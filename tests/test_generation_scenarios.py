import json
import unittest

from qa.generation.qa_generation_flow import (
    call_question_editor_llm,
    call_scenario_planner_llm,
)
from qa.generation.structure_units import plan_generation_units
from qa.text_to_qa_pipeline import _deduplicate_document_questions


def _chunk(index, section_path, title, text, *, parent="1", fragment=1, fragment_count=1):
    return {
        "chunk_id": f"c{index}",
        "chunk_index": index,
        "section_chunk_index": fragment,
        "section_path": section_path,
        "section_parent_path": parent,
        "section_level": 2,
        "section_is_leaf": True,
        "title_path": title,
        "fragment_group_id": f"g-{section_path}",
        "fragment_index": fragment,
        "fragment_count": fragment_count,
        "content_kind": "mixed" if "图片" in text else "text",
        "source_asset_ids": ["img-1"] if "图片" in text else [],
        "text": text,
        "text_for_embedding": text,
    }


class _EditorClient:
    def __init__(self, decision, question="", reason=""):
        self.payload = {"decision": decision, "question": question, "reason": reason}

    def create_chat_completion_text(self, **_kwargs):
        return json.dumps(self.payload, ensure_ascii=False)


class _ScenarioClient:
    def __init__(self, items):
        self.items = items

    def create_chat_completion_text(self, **_kwargs):
        return json.dumps({"items": self.items}, ensure_ascii=False)


class GenerationScenarioTests(unittest.TestCase):
    def test_scenario_llm_validation_keeps_typed_candidate_pools_separate(self):
        chunk = _chunk(1, "1.1", "文档>材料", "应提交身份证明。")
        captured = []

        def planner(materials, _count, _mode):
            return call_scenario_planner_llm(
                client=_ScenarioClient(
                    [
                        {
                            "scenario_type": "summary",
                            "intent": "错误类型",
                            "reader_need": "总结材料",
                            "material_ids": [materials[0].material_id],
                        }
                    ]
                ),
                model="test",
                section_materials=list(materials),
                requested_count=1,
                qa_detail_mode="point",
                prompt_language="zh",
                request_timeout=10,
                debug_writer=captured.append,
            )

        plan = plan_generation_units(
            [chunk],
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=planner,
        )

        self.assertEqual("point", plan.units[0].qa_mode)
        self.assertEqual(1, captured[0]["dropped_validation_reasons"]["scenario_type_mismatch"])

    def test_scenario_llm_validation_rejects_multi_material_point(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>时限", "五个工作日内办结。"),
        ]
        captured = []

        def planner(materials, _count, _mode):
            return call_scenario_planner_llm(
                client=_ScenarioClient(
                    [
                        {
                            "scenario_type": "point",
                            "intent": "错误合并",
                            "reader_need": "同时了解材料和时限",
                            "material_ids": [materials[0].material_id, materials[1].material_id],
                        }
                    ]
                ),
                model="test",
                section_materials=list(materials),
                requested_count=1,
                qa_detail_mode="point",
                prompt_language="zh",
                request_timeout=10,
                debug_writer=captured.append,
            )

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=planner,
        )

        self.assertEqual(1, len(plan.units))
        self.assertEqual(["section-1"], plan.units[0].material_ids)
        self.assertEqual(
            "llm_point_pool_underfilled",
            plan.units[0].debug["raw_scenario"]["fallback_reason"],
        )
        self.assertEqual(
            1,
            captured[0]["dropped_validation_reasons"]["point_requires_one_material"],
        )

    def test_same_section_fragments_and_image_are_one_material(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "第一段正文。", fragment=1, fragment_count=2),
            _chunk(2, "1.1", "文档>材料", "第二段正文和图片说明。", fragment=2, fragment_count=2),
            _chunk(3, "1.2", "文档>时限", "办理期限为五个工作日。"),
        ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=lambda materials, count, mode: [
                {
                    "scenario_type": "point",
                    "intent": "询问材料内容",
                    "reader_need": "了解申请材料",
                    "material_ids": [materials[0].material_id],
                }
            ],
        )

        self.assertEqual(2, len(plan.section_materials))
        first = plan.section_materials[0]
        self.assertEqual([1, 2], first.source_chunk_indexes)
        self.assertIn("第一段正文", first.material_text)
        self.assertIn("第二段正文和图片说明", first.material_text)
        self.assertEqual(["img-1"], first.source_asset_ids)
        self.assertEqual([1, 2], plan.units[0].source_chunk_indexes)

    def test_repeated_markdown_heading_is_removed_after_first_fragment(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "## 材料\n第一段正文。", fragment=1, fragment_count=2),
            _chunk(2, "1.1", "文档>材料", "## 材料\n第二段正文。", fragment=2, fragment_count=2),
        ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=lambda materials, count, mode: [
                {
                    "scenario_type": "point",
                    "intent": "询问材料",
                    "reader_need": "了解材料",
                    "material_ids": [materials[0].material_id],
                }
            ],
        )

        self.assertEqual(1, plan.section_materials[0].material_text.count("## 材料"))
        self.assertIn("第一段正文", plan.section_materials[0].material_text)
        self.assertIn("第二段正文", plan.section_materials[0].material_text)

    def test_auto_planning_uses_bounded_typed_batches_and_global_mix(self):
        chunks = [
            _chunk(index, f"1.{index}", f"文档>第{index}节", f"第{index}节规定了事实{index}。")
            for index in range(1, 7)
        ]
        calls = []

        def planner(materials, count, mode):
            calls.append((mode, len(materials), count))
            if mode == "point":
                return [
                    {
                        "scenario_type": "point",
                        "intent": f"询问{material.material_id}",
                        "reader_need": f"了解{material.material_id}",
                        "material_ids": [material.material_id],
                    }
                    for material in materials[:count]
                ]
            if len(materials) < 2 or count <= 0:
                return []
            return [
                {
                    "scenario_type": "summary",
                    "intent": "总结相邻要求",
                    "reader_need": "共同了解两项要求",
                    "material_ids": [materials[0].material_id, materials[1].material_id],
                }
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=4,
            qa_per_chunk=1,
            qa_detail_mode="auto",
            chunk_size=600,
            scenario_planning_batch_chars=500,
            scenario_planner=planner,
        )

        self.assertTrue(calls)
        self.assertTrue(all(mode in {"point", "summary"} for mode, _size, _count in calls))
        self.assertTrue(all(size < len(chunks) for _mode, size, _count in calls))
        self.assertEqual(4, len(plan.units))
        self.assertEqual(1, plan.scenario_selected_by_type["summary"])
        self.assertEqual(3, plan.scenario_selected_by_type["point"])

    def test_point_budget_covers_distinct_small_batches(self):
        chunks = [
            _chunk(index, f"1.{index}", f"文档>第{index}节", f"第{index}节规定了事实{index}。")
            for index in range(1, 4)
        ]
        called_material_ids = []

        def planner(materials, count, mode):
            self.assertEqual("point", mode)
            called_material_ids.extend(material.material_id for material in materials[:count])
            return [
                {
                    "scenario_type": "point",
                    "intent": f"询问{material.material_id}",
                    "reader_need": f"了解{material.material_id}",
                    "material_ids": [material.material_id],
                }
                for material in materials[:count]
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=3,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planning_batch_chars=500,
            scenario_planner=planner,
        )

        self.assertEqual(["section-1", "section-2", "section-3"], called_material_ids)
        self.assertEqual(3, len(plan.units))

    def test_one_section_can_supply_multiple_distinct_point_scenarios(self):
        chunks = [
            _chunk(
                1,
                "1.1",
                "文档>办理要求",
                "申请人应提交身份证明，受理后五个工作日内办结。",
            )
        ]

        def planner(materials, count, mode):
            self.assertEqual("point", mode)
            self.assertEqual(2, count)
            material_id = materials[0].material_id
            return [
                {
                    "scenario_type": "point",
                    "intent": "询问申请材料",
                    "reader_need": "了解需要提交什么",
                    "material_ids": [material_id],
                },
                {
                    "scenario_type": "point",
                    "intent": "询问办理时限",
                    "reader_need": "了解多久办结",
                    "material_ids": [material_id],
                },
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=2,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=planner,
        )

        self.assertEqual(2, len(plan.units))
        self.assertEqual([["section-1"], ["section-1"]], [unit.material_ids for unit in plan.units])
        self.assertEqual(
            {"询问申请材料", "询问办理时限"},
            {unit.scenario_intent for unit in plan.units},
        )

    def test_auto_budget_flows_back_to_point_when_summary_pool_is_short(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>时限", "办理期限为五个工作日。"),
            _chunk(3, "1.3", "文档>费用", "办理不收取费用。"),
        ]

        def planner(materials, _count, _mode):
            return [
                {
                    "scenario_type": "point",
                    "intent": f"询问{material.title_path}",
                    "reader_need": f"了解{material.title_path}",
                    "material_ids": [material.material_id],
                }
                for material in materials
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=3,
            qa_per_chunk=1,
            qa_detail_mode="auto",
            chunk_size=600,
            scenario_planner=planner,
        )

        self.assertEqual(3, len(plan.units))
        self.assertEqual({"point": 3, "summary": 0}, plan.scenario_selected_by_type)

    def test_auto_uses_deterministic_point_fallback_when_planner_returns_only_summary(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>时限", "办理期限为五个工作日。"),
            _chunk(3, "1.3", "文档>费用", "办理不收取费用。"),
        ]

        def planner(materials, _count, mode):
            if mode == "point":
                return []
            return [
                {
                    "scenario_type": "summary",
                    "intent": "总结材料和时限",
                    "reader_need": "了解办理要求",
                    "material_ids": [materials[0].material_id, materials[1].material_id],
                }
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=3,
            qa_per_chunk=1,
            qa_detail_mode="auto",
            chunk_size=600,
            scenario_planner=planner,
        )

        self.assertEqual(3, len(plan.units))
        self.assertEqual({"point": 2, "summary": 1}, plan.scenario_selected_by_type)
        self.assertTrue(
            any(
                unit.debug["raw_scenario"].get("fallback_reason")
                == "llm_point_pool_underfilled"
                for unit in plan.units
                if unit.qa_mode == "point"
            )
        )

    def test_summary_scenario_can_bind_related_sibling_sections_only_when_planned(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>时限", "办理期限为五个工作日。"),
        ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="summary",
            chunk_size=600,
            scenario_planner=lambda materials, count, mode: [
                {
                    "scenario_type": "summary",
                    "intent": "总结办理要求",
                    "reader_need": "一次了解材料和时限",
                    "material_ids": [material.material_id for material in materials],
                }
            ],
        )

        self.assertEqual([1, 2], plan.units[0].source_chunk_indexes)
        self.assertEqual("summary", plan.units[0].qa_mode)

    def test_question_editor_supports_keep_rewrite_and_drop(self):
        common = {
            "model": "test",
            "candidate": {"question": "原问题？", "question_type": "简答题"},
            "source_material": "产假增加六十天。",
            "scenario_intent": "询问增加产假天数",
            "reader_need": "了解额外产假",
            "qa_detail_mode": "point",
            "prompt_language": "zh",
            "request_timeout": 10,
        }
        kept, kept_status = call_question_editor_llm(client=_EditorClient("keep"), **common)
        rewritten, rewritten_status = call_question_editor_llm(
            client=_EditorClient("rewrite", "生育后还能增加多少天产假？"),
            **common,
        )
        dropped, dropped_status = call_question_editor_llm(client=_EditorClient("drop"), **common)

        self.assertEqual("原问题？", kept["question"])
        self.assertEqual("keep", kept_status)
        self.assertEqual("生育后还能增加多少天产假？", rewritten["question"])
        self.assertEqual("rewrite", rewritten_status)
        self.assertIsNone(dropped)
        self.assertEqual("drop", dropped_status)

    def test_document_question_dedup_ignores_spacing_and_punctuation(self):
        items = [
            {"question": "办理需要多久？", "answer": "五天"},
            {"question": "办理 需要 多久", "answer": "五个工作日"},
            {"question": "需要提交哪些材料？", "answer": "身份证明"},
        ]

        deduped, dropped = _deduplicate_document_questions(items)

        self.assertEqual(2, len(deduped))
        self.assertEqual(1, dropped)


if __name__ == "__main__":
    unittest.main()
