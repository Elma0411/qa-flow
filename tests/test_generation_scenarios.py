import json
import unittest

from qa.generation.qa_generation_flow import (
    call_question_editor_llm,
    call_scenario_planner_llm,
)
from qa.generation.structure_units import plan_generation_units
from qa.text_to_qa_pipeline import _deduplicate_document_questions


def _chunk(index, section_path, title, text, *, image_materials=None, fragment=1, fragment_count=1):
    return {
        "chunk_id": f"c{index}",
        "chunk_index": index,
        "section_chunk_index": fragment,
        "section_path": section_path,
        "section_parent_path": "文档",
        "section_level": 2,
        "section_is_leaf": True,
        "title_path": title,
        "fragment_group_id": f"g-{section_path}",
        "fragment_index": fragment,
        "fragment_count": fragment_count,
        "content_kind": "text",
        "source_asset_ids": [],
        "text": text,
        "text_for_embedding": text,
        "image_materials": image_materials or [],
    }


class _JsonClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def create_chat_completion_text(self, **kwargs):
        self.messages = kwargs["messages"]
        return json.dumps(self.payload, ensure_ascii=False)


class GenerationScenarioTests(unittest.TestCase):
    def test_planner_uses_aliases_and_rejects_internal_material_ids(self):
        chunks = [_chunk(1, "1.1", "文档>材料", "应提交身份证明。")]
        captured = []
        client = _JsonClient(
            {
                "items": [
                    {
                        "scenario_type": "point",
                        "intent": "了解材料要求",
                        "reader_need": "知道需要提交什么",
                        "required_material_refs": ["section-1"],
                        "optional_material_refs": [],
                        "evidence_mode": "text",
                        "required_image_refs": [],
                    }
                ]
            }
        )

        def planner(materials, count, mode, **kwargs):
            return call_scenario_planner_llm(
                client=client,
                model="test",
                section_materials=list(materials),
                requested_count=count,
                qa_detail_mode=mode,
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
        self.assertEqual("llm_point_pool_underfilled", plan.units[0].debug["raw_scenario"]["fallback_reason"])
        self.assertEqual(1, captured[0]["dropped_validation_reasons"]["unknown_material_ref"])
        self.assertNotIn("section-1", client.messages[1]["content"])
        self.assertIn("主材料-A", client.messages[1]["content"])

    def test_point_requires_one_required_material(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>时限", "五个工作日内办结。"),
        ]

        def planner(materials, _count, _mode, **_kwargs):
            return [
                {
                    "scenario_type": "point",
                    "intent": "错误合并",
                    "reader_need": "同时了解材料和时限",
                    "required_material_ids": [material.material_id for material in materials],
                    "optional_material_ids": [],
                    "evidence_mode": "text",
                    "required_image_ids": [],
                }
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(1, len(plan.units))
        self.assertEqual("llm_point_pool_underfilled", plan.units[0].debug["raw_scenario"]["fallback_reason"])

    def test_summary_rejects_more_than_three_required_materials(self):
        chunks = [
            _chunk(index, f"1.{index}", f"文档>第{index}节", f"事实{index}。")
            for index in range(1, 5)
        ]

        def planner(materials, _count, _mode, **_kwargs):
            return [
                {
                    "scenario_type": "summary",
                    "intent": "概括整份手册",
                    "reader_need": "了解所有内容",
                    "required_material_ids": [material.material_id for material in materials],
                    "optional_material_ids": [],
                    "evidence_mode": "text",
                    "required_image_ids": [],
                }
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="summary",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual([], plan.units)

    def test_same_section_fragments_and_images_stay_one_material(self):
        chunks = [
            _chunk(1, "1.1", "文档>办理", "第一段正文。", fragment=1, fragment_count=2),
            _chunk(
                2,
                "1.1",
                "文档>办理",
                "第二段正文。",
                fragment=2,
                fragment_count=2,
                image_materials=[
                    {
                        "image_id": "image-1",
                        "description": "页面提供导出按钮。",
                        "context_before": "正文",
                        "context_after": "结束",
                    }
                ],
            ),
        ]

        def planner(materials, _count, _mode, **_kwargs):
            material = materials[0]
            return [
                {
                    "scenario_type": "summary",
                    "intent": "了解页面操作",
                    "reader_need": "查看并导出记录",
                    "required_material_ids": [material.material_id],
                    "optional_material_ids": [],
                    "evidence_mode": "mixed",
                    "required_image_ids": ["image-1"],
                }
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="summary",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(1, len(plan.section_materials))
        self.assertEqual([1, 2], plan.section_materials[0].source_chunk_indexes)
        self.assertEqual(1, len(plan.section_materials[0].image_materials))
        self.assertEqual("mixed", plan.units[0].evidence_mode)
        self.assertEqual(["image-1"], plan.units[0].required_image_ids)

    def test_required_image_promotes_its_parent_material_before_writing(self):
        chunks = [
            _chunk(1, "1.1", "文档>渠道", "可通过平台办理。"),
            _chunk(
                2,
                "1.2",
                "文档>界面",
                "审核记录页面。",
                image_materials=[
                    {
                        "image_id": "image-2",
                        "description": "页面提供导出按钮。",
                        "context_before": "",
                        "context_after": "",
                    }
                ],
            ),
        ]

        def planner(materials, _count, _mode, **_kwargs):
            return [
                {
                    "scenario_type": "summary",
                    "intent": "查看并导出申报记录",
                    "reader_need": "了解平台渠道和导出操作",
                    "required_material_ids": [materials[0].material_id],
                    "optional_material_ids": [materials[1].material_id],
                    "evidence_mode": "mixed",
                    "required_image_ids": ["image-2"],
                }
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="summary",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(["section-1", "section-2"], plan.units[0].required_material_ids)
        self.assertEqual([], plan.units[0].optional_material_ids)

    def test_same_visual_flow_is_deduplicated_across_sections(self):
        flow_description = "流程从人员申报开始，经单位诚信申报和经办机构审核，审核通过后进入缴费，审核不通过后返回重新申报。"
        chunks = [
            _chunk(
                1,
                "1.1",
                "文档>经办机构流程",
                "流程说明。",
                image_materials=[{"image_id": "image-1", "description": flow_description, "context_before": "", "context_after": ""}],
            ),
            _chunk(
                2,
                "1.2",
                "文档>参保单位流程",
                "流程说明。",
                image_materials=[{"image_id": "image-2", "description": flow_description, "context_before": "", "context_after": ""}],
            ),
        ]

        def planner(materials, _count, _mode, **_kwargs):
            return [
                {
                    "scenario_type": "point",
                    "intent": f"了解{material.title_path}的完整流程",
                    "reader_need": "了解申报流程",
                    "required_material_ids": [material.material_id],
                    "optional_material_ids": [],
                    "evidence_mode": "visual",
                    "required_image_ids": [material.image_materials[0].image_id],
                }
                for material in materials
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=2,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(1, len(plan.units))

    def test_editor_returns_one_final_question_without_contract_fields(self):
        client = _JsonClient({"question": "材料齐全后，审核需要多久？"})
        edited, status = call_question_editor_llm(
            client=client,
            model="test",
            candidate={"question": "申请材料齐全并受理后，应当在多少个工作日内完成审核？", "question_type": "简答题"},
            source_material="材料齐全后五个工作日内完成审核。",
            scenario_intent="询问审核期限",
            reader_need="了解审核需要多久",
            qa_detail_mode="point",
            prompt_language="zh",
            request_timeout=10,
        )
        self.assertEqual("edited", status)
        self.assertEqual("材料齐全后，审核需要多久？", edited["question"])
        self.assertEqual({"question", "question_type"}, set(edited))
        self.assertNotIn("required_material_refs", client.messages[1]["content"])

    def test_auto_falls_back_to_points_when_summary_pool_is_empty(self):
        chunks = [
            _chunk(1, "1.1", "文档>材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>时限", "五个工作日内办结。"),
        ]

        def planner(materials, _count, mode, **_kwargs):
            if mode == "summary":
                return []
            return [
                {
                    "scenario_type": "point",
                    "intent": f"了解{material.title_path}",
                    "reader_need": f"了解{material.title_path}",
                    "required_material_ids": [material.material_id],
                    "optional_material_ids": [],
                    "evidence_mode": "text",
                    "required_image_ids": [],
                }
                for material in materials
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=2,
            qa_per_chunk=1,
            qa_detail_mode="auto",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(2, len(plan.units))
        self.assertTrue(all(unit.qa_mode == "point" for unit in plan.units))

    def test_document_dedup_removes_cross_mode_semantic_duplicate(self):
        items = [
            {"question": "单位缴费基数诚信申报流程是什么？", "qa_generation_material_ids": ["section-1"]},
            {"question": "单位缴费基数诚信申报流程是什么", "qa_generation_material_ids": ["section-1"]},
            {"question": "申报可以通过哪些渠道办理？", "qa_generation_material_ids": ["section-2"]},
        ]
        deduped, dropped = _deduplicate_document_questions(items)
        self.assertEqual(2, len(deduped))
        self.assertEqual(1, dropped)


if __name__ == "__main__":
    unittest.main()
