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


class _SequentialJsonClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.messages = []

    def create_chat_completion_text(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


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
                    "summary_hops": [
                        {
                            "sub_question": f"事实{index}是什么？",
                            "material_id": material.material_id,
                            "evidence_mode": "text",
                            "required_image_ids": [],
                        }
                        for index, material in enumerate(materials, start=1)
                    ],
                    "optional_material_ids": [],
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

    def test_summary_type_mismatch_gets_one_targeted_planner_retry(self):
        chunks = [
            _chunk(1, "1.1", "文档>申请材料", "应提交身份证明。"),
            _chunk(2, "1.2", "文档>办理时限", "五个工作日内办结。"),
        ]
        client = _SequentialJsonClient(
            [
                {
                    "items": [
                        {
                            "scenario_type": "point",
                            "intent": "了解办理要求",
                            "reader_need": "知道办理条件",
                            "required_material_refs": ["主材料-A", "主材料-B"],
                            "optional_material_refs": [],
                            "evidence_mode": "text",
                            "required_image_refs": [],
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "scenario_type": "summary",
                            "intent": "完成办理前需要整体了解哪些要求？",
                            "reader_need": "统筹准备办理事项",
                            "summary_hops": [
                                {
                                    "sub_question": "需要提交什么申请材料？",
                                    "material_ref": "主材料-A",
                                    "evidence_mode": "text",
                                    "image_refs": [],
                                },
                                {
                                    "sub_question": "办理时限是多久？",
                                    "material_ref": "主材料-B",
                                    "evidence_mode": "text",
                                    "image_refs": [],
                                },
                            ],
                            "optional_material_refs": [],
                        }
                    ]
                },
            ]
        )
        captured = []

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
                planning_batch_index=kwargs.get("planning_batch_index"),
                planning_batch_count=kwargs.get("planning_batch_count"),
            )

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="summary",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(1, len(plan.units))
        self.assertEqual("summary", plan.units[0].qa_mode)
        self.assertEqual(["section-1", "section-2"], plan.units[0].required_material_ids)
        self.assertEqual(2, len(plan.units[0].summary_hops))
        self.assertEqual("hop-1", plan.units[0].summary_hops[0]["hop_id"])
        self.assertEqual(2, len(client.messages))
        self.assertIn("本批次只接受 `summary`", client.messages[1][-1]["content"])
        self.assertEqual(2, captured[0]["planning_attempt_count"])
        self.assertEqual("scenario_type_mismatch", captured[0]["planner_retry_reason"])
        planner_detail = plan.summary()["scenario_planner_batch_details"]["summary"][0]
        self.assertEqual(2, planner_detail["planning_attempt_count"])
        self.assertGreaterEqual(float(planner_detail["planner_seconds"]), 0.0)
        self.assertEqual(
            "文档>申请材料",
            planner_detail["scenarios"][0]["summary_hops"][0]["material_path"],
        )
        self.assertEqual(
            "办理时限是多久？",
            planner_detail["scenarios"][0]["summary_hops"][1]["sub_question"],
        )

    def test_summary_with_only_one_atomic_hop_is_rejected(self):
        chunks = [_chunk(1, "1.1", "文档>材料", "应提交身份证明。")]

        def planner(materials, _count, _mode, **_kwargs):
            return [
                {
                    "scenario_type": "summary",
                    "intent": "了解申请要求",
                    "reader_need": "准备申请",
                    "summary_hops": [
                        {
                            "sub_question": "需要提交什么材料？",
                            "material_id": materials[0].material_id,
                            "evidence_mode": "text",
                            "required_image_ids": [],
                        }
                    ],
                    "optional_material_ids": [],
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

    def test_summary_mode_and_required_images_are_derived_from_hops(self):
        chunks = [
            _chunk(
                1,
                "1.1",
                "文档>申报结果",
                "审核通过后可查看申报记录。",
                image_materials=[
                    {
                        "image_id": "image-1",
                        "description": "页面提供导出按钮。",
                        "context_before": "",
                        "context_after": "",
                    }
                ],
            )
        ]
        client = _JsonClient(
            {
                "items": [
                    {
                        "scenario_type": "summary",
                        "intent": "了解申报结果的查看和导出方式",
                        "reader_need": "查看并保存申报结果",
                        "summary_hops": [
                            {
                                "sub_question": "审核通过后可以查看什么？",
                                "material_ref": "主材料-A",
                                "evidence_mode": "text",
                                "image_refs": [],
                            },
                            {
                                "sub_question": "页面通过什么控件导出记录？",
                                "material_ref": "主材料-A",
                                "evidence_mode": "visual",
                                "image_refs": ["图片-A"],
                            },
                        ],
                        "optional_material_refs": [],
                        # These conflicting top-level values are deliberately
                        # ignored for Summary; hops own the evidence contract.
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
                planning_batch_index=kwargs.get("planning_batch_index"),
                planning_batch_count=kwargs.get("planning_batch_count"),
            )

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="summary",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(1, len(plan.units))
        self.assertEqual("mixed", plan.units[0].evidence_mode)
        self.assertEqual(["image-1"], plan.units[0].required_image_ids)
        self.assertEqual("visual", plan.units[0].summary_hops[1]["evidence_mode"])
        self.assertNotIn('"material_ref":"主材料-B"', client.messages[0]["content"])

    def test_visual_alternative_is_promoted_over_same_material_text_candidate(self):
        chunks = [
            _chunk(
                1,
                "1.1",
                "文档>网上申报操作",
                "可通过平台办理申报。",
                image_materials=[
                    {
                        "image_id": "image-1",
                        "description": "上传前必须勾选同意协议，提交按钮才可使用。",
                        "context_before": "",
                        "context_after": "",
                    }
                ],
            )
        ]

        def planner(materials, _count, _mode, **_kwargs):
            material = materials[0]
            return [
                {
                    "scenario_type": "point",
                    "intent": "了解网上申报入口",
                    "reader_need": "找到办理渠道",
                    "required_material_ids": [material.material_id],
                    "optional_material_ids": [],
                    "evidence_mode": "text",
                    "required_image_ids": [],
                },
                {
                    "scenario_type": "point",
                    "intent": "上传前需要完成什么确认操作？",
                    "reader_need": "避免提交按钮不可用",
                    "required_material_ids": [material.material_id],
                    "optional_material_ids": [],
                    "evidence_mode": "visual",
                    "required_image_ids": ["image-1"],
                },
            ]

        plan = plan_generation_units(
            chunks,
            qa_total_limit=1,
            qa_per_chunk=1,
            qa_detail_mode="point",
            chunk_size=600,
            scenario_planner=planner,
        )
        self.assertEqual(["image-1"], plan.units[0].required_image_ids)
        self.assertEqual(
            "promoted_visual_alternative_same_material",
            plan.units[0].debug["selection_adjustment"],
        )

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
                    "summary_hops": [
                        {
                            "sub_question": "页面可以查看哪些记录？",
                            "material_id": material.material_id,
                            "evidence_mode": "text",
                            "required_image_ids": [],
                        },
                        {
                            "sub_question": "页面通过什么按钮导出记录？",
                            "material_id": material.material_id,
                            "evidence_mode": "mixed",
                            "required_image_ids": ["image-1"],
                        },
                    ],
                    "optional_material_ids": [],
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
                    "summary_hops": [
                        {
                            "sub_question": "可以通过什么平台查看申报记录？",
                            "material_id": materials[0].material_id,
                            "evidence_mode": "text",
                            "required_image_ids": [],
                        },
                        {
                            "sub_question": "审核记录页面通过什么控件导出数据？",
                            "material_id": materials[1].material_id,
                            "evidence_mode": "visual",
                            "required_image_ids": ["image-2"],
                        },
                    ],
                    "optional_material_ids": [],
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

    def test_summary_is_not_dropped_when_it_extends_a_point_fact(self):
        items = [
            {
                "question": "本操作说明适用于哪些参保单位？",
                "source_fact_text": "该操作文档适用于参加城镇职工养老保险的参保单位。",
                "qa_generation_material_ids": ["section-1"],
                "qa_generation_unit_mode": "point",
            },
            {
                "question": "哪些用人单位需要办理缴费基数诚信申报？",
                "source_fact_text": "该操作文档适用于参加城镇职工养老保险的参保单位。适用范围为全省参加城镇职工养老保险的用人单位（含个体工商户），不包含自由职业者单位。",
                "qa_generation_material_ids": ["section-2"],
                "qa_generation_unit_mode": "summary",
            },
        ]
        deduped, dropped = _deduplicate_document_questions(items)
        self.assertEqual(2, len(deduped))
        self.assertEqual(0, dropped)

    def test_cross_mode_equivalent_grounded_fact_is_still_deduplicated(self):
        shared_fact = "缴费基数诚信申报未生效前，只影响正常的单位缴费核定业务。"
        items = [
            {
                "question": "诚信申报未生效会影响什么业务？",
                "source_fact_text": shared_fact,
                "qa_generation_material_ids": ["section-1"],
                "qa_generation_unit_mode": "point",
            },
            {
                "question": "缴费基数诚信申报未生效时会影响哪些业务？",
                "source_fact_text": shared_fact,
                "qa_generation_material_ids": ["section-2"],
                "qa_generation_unit_mode": "summary",
            },
        ]
        deduped, dropped = _deduplicate_document_questions(items)
        self.assertEqual(1, len(deduped))
        self.assertEqual(1, dropped)
        self.assertEqual("summary", deduped[0]["qa_generation_unit_mode"])

    def test_cross_mode_similar_wording_does_not_drop_composite_summary(self):
        items = [
            {
                "question": "未按时完成诚信申报会有什么影响？",
                "source_fact_text": "未按时完成申报将无法办理缴费核定。",
                "qa_generation_material_ids": ["section-1"],
                "qa_generation_unit_mode": "point",
            },
            {
                "question": "未按时完成缴费基数诚信申报会有哪些影响？",
                "source_fact_text": (
                    "未按时完成申报将无法办理缴费核定；单位整体补收、个人缴费核定"
                    "和个人补费等其他核定业务不受影响。"
                ),
                "qa_generation_material_ids": ["section-1", "section-2"],
                "qa_generation_unit_mode": "summary",
            },
        ]
        deduped, dropped = _deduplicate_document_questions(items)
        self.assertEqual(2, len(deduped))
        self.assertEqual(0, dropped)


if __name__ == "__main__":
    unittest.main()
