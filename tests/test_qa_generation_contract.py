import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.debug.qa_store import get_debug_map, upsert_qa_debug_items
from app.services.storage.consolidation import build_consolidated_entry
from app.services.unsupervised_evaluation import compute_unsupervised_average_score
from qa.generation.evidence_units import QADocumentEvidenceIndex
from qa.generation.qa_generation_flow import (
    call_candidate_question_llm,
    call_evidence_answer_llm,
    call_question_editor_llm,
)
from qa.pipeline_runtime import parse_one_step_pipeline_runtime
from qa.prompts.qa_generation_prompts import (
    build_candidate_question_system_prompt,
    build_evidence_answer_system_prompt,
    build_planner_category_profile,
    build_question_editor_system_prompt,
    build_scenario_planner_system_prompt,
)
from qa.retrieval import EvidenceWindow
from qa.text_to_qa_pipeline import _unit_latency_percentiles
from qa.validation import validate_and_normalize_item_with_reason


class _StaticChatClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = []

    def create_chat_completion_text(self, **kwargs):
        self.messages = kwargs["messages"]
        return json.dumps(self.payload, ensure_ascii=False)


class _SequentialChatClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.messages = []

    def create_chat_completion_text(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return json.dumps(self.payloads.pop(0), ensure_ascii=False)


def _generation_unit(*, mode="text", required_images=None, required_materials=None):
    required_images = list(required_images or [])
    required_materials = list(required_materials or ["section-1"])
    chunks = [
        {"chunk_id": "chunk-1", "chunk_index": 1, "title_path": "办理 > 正文", "text": "正文事实。"},
        {"chunk_id": "chunk-2", "chunk_index": 2, "title_path": "办理 > 时限", "text": "时限事实。"},
    ]
    ref_map = {
        "正文证据-1": {
            "chunk_id": "chunk-1",
            "chunk_index": 1,
            "title_path": "办理 > 正文",
            "material_id": "section-1",
            "role": "primary_source",
        },
        "正文证据-2": {
            "chunk_id": "chunk-2",
            "chunk_index": 2,
            "title_path": "办理 > 时限",
            "material_id": "section-2",
            "role": "primary_source",
        },
    }
    rendered = "【必需正文证据】\n\n正文证据-1\n正文：正文事实。"
    evidence_text_by_ref = {
        "正文证据-1": "正文证据-1\n正文：正文事实。",
        "正文证据-2": "正文证据-2\n正文：时限事实。",
    }
    if mode in {"visual", "mixed"}:
        ref_map["图片证据-1"] = {
            "chunk_id": "chunk-1",
            "chunk_index": 1,
            "title_path": "办理 > 正文",
            "material_id": "section-1",
            "image_id": "image-1",
            "role": "primary_visual",
        }
        rendered += "\n\n【必需图片证据】\n\n图片证据-1\n图片事实：页面显示导出按钮。"
        evidence_text_by_ref["图片证据-1"] = "图片证据-1\n图片事实：页面显示导出按钮。"
    return {
        "source_chunk": chunks[0],
        "source_chunks": chunks,
        "source_unit_text": "正文事实。",
        "qa_generation_unit_text": rendered,
        "evidence_chunk_ids": [],
        "qa_generation_unit_id": "unit-1",
        "source_unit": {
            "required_material_ids": required_materials,
            "material_ids": required_materials,
            "required_image_ids": required_images,
            "evidence_mode": mode,
        },
        "llm_evidence_ref_map": ref_map,
        "llm_evidence_text_by_ref": evidence_text_by_ref,
    }


class QAGenerationContractTests(unittest.TestCase):
    def test_unsupervised_average_uses_three_visible_metrics_and_fixed_denominator(self):
        self.assertAlmostEqual(
            0.5,
            compute_unsupervised_average_score(
                {"faithfulness": 1, "answerability": 0.5, "coverage_score": 0}
            ),
        )
        self.assertAlmostEqual(
            1 / 3,
            compute_unsupervised_average_score(
                {"faithfulness": 1, "answerability": "bad", "coverage_score": float("nan")}
            ),
        )

    def test_prompt_surface_is_closed_to_each_model_stage(self):
        candidate_prompt = build_candidate_question_system_prompt(
            language_code="zh",
            language_instruction="请使用简体中文。",
            qa_detail_mode="point",
            style_example="风格示例：申请受理后，审核需要多久？",
        )
        editor_prompt = build_question_editor_system_prompt(
            language_code="zh",
            language_instruction="请使用简体中文。",
            qa_detail_mode="summary",
        )
        answer_prompt = build_evidence_answer_system_prompt(
            language_code="zh",
            language_instruction="请使用简体中文。",
            qa_detail_mode="mixed",
            question_type="简答题",
        )
        for prompt in (candidate_prompt, editor_prompt, answer_prompt):
            self.assertNotIn("chunk_id", prompt)
            self.assertNotIn("snippet", prompt)
            self.assertNotIn("typed_primary_materials", prompt)
            self.assertNotIn("difficulty_level", prompt)
            self.assertNotIn("difficulty_score", prompt)
            self.assertNotIn("question_type_reason", prompt)
        self.assertIn('{"question":"..."}', candidate_prompt)
        self.assertIn('{"question":"..."}', editor_prompt)
        self.assertIn("图片证据", answer_prompt)
        self.assertIn("肯否关系", candidate_prompt)
        self.assertIn("不要逐项复述", candidate_prompt)

    def test_summary_planner_prompt_only_allows_summary_schema(self):
        prompt = build_scenario_planner_system_prompt(
            language_code="zh",
            language_instruction="请使用简体中文。",
            requested_count=2,
            qa_detail_mode="summary",
        )
        self.assertIn('"scenario_type":"summary"', prompt)
        self.assertNotIn('"scenario_type":"point|summary"', prompt)
        self.assertIn("本批次只规划总结场景", prompt)

    def test_category_profile_stays_in_planning_and_few_shot_is_style_only(self):
        profile = build_planner_category_profile(
            knowledge_category="法律法规/部委规章/其他",
            language_code="zh",
        )
        self.assertIn("读者画像", profile)
        prompt = build_candidate_question_system_prompt(
            language_code="zh",
            language_instruction="请使用简体中文。",
            qa_detail_mode="point",
            style_example="风格示例：申请受理后，审核需要多久？",
        )
        self.assertIn("风格示例", prompt)
        self.assertNotIn("分类专用模板", prompt)
        self.assertNotIn("few-shot", prompt)

    def test_candidate_writer_receives_only_a_writing_brief_and_backfills_question_type(self):
        client = _StaticChatClient({"question": "婚前医学检查费用由谁承担？"})
        items = call_candidate_question_llm(
            client=client,
            model="test-model",
            source_chunk_text="结婚登记前参加婚前医学检查的费用按规定承担。",
            source_chunk_meta={
                "chunk_id": "private-chunk-id",
                "title_path": "婚姻登记 > 婚前医学检查",
                "qa_generation_unit_subject_label": "婚前医学检查",
                "qa_generation_unit_scenario_intent": "明确费用承担主体",
                "qa_generation_unit_reader_need": "了解费用由谁承担",
            },
            candidate_count=1,
            prompt_language="zh",
            question_type_plan=["简答题"],
            few_shot_examples=None,
            request_timeout=10,
            qa_detail_mode="point",
        )
        self.assertEqual([{"question": "婚前医学检查费用由谁承担？", "question_type": "简答题"}], items)
        user_content = client.messages[1]["content"]
        self.assertIn("提问焦点", user_content)
        self.assertIn("回答依据", user_content)
        self.assertNotIn("private-chunk-id", user_content)
        self.assertNotIn("material_ref", user_content)
        self.assertNotIn("evidence_mode", user_content)

    def test_editor_changes_only_wording_and_preserves_frozen_contract(self):
        client = _StaticChatClient({"question": "申报通过后，可以在平台上查看或导出哪些信息？"})
        frozen_meta = {
            "qa_generation_unit_subject_label": "单位缴费基数诚信申报",
            "qa_generation_unit_scenario_intent": "了解审核通过后的界面操作",
            "qa_generation_unit_reader_need": "查看审核状态并导出申报记录",
            "qa_generation_unit_evidence_mode": "mixed",
            "qa_generation_unit_required_material_ids": ["section-1"],
            "qa_generation_unit_required_image_ids": ["image-1"],
            "qa_generation_unit_material_ref_map": {"主材料-A": "section-1"},
            "qa_generation_unit_image_ref_map": {"图片-A": "image-1"},
            "qa_generation_unit_prompt_materials": [
                {
                    "material_ref": "主材料-A",
                    "text_content": "审核通过后可查看申报记录。",
                    "image_materials": [{"image_ref": "图片-A", "description": "页面提供导出按钮。"}],
                }
            ],
        }
        candidate = {"question": "审核通过后该如何处理？", "question_type": "简答题"}
        edited, status = call_question_editor_llm(
            client=client,
            model="test-model",
            candidate=candidate,
            source_material="审核通过后可查看申报记录。",
            scenario_intent="了解审核通过后的界面操作",
            reader_need="查看审核状态并导出申报记录",
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            source_chunk_meta=frozen_meta,
        )
        self.assertEqual("edited", status)
        self.assertEqual("申报通过后，可以在平台上查看或导出哪些信息？", edited["question"])
        self.assertEqual({"question", "question_type"}, set(edited))
        user_content = client.messages[1]["content"]
        self.assertIn("视觉回答依据", user_content)
        self.assertNotIn("required_image_refs", user_content)
        self.assertNotIn("evidence_mode", user_content)

    def test_editor_preserves_binary_permission_question_form(self):
        records = []
        client = _StaticChatClient({"question": "申报成功后，人员缴费基数应当如何修改？"})
        candidate = {"question": "申报成功后，还能修改人员缴费基数吗？", "question_type": "简答题"}
        edited, status = call_question_editor_llm(
            client=client,
            model="test-model",
            candidate=candidate,
            source_material="一旦申报成功，将不能再修改人员缴费基数。",
            scenario_intent="确认申报成功后能否修改人员缴费基数",
            reader_need="确认申报成功后的修改限制",
            qa_detail_mode="point",
            prompt_language="zh",
            request_timeout=10,
            debug_writer=records.append,
        )
        self.assertEqual("edited", status)
        self.assertEqual(candidate["question"], edited["question"])
        self.assertEqual("preserved_binary_question_form", records[0]["editor_decision"])

    def test_editor_replaces_document_deictic_with_business_object(self):
        client = _StaticChatClient({"question": "这份操作说明适用于哪些参保单位？"})
        edited, status = call_question_editor_llm(
            client=client,
            model="test-model",
            candidate={"question": "本操作说明适用于哪些参保单位？", "question_type": "简答题"},
            source_material="该操作文档适用于参加城镇职工养老保险的参保单位。",
            scenario_intent="明确适用单位",
            reader_need="判断是否属于适用范围",
            qa_detail_mode="point",
            prompt_language="zh",
            request_timeout=10,
            source_chunk_meta={
                "qa_generation_unit_material_paths": [
                    "陕西省通知>单位缴费基数诚信申报操作使用说明（参保单位版）>一.适用范围"
                ],
            },
        )
        self.assertEqual("edited", status)
        self.assertNotIn("这份操作说明", edited["question"])
        self.assertIn("单位缴费基数诚信申报操作使用说明", edited["question"])

    def test_evidence_renderer_separates_text_and_image_blocks(self):
        index = QADocumentEvidenceIndex(
            chunks=[
                {
                    "chunk_id": "primary-id",
                    "chunk_index": 1,
                    "title_path": "内部标题 > 第一条",
                    "section_path": "1",
                    "text": "正文事实。\n【图片描述：旧的扁平图片事实。】",
                    "retrieval_text": "正文事实。",
                },
                {
                    "chunk_id": "supplement-id",
                    "chunk_index": 2,
                    "title_path": "内部标题 > 第二条",
                    "section_path": "2",
                    "text": "补充事实。",
                    "retrieval_text": "补充事实。",
                },
            ],
            embeddings=[[1.0], [0.5]],
        )
        generation_unit = index.build_generation_unit(
            source_chunk_index=1,
            source_unit={
                "source_chunk_indexes": [1],
                "required_material_ids": ["section-1"],
                "material_source_chunk_indexes": {"section-1": [1]},
                "material_ref_map": {"主材料-A": "section-1"},
                "image_ref_map": {"图片-A": "image-1"},
                "required_image_ids": ["image-1"],
                "evidence_mode": "mixed",
                "prompt_materials": [
                    {
                        "material_ref": "主材料-A",
                        "node_path": "内部标题 > 第一条",
                        "text_content": "正文事实。",
                        "image_materials": [{"image_ref": "图片-A", "description": "页面提供导出按钮。"}],
                    }
                ],
            },
            question="审核通过后可以做什么？",
            retrieval_result={
                "selected_windows": [
                    EvidenceWindow(
                        window_id="window-1",
                        chunk_ids=("supplement-id",),
                        reason="atomic",
                        text="补充事实。",
                        title_path="内部标题 > 第二条",
                        anchor_chunk_ids=("supplement-id",),
                    )
                ],
                "trace": {},
            },
            final_evidence_k=5,
            evidence_token_budget=4000,
        )
        rendered = generation_unit["qa_generation_unit_text"]
        self.assertIn("正文证据-1", rendered)
        self.assertIn("图片证据-1", rendered)
        self.assertIn("补充正文证据-1", rendered)
        self.assertNotIn("primary-id", rendered)
        self.assertNotIn("旧的扁平图片事实", rendered)
        image_ref = generation_unit["llm_evidence_ref_map"]["图片证据-1"]
        self.assertEqual("image-1", image_ref["image_id"])
        self.assertEqual("primary_visual", image_ref["role"])
        self.assertIn(
            "页面提供导出按钮。",
            generation_unit["llm_evidence_text_by_ref"]["图片证据-1"],
        )

    def test_mixed_answer_requires_text_and_visual_citations(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "answer": "审核通过后，单位可查看申报记录并使用导出按钮导出信息。",
                        "answer_explanation": "审核状态和导出操作共同支持后续查询与留档。",
                        "source_fact_text": "审核通过后可查看申报记录；页面提供导出按钮。",
                        "evidence_usage": [
                            {"evidence_ref": "正文证据-1", "role": "primary_source"},
                            {"evidence_ref": "图片证据-1", "role": "primary_visual"},
                        ],
                    }
                ]
            }
        )
        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={"question": "审核通过后可以做什么？", "question_type": "简答题"},
            generation_unit=_generation_unit(mode="mixed", required_images=["image-1"]),
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )
        self.assertEqual("ok", reason)
        self.assertEqual("mixed", item["evidence_mode"])
        self.assertEqual("image-1", item["evidence_usage"][1]["image_id"])
        self.assertIn("正文证据-1", item["qa_evaluation_evidence_text"])
        self.assertIn("图片证据-1", item["qa_evaluation_evidence_text"])
        self.assertNotIn("正文证据-2", item["qa_evaluation_evidence_text"])
        self.assertIn("图片证据-1", client.messages[1]["content"])
        self.assertNotIn("typed_primary_materials", client.messages[1]["content"])

    def test_mixed_answer_without_visual_citation_is_rejected(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "answer": "审核通过后可查看申报记录。",
                        "answer_explanation": "正文说明了查询方式。",
                        "source_fact_text": "审核通过后可查看申报记录。",
                        "evidence_usage": [{"evidence_ref": "正文证据-1", "role": "primary_source"}],
                    }
                ]
            }
        )
        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={"question": "审核通过后可以做什么？", "question_type": "简答题"},
            generation_unit=_generation_unit(mode="mixed", required_images=["image-1"]),
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )
        self.assertIsNone(item)
        self.assertEqual("missing_required_visual_evidence", reason)

    def test_visual_contract_gets_one_targeted_answer_retry(self):
        client = _SequentialChatClient(
            [
                {
                    "items": [
                        {
                            "answer": "审核通过后可查看申报记录。",
                            "answer_explanation": "正文说明了查询方式。",
                            "source_fact_text": "审核通过后可查看申报记录。",
                            "evidence_usage": [{"evidence_ref": "正文证据-1", "role": "primary_source"}],
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "answer": "审核通过后可查看申报记录，并通过页面导出按钮导出信息。",
                            "answer_explanation": "正文和图片按钮共同支持查询与导出。",
                            "source_fact_text": "审核通过后可查看申报记录；页面提供导出按钮。",
                            "evidence_usage": [
                                {"evidence_ref": "正文证据-1", "role": "primary_source"},
                                {"evidence_ref": "图片证据-1", "role": "primary_visual"},
                            ],
                        }
                    ]
                },
            ]
        )
        answer_audit = {}
        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={"question": "审核通过后可以做什么？", "question_type": "简答题"},
            generation_unit=_generation_unit(mode="mixed", required_images=["image-1"]),
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
            answer_audit=answer_audit,
        )
        self.assertEqual("ok", reason)
        self.assertIsNotNone(item)
        self.assertEqual(2, len(client.messages))
        self.assertIn("必须使用并在 evidence_usage 中引用图片证据", client.messages[1][1]["content"])
        self.assertEqual(2, answer_audit["answer_attempt_count"])
        self.assertEqual("missing_required_visual_evidence", answer_audit["answer_retry_reason"])

    def test_summary_answer_requires_all_frozen_required_materials(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "answer": "需要提交申请表。",
                        "answer_explanation": "回答只覆盖材料。",
                        "source_fact_text": "提交申请表。",
                        "evidence_usage": [{"evidence_ref": "正文证据-1", "role": "primary_source"}],
                    }
                ]
            }
        )
        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={"question": "办理需要哪些材料和多长时间？", "question_type": "简答题"},
            generation_unit=_generation_unit(required_materials=["section-1", "section-2"]),
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )
        self.assertIsNone(item)
        self.assertEqual("incomplete_primary_material_coverage", reason)

    def test_consolidation_and_debug_have_no_difficulty_fields(self):
        qa_item = {
            "id": "qa-1",
            "question": "谁负责办理？",
            "answer": "登记机关负责办理。",
            "source": "c1",
            "source_fact_text": "登记机关负责办理。",
            "qa_evaluation_evidence_text": "正文证据-1\n正文：登记机关负责办理。",
            "question_type": "简答题",
            "evidence_usage": [
                {
                    "evidence_ref": "正文证据-1",
                    "role": "primary_source",
                    "chunk_id": "c1",
                    "chunk_index": 1,
                    "title_path": "办理要求",
                    "material_id": "section-1",
                }
            ],
        }
        entry = build_consolidated_entry(
            task_id="task-contract",
            original_filename="contract.md",
            facts=[],
            categorized_facts=[],
            qa_data=[qa_item],
            evaluation_results=None,
            filtered_qa_data=None,
            include_evaluation=False,
            evaluation_method="llm",
            filter_by_threshold=False,
            score_threshold=0.7,
            chunk_size=600,
            qa_per_chunk=1,
            qa_detail_mode="point",
            prompt_language="zh",
            llm_model="test-model",
        )
        consolidated = entry["payload"]["items"][0]
        self.assertNotIn("difficulty_level", consolidated)
        self.assertNotIn("difficulty_score", consolidated)
        self.assertNotIn("question_type_reason", consolidated)
        self.assertEqual(
            "正文证据-1\n正文：登记机关负责办理。",
            consolidated["qa_evaluation_evidence_text"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "qa-debug.sqlite3")
            with patch.dict("os.environ", {"QA_DEBUG_DB_PATH": db_path}):
                upsert_qa_debug_items([consolidated])
                debug_payload = get_debug_map([consolidated["id"]])[consolidated["id"]]
        self.assertNotIn("difficulty_level", debug_payload)
        self.assertNotIn("difficulty_score", debug_payload)
        self.assertEqual(
            "正文证据-1\n正文：登记机关负责办理。",
            debug_payload["qa_evaluation_evidence_text"],
        )

    def test_zero_final_evidence_k_is_preserved(self):
        runtime = parse_one_step_pipeline_runtime(
            {"chunk_size": 600, "qa_per_chunk": 1, "final_evidence_k": 0}
        )
        self.assertEqual(0, runtime.final_evidence_k)

    def test_generation_latency_percentiles_include_editor_and_answer(self):
        stats = _unit_latency_percentiles(
            [
                {"timing": {"chunk_total_seconds": 10, "question_editor_seconds": 3, "answer_generation_seconds": 4}},
                {"timing": {"chunk_total_seconds": 20, "question_editor_seconds": 5, "answer_generation_seconds": 8}},
                {"timing": {"chunk_total_seconds": 40, "question_editor_seconds": 9, "answer_generation_seconds": 16}},
            ]
        )
        self.assertEqual(20.0, stats["chunk_total_seconds"]["p50"])
        self.assertGreater(stats["chunk_total_seconds"]["p95"], 35.0)
        self.assertEqual(5.0, stats["question_editor_seconds"]["p50"])
        self.assertEqual(8.0, stats["answer_generation_seconds"]["p50"])


if __name__ == "__main__":
    unittest.main()
