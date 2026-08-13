import json
import unittest

from qa.generation.qa_generation_flow import (
    call_candidate_question_llm,
    call_evidence_answer_llm,
)
from qa.pipeline_runtime import parse_one_step_pipeline_runtime
from qa.prompts.qa_generation_prompts import (
    build_candidate_question_system_prompt,
    build_evidence_answer_system_prompt,
    build_question_editor_system_prompt,
)
from qa.validation import validate_and_normalize_item_with_reason


class _StaticChatClient:
    def __init__(self, payload):
        self.payload = payload

    def create_chat_completion_text(self, **_kwargs):
        return json.dumps(self.payload, ensure_ascii=False)


class QAGenerationContractTests(unittest.TestCase):
    def test_summary_candidate_prompt_requires_one_question_per_item(self):
        zh_prompt = build_candidate_question_system_prompt(
            language_code="zh",
            language_instruction="请使用中文。",
            candidate_count=2,
            question_type_plan=["简答题", "简答题"],
            few_shot_examples=None,
            qa_detail_mode="summary",
        )
        en_prompt = build_candidate_question_system_prompt(
            language_code="en",
            language_instruction="Use English.",
            candidate_count=2,
            question_type_plan=["简答题", "简答题"],
            few_shot_examples=None,
            qa_detail_mode="summary",
        )

        self.assertIn("每个 item 只写一个围绕同一读者需求的完整问句", zh_prompt)
        self.assertIn("“总结”是答案可以组织多个相关事实", zh_prompt)
        self.assertNotIn("source_anchor_text", zh_prompt)
        self.assertIn("one standalone question around one coherent reader need", en_prompt)
        self.assertIn("Summary means the answer may organize related facts", en_prompt)
        self.assertNotIn("source_anchor_text", en_prompt)

    def test_candidate_prompt_requires_natural_user_questions(self):
        zh_prompt = build_candidate_question_system_prompt(
            language_code="zh",
            language_instruction="请使用中文。",
            candidate_count=1,
            question_type_plan=["简答题"],
            few_shot_examples=None,
            knowledge_category="法律法规/婚姻登记",
            qa_detail_mode="point",
        )
        en_prompt = build_candidate_question_system_prompt(
            language_code="en",
            language_instruction="Use English.",
            candidate_count=1,
            question_type_plan=["简答题"],
            few_shot_examples=None,
            knowledge_category="法律法规/婚姻登记",
            qa_detail_mode="point",
        )

        self.assertIn("严格遵循输入给出的 scenario_intent 和 reader_need", zh_prompt)
        self.assertIn("请使用中文。", zh_prompt)
        self.assertIn("不要把原文一句话的前半句改成问题", zh_prompt)
        self.assertIn("例如写", zh_prompt)
        self.assertIn("生育后还能增加多少天产假", zh_prompt)
        self.assertIn("默认提问者：办事人", zh_prompt)
        self.assertIn("Follow the supplied scenario_intent and reader_need", en_prompt)
        self.assertIn("Use English.", en_prompt)
        self.assertIn("Do not convert the first half", en_prompt)
        self.assertIn("Example: write", en_prompt)
        self.assertIn("Default questioner: An applicant", en_prompt)

    def test_candidate_input_contains_only_readable_source_material(self):
        class RecordingClient:
            def __init__(self):
                self.messages = []

            def create_chat_completion_text(self, **kwargs):
                self.messages = kwargs["messages"]
                return json.dumps(
                    {
                        "items": [
                            {
                                "question": "婚前医学检查费用如何承担？",
                                "question_type": "简答题",
                                "difficulty_level": "中等",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

        client = RecordingClient()
        call_candidate_question_llm(
            client=client,
            model="test-model",
            source_chunk_text="结婚登记前参加婚前医学检查的费用按规定承担。",
            source_chunk_meta={
                "chunk_id": "chunk-1",
                "title_path": "婚姻登记 > 婚前医学检查",
            },
            candidate_count=1,
            prompt_language="zh",
            question_type_plan=["简答题"],
            few_shot_examples=None,
            request_timeout=10,
            knowledge_category="法律法规/婚姻登记",
            qa_detail_mode="point",
        )

        user_content = client.messages[1]["content"]
        self.assertIn("主来源材料", user_content)
        self.assertIn("结婚登记前参加婚前医学检查的费用按规定承担。", user_content)
        self.assertNotIn("chunk-1", user_content)
        self.assertNotIn("title_path", user_content)
        self.assertNotIn("婚姻登记 > 婚前医学检查", user_content)

    def test_answer_prompt_does_not_reject_candidate_as_quality_decision(self):
        zh_prompt = build_evidence_answer_system_prompt(
            language_code="zh",
            language_instruction="请使用中文。",
            qa_detail_mode="summary",
        )
        en_prompt = build_evidence_answer_system_prompt(
            language_code="en",
            language_instruction="Use English.",
            qa_detail_mode="summary",
        )

        self.assertIn("不得基于质量判断输出空 items", zh_prompt)
        self.assertIn("必须引用回答该问题所必需的每份主材料", zh_prompt)
        self.assertIn("只输出包含 1 个 item", zh_prompt)
        self.assertNotIn('{"items":[]}', zh_prompt)
        self.assertIn("Do not output an empty items list as a quality decision", en_prompt)
        self.assertIn("exactly one item", en_prompt)
        self.assertIn("cite every primary material required by the question", en_prompt)
        self.assertNotIn('{"items":[]}', en_prompt)
        self.assertNotIn("- retrieval_query\n", zh_prompt)
        self.assertNotIn("- must_have_terms\n", zh_prompt)
        self.assertNotIn("- retrieval_query\n", en_prompt)
        self.assertNotIn("- must_have_terms\n", en_prompt)

    def test_question_editor_prompt_contains_clause_shape_rewrite_examples(self):
        zh_prompt = build_question_editor_system_prompt(
            language_code="zh",
            language_instruction="请使用中文。",
            qa_detail_mode="point",
        )
        en_prompt = build_question_editor_system_prompt(
            language_code="en",
            language_instruction="Use English.",
            qa_detail_mode="point",
        )

        self.assertIn("女职工生育后，可以额外休多少天产假", zh_prompt)
        self.assertIn("把原文的条件从句直接搬到逗号前", zh_prompt)
        self.assertIn("not natural merely because it is grammatical", en_prompt)

    def test_answer_prompt_requires_standalone_reader_explanation(self):
        zh_prompt = build_evidence_answer_system_prompt(
            language_code="zh",
            language_instruction="请使用中文。",
            qa_detail_mode="summary",
        )
        en_prompt = build_evidence_answer_system_prompt(
            language_code="en",
            language_instruction="Use English.",
            qa_detail_mode="summary",
        )

        self.assertIn("1 到 2 句完整、面向读者的说明", zh_prompt)
        self.assertIn("不是证据追踪，也不是原文句子的后半截", zh_prompt)
        self.assertIn("source_fact_text 和 evidence_usage 完成", zh_prompt)
        self.assertIn("标签仅用于证据追踪", zh_prompt)
        self.assertIn("不要以“该优惠、该答案、此项、上述、其中、它”等指代词开头", zh_prompt)
        self.assertIn("农村独生子女或双女户父母参加新型农村合作医疗时", zh_prompt)
        self.assertIn("办理流程、申请手续、主管机关或期限不得自行补全", zh_prompt)
        self.assertIn("complete, reader-facing clarification", en_prompt)
        self.assertIn("not a citation trace or a continuation of a source sentence", en_prompt)
        self.assertIn('not a deictic phrase such as "this benefit"', en_prompt)
        self.assertIn("Eligible rural one-child or two-daughter families", en_prompt)
        self.assertIn("Do not invent an amount, ratio, procedure", en_prompt)

    def test_zero_final_evidence_k_is_preserved(self):
        runtime = parse_one_step_pipeline_runtime(
            {
                "chunk_size": 600,
                "qa_per_chunk": 1,
                "final_evidence_k": 0,
            }
        )

        self.assertEqual(0, runtime.final_evidence_k)

    def test_candidate_generation_discards_unrequested_planning_fields(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "question": "其中应当如何处理？",
                        "source_anchor_text": "旧字段不应进入候选题结果。",
                        "retrieval_query": "处理方式",
                        "must_have_terms": ["处理"],
                        "answer_scope_hint": "source_primary",
                        "question_type": "简答题",
                        "difficulty_level": "中等",
                    }
                ]
            }
        )

        items = call_candidate_question_llm(
            client=client,
            model="test-model",
            source_chunk_text="输入块只包含另一段文字。",
            source_chunk_meta={"chunk_id": "chunk-1"},
            candidate_count=1,
            prompt_language="zh",
            question_type_plan=["简答题"],
            few_shot_examples=None,
            request_timeout=10,
            qa_detail_mode="summary",
        )

        self.assertEqual(1, len(items))
        self.assertEqual("其中应当如何处理？", items[0]["question"])
        self.assertNotIn("source_anchor_text", items[0])
        self.assertNotIn("retrieval_query", items[0])
        self.assertNotIn("must_have_terms", items[0])
        self.assertNotIn("answer_scope_hint", items[0])

    def test_answer_semantic_rules_do_not_drop_normalized_item(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "question": "模型自行改写的问题？",
                        "answer": "其中由相关人员处理。",
                        "answer_explanation": "其中给出了处理主体。",
                        "source_fact_text": "这条事实并未出现在证据中。",
                        "source": "模型来源",
                        "evidence_usage": [
                            {
                                "evidence_ref": "主材料-1",
                                "role": "primary_source",
                                "snippet": "费用由责任主体承担。",
                                "usage": "支持费用承担结论",
                            }
                        ],
                        "question_type": "简答题",
                        "difficulty_level": "中等",
                    }
                ]
            }
        )
        candidate_question = "费用承担规则如何规定？"

        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={
                "question": candidate_question,
                "source_anchor_text": "旧字段不应进入答案生成输入。",
                "question_type": "简答题",
                "difficulty_level": "中等",
            },
            generation_unit={
                "source_chunk": {
                    "chunk_id": "chunk-1",
                    "chunk_index": 1,
                    "text": "费用由责任主体承担。",
                },
                "source_unit_text": "费用由责任主体承担。",
                "qa_generation_unit_text": "【主来源材料】\n主材料-1\n费用由责任主体承担。",
                "evidence_chunk_ids": [],
                "qa_generation_unit_id": "unit-1",
                "llm_evidence_ref_map": {
                    "主材料-1": {
                        "chunk_id": "chunk-1",
                        "chunk_index": 1,
                        "role": "primary_source",
                    }
                },
            },
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )

        self.assertEqual("ok", reason)
        self.assertIsNotNone(item)
        self.assertEqual(candidate_question, item["question"])
        self.assertEqual("这条事实并未出现在证据中。", item["source_fact_text"])
        self.assertNotIn("source_anchor_text", item)
        self.assertEqual("chunk-1", item["evidence_usage"][0]["chunk_id"])
        self.assertEqual("主材料-1", item["evidence_usage"][0]["evidence_ref"])

    def test_answer_source_attribution_uses_actually_cited_primary_materials(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "question": "办理需要哪些材料和多长时间？",
                        "answer": "需提交申请表，办理期限为五个工作日。",
                        "answer_explanation": "申请材料和办理期限共同构成办理要求。",
                        "source_fact_text": "提交申请表；五个工作日内办结。",
                        "source": "文本内容",
                        "evidence_usage": [
                            {"evidence_ref": "主材料-2", "snippet": "提交申请表", "usage": "材料"},
                            {"evidence_ref": "主材料-3", "snippet": "五个工作日", "usage": "时限"},
                        ],
                        "question_type": "简答题",
                        "difficulty_level": "中等",
                    }
                ]
            }
        )
        source_chunks = [
            {"chunk_id": "c1", "chunk_index": 1, "title_path": "总则", "text": "总则。"},
            {"chunk_id": "c2", "chunk_index": 2, "title_path": "材料", "text": "提交申请表。"},
            {"chunk_id": "c3", "chunk_index": 3, "title_path": "时限", "text": "五个工作日内办结。"},
        ]
        ref_map = {
            f"主材料-{index}": {
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "title_path": chunk["title_path"],
                "role": "primary_source",
            }
            for index, chunk in enumerate(source_chunks, 1)
        }

        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={"question": "办理需要哪些材料和多长时间？", "question_type": "简答题"},
            generation_unit={
                "source_chunk": source_chunks[0],
                "source_chunks": source_chunks,
                "source_unit_text": "\n".join(chunk["text"] for chunk in source_chunks),
                "qa_generation_unit_text": "主材料-1\n总则。\n主材料-2\n提交申请表。\n主材料-3\n五个工作日内办结。",
                "evidence_chunk_ids": [],
                "qa_generation_unit_id": "unit-1",
                "llm_evidence_ref_map": ref_map,
            },
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )

        self.assertEqual("ok", reason)
        self.assertEqual("c2", item["source_chunk_id"])
        self.assertEqual(2, item["source_chunk_index"])
        self.assertEqual("材料", item["source_chunk_title_path"])
        self.assertEqual(["c2", "c3"], item["source_chunk_ids"])
        self.assertEqual([2, 3], item["source_chunk_indexes"])
        self.assertEqual(["材料", "时限"], item["source_chunk_title_paths"])

    def test_summary_answer_requires_primary_coverage_for_each_bound_material(self):
        client = _StaticChatClient(
            {
                "items": [
                    {
                        "question": "办理需要哪些材料和多长时间？",
                        "answer": "需要提交申请表。",
                        "answer_explanation": "回答只覆盖了材料。",
                        "source_fact_text": "提交申请表。",
                        "source": "文本内容",
                        "evidence_usage": [
                            {"evidence_ref": "主材料-1", "snippet": "提交申请表", "usage": "材料"},
                        ],
                        "question_type": "简答题",
                        "difficulty_level": "中等",
                    }
                ]
            }
        )
        source_chunks = [
            {"chunk_id": "c1", "chunk_index": 1, "title_path": "材料", "text": "提交申请表。"},
            {"chunk_id": "c2", "chunk_index": 2, "title_path": "时限", "text": "五日内办结。"},
        ]
        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={"question": "办理需要哪些材料和多长时间？", "question_type": "简答题"},
            generation_unit={
                "source_chunk": source_chunks[0],
                "source_chunks": source_chunks,
                "source_unit": {
                    "material_ids": ["section-1", "section-2"],
                    "material_source_chunk_indexes": {
                        "section-1": [1],
                        "section-2": [2],
                    },
                },
                "source_unit_text": "提交申请表。\n五日内办结。",
                "qa_generation_unit_text": "主材料-1\n提交申请表。\n主材料-2\n五日内办结。",
                "evidence_chunk_ids": [],
                "qa_generation_unit_id": "unit-coverage",
                "llm_evidence_ref_map": {
                    "主材料-1": {"chunk_id": "c1", "chunk_index": 1, "role": "primary_source"},
                    "主材料-2": {"chunk_id": "c2", "chunk_index": 2, "role": "primary_source"},
                },
            },
            qa_detail_mode="summary",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )

        self.assertIsNone(item)
        self.assertEqual("incomplete_summary_primary_coverage", reason)

    def test_answer_input_hides_internal_metadata_and_planning_fields(self):
        class RecordingClient:
            def __init__(self):
                self.messages = []

            def create_chat_completion_text(self, **kwargs):
                self.messages = kwargs["messages"]
                return json.dumps(
                    {
                        "items": [
                            {
                                "question": "费用由谁承担？",
                                "answer": "费用由责任主体承担。",
                                "answer_explanation": "责任主体承担相关费用。",
                                "source_fact_text": "费用由责任主体承担。",
                                "source": "文本内容",
                                "question_type": "简答题",
                                "difficulty_level": "简单",
                                "evidence_usage": [
                                    {
                                        "evidence_ref": "主材料-1",
                                        "snippet": "费用由责任主体承担。",
                                        "usage": "支持答案",
                                    }
                                ],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )

        client = RecordingClient()
        item, reason = call_evidence_answer_llm(
            client=client,
            model="test-model",
            candidate={
                "question": "费用由谁承担？",
                "question_type": "简答题",
                "difficulty_level": "简单",
            },
            generation_unit={
                "source_chunk": {
                    "chunk_id": "secret-chunk-id",
                    "chunk_index": 42,
                    "title_path": "内部���题 > 不应出现",
                    "text": "费用由责任主体承担。",
                },
                "source_unit_text": "费用由责任主体承担。",
                "qa_generation_unit_text": "【主来源材料】\n主材料-1\n费用由责任主体承担。",
                "evidence_chunk_ids": ["supplement-id"],
                "qa_generation_unit_id": "unit-1",
                "llm_evidence_ref_map": {
                    "主材料-1": {
                        "chunk_id": "secret-chunk-id",
                        "chunk_index": 42,
                        "role": "primary_source",
                    }
                },
            },
            qa_detail_mode="point",
            prompt_language="zh",
            request_timeout=10,
            item_normalizer_with_reason=validate_and_normalize_item_with_reason,
            source_override_handler=lambda *_args, **_kwargs: None,
        )

        self.assertEqual("ok", reason)
        self.assertIsNotNone(item)
        user_content = client.messages[1]["content"]
        self.assertIn("主材料-1", user_content)
        self.assertIn("费用由责任主体承担。", user_content)
        self.assertNotIn("secret-chunk-id", user_content)
        self.assertNotIn("内部标题", user_content)
        self.assertNotIn("retrieval_query", user_content)
        self.assertNotIn("must_have_terms", user_content)
        self.assertEqual("secret-chunk-id", item["evidence_usage"][0]["chunk_id"])

    def test_pipeline_settings_includes_retrieval_module(self):
        from pathlib import Path

        app_js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        start = app_js.index("function openPipelineSettingsModal()")
        end = app_js.index("function createModuleCard", start)
        modal_source = app_js[start:end]
        self.assertIn("'pipeline.retrieval'", modal_source)

    def test_evidence_rendering_uses_readable_refs_and_preserves_mapping(self):
        from qa.generation.evidence_units import QADocumentEvidenceIndex
        from qa.retrieval import EvidenceWindow

        index = QADocumentEvidenceIndex(
            chunks=[
                {
                    "chunk_id": "primary-id",
                    "chunk_index": 1,
                    "title_path": "内部标题 > 第一条",
                    "section_path": "1",
                    "text": "主材料事实。",
                    "retrieval_text": "主材料事实。",
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
            source_unit={"source_chunk_indexes": [1]},
            question="主材料事实是什么？",
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
        self.assertIn("主材料-1", rendered)
        self.assertIn("检索证据-1", rendered)
        self.assertNotIn("primary-id", rendered)
        self.assertNotIn("supplement-id", rendered)
        self.assertIn("标题路径：内部标题 > 第二条", rendered)
        self.assertEqual("primary-id", generation_unit["llm_evidence_ref_map"]["主材料-1"]["chunk_id"])
        self.assertEqual("supplement-id", generation_unit["llm_evidence_ref_map"]["检索证据-1"]["chunk_id"])


if __name__ == "__main__":
    unittest.main()
