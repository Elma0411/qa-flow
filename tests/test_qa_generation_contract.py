import json
import unittest

from qa.generation.qa_generation_flow import (
    call_candidate_question_llm,
    call_evidence_answer_llm,
)
from qa.prompts.qa_generation_prompts import (
    build_candidate_question_system_prompt,
    build_evidence_answer_system_prompt,
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

        self.assertIn("question 只能包含一个完整问句", zh_prompt)
        self.assertIn("每题最多使用一个问号", zh_prompt)
        self.assertIn("“总结型”描述的是答案组织方式", zh_prompt)
        self.assertNotIn("source_anchor_text", zh_prompt)
        self.assertIn("exactly one standalone question sentence", en_prompt)
        self.assertIn("Use at most one question mark", en_prompt)
        self.assertIn('"Summary" describes the expected answer', en_prompt)
        self.assertNotIn("source_anchor_text", en_prompt)

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
        self.assertIn("只输出包含 1 个 item", zh_prompt)
        self.assertNotIn('{"items":[]}', zh_prompt)
        self.assertIn("Do not output an empty items list as a quality decision", en_prompt)
        self.assertIn("exactly one item", en_prompt)
        self.assertNotIn('{"items":[]}', en_prompt)

    def test_candidate_generation_discards_source_anchor_text(self):
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
                "retrieval_query": "费用承担",
                "must_have_terms": ["费用", "承担"],
                "answer_scope": "source_primary",
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
                "qa_generation_unit_text": "【主来源块】费用由责任主体承担。",
                "evidence_chunk_ids": [],
                "qa_generation_unit_id": "unit-1",
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


if __name__ == "__main__":
    unittest.main()
