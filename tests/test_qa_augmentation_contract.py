import json
import unittest

from qa.augmentation import augment_qa_pairs


class _SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create_chat_completion_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class QAAugmentationContractTests(unittest.TestCase):
    def test_augmented_question_passes_through_the_same_editor(self):
        client = _SequenceClient([
            json.dumps([{
                "question": "根据该条例，女职工还能休多少天？",
                "answer": "可以增加六十天产假。",
            }], ensure_ascii=False),
            json.dumps({
                "decision": "rewrite",
                "question": "女职工生育后可以增加多少天产假？",
                "reason": "去掉来源视角和模糊指代",
                "required_material_refs": ["主材料-A"],
                "optional_material_refs": [],
                "evidence_mode": "text",
                "required_image_refs": [],
            }, ensure_ascii=False),
        ])
        source = [{
            "question": "女职工生育后可以增加多少天产假？",
            "answer": "可以增加六十天产假。",
            "question_type": "简答题",
            "source_fact_text": "女职工依法生育子女的，增加产假六十天。",
            "qa_generation_scenario_intent": "询问增加产假天数",
            "qa_generation_reader_need": "了解生育后的额外产假",
            "qa_generation_unit_mode": "point",
        }]

        augmented = augment_qa_pairs(
            source,
            augment_per_qa=1,
            client=client,
            model="test-model",
            max_workers=1,
        )

        self.assertEqual(1, len(augmented))
        self.assertEqual("女职工生育后可以增加多少天产假？", augmented[0]["question"])

    def test_summary_augmentation_editor_receives_atomic_needs(self):
        client = _SequenceClient([
            json.dumps([{
                "question": "办理申请需要准备什么，多久办完？",
                "answer": "提交申请表，五个工作日内办结。",
            }], ensure_ascii=False),
            json.dumps({
                "question": "办理申请需要提交什么材料，多久可以办结？",
            }, ensure_ascii=False),
        ])
        source = [{
            "question": "办理申请需要提交什么材料，多久可以办结？",
            "answer": "提交申请表，五个工作日内办结。",
            "question_type": "简答题",
            "source_fact_text": "提交申请表；五个工作日内办结。",
            "qa_generation_scenario_intent": "了解申请材料和办理时限",
            "qa_generation_reader_need": "一次准备好申请",
            "qa_generation_unit_mode": "summary",
            "qa_generation_summary_hops": [
                {
                    "hop_id": "hop-1",
                    "sub_question": "需要提交什么申请材料？",
                    "material_id": "section-1",
                    "evidence_mode": "text",
                    "required_image_ids": [],
                },
                {
                    "hop_id": "hop-2",
                    "sub_question": "办理时限是多久？",
                    "material_id": "section-2",
                    "evidence_mode": "text",
                    "required_image_ids": [],
                },
            ],
        }]

        augmented = augment_qa_pairs(
            source,
            augment_per_qa=1,
            client=client,
            model="test-model",
            max_workers=1,
        )

        self.assertEqual(1, len(augmented))
        editor_content = client.calls[1]["messages"][1]["content"]
        self.assertIn("需要提交什么申请材料", editor_content)
        self.assertIn("办理时限是多久", editor_content)
        self.assertNotIn("section-1", editor_content)


if __name__ == "__main__":
    unittest.main()
