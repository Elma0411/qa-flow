import unittest

from qa.generation.qa_generation_flow import _summary_question_shape_reason
from qa.generation.text_quality_filters import contains_ambiguous_reference
from qa.grounding.source_fact_grounding import validate_source_fact_grounding


class GenerationQualityFilterTests(unittest.TestCase):
    def test_summary_question_shape_only_rejects_generic_flat_lists(self):
        self.assertEqual(
            "",
            _summary_question_shape_reason(
                "考勤异常处理需要覆盖哪些步骤？",
                language_code="zh",
            ),
        )
        self.assertEqual(
            "",
            _summary_question_shape_reason(
                "资产验收需要哪些材料和记录？",
                language_code="zh",
            ),
        )
        self.assertEqual(
            "summary_question_too_shallow_list",
            _summary_question_shape_reason("该部分有哪些内容？", language_code="zh"),
        )
        self.assertEqual(
            "",
            _summary_question_shape_reason(
                "Which records and approvals are required for asset acceptance?",
                language_code="en",
            ),
        )
        self.assertEqual(
            "summary_question_too_shallow_list",
            _summary_question_shape_reason("What items are listed?", language_code="en"),
        )

    def test_summary_grounding_accepts_majority_supported_segments(self):
        fact = (
            "外观检查由使用部门负责；"
            "数量核对由使用部门负责；"
            "异常处理由采购专员跟进"
        )
        chunk = "外观检查和数量核对由使用部门负责。供应商资质由采购专员审核。"

        ok, reason = validate_source_fact_grounding(
            fact,
            chunk_text=chunk,
            qa_detail_mode="summary",
            language_code="zh",
        )

        self.assertTrue(ok, reason)

    def test_unresolved_reference_filter_allows_local_headed_references(self):
        self.assertFalse(
            contains_ambiguous_reference(
                "资产验收包括外观检查和数量核对，其中外观检查由使用部门负责。",
                language_code="zh",
            )
        )
        self.assertTrue(
            contains_ambiguous_reference(
                "其中外观检查由使用部门负责。",
                language_code="zh",
            )
        )
        self.assertFalse(
            contains_ambiguous_reference(
                "This asset acceptance process requires inspection records.",
                language_code="en",
            )
        )
        self.assertTrue(
            contains_ambiguous_reference(
                "This requires inspection records.",
                language_code="en",
            )
        )


if __name__ == "__main__":
    unittest.main()
