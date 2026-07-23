import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.unsupervised_evaluation.model_options import (
    normalize_evaluation_model_name,
    resolve_evaluation_model_path,
    validate_evaluation_model_name,
)


class UnsupervisedModelOptionsTests(unittest.TestCase):
    def test_auto_is_an_empty_override(self):
        self.assertEqual(
            normalize_evaluation_model_name("auto", kind="coverage_embedding"),
            "",
        )
        self.assertEqual(
            normalize_evaluation_model_name(None, kind="answerability_qa"),
            "",
        )

    def test_new_models_are_allowed(self):
        self.assertEqual(
            normalize_evaluation_model_name(
                "xlm_roberta_large_xnli",
                kind="faithfulness_nli",
            ),
            "xlm_roberta_large_xnli",
        )
        self.assertEqual(
            normalize_evaluation_model_name(
                "deepset_xlm_roberta_large_squad2",
                kind="answerability_qa",
            ),
            "deepset_xlm_roberta_large_squad2",
        )
        self.assertEqual(
            normalize_evaluation_model_name(
                "qwen3_embedding_4b",
                kind="coverage_embedding",
            ),
            "qwen3_embedding_4b",
        )

    def test_model_kind_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_evaluation_model_name(
                "qwen3_embedding_0_6b",
                kind="faithfulness_nli",
            )

    def test_selected_model_requires_a_local_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "qwen3_embedding_0_6b"
            model_dir.mkdir()
            with patch(
                "app.services.unsupervised_evaluation.model_options.resolve_model_reference",
                return_value=str(model_dir),
            ):
                self.assertEqual(
                    validate_evaluation_model_name(
                        "qwen3_embedding_0_6b",
                        kind="coverage_embedding",
                    ),
                    "qwen3_embedding_0_6b",
                )
                self.assertEqual(
                    resolve_evaluation_model_path(
                        "qwen3_embedding_0_6b",
                        kind="coverage_embedding",
                    ),
                    str(model_dir),
                )

    def test_missing_selected_model_is_rejected(self):
        with patch(
            "app.services.unsupervised_evaluation.model_options.resolve_model_reference",
            return_value="/tmp/qa-flow-model-does-not-exist",
        ):
            with self.assertRaises(ValueError):
                validate_evaluation_model_name(
                    "qwen3_embedding_0_6b",
                    kind="coverage_embedding",
                )


if __name__ == "__main__":
    unittest.main()
