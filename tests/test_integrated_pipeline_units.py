import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.document_processing.ocr_processor.ocr_models import ImageInfo
from app.services.document_processing import normalize_output_format, resolve_result_output_file
from app.services.image_understanding.classifier_service.class_config import (
    BUILTIN_CLASS_CONFIGS,
    load_class_configs_from_file,
)
from app.services.integrated_pipeline.ocr_worker import resolve_ocr_replace_images
from app.services.integrated_pipeline.service import (
    IntegratedPipelineRunner,
    _format_image_before_context,
    _chunk_context_around_marker,
    _format_image_side_context,
    _map_image_analysis_progress_stage,
    _normalize_pdf_docx_strategy,
    _resolve_positive_int,
)
from app.services.integrated_pipeline.markers import (
    extract_marker_ids,
    locate_markers_in_chunks,
    replace_image_divs_with_markers,
    restore_markers_in_text,
)
from app.services.integrated_pipeline.placement import normalize_fit_score, parse_placement_response
from app.services.integrated_pipeline.summary import ChunkSummaryService, normalize_summary_mode
from app.services.integrated_pipeline.models import ChunkContext
from app.services.llm.vlm_client import VLMClientConfig


class IntegratedPipelineUnitTests(unittest.TestCase):
    def test_image_divs_are_replaced_with_markers(self):
        markdown = 'before <div><img src="imgs/img_1.jpg"/></div> after'
        info = ImageInfo(
            image_id="img_1",
            file_path=Path("imgs/img_1.jpg"),
            page_number=1,
            div_tag='<div><img src="imgs/img_1.jpg"/></div>',
        )

        marked, markers = replace_image_divs_with_markers(markdown, [info])

        self.assertIn("[[IMAGE_REF:img_1]]", marked)
        self.assertEqual(["img_1"], [marker.image_id for marker in markers])
        self.assertEqual(["img_1"], extract_marker_ids(marked))

    def test_restore_markers_in_text(self):
        text = "alpha [[IMAGE_REF:img_1]] beta [[IMAGE_REF:img_2]]"
        restored = restore_markers_in_text(text, {"img_1": "图片事实"})

        self.assertIn("【图片描述：图片事实】", restored)
        self.assertNotIn("[[IMAGE_REF:img_2]]", restored)

    def test_marker_locations_are_indexed_by_chunk(self):
        locations = locate_markers_in_chunks(
            [
                {"chunk_index": 1, "text": "alpha [[IMAGE_REF:img_1]]"},
                {"chunk_index": 2, "text": "beta [[IMAGE_REF:img_2]]"},
            ]
        )

        self.assertEqual({"img_1": 1, "img_2": 2}, locations)

    def test_ocr_markdown_chunking_normalizes_original_pdf_name(self):
        from qa.chunking.easy_dataset import build_tree_chunks_easy_dataset

        chunks, meta, report = build_tree_chunks_easy_dataset(
            "# 标题\n\nOCR 正文内容。",
            chunk_size=80,
            original_filename="demo2.pdf",
            task_id="unit-task",
            doc_id="unit-doc",
            split_type="text",
        )

        self.assertEqual(1, len(chunks))
        self.assertEqual("demo2.pdf", meta[0]["original_filename"])
        self.assertEqual("text", report["effective_split_type"])

    def test_chunk_context_uses_full_marker_split_without_other_image_refs(self):
        text = (
            "标题前缀\n"
            "图片前第一句。图片前第二句。"
            "[[IMAGE_REF:img_1]]"
            "图片后第一句。[[IMAGE_REF:img_2]]图片后第二句。"
        )

        before, after = _chunk_context_around_marker(text, "[[IMAGE_REF:img_1]]")

        self.assertEqual("标题前缀\n图片前第一句。图片前第二句。", before)
        self.assertEqual("图片后第一句。图片后第二句。", after)

    def test_format_image_side_context_labels_empty_text(self):
        self.assertEqual("图片之前内容：\n（无）", _format_image_side_context("图片之前内容", ""))

    def test_format_image_before_context_keeps_chunk_summary(self):
        formatted = _format_image_before_context("chunk 摘要", "图片前正文")

        self.assertIn("图片所在 chunk 摘要：\nchunk 摘要", formatted)
        self.assertIn("图片之前内容：\n图片前正文", formatted)

    def test_integrated_image_classification_progress_is_separate_stage(self):
        self.assertEqual(
            "image_classification",
            _map_image_analysis_progress_stage("image_classification"),
        )
        self.assertEqual("doc_image_analysis", _map_image_analysis_progress_stage("vlm_call"))

    def test_parse_placement_response_applies_threshold(self):
        decision = parse_placement_response(
            "img_1",
            '{"accepted": true, "score": 0.7, "reason": "fits"}',
            0.65,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(0.7, decision.score)

        rejected = parse_placement_response(
            "img_1",
            '{"accepted": true, "score": 0.5, "reason": "weak"}',
            0.65,
        )
        self.assertFalse(rejected.accepted)

        invalid = parse_placement_response("img_1", "not json", 0.65)
        self.assertFalse(invalid.accepted)
        self.assertEqual("invalid_json", invalid.error)

    def test_summary_mode_normalization(self):
        self.assertEqual("lightweight", normalize_summary_mode(""))
        self.assertEqual("lightweight", normalize_summary_mode("bad"))
        self.assertEqual("llm", normalize_summary_mode("llm"))

    def test_lightweight_summary_uses_title_path_and_excerpt(self):
        summaries = ChunkSummaryService(mode="lightweight").summarize(
            [
                ChunkContext(
                    chunk_index=1,
                    chunk_id="c1",
                    text="正文内容",
                    title_path="A > B",
                    path_summary="路径摘要",
                )
            ]
        )

        self.assertIn("标题路径：A > B", summaries[0].summary)
        self.assertIn("路径摘要：路径摘要", summaries[0].summary)
        self.assertIn("正文内容", summaries[0].summary)

    def test_fit_score_normalization(self):
        self.assertEqual(1.0, normalize_fit_score(2.0))
        self.assertEqual(0.0, normalize_fit_score(-1.0))

    def test_ocr_process_output_selection(self):
        self.assertEqual("text", normalize_output_format("bad"))
        self.assertEqual("markdown", normalize_output_format("md"))
        self.assertEqual("ocr_markdown", normalize_output_format("ocr-md"))

        path, media_type = resolve_result_output_file(
            {
                "output_text_file": "/tmp/out.txt",
                "output_markdown_file": "/tmp/out.md",
                "ocr_markdown_file": "/tmp/ocr.md",
            },
            "ocr_markdown",
        )

        self.assertEqual(Path("/tmp/ocr.md"), path)
        self.assertEqual("text/markdown", media_type)

    def test_vlm_config_uses_env_and_requires_missing_values(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "VLM_API_BASE"):
                VLMClientConfig.from_values(
                    api_base=None,
                    model_name=None,
                    api_key=None,
                    timeout_seconds=5,
                )

        env = {
            "VLM_API_BASE": "http://env.example/v1",
            "VLM_MODEL_NAME": "env-model",
            "VLM_API_KEY": "env-key",
            "VLM_MODEL_VERSION": "env-version",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VLMClientConfig.from_values(
                api_base=None,
                model_name=None,
                api_key=None,
                timeout_seconds=5,
            )

        self.assertEqual("http://env.example/v1", config.base_url)
        self.assertEqual("env-model", config.model_name)
        self.assertEqual("env-key", config.api_key)
        self.assertEqual("env-version", config.model_version)

    def test_vlm_config_request_values_override_env(self):
        env = {
            "VLM_API_BASE": "http://env.example/v1",
            "VLM_MODEL_NAME": "env-model",
            "VLM_API_KEY": "env-key",
        }
        with patch.dict(os.environ, env, clear=True):
            config = VLMClientConfig.from_values(
                api_base="http://request.example/v1/chat/completions",
                model_name="request-model",
                api_key="request-key",
                timeout_seconds=5,
            )

        self.assertEqual("http://request.example/v1", config.base_url)
        self.assertEqual("request-model", config.model_name)
        self.assertEqual("request-key", config.api_key)

    def test_ocr_replace_images_env_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(resolve_ocr_replace_images(default=True))

        with patch.dict(os.environ, {"OCR_REPLACE_IMAGES": "false"}, clear=True):
            self.assertFalse(resolve_ocr_replace_images(default=True))
            self.assertTrue(resolve_ocr_replace_images(True, default=True))

        runner = IntegratedPipelineRunner(
            task_id="unit-test",
            chunk_size=10,
            replace_images=False,
        )
        self.assertFalse(runner.replace_images)

    def test_integrated_concurrency_uses_request_then_env(self):
        with patch.dict(os.environ, {"DOC_MAX_CONCURRENCY": "3"}, clear=True):
            self.assertEqual(
                3,
                _resolve_positive_int(
                    None,
                    env_name="DOC_MAX_CONCURRENCY",
                    default=1,
                    maximum=8,
                ),
            )
            self.assertEqual(
                5,
                _resolve_positive_int(
                    5,
                    env_name="DOC_MAX_CONCURRENCY",
                    default=1,
                    maximum=8,
                ),
            )
            self.assertEqual(
                8,
                _resolve_positive_int(
                    99,
                    env_name="DOC_MAX_CONCURRENCY",
                    default=1,
                    maximum=8,
                ),
            )

    def test_docx_strategy_is_forced_to_pdf(self):
        self.assertEqual("pdf", _normalize_pdf_docx_strategy("native"))

        runner = IntegratedPipelineRunner(
            task_id="unit-test",
            chunk_size=10,
            docx_strategy="native",
        )
        self.assertEqual("pdf", runner.docx_strategy)

    def test_classifier_classes_use_builtin_fallback_shape(self):
        self.assertEqual(10, len(BUILTIN_CLASS_CONFIGS))
        self.assertEqual("其他", BUILTIN_CLASS_CONFIGS[0].model_label)

    def test_classifier_classes_load_from_json_file(self):
        payload = [
            {
                "class_id": 0,
                "model_label": "alpha",
                "category_key": "alpha_key",
                "display_name": "Alpha",
            },
            {
                "class_id": 1,
                "model_label": "beta",
                "category_key": "beta_key",
                "display_name": "Beta",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "classes.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            configs = load_class_configs_from_file(path)

        self.assertEqual(["alpha", "beta"], [config.model_label for config in configs])
        self.assertEqual(["alpha_key", "beta_key"], [config.category_key for config in configs])

    def test_classifier_classes_reject_invalid_json_shape(self):
        payload = [
            {
                "class_id": 1,
                "model_label": "beta",
                "category_key": "beta_key",
                "display_name": "Beta",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "classes.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "sequential"):
                load_class_configs_from_file(path)


if __name__ == "__main__":
    unittest.main()
