"""
Pipeline orchestration for PDF OCR, image analysis, and text integration.
"""

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.services.document_processing.ocr_processor.ocr_models import OCRResult
from app.services.document_processing.ocr_processor.ocr_processor import SimpleOCRProcessor
from app.services.document_processing.text_integrator.text_processor import SimpleTextIntegrator
from app.services.image_understanding.analyzer import analyze_images_simple
from app.services.image_understanding.image_models import AnalysisResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    *,
    stage: str,
    state: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(
            {
                "stage": stage,
                "state": state,
                "message": message,
                "extra": extra or {},
            }
        )
    except Exception:
        logger.debug("PDF pipeline progress callback failed", exc_info=True)


def _normalize_classification_confidence_threshold(value: Optional[float]) -> float:
    if value is None:
        return 0.0

    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("classification_confidence_threshold must be between 0.0 and 1.0")
    return threshold


class PDFPipeline:
    """
    End-to-end PDF processing pipeline.
    """

    _ocr_processor_cache: Dict[tuple, SimpleOCRProcessor] = {}

    def __init__(
        self,
        model_base_dir: str,
        use_gpu: bool = True,
        output_base_dir: str = None,
        ocr_processor: Optional[SimpleOCRProcessor] = None,
        use_api: bool = True,
        replace_images: bool = True,
        remove_watermark: bool = True,
        watermark_dpi: int = 200,
        vlm_api_base: str = None,
        vlm_model_name: str = None,
        vlm_api_key: str = None,
        vlm_api_type: str = "openai",
        vlm_model_version: str = None,
        vlm_client: Optional[Any] = None,
        classifier_api_base: str = None,
        classifier_timeout: int = 30,
        classification_confidence_threshold: float = 0.0,
    ):
        self.use_gpu = use_gpu
        self.model_base_dir = model_base_dir
        self.replace_images = replace_images
        self.remove_watermark = remove_watermark
        self.watermark_dpi = watermark_dpi
        self.use_api = use_api
        self.vlm_api_base = vlm_api_base
        self.vlm_model_name = vlm_model_name
        self.vlm_api_key = vlm_api_key
        self.vlm_api_type = vlm_api_type
        self.vlm_model_version = vlm_model_version
        self.vlm_client = vlm_client
        self.classifier_api_base = classifier_api_base
        self.classifier_timeout = classifier_timeout
        self.classification_confidence_threshold = _normalize_classification_confidence_threshold(
            classification_confidence_threshold
        )

        if output_base_dir:
            self.output_base_dir = Path(output_base_dir)
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_base_dir = None

        if ocr_processor is not None:
            self.ocr_processor = ocr_processor
            logger.info("Using injected shared OCR processor instance")
        else:
            ocr_cache_key = (
                str(Path(model_base_dir).resolve()),
                bool(use_gpu),
                bool(replace_images),
            )
            if ocr_cache_key not in self._ocr_processor_cache:
                logger.info("Creating shared OCR processor instance")
                self._ocr_processor_cache[ocr_cache_key] = SimpleOCRProcessor(
                    model_base_dir=model_base_dir,
                    use_gpu=use_gpu,
                    replace_images=replace_images,
                )
            else:
                logger.info("Reusing shared OCR processor instance")

            self.ocr_processor = self._ocr_processor_cache[ocr_cache_key]
        self.text_integrator = SimpleTextIntegrator()

        logger.info("PDF pipeline initialized")
        logger.info(f"GPU mode: {'enabled' if use_gpu else 'disabled'}")
        logger.info(
            f"Watermark removal: {'enabled' if remove_watermark else 'disabled'} "
            f"(DPI: {watermark_dpi})"
        )
        logger.info(f"Classifier API base: {self.classifier_api_base or 'not configured'}")
        logger.info(f"Classification confidence threshold: {self.classification_confidence_threshold:.3f}")
        logger.info(f"VLM API type: {self.vlm_api_type}")
        if self.output_base_dir:
            logger.info(f"Output base dir: {self.output_base_dir}")
        else:
            logger.info("Output base dir: using request-specific directories")

    def _resolve_output_dir(self, input_stem: str, custom_output_dir: Optional[str]) -> Path:
        if custom_output_dir:
            output_dir = Path(custom_output_dir)
        elif self.output_base_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.output_base_dir / f"{input_stem}_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"pdf_output_{input_stem}_{timestamp}")

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _complete_from_ocr_result(
        self,
        *,
        ocr_result: OCRResult,
        output_dir: Path,
        total_start_time: float,
        ocr_time: float,
        enable_image_analysis: bool,
        enable_classification: bool,
        classification_confidence_threshold: Optional[float],
        completion_label: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        pdf_name = ocr_result.pdf_name
        effective_classification_confidence_threshold = (
            self.classification_confidence_threshold
            if classification_confidence_threshold is None
            else _normalize_classification_confidence_threshold(classification_confidence_threshold)
        )

        logger.info("Step 2: image analysis")
        image_analysis_dir = output_dir / "image_analysis"
        image_analysis_time = 0.0

        if enable_image_analysis and ocr_result.images_info:
            _emit_progress(
                progress_callback,
                stage="image_analysis",
                state="processing",
                message=f"图片理解中：共 {len(ocr_result.images_info)} 张图片",
                extra={"total_images": len(ocr_result.images_info)},
            )
            analysis_result = analyze_images_simple(
                ocr_result=ocr_result,
                output_dir=str(image_analysis_dir),
                use_api=self.use_api,
                api_base=self.vlm_api_base,
                model_name=self.vlm_model_name,
                api_key=self.vlm_api_key,
                vlm_api_type=self.vlm_api_type,
                model_version=self.vlm_model_version,
                vlm_client=self.vlm_client,
                enable_classification=enable_classification,
                classifier_api_base=self.classifier_api_base,
                classifier_timeout=self.classifier_timeout,
                classification_confidence_threshold=effective_classification_confidence_threshold,
                progress_callback=progress_callback,
            )
            image_analysis_time = float(analysis_result.processing_time)
            logger.info(
                f"Image analysis completed - analyzed: "
                f"{analysis_result.analyzed_images}/{analysis_result.total_images}, "
                f"time: {image_analysis_time:.2f}s"
            )
            _emit_progress(
                progress_callback,
                stage="image_analysis",
                state="completed",
                message=f"图片理解完成：{analysis_result.analyzed_images}/{analysis_result.total_images}",
                extra={
                    "total_images": analysis_result.total_images,
                    "analyzed_images": analysis_result.analyzed_images,
                    "failed_images": analysis_result.total_images - analysis_result.analyzed_images,
                    "image_analysis_seconds": image_analysis_time,
                },
            )
        else:
            image_analysis_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Skipping image analysis stage")
            analysis_result = AnalysisResult(
                pdf_name=pdf_name,
                total_images=len(ocr_result.images_info),
                analyzed_images=0,
                descriptions=[],
                processing_time=0.0,
                output_dir=image_analysis_dir,
            )
            _emit_progress(
                progress_callback,
                stage="image_analysis",
                state="completed",
                message="图片理解跳过",
                extra={"total_images": len(ocr_result.images_info), "enabled": enable_image_analysis},
            )

        logger.info("Step 3: text integration")
        text_integration_dir = output_dir / "text_integration"
        _emit_progress(progress_callback, stage="text_integration", state="processing", message="文本整合中")

        integration_result = self.text_integrator.integrate(
            ocr_result=ocr_result,
            analysis_result=analysis_result,
            output_dir=str(text_integration_dir),
            insert_image_descriptions=enable_image_analysis and analysis_result.analyzed_images > 0,
        )

        text_integration_time = float(integration_result.processing_time)
        logger.info(
            f"Text integration completed - text length: "
            f"{integration_result.integrated_text_length}, "
            f"time: {text_integration_time:.2f}s"
        )
        _emit_progress(
            progress_callback,
            stage="text_integration",
            state="completed",
            message="文本整合完成",
            extra={
                "integrated_markdown_length": integration_result.integrated_markdown_length,
                "integrated_text_length": integration_result.integrated_text_length,
                "replaced_images": integration_result.replaced_images,
                "text_integration_seconds": text_integration_time,
            },
        )

        total_time = time.perf_counter() - total_start_time
        raw_stage_processing_times = {
            "ocr": ocr_time,
            "image_analysis": image_analysis_time,
            "text_integration": text_integration_time,
        }
        stage_processing_times = {
            stage_name: round(stage_time, 2)
            for stage_name, stage_time in raw_stage_processing_times.items()
        }
        accounted_stage_time = sum(raw_stage_processing_times.values())
        orchestration_overhead_time = max(0.0, round(total_time - accounted_stage_time, 2))
        ocr_output_dir = Path(ocr_result.output_dir)

        result = {
            "pdf_name": pdf_name,
            "total_pages": ocr_result.total_pages,
            "total_images": analysis_result.total_images,
            "analyzed_images": analysis_result.analyzed_images,
            "replaced_images": integration_result.replaced_images,
            "classification_enabled": enable_classification,
            "classification_confidence_threshold": effective_classification_confidence_threshold,
            "final_markdown_length": integration_result.integrated_markdown_length,
            "final_text_length": integration_result.integrated_text_length,
            "total_processing_time": round(total_time, 2),
            "stage_processing_times": stage_processing_times,
            "orchestration_overhead_time": orchestration_overhead_time,
            "output_file": str(integration_result.output_file),
            "output_text_file": str(integration_result.text_output_file),
            "output_markdown_file": str(integration_result.markdown_output_file),
            "ocr_markdown_file": str(ocr_output_dir / f"{pdf_name}.md"),
            "output_dir": str(output_dir),
        }

        logger.info("=" * 60)
        logger.info(f"{completion_label} completed")
        logger.info("=" * 60)
        logger.info(f"PDF name: {pdf_name}")
        logger.info(f"Total pages: {ocr_result.total_pages}")
        logger.info(f"Total images: {analysis_result.total_images}")
        logger.info(f"Analyzed images: {analysis_result.analyzed_images}")
        logger.info(f"Replaced images: {integration_result.replaced_images}")
        logger.info(f"Classification enabled: {enable_classification}")
        logger.info(
            f"Classification confidence threshold: {effective_classification_confidence_threshold:.3f}"
        )
        logger.info(f"Final markdown length: {integration_result.integrated_markdown_length}")
        logger.info(f"Final text length: {integration_result.integrated_text_length}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Stage processing times: {stage_processing_times}")
        logger.info(f"Orchestration overhead time: {orchestration_overhead_time:.2f}s")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Markdown file: {integration_result.markdown_output_file}")
        logger.info(f"Text file: {integration_result.text_output_file}")
        logger.info("=" * 60)
        _emit_progress(
            progress_callback,
            stage="document_output",
            state="completed",
            message=f"{completion_label} 完成",
            extra={
                "total_processing_time": round(total_time, 2),
                "stage_processing_times": stage_processing_times,
            },
        )

        return result

    def process_from_ocr_result(
        self,
        ocr_result: OCRResult,
        custom_output_dir: Optional[str] = None,
        enable_image_analysis: bool = True,
        enable_classification: bool = False,
        classification_confidence_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        ocr_time = float(ocr_result.processing_time)
        total_start_time = time.perf_counter() - ocr_time
        output_dir = self._resolve_output_dir(ocr_result.pdf_name, custom_output_dir)

        logger.info("=" * 60)
        logger.info(f"Start processing OCRResult: {ocr_result.pdf_name}")
        logger.info(f"OCRResult output dir: {ocr_result.output_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("=" * 60)

        try:
            return self._complete_from_ocr_result(
                ocr_result=ocr_result,
                output_dir=output_dir,
                total_start_time=total_start_time,
                ocr_time=ocr_time,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                classification_confidence_threshold=classification_confidence_threshold,
                completion_label="Document tail processing",
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Processing from OCRResult failed: {e}")
            raise

    def process_pdf(
        self,
        pdf_path: str,
        custom_output_dir: Optional[str] = None,
        enable_image_analysis: bool = True,
        enable_classification: bool = False,
        remove_watermark: Optional[bool] = None,
        watermark_dpi: Optional[int] = None,
        classification_confidence_threshold: Optional[float] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        total_start_time = time.perf_counter()
        effective_remove_watermark = (
            self.remove_watermark
            if remove_watermark is None
            else bool(remove_watermark)
        )
        effective_watermark_dpi = (
            self.watermark_dpi
            if watermark_dpi is None
            else int(watermark_dpi)
        )

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"Input file not found: {pdf_path}")

        output_dir = self._resolve_output_dir(pdf_path_obj.stem, custom_output_dir)
        pdf_name = pdf_path_obj.stem

        logger.info("=" * 60)
        logger.info(f"Start processing input: {pdf_name}")
        logger.info(f"Input path: {pdf_path}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("=" * 60)

        try:
            logger.info("Step 1: OCR")
            ocr_output_dir = output_dir / "ocr_output"

            ocr_result = self.ocr_processor.process_pdf(
                pdf_path=str(pdf_path),
                output_dir=str(ocr_output_dir),
                remove_watermark=effective_remove_watermark,
                watermark_dpi=effective_watermark_dpi,
                progress_callback=progress_callback,
            )

            ocr_time = float(ocr_result.processing_time)
            logger.info(
                f"OCR completed - pages: {ocr_result.total_pages}, "
                f"images: {len(ocr_result.images_info)}, time: {ocr_time:.2f}s"
            )

            return self._complete_from_ocr_result(
                ocr_result=ocr_result,
                output_dir=output_dir,
                total_start_time=total_start_time,
                ocr_time=ocr_time,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                classification_confidence_threshold=classification_confidence_threshold,
                completion_label="PDF processing",
                progress_callback=progress_callback,
            )

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
