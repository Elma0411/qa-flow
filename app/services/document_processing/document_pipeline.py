"""
Document format router ahead of the PDF pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.services.document_processing.input_adapters.common import IMAGE_SUFFIXES, PDF_SUFFIXES
from app.services.document_processing.input_adapters.libreoffice_adapter import convert_to_pdf
from app.services.document_processing.input_adapters.ofd_adapter import ofd_to_pdf
from app.services.document_processing.pipeline import PDFPipeline


logger = logging.getLogger(__name__)

DOCX_STRATEGIES = {"pdf"}
SUPPORTED_DOCUMENT_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | {".ofd", ".docx", ".doc"}


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
        logger.debug("Document pipeline progress callback failed", exc_info=True)


class DocumentPipeline:
    """
    Route supported input formats to PDFPipeline-backed processing.
    """

    def __init__(self, pdf_pipeline: PDFPipeline):
        self.pdf_pipeline = pdf_pipeline

    def process_file(
        self,
        file_path: str,
        custom_output_dir: str,
        enable_image_analysis: bool,
        enable_classification: bool,
        remove_watermark: bool,
        watermark_dpi: int,
        docx_strategy: str = "auto",
        classification_confidence_threshold: float = 0.0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        suffix = input_path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}")
        _emit_progress(
            progress_callback,
            stage="format_routing",
            state="processing",
            message=f"识别输入格式：{suffix.lstrip('.') or 'unknown'}",
            extra={"suffix": suffix},
        )

        output_dir = Path(custom_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if suffix in PDF_SUFFIXES or suffix in IMAGE_SUFFIXES:
            result = self._process_pdf_pipeline(
                input_path=input_path,
                output_dir=output_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                classification_confidence_threshold=classification_confidence_threshold,
                progress_callback=progress_callback,
            )
            return self._with_metadata(
                result,
                input_type=suffix.lstrip("."),
                normalized_input_path=str(input_path),
                processing_path="direct->ocr_pipeline",
                docx_strategy=None,
            )

        normalized_dir = output_dir / "normalized_input"
        normalized_dir.mkdir(parents=True, exist_ok=True)

        if suffix == ".ofd":
            _emit_progress(progress_callback, stage="format_conversion", state="processing", message="OFD 转 PDF 中")
            normalized_pdf = Path(ofd_to_pdf(str(input_path), str(normalized_dir)))
            _emit_progress(
                progress_callback,
                stage="format_conversion",
                state="completed",
                message="OFD 转 PDF 完成",
                extra={"normalized_input_path": str(normalized_pdf)},
            )
            result = self._process_pdf_pipeline(
                input_path=normalized_pdf,
                output_dir=output_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                classification_confidence_threshold=classification_confidence_threshold,
                progress_callback=progress_callback,
            )
            return self._with_metadata(
                result,
                input_type="ofd",
                normalized_input_path=str(normalized_pdf),
                processing_path="ofd->pdf->ocr_pipeline",
                docx_strategy=None,
            )

        if suffix == ".doc":
            _emit_progress(progress_callback, stage="format_conversion", state="processing", message="DOC 转 PDF 中")
            normalized_pdf = Path(convert_to_pdf(str(input_path), str(normalized_dir)))
            _emit_progress(
                progress_callback,
                stage="format_conversion",
                state="completed",
                message="DOC 转 PDF 完成",
                extra={"normalized_input_path": str(normalized_pdf)},
            )
            result = self._process_pdf_pipeline(
                input_path=normalized_pdf,
                output_dir=output_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                classification_confidence_threshold=classification_confidence_threshold,
                progress_callback=progress_callback,
            )
            return self._with_metadata(
                result,
                input_type="doc",
                normalized_input_path=str(normalized_pdf),
                processing_path="doc->pdf->ocr_pipeline",
                docx_strategy=None,
            )

        return self._process_docx(
            input_path=input_path,
            output_dir=output_dir,
            normalized_dir=normalized_dir,
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            docx_strategy=docx_strategy,
            classification_confidence_threshold=classification_confidence_threshold,
            progress_callback=progress_callback,
        )

    def _process_docx(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        normalized_dir: Path,
        enable_image_analysis: bool,
        enable_classification: bool,
        remove_watermark: bool,
        watermark_dpi: int,
        docx_strategy: str,
        classification_confidence_threshold: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        strategy = self._normalize_docx_strategy(docx_strategy)
        return self._process_docx_as_pdf(
            input_path=input_path,
            output_dir=output_dir,
            normalized_dir=normalized_dir,
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            requested_strategy=strategy,
            classification_confidence_threshold=classification_confidence_threshold,
            progress_callback=progress_callback,
        )

    def _process_docx_as_pdf(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        normalized_dir: Path,
        enable_image_analysis: bool,
        enable_classification: bool,
        remove_watermark: bool,
        watermark_dpi: int,
        requested_strategy: str,
        classification_confidence_threshold: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        _emit_progress(progress_callback, stage="format_conversion", state="processing", message="DOCX 转 PDF 中")
        normalized_pdf = Path(convert_to_pdf(str(input_path), str(normalized_dir)))
        _emit_progress(
            progress_callback,
            stage="format_conversion",
            state="completed",
            message="DOCX 转 PDF 完成",
            extra={"normalized_input_path": str(normalized_pdf)},
        )
        result = self._process_pdf_pipeline(
            input_path=normalized_pdf,
            output_dir=output_dir,
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            classification_confidence_threshold=classification_confidence_threshold,
            progress_callback=progress_callback,
        )
        metadata = self._with_metadata(
            result,
            input_type="docx",
            normalized_input_path=str(normalized_pdf),
            processing_path="docx->pdf->ocr_pipeline",
            docx_strategy=requested_strategy,
        )
        return metadata

    def _process_pdf_pipeline(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        enable_image_analysis: bool,
        enable_classification: bool,
        remove_watermark: bool,
        watermark_dpi: int,
        classification_confidence_threshold: float,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        return self.pdf_pipeline.process_pdf(
            pdf_path=str(input_path),
            custom_output_dir=str(output_dir),
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            classification_confidence_threshold=classification_confidence_threshold,
            progress_callback=progress_callback,
        )

    def _normalize_docx_strategy(self, raw_strategy: str) -> str:
        return "pdf"

    def _with_metadata(
        self,
        result: Dict[str, Any],
        *,
        input_type: str,
        normalized_input_path: str,
        processing_path: str,
        docx_strategy: str | None,
    ) -> Dict[str, Any]:
        updated = dict(result)
        updated.update(
            {
                "input_type": input_type,
                "normalized_input_path": normalized_input_path,
                "processing_path": processing_path,
            }
        )
        if docx_strategy is not None:
            updated["docx_strategy"] = docx_strategy
        return updated
