"""
Document format router ahead of the PDF pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from app.services.document_processing.input_adapters.common import IMAGE_SUFFIXES, PDF_SUFFIXES
from app.services.document_processing.input_adapters.libreoffice_adapter import convert_to_pdf
from app.services.document_processing.input_adapters.ofd_adapter import ofd_to_pdf
from app.services.document_processing.pipeline import PDFPipeline


logger = logging.getLogger(__name__)

DOCX_STRATEGIES = {"auto", "native", "pdf"}
SUPPORTED_DOCUMENT_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | {".ofd", ".docx", ".doc"}


class DocumentPipeline:
    """
    Route supported input formats to either PDFPipeline or a native adapter.
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
    ) -> Dict[str, Any]:
        input_path = Path(file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        suffix = input_path.suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}")

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
            normalized_pdf = Path(ofd_to_pdf(str(input_path), str(normalized_dir)))
            result = self._process_pdf_pipeline(
                input_path=normalized_pdf,
                output_dir=output_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                classification_confidence_threshold=classification_confidence_threshold,
            )
            return self._with_metadata(
                result,
                input_type="ofd",
                normalized_input_path=str(normalized_pdf),
                processing_path="ofd->pdf->ocr_pipeline",
                docx_strategy=None,
            )

        if suffix == ".doc":
            normalized_pdf = Path(convert_to_pdf(str(input_path), str(normalized_dir)))
            result = self._process_pdf_pipeline(
                input_path=normalized_pdf,
                output_dir=output_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                classification_confidence_threshold=classification_confidence_threshold,
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
    ) -> Dict[str, Any]:
        strategy = self._normalize_docx_strategy(docx_strategy)

        if strategy == "pdf":
            return self._process_docx_as_pdf(
                input_path=input_path,
                output_dir=output_dir,
                normalized_dir=normalized_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                fallback_reason=None,
                requested_strategy=strategy,
                classification_confidence_threshold=classification_confidence_threshold,
            )

        try:
            result = self._process_docx_native(
                input_path=input_path,
                output_dir=output_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                classification_confidence_threshold=classification_confidence_threshold,
            )
            return self._with_metadata(
                result,
                input_type="docx",
                normalized_input_path=str(input_path),
                processing_path="docx->native->image_analysis->text_integration",
                docx_strategy=strategy,
            )
        except Exception as exc:
            if strategy == "native":
                raise

            logger.warning(
                "DOCX native processing failed, falling back to PDF conversion: %s",
                exc,
                exc_info=True,
            )
            return self._process_docx_as_pdf(
                input_path=input_path,
                output_dir=output_dir,
                normalized_dir=normalized_dir,
                enable_image_analysis=enable_image_analysis,
                enable_classification=enable_classification,
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                fallback_reason=str(exc),
                requested_strategy=strategy,
                classification_confidence_threshold=classification_confidence_threshold,
            )

    def _process_docx_native(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        enable_image_analysis: bool,
        enable_classification: bool,
        classification_confidence_threshold: float,
    ) -> Dict[str, Any]:
        from app.services.document_processing.input_adapters.docx_native_adapter import parse_docx_to_ocr_result

        ocr_result = parse_docx_to_ocr_result(
            docx_path=str(input_path),
            output_dir=str(output_dir),
        )
        return self.pdf_pipeline.process_from_ocr_result(
            ocr_result=ocr_result,
            custom_output_dir=str(output_dir),
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            classification_confidence_threshold=classification_confidence_threshold,
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
        fallback_reason: str | None,
        requested_strategy: str,
        classification_confidence_threshold: float,
    ) -> Dict[str, Any]:
        normalized_pdf = Path(convert_to_pdf(str(input_path), str(normalized_dir)))
        result = self._process_pdf_pipeline(
            input_path=normalized_pdf,
            output_dir=output_dir,
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            classification_confidence_threshold=classification_confidence_threshold,
        )
        metadata = self._with_metadata(
            result,
            input_type="docx",
            normalized_input_path=str(normalized_pdf),
            processing_path="docx->pdf->ocr_pipeline",
            docx_strategy=requested_strategy,
        )
        if fallback_reason:
            metadata["docx_native_fallback_reason"] = fallback_reason
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
    ) -> Dict[str, Any]:
        return self.pdf_pipeline.process_pdf(
            pdf_path=str(input_path),
            custom_output_dir=str(output_dir),
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            classification_confidence_threshold=classification_confidence_threshold,
        )

    def _normalize_docx_strategy(self, raw_strategy: str) -> str:
        strategy = str(raw_strategy or "auto").strip().lower()
        if strategy not in DOCX_STRATEGIES:
            raise ValueError(
                f"Unsupported docx_strategy={raw_strategy!r}; expected one of "
                f"{', '.join(sorted(DOCX_STRATEGIES))}"
            )
        return strategy

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
