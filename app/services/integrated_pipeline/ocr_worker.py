"""OCR worker manager for the integrated pipeline."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.core.config import CONFIG
from app.services.document_processing.input_adapters.common import IMAGE_SUFFIXES, PDF_SUFFIXES
from app.services.document_processing.ocr_processor.ocr_models import OCRResult


def _resolve_env_bool(name: str, *, override: Optional[bool] = None, default: bool = True) -> bool:
    if override is not None:
        return bool(override)
    raw = str(os.getenv(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def resolve_ocr_use_gpu(default: bool = True) -> bool:
    return _resolve_env_bool("OCR_USE_GPU", default=default)


def resolve_ocr_replace_images(override: Optional[bool] = None, default: bool = True) -> bool:
    return _resolve_env_bool("OCR_REPLACE_IMAGES", override=override, default=default)


class OCRWorkerManager:
    """Owns process-local OCR processor instances keyed by model/device config."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._processors: Dict[tuple, object] = {}

    def extract(
        self,
        *,
        file_path: str,
        output_dir: str,
        docx_strategy: str = "pdf",
        remove_watermark: bool = False,
        watermark_dpi: int = 200,
        replace_images: bool = True,
        use_gpu: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> OCRResult:
        input_path = Path(file_path)
        suffix = input_path.suffix.lower()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if suffix in PDF_SUFFIXES or suffix in IMAGE_SUFFIXES:
            return self._process_pdf_or_image(
                file_path=str(input_path),
                output_dir=str(output_path / "ocr_output"),
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                replace_images=replace_images,
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )

        normalized_dir = output_path / "normalized_input"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        if suffix == ".ofd":
            from app.services.document_processing.input_adapters.ofd_adapter import ofd_to_pdf

            normalized_pdf = ofd_to_pdf(str(input_path), str(normalized_dir))
            return self._process_pdf_or_image(
                file_path=normalized_pdf,
                output_dir=str(output_path / "ocr_output"),
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                replace_images=replace_images,
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )
        if suffix == ".doc":
            from app.services.document_processing.input_adapters.libreoffice_adapter import convert_to_pdf

            normalized_pdf = convert_to_pdf(str(input_path), str(normalized_dir))
            return self._process_pdf_or_image(
                file_path=normalized_pdf,
                output_dir=str(output_path / "ocr_output"),
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                replace_images=replace_images,
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )
        if suffix == ".docx":
            from app.services.document_processing.input_adapters.libreoffice_adapter import convert_to_pdf

            normalized_pdf = convert_to_pdf(str(input_path), str(normalized_dir))
            return self._process_pdf_or_image(
                file_path=normalized_pdf,
                output_dir=str(output_path / "ocr_output"),
                remove_watermark=remove_watermark,
                watermark_dpi=watermark_dpi,
                replace_images=replace_images,
                use_gpu=use_gpu,
                progress_callback=progress_callback,
            )
        raise ValueError(f"Unsupported integrated OCR input type: {suffix}")

    def _process_pdf_or_image(
        self,
        *,
        file_path: str,
        output_dir: str,
        remove_watermark: bool,
        watermark_dpi: int,
        replace_images: bool,
        use_gpu: bool,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> OCRResult:
        processor = self._get_processor(replace_images=replace_images, use_gpu=use_gpu)
        return processor.process_pdf(
            pdf_path=file_path,
            output_dir=output_dir,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            progress_callback=progress_callback,
        )

    def _get_processor(self, *, replace_images: bool, use_gpu: bool) -> object:
        from app.services.document_processing.ocr_processor.ocr_processor import (
            SimpleOCRProcessor,
            resolve_model_base_dir,
        )

        model_base_dir = resolve_model_base_dir(
            os.getenv("MODEL_BASE_DIR", os.path.join(CONFIG["models_dir"], "ocr"))
        )
        cache_key = (str(model_base_dir), bool(use_gpu), bool(replace_images))
        with self._lock:
            processor = self._processors.get(cache_key)
            if processor is None:
                processor = SimpleOCRProcessor(
                    model_base_dir=str(model_base_dir),
                    use_gpu=use_gpu,
                    replace_images=replace_images,
                )
                self._processors[cache_key] = processor
            return processor


_DEFAULT_MANAGER = OCRWorkerManager()


def get_ocr_worker_manager() -> OCRWorkerManager:
    return _DEFAULT_MANAGER
