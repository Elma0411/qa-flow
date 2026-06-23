"""Public facade for document extraction and OCR processing.

Heavy OCR dependencies are imported lazily so the main API can start in
environments where PaddleOCR is only available in the OCR runtime.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "normalize_output_format": ".compat_output",
    "DocumentPipeline": ".document_pipeline",
    "ImageInfo": ".ocr_processor.ocr_models",
    "OCRResult": ".ocr_processor.ocr_models",
    "PDFPipeline": ".pipeline",
    "resolve_result_output_file": ".compat_output",
    "SimpleOCRProcessor": ".ocr_processor.ocr_processor",
    "resolve_model_base_dir": ".ocr_processor.ocr_processor",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
