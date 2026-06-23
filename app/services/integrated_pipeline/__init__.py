"""Public facade for the OCR-image-QA integrated pipeline."""

__all__ = ["resolve_uploaded_files_with_integrated_processing"]


def __getattr__(name: str):
    if name == "resolve_uploaded_files_with_integrated_processing":
        from .service import resolve_uploaded_files_with_integrated_processing

        return resolve_uploaded_files_with_integrated_processing
    raise AttributeError(name)
