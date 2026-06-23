"""
Shared helpers for document input adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class DocumentConversionError(RuntimeError):
    """Raised when an input adapter cannot normalize a document to PDF."""


PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
OFFICE_SUFFIXES = {".doc", ".docx"}


def ensure_output_dir(output_dir: str | Path) -> Path:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    return output_dir_path


def find_pdf_for_stem(output_dir: Path, stem: str) -> Path | None:
    expected = output_dir / f"{stem}.pdf"
    if expected.exists():
        return expected

    stem_lower = stem.lower()
    candidates = [
        path
        for path in output_dir.glob("*.pdf")
        if path.stem.lower() == stem_lower
    ]
    if candidates:
        return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]

    return None


def supported_suffixes(extra_suffixes: Iterable[str] = ()) -> set[str]:
    return PDF_SUFFIXES | IMAGE_SUFFIXES | OFFICE_SUFFIXES | {suffix.lower() for suffix in extra_suffixes}
