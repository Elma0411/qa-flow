"""Output selection helpers for the dw-compatible OCR route."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple


def normalize_output_format(raw_value: Optional[str]) -> str:
    normalized = str(raw_value or "text").strip().lower()
    if normalized in {"markdown", "md"}:
        return "markdown"
    if normalized in {"ocr_markdown", "ocr-md", "ocr_md"}:
        return "ocr_markdown"
    return "text"


def resolve_result_output_file(result: Mapping[str, object], output_format: str) -> Tuple[Path, str]:
    if output_format == "markdown":
        return Path(str(result.get("output_markdown_file") or "")), "text/markdown"
    if output_format == "ocr_markdown":
        return Path(str(result.get("ocr_markdown_file") or "")), "text/markdown"
    return Path(str(result.get("output_text_file") or result.get("output_file") or "")), "text/plain"
