# 文件作用：处理 Easy Dataset 输入文件的 Markdown 化预处理。
# 关联说明：被 easy_dataset.py 的 split_file 调用，和 epub_preprocessing 共同完成文件读取阶段。

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .easy_dataset_common import _normalize_filename_to_markdown
from .easy_dataset_errors import EasyDatasetChunkingError
from .epub_preprocessing import EasyDatasetEpubError, process_epub

def _require_mammoth():
    try:
        import mammoth  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise EasyDatasetChunkingError(
            "missing_dependency",
            "Missing dependency `mammoth` required for .docx preprocessing",
        ) from exc
    return mammoth
def _require_markdownify():
    try:
        from markdownify import markdownify  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency error path
        raise EasyDatasetChunkingError(
            "missing_dependency",
            "Missing dependency `markdownify` required for HTML/EPUB/DOCX Markdown conversion",
        ) from exc
    return markdownify
def preprocess_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    ext = path.suffix.lower()
    buffer = path.read_bytes()
    base_name = path.name

    if ext == ".md":
        return {
            "sourcePath": str(path),
            "sourceExt": ext,
            "fileName": base_name,
            "normalizedFileName": base_name,
            "content": buffer.decode("utf-8"),
        }
    if ext == ".txt":
        return {
            "sourcePath": str(path),
            "sourceExt": ext,
            "fileName": base_name,
            "normalizedFileName": _normalize_filename_to_markdown(base_name, ext),
            "content": buffer.decode("utf-8"),
        }
    if ext == ".docx":
        mammoth = _require_mammoth()
        markdownify = _require_markdownify()
        with path.open("rb") as handle:
            html_result = mammoth.convert_to_html(handle)
        return {
            "sourcePath": str(path),
            "sourceExt": ext,
            "fileName": base_name,
            "normalizedFileName": _normalize_filename_to_markdown(base_name, ext),
            "content": markdownify(str(html_result.value or ""), heading_style="ATX"),
        }
    if ext == ".epub":
        try:
            epub_markdown = process_epub(buffer)
        except EasyDatasetEpubError as exc:
            raise EasyDatasetChunkingError("epub_preprocess_failed", str(exc)) from exc
        return {
            "sourcePath": str(path),
            "sourceExt": ext,
            "fileName": base_name,
            "normalizedFileName": _normalize_filename_to_markdown(base_name, ext),
            "content": epub_markdown,
        }
    if ext == ".pdf":
        raise EasyDatasetChunkingError(
            "unsupported_extension",
            "PDF preprocessing is not included in this standalone extraction",
        )
    raise EasyDatasetChunkingError(
        "unsupported_extension",
        f"Unsupported file extension: {ext}",
    )

__all__ = ["preprocess_file"]
