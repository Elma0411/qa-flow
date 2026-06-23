# 文件作用：提供 Easy Dataset 切块内部共用的小型路径/文件名工具。
# 关联说明：被 easy_dataset.py、manual 和 preprocessing 模块复用，避免工具函数散落。

from __future__ import annotations

from pathlib import Path

def _normalize_filename_to_markdown(filename: str, source_ext: str) -> str:
    source_path = Path(filename or "document")
    suffix = source_ext.lower()
    if suffix in {".txt", ".docx", ".epub"}:
        return source_path.with_suffix(".md").name
    return source_path.name
def _path_stem(file_name: str) -> str:
    stem = Path(str(file_name or "document.md")).stem.strip()
    return stem or "document"

__all__ = ["_normalize_filename_to_markdown", "_path_stem"]
