# 文件作用：构建带层级路径的文档树状 chunk。
# 关联说明：调用 easy_dataset 生成树状 chunk，是 app pipeline 使用的树结构切块入口。

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .easy_dataset import (
    ENGINE_VERSION,
    build_tree_chunks_easy_dataset,
)


def build_tree_chunks(
    text: str,
    *,
    chunk_size: int,
    original_filename: str,
    task_id: str,
    doc_id: str,
    prefix_max_depth: int = 4,
    title_sep: str = ">",
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    split_type: Optional[str] = None,
    text_split_min_length: Optional[int] = None,
    text_split_max_length: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separator: Optional[str] = None,
    separators: Optional[List[str]] = None,
    split_language: Optional[str] = None,
    custom_separator: Optional[str] = None,
    manual_split_points: Optional[List[Dict[str, Any]]] = None,
    force_heading_correction: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    return build_tree_chunks_easy_dataset(
        text,
        chunk_size=chunk_size,
        original_filename=original_filename,
        task_id=task_id,
        doc_id=doc_id,
        prefix_max_depth=prefix_max_depth,
        title_sep=title_sep,
        debug_writer=debug_writer,
        split_type=split_type,
        text_split_min_length=text_split_min_length,
        text_split_max_length=text_split_max_length,
        chunk_overlap=chunk_overlap,
        separator=separator,
        separators=separators,
        split_language=split_language,
        custom_separator=custom_separator,
        manual_split_points=manual_split_points,
        force_heading_correction=force_heading_correction,
    )


__all__ = ["ENGINE_VERSION", "build_tree_chunks"]
