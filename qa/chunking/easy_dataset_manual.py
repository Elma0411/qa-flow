# 文件作用：处理 Easy Dataset 手工切分点校验、预览和切分。
# 关联说明：被 easy_dataset.py 的 split_content 调用，和自动 split modes 并列。

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .easy_dataset_common import _path_stem
from .easy_dataset_errors import EasyDatasetChunkingError

def normalize_split_points(split_points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen_positions: set[int] = set()
    for index, point in enumerate(split_points):
        if not isinstance(point, dict):
            raise EasyDatasetChunkingError(
                "invalid_split_points",
                f"Split point at index {index} must be an object with a `position` field",
            )
        if "position" not in point:
            raise EasyDatasetChunkingError(
                "invalid_split_points",
                f"Split point at index {index} is missing `position`",
            )
        try:
            position = int(point.get("position"))
        except Exception as exc:
            raise EasyDatasetChunkingError(
                "invalid_split_points",
                f"Split point at index {index} has an invalid `position` value",
            ) from exc
        if position < 0:
            raise EasyDatasetChunkingError(
                "invalid_split_points",
                f"Split point at index {index} must be >= 0",
            )
        if position in seen_positions:
            raise EasyDatasetChunkingError(
                "invalid_split_points",
                f"Duplicate split point position detected: {position}",
            )
        seen_positions.add(position)
        normalized.append(
            {
                "id": point.get("id") or index + 1,
                "position": position,
                "preview": str(point.get("preview") or ""),
            }
        )
    normalized.sort(key=lambda item: int(item["position"]))
    return normalized
def preview_split_points(content: str, split_points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_points = normalize_split_points(split_points)
    chunks: List[Dict[str, Any]] = []
    start_position = 0
    for index, point in enumerate(sorted_points, start=1):
        end_position = int(point["position"])
        chunk_content = str(content or "")[start_position:end_position]
        if chunk_content.strip():
            preview = chunk_content[:20] + ("..." if len(chunk_content) > 20 else "")
            chunks.append(
                {
                    "index": index,
                    "length": len(chunk_content),
                    "preview": preview,
                }
            )
        start_position = end_position
    last_chunk = str(content or "")[start_position:]
    if last_chunk.strip():
        preview = last_chunk[:20] + ("..." if len(last_chunk) > 20 else "")
        chunks.append(
            {
                "index": len(chunks) + 1,
                "length": len(last_chunk),
                "preview": preview,
            }
        )
    return chunks
def manual_split(
    *,
    content: str,
    file_name: str = "document.md",
    split_points: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sorted_points = normalize_split_points(split_points)
    chunks: List[Dict[str, Any]] = []
    start_position = 0
    base_name = _path_stem(file_name)
    total_chunks = len(sorted_points) + 1

    for index, point in enumerate(sorted_points, start=1):
        end_position = int(point["position"])
        chunk_content = str(content or "")[start_position:end_position]
        if chunk_content.strip():
            chunks.append(
                {
                    "name": f"{base_name}-part-{index}",
                    "fileName": file_name,
                    "content": chunk_content,
                    "summary": f"{base_name} Custom Split {index}/{total_chunks}",
                    "size": len(chunk_content),
                    "titlePathParts": [base_name, f"Custom Split {index}"],
                    "splitType": "manual",
                }
            )
        start_position = end_position

    last_chunk = str(content or "")[start_position:]
    if last_chunk.strip():
        chunks.append(
            {
                "name": f"{base_name}-part-{total_chunks}",
                "fileName": file_name,
                "content": last_chunk,
                "summary": f"{base_name} Custom Split {total_chunks}/{total_chunks}",
                "size": len(last_chunk),
                "titlePathParts": [base_name, f"Custom Split {total_chunks}"],
                "splitType": "manual",
            }
        )
    return chunks

__all__ = ["manual_split", "normalize_split_points", "preview_split_points"]
