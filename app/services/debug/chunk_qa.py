# 文件作用：生成和保存 chunk 到 QA 的调试记录。
# 关联说明：记录 chunk 级 QA 调试信息，qa_store.py 负责通用调试存储。

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.core.config import CONFIG
from app.services.pipeline_state import get_pipeline_task_status


def _normalize_path(raw_path: Any) -> str:
    text = str(raw_path or "").strip()
    if not text:
        return ""
    candidates = [text]
    base_name = os.path.basename(text)
    outputs_dir = str(CONFIG["outputs_dir"])
    if base_name:
        candidates.append(os.path.join(outputs_dir, base_name))
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.exists(normalized):
            return normalized
    return ""


def _iter_output_paths(task_status: Dict[str, Any]) -> Iterable[str]:
    outputs = task_status.get("outputs") if isinstance(task_status.get("outputs"), list) else []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        for key in ("consolidated_json",):
            path = _normalize_path(output.get(key))
            if path:
                yield path


def _load_consolidated_items(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def load_chunk_qa_items_from_artifacts(
    *,
    task_id: str,
    chunk_id: str,
    only_filtered: bool = False,
    page: int = 1,
    page_size: int = 100,
) -> Dict[str, Any]:
    safe_task_id = str(task_id or "").strip()
    safe_chunk_id = str(chunk_id or "").strip()
    if not safe_task_id or not safe_chunk_id:
        return {
            "success": False,
            "source": "artifacts",
            "total": 0,
            "page": max(1, int(page)),
            "page_size": max(1, min(200, int(page_size))),
            "items": [],
        }

    task_status = get_pipeline_task_status(safe_task_id) or {}
    paths = list(_iter_output_paths(task_status))
    if not paths:
        return {
            "success": True,
            "source": "artifacts",
            "total": 0,
            "page": max(1, int(page)),
            "page_size": max(1, min(200, int(page_size))),
            "items": [],
        }

    matched: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for item in _load_consolidated_items(path):
            source_chunk_id = str(item.get("source_chunk_id") or item.get("source") or "").strip()
            if source_chunk_id != safe_chunk_id:
                continue
            if only_filtered and not bool(item.get("filtered")):
                continue
            qa_id = str(item.get("id") or "").strip()
            key = qa_id or f"{path}::{len(matched)}"
            matched[key] = dict(item)

    items = sorted(
        matched.values(),
        key=lambda row: (
            0 if row.get("is_primary") else 1,
            int(row.get("created_at") or 0),
            str(row.get("id") or ""),
        ),
    )

    safe_page = max(1, int(page))
    safe_size = max(1, min(200, int(page_size)))
    start = (safe_page - 1) * safe_size
    end = start + safe_size
    return {
        "success": True,
        "source": "artifacts",
        "task_id": safe_task_id,
        "chunk_id": safe_chunk_id,
        "total": len(items),
        "page": safe_page,
        "page_size": safe_size,
        "items": items[start:end],
    }


__all__ = ["load_chunk_qa_items_from_artifacts"]
