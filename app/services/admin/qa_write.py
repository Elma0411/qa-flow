# 文件作用：封装管理端问答条目的写入、更新、删除和导出操作。
# 关联说明：依赖 qa_query 做校验，依赖 qa_common/meta 完成问答写回和管理字段更新。

import json
import os
import time
from typing import Any, Dict, List, Optional

from app.core.config import CONFIG
from app.services import milvus as milvus_service
from .qa_common import (
    AdminMilvusError,
    _ensure_milvus_ready,
    _expr_in,
    _get_allowed_fields,
)
from .qa_query import get_qa_item
from app.services.debug import delete_debug_entries


def fetch_records_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    _ensure_milvus_ready()
    allowed_fields = _get_allowed_fields()
    allowed_set = set(allowed_fields)
    output_fields = list(allowed_fields)

    records: Dict[str, Dict[str, Any]] = {}
    chunk_size = 200
    for index in range(0, len(ids), chunk_size):
        chunk = [str(item) for item in ids[index : index + chunk_size] if item]
        if not chunk:
            continue
        expr = _expr_in("id", chunk)
        rows = milvus_service.milvus_client.query(  # type: ignore[union-attr]
            expr=expr,
            output_fields=output_fields,
            limit=len(chunk),
        )
        for row in rows:
            qa_id = str(row.get("id") or "")
            if not qa_id:
                continue
            records[qa_id] = {key: value for key, value in dict(row).items() if key in allowed_set}
    return records


def replace_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    _ensure_milvus_ready()
    allowed_fields = set(_get_allowed_fields())
    ids = [str(record.get("id")) for record in records if record.get("id")]
    chunk_size = 200
    for index in range(0, len(ids), chunk_size):
        chunk_ids = [item for item in ids[index : index + chunk_size] if item]
        if not chunk_ids:
            continue
        expr = _expr_in("id", chunk_ids)
        try:
            milvus_service.milvus_client.delete(expr=expr)  # type: ignore[union-attr]
            milvus_service.milvus_client.flush()  # type: ignore[union-attr]
        except Exception as exc:
            raise AdminMilvusError(f"Milvus delete 失败: {exc}") from exc

        chunk_records = []
        id_set = set(chunk_ids)
        for record in records:
            qa_id = str(record.get("id") or "")
            if qa_id in id_set:
                chunk_records.append(
                    {key: value for key, value in record.items() if key in allowed_fields}
                )
        if not chunk_records:
            continue
        try:
            milvus_service.milvus_client.insert(chunk_records)  # type: ignore[union-attr]
            milvus_service.milvus_client.flush()  # type: ignore[union-attr]
        except Exception as exc:
            raise AdminMilvusError(f"Milvus insert 失败: {exc}") from exc


def update_qa_item_fields(qa_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    qa_id = str(qa_id)
    records = fetch_records_by_ids([qa_id])
    record = records.get(qa_id)
    if not record:
        raise KeyError("记录不存在")
    for key, value in patch.items():
        record[key] = value
    replace_records([record])
    return get_qa_item(qa_id)


def batch_update_fields(ids: List[str], patch: Dict[str, Any]) -> Dict[str, Any]:
    ids = [str(item) for item in ids if item]
    if not ids:
        return {"updated": [], "missing": [], "failed": []}
    records = fetch_records_by_ids(ids)
    missing = [item for item in ids if item not in records]
    updated_records: List[Dict[str, Any]] = []
    for qa_id, record in records.items():
        for key, value in patch.items():
            record[key] = value
        updated_records.append(record)
    replace_records(updated_records)
    return {"updated": list(records.keys()), "missing": missing, "failed": []}


def hard_delete(ids: List[str]) -> Dict[str, Any]:
    ids = [str(item) for item in ids if item]
    if not ids:
        return {"deleted": [], "failed": []}
    _ensure_milvus_ready()
    deleted: List[str] = []
    failed: List[Dict[str, str]] = []
    chunk_size = 200
    for index in range(0, len(ids), chunk_size):
        chunk = ids[index : index + chunk_size]
        expr = _expr_in("id", chunk)
        try:
            milvus_service.milvus_client.delete(expr=expr)  # type: ignore[union-attr]
            milvus_service.milvus_client.flush()  # type: ignore[union-attr]
            deleted.extend(chunk)
        except Exception as exc:
            failed.append({"expr": expr, "error": str(exc)})
    if deleted:
        try:
            delete_debug_entries(deleted)
        except Exception:
            pass
    return {"deleted": deleted, "failed": failed}


def export_items_to_json(
    items: List[Dict[str, Any]],
    *,
    prefix: str,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    outputs_dir = str(CONFIG["outputs_dir"])
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = int(time.time())
    path = os.path.join(outputs_dir, f"{prefix}_{timestamp}.json")
    payload: Dict[str, Any] = {
        "exported_at": timestamp,
        "total": len(items),
        "items": items,
    }
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


__all__ = [
    "batch_update_fields",
    "export_items_to_json",
    "fetch_records_by_ids",
    "hard_delete",
    "replace_records",
    "update_qa_item_fields",
]
