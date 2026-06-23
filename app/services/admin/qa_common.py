# 文件作用：提供管理端问答服务共用的错误处理和字段转换工具。
# 关联说明：被 qa_query 和 qa_write 共同复用，集中处理 Milvus 字段和异常。

import json
from typing import Any, Dict, Iterable, List

from app.core.logger import logger
from app.services import milvus as milvus_service
from .meta import AdminMeta


class AdminMilvusError(RuntimeError):
    pass


def _ensure_milvus_ready() -> None:
    if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
        raise AdminMilvusError("Milvus服务不可用，请确认向量数据库已启动并连接")
    try:
        milvus_service.milvus_client.load()
    except Exception as exc:
        raise AdminMilvusError(f"Milvus collection load 失败: {exc}") from exc


def _get_allowed_fields() -> List[str]:
    _ensure_milvus_ready()
    try:
        fields = [f.name for f in milvus_service.milvus_client.schema.fields]  # type: ignore[union-attr]
        fields.sort()
        return fields
    except Exception as exc:
        raise AdminMilvusError(f"读取 Milvus schema 失败: {exc}") from exc


def _resolve_source_field() -> str:
    try:
        fields = set(_get_allowed_fields())
        if "source" in fields:
            return "source"
        if "source_id" in fields:
            return "source_id"
    except Exception:
        return "source"
    return "source"


def _escape_expr_value(value: str) -> str:
    return str(value).replace('"', '\\"')


def _expr_in(field: str, values: Iterable[str]) -> str:
    escaped = [_escape_expr_value(v) for v in values if v]
    quoted = ",".join(f'"{v}"' for v in escaped)
    return f"{field} in [{quoted}]"


def _parse_json_field(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _default_admin_meta(qa_id: str) -> AdminMeta:
    return AdminMeta(
        id=str(qa_id),
        is_active=True,
        review_status=None,
        review_note=None,
        updated_at=None,
    )


def _is_evaluated(row: Dict[str, Any]) -> bool:
    method = row.get("evaluation_method")
    if isinstance(method, str) and method.strip():
        return True
    for key in ("llm_scores", "local_scores"):
        val = row.get(key)
        if isinstance(val, str) and val.strip() and val.strip() not in ("{}", "null"):
            return True
    return False


def _fetch_rows(
    expr: str,
    output_fields: List[str],
    *,
    batch_size: int = 1000,
    max_rows: int = 50000,
) -> List[Dict[str, Any]]:
    _ensure_milvus_ready()
    rows_out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        rows = milvus_service.milvus_client.query(  # type: ignore[union-attr]
            expr=expr,
            output_fields=output_fields,
            offset=offset,
            limit=min(batch_size, max_rows - len(rows_out)),
        )
        if not rows:
            break
        rows_out.extend(rows)
        offset += len(rows)
        if len(rows) < batch_size:
            break
        if len(rows_out) >= max_rows:
            logger.warning("admin list reached max_rows=%s; expr=%s", max_rows, expr)
            break
    return rows_out


__all__ = [
    "AdminMilvusError",
    "_default_admin_meta",
    "_ensure_milvus_ready",
    "_escape_expr_value",
    "_expr_in",
    "_fetch_rows",
    "_get_allowed_fields",
    "_is_evaluated",
    "_parse_json_field",
    "_resolve_source_field",
]
