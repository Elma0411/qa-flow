# 文件作用：构建、入库、查询文档块树及其关联问答。
# 关联说明：与 milvus/storage 服务并列，专门维护文档块树结构和块到 QA 的关系。

import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import CONFIG
from app.core.logger import logger
from app.services import milvus as milvus_service

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

    _PYMILVUS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    Collection = Any  # type: ignore
    CollectionSchema = Any  # type: ignore
    DataType = Any  # type: ignore
    FieldSchema = Any  # type: ignore
    utility = None  # type: ignore
    _PYMILVUS_AVAILABLE = False


DOC_TREE_CHUNKS_COLLECTION = str(
    (CONFIG.get("milvus") or {}).get("doc_tree_chunks_collection") or "doc_tree_chunks"
).strip() or "doc_tree_chunks"

doc_tree_chunks_client: Optional[Collection] = None


def build_doc_id(original_filename: str, text: str) -> str:
    """
    Stable doc_id for the same (filename + content).

    - Changing content changes doc_id.
    - Same content but different filename gets different doc_id (helps human grouping).
    """
    name = str(original_filename or "").strip()
    content_hash = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    return hashlib.sha1(f"{name}|||{content_hash}".encode("utf-8")).hexdigest()


def _truncate(text: Any, max_len: int) -> str:
    s = str(text or "")
    if max_len <= 0:
        return s
    return s[:max_len]


def ensure_doc_tree_chunks_initialized() -> Tuple[bool, str]:
    """
    Ensure the doc_tree_chunks collection exists and `doc_tree_chunks_client` is ready.
    """
    global doc_tree_chunks_client

    if not milvus_service.MILVUS_AVAILABLE or not _PYMILVUS_AVAILABLE:
        return False, "Milvus 相关库未安装"
    if utility is None:
        return False, "pymilvus utility 不可用"

    try:
        if not milvus_service.milvus_client:
            ok, msg = milvus_service.init_milvus()
            if not ok:
                return False, f"Milvus 初始化失败: {msg}"

        if utility.has_collection(DOC_TREE_CHUNKS_COLLECTION):
            doc_tree_chunks_client = Collection(DOC_TREE_CHUNKS_COLLECTION)
            return True, "doc_tree_chunks 已连接"

        vector_dim = int((CONFIG.get("milvus") or {}).get("vector_dim") or 1024)
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="original_filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="index_path", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="title_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="parent_index_path", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="root_index_path", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="level", dtype=DataType.INT64),
            FieldSchema(name="is_leaf", dtype=DataType.BOOL),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
            FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, "Document tree chunks for traceability and retrieval")
        collection = Collection(DOC_TREE_CHUNKS_COLLECTION, schema)

        metric_type_cfg = (CONFIG.get("milvus") or {}).get("metric_type") or "IP"
        metric_type_cfg = str(metric_type_cfg).strip().upper() or "IP"
        index_params = {
            "index_type": (CONFIG.get("milvus") or {}).get("index_type") or "HNSW",
            "metric_type": metric_type_cfg,
            "params": (CONFIG.get("milvus") or {}).get("index_params") or {"M": 16, "efConstruction": 200},
        }
        try:
            collection.create_index(field_name="embedding_vector", index_params=index_params)
        except Exception as exc:
            msg = str(exc)
            if metric_type_cfg == "COSINE" and ("metric type not found" in msg or "not supported" in msg):
                fallback_params = dict(index_params)
                fallback_params["metric_type"] = "IP"
                collection.create_index(field_name="embedding_vector", index_params=fallback_params)
                logger.warning("doc_tree_chunks: COSINE 不受支持，已回退为 IP（请确保向量已归一化）")
            else:
                raise

        for scalar_field in (
            "doc_id",
            "task_id",
            "original_filename",
            "index_path",
            "root_index_path",
            "chunk_index",
        ):
            try:
                collection.create_index(field_name=scalar_field)
            except Exception:
                pass

        doc_tree_chunks_client = Collection(DOC_TREE_CHUNKS_COLLECTION)
        return True, "doc_tree_chunks 已创建并连接"
    except Exception as exc:
        return False, f"doc_tree_chunks 初始化失败: {exc}"


def store_doc_tree_chunks(
    chunks_meta: List[Dict[str, Any]],
    *,
    enable: bool = True,
) -> Dict[str, Any]:
    """
    Store doc chunks (leaf nodes) into the doc_tree_chunks collection.

    `chunks_meta` should include at least:
      - id, doc_id, task_id, original_filename, chunk_index
      - index_path, title_path, text, text_for_embedding
    """
    if not enable:
        return {"success": False, "message": "chunk 入库未启用", "stored_count": 0}

    ok, msg = ensure_doc_tree_chunks_initialized()
    if not ok or not doc_tree_chunks_client:
        return {"success": False, "message": msg, "stored_count": 0}

    if not chunks_meta:
        return {"success": True, "message": "没有 chunk 需要入库", "stored_count": 0}

    try:
        doc_tree_chunks_client.load()
    except Exception:
        pass

    now_ts = int(time.time())
    prepared_rows: List[Dict[str, Any]] = []
    embed_texts: List[str] = []

    for raw in chunks_meta:
        if not isinstance(raw, dict):
            continue
        chunk_id = str(raw.get("chunk_id") or raw.get("id") or "").strip()
        if not chunk_id:
            continue
        title_path = str(raw.get("title_path") or "").strip()
        text = str(raw.get("text") or "").strip()
        text_for_embedding = str(raw.get("text_for_embedding") or "").strip() or (
            (title_path + "\n" + text).strip() if title_path else text
        )

        row: Dict[str, Any] = {
            "id": _truncate(chunk_id, 128),
            "doc_id": _truncate(raw.get("doc_id"), 128),
            "task_id": _truncate(raw.get("task_id"), 128),
            "original_filename": _truncate(raw.get("original_filename"), 512),
            "chunk_index": int(raw.get("chunk_index") or 0),
            "index_path": _truncate(raw.get("index_path"), 128),
            "title_path": _truncate(title_path, 2048),
            "parent_index_path": _truncate(raw.get("parent_index_path"), 128),
            "root_index_path": _truncate(raw.get("root_index_path"), 128),
            "level": int(raw.get("level") or 0),
            "is_leaf": bool(raw.get("is_leaf", True)),
            "text": _truncate(text, 16384),
            "created_at": int(raw.get("created_at") or now_ts),
        }
        prepared_rows.append(row)
        embed_texts.append(text_for_embedding)

    if not prepared_rows:
        return {"success": False, "message": "无有效 chunk 记录可入库", "stored_count": 0}

    # Best-effort de-dup by primary key
    try:
        ids_to_delete = [r["id"] for r in prepared_rows if r.get("id")]
        if ids_to_delete:
            chunk_size = 200
            for i in range(0, len(ids_to_delete), chunk_size):
                chunk = [x for x in ids_to_delete[i : i + chunk_size] if x]
                if not chunk:
                    continue
                expr = "id in [" + ",".join(f'\"{x}\"' for x in chunk) + "]"
                doc_tree_chunks_client.delete(expr=expr)
            doc_tree_chunks_client.flush()
    except Exception as exc:
        logger.warning("doc_tree_chunks de-dup skipped: %s", exc)

    embeddings = milvus_service.generate_embeddings(embed_texts)
    for idx, emb in enumerate(embeddings):
        prepared_rows[idx]["embedding_vector"] = emb

    doc_tree_chunks_client.insert(prepared_rows)
    doc_tree_chunks_client.flush()
    return {
        "success": True,
        "message": f"成功写入 doc_tree_chunks: {len(prepared_rows)}",
        "stored_count": len(prepared_rows),
        "collection_name": DOC_TREE_CHUNKS_COLLECTION,
    }


def list_docs_by_task(task_id: str, *, max_rows: int = 16384) -> Dict[str, Any]:
    ok, msg = ensure_doc_tree_chunks_initialized()
    if not ok or not doc_tree_chunks_client:
        return {"success": False, "message": msg, "docs": []}

    safe_task_id = str(task_id or "").replace('"', '\\"').strip()
    if not safe_task_id:
        return {"success": False, "message": "task_id 不能为空", "docs": []}

    try:
        doc_tree_chunks_client.load()
    except Exception:
        pass

    expr = f'task_id == "{safe_task_id}"'
    rows: List[Dict[str, Any]] = []
    try:
        rows = doc_tree_chunks_client.query(
            expr=expr,
            output_fields=["doc_id", "original_filename", "created_at"],
            limit=max_rows,
        )
    except Exception as exc:
        return {"success": False, "message": f"查询失败: {exc}", "docs": []}

    docs: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        doc_id = str((r or {}).get("doc_id") or "").strip()
        if not doc_id:
            continue
        info = docs.get(doc_id) or {
            "doc_id": doc_id,
            "original_filename": (r or {}).get("original_filename"),
            "chunk_count": 0,
            "created_at": (r or {}).get("created_at"),
        }
        info["chunk_count"] = int(info.get("chunk_count") or 0) + 1
        docs[doc_id] = info

    doc_list = sorted(docs.values(), key=lambda x: (str(x.get("original_filename") or ""), str(x.get("doc_id") or "")))
    return {"success": True, "message": "ok", "task_id": safe_task_id, "docs": doc_list}


def fetch_chunks_by_doc_id(
    doc_id: str,
    *,
    task_id: Optional[str] = None,
    include_text: bool = True,
    max_rows: int = 16384,
) -> Dict[str, Any]:
    ok, msg = ensure_doc_tree_chunks_initialized()
    if not ok or not doc_tree_chunks_client:
        return {"success": False, "message": msg, "chunks": []}

    safe_doc_id = str(doc_id or "").strip()
    if not safe_doc_id:
        return {"success": False, "message": "doc_id 不能为空", "chunks": []}

    safe_task_id = None
    if task_id is not None:
        safe_task_id = str(task_id or "").strip() or None

    try:
        doc_tree_chunks_client.load()
    except Exception:
        pass

    fields = [
        "id",
        "doc_id",
        "task_id",
        "original_filename",
        "chunk_index",
        "index_path",
        "title_path",
        "parent_index_path",
        "root_index_path",
        "level",
        "is_leaf",
        "created_at",
    ]
    if include_text:
        fields.append("text")

    expr_parts = [f"doc_id == {json.dumps(safe_doc_id)}"]
    if safe_task_id:
        expr_parts.append(f"task_id == {json.dumps(safe_task_id)}")
    expr = " and ".join(expr_parts)
    try:
        rows = doc_tree_chunks_client.query(expr=expr, output_fields=fields, limit=max_rows)
    except Exception as exc:
        return {"success": False, "message": f"查询失败: {exc}", "chunks": []}

    chunks = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        chunks.append(
            {
                "chunk_id": r.get("id"),
                "doc_id": r.get("doc_id"),
                "task_id": r.get("task_id"),
                "original_filename": r.get("original_filename"),
                "chunk_index": r.get("chunk_index"),
                "index_path": r.get("index_path"),
                "title_path": r.get("title_path"),
                "parent_index_path": r.get("parent_index_path"),
                "root_index_path": r.get("root_index_path"),
                "level": r.get("level"),
                "is_leaf": r.get("is_leaf"),
                "created_at": r.get("created_at"),
                "text": r.get("text") if include_text else None,
            }
        )
    chunks.sort(key=lambda x: (int(x.get("chunk_index") or 0), str(x.get("chunk_id") or "")))
    return {
        "success": True,
        "message": "ok",
        "doc_id": safe_doc_id,
        "task_id": safe_task_id,
        "chunks": chunks,
    }


def get_chunk_by_id(chunk_id: str) -> Dict[str, Any]:
    ok, msg = ensure_doc_tree_chunks_initialized()
    if not ok or not doc_tree_chunks_client:
        return {"success": False, "message": msg}

    safe_id = str(chunk_id or "").replace('"', '\\"').strip()
    if not safe_id:
        return {"success": False, "message": "chunk_id 不能为空"}

    try:
        doc_tree_chunks_client.load()
    except Exception:
        pass

    try:
        rows = doc_tree_chunks_client.query(
            expr=f'id == "{safe_id}"',
            output_fields=[
                "id",
                "doc_id",
                "task_id",
                "original_filename",
                "chunk_index",
                "index_path",
                "title_path",
                "parent_index_path",
                "root_index_path",
                "level",
                "is_leaf",
                "text",
                "created_at",
            ],
            limit=1,
        )
    except Exception as exc:
        return {"success": False, "message": f"查询失败: {exc}"}

    if not rows:
        return {"success": False, "message": "chunk 不存在", "chunk_id": safe_id}

    r = rows[0]
    return {
        "success": True,
        "chunk": {
            "chunk_id": r.get("id"),
            "doc_id": r.get("doc_id"),
            "task_id": r.get("task_id"),
            "original_filename": r.get("original_filename"),
            "chunk_index": r.get("chunk_index"),
            "index_path": r.get("index_path"),
            "title_path": r.get("title_path"),
            "parent_index_path": r.get("parent_index_path"),
            "root_index_path": r.get("root_index_path"),
            "level": r.get("level"),
            "is_leaf": r.get("is_leaf"),
            "text": r.get("text"),
            "created_at": r.get("created_at"),
        },
    }


__all__ = [
    "DOC_TREE_CHUNKS_COLLECTION",
    "build_doc_id",
    "ensure_doc_tree_chunks_initialized",
    "store_doc_tree_chunks",
    "list_docs_by_task",
    "fetch_chunks_by_doc_id",
    "get_chunk_by_id",
]
