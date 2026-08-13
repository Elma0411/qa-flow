"""Versioned storage and querying for structural document content chunks."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


DOC_TREE_CHUNKS_SCHEMA_VERSION = 2
LEGACY_DOC_TREE_CHUNKS_COLLECTION = "doc_tree_chunks"
DOC_TREE_CHUNKS_COLLECTION = str(
    (CONFIG.get("milvus") or {}).get("doc_tree_chunks_v2_collection")
    or "doc_content_chunks_v2"
).strip() or "doc_content_chunks_v2"


def build_doc_id(original_filename: str, text: str) -> str:
    name = str(original_filename or "").strip()
    content_hash = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    return hashlib.sha1(f"{name}|||{content_hash}".encode("utf-8")).hexdigest()


def _truncate(value: Any, max_len: int) -> str:
    return str(value or "")[:max_len]


def _json_list(value: Any) -> str:
    values = value if isinstance(value, list) else []
    return json.dumps([str(item) for item in values if str(item)], ensure_ascii=False)


class DocumentChunkStore:
    """Own the process-wide Milvus collection handle for structural chunks."""

    _QUERY_FIELDS = [
        "id",
        "doc_id",
        "task_id",
        "original_filename",
        "chunk_index",
        "section_chunk_index",
        "section_path",
        "section_parent_path",
        "section_level",
        "section_is_leaf",
        "title_path",
        "fragment_group_id",
        "fragment_index",
        "fragment_count",
        "content_kind",
        "source_asset_ids_json",
        "created_at",
        "schema_version",
    ]

    def __init__(self, collection_name: str = DOC_TREE_CHUNKS_COLLECTION) -> None:
        self.collection_name = str(collection_name or "").strip()
        self._client: Optional[Collection] = None
        self._init_lock = threading.Lock()
        self._write_lock = threading.Lock()

    @property
    def client(self) -> Optional[Collection]:
        return self._client

    def ensure_initialized(self) -> Tuple[bool, str]:
        if not milvus_service.MILVUS_AVAILABLE or not _PYMILVUS_AVAILABLE:
            return False, "Milvus 相关库未安装"
        if utility is None:
            return False, "pymilvus utility 不可用"
        with self._init_lock:
            try:
                if not milvus_service.milvus_client:
                    ok, message = milvus_service.init_milvus()
                    if not ok:
                        return False, f"Milvus 初始化失败: {message}"
                if utility.has_collection(self.collection_name):
                    collection = Collection(self.collection_name)
                    expected = {
                        "section_path",
                        "section_parent_path",
                        "section_is_leaf",
                        "fragment_group_id",
                        "content_kind",
                        "source_asset_ids_json",
                        "schema_version",
                    }
                    existing = {field.name for field in collection.schema.fields}
                    missing = sorted(expected - existing)
                    if missing:
                        return False, (
                            f"{self.collection_name} schema 不完整，缺少 {missing}；"
                            "请更换 v2 集合名或显式重建，系统不会回退旧集合"
                        )
                    collection.load()
                    self._client = collection
                    return True, f"{self.collection_name} 已连接"

                vector_dim = int((CONFIG.get("milvus") or {}).get("vector_dim") or 1024)
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                    FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=128),
                    FieldSchema(name="original_filename", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="section_chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="section_parent_path", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="section_level", dtype=DataType.INT64),
                    FieldSchema(name="section_is_leaf", dtype=DataType.BOOL),
                    FieldSchema(name="title_path", dtype=DataType.VARCHAR, max_length=2048),
                    FieldSchema(name="fragment_group_id", dtype=DataType.VARCHAR, max_length=128),
                    FieldSchema(name="fragment_index", dtype=DataType.INT64),
                    FieldSchema(name="fragment_count", dtype=DataType.INT64),
                    FieldSchema(name="content_kind", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="source_asset_ids_json", dtype=DataType.VARCHAR, max_length=8192),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
                    FieldSchema(name="created_at", dtype=DataType.INT64),
                    FieldSchema(name="schema_version", dtype=DataType.INT64),
                ]
                collection = Collection(
                    self.collection_name,
                    CollectionSchema(fields, "QA Flow structural content chunks schema v2"),
                )
                metric_type = str((CONFIG.get("milvus") or {}).get("metric_type") or "IP").upper()
                index_params = {
                    "index_type": (CONFIG.get("milvus") or {}).get("index_type") or "HNSW",
                    "metric_type": metric_type,
                    "params": (CONFIG.get("milvus") or {}).get("index_params") or {"M": 16, "efConstruction": 200},
                }
                try:
                    collection.create_index("embedding_vector", index_params=index_params)
                except Exception as exc:
                    if metric_type == "COSINE" and "metric" in str(exc).lower():
                        collection.create_index(
                            "embedding_vector",
                            index_params={**index_params, "metric_type": "IP"},
                        )
                        logger.warning(
                            "%s: COSINE unsupported; using normalized-vector IP",
                            self.collection_name,
                        )
                    else:
                        raise
                for field_name in (
                    "doc_id",
                    "task_id",
                    "original_filename",
                    "section_path",
                    "fragment_group_id",
                    "chunk_index",
                ):
                    try:
                        collection.create_index(field_name)
                    except Exception:
                        pass
                collection.load()
                self._client = collection
                return True, f"{self.collection_name} 已创建并连接"
            except Exception as exc:
                return False, f"{self.collection_name} 初始化失败: {exc}"

    @staticmethod
    def _validate_and_prepare(
        chunks_meta: Sequence[Dict[str, Any]],
        *,
        expected_task_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not chunks_meta:
            raise ValueError("chunks must contain at least one v2 content chunk")
        expected_task = str(expected_task_id or "").strip()
        now = int(time.time())
        rows: List[Dict[str, Any]] = []
        embedding_texts: List[str] = []
        seen_ids: set[str] = set()
        seen_positions: set[Tuple[str, str, int]] = set()
        for row_index, raw in enumerate(chunks_meta, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"chunk #{row_index} must be an object")
            chunk_id = str(raw.get("chunk_id") or "").strip()
            task_id = str(raw.get("task_id") or "").strip()
            doc_id = str(raw.get("doc_id") or "").strip()
            original_filename = str(raw.get("original_filename") or "").strip()
            section_path = str(raw.get("section_path") or "").strip()
            fragment_group_id = str(raw.get("fragment_group_id") or "").strip()
            content_kind = str(raw.get("content_kind") or "").strip()
            text = str(raw.get("text") or "").strip()
            required_values = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "task_id": task_id,
                "original_filename": original_filename,
                "section_path": section_path,
                "fragment_group_id": fragment_group_id,
                "content_kind": content_kind,
                "text": text,
            }
            missing = [key for key, value in required_values.items() if not value]
            if missing:
                raise ValueError(f"chunk #{row_index} v2 contract missing={missing}")
            if expected_task and task_id != expected_task:
                raise ValueError("every rebuild chunk must carry the requested task_id")
            if chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id in rebuild payload: {chunk_id}")
            seen_ids.add(chunk_id)

            chunk_index = int(raw.get("chunk_index") or 0)
            section_chunk_index = int(raw.get("section_chunk_index") or 0)
            section_level = int(raw.get("section_level") or 0)
            fragment_index = int(raw.get("fragment_index") or 0)
            fragment_count = int(raw.get("fragment_count") or 0)
            if chunk_index < 1 or section_chunk_index < 1 or section_level < 1:
                raise ValueError(f"chunk #{row_index} has invalid structural index")
            if fragment_index < 1 or fragment_count < fragment_index:
                raise ValueError(f"chunk #{row_index} has invalid fragment position")
            position = (task_id, doc_id, chunk_index)
            if position in seen_positions:
                raise ValueError(
                    f"duplicate chunk_index={chunk_index} for task_id={task_id}, doc_id={doc_id}"
                )
            seen_positions.add(position)

            title_path = str(raw.get("title_path") or "").strip()
            embedding_text = str(raw.get("text_for_embedding") or "").strip() or (
                f"{title_path}\n{text}".strip() if title_path else text
            )
            rows.append(
                {
                    "id": _truncate(chunk_id, 128),
                    "doc_id": _truncate(doc_id, 128),
                    "task_id": _truncate(task_id, 128),
                    "original_filename": _truncate(original_filename, 512),
                    "chunk_index": chunk_index,
                    "section_chunk_index": section_chunk_index,
                    "section_path": _truncate(section_path, 256),
                    "section_parent_path": _truncate(raw.get("section_parent_path"), 256),
                    "section_level": section_level,
                    "section_is_leaf": bool(raw.get("section_is_leaf", True)),
                    "title_path": _truncate(title_path, 2048),
                    "fragment_group_id": _truncate(fragment_group_id, 128),
                    "fragment_index": fragment_index,
                    "fragment_count": fragment_count,
                    "content_kind": _truncate(content_kind, 64),
                    "source_asset_ids_json": _truncate(_json_list(raw.get("source_asset_ids")), 8192),
                    "text": _truncate(text, 65535),
                    "created_at": int(raw.get("created_at") or now),
                    "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
                }
            )
            embedding_texts.append(embedding_text)
        return rows, embedding_texts

    @staticmethod
    def _attach_embeddings(
        rows: List[Dict[str, Any]],
        embedding_texts: Sequence[str],
    ) -> None:
        embeddings = milvus_service.generate_embeddings(list(embedding_texts))
        if len(embeddings) != len(rows):
            raise RuntimeError(
                f"embedding count mismatch: rows={len(rows)}, embeddings={len(embeddings)}"
            )
        for row, embedding in zip(rows, embeddings):
            row["embedding_vector"] = embedding

    @staticmethod
    def _delete_ids(client: Collection, ids: Sequence[str]) -> None:
        for start in range(0, len(ids), 200):
            expression = "id in [" + ",".join(
                json.dumps(value) for value in ids[start : start + 200]
            ) + "]"
            client.delete(expr=expression)

    def store(
        self,
        chunks_meta: List[Dict[str, Any]],
        *,
        enable: bool = True,
    ) -> Dict[str, Any]:
        if not enable:
            return {"success": False, "message": "chunk 溯源索引保存未启用", "stored_count": 0}
        if not chunks_meta:
            return {"success": True, "message": "没有 chunk 需要入库", "stored_count": 0}
        rows, embedding_texts = self._validate_and_prepare(chunks_meta)
        ok, message = self.ensure_initialized()
        client = self._client
        if not ok or client is None:
            return {"success": False, "message": message, "stored_count": 0}
        self._attach_embeddings(rows, embedding_texts)
        with self._write_lock:
            self._delete_ids(client, [row["id"] for row in rows])
            client.insert(rows)
            client.flush()
        return {
            "success": True,
            "message": f"成功写入 {self.collection_name}: {len(rows)}",
            "stored_count": len(rows),
            "collection_name": self.collection_name,
            "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
        }

    @staticmethod
    def _decode_row(row: Dict[str, Any], *, include_text: bool) -> Dict[str, Any]:
        try:
            assets = json.loads(str(row.get("source_asset_ids_json") or "[]"))
        except Exception:
            assets = []
        return {
            "chunk_id": row.get("id"),
            "doc_id": row.get("doc_id"),
            "task_id": row.get("task_id"),
            "original_filename": row.get("original_filename"),
            "chunk_index": row.get("chunk_index"),
            "section_chunk_index": row.get("section_chunk_index"),
            "section_path": row.get("section_path"),
            "section_parent_path": row.get("section_parent_path"),
            "section_level": row.get("section_level"),
            "section_is_leaf": row.get("section_is_leaf"),
            "title_path": row.get("title_path"),
            "fragment_group_id": row.get("fragment_group_id"),
            "fragment_index": row.get("fragment_index"),
            "fragment_count": row.get("fragment_count"),
            "content_kind": row.get("content_kind"),
            "source_asset_ids": assets if isinstance(assets, list) else [],
            "created_at": row.get("created_at"),
            "schema_version": row.get("schema_version"),
            "text": row.get("text") if include_text else None,
        }

    def list_docs_by_task(self, task_id: str, *, max_rows: int = 16384) -> Dict[str, Any]:
        ok, message = self.ensure_initialized()
        client = self._client
        if not ok or client is None:
            return {"success": False, "message": message, "docs": []}
        value = str(task_id or "").strip()
        if not value:
            return {"success": False, "message": "task_id 不能为空", "docs": []}
        try:
            rows = client.query(
                expr=f"task_id == {json.dumps(value)}",
                output_fields=["doc_id", "original_filename", "created_at"],
                limit=max_rows,
            )
        except Exception as exc:
            return {"success": False, "message": f"查询失败: {exc}", "docs": []}
        docs: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            doc_id = str(row.get("doc_id") or "").strip()
            if not doc_id:
                continue
            entry = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "original_filename": row.get("original_filename"),
                    "chunk_count": 0,
                    "created_at": row.get("created_at"),
                    "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
                },
            )
            entry["chunk_count"] += 1
        values = sorted(
            docs.values(),
            key=lambda item: (str(item.get("original_filename") or ""), item["doc_id"]),
        )
        return {"success": True, "message": "ok", "task_id": value, "docs": values}

    def fetch_chunks_by_doc_id(
        self,
        doc_id: str,
        *,
        task_id: Optional[str] = None,
        include_text: bool = True,
        max_rows: int = 16384,
    ) -> Dict[str, Any]:
        ok, message = self.ensure_initialized()
        client = self._client
        if not ok or client is None:
            return {"success": False, "message": message, "chunks": []}
        value = str(doc_id or "").strip()
        if not value:
            return {"success": False, "message": "doc_id 不能为空", "chunks": []}
        expressions = [f"doc_id == {json.dumps(value)}"]
        task_value = str(task_id or "").strip()
        if task_value:
            expressions.append(f"task_id == {json.dumps(task_value)}")
        fields = list(self._QUERY_FIELDS) + (["text"] if include_text else [])
        try:
            rows = client.query(
                expr=" and ".join(expressions),
                output_fields=fields,
                limit=max_rows,
            )
        except Exception as exc:
            return {"success": False, "message": f"查询失败: {exc}", "chunks": []}
        chunks = [
            self._decode_row(row, include_text=include_text)
            for row in rows
            if isinstance(row, dict)
        ]
        chunks.sort(
            key=lambda item: (int(item.get("chunk_index") or 0), str(item.get("chunk_id") or ""))
        )
        return {
            "success": True,
            "message": "ok",
            "doc_id": value,
            "task_id": task_value or None,
            "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
            "chunks": chunks,
        }

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        ok, message = self.ensure_initialized()
        client = self._client
        if not ok or client is None:
            return {"success": False, "message": message}
        value = str(chunk_id or "").strip()
        if not value:
            return {"success": False, "message": "chunk_id 不能为空"}
        try:
            rows = client.query(
                expr=f"id == {json.dumps(value)}",
                output_fields=list(self._QUERY_FIELDS) + ["text"],
                limit=1,
            )
        except Exception as exc:
            return {"success": False, "message": f"查询失败: {exc}"}
        if not rows:
            return {"success": False, "message": "chunk 不存在", "chunk_id": value}
        return {"success": True, "chunk": self._decode_row(rows[0], include_text=True)}

    def rebuild(self, chunks_meta: List[Dict[str, Any]], *, task_id: str) -> Dict[str, Any]:
        """Validate and embed first, then atomically replace one task's visible rows."""
        value = str(task_id or "").strip()
        if not value:
            raise ValueError("task_id is required for explicit chunk rebuild")
        rows, embedding_texts = self._validate_and_prepare(
            chunks_meta,
            expected_task_id=value,
        )
        filenames = {str(row["original_filename"]) for row in rows}
        if len(filenames) != 1:
            raise ValueError("one rebuild request must contain exactly one original_filename")
        original_filename = next(iter(filenames))
        self._attach_embeddings(rows, embedding_texts)
        ok, message = self.ensure_initialized()
        client = self._client
        if not ok or client is None:
            return {"success": False, "message": message, "stored_count": 0}

        old_rows: List[Dict[str, Any]] = []
        scope_expr = (
            f"task_id == {json.dumps(value)} and "
            f"original_filename == {json.dumps(original_filename)}"
        )
        with self._write_lock:
            try:
                old_rows = client.query(
                    expr=scope_expr,
                    output_fields=list(self._QUERY_FIELDS) + ["text", "embedding_vector"],
                    limit=16384,
                )
            except Exception as exc:
                return {
                    "success": False,
                    "message": f"读取待替换任务失败，未修改现有数据: {exc}",
                    "stored_count": 0,
                }
            old_ids = [str(row.get("id") or "") for row in old_rows if str(row.get("id") or "")]
            new_ids = [str(row["id"]) for row in rows]
            new_id_set = set(new_ids)
            try:
                client.upsert(rows)
                client.flush()
                stale_ids = [chunk_id for chunk_id in old_ids if chunk_id not in new_id_set]
                if stale_ids:
                    self._delete_ids(client, stale_ids)
                    client.flush()
            except Exception as exc:
                logger.exception("%s rebuild failed for task_id=%s", self.collection_name, value)
                try:
                    old_id_set = set(old_ids)
                    inserted_ids = [chunk_id for chunk_id in new_ids if chunk_id not in old_id_set]
                    if inserted_ids:
                        self._delete_ids(client, inserted_ids)
                    if old_rows:
                        client.upsert(old_rows)
                    client.flush()
                except Exception as restore_exc:
                    raise RuntimeError(
                        "v2 rebuild failed and rollback failed; "
                        f"write_error={exc}; rollback_error={restore_exc}"
                    ) from restore_exc
                return {
                    "success": False,
                    "message": f"v2 重建写入失败，原任务数据已恢复: {exc}",
                    "stored_count": 0,
                    "rebuild_task_id": value,
                }
        return {
            "success": True,
            "message": f"成功重建 {self.collection_name}: {len(rows)}",
            "stored_count": len(rows),
            "collection_name": self.collection_name,
            "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
            "rebuild_task_id": value,
            "rebuild_original_filename": original_filename,
            "replaced_count": len(old_rows),
        }


DOCUMENT_CHUNK_STORE = DocumentChunkStore()


def ensure_doc_tree_chunks_initialized() -> Tuple[bool, str]:
    return DOCUMENT_CHUNK_STORE.ensure_initialized()


def store_doc_tree_chunks(
    chunks_meta: List[Dict[str, Any]],
    *,
    enable: bool = True,
) -> Dict[str, Any]:
    return DOCUMENT_CHUNK_STORE.store(chunks_meta, enable=enable)


def list_docs_by_task(task_id: str, *, max_rows: int = 16384) -> Dict[str, Any]:
    return DOCUMENT_CHUNK_STORE.list_docs_by_task(task_id, max_rows=max_rows)


def fetch_chunks_by_doc_id(
    doc_id: str,
    *,
    task_id: Optional[str] = None,
    include_text: bool = True,
    max_rows: int = 16384,
) -> Dict[str, Any]:
    return DOCUMENT_CHUNK_STORE.fetch_chunks_by_doc_id(
        doc_id,
        task_id=task_id,
        include_text=include_text,
        max_rows=max_rows,
    )


def get_chunk_by_id(chunk_id: str) -> Dict[str, Any]:
    return DOCUMENT_CHUNK_STORE.get_chunk_by_id(chunk_id)


def rebuild_doc_tree_chunks(
    chunks_meta: List[Dict[str, Any]],
    *,
    task_id: str,
) -> Dict[str, Any]:
    return DOCUMENT_CHUNK_STORE.rebuild(chunks_meta, task_id=task_id)


__all__ = [
    "DOC_TREE_CHUNKS_COLLECTION",
    "DOC_TREE_CHUNKS_SCHEMA_VERSION",
    "DOCUMENT_CHUNK_STORE",
    "DocumentChunkStore",
    "LEGACY_DOC_TREE_CHUNKS_COLLECTION",
    "build_doc_id",
    "ensure_doc_tree_chunks_initialized",
    "fetch_chunks_by_doc_id",
    "get_chunk_by_id",
    "list_docs_by_task",
    "rebuild_doc_tree_chunks",
    "store_doc_tree_chunks",
]
