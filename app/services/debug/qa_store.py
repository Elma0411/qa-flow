# 文件作用：读写问答调试样本和本地调试存储。
# 关联说明：为 chunk_qa.py 和调试接口提供本地 QA 样本读写。

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List

from app.core.config import CONFIG

_SQLITE_MAX_VARS = 900


def _get_db_path() -> str:
    configured = os.environ.get("QA_DEBUG_DB_PATH")
    if configured:
        return configured
    outputs_dir = str(CONFIG["outputs_dir"])
    return os.path.join(outputs_dir, "qa_debug.sqlite3")


def _connect() -> sqlite3.Connection:
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_debug_payloads (
                qa_id TEXT PRIMARY KEY,
                task_id TEXT,
                original_filename TEXT,
                source_chunk_id TEXT,
                source_chunk_index INTEGER,
                source_chunk_title_path TEXT,
                question TEXT,
                answer TEXT,
                created_at INTEGER,
                updated_at INTEGER,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_debug_source_chunk_id ON qa_debug_payloads(source_chunk_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_debug_task_id ON qa_debug_payloads(task_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qa_debug_original_filename ON qa_debug_payloads(original_filename)"
        )
        conn.commit()


def _chunked(values: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [values]
    return [values[index : index + size] for index in range(0, len(values), size)]


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_loads(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_debug_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": item.get("id"),
        "task_id": item.get("task_id"),
        "original_filename": item.get("original_filename"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "question_type": item.get("question_type"),
        "question_type_reason": item.get("question_type_reason"),
        "knowledge_category": item.get("knowledge_category"),
        "knowledge_category_reason": item.get("knowledge_category_reason"),
        "knowledge_category_confidence": item.get("knowledge_category_confidence"),
        "difficulty_level": item.get("difficulty_level"),
        "difficulty_score": item.get("difficulty_score"),
        "options": item.get("options"),
        "correct_option": item.get("correct_option"),
        "answer_explanation": item.get("answer_explanation"),
        "filtered": item.get("filtered"),
        "average_score": item.get("average_score"),
        "evaluation_method": item.get("evaluation_method"),
        "evaluation": item.get("evaluation"),
        "unsupervised_evaluation": item.get("unsupervised_evaluation"),
        "source": item.get("source"),
        "source_fact_id": item.get("source_fact_id"),
        "source_fact_text": item.get("source_fact_text"),
        "source_anchor_text": item.get("source_anchor_text"),
        "source_chunk_id": item.get("source_chunk_id") or item.get("source"),
        "source_chunk_index": item.get("source_chunk_index"),
        "source_chunk_title_path": item.get("source_chunk_title_path"),
        "evidence_chunk_ids": item.get("evidence_chunk_ids") or [],
        "qa_generation_unit_id": item.get("qa_generation_unit_id"),
        "qa_generation_unit_text": item.get("qa_generation_unit_text"),
        "retrieval_query": item.get("retrieval_query"),
        "must_have_terms": item.get("must_have_terms") or [],
        "answer_scope": item.get("answer_scope"),
        "evidence_usage": item.get("evidence_usage") or [],
        "retrieval_trace": item.get("retrieval_trace") or {},
        "filter_basis": item.get("filter_basis"),
        "is_primary": item.get("is_primary"),
        "is_augmented": item.get("is_augmented"),
        "variant_of": item.get("variant_of"),
        "created_at": item.get("created_at"),
    }
    return payload


def upsert_qa_debug_items(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[tuple[Any, ...]] = []
    now = int(time.time())
    for item in items or []:
        if not isinstance(item, dict):
            continue
        qa_id = str(item.get("id") or "").strip()
        if not qa_id:
            continue
        payload = _build_debug_payload(item)
        rows.append(
            (
                qa_id,
                str(item.get("task_id") or "").strip() or None,
                str(item.get("original_filename") or "").strip() or None,
                str(payload.get("source_chunk_id") or "").strip() or None,
                int(payload.get("source_chunk_index") or 0) or None,
                str(payload.get("source_chunk_title_path") or "").strip() or None,
                str(item.get("question") or "").strip() or None,
                str(item.get("answer") or "").strip() or None,
                int(item.get("created_at") or now),
                now,
                _json_dumps(payload),
            )
        )
    if not rows:
        return {"success": True, "stored_count": 0}

    init_db()
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO qa_debug_payloads (
                qa_id,
                task_id,
                original_filename,
                source_chunk_id,
                source_chunk_index,
                source_chunk_title_path,
                question,
                answer,
                created_at,
                updated_at,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(qa_id) DO UPDATE SET
                task_id = excluded.task_id,
                original_filename = excluded.original_filename,
                source_chunk_id = excluded.source_chunk_id,
                source_chunk_index = excluded.source_chunk_index,
                source_chunk_title_path = excluded.source_chunk_title_path,
                question = excluded.question,
                answer = excluded.answer,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                payload_json = excluded.payload_json
            """,
            rows,
        )
        conn.commit()
    return {"success": True, "stored_count": len(rows)}


def get_debug_map(qa_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    id_list = [str(item) for item in qa_ids if str(item or "").strip()]
    if not id_list:
        return {}
    init_db()
    result: Dict[str, Dict[str, Any]] = {}
    with _connect() as conn:
        for chunk in _chunked(id_list, _SQLITE_MAX_VARS):
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT qa_id, payload_json FROM qa_debug_payloads "
                f"WHERE qa_id IN ({placeholders})"
            )
            for row in conn.execute(sql, chunk):
                qa_id = str(row["qa_id"])
                result[qa_id] = _json_loads(row["payload_json"])
    return result


def delete_debug_entries(qa_ids: Iterable[str]) -> int:
    id_list = [str(item) for item in qa_ids if str(item or "").strip()]
    if not id_list:
        return 0
    init_db()
    removed = 0
    with _connect() as conn:
        for chunk in _chunked(id_list, _SQLITE_MAX_VARS):
            placeholders = ",".join("?" for _ in chunk)
            cur = conn.execute(
                f"DELETE FROM qa_debug_payloads WHERE qa_id IN ({placeholders})",
                chunk,
            )
            removed += int(cur.rowcount or 0)
        conn.commit()
    return removed


__all__ = ["delete_debug_entries", "get_debug_map", "init_db", "upsert_qa_debug_items"]
