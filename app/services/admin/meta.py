# 文件作用：维护问答条目的管理元数据存储与查询。
# 关联说明：为 qa_query 和 qa_write 提供问答管理元数据读写。

import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

from app.core.config import CONFIG

_SQLITE_MAX_VARS = 900


@dataclass(frozen=True)
class AdminMeta:
    id: str
    is_active: bool = True
    review_status: Optional[str] = None
    review_note: Optional[str] = None
    updated_at: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _get_db_path() -> str:
    configured = os.environ.get("ADMIN_META_DB_PATH")
    if configured:
        return configured
    outputs_dir = str(CONFIG["outputs_dir"])
    return os.path.join(outputs_dir, "admin_meta.sqlite3")


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
            CREATE TABLE IF NOT EXISTS admin_meta (
                id TEXT PRIMARY KEY,
                is_active INTEGER NOT NULL DEFAULT 1,
                review_status TEXT,
                review_note TEXT,
                updated_at INTEGER
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_meta_is_active ON admin_meta(is_active)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_meta_review_status ON admin_meta(review_status)"
        )
        conn.commit()

def _chunked(values: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [values]
    return [values[i : i + size] for i in range(0, len(values), size)]


def get_meta_map(ids: Iterable[str]) -> Dict[str, AdminMeta]:
    id_list = [str(i) for i in ids if i]
    if not id_list:
        return {}
    init_db()
    out: Dict[str, AdminMeta] = {}
    with _connect() as conn:
        for chunk in _chunked(id_list, _SQLITE_MAX_VARS):
            placeholders = ",".join("?" for _ in chunk)
            sql = (
                "SELECT id, is_active, review_status, review_note, updated_at "
                f"FROM admin_meta WHERE id IN ({placeholders})"
            )
            for row in conn.execute(sql, chunk):
                out[str(row["id"])] = AdminMeta(
                    id=str(row["id"]),
                    is_active=bool(row["is_active"]),
                    review_status=row["review_status"],
                    review_note=row["review_note"],
                    updated_at=int(row["updated_at"])
                    if row["updated_at"] is not None
                    else None,
                )
    return out


def upsert_meta(
    qa_id: str,
    *,
    is_active: Optional[bool] = None,
    review_status: Optional[str] = None,
    review_note: Optional[str] = None,
) -> AdminMeta:
    if not qa_id:
        raise ValueError("qa_id is required")
    init_db()
    now = int(time.time())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO admin_meta (id, is_active, review_status, review_note, updated_at)
            VALUES (?, COALESCE(?, 1), ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                is_active = COALESCE(excluded.is_active, admin_meta.is_active),
                review_status = COALESCE(excluded.review_status, admin_meta.review_status),
                review_note = COALESCE(excluded.review_note, admin_meta.review_note),
                updated_at = excluded.updated_at
            """,
            (
                str(qa_id),
                None if is_active is None else (1 if is_active else 0),
                review_status,
                review_note,
                now,
            ),
        )
        conn.commit()
    return get_or_default(qa_id)


def batch_upsert(
    ids: Iterable[str],
    *,
    is_active: Optional[bool] = None,
    review_status: Optional[str] = None,
    review_note: Optional[str] = None,
) -> Dict[str, AdminMeta]:
    id_list = [str(i) for i in ids if i]
    if not id_list:
        return {}
    init_db()
    now = int(time.time())
    with _connect() as conn:
        for qa_id in id_list:
            conn.execute(
                """
                INSERT INTO admin_meta (id, is_active, review_status, review_note, updated_at)
                VALUES (?, COALESCE(?, 1), ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    is_active = COALESCE(excluded.is_active, admin_meta.is_active),
                    review_status = COALESCE(excluded.review_status, admin_meta.review_status),
                    review_note = COALESCE(excluded.review_note, admin_meta.review_note),
                    updated_at = excluded.updated_at
                """,
                (
                    qa_id,
                    None if is_active is None else (1 if is_active else 0),
                    review_status,
                    review_note,
                    now,
                ),
            )
        conn.commit()
    return {qa_id: get_or_default(qa_id) for qa_id in id_list}


def delete_meta(ids: Iterable[str]) -> int:
    id_list = [str(i) for i in ids if i]
    if not id_list:
        return 0
    init_db()
    removed = 0
    with _connect() as conn:
        for chunk in _chunked(id_list, _SQLITE_MAX_VARS):
            placeholders = ",".join("?" for _ in chunk)
            cur = conn.execute(
                f"DELETE FROM admin_meta WHERE id IN ({placeholders})", chunk
            )
            removed += int(cur.rowcount or 0)
        conn.commit()
    return removed


def list_ids(
    *,
    is_active: Optional[bool] = None,
    review_status: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> List[str]:
    """
    List QA ids that have admin_meta rows.

    Note: records without admin_meta row are implicitly treated as `is_active=True`
    and `review_status=None`.
    """
    init_db()
    where_parts: List[str] = []
    params: List[object] = []
    if is_active is not None:
        where_parts.append("is_active = ?")
        params.append(1 if is_active else 0)
    if review_status is not None:
        where_parts.append("review_status = ?")
        params.append(str(review_status))
    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    sql = (
        "SELECT id FROM admin_meta"
        f"{where_sql} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([int(limit), int(offset)])
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [str(r["id"]) for r in rows]


def count_ids(*, is_active: Optional[bool] = None, review_status: Optional[str] = None) -> int:
    init_db()
    where_parts: List[str] = []
    params: List[object] = []
    if is_active is not None:
        where_parts.append("is_active = ?")
        params.append(1 if is_active else 0)
    if review_status is not None:
        where_parts.append("review_status = ?")
        params.append(str(review_status))
    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    sql = f"SELECT COUNT(1) AS c FROM admin_meta{where_sql}"
    with _connect() as conn:
        row = conn.execute(sql, params).fetchone()
    return int(row["c"] if row and row["c"] is not None else 0)


def get_or_default(qa_id: str) -> AdminMeta:
    meta = get_meta_map([qa_id]).get(str(qa_id))
    if meta:
        return meta
    return AdminMeta(id=str(qa_id), is_active=True, review_status=None, review_note=None, updated_at=None)


class AdminMetaStore:
    """Facade object for admin metadata storage and query state."""

    def init_db(self) -> None:
        init_db()

    def get_meta_map(self, ids: Iterable[str]) -> Dict[str, AdminMeta]:
        return get_meta_map(ids)

    def upsert_meta(
        self,
        qa_id: str,
        *,
        is_active: Optional[bool] = None,
        review_status: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> AdminMeta:
        return upsert_meta(
            qa_id,
            is_active=is_active,
            review_status=review_status,
            review_note=review_note,
        )

    def batch_upsert(
        self,
        ids: Iterable[str],
        *,
        is_active: Optional[bool] = None,
        review_status: Optional[str] = None,
        review_note: Optional[str] = None,
    ) -> Dict[str, AdminMeta]:
        return batch_upsert(
            ids,
            is_active=is_active,
            review_status=review_status,
            review_note=review_note,
        )

    def delete_meta(self, ids: Iterable[str]) -> int:
        return delete_meta(ids)

    def list_ids(
        self,
        *,
        is_active: Optional[bool] = None,
        review_status: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[str]:
        return list_ids(
            is_active=is_active,
            review_status=review_status,
            limit=limit,
            offset=offset,
        )

    def count_ids(
        self,
        *,
        is_active: Optional[bool] = None,
        review_status: Optional[str] = None,
    ) -> int:
        return count_ids(is_active=is_active, review_status=review_status)

    def get_or_default(self, qa_id: str) -> AdminMeta:
        return get_or_default(qa_id)


ADMIN_META_STORE = AdminMetaStore()


__all__ = [
    "AdminMeta",
    "AdminMetaStore",
    "ADMIN_META_STORE",
    "batch_upsert",
    "count_ids",
    "delete_meta",
    "get_meta_map",
    "get_or_default",
    "init_db",
    "list_ids",
    "upsert_meta",
]
