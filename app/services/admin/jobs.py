# 文件作用：维护管理端后台作业的状态、取消标记和结果记录。
# 关联说明：与 meta/qa 服务并列，负责管理端异步作业状态而不直接处理问答内容。

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import CONFIG


@dataclass
class AdminJob:
    job_id: str
    job_type: str
    status: str
    created_at: int
    params: Optional[Dict[str, Any]] = None
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    total: int = 0
    processed: int = 0
    failed: int = 0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _job_to_dict(job: AdminJob) -> Dict[str, Any]:
    return asdict(job)


def _artifact_path_exists(raw_path: Any) -> bool:
    text = str(raw_path or "").strip()
    if not text:
        return False
    candidates = [text]
    base_name = os.path.basename(text)
    if base_name:
        candidates.append(os.path.join(str(CONFIG["outputs_dir"]), base_name))
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return True
    return False


def _refresh_eval_result_state(result: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    refreshed = dict(result or {})
    changed = False
    files = refreshed.get("files") if isinstance(refreshed.get("files"), dict) else {}
    refreshed_files = dict(files)
    history_source = str(refreshed.get("history_source") or "").strip().lower()
    declared_paths = 0
    existing_paths = 0

    for key in ADMIN_JOB_STORE.result_file_keys:
        raw = str(refreshed_files.get(key) or "").strip()
        if not raw:
            continue
        declared_paths += 1
        if _artifact_path_exists(raw):
            existing_paths += 1
            continue
        refreshed_files[key] = None
        changed = True

    if refreshed_files != files:
        refreshed["files"] = refreshed_files
        changed = True

    if history_source == "milvus":
        if refreshed.get("artifacts_deleted") is not True:
            refreshed["artifacts_deleted"] = True
            changed = True
        if refreshed.get("artifacts_expire_at") is not None:
            refreshed["artifacts_expire_at"] = None
            changed = True
    elif declared_paths > 0 and existing_paths == 0:
        if refreshed.get("artifacts_deleted") is not True:
            refreshed["artifacts_deleted"] = True
            changed = True
    elif existing_paths > 0 and refreshed.get("artifacts_deleted") is True:
        refreshed["artifacts_deleted"] = False
        changed = True

    return refreshed, changed


def _refresh_job_runtime_state(job: AdminJob) -> bool:
    if not isinstance(job.result, dict):
        return False
    refreshed_result, changed = _refresh_eval_result_state(job.result)
    if changed:
        job.result = refreshed_result
    return changed


def _normalize_loaded_job(raw: Dict[str, Any]) -> Optional[AdminJob]:
    if not isinstance(raw, dict):
        return None
    job_id = str(raw.get("job_id") or "").strip()
    job_type = str(raw.get("job_type") or "").strip()
    status = str(raw.get("status") or "").strip()
    if not job_id or not job_type or not status:
        return None
    params = raw.get("params")
    result = raw.get("result")
    logs = raw.get("logs")
    job = AdminJob(
        job_id=job_id,
        job_type=job_type,
        status=status,
        created_at=int(raw.get("created_at") or int(time.time())),
        params=params if isinstance(params, dict) else None,
        started_at=int(raw["started_at"]) if raw.get("started_at") is not None else None,
        finished_at=int(raw["finished_at"]) if raw.get("finished_at") is not None else None,
        total=int(raw.get("total") or 0),
        processed=int(raw.get("processed") or 0),
        failed=int(raw.get("failed") or 0),
        message=str(raw.get("message") or ""),
        result=result if isinstance(result, dict) else None,
        error=str(raw.get("error")) if raw.get("error") is not None else None,
        logs=[str(x) for x in logs[-200:]] if isinstance(logs, list) else [],
    )
    if job.status in {"queued", "running"}:
        job.status = "failed"
        if not job.message:
            job.message = "interrupted by process restart"
        if not job.error:
            job.error = "job interrupted by process restart"
        if job.finished_at is None:
            job.finished_at = int(time.time())
    return job


def _persist_jobs_locked() -> None:
    os.makedirs(os.path.dirname(ADMIN_JOB_STORE.job_store_path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": int(time.time()),
        "jobs": [_job_to_dict(job) for job in sorted(ADMIN_JOB_STORE.jobs.values(), key=lambda item: item.created_at)],
    }
    tmp_path = f"{ADMIN_JOB_STORE.job_store_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, ADMIN_JOB_STORE.job_store_path)


def _load_jobs_from_disk() -> None:
    if not os.path.exists(ADMIN_JOB_STORE.job_store_path):
        return
    try:
        with open(ADMIN_JOB_STORE.job_store_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return
    jobs_raw = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs_raw, list):
        return
    loaded: Dict[str, AdminJob] = {}
    for item in jobs_raw:
        job = _normalize_loaded_job(item)
        if job is None:
            continue
        loaded[job.job_id] = job
    with ADMIN_JOB_STORE.store_lock:
        ADMIN_JOB_STORE.jobs.clear()
        ADMIN_JOB_STORE.jobs.update(loaded)
        if loaded:
            _persist_jobs_locked()


def _backfill_eval_jobs_locked() -> bool:
    outputs_dir = str(CONFIG["outputs_dir"])
    if not os.path.isdir(outputs_dir):
        return False
    changed = False
    for name in os.listdir(outputs_dir):
        if not name.startswith("eval_job_") or not name.endswith("_summary.json"):
            continue
        path = os.path.join(outputs_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id or job_id in ADMIN_JOB_STORE.jobs:
            continue
        finished_at = int(os.path.getmtime(path))
        total_seconds = 0
        timing = payload.get("timing")
        if isinstance(timing, dict):
            try:
                total_seconds = max(0, int(float(timing.get("total_seconds") or 0)))
            except Exception:
                total_seconds = 0
        started_at = max(0, finished_at - total_seconds) if total_seconds > 0 else None
        counts = payload.get("counts")
        total = int(counts.get("total") or 0) if isinstance(counts, dict) else 0
        params: Dict[str, Any] = {}
        if payload.get("dataset_name") is not None:
            params["dataset_name"] = str(payload.get("dataset_name") or "")
        mapping = payload.get("mapping")
        if isinstance(mapping, dict):
            params.update(mapping)
        ADMIN_JOB_STORE.jobs[job_id] = AdminJob(
            job_id=job_id,
            job_type="eval",
            status="succeeded",
            created_at=started_at or finished_at,
            params=params or None,
            started_at=started_at,
            finished_at=finished_at,
            total=total,
            processed=total,
            failed=0,
            message="done",
            result=payload,
            error=None,
            logs=[],
        )
        changed = True
    return changed


def _initialize_job_store() -> None:
    _load_jobs_from_disk()
    with ADMIN_JOB_STORE.store_lock:
        if _backfill_eval_jobs_locked():
            _persist_jobs_locked()


def create_job(
    job_type: str,
    *,
    total: int = 0,
    message: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> AdminJob:
    with ADMIN_JOB_STORE.store_lock:
        job_id = uuid.uuid4().hex
        job = AdminJob(
            job_id=job_id,
            job_type=str(job_type),
            status="queued",
            created_at=int(time.time()),
            total=int(total or 0),
            message=str(message or ""),
            params=params,
        )
        ADMIN_JOB_STORE.jobs[job_id] = job
        _persist_jobs_locked()
        return job


def get_job(job_id: str) -> Optional[AdminJob]:
    with ADMIN_JOB_STORE.store_lock:
        job = ADMIN_JOB_STORE.jobs.get(str(job_id))
        if not job:
            return None
        if _refresh_job_runtime_state(job):
            _persist_jobs_locked()
        return job


def list_jobs(limit: int = 50) -> List[AdminJob]:
    with ADMIN_JOB_STORE.store_lock:
        items = list(ADMIN_JOB_STORE.jobs.values())
        changed = False
        for job in items:
            if _refresh_job_runtime_state(job):
                changed = True
        if changed:
            _persist_jobs_locked()
        items.sort(key=lambda j: j.created_at, reverse=True)
        return items[: max(1, int(limit))]


def set_task(job_id: str, task: asyncio.Task) -> None:
    with ADMIN_JOB_STORE.store_lock:
        ADMIN_JOB_STORE.tasks[str(job_id)] = task


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    processed: Optional[int] = None,
    failed: Optional[int] = None,
    total: Optional[int] = None,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    append_log: Optional[str] = None,
) -> Optional[AdminJob]:
    with ADMIN_JOB_STORE.store_lock:
        job = ADMIN_JOB_STORE.jobs.get(str(job_id))
        if not job:
            return None
        if status is not None:
            job.status = str(status)
        if message is not None:
            job.message = str(message)
        if processed is not None:
            job.processed = int(processed)
        if failed is not None:
            job.failed = int(failed)
        if total is not None:
            job.total = int(total)
        if result is not None:
            job.result = result
        if error is not None:
            job.error = str(error)
        if append_log:
            job.logs.append(str(append_log))
            if len(job.logs) > 200:
                job.logs = job.logs[-200:]
        _persist_jobs_locked()
        return job


def start_job(job_id: str, *, total: Optional[int] = None, message: Optional[str] = None) -> Optional[AdminJob]:
    with ADMIN_JOB_STORE.store_lock:
        job = ADMIN_JOB_STORE.jobs.get(str(job_id))
        if not job:
            return None
        job.status = "running"
        if message is not None:
            job.message = str(message)
        if total is not None:
            job.total = int(total)
        if job.started_at is None:
            job.started_at = int(time.time())
        _persist_jobs_locked()
        return job


def complete_job(job_id: str, *, result: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Optional[AdminJob]:
    with ADMIN_JOB_STORE.store_lock:
        job = ADMIN_JOB_STORE.jobs.get(str(job_id))
        if not job:
            return None
        job.status = "succeeded"
        if result is not None:
            job.result = result
        if message is not None:
            job.message = str(message)
        job.finished_at = int(time.time())
        ADMIN_JOB_STORE.tasks.pop(str(job_id), None)
        _persist_jobs_locked()
        return job


def fail_job(job_id: str, *, error: str, result: Optional[Dict[str, Any]] = None) -> Optional[AdminJob]:
    with ADMIN_JOB_STORE.store_lock:
        job = ADMIN_JOB_STORE.jobs.get(str(job_id))
        if not job:
            return None
        job.status = "failed"
        job.error = str(error)
        if result is not None:
            job.result = result
        job.finished_at = int(time.time())
        ADMIN_JOB_STORE.tasks.pop(str(job_id), None)
        _persist_jobs_locked()
        return job


def cancel_job(job_id: str) -> bool:
    with ADMIN_JOB_STORE.store_lock:
        job_id = str(job_id)
        job = ADMIN_JOB_STORE.jobs.get(job_id)
        if not job:
            return False
        task = ADMIN_JOB_STORE.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        job.status = "canceled"
        job.message = "canceled by user"
        job.finished_at = int(time.time())
        ADMIN_JOB_STORE.tasks.pop(job_id, None)
        _persist_jobs_locked()
        return True


def delete_job(job_id: str) -> bool:
    with ADMIN_JOB_STORE.store_lock:
        normalized = str(job_id or "").strip()
        if not normalized or normalized not in ADMIN_JOB_STORE.jobs:
            return False
        ADMIN_JOB_STORE.jobs.pop(normalized, None)
        ADMIN_JOB_STORE.tasks.pop(normalized, None)
        _persist_jobs_locked()
        return True


class AdminJobStore:
    """Facade object for admin job lifecycle state."""

    def __init__(self) -> None:
        self.jobs: Dict[str, AdminJob] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.store_lock = threading.RLock()
        self.job_store_path = os.path.join(str(CONFIG["outputs_dir"]), "admin_jobs_store.json")
        self.result_file_keys = ("scored_jsonl", "summary_json")

    def create_job(
        self,
        job_type: str,
        *,
        total: int = 0,
        message: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> AdminJob:
        return create_job(job_type, total=total, message=message, params=params)

    def get_job(self, job_id: str) -> Optional[AdminJob]:
        return get_job(job_id)

    def list_jobs(self, limit: int = 50) -> List[AdminJob]:
        return list_jobs(limit)

    def set_task(self, job_id: str, task: asyncio.Task) -> None:
        set_task(job_id, task)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        message: Optional[str] = None,
        processed: Optional[int] = None,
        failed: Optional[int] = None,
        total: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        append_log: Optional[str] = None,
    ) -> Optional[AdminJob]:
        return update_job(
            job_id,
            status=status,
            message=message,
            processed=processed,
            failed=failed,
            total=total,
            result=result,
            error=error,
            append_log=append_log,
        )

    def start_job(
        self,
        job_id: str,
        *,
        total: Optional[int] = None,
        message: Optional[str] = None,
    ) -> Optional[AdminJob]:
        return start_job(job_id, total=total, message=message)

    def complete_job(
        self,
        job_id: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> Optional[AdminJob]:
        return complete_job(job_id, result=result, message=message)

    def fail_job(
        self,
        job_id: str,
        *,
        error: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[AdminJob]:
        return fail_job(job_id, error=error, result=result)

    def cancel_job(self, job_id: str) -> bool:
        return cancel_job(job_id)

    def delete_job(self, job_id: str) -> bool:
        return delete_job(job_id)


ADMIN_JOB_STORE = AdminJobStore()
_initialize_job_store()


__all__ = [
    "AdminJob",
    "AdminJobStore",
    "ADMIN_JOB_STORE",
    "cancel_job",
    "complete_job",
    "create_job",
    "delete_job",
    "fail_job",
    "get_job",
    "list_jobs",
    "set_task",
    "start_job",
    "update_job",
]
