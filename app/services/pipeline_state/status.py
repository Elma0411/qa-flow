# 文件作用：维护流水线任务状态文件的读写、更新和兼容清理。
# 关联说明：与 storage/artifacts 配合，记录任务状态但不直接处理产物内容。

import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import CONFIG
from app.core.time_utils import normalize_pipeline_timestamps, to_local_epoch_seconds


_STORE_LOCK = threading.RLock()
_STORE_PATH = os.path.join(
    str(CONFIG["outputs_dir"]),
    "pipeline_jobs_store.json",
)
_TASKS: Dict[str, Dict[str, Any]] = {}
_LEGACY_STATUS_RE = re.compile(r"^(?P<task_id>.+)_status_\d+\.json$")
_OUTPUT_PATH_KEYS = ("consolidated_json", "consolidated_csv", "evaluation_json", "debug_jsonl")
_OUTPUT_LIST_PATH_KEYS = ("evaluation_json_files", "debug_json_files")


def _now_ts() -> int:
    return int(time.time())


def _store_dir() -> str:
    base = os.path.dirname(_STORE_PATH) or "."
    os.makedirs(base, exist_ok=True)
    return base


def get_pipeline_store_path() -> str:
    _store_dir()
    return _STORE_PATH


def _task_sort_key(task: Dict[str, Any]) -> Tuple[int, str]:
    updated = task.get("updated_at")
    finished = task.get("finished_at")
    started = task.get("started_at")
    created = task.get("created_at")
    for raw in (updated, finished, started, created):
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return int(raw), str(task.get("task_id") or "")
        text = str(raw).strip()
        if not text:
            continue
        if text.isdigit():
            return int(text), str(task.get("task_id") or "")
        parsed = to_local_epoch_seconds(text, naive_assumption="utc")
        if parsed is not None:
            return parsed, str(task.get("task_id") or "")
    return 0, str(task.get("task_id") or "")


def _normalize_loaded_task(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    task_id = str(raw.get("task_id") or "").strip()
    if not task_id:
        return None
    task = dict(raw)
    task["task_id"] = task_id
    task.setdefault("status", "")
    task.setdefault("message", "")
    return normalize_pipeline_timestamps(task)


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


def _refresh_output_artifact_state(output: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    refreshed = dict(output or {})
    changed = False
    history_source = str(refreshed.get("history_source") or "").strip().lower()
    declared_paths = 0
    existing_paths = 0

    for key in _OUTPUT_PATH_KEYS:
        raw = str(refreshed.get(key) or "").strip()
        if not raw:
            continue
        declared_paths += 1
        if _artifact_path_exists(raw):
            existing_paths += 1
            continue
        refreshed[key] = None
        changed = True

    for key in _OUTPUT_LIST_PATH_KEYS:
        values = refreshed.get(key)
        if not isinstance(values, list):
            continue
        kept: List[str] = []
        list_changed = False
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                list_changed = True
                continue
            declared_paths += 1
            if _artifact_path_exists(text):
                existing_paths += 1
                kept.append(text)
                continue
            list_changed = True
        if list_changed or len(kept) != len(values):
            refreshed[key] = kept
            changed = True

    if history_source == "milvus":
        if existing_paths > 0:
            if refreshed.get("artifacts_deleted") is True:
                refreshed["artifacts_deleted"] = False
                changed = True
        else:
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


def _refresh_task_artifact_state(task: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    refreshed = dict(task or {})
    changed = False
    outputs = refreshed.get("outputs")
    if not isinstance(outputs, list):
        return refreshed, changed

    new_outputs: List[Dict[str, Any]] = []
    has_existing_artifacts = False
    has_milvus_outputs = False
    has_deleted_outputs = False

    for output in outputs:
        if not isinstance(output, dict):
            continue
        updated_output, output_changed = _refresh_output_artifact_state(output)
        new_outputs.append(updated_output)
        changed = changed or output_changed or updated_output != output
        if str(updated_output.get("history_source") or "").strip().lower() == "milvus":
            has_milvus_outputs = True
        if any(str(updated_output.get(key) or "").strip() for key in _OUTPUT_PATH_KEYS):
            has_existing_artifacts = True
        for key in _OUTPUT_LIST_PATH_KEYS:
            if isinstance(updated_output.get(key), list) and updated_output.get(key):
                has_existing_artifacts = True
        if updated_output.get("artifacts_deleted") is True:
            has_deleted_outputs = True

    if new_outputs != outputs:
        refreshed["outputs"] = new_outputs
        changed = True

    if has_existing_artifacts:
        if refreshed.get("artifacts_deleted") is True:
            refreshed["artifacts_deleted"] = False
            changed = True
    elif has_milvus_outputs or has_deleted_outputs:
        if refreshed.get("artifacts_deleted") is not True:
            refreshed["artifacts_deleted"] = True
            changed = True
    return refreshed, changed


def _persist_tasks_locked() -> None:
    _store_dir()
    payload = {
        "version": 1,
        "updated_at": _now_ts(),
        "tasks": sorted(_TASKS.values(), key=_task_sort_key, reverse=True),
    }
    tmp_path = f"{_STORE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, _STORE_PATH)


def _load_tasks_from_disk() -> bool:
    if not os.path.exists(_STORE_PATH):
        return False
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    tasks_raw = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks_raw, list):
        return False
    loaded: Dict[str, Dict[str, Any]] = {}
    changed = False
    for item in tasks_raw:
        task = _normalize_loaded_task(item)
        if task is None:
            continue
        if isinstance(item, dict) and task != item:
            changed = True
        loaded[str(task["task_id"])] = task
    with _STORE_LOCK:
        _TASKS.clear()
        _TASKS.update(loaded)
    return changed


def _backfill_legacy_status_files_locked() -> bool:
    outputs_dir = str(CONFIG["outputs_dir"])
    if not os.path.isdir(outputs_dir):
        return False
    latest_by_task: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    for name in os.listdir(outputs_dir):
        match = _LEGACY_STATUS_RE.match(name)
        if not match:
            continue
        path = os.path.join(outputs_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        task = _normalize_loaded_task(payload if isinstance(payload, dict) else {})
        if task is None:
            task = _normalize_loaded_task({"task_id": match.group("task_id"), **(payload or {})})
        if task is None:
            continue
        mtime = os.path.getmtime(path)
        current = latest_by_task.get(str(task["task_id"]))
        if current is None or mtime >= current[0]:
            latest_by_task[str(task["task_id"])] = (mtime, path, task)

    if not latest_by_task:
        return False

    changed = False
    imported_paths: List[str] = []
    for task_id, (_mtime, path, task) in latest_by_task.items():
        existing = _TASKS.get(task_id)
        if existing is None or _task_sort_key(task) >= _task_sort_key(existing):
            _TASKS[task_id] = task
            changed = True
        imported_paths.append(path)

    if changed:
        _persist_tasks_locked()

    for path in imported_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            continue
    return changed


def _initialize_store() -> None:
    changed = _load_tasks_from_disk()
    with _STORE_LOCK:
        backfilled = _backfill_legacy_status_files_locked()
        if (changed or backfilled) and _TASKS:
            _persist_tasks_locked()


def upsert_pipeline_task_status(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = str(task_id or "").strip()
    if not normalized:
        raise ValueError("task_id 不能为空")
    task = dict(payload or {})
    task["task_id"] = normalized
    with _STORE_LOCK:
        _TASKS[normalized] = task
        _persist_tasks_locked()
        return dict(task)


def get_pipeline_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    normalized = str(task_id or "").strip()
    with _STORE_LOCK:
        task = _TASKS.get(normalized)
        if not isinstance(task, dict):
            return None
        refreshed, changed = _refresh_task_artifact_state(task)
        if changed:
            _TASKS[normalized] = refreshed
            _persist_tasks_locked()
        return dict(refreshed)


def list_pipeline_task_statuses(limit: int = 50) -> List[Dict[str, Any]]:
    with _STORE_LOCK:
        refreshed_items: List[Dict[str, Any]] = []
        changed = False
        for task_id, task in list(_TASKS.items()):
            refreshed, item_changed = _refresh_task_artifact_state(task)
            if item_changed:
                _TASKS[task_id] = refreshed
                changed = True
            refreshed_items.append(refreshed)
        if changed:
            _persist_tasks_locked()
        items = sorted(refreshed_items, key=_task_sort_key, reverse=True)
        return [dict(item) for item in items[: max(1, int(limit))]]


def delete_pipeline_task_status(task_id: str) -> bool:
    normalized = str(task_id or "").strip()
    if not normalized:
        return False
    with _STORE_LOCK:
        if normalized not in _TASKS:
            return False
        _TASKS.pop(normalized, None)
        _persist_tasks_locked()
        return True


_initialize_store()


__all__ = [
    "delete_pipeline_task_status",
    "get_pipeline_store_path",
    "get_pipeline_task_status",
    "list_pipeline_task_statuses",
    "upsert_pipeline_task_status",
]
