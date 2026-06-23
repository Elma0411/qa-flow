# 文件作用：登记、续期、过期清理和删除流水线临时产物。
# 关联说明：与 storage/pipeline_state 配合，负责产物生命周期而不生成业务结果。

from __future__ import annotations

import asyncio
import glob
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from app.core.config import CONFIG
from app.core.logger import logger

class ArtifactLifecycleService:
    """Stateful facade for artifact lifecycle orchestration."""

    LONG_LIVED_SUFFIXES = ("_status.json",)
    TEMP_PATTERNS = (
        "eval_job_*_input_*",
        "eval_job_*_scored.jsonl",
        "eval_job_*_summary.json",
        "eval_import_*_consolidated_*.json",
        "*_consolidated_*.json",
        "*_consolidated_*.csv",
        "*_evaluation_*.json",
        "*_one_step_debug_*.jsonl",
    )

    def __init__(self) -> None:
        self.registry_lock = threading.RLock()
        self.registry_path = os.path.join(
            str(CONFIG["outputs_dir"]), "artifact_lifecycle_registry.json"
        )
        self.cleanup_task: Optional[asyncio.Task] = None
        self.cleanup_stop = False
        self.long_lived_names = {
            "admin_jobs_store.json",
            "pipeline_jobs_store.json",
            "ocr_configs.json",
            "llm_configs.json",
            "admin_meta.sqlite3",
            os.path.basename(self.registry_path),
        }


def _outputs_dir() -> str:
    base = str(CONFIG["outputs_dir"])
    os.makedirs(base, exist_ok=True)
    return os.path.abspath(base)


def _now_ts() -> int:
    return int(time.time())


def _normalize_path(path: str) -> str:
    return os.path.abspath(str(path or "")).replace("\\", "/")


def _is_long_lived_path(path: str) -> bool:
    name = os.path.basename(str(path or ""))
    if name in ARTIFACT_LIFECYCLE.long_lived_names:
        return True
    if "_status_" in name and name.endswith(".json"):
        return True
    for suffix in ARTIFACT_LIFECYCLE.LONG_LIVED_SUFFIXES:
        if name.endswith(suffix):
            return True
    return False


def _load_registry_locked() -> Dict[str, Any]:
    registry_path = ARTIFACT_LIFECYCLE.registry_path
    if not os.path.exists(registry_path):
        return {"version": 1, "updated_at": _now_ts(), "entries": []}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {"version": 1, "updated_at": _now_ts(), "entries": []}
    if not isinstance(payload, dict):
        return {"version": 1, "updated_at": _now_ts(), "entries": []}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    payload["version"] = 1
    payload["updated_at"] = int(payload.get("updated_at") or _now_ts())
    payload["entries"] = [entry for entry in entries if isinstance(entry, dict)]
    return payload


def _save_registry_locked(payload: Dict[str, Any]) -> None:
    registry_path = ARTIFACT_LIFECYCLE.registry_path
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    payload = dict(payload or {})
    payload["version"] = 1
    payload["updated_at"] = _now_ts()
    tmp_path = f"{registry_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, registry_path)


def _iter_entries_by_owner(
    entries: Iterable[Dict[str, Any]],
    *,
    owner_id: str,
    owner_kind: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    owner_id_norm = str(owner_id or "").strip()
    owner_kind_norm = str(owner_kind or "").strip() or None
    for entry in entries:
        if str(entry.get("owner_id") or "").strip() != owner_id_norm:
            continue
        if owner_kind_norm and str(entry.get("owner_kind") or "").strip() != owner_kind_norm:
            continue
        yield entry


def register_temporary_artifacts(
    *,
    owner_kind: str,
    owner_id: str,
    artifact_kind: str,
    paths: Iterable[str],
    ttl_seconds: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    owner_kind_norm = str(owner_kind or "").strip()
    owner_id_norm = str(owner_id or "").strip()
    artifact_kind_norm = str(artifact_kind or "").strip()
    if not owner_kind_norm or not owner_id_norm or not artifact_kind_norm:
        return []
    normalized_paths = []
    for path in paths or []:
        norm = _normalize_path(path)
        if not norm or _is_long_lived_path(norm):
            continue
        normalized_paths.append(norm)
    if not normalized_paths:
        return []

    expire_at = _now_ts() + max(1, int(ttl_seconds or 1))
    created_at = _now_ts()
    extra = dict(metadata or {})
    with ARTIFACT_LIFECYCLE.registry_lock:
        registry = _load_registry_locked()
        entries = registry.get("entries") or []
        existing_map = {
            str(entry.get("path") or ""): entry
            for entry in entries
            if isinstance(entry, dict)
        }
        for norm in normalized_paths:
            current = existing_map.get(norm)
            base_entry = {
                "path": norm,
                "owner_kind": owner_kind_norm,
                "owner_id": owner_id_norm,
                "artifact_kind": artifact_kind_norm,
                "created_at": int(current.get("created_at") or created_at) if isinstance(current, dict) else created_at,
                "expire_at": expire_at,
                "deleted_at": None,
                "delete_reason": None,
            }
            if extra:
                base_entry["metadata"] = extra
            existing_map[norm] = base_entry
        registry["entries"] = list(existing_map.values())
        _save_registry_locked(registry)
    return normalized_paths


def get_owner_artifact_expire_at(owner_kind: str, owner_id: str) -> Optional[int]:
    with ARTIFACT_LIFECYCLE.registry_lock:
        registry = _load_registry_locked()
        expire_vals = []
        for entry in _iter_entries_by_owner(
            registry.get("entries") or [],
            owner_kind=owner_kind,
            owner_id=owner_id,
        ):
            if entry.get("deleted_at"):
                continue
            try:
                expire_vals.append(int(entry.get("expire_at") or 0))
            except Exception:
                continue
        return max(expire_vals) if expire_vals else None


def delete_artifacts_now(
    *,
    owner_kind: str,
    owner_id: str,
    artifact_kinds: Optional[Iterable[str]] = None,
    paths: Optional[Iterable[str]] = None,
    reason: str,
) -> List[str]:
    owner_kind_norm = str(owner_kind or "").strip()
    owner_id_norm = str(owner_id or "").strip()
    path_filter = {_normalize_path(path) for path in (paths or []) if str(path or "").strip()}
    kinds_filter = {str(kind).strip() for kind in (artifact_kinds or []) if str(kind).strip()}
    deleted: List[str] = []

    with ARTIFACT_LIFECYCLE.registry_lock:
        registry = _load_registry_locked()
        entries = registry.get("entries") or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("owner_id") or "").strip() != owner_id_norm:
                continue
            if owner_kind_norm and str(entry.get("owner_kind") or "").strip() != owner_kind_norm:
                continue
            if entry.get("deleted_at"):
                continue
            if kinds_filter and str(entry.get("artifact_kind") or "").strip() not in kinds_filter:
                continue
            norm = _normalize_path(entry.get("path") or "")
            if path_filter and norm not in path_filter:
                continue
            if norm and os.path.exists(norm):
                try:
                    os.remove(norm)
                    deleted.append(norm)
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    logger.warning("Delete artifact failed path=%s error=%s", norm, exc)
                    continue
            entry["deleted_at"] = _now_ts()
            entry["delete_reason"] = str(reason or "deleted")
        _save_registry_locked(registry)
    return deleted


def delete_paths_now(paths: Iterable[str], *, reason: str) -> List[str]:
    deleted: List[str] = []
    normalized = [_normalize_path(path) for path in paths or [] if str(path or "").strip()]
    if not normalized:
        return deleted
    for norm in normalized:
        if _is_long_lived_path(norm):
            continue
        try:
            if os.path.exists(norm):
                os.remove(norm)
                deleted.append(norm)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Delete path failed path=%s error=%s", norm, exc)
    with ARTIFACT_LIFECYCLE.registry_lock:
        registry = _load_registry_locked()
        changed = False
        for entry in registry.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            norm = _normalize_path(entry.get("path") or "")
            if norm not in normalized:
                continue
            if entry.get("deleted_at"):
                continue
            entry["deleted_at"] = _now_ts()
            entry["delete_reason"] = str(reason or "deleted")
            changed = True
        if changed:
            _save_registry_locked(registry)
    return deleted


def cleanup_expired_artifacts() -> List[str]:
    now = _now_ts()
    deleted: List[str] = []
    with ARTIFACT_LIFECYCLE.registry_lock:
        registry = _load_registry_locked()
        entries = registry.get("entries") or []
        changed = False
        compacted: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            norm = _normalize_path(entry.get("path") or "")
            deleted_at = entry.get("deleted_at")
            expire_at = int(entry.get("expire_at") or 0)
            if deleted_at:
                if now - int(deleted_at) > 3600:
                    changed = True
                    continue
                compacted.append(entry)
                continue
            if expire_at > 0 and expire_at <= now:
                if norm and os.path.exists(norm):
                    try:
                        os.remove(norm)
                        deleted.append(norm)
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        logger.warning("Cleanup artifact failed path=%s error=%s", norm, exc)
                        compacted.append(entry)
                        continue
                entry["deleted_at"] = now
                entry["delete_reason"] = "expired"
                changed = True
            compacted.append(entry)
        if changed or len(compacted) != len(entries):
            registry["entries"] = compacted
            _save_registry_locked(registry)
    return deleted


def _register_scan_path(path: str, *, ttl_seconds: int) -> None:
    norm = _normalize_path(path)
    if not norm or not os.path.exists(norm) or _is_long_lived_path(norm):
        return
    name = os.path.basename(norm)
    owner_kind = "orphan"
    owner_id = name
    artifact_kind = "unknown"

    if name.startswith("eval_job_") and "_input_" in name:
        owner_kind = "eval_job"
        owner_id = name.split("_input_")[0].replace("eval_job_", "", 1)
        artifact_kind = "eval_input"
    elif name.startswith("eval_job_") and name.endswith("_scored.jsonl"):
        owner_kind = "eval_job"
        owner_id = name.replace("eval_job_", "", 1).replace("_scored.jsonl", "", 1)
        artifact_kind = "eval_scored"
    elif name.startswith("eval_job_") and name.endswith("_summary.json"):
        owner_kind = "eval_job"
        owner_id = name.replace("eval_job_", "", 1).replace("_summary.json", "", 1)
        artifact_kind = "eval_summary"
    elif name.startswith("eval_import_") and "_consolidated_" in name:
        owner_kind = "eval_ingest"
        owner_id = name
        artifact_kind = "eval_ingest_consolidated"
    elif "_one_step_debug_" in name:
        owner_kind = "pipeline_task"
        owner_id = name.split("_one_step_debug_")[0]
        artifact_kind = "pipeline_debug"
    elif "_evaluation_" in name:
        owner_kind = "pipeline_task"
        owner_id = name.split("_evaluation_")[0]
        artifact_kind = "pipeline_evaluation"
    elif "_consolidated_" in name:
        owner_kind = "pipeline_task"
        owner_id = name.split("_consolidated_")[0]
        artifact_kind = "pipeline_output"

    try:
        mtime = int(os.path.getmtime(norm))
    except Exception:
        mtime = _now_ts()
    expire_at = max(mtime + max(1, int(ttl_seconds or 1)), _now_ts() + 60)
    register_temporary_artifacts(
        owner_kind=owner_kind,
        owner_id=owner_id,
        artifact_kind=artifact_kind,
        paths=[norm],
        ttl_seconds=max(1, expire_at - _now_ts()),
    )


def initialize_artifact_lifecycle(*, ttl_seconds: int = 86400) -> None:
    base = _outputs_dir()
    with ARTIFACT_LIFECYCLE.registry_lock:
        registry = _load_registry_locked()
        known_paths = {
            _normalize_path(entry.get("path") or "")
            for entry in (registry.get("entries") or [])
            if isinstance(entry, dict)
        }
    for pattern in ARTIFACT_LIFECYCLE.TEMP_PATTERNS:
        for path in glob.glob(os.path.join(base, pattern)):
            norm = _normalize_path(path)
            if norm in known_paths:
                continue
            _register_scan_path(norm, ttl_seconds=ttl_seconds)
    cleanup_expired_artifacts()


async def _cleanup_loop(interval_seconds: int) -> None:
    try:
        while not ARTIFACT_LIFECYCLE.cleanup_stop:
            try:
                removed = cleanup_expired_artifacts()
                if removed:
                    logger.info("Artifact cleanup removed %d expired files", len(removed))
            except Exception as exc:
                logger.warning("Artifact cleanup loop failed: %s", exc)
            await asyncio.sleep(max(30, int(interval_seconds or 300)))
    except asyncio.CancelledError:
        raise


def start_artifact_cleanup_loop(*, interval_seconds: int = 300) -> None:
    if ARTIFACT_LIFECYCLE.cleanup_task is not None and not ARTIFACT_LIFECYCLE.cleanup_task.done():
        return
    ARTIFACT_LIFECYCLE.cleanup_stop = False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    ARTIFACT_LIFECYCLE.cleanup_task = loop.create_task(_cleanup_loop(interval_seconds))


async def stop_artifact_cleanup_loop() -> None:
    ARTIFACT_LIFECYCLE.cleanup_stop = True
    task = ARTIFACT_LIFECYCLE.cleanup_task
    ARTIFACT_LIFECYCLE.cleanup_task = None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


ARTIFACT_LIFECYCLE = ArtifactLifecycleService()


__all__ = [
    "ARTIFACT_LIFECYCLE",
    "ArtifactLifecycleService",
    "cleanup_expired_artifacts",
    "delete_artifacts_now",
    "delete_paths_now",
    "get_owner_artifact_expire_at",
    "initialize_artifact_lifecycle",
    "register_temporary_artifacts",
    "start_artifact_cleanup_loop",
    "stop_artifact_cleanup_loop",
]
