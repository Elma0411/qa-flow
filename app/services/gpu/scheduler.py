# 文件作用：控制 GPU 作业准入、阶段标记、释放和显存清理。
# 关联说明：与 knowledge_tagging/evaluation 等模型服务配合，统一管理 GPU 作业阶段。

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from app.core.config import CONFIG
from app.core.logger import logger

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


GPU_QUEUE_LIMIT = max(0, _env_int("GPU_STAGE_QUEUE_LIMIT", 8))
GPU_STAGE_POLL_SECONDS = max(1, _env_int("GPU_STAGE_POLL_SECONDS", 2))

_LOCK = threading.RLock()
_ADMITTED: Dict[str, Dict[str, object]] = {}
_LEASES: Dict[str, "GpuLease"] = {}
_LEASE_SEQ = 0


@dataclass(frozen=True)
class GpuLease:
    lease_id: str
    job_id: str
    stage_name: str
    device: str
    gpu_index: int
    acquired_at: int


def _query_visible_gpus() -> List[Dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return []
    gpus: List[Dict[str, int]] = []
    for line in str(result.stdout or "").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            mem_free = int(parts[1])
        except Exception:
            continue
        gpus.append({"index": idx, "memory_free_mb": mem_free})
    return gpus


def scheduler_enabled() -> bool:
    return len(_query_visible_gpus()) > 0


def visible_gpu_count() -> int:
    return len(_query_visible_gpus())


def admit_gpu_job(job_id: str, *, job_type: str) -> Dict[str, object]:
    normalized = str(job_id or "").strip()
    if not normalized:
        return {"accepted": False, "reason": "missing_job_id"}
    gpus = _query_visible_gpus()
    if not gpus:
        return {"accepted": True, "queued": False, "scheduler_enabled": False, "visible_gpus": 0}
    with _LOCK:
        if normalized in _ADMITTED:
            return {"accepted": True, "queued": False, "scheduler_enabled": True, "visible_gpus": len(gpus)}
        capacity = len(gpus) + int(GPU_QUEUE_LIMIT)
        if len(_ADMITTED) >= capacity:
            return {
                "accepted": False,
                "reason": "gpu_queue_full",
                "scheduler_enabled": True,
                "visible_gpus": len(gpus),
                "queue_limit": int(GPU_QUEUE_LIMIT),
                "current_jobs": len(_ADMITTED),
            }
        _ADMITTED[normalized] = {
            "job_type": str(job_type or ""),
            "admitted_at": int(time.time()),
        }
        return {
            "accepted": True,
            "queued": len(_ADMITTED) > len(gpus),
            "scheduler_enabled": True,
            "visible_gpus": len(gpus),
            "queue_limit": int(GPU_QUEUE_LIMIT),
            "current_jobs": len(_ADMITTED),
        }


def release_gpu_job(job_id: str) -> None:
    normalized = str(job_id or "").strip()
    if not normalized:
        return
    with _LOCK:
        _ADMITTED.pop(normalized, None)
        orphan_lease_ids = [lease_id for lease_id, lease in _LEASES.items() if lease.job_id == normalized]
    for lease_id in orphan_lease_ids:
        release_gpu_stage_lease_by_id(lease_id)


def _pick_free_gpu_locked() -> Optional[Dict[str, int]]:
    gpus = _query_visible_gpus()
    busy = {lease.gpu_index for lease in _LEASES.values()}
    free = [gpu for gpu in gpus if int(gpu.get("index", -1)) not in busy]
    if not free:
        return None
    free.sort(key=lambda gpu: (int(gpu.get("memory_free_mb", 0)), int(gpu.get("index", 0))), reverse=True)
    return free[0]


def acquire_gpu_stage_lease(job_id: str, stage_name: str) -> Optional[GpuLease]:
    normalized = str(job_id or "").strip()
    stage = str(stage_name or "").strip() or "gpu_stage"
    if not normalized:
        raise RuntimeError("GPU stage lease requires job_id")
    if not scheduler_enabled():
        return None
    deadline_sleep = max(1, int(GPU_STAGE_POLL_SECONDS))
    global _LEASE_SEQ
    while True:
        with _LOCK:
            if normalized not in _ADMITTED:
                raise RuntimeError(f"GPU job not admitted: {normalized}")
            picked = _pick_free_gpu_locked()
            if picked is not None:
                _LEASE_SEQ += 1
                gpu_index = int(picked.get("index") or 0)
                lease = GpuLease(
                    lease_id=f"{normalized}:{_LEASE_SEQ}",
                    job_id=normalized,
                    stage_name=stage,
                    device=f"cuda:{gpu_index}",
                    gpu_index=gpu_index,
                    acquired_at=int(time.time()),
                )
                _LEASES[lease.lease_id] = lease
                logger.info("GPU lease acquired job=%s stage=%s device=%s", normalized, stage, lease.device)
                return lease
        time.sleep(deadline_sleep)


def release_gpu_stage_lease(lease: Optional[GpuLease]) -> None:
    if lease is None:
        return
    release_gpu_stage_lease_by_id(lease.lease_id)


def release_gpu_stage_lease_by_id(lease_id: str) -> None:
    normalized = str(lease_id or "").strip()
    if not normalized:
        return
    with _LOCK:
        lease = _LEASES.pop(normalized, None)
    if lease is not None:
        logger.info("GPU lease released job=%s stage=%s device=%s", lease.job_id, lease.stage_name, lease.device)


def clear_cuda_runtime_for_device(device: Optional[str]) -> None:
    resolved = str(device or "").strip().lower()
    if not resolved.startswith("cuda"):
        return
    if torch is None or not torch.cuda.is_available():
        return
    try:
        if ":" in resolved:
            idx = int(resolved.split(":", 1)[1])
            if 0 <= idx < torch.cuda.device_count():
                with torch.cuda.device(idx):
                    torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


@contextlib.asynccontextmanager
async def gpu_stage_async(job_id: str, stage_name: str):
    lease = await asyncio.to_thread(acquire_gpu_stage_lease, job_id, stage_name)
    try:
        yield (lease.device if lease is not None else None)
    finally:
        await asyncio.to_thread(release_gpu_stage_lease, lease)


@contextlib.contextmanager
def gpu_stage(job_id: str, stage_name: str):
    lease = acquire_gpu_stage_lease(job_id, stage_name)
    try:
        yield (lease.device if lease is not None else None)
    finally:
        release_gpu_stage_lease(lease)


def get_scheduler_snapshot() -> Dict[str, object]:
    gpus = _query_visible_gpus()
    with _LOCK:
        busy = [
            {
                "job_id": lease.job_id,
                "stage_name": lease.stage_name,
                "device": lease.device,
                "gpu_index": lease.gpu_index,
                "acquired_at": lease.acquired_at,
            }
            for lease in _LEASES.values()
        ]
        admitted_jobs = list(_ADMITTED.keys())
    return {
        "enabled": bool(gpus),
        "visible_gpus": gpus,
        "queue_limit": int(GPU_QUEUE_LIMIT),
        "admitted_jobs": admitted_jobs,
        "active_leases": busy,
    }


__all__ = [
    "GPU_QUEUE_LIMIT",
    "GPU_STAGE_POLL_SECONDS",
    "admit_gpu_job",
    "clear_cuda_runtime_for_device",
    "get_scheduler_snapshot",
    "gpu_stage",
    "gpu_stage_async",
    "release_gpu_job",
    "scheduler_enabled",
    "visible_gpu_count",
]
