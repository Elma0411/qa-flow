# 文件作用：提供无监督评价指标共用的模型运行时工具。
# 关联说明：被 unsupervised_* 指标模块复用，统一 device 选择、缓存锁和按设备释放逻辑。

from __future__ import annotations

import gc
import os
import threading
from typing import Any, Dict, Hashable, Iterable, MutableMapping, Optional


def resolve_first_existing_model_path(paths: Iterable[str]) -> str:
    candidates = tuple(str(path or "").strip() for path in paths if str(path or "").strip())
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else ""


def select_torch_device(
    device: Optional[str],
    *,
    default_device: str,
    torch_module: Any,
) -> str:
    raw = (device or default_device or "cpu").strip().lower()
    if raw in {"cuda", "gpu"}:
        if torch_module is not None and torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
    if raw.startswith("cuda:"):
        if torch_module is None or not torch_module.cuda.is_available():
            return "cpu"
        try:
            idx = int(raw.split(":", 1)[1])
        except Exception:
            return "cpu"
        return f"cuda:{idx}" if 0 <= idx < torch_module.cuda.device_count() else "cpu"
    if raw == "auto":
        if torch_module is not None and torch_module.cuda.is_available():
            return "cuda"
        return "cpu"
    return raw if raw == "cpu" else "cpu"


def get_or_create_infer_lock(
    locks: MutableMapping[Hashable, threading.Lock],
    guard: threading.Lock,
    cache_key: Hashable,
) -> threading.Lock:
    with guard:
        lock = locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            locks[cache_key] = lock
        return lock


def release_cached_models_for_device(
    model_cache: MutableMapping[Any, Any],
    infer_locks: MutableMapping[Any, threading.Lock],
    model_lock: threading.Lock,
    resolved_device: str,
    *,
    torch_module: Any,
) -> None:
    with model_lock:
        keys_to_remove = [
            key
            for key in list(model_cache.keys())
            if isinstance(key, tuple) and len(key) >= 2 and key[1] == resolved_device
        ]
        for key in keys_to_remove:
            model_cache.pop(key, None)
            infer_locks.pop(key, None)
    gc.collect()
    if torch_module is not None and resolved_device.startswith("cuda") and torch_module.cuda.is_available():
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "get_or_create_infer_lock",
    "release_cached_models_for_device",
    "resolve_first_existing_model_path",
    "select_torch_device",
]
