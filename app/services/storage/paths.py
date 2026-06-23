# 文件作用：解析输出目录、下载路径和安全文件名。
# 关联说明：被 uploads、consolidation、routers 复用，统一输出路径和安全文件名。

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from app.core.config import (
    CONFIG,
    DEFAULT_BATCH_CONCURRENCY,
    MAX_BATCH_CONCURRENCY,
)


def sanitize_filename(name: str) -> str:
    """
    Convert filenames into safe ASCII identifiers:
    - 尝试将中文转为拼音（如缺少依赖则直接降级）
    - 非 ASCII 统一转为下划线
    - 去重连续下划线并截断过长字符串
    """
    base = os.path.splitext(os.path.basename(name))[0]
    transliterated = base
    try:
        from pypinyin import lazy_pinyin  # type: ignore
        transliterated = "".join(lazy_pinyin(base))
    except Exception:
        transliterated = base

    safe_chars: List[str] = []
    for ch in transliterated:
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_")):
            safe_chars.append(ch)
        elif ch in ("-", "_"):
            safe_chars.append("_")
        else:
            safe_chars.append("_")
    sanitized = re.sub(r"_+", "_", "".join(safe_chars)).strip("_")
    if not sanitized:
        sanitized = "file"
    return sanitized[:120]



def get_output_path(prefix: str, ext: str) -> str:
    os.makedirs(CONFIG["outputs_dir"], exist_ok=True)
    timestamp = int(time.time())
    return f"{CONFIG['outputs_dir']}/{prefix}_{timestamp}{ext}"


def resolve_batch_concurrency(value: Optional[int]) -> int:
    try:
        target = int(value) if value is not None else DEFAULT_BATCH_CONCURRENCY
    except (TypeError, ValueError):
        target = DEFAULT_BATCH_CONCURRENCY
    return max(1, min(target, MAX_BATCH_CONCURRENCY))


def write_status_file(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def cleanup_outputs(age_seconds: int = 60) -> List[str]:
    """
    删除 outputs_dir 中早于指定秒数的文件，返回已删除列表。
    默认 60 秒，方便跑完任务后批量清理。
    """
    base = CONFIG["outputs_dir"]
    if not os.path.exists(base):
        return []
    now = time.time()
    removed: List[str] = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(path)
                if now - mtime > age_seconds:
                    os.remove(path)
                    removed.append(path)
            except Exception:
                continue
    return removed

__all__ = [
    'cleanup_outputs',
    'get_output_path',
    'resolve_batch_concurrency',
    'sanitize_filename',
    'write_status_file',
]
