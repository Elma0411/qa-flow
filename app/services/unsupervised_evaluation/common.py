# 文件作用：提供无监督评估共用的分组、数值和哈希工具。
# 关联说明：被 runners、aggregation、service 共享，放置无监督评估通用工具。

from __future__ import annotations

import hashlib
import re
from typing import Any

_RE_WS = re.compile(r"\s+")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _context_group_id(text: str) -> str:
    norm = _RE_WS.sub(" ", str(text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()
    return "sha1:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()


__all__ = ["_context_group_id", "_safe_float"]
