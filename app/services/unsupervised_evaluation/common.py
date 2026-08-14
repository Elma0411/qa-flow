# 文件作用：提供无监督评估共用的分组、数值和哈希工具。
# 关联说明：被 runners、aggregation、service 共享，放置无监督评估通用工具。

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

_RE_WS = re.compile(r"\s+")
UNSUPERVISED_AVERAGE_METRICS = ("faithfulness", "answerability", "coverage_score")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def compute_unsupervised_average_score(scores: Any) -> float:
    """Return the item-level arithmetic mean of the three displayed metrics.

    Missing or malformed values count as zero so a record cannot pass a quality
    threshold merely because one of the required metrics was absent.  ``p`` is
    accepted only as the legacy spelling of answerability for old artifacts.
    """
    raw_scores = scores
    if isinstance(raw_scores, str):
        try:
            raw_scores = json.loads(raw_scores)
        except Exception:
            raw_scores = {}
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    values = []
    for metric in UNSUPERVISED_AVERAGE_METRICS:
        raw_value = raw_scores.get(metric)
        if metric == "answerability" and raw_value is None:
            raw_value = raw_scores.get("p")
        # The displayed metrics are unit scores.  A missing, non-numeric,
        # non-finite, or out-of-range value is invalid and must contribute
        # exactly zero; do not clamp invalid data into a passing score.
        try:
            value = float(raw_value)
        except Exception:
            value = 0.0
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            value = 0.0
        values.append(value)
    return float(sum(values) / len(values))


def _context_group_id(text: str) -> str:
    norm = _RE_WS.sub(" ", str(text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()
    return "sha1:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()


__all__ = [
    "UNSUPERVISED_AVERAGE_METRICS",
    "_context_group_id",
    "_safe_float",
    "compute_unsupervised_average_score",
]
