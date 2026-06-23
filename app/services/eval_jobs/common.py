# 文件作用：提供评测作业共用路径、状态和 JSONL 读写工具。
# 关联说明：被 dataset、run、result、service 共享，集中处理作业路径和 JSONL 状态。

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _unsupervised_scores_from_suite_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    raw = summary.get("scores") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "faithfulness": _safe_float(raw.get("faithfulness"), 0.0),
        "answerability": _safe_float(raw.get("p"), 0.0),
        "coverage_recall_soft": _safe_float(raw.get("r_soft"), 0.0),
        "coverage_self": _safe_float(raw.get("coverage_self"), 0.0),
        "coverage_score": _safe_float(raw.get("coverage_score"), 0.0),
        "unsupervised_f1": _safe_float(raw.get("f1"), 0.0),
    }


def _chunk(items: list[Dict[str, Any]], size: int) -> list[list[Dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[index : index + size] for index in range(0, len(items), size)]


def _sanitize_task_token(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "dataset"


def iter_scored_items(scored_jsonl_path: str) -> Iterable[Dict[str, Any]]:
    with open(scored_jsonl_path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if obj.get("id") == "__SUMMARY__":
                continue
            yield obj


__all__ = [
    "_chunk",
    "_safe_float",
    "_sanitize_task_token",
    "_unsupervised_scores_from_suite_summary",
    "_write_json",
    "_write_jsonl",
    "iter_scored_items",
]
