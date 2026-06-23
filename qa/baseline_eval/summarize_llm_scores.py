# 文件作用：汇总基线评测输出中的 LLM 评分统计。
# 关联说明：读取同目录 baseline 脚本输出的 JSONL，做跨实验评分汇总。

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from app.core.runtime_paths import OUTPUTS_DIR


METRICS = ("correctness", "semantic_equivalence", "style_similarity")


@dataclass
class MetricStats:
    count: int = 0
    total: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += float(value)

    @property
    def mean(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return self.total / self.count


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield {"_parse_error": True, "_line_no": line_no}
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                yield {"_not_object": True, "_line_no": line_no}


def _extract_metric_score(record: Dict[str, Any], metric: str) -> Optional[float]:
    """
    Best-effort extraction for multiple JSONL schemas:
    1) {"scores": {"correctness": {"score": 0.9}}}
    2) {"correctness": {"score": 0.9}}
    3) {"scores": {"correctness": 0.9}}
    """
    if metric in record and isinstance(record[metric], dict) and "score" in record[metric]:
        try:
            return float(record[metric]["score"])
        except (TypeError, ValueError):
            return None

    scores = record.get("scores")
    if isinstance(scores, dict):
        block = scores.get(metric)
        if isinstance(block, dict) and "score" in block:
            try:
                return float(block["score"])
            except (TypeError, ValueError):
                return None
        if isinstance(block, (int, float)):
            return float(block)

    return None


def _summarize_file(path: str) -> Tuple[Dict[str, MetricStats], Dict[str, int]]:
    stats = {m: MetricStats() for m in METRICS}
    meta = {
        "lines_total": 0,
        "lines_json_error": 0,
        "lines_not_object": 0,
        "lines_with_any_metric": 0,
    }

    for rec in _iter_jsonl(path):
        meta["lines_total"] += 1
        if rec.get("_parse_error"):
            meta["lines_json_error"] += 1
            continue
        if rec.get("_not_object"):
            meta["lines_not_object"] += 1
            continue

        found_any = False
        for metric in METRICS:
            val = _extract_metric_score(rec, metric)
            if val is None:
                continue
            if not (0.0 <= val <= 1.0):
                continue
            stats[metric].add(val)
            found_any = True
        if found_any:
            meta["lines_with_any_metric"] += 1

    return stats, meta


def _format_mean(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _resolve_default_dir() -> str:
    for candidate in (OUTPUTS_DIR, "qa/outputs", "qa/output", "outputs"):
        if os.path.isdir(candidate):
            return candidate
    return OUTPUTS_DIR


def _display_width(text: str) -> int:
    """
    Approximate terminal display width.
    Treat East Asian Wide/Fullwidth chars as width=2, others width=1.
    """
    import unicodedata

    width = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, width: int, align: str) -> str:
    pad = width - _display_width(text)
    if pad <= 0:
        return text
    if align == "right":
        return (" " * pad) + text
    return text + (" " * pad)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "汇总输出目录下所有 JSONL 的 LLM 评审平均分（常用于基准评测输出）："
            "correctness / semantic_equivalence / style_similarity。"
        )
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=_resolve_default_dir(),
        help="包含 JSONL 文件的目录（默认优先探测 runtime_assets/outputs）。",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default="*.jsonl",
        help="文件匹配模式（默认 *.jsonl）。",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="只输出 file + 三个 mean（不包含 lines/any_metric/json_error）。",
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="输出 TSV（tab 分隔），便于脚本解析；默认输出对齐表格。",
    )
    args = parser.parse_args()

    base_dir = args.dir
    if not os.path.isdir(base_dir):
        raise SystemExit(f"目录不存在: {base_dir}")

    pattern = os.path.join(base_dir, args.glob_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"未找到匹配文件: {pattern}")

    overall = {m: MetricStats() for m in METRICS}
    if args.brief:
        columns = ("file", "correctness_mean", "semantic_mean", "style_mean")
    else:
        columns = (
            "file",
            "lines",
            "any_metric",
            "json_error",
            "correctness_mean",
            "semantic_mean",
            "style_mean",
        )

    rows = []
    for path in files:
        stats, meta = _summarize_file(path)
        for m in METRICS:
            overall[m].count += stats[m].count
            overall[m].total += stats[m].total

        rows.append(
            {
                "file": os.path.basename(path),
                "lines": str(meta["lines_total"]),
                "any_metric": str(meta["lines_with_any_metric"]),
                "json_error": str(meta["lines_json_error"]),
                "correctness_mean": _format_mean(stats["correctness"].mean),
                "semantic_mean": _format_mean(stats["semantic_equivalence"].mean),
                "style_mean": _format_mean(stats["style_similarity"].mean),
            }
        )

    overall_row = {
        "file": "OVERALL",
        "lines": "-",
        "any_metric": "-",
        "json_error": "-",
        "correctness_mean": _format_mean(overall["correctness"].mean),
        "semantic_mean": _format_mean(overall["semantic_equivalence"].mean),
        "style_mean": _format_mean(overall["style_similarity"].mean),
    }

    if args.tsv:
        print("\t".join(columns))
        for row in rows:
            print("\t".join(row[col] for col in columns))
        print("\t".join(overall_row[col] for col in columns))
        return

    all_rows = rows + [overall_row]
    widths = {}
    for col in columns:
        max_width = _display_width(col)
        for row in all_rows:
            max_width = max(max_width, _display_width(row[col]))
        widths[col] = max_width

    right_align = {
        "lines",
        "any_metric",
        "json_error",
        "correctness_mean",
        "semantic_mean",
        "style_mean",
    }
    print(
        "  ".join(
            _pad(col, widths[col], "right" if col in right_align else "left")
            for col in columns
        )
    )
    for row in all_rows:
        print(
            "  ".join(
                _pad(row[col], widths[col], "right" if col in right_align else "left")
                for col in columns
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
