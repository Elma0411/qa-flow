"""
三元组评估脚本（JSONL）

用途：
- 对 `context + question + answer` 的三元组数据进行评估；
- 支持三种评估方式：LLM 评估 / NLP 自动指标 / 无监督套件（Faithfulness+Answerability+Coverage+F1；另可选 fluency_ppl）。

输入格式（JSONL 每行一个对象，至少包含）：
- id: str（可选；缺失会自动生成）
- context: str（证据段落；脚本会映射到 source_fact_text）
- question: str
- answer: str

输出：
- 默认输出 JSONL：每行包含原始字段 + evaluation/local/unsupervised_evaluation 等评分字段；
- 同时会写入一个汇总行：{"id":"__SUMMARY__", ...}。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import (  # noqa: E402
    AUTO_EVAL_MAX_ITEMS_PER_REQUEST,
    CONFIG,
    LLM_EVALUATION_METRICS,
)
from app.services.unsupervised_evaluation import (  # noqa: E402
    UNSUPERVISED_EVALUATION_AVAILABLE,
    execute_unsupervised_suite_blocking,
)


EvalMode = Literal["llm", "nlp", "unsupervised", "all"]
PrimaryScore = Literal["none", "llm", "nlp", "unsupervised"]


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _ensure_id(item: Dict[str, Any], idx: int) -> str:
    raw = item.get("id")
    if raw is None or str(raw).strip() == "":
        return f"row_{idx + 1:06d}"
    return str(raw)


def _coerce_triple_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        row = dict(it)
        row["id"] = _ensure_id(row, idx)
        context = row.get("context")
        if context is None:
            context = row.get("source_fact_text") or row.get("source_fact") or row.get("source")
        row["context"] = "" if context is None else str(context)
        row["question"] = "" if row.get("question") is None else str(row.get("question"))
        row["answer"] = "" if row.get("answer") is None else str(row.get("answer"))
        # normalize to API-friendly reference field for evaluators
        row["source_fact_text"] = row.get("source_fact_text") or row["context"]
        out.append(row)
    return out


def _compute_llm_average(result_row: Dict[str, Any], criteria_list: List[str]) -> float:
    ev = result_row.get("evaluation") or {}
    if not isinstance(ev, dict):
        return 0.0
    vals: List[float] = []
    for k in criteria_list:
        entry = ev.get(k)
        if isinstance(entry, dict) and isinstance(entry.get("score"), (int, float)):
            vals.append(float(entry["score"]))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _run_llm_eval(
    rows: List[Dict[str, Any]],
    *,
    criteria_list: List[str],
    max_eval_concurrency: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    from app.services.evaluation import execute_llm_evaluation_blocking  # noqa: E402

    llm_config = {
        "api_key": CONFIG.get("api_key"),
        "base_url": CONFIG.get("base_url"),
        "model": CONFIG.get("model"),
        "max_retries": CONFIG.get("max_retries"),
        "request_timeout": CONFIG.get("request_timeout"),
    }
    started = time.time()
    summary: Dict[str, Any] = {
        "method": "llm",
        "criteria": criteria_list,
        "max_eval_concurrency": max_eval_concurrency,
        "model": llm_config.get("model"),
    }
    # execute_llm_evaluation_blocking expects a list of qa dicts and will write a temp file internally.
    res = execute_llm_evaluation_blocking(
        rows,
        criteria_list,
        max_eval_concurrency=max_eval_concurrency,
        llm_config=llm_config,
    )
    if not isinstance(res, dict) or not isinstance(res.get("results"), list):
        summary["error"] = "llm evaluation returned empty"
        summary["duration_seconds"] = time.time() - started
        return {}, summary

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in res.get("results") or []:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or "").strip()
        if not rid:
            continue
        avg = _compute_llm_average(r, criteria_list)
        r["average_score"] = float(avg)
        r["evaluation_method"] = "llm"
        by_id[rid] = r

    summary["evaluated"] = len(by_id)
    summary["duration_seconds"] = time.time() - started
    return by_id, summary


def _chunk(items: List[Dict[str, Any]], n: int) -> List[List[Dict[str, Any]]]:
    if n <= 0:
        return [items]
    return [items[i : i + n] for i in range(0, len(items), n)]


def _run_nlp_eval(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    from app.services.evaluation import execute_local_evaluation_blocking  # noqa: E402

    started = time.time()
    summary: Dict[str, Any] = {
        "method": "nlp",
        "local_auto_metrics": True,
        "max_items_per_request": int(AUTO_EVAL_MAX_ITEMS_PER_REQUEST),
    }
    if not rows:
        summary["evaluated"] = 0
        summary["duration_seconds"] = time.time() - started
        return {}, summary

    by_id: Dict[str, Dict[str, Any]] = {}
    for part in _chunk(rows, int(AUTO_EVAL_MAX_ITEMS_PER_REQUEST)):
        res = execute_local_evaluation_blocking(part, use_local_models=True)
        if not isinstance(res, dict) or not isinstance(res.get("results"), list):
            continue
        for r in res.get("results") or []:
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id") or "").strip()
            if not rid:
                continue
            r["evaluation_method"] = "local"
            by_id[rid] = r

    summary["evaluated"] = len(by_id)
    summary["duration_seconds"] = time.time() - started
    return by_id, summary


def _run_unsupervised(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    started = time.time()
    if not UNSUPERVISED_EVALUATION_AVAILABLE:
        return {
            "method": "unsupervised_suite_v1",
            "available": False,
            "error": "unsupervised evaluation dependencies missing",
            "duration_seconds": time.time() - started,
        }
    # Keep per-item details (e.g. faithfulness hypothesis / answerability spans / coverage units)
    # so the JSONL can be used for debugging and manual inspection.
    summary = execute_unsupervised_suite_blocking(
        rows,
        only_primary=True,
        prune_item_details=False,
    )
    if not isinstance(summary, dict):
        summary = {"method": "unsupervised_suite_v1", "scores": {}, "error": "suite returned empty"}

    ordered: Dict[str, Any] = {"method": summary.get("method") or "unsupervised_suite_v1", "scores": summary.get("scores") or {}}
    for k, v in summary.items():
        if k in {"method", "scores"}:
            continue
        ordered[k] = v
    ordered["duration_seconds"] = time.time() - started
    return ordered


def _apply_primary_score(
    rows: List[Dict[str, Any]],
    *,
    primary: PrimaryScore,
) -> None:
    primary = str(primary or "none").strip().lower()  # type: ignore[assignment]
    if primary not in {"none", "llm", "nlp", "unsupervised"}:
        primary = "none"

    if primary == "none":
        return

    for r in rows:
        if primary == "llm":
            raw = r.get("llm_average_score")
            if isinstance(raw, (int, float, str)):
                try:
                    r["average_score"] = float(raw)
                    r["evaluation_method"] = "llm"
                except Exception:
                    continue
            continue

        if primary == "nlp":
            raw = r.get("nlp_average_score")
            if isinstance(raw, (int, float, str)):
                try:
                    r["average_score"] = float(raw)
                    r["evaluation_method"] = "local"
                except Exception:
                    continue
            continue

        ue = r.get("unsupervised_evaluation")
        scores = ue.get("scores") if isinstance(ue, dict) else {}
        raw = scores.get("unsupervised_f1") if isinstance(scores, dict) else None
        if isinstance(raw, (int, float, str)):
            try:
                r["average_score"] = float(raw)
                r["evaluation_method"] = "unsupervised_f1"
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate triples jsonl with llm/nlp/unsupervised suite.")
    ap.add_argument("--input", "-i", required=True, help="Input triples jsonl path")
    ap.add_argument("--output", "-o", required=True, help="Output jsonl path")
    ap.add_argument(
        "--skip-items",
        type=int,
        default=0,
        help="Skip first N items from input (default: 0)",
    )
    ap.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Evaluate at most N items after skipping (0 means all, default: 0)",
    )
    ap.add_argument(
        "--mode",
        choices=["llm", "nlp", "unsupervised", "all"],
        default="all",
        help="Evaluation mode (default: all)",
    )
    ap.add_argument(
        "--summary-style",
        choices=["full", "minimal"],
        default=None,
        help=(
            "Output format of the last __SUMMARY__ line. "
            "Default: minimal for --mode unsupervised, otherwise full."
        ),
    )
    ap.add_argument(
        "--primary-score",
        choices=["none", "llm", "nlp", "unsupervised"],
        default="unsupervised",
        help="Which score to write into average_score/evaluation_method for filtering (default: unsupervised)",
    )
    ap.add_argument(
        "--llm-criteria",
        default=",".join(LLM_EVALUATION_METRICS),
        help="Comma-separated criteria list for LLM eval",
    )
    ap.add_argument("--llm-max-concurrency", type=int, default=8, help="LLM eval concurrency (default: 8)")

    args = ap.parse_args()
    items = _load_jsonl(args.input)
    if not items:
        raise SystemExit(f"No valid jsonl items loaded: {args.input}")

    rows_all = _coerce_triple_rows(items)
    skip_items = max(0, int(args.skip_items or 0))
    max_items = max(0, int(args.max_items or 0))
    rows = rows_all[skip_items:]
    if max_items > 0:
        rows = rows[:max_items]
    mode: EvalMode = args.mode
    summary_style = str(args.summary_style or "").strip().lower()
    if summary_style not in {"full", "minimal"}:
        summary_style = "minimal" if mode == "unsupervised" else "full"

    summaries: Dict[str, Any] = {
        "input": args.input,
        "mode": mode,
        "total_loaded": len(rows_all),
        "skip_items": skip_items,
        "max_items": max_items,
        "total": len(rows),
    }

    # ---- LLM eval ----
    if mode in ("llm", "all"):
        criteria_list = [c.strip() for c in str(args.llm_criteria or "").split(",") if c.strip()]
        if not criteria_list:
            criteria_list = list(LLM_EVALUATION_METRICS)
        by_id, s = _run_llm_eval(
            rows,
            criteria_list=criteria_list,
            max_eval_concurrency=max(1, int(args.llm_max_concurrency or 1)),
        )
        summaries["llm"] = s
        for r in rows:
            rid = str(r.get("id") or "")
            got = by_id.get(rid)
            if not got:
                continue
            # store in a consistent shape: evaluation.llm + average_score
            r["evaluation"] = {"llm": got.get("evaluation")}
            r["llm_average_score"] = got.get("average_score")
            # Keep top-level average_score as LLM average for now; may be overridden by primary-score.
            r["average_score"] = got.get("average_score")
            r["evaluation_method"] = "llm"

    # ---- NLP auto metrics ----
    if mode in ("nlp", "all"):
        by_id, s = _run_nlp_eval(rows)
        summaries["nlp"] = s
        for r in rows:
            rid = str(r.get("id") or "")
            got = by_id.get(rid)
            if not got:
                continue
            ev = r.get("evaluation")
            if not isinstance(ev, dict):
                ev = {}
            ev["local"] = got.get("evaluation")
            r["evaluation"] = ev
            r["nlp_average_score"] = got.get("average_score")
            if mode == "nlp":
                r["average_score"] = got.get("average_score")
                r["evaluation_method"] = "local"

    # ---- Unsupervised suite ----
    if mode in ("unsupervised", "all"):
        summaries["unsupervised"] = _run_unsupervised(rows)
        # average_score will be set by primary-score selection below

    _apply_primary_score(rows, primary=args.primary_score)

    summaries["completed_at"] = time.time()
    out_rows = list(rows)

    if summary_style == "minimal":
        minimal: Dict[str, Any] = {"mode": str(mode)}
        if mode in ("unsupervised", "all") and isinstance(summaries.get("unsupervised"), dict):
            raw = summaries.get("unsupervised", {}).get("scores") or {}
            if not isinstance(raw, dict):
                raw = {}
            minimal = {
                "mode": "unsupervised",
                "scores": {
                    "faithfulness": float(raw.get("faithfulness") or 0.0),
                    "answerability": float(raw.get("p") or 0.0),
                    "coverage_recall_soft": float(raw.get("r_soft") or 0.0),
                    "coverage_self": float(raw.get("coverage_self") or 0.0),
                    "coverage_score": float(raw.get("coverage_score") or 0.0),
                    "unsupervised_f1": float(raw.get("f1") or 0.0),
                },
            }
        out_rows.append({"id": "__SUMMARY__", "summary": minimal})
    else:
        out_rows.append({"id": "__SUMMARY__", "summary": summaries})
    _write_jsonl(args.output, out_rows)
    print(f"Saved: {args.output} (rows={len(out_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
