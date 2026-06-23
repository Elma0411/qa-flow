"""
eval_generate_report.py

读取评测输出（JSONL/JSON），自动调用 `scripts/eval_jsonl_metrics.py` 计算指标，
并生成一个“带字段 + 指标”的结果文件（JSONL 或 JSON），同时可导出 CSV。

典型用法（PowerShell）：

1) 输入为 benchmark_synthetic_qa 的 scored JSONL（包含 context/ref_*/gen_*）：
   python scripts/eval_generate_report.py `
     --input "qa/outputs/bench_sgcc_dev_triples_scored1000.jsonl" `
     --output "qa/outputs/bench_sgcc_dev_triples_report.jsonl" `
     --csv-output "qa/outputs/bench_sgcc_dev_triples_report.csv" `
     --lang zh `
     --skip-empty

2) 指定本地 BERTScore 模型（HF 名称或本地路径）：
   python scripts/eval_generate_report.py --input "..." --output "..." --lang zh `
     --bertscore-model "runtime_assets/models/chinese_bert_wwm_ext_pytorch"

输出说明：
- 每条记录包含：context/ref_question/ref_answer/gen_question/gen_answer + 各项指标分数。
- 最后一条记录（JSONL 的最后一行 / JSON list 的最后一个元素）为平均值汇总：
  {"id":"__AVERAGE__", "metrics": {...}, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_items(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    lower = path.lower()
    if lower.endswith(".jsonl"):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
        return items

    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("results", "items", "data"):
                val = data.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)]
        raise ValueError(f"Unsupported JSON structure in {path}: expected list or dict with results/items/data list")

    raise ValueError(f"Unsupported input type (expect .jsonl or .json): {path}")


def _ensure_id(item: Dict[str, Any], idx: int, *, id_key: str) -> str:
    raw = item.get(id_key)
    if raw is None or str(raw).strip() == "":
        return f"row_{idx + 1:06d}"
    return str(raw)


def _extract_fields(
    item: Dict[str, Any],
    *,
    context_key: str,
    ref_question_key: str,
    ref_answer_key: str,
    gen_question_key: str,
    gen_answer_key: str,
) -> Tuple[str, str, str, str, str]:
    context = str(item.get(context_key, "") or "")

    # scored jsonl usually uses ref_*; triples jsonl usually uses question/answer
    ref_question = item.get(ref_question_key)
    if ref_question is None:
        ref_question = item.get("question", "")
    ref_answer = item.get(ref_answer_key)
    if ref_answer is None:
        ref_answer = item.get("answer", "")

    gen_question = str(item.get(gen_question_key, "") or "")
    gen_answer = str(item.get(gen_answer_key, "") or "")

    return (
        context,
        str(ref_question or ""),
        str(ref_answer or ""),
        gen_question,
        gen_answer,
    )


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "id",
        "context",
        "ref_question",
        "ref_answer",
        "gen_question",
        "gen_answer",
        "BERTScore_P",
        "BERTScore_R",
        "BERTScore_F1",
        "ROUGE_L_F1",
        "Token_F1",
        "BLEU",
        "EM",
    ]
    # Allow extra fields (e.g., count/skipped_empty) to be appended
    extras: List[str] = []
    for r in rows:
        for k in r.keys():
            if k in fieldnames or k == "metrics":
                continue
            if k not in extras:
                extras.append(k)
    fieldnames.extend(extras)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
            row = {
                "id": r.get("id", ""),
                "context": r.get("context", ""),
                "ref_question": r.get("ref_question", ""),
                "ref_answer": r.get("ref_answer", ""),
                "gen_question": r.get("gen_question", ""),
                "gen_answer": r.get("gen_answer", ""),
                "BERTScore_P": metrics.get("BERTScore_P", 0.0),
                "BERTScore_R": metrics.get("BERTScore_R", 0.0),
                "BERTScore_F1": metrics.get("BERTScore_F1", 0.0),
                "ROUGE_L_F1": metrics.get("ROUGE_L_F1", 0.0),
                "Token_F1": metrics.get("Token_F1", 0.0),
                "BLEU": metrics.get("BLEU", 0.0),
                "EM": metrics.get("EM", 0.0),
            }
            for k in extras:
                row[k] = r.get(k, "")
            writer.writerow(row)


def _run_eval_jsonl_metrics(
    *,
    temp_input_jsonl: str,
    temp_metrics_jsonl: str,
    lang: str,
    skip_empty: bool,
    bertscore_model: Optional[str],
    bertscore_num_layers: Optional[int],
    pred_key: str,
    ref_key: str,
) -> None:
    cmd = [
        sys.executable,
        os.path.join("scripts", "eval_jsonl_metrics.py"),
        "--input",
        temp_input_jsonl,
        "--pred_key",
        pred_key,
        "--ref_key",
        ref_key,
        "--lang",
        lang,
        "--save_per_item",
        temp_metrics_jsonl,
    ]
    if skip_empty:
        cmd.append("--skip-empty")
    if bertscore_model:
        cmd.extend(["--bertscore_model", bertscore_model])
    if bertscore_num_layers is not None:
        cmd.extend(["--bertscore_num_layers", str(int(bertscore_num_layers))])

    subprocess.run(cmd, check=True)


def _load_metrics_jsonl(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    average: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            if obj.get("id") == "__AVERAGE__":
                average = obj
                continue
            _id = str(obj.get("id", "") or "")
            if _id:
                by_id[_id] = obj
    return by_id, average


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read benchmark jsonl/json, call eval_jsonl_metrics.py, and output enriched JSON/CSV report.",
    )
    ap.add_argument("--input", "-i", required=True, help="Input .jsonl or .json file")
    ap.add_argument("--output", "-o", required=True, help="Output .jsonl or .json file")
    ap.add_argument("--csv-output", default="", help="Optional CSV output path")

    ap.add_argument("--id-key", default="id", help="ID key in input objects (default: id)")
    ap.add_argument("--context-key", default="context", help="Context key (default: context)")
    ap.add_argument("--ref-question-key", default="ref_question", help="Reference question key (default: ref_question)")
    ap.add_argument("--ref-answer-key", default="ref_answer", help="Reference answer key (default: ref_answer)")
    ap.add_argument("--gen-question-key", default="gen_question", help="Generated question key (default: gen_question)")
    ap.add_argument("--gen-answer-key", default="gen_answer", help="Generated answer key (default: gen_answer)")

    ap.add_argument("--pred-key", default="gen_answer", help="Prediction key for metrics (default: gen_answer)")
    ap.add_argument("--ref-key", default="ref_answer", help="Reference key for metrics (default: ref_answer)")
    ap.add_argument("--lang", default="zh", help="BERTScore lang, e.g., zh/en (default: zh)")
    ap.add_argument("--skip-empty", action="store_true", help="Skip items where pred/ref is empty")
    ap.add_argument("--bertscore-model", default="", help="Override BERTScore model (HF name or local path)")
    ap.add_argument("--bertscore-num-layers", type=int, default=None, help="Override BERTScore num_layers")

    args = ap.parse_args()

    items = _load_items(args.input)
    if not items:
        raise SystemExit(f"No valid items loaded from: {args.input}")

    # Prepare rows (keep input order)
    extracted: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        _id = _ensure_id(it, idx, id_key=str(args.id_key))
        context, ref_q, ref_a, gen_q, gen_a = _extract_fields(
            it,
            context_key=str(args.context_key),
            ref_question_key=str(args.ref_question_key),
            ref_answer_key=str(args.ref_answer_key),
            gen_question_key=str(args.gen_question_key),
            gen_answer_key=str(args.gen_answer_key),
        )

        row = {
            "id": _id,
            "context": context,
            "ref_question": ref_q,
            "ref_answer": ref_a,
            "gen_question": gen_q,
            "gen_answer": gen_a,
        }
        extracted.append(row)

    # Build a temp JSONL for metrics script (only needs id + pred/ref)
    with tempfile.TemporaryDirectory(prefix="eval_report_") as tmp_dir:
        tmp_input = os.path.join(tmp_dir, "eval_input.jsonl")
        tmp_metrics = os.path.join(tmp_dir, "eval_metrics.jsonl")

        with open(tmp_input, "w", encoding="utf-8") as f:
            for r in extracted:
                pred = str(r.get("gen_answer", "") or "")
                ref = str(r.get("ref_answer", "") or "")
                if args.skip_empty and (not pred.strip() or not ref.strip()):
                    continue
                obj = {
                    "id": r["id"],
                    str(args.pred_key): pred,
                    str(args.ref_key): ref,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        _run_eval_jsonl_metrics(
            temp_input_jsonl=tmp_input,
            temp_metrics_jsonl=tmp_metrics,
            lang=str(args.lang),
            skip_empty=bool(args.skip_empty),
            bertscore_model=str(args.bertscore_model).strip() or None,
            bertscore_num_layers=args.bertscore_num_layers,
            pred_key=str(args.pred_key),
            ref_key=str(args.ref_key),
        )

        metrics_by_id, avg_record = _load_metrics_jsonl(tmp_metrics)

    # Join back into enriched rows
    out_rows: List[Dict[str, Any]] = []
    for r in extracted:
        pred = str(r.get("gen_answer", "") or "")
        ref = str(r.get("ref_answer", "") or "")
        if args.skip_empty and (not pred.strip() or not ref.strip()):
            continue
        m = metrics_by_id.get(str(r["id"]))
        out = dict(r)
        out["metrics"] = (m or {}).get("metrics", {}) if isinstance(m, dict) else {}
        out_rows.append(out)

    # Append average line as the last record
    avg_metrics = (avg_record or {}).get("metrics", {}) if isinstance(avg_record, dict) else {}
    summary = {
        "id": "__AVERAGE__",
        "count": int((avg_record or {}).get("count", len(out_rows))) if isinstance(avg_record, dict) else len(out_rows),
        "skipped_empty": int((avg_record or {}).get("skipped_empty", 0)) if isinstance(avg_record, dict) else 0,
        "pred_key": str(args.pred_key),
        "ref_key": str(args.ref_key),
        "metrics": avg_metrics,
    }
    out_rows.append(summary)

    # Write JSON/JSONL
    if args.output.lower().endswith(".json"):
        _write_json(args.output, out_rows)
    else:
        _write_jsonl(args.output, out_rows)

    if str(args.csv_output).strip():
        _write_csv(str(args.csv_output).strip(), out_rows)

    print(f"[OK] Wrote: {args.output}")
    if str(args.csv_output).strip():
        print(f"[OK] Wrote: {str(args.csv_output).strip()}")


if __name__ == "__main__":  # pragma: no cover
    main()
