# 文件作用：命令行评估知识三级标签模型效果。
# 关联说明：调用 modeling/dataset_io，对 train.py 产物进行离线评估。

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from qa.knowledge_tagging_3lvl.dataset_io import read_jsonl, write_json
from qa.knowledge_tagging_3lvl.modeling import load_model


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to 三级知识标签.txt (kept for future extensions)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--by-source", action="store_true", help="Report accuracy grouped by source.name")
    ap.add_argument("--out", default="", help="Optional report path (json)")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    model, tokenizer, spec = load_model(args.model_dir, device=device)

    rows = read_jsonl(args.test)

    label_maps = json.loads(Path(args.model_dir, "labels.json").read_text(encoding="utf-8"))
    leaf_paths = [x["path"] for x in label_maps["leaf_labels"]]
    leaf_to_id = label_maps.get("leaf_to_id", {p: i for i, p in enumerate(leaf_paths)})

    correct = 0
    correct_topk = 0
    total = 0
    confs: list[float] = []
    bad = Counter()
    by_src = {}

    k = max(1, int(args.topk))
    k = min(k, len(leaf_paths))
    bs = max(1, int(args.batch_size))

    for start in range(0, len(rows), bs):
        batch_rows = rows[start : start + bs]
        texts = [r["text"] for r in batch_rows]
        truths = [r["label_path"] for r in batch_rows]
        sources = [(r.get("source") or {}).get("name") or "unknown" for r in batch_rows]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=spec.max_length,
            return_tensors="pt",
        )
        enc = {k2: v.to(device) for k2, v in enc.items()}
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            probs = torch.softmax(out["logits3"], dim=-1)
            topk = probs.topk(k=k, dim=-1)
            pred_ids = topk.indices[:, 0].tolist()
            max_confs = topk.values[:, 0].tolist()
            topk_ids = topk.indices.tolist()

        for truth, pred_id, conf, cand_ids, src in zip(truths, pred_ids, max_confs, topk_ids, sources):
            pred = leaf_paths[int(pred_id)]
            total += 1
            confs.append(float(conf))
            if pred == truth:
                correct += 1
                if args.by_source:
                    by_src.setdefault(src, {"total": 0, "correct": 0})["correct"] += 1
            else:
                bad[(truth, pred)] += 1
            if args.by_source:
                by_src.setdefault(src, {"total": 0, "correct": 0})["total"] += 1
            truth_id = leaf_to_id.get(truth)
            if truth_id is not None and int(truth_id) in cand_ids:
                correct_topk += 1

    acc = correct / max(1, total)
    acc_topk = correct_topk / max(1, total)
    report = {
        "total": total,
        "acc_leaf": acc,
        "acc_leaf_topk": acc_topk,
        "topk": k,
        "avg_conf": sum(confs) / max(1, len(confs)),
        "most_common_errors": [
            {"truth": t, "pred": p, "count": c} for (t, p), c in bad.most_common(20)
        ],
    }
    if args.by_source:
        report["by_source"] = {
            k2: {
                "total": int(v["total"]),
                "acc_leaf": (float(v["correct"]) / max(1, int(v["total"]))),
            }
            for k2, v in sorted(by_src.items(), key=lambda kv: kv[0])
        }

    # `conda run` on Windows can crash when stdout contains non-GBK characters.
    # Keep stdout ASCII-only; the report file is still written in UTF-8.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.out:
        write_json(args.out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
