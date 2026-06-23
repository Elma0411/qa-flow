# 文件作用：命令行构建知识三级标签训练数据集。
# 关联说明：调用 synth、taxonomy、dataset_io，为 train.py 准备数据。

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from qa.knowledge_tagging_3lvl.dataset_io import write_json, write_jsonl
from qa.knowledge_tagging_3lvl.synth import synthesize_examples
from qa.knowledge_tagging_3lvl.taxonomy import build_label_mappings, parse_taxonomy
from qa.knowledge_tagging_3lvl.text_cleaning import clean_for_model


def _dedup(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        text = clean_for_model(r.get("text", ""))
        if not text:
            continue
        src = r.get("source") or {}
        url = (src.get("url") or "").strip()
        if url:
            key = ("url", url)
        else:
            key = ("text", r.get("label_path", ""), text)
        if key in seen:
            continue
        seen.add(key)
        r = dict(r)
        r["text"] = text
        out.append(r)
    return out


def _stratified_split(
    rows: list[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label_path"], []).append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for _label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _group_key(r: dict) -> str:
    src = r.get("source") or {}
    url = (src.get("url") or "").strip()
    if url:
        return url
    return clean_for_model(r.get("text", ""))


def _stratified_group_split(
    rows: list[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label_path"], []).append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for _label, items in by_label.items():
        groups: dict[str, list[dict]] = {}
        for it in items:
            groups.setdefault(_group_key(it), []).append(it)
        group_items = list(groups.items())
        rng.shuffle(group_items)

        # If too few groups, fall back to per-item split.
        if len(group_items) < 3:
            rng.shuffle(items)
            n = len(items)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train.extend(items[:n_train])
            val.extend(items[n_train : n_train + n_val])
            test.extend(items[n_train + n_val :])
            continue

        n_groups = len(group_items)
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio))
        if n_train + n_val >= n_groups:
            n_train = max(1, n_groups - 2)
            n_val = 1

        for _gid, grp in group_items[:n_train]:
            train.extend(grp)
        for _gid, grp in group_items[n_train : n_train + n_val]:
            val.extend(grp)
        for _gid, grp in group_items[n_train + n_val :]:
            test.extend(grp)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to 三级知识标签.txt")
    ap.add_argument("--out", required=True, help="Output dataset dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synth-per-label", type=int, default=60)
    ap.add_argument("--crawl-openstd", action="store_true")
    ap.add_argument("--openstd-max-per-type", type=int, default=400)
    ap.add_argument("--crawl-govcn", action="store_true", help="Crawl gov.cn policy pages (title + first paragraphs)")
    ap.add_argument("--govcn-max-pages", type=int, default=20)
    ap.add_argument("--govcn-max-items", type=int, default=300)
    ap.add_argument("--govcn-max-paragraphs", type=int, default=3)
    ap.add_argument("--govcn-max-chars", type=int, default=1200)
    ap.add_argument("--min-label-confidence", type=float, default=0.0)
    ap.add_argument("--no-group-split", action="store_true", help="Disable group split by source.url")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    leaves = parse_taxonomy(args.labels)
    label_maps = build_label_mappings(leaves)
    write_json(out_dir / "labels.json", label_maps)

    rows: list[dict] = []

    rows.extend(synthesize_examples(leaves, per_label=args.synth_per_label, seed=args.seed))

    if args.crawl_openstd:
        from qa.knowledge_tagging_3lvl.openstd_gb import iter_openstd_seed_examples

        rows.extend(list(iter_openstd_seed_examples(max_per_type=args.openstd_max_per_type)))

    if args.crawl_govcn:
        from qa.knowledge_tagging_3lvl.govcn import iter_govcn_seed_examples

        rows.extend(
            list(
                iter_govcn_seed_examples(
                    max_pages=args.govcn_max_pages,
                    max_items=args.govcn_max_items,
                    max_paragraphs=args.govcn_max_paragraphs,
                    max_chars=args.govcn_max_chars,
                )
            )
        )

    rows = _dedup(rows)
    if args.min_label_confidence > 0:
        rows = [r for r in rows if float(r.get("label_confidence", 1.0)) >= float(args.min_label_confidence)]

    known = set(label_maps["leaf_to_id"].keys())
    rows = [r for r in rows if r.get("label_path") in known]

    if args.no_group_split:
        train, val, test = _stratified_split(rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    else:
        train, val, test = _stratified_group_split(rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    test_web = [r for r in test if (r.get("source") or {}).get("name") != "synthetic"]

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)
    if test_web:
        write_jsonl(out_dir / "test_web.jsonl", test_web)

    stats = {
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "test_web": len(test_web),
        "num_leaf_labels": len(known),
    }
    write_json(out_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
