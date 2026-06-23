# 文件作用：将 DuReader Robust 数据转换为三元组 JSONL。
# 关联说明：与内层 dureader_robust evaluate.py 配套，面向 DuReader Robust 数据源。

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 DuReader-Robust 的 train/dev.json 转换为本项目 triples JSONL（dataset/split/id/context/question/answer）。",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 JSON 文件路径（如 qa/dataset/dureader_robust/dureader_robust-data/train.json）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 JSONL 文件路径（如 qa/dataset/dureader_robust/dureader_robust_train_triples.jsonl）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="写入 split 字段（默认 train）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dureader_robust",
        help="写入 dataset 字段（默认 dureader_robust）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多导出多少条 QA（0 表示不限制）",
    )
    return parser.parse_args()


def _iter_qas(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    DuReader-Robust 的结构（常见）：
    {"data":[{"paragraphs":[{"context":str,"qas":[{"question":str,"id":str,"answers":[{"text":str,"answer_start":int}]}]}]}]}]}
    """
    for entry in data.get("data") or []:
        for para in entry.get("paragraphs") or []:
            context = para.get("context") or ""
            for qa in para.get("qas") or []:
                yield {
                    "context": context,
                    "qa": qa,
                }


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"输入文件不存在: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "data" not in raw:
        raise SystemExit("输入 JSON 结构不符合预期：需要顶层包含 data 字段。")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    written = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8", newline="\n") as out_f:
        for item in _iter_qas(raw):
            context = str(item.get("context") or "")
            qa = item.get("qa") or {}
            if not isinstance(qa, dict):
                skipped += 1
                continue

            question = str(qa.get("question") or "")
            qa_id = qa.get("id")
            qa_id = str(qa_id) if qa_id is not None else f"{written + 1}"

            answers = qa.get("answers") or []
            answer_text = ""
            if isinstance(answers, list) and answers:
                a0 = answers[0]
                if isinstance(a0, dict):
                    answer_text = str(a0.get("text") or "")
                elif isinstance(a0, str):
                    answer_text = a0

            if not (context and question and answer_text):
                skipped += 1
                continue

            rec = {
                "dataset": args.dataset,
                "split": args.split,
                "id": qa_id,
                "context": context,
                "question": question,
                "answer": answer_text,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(
        f"转换完成: 写入 {written} 条，跳过 {skipped} 条。输出: {args.output}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

