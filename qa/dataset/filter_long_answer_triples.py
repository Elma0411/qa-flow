# 文件作用：筛选答案长度满足阈值的三元组样本。
# 关联说明：接在 convert_* 脚本之后，用于筛出适合评测的长答案样本。

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从本项目 triples JSONL（context/question/answer）中筛选“长答案”子集。\n"
            "默认按 answer 字符数 >= 阈值筛选；可选要求答案包含分隔符（标点/换行等），"
            "以更偏向“句子/列表型”的长答案。"
        ),
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="输入 triples JSONL 路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 JSONL 路径（不会覆盖输入）")
    parser.add_argument(
        "--min-answer-len",
        type=int,
        default=30,
        help="答案最小字符数（默认 30）",
    )
    parser.add_argument(
        "--max-answer-len",
        type=int,
        default=0,
        help="答案最大字符数（0 表示不限制）",
    )
    parser.add_argument(
        "--require-sep",
        action="store_true",
        help="要求答案包含至少一个“分隔符字符”（见 --sep-chars），用于筛出更像句子/列表的长答案",
    )
    parser.add_argument(
        "--sep-chars",
        type=str,
        default="。！？；，、;,:.\n",
        help="分隔符字符集合（默认含中文/英文标点及换行）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多输出多少条（0 表示不限制）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="在筛选通过的样本中随机抽样（配合 --max-samples 使用）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    return parser.parse_args()


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
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
                yield obj


def _passes(
    record: Dict[str, Any],
    min_answer_len: int,
    max_answer_len: int,
    require_sep: bool,
    sep_chars: str,
) -> Tuple[bool, Optional[str]]:
    answer = record.get("answer")
    if answer is None:
        return False, None
    answer_text = str(answer).strip()
    if not answer_text:
        return False, None

    answer_len = len(answer_text)
    if answer_len < min_answer_len:
        return False, answer_text
    if max_answer_len and answer_len > max_answer_len:
        return False, answer_text
    if require_sep:
        sep_set = set(sep_chars)
        if not any(ch in sep_set for ch in answer_text):
            return False, answer_text
    return True, answer_text


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"输入文件不存在: {args.input}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    kept: List[Dict[str, Any]] = []
    total = 0
    skipped_invalid = 0
    skipped_filter = 0

    for rec in _iter_jsonl(args.input):
        total += 1
        if not isinstance(rec.get("context"), str) or not isinstance(rec.get("question"), str):
            skipped_invalid += 1
            continue
        ok, _ = _passes(
            rec,
            min_answer_len=max(0, int(args.min_answer_len)),
            max_answer_len=max(0, int(args.max_answer_len)),
            require_sep=bool(args.require_sep),
            sep_chars=str(args.sep_chars),
        )
        if not ok:
            skipped_filter += 1
            continue
        kept.append(rec)

    if args.shuffle and args.max_samples and len(kept) > args.max_samples:
        rng = random.Random(int(args.seed))
        rng.shuffle(kept)
        kept = kept[: args.max_samples]
    elif args.max_samples:
        kept = kept[: args.max_samples]

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        "筛选完成:"
        f" 输入 {total} 条，保留 {len(kept)} 条，"
        f"无效 {skipped_invalid} 条，未通过筛选 {skipped_filter} 条。"
        f" 输出: {args.output}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

