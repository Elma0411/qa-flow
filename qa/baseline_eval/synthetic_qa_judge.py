# 文件作用：兼容保留 baseline 评审 CLI 入口。
# 关联说明：真正的 LLM-as-a-judge 逻辑已下沉到 qa.baseline_eval.judge，本文件仅保留脚本调用兼容。

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from qa.baseline_eval.judge import _load_config_from_env, build_judge_client, judge_pair


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LLM-as-a-judge 对 (context,ref QA,gen QA) 进行三维打分的小工具",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 JSONL，每行包含 context, ref_question, ref_answer, gen_question, gen_answer",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 JSONL，附加 scores 字段",
    )
    return parser.parse_args()


def _iter_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def main() -> None:
    args = _parse_args()
    cfg = _load_config_from_env()
    client = build_judge_client(cfg)
    samples = _iter_jsonl(args.input)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for item in samples:
            ctx = item.get("context", "")
            ref_q = item.get("ref_question", "")
            ref_a = item.get("ref_answer", "")
            gen_q = item.get("gen_question", "")
            gen_a = item.get("gen_answer", "")
            scores = judge_pair(client, ctx, ref_q, ref_a, gen_q, gen_a, cfg)
            item["scores"] = scores
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":  # pragma: no cover
    main()
