import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 benchmark_synthetic_qa 输出的 JSONL 做快速统计与抽样查看。"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="评测结果 JSONL 路径（benchmark_synthetic_qa 的 -o 输出）。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="展示语义相似度最低的样本数量，默认 5。",
    )
    return parser.parse_args()


def _stats(arr: List[float]) -> str:
    if not arr:
        return "n/a"
    return (
        f"avg={statistics.mean(arr):.3f}, "
        f"p50={statistics.median(arr):.3f}, "
        f"max={max(arr):.3f}"
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main() -> None:
    args = parse_args()
    recs = load_jsonl(Path(args.input))
    if not recs:
        print("无记录")
        return

    correctness: List[float] = []
    semantic: List[float] = []
    style: List[float] = []
    align: List[float] = []
    gen_time: List[float] = []
    wall_time: List[float] = []

    scored: List[Tuple[float, Dict[str, Any]]] = []

    for r in recs:
        scores = r.get("scores") or {}
        for arr, key in (
            (correctness, "correctness"),
            (semantic, "semantic_equivalence"),
            (style, "style_similarity"),
        ):
            val = (scores.get(key) or {}).get("score")
            if isinstance(val, (int, float)):
                arr.append(float(val))
        align_score = (r.get("alignment") or {}).get("alignment_score")
        if isinstance(align_score, (int, float)):
            align.append(float(align_score))
        timing = r.get("timing") or {}
        if isinstance(timing.get("generation_seconds"), (int, float)):
            gen_time.append(float(timing["generation_seconds"]))
        if isinstance(timing.get("wall_seconds"), (int, float)):
            wall_time.append(float(timing["wall_seconds"]))

        se = (scores.get("semantic_equivalence") or {}).get("score")
        if isinstance(se, (int, float)):
            scored.append((float(se), r))

    scored.sort(key=lambda x: x[0])

    print(f"总条数: {len(recs)}")
    print(f"correctness: {_stats(correctness)}")
    print(f"semantic_equivalence: {_stats(semantic)}")
    print(f"style_similarity: {_stats(style)}")
    print(f"alignment_score: {_stats(align)}")
    print(f"generation_seconds (per context): {_stats(gen_time)}")
    print(f"wall_seconds (per context): {_stats(wall_time)}")

    topk = max(1, args.topk)
    if scored:
        print(f"\n语义相似度最低的 {topk} 条：")
        for se, rec in scored[:topk]:
            ref_q = rec.get("ref_question", "")
            ref_a = rec.get("ref_answer", "")
            gen_q = rec.get("gen_question", "")
            gen_a = rec.get("gen_answer", "")
            print("-" * 40)
            print(f"semantic_equivalence={se:.3f}")
            print(f"ref_q: {ref_q}")
            print(f"ref_a: {ref_a}")
            print(f"gen_q: {gen_q}")
            print(f"gen_a: {gen_a}")


if __name__ == "__main__":
    main()
