# 文件作用：将 DuReader2 数据转换为三元组 JSONL。
# 关联说明：与 convert_dureader_robust_to_triples 并列，面向 DuReader2 数据源。

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将 DuReader 2.0 raw JSON（每行一个样本）转换为本项目 triples JSONL："
            "dataset/split/id/context/question/answer。"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="输入 JSONL（raw）路径，如 qa/dataset/dureader_2_0/raw/devset/search.dev.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 triples JSONL 路径，如 qa/dataset/dureader_2_0/dureader2_search_dev_triples.jsonl",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="写入 split 字段（默认 dev）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dureader2",
        help="写入 dataset 字段前缀（默认 dureader2；实际会附加 subset）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="search",
        choices=["search", "zhidao"],
        help="子集名称（默认 search）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多导出多少条（0 表示不限制）",
    )
    parser.add_argument(
        "--docs-topk",
        type=int,
        default=3,
        help="最多使用多少篇文档拼接 context（默认 3）",
    )
    parser.add_argument(
        "--paras-topk",
        type=int,
        default=2,
        help="每篇文档最多使用多少段落拼接 context（默认 2）",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=2000,
        help="context 最大字符数，超过截断（默认 2000）",
    )
    parser.add_argument(
        "--answer-strategy",
        choices=["shortest", "first", "join"],
        default="shortest",
        help="参考答案选择策略：shortest=取最短答案（默认）；first=取 answers[0]；join=用分号拼接前 3 个答案",
    )
    return parser.parse_args()


def _iter_raw(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _pick_answer(sample: Dict[str, Any], strategy: str) -> str:
    answers = [a.strip() for a in (sample.get("answers") or []) if isinstance(a, str) and a.strip()]
    if not answers:
        return ""
    if strategy == "first":
        return answers[0]
    if strategy == "join":
        return "；".join(answers[:3])
    # shortest
    return sorted(answers, key=len)[0]


def _flatten_entity_answers(sample: Dict[str, Any], limit: int = 50) -> List[str]:
    """
    entity_answers 常见形式是 List[List[str]]，用于实体类问题的候选答案。
    这里仅作为“找证据段落”的辅助，不强行替代 answers。
    """
    out: List[str] = []
    ea = sample.get("entity_answers")
    if not isinstance(ea, list):
        return out
    for group in ea:
        if isinstance(group, list):
            for x in group:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
                    if len(out) >= limit:
                        return out
    return out


def _select_documents(sample: Dict[str, Any], docs_topk: int) -> List[Dict[str, Any]]:
    docs = [d for d in (sample.get("documents") or []) if isinstance(d, dict)]
    if not docs:
        return []
    selected = [d for d in docs if d.get("is_selected") is True]
    if selected:
        docs = selected
    return docs[: max(1, docs_topk)]


def _build_context(
    sample: Dict[str, Any],
    answer_hint: str,
    docs_topk: int,
    paras_topk: int,
    max_chars: int,
) -> str:
    """
    用“少量文档 + 少量段落”的方式构造 context，避免超长。
    优先选择包含答案（或实体候选答案）的段落作为证据段落。
    """
    docs = _select_documents(sample, docs_topk=docs_topk)
    if not docs:
        return ""

    answer_candidates: List[str] = []
    if answer_hint:
        answer_candidates.append(answer_hint)
        # 也尝试拆成更短的片段以便命中（非常长的答案可能是多项列表）
        if "、" in answer_hint:
            parts = [p.strip() for p in answer_hint.split("、") if p.strip()]
            answer_candidates.extend(parts[:10])
    answer_candidates.extend(_flatten_entity_answers(sample, limit=50))

    paras: List[str] = []
    total = 0

    def _add_para(p: str) -> None:
        nonlocal total
        if not p:
            return
        p = p.strip()
        if not p:
            return
        if total >= max_chars:
            return
        remaining = max_chars - total
        if len(p) > remaining:
            p = p[:remaining]
        paras.append(p)
        total += len(p) + 1

    for doc in docs:
        doc_paras = [p for p in (doc.get("paragraphs") or []) if isinstance(p, str) and p.strip()]
        if not doc_paras:
            continue

        matched: List[str] = []
        if answer_candidates:
            for p in doc_paras:
                if any(a and a in p for a in answer_candidates):
                    matched.append(p)

        used = 0
        for p in matched:
            _add_para(p)
            used += 1
            if used >= paras_topk:
                break
            if total >= max_chars:
                break

        if used < paras_topk and total < max_chars:
            for p in doc_paras:
                if p in matched:
                    continue
                _add_para(p)
                used += 1
                if used >= paras_topk or total >= max_chars:
                    break

        if total >= max_chars:
            break

    return "\n".join(paras).strip()


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.input):
        raise SystemExit(f"输入文件不存在: {args.input}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    dataset_name = f"{args.dataset}_{args.subset}"
    written = 0
    skipped = 0

    with open(args.output, "w", encoding="utf-8", newline="\n") as out_f:
        for sample in _iter_raw(args.input):
            q = sample.get("question")
            q = str(q).strip() if isinstance(q, str) else ""
            if not q:
                skipped += 1
                continue

            answer = _pick_answer(sample, args.answer_strategy)
            if not answer:
                skipped += 1
                continue

            context = _build_context(
                sample,
                answer_hint=answer,
                docs_topk=max(1, args.docs_topk),
                paras_topk=max(1, args.paras_topk),
                max_chars=max(200, args.max_context_chars),
            )
            if not context:
                skipped += 1
                continue

            qid = sample.get("question_id")
            qid = str(qid) if qid is not None else f"{written + 1}"
            rec = {
                "dataset": dataset_name,
                "split": args.split,
                "id": f"{args.subset}_{qid}",
                "context": context,
                "question": q,
                "answer": answer,
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(f"转换完成: 写入 {written} 条，跳过 {skipped} 条。输出: {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()

