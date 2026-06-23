# 文件作用：运行 BM25 和规则模板问答生成基线评测。
# 关联说明：复用 benchmark_synthetic_qa 的对齐和 synthetic_qa_judge 的评审，替换生成策略为 BM25/规则。

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

# 确保可以从项目根目录导入 qa / app 等包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qa.baseline_eval.common import ALIGN_THRESHOLD, ALIGN_WEIGHTS, _iter_triples, _pick_best_aligned_qa
from qa.qa_evaluation.qa_quality_evaluator import QAEvaluator
from qa.baseline_eval.judge import _load_config_from_env, build_judge_client, judge_pair

try:
    import jieba
    import jieba.posseg as pseg

    JIEBA_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖可选
    jieba = None
    pseg = None  # type: ignore
    JIEBA_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    BM25_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖可选
    BM25_AVAILABLE = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BM25/依存模板基线：仅用规则+BM25 从上下文生成问答，再复用现有嵌入对齐与 LLM 评审。",
    )
    parser.add_argument(
        "--triples-file",
        "-t",
        type=str,
        required=True,
        help="三元组 JSONL 文件（至少包含 context/question/answer 字段）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 JSONL 文件路径（写入对齐 + 评审结果）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="最多评测多少条样本（默认 200）",
    )
    parser.add_argument(
        "--qa-per-context",
        type=int,
        default=3,
        help="每个 context 生成的候选问答数（默认 3）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="并发评审线程数（默认 50）",
    )
    parser.add_argument(
        "--bm25-topk",
        type=int,
        default=20,
        help="BM25 选择的句子上限（默认 20）",
    )
    parser.add_argument(
        "--keyword-topk",
        type=int,
        default=15,
        help="用于 BM25 的关键词数量（默认 15，需 jieba）",
    )
    parser.add_argument(
        "--min-sent-len",
        type=int,
        default=8,
        help="候选句子最小长度（字符，默认 8）",
    )
    parser.add_argument(
        "--max-sent-len",
        type=int,
        default=120,
        help="候选句子最大长度（字符，默认 120，超长会截断）",
    )
    parser.add_argument(
        "--save-generated",
        type=str,
        help="可选：将生成的候选 QA 记录到指定 JSONL，便于检查基线质量",
    )
    return parser.parse_args()


def _split_sentences(text: str, min_len: int, max_len: int) -> List[str]:
    """粗分句并限制长度。"""
    parts = re.split(r"[。！？?!；;]", text or "")
    sentences: List[str] = []
    for p in parts:
        s = p.strip()
        if min_len <= len(s) <= max_len:
            sentences.append(s)
        elif len(s) > max_len:
            sentences.append(s[:max_len])
    return sentences


def _extract_keywords(text: str, topk: int) -> List[str]:
    """基于 jieba 取高频名词/动词作为 BM25 查询词。"""
    if not JIEBA_AVAILABLE or not text:
        return []
    freq: Dict[str, int] = {}
    for word, flag in pseg.cut(text):
        if len(word) < 2:
            continue
        if flag.startswith(("n", "v")):
            freq[word] = freq.get(word, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:topk]


def _build_question(sentence: str) -> str:
    """
    简易模板把陈述句转问句：
    - 定义类：“X 是...” -> “X 是什么？”
    - 时间/地点类：含“在/于/于...年” -> “何时/何地发生？”
    - 数值类：含数字 -> “数值/结果是多少？”
    - 回退：以片段作引用问。
    """
    if not sentence:
        return ""

    if "是" in sentence:
        left = sentence.split("是", 1)[0].strip()
        if 2 <= len(left) <= 24:
            return f"文中提到的{left}是什么？"

    if re.search(r"\d", sentence):
        return "文中提到的关键数值或时间是多少？"

    for marker in ("在", "于"):
        if marker in sentence:
            return f"文中描述的事件发生{marker}何时或何地？"

    snippet = sentence[:24]
    return f"文中与“{snippet}”相关的内容是什么？"


def build_bm25_rule_generator(
    bm25_topk: int,
    keyword_topk: int,
    min_len: int,
    max_len: int,
) -> Any:
    """
    返回一个函数：输入 context，输出若干 {question, answer}。
    若装有 rank_bm25 + jieba，则用关键词做 BM25 排序；否则回退长度/数字启发式。
    """

    def _generate(context: str, limit: int) -> List[Dict[str, Any]]:
        sentences = _split_sentences(context, min_len, max_len)
        if not sentences:
            return []

        ranked_indices: List[int] = list(range(len(sentences)))

        if BM25_AVAILABLE and JIEBA_AVAILABLE:
            tokens_list = [jieba.lcut(s) for s in sentences]
            keywords = _extract_keywords(context, keyword_topk)
            if not keywords:
                keywords = [w for w in jieba.lcut(context) if len(w) > 1][:keyword_topk]
            if keywords:
                bm25 = BM25Okapi(tokens_list)
                scores = bm25.get_scores(keywords)
                ranked_indices = sorted(
                    range(len(sentences)),
                    key=lambda i: scores[i],
                    reverse=True,
                )

        if not BM25_AVAILABLE:
            ranked_indices = sorted(
                range(len(sentences)),
                key=lambda i: (bool(re.search(r"\d", sentences[i])), len(sentences[i])),
                reverse=True,
            )

        candidates = []
        for idx in ranked_indices:
            candidates.append(sentences[idx])
            if len(candidates) >= max(limit * 3, bm25_topk):
                break

        qas: List[Dict[str, Any]] = []
        for sent in candidates:
            question = _build_question(sent)
            if not question:
                continue
            qas.append({"question": question, "answer": sent})
            if len(qas) >= limit:
                break
        return qas

    return _generate


def _evaluate_context_group(
    generator: Any,
    context: str,
    triples: List[Dict[str, Any]],
    qa_per_context: int,
    aligner: QAEvaluator,
    judge_client: Any,
    wall_start: float,
    save_generated_path: str | None = None,
) -> List[Dict[str, Any]]:
    """对单个 context 生成候选 QA，做对齐 + 评审。"""
    gen_start = time.perf_counter()
    synthetic_qas = generator(context, qa_per_context)
    gen_duration = time.perf_counter() - gen_start
    wall_duration = time.perf_counter() - wall_start

    if save_generated_path:
        os.makedirs(os.path.dirname(save_generated_path) or ".", exist_ok=True)
        record = {
            "context_id": triples[0].get("id") if triples else None,
            "dataset": triples[0].get("dataset") if triples else None,
            "context": context,
            "synthetic_qas": synthetic_qas,
            "ref_triples": [
                {
                    "id": t.get("id"),
                    "ref_question": t.get("question"),
                    "ref_answer": t.get("answer"),
                }
                for t in triples
            ],
        }
        with open(save_generated_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    results: List[Dict[str, Any]] = []

    if not synthetic_qas:
        for triple in triples:
            ref_question = triple.get("question", "")
            ref_answer = triple.get("answer", "")
            results.append(
                {
                    "id": triple.get("id"),
                    "dataset": triple.get("dataset"),
                    "context": context,
                    "ref_question": ref_question,
                    "ref_answer": ref_answer,
                    "gen_question": "",
                    "gen_answer": "",
                    "timing": {
                        "generation_seconds": gen_duration,
                        "wall_seconds": wall_duration,
                    },
                    "alignment": {
                        "question_similarity": 0.0,
                        "answer_similarity": 0.0,
                        "alignment_score": 0.0,
                    },
                    "scores": {
                        "correctness": {
                            "score": 0.0,
                            "reasons": "未能生成合成问答",
                        },
                        "semantic_equivalence": {
                            "score": 0.0,
                            "reasons": "未能生成合成问答",
                        },
                        "style_similarity": {
                            "score": 0.0,
                            "reasons": "未能生成合成问答",
                        },
                    },
                }
            )
        return results

    for triple in triples:
        ref_question = triple.get("question", "")
        ref_answer = triple.get("answer", "")

        best_qa, align_info = _pick_best_aligned_qa(
            aligner,
            ref_question=ref_question,
            ref_answer=ref_answer,
            synthetic_qas=synthetic_qas,
            align_weights=ALIGN_WEIGHTS,
            align_threshold=ALIGN_THRESHOLD,
        )
        if not best_qa:
            continue

        gen_question = best_qa.get("question", "")
        gen_answer = best_qa.get("answer", "")

        scores = judge_pair(
            judge_client,
            context=context,
            ref_question=ref_question,
            ref_answer=ref_answer,
            gen_question=gen_question,
            gen_answer=gen_answer,
        )

        results.append(
            {
                "id": triple.get("id"),
                "dataset": triple.get("dataset"),
                "context": context,
                "ref_question": ref_question,
                "ref_answer": ref_answer,
                "gen_question": gen_question,
                "gen_answer": gen_answer,
                "timing": {
                    "generation_seconds": gen_duration,
                    "wall_seconds": wall_duration,
                },
                "alignment": align_info,
                "scores": scores,
            }
        )

    return results


def main() -> None:
    args = _parse_args()
    cfg = _load_config_from_env()
    judge_client = build_judge_client(cfg)

    generator = build_bm25_rule_generator(
        bm25_topk=max(1, args.bm25_topk),
        keyword_topk=max(1, args.keyword_topk),
        min_len=max(1, args.min_sent_len),
        max_len=max(args.min_sent_len, args.max_sent_len),
    )

    try:
        aligner = QAEvaluator(use_local_models=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"对齐所需本地模型未就绪，请先下载 qa_evaluation 所需模型: {exc}")

    if not os.path.exists(args.triples_file):
        raise SystemExit(f"未找到三元组文件: {args.triples_file}")

    triples_iter = list(_iter_triples(args.triples_file, args.max_samples))
    if not triples_iter:
        raise SystemExit(f"未在 {args.triples_file} 中读取到有效样本")

    context_groups: Dict[str, List[Dict[str, Any]]] = {}
    for triple in triples_iter:
        ctx = triple.get("context", "") or ""
        context_groups.setdefault(ctx, []).append(triple)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = len(context_groups)
    processed = 0
    wall_start = time.perf_counter()
    judge_max_concurrency = max(
        1, int(os.environ.get("BENCH_JUDGE_MAX_CONCURRENCY", "50"))
    )
    worker_count = max(1, min(args.max_workers, judge_max_concurrency))

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_context_group,
                    generator,
                    context,
                    triples,
                    args.qa_per_context,
                    aligner,
                    judge_client,
                    time.perf_counter(),
                    args.save_generated,
                )
                for context, triples in context_groups.items()
            ]
            for fut in as_completed(futures):
                scored_list = fut.result()
                processed += 1
                print(f"[{processed}/{total}] context done, scored {len(scored_list)} samples")
                for scored in scored_list:
                    out_f.write(json.dumps(scored, ensure_ascii=False) + "\n")

    wall_elapsed = time.perf_counter() - wall_start
    print(
        f"Total wall-clock time: {wall_elapsed:.2f}s for {total} contexts "
        f"(max_workers={worker_count}, judge_cap={judge_max_concurrency})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
