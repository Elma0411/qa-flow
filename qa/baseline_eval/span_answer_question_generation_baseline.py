# 文件作用：运行基于答案 span 的问题生成基线评测。
# 关联说明：复用 benchmark_synthetic_qa 的对齐和 synthetic_qa_judge 的评审，替换生成策略为 span-aware QG。

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
except Exception:  # pragma: no cover - 可选依赖
    jieba = None
    pseg = None  # type: ignore
    JIEBA_AVAILABLE = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Span+Answer-aware QG 基线：仅输入 context，通过规则/分词自动选答案 span，"
            "再用 HuggingFace text2text QG 模型（context+answer→question）生成问题，"
            "得到候选问答对后复用现有嵌入对齐与 LLM 评审。"
        ),
    )
    parser.add_argument(
        "--triples-file",
        "-t",
        type=str,
        required=True,
        help="三元组 JSONL，需包含 context/question/answer 字段。",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="输出 JSONL（对齐+评审结果）。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="最多读取多少条样本（默认 200）。",
    )
    parser.add_argument(
        "--qa-per-context",
        type=int,
        default=6,
        help="每个 context 生成多少候选问答（默认 6）。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="并发评审线程数（默认 20）。",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lmqg/mt5-small-zhquad-qg",
        help="QG 模型名称或本地路径（默认 lmqg/mt5-small-zhquad-qg）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备：cpu 或 cuda（默认 cpu）。",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam size（默认 4）。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=64,
        help="生成最大 token 数（默认 64）。",
    )
    parser.add_argument(
        "--answer-markup",
        type=str,
        default="hl",
        choices=["tags", "hl"],
        help=(
            "如何把 answer span 提供给 QG 模型："
            "hl=在 context 内用 <hl> A <hl> 标记（适配 lmqg/*-zhquad-qg 等）；"
            "tags='<answer> A <context> C'（适配 iarfmoose/t5-base-question-generator 等）。"
        ),
    )
    parser.add_argument(
        "--span-strategy",
        type=str,
        default="mix",
        choices=["mix", "regex", "jieba"],
        help="自动选答案 span 的策略：mix/regex/jieba（默认 mix）。",
    )
    parser.add_argument(
        "--max-span-candidates",
        type=int,
        default=60,
        help="最多保留多少个候选 span（默认 60）。",
    )
    parser.add_argument(
        "--min-span-len",
        type=int,
        default=2,
        help="答案 span 最小长度（字符，默认 2）。",
    )
    parser.add_argument(
        "--max-span-len",
        type=int,
        default=30,
        help="答案 span 最大长度（字符，默认 30）。",
    )
    parser.add_argument(
        "--save-generated",
        type=str,
        help="可选：将每个 context 的候选问答（含 span 候选）写入指定 JSONL 便于排查。",
    )
    return parser.parse_args()


def _build_qg_pipeline(model_name: str, device: str):
    try:
        from transformers import (  # type: ignore
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )
    except Exception as exc:  # pragma: no cover - 环境缺依赖
        raise ImportError(
            "缺少 transformers 依赖，请先安装：pip install transformers sentencepiece"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device_id = 0 if device and device.lower().startswith("cuda") else -1
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
    )


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _clean_context(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _build_qg_prompt(context: str, answer: str, answer_markup: str) -> str:
    normalized_answer = (answer or "").strip()
    if not normalized_answer:
        return context

    if answer_markup == "hl":
        idx = context.find(normalized_answer)
        if idx == -1:
            return f"{context} <hl> {normalized_answer} <hl>"
        return (
            context[:idx]
            + f"<hl> {normalized_answer} <hl>"
            + context[idx + len(normalized_answer) :]
        )

    return f"<answer> {normalized_answer} <context> {context}"


def _extract_question(text: str) -> str:
    """
    尝试从 QG 模型输出中提取问题（通常只输出 question）。
    兼容 'question: ...' / '问题: ...' / 纯问句。
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = raw.splitlines()[0].strip()

    for prefix in ("question:", "Question:", "问题:", "问题："):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :].strip()
            break

    raw = raw.strip(" \t\r\n：:，,")
    if not raw:
        return ""
    if raw.endswith(("?", "？")):
        return raw
    # 常见模型会省略问号
    return raw + "？"


_RE_SPAN_PATTERNS: List[str] = [
    r"《[^》]{2,60}》",
    r"\d{4}年\d{1,2}月\d{1,2}日",
    r"\d{4}年\d{1,2}月",
    r"\d{4}年",
    r"\d{1,2}月\d{1,2}日",
    r"\d+(?:\.\d+)?(?:万|亿)?元",
    r"\d+(?:\.\d+)?%",
    r"\d+(?:\.\d+)?(?:万|亿)?(?:人|次|个|条|份|公里|米|吨|小时|分钟|天|岁)",
    r"[\u4e00-\u9fff]{2,20}(?:公司|法院|医院|银行|基金|大学|学院|委员会|集团|有限公司)",
    r"[\u4e00-\u9fff]{1,4}[xX×]{1,3}\d*",
]


def _extract_spans_regex(context: str) -> List[str]:
    spans: List[str] = []
    if not context:
        return spans
    for pat in _RE_SPAN_PATTERNS:
        try:
            for m in re.finditer(pat, context):
                s = (m.group(0) or "").strip()
                if s:
                    spans.append(s)
        except re.error:
            continue
    return spans


def _extract_spans_jieba(context: str) -> List[str]:
    if not (JIEBA_AVAILABLE and pseg and context):
        return []

    freq: Dict[str, int] = {}
    for word, flag in pseg.cut(context):
        w = (word or "").strip()
        if len(w) < 2:
            continue
        if not flag:
            continue
        if flag.startswith(("n", "nr", "ns", "nt", "nz", "t", "m")):
            freq[w] = freq.get(w, 0) + 1

    return sorted(freq, key=freq.get, reverse=True)


def _span_score(span: str) -> int:
    if not span:
        return -999
    score = 0
    if "《" in span and "》" in span:
        score += 6
    if re.search(r"\d", span):
        score += 4
    if re.search(r"[一二三四五六七八九十百千万亿零〇两]", span):
        score += 2
    if any(u in span for u in ("年", "月", "日", "元", "%", "人", "次", "公里", "米", "吨")):
        score += 2
    if any(suf in span for suf in ("公司", "法院", "银行", "医院", "基金", "大学", "学院", "委员会", "集团")):
        score += 2
    if any(ch in span for ch in ("x", "X", "×")):
        score += 1
    if 2 <= len(span) <= 8:
        score += 1
    if len(span) > 20:
        score -= 1
    return score


def _select_answer_spans(
    context: str,
    strategy: str,
    max_candidates: int,
    min_len: int,
    max_len: int,
) -> Tuple[List[str], List[str]]:
    """
    从 context 自动提取候选答案 span。

    返回: (span_candidates, warnings)
    """
    warnings: List[str] = []
    ctx = _clean_context(context)
    if not ctx:
        return [], warnings

    spans: List[str] = []
    strategy_norm = (strategy or "mix").lower()
    if strategy_norm in {"mix", "regex"}:
        spans.extend(_extract_spans_regex(ctx))
    if strategy_norm in {"mix", "jieba"}:
        if not JIEBA_AVAILABLE:
            if strategy_norm == "jieba":
                warnings.append("span-strategy=jieba 但未安装 jieba，将回退为 regex。")
            spans.extend(_extract_spans_regex(ctx))
        else:
            spans.extend(_extract_spans_jieba(ctx))

    # 过滤 + 去重（保序）
    dedup: List[str] = []
    seen: set[str] = set()
    for s in spans:
        cand = (s or "").strip()
        if not cand:
            continue
        if "<hl>" in cand or "<answer>" in cand:
            continue
        if len(cand) > max_len:
            continue
        if len(cand) < min_len:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        dedup.append(cand)

    # 进一步按启发式打分排序（优先更像“答案”的 span）
    ranked = sorted(dedup, key=lambda s: (_span_score(s), len(s)), reverse=True)
    ranked = ranked[: max(1, int(max_candidates))]

    if not ranked:
        warnings.append("未从 context 中提取到有效 answer span（可尝试安装 jieba 或调大 max-span-candidates/max-span-len）。")
    return ranked, warnings


def _generate_qas_for_context(
    pipe: Any,
    context: str,
    qa_per_context: int,
    model_name: str,
    num_beams: int,
    max_length: int,
    answer_markup: str,
    span_strategy: str,
    max_span_candidates: int,
    min_span_len: int,
    max_span_len: int,
) -> Tuple[List[Dict[str, Any]], str | None, List[str], List[str]]:
    warnings: List[str] = []
    ctx = _clean_context(context)
    qas: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str]] = set()

    span_candidates, span_warnings = _select_answer_spans(
        ctx,
        strategy=span_strategy,
        max_candidates=max_span_candidates,
        min_len=min_span_len,
        max_len=max_span_len,
    )
    warnings.extend(span_warnings)

    if model_name.startswith("lmqg/") and model_name.endswith("-qg") and answer_markup != "hl":
        warnings.append("lmqg/*-qg 通常需要用 <hl> 标记答案；建议使用 --answer-markup hl。")

    try:
        gen_kwargs: Dict[str, Any] = {
            "num_beams": max(1, int(num_beams)),
            "num_return_sequences": 1,
            "max_new_tokens": max(8, int(max_length)),
        }
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}", warnings, span_candidates

    def _pipe_call(prompt: str) -> Any:
        try:
            return pipe(prompt, **gen_kwargs)
        except TypeError as exc:
            if "max_new_tokens" not in str(exc):
                raise
            gen_kwargs.pop("max_new_tokens", None)
            gen_kwargs["max_length"] = max(8, int(max_length))
            return pipe(prompt, **gen_kwargs)

    context_has_cjk = _has_cjk(ctx)
    max_need = max(1, int(qa_per_context))

    for span in span_candidates:
        prompt = _build_qg_prompt(context=ctx, answer=span, answer_markup=answer_markup)
        try:
            outputs = _pipe_call(prompt)
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}", warnings, span_candidates

        if not isinstance(outputs, list) or not outputs:
            continue
        item = outputs[0] if isinstance(outputs[0], dict) else None
        text = item.get("generated_text") if isinstance(item, dict) else None
        if not text:
            continue
        question = _extract_question(text)
        if not question:
            continue
        if context_has_cjk and not _has_cjk(question):
            continue
        key = (question, span)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        qas.append({"question": question, "answer": span})
        if len(qas) >= max_need:
            break

    if context_has_cjk and qas and not any(_has_cjk(qa.get("question", "")) for qa in qas):
        warnings.append("context 含中文但生成问题不含中文字符；模型/分词器可能不适配中文。")

    return qas, None, warnings, span_candidates


def _evaluate_context_group(
    pipe: Any,
    context: str,
    triples: List[Dict[str, Any]],
    qa_per_context: int,
    gen_params: Dict[str, Any],
    aligner: QAEvaluator,
    judge_client: Any,
    wall_start: float,
    save_generated_path: str | None = None,
) -> List[Dict[str, Any]]:
    gen_start = time.perf_counter()
    synthetic_qas, generation_error, generation_warnings, span_candidates = _generate_qas_for_context(
        pipe,
        context,
        qa_per_context=qa_per_context,
        model_name=gen_params["model_name"],
        num_beams=gen_params["num_beams"],
        max_length=gen_params["max_length"],
        answer_markup=gen_params["answer_markup"],
        span_strategy=gen_params["span_strategy"],
        max_span_candidates=gen_params["max_span_candidates"],
        min_span_len=gen_params["min_span_len"],
        max_span_len=gen_params["max_span_len"],
    )
    gen_duration = time.perf_counter() - gen_start
    wall_duration = time.perf_counter() - wall_start

    if save_generated_path:
        os.makedirs(os.path.dirname(save_generated_path) or ".", exist_ok=True)
        record = {
            "context_id": triples[0].get("id") if triples else None,
            "dataset": triples[0].get("dataset") if triples else None,
            "context": context,
            "generation": {
                "qa_per_context": qa_per_context,
                "model_name": gen_params["model_name"],
                "num_beams": gen_params["num_beams"],
                "max_length": gen_params["max_length"],
                "answer_markup": gen_params["answer_markup"],
                "span_strategy": gen_params["span_strategy"],
                "max_span_candidates": gen_params["max_span_candidates"],
                "min_span_len": gen_params["min_span_len"],
                "max_span_len": gen_params["max_span_len"],
                "error": generation_error,
                "warnings": generation_warnings,
            },
            "span_candidates": span_candidates,
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
                    "generation_error": generation_error,
                    "generation_warnings": generation_warnings,
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

    pipe = _build_qg_pipeline(args.model_name, args.device)

    gen_params = {
        "model_name": args.model_name,
        "num_beams": max(1, int(args.num_beams)),
        "max_length": max(8, int(args.max_length)),
        "answer_markup": args.answer_markup,
        "span_strategy": args.span_strategy,
        "max_span_candidates": max(1, int(args.max_span_candidates)),
        "min_span_len": max(1, int(args.min_span_len)),
        "max_span_len": max(2, int(args.max_span_len)),
    }

    if gen_params["span_strategy"] == "jieba" and not JIEBA_AVAILABLE:
        print(
            "[WARN] 你选择了 --span-strategy jieba 但环境未安装 jieba，将回退为 regex。",
            file=sys.stderr,
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
        1, int(os.environ.get("BENCH_JUDGE_MAX_CONCURRENCY", str(args.max_workers)))
    )
    worker_count = max(1, min(args.max_workers, judge_max_concurrency))

    qa_per_context = max(1, int(args.qa_per_context))

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_context_group,
                    pipe,
                    context,
                    triples,
                    qa_per_context,
                    gen_params,
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
