# 文件作用：运行项目主 QA 生成流程的合成数据基准评测。
# 关联说明：作为主流程基准，给其他 baseline 脚本复用样本读取和对齐工具。

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from openai import OpenAI
import hashlib
import threading

# 确保可以从项目根目录导入 qa / app 等包（支持直接 python qa/baseline_eval/benchmark_synthetic_qa.py 运行）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qa import process_text_to_qa_one_step
from qa.baseline_eval.common import ALIGN_THRESHOLD, ALIGN_WEIGHTS, _iter_triples, _pick_best_aligned_qa
from qa.baseline_eval.judge import _load_config_from_env, build_judge_client, judge_pair


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "基于 triples JSONL（context/question/answer），使用完整 QA pipeline facade"
            "生成候选 QA，再做嵌入对齐 + LLM-as-a-judge 评审，输出 scored JSONL。"
        ),
    )
    parser.add_argument(
        "--triples-file",
        "-t",
        type=str,
        required=True,
        help="三元组 JSONL 文件（如 F:\\qa\\dataset\\cmrc2018_dev_triples.jsonl）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="评测输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2500,
        help="最多评测多少条样本（默认 2500）",
    )
    parser.add_argument(
        "--qa-per-context",
        type=int,
        default=1,
        help="兼容旧参数：若未显式设置 --qa-per-chunk，则用该值作为 qa_per_chunk（默认 1）",
    )
    parser.add_argument(
        "--qa-per-chunk",
        type=int,
        default=0,
        help="每个 chunk 期望生成的主问答条数（默认 0=自动：优先用 --qa-per-context，再回退为 1）",
    )
    parser.add_argument(
        "--qa-per-fact",
        type=int,
        default=1,
        help="兼容旧参数（已不再使用 facts 流水线）：若未设置 --qa-per-chunk/--qa-per-context，则回退用它",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=600,
        help="一步式生成时的 chunk 大小（默认 600；过大可能降低遵循性/增加成本）",
    )
    parser.add_argument(
        "--max-candidates-per-context",
        type=int,
        default=50,
        help="每个 context 最多保留多少条候选 QA 进入对齐阶段（默认 50；0 表示不截断）",
    )
    parser.add_argument(
        "--qa-detail-mode",
        choices=["point", "summary"],
        default="point",
        help="问答粒度：point=单点事实直答；summary=多点合并用于总结/对比/推理（默认 point）",
    )
    parser.add_argument(
        "--prompt-language",
        choices=["auto", "zh", "en"],
        default="auto",
        help="提示词语言：auto/zh/en（默认 auto）",
    )
    parser.add_argument(
        "--question-type-mode",
        choices=["fixed", "mixed"],
        default="fixed",
        help="题型模式：fixed=固定使用首个题型；mixed=在题型集合内混合（默认 fixed）",
    )
    parser.add_argument(
        "--question-types",
        type=str,
        default="简答题",
        help="题型列表（逗号分隔）：简答题/单选题/判断题/计算题；默认 '简答题'",
    )
    parser.add_argument(
        "--question-type-weights",
        type=str,
        default="",
        help='题型权重（仅 mixed 生效），JSON，如 {"简答题":0.6,"判断题":0.4}',
    )
    parser.add_argument(
        "--fact-concurrency",
        type=int,
        default=100,
        help="兼容旧参数：已不使用（facts 流水线）",
    )
    parser.add_argument(
        "--categorize-concurrency",
        type=int,
        default=50,
        help="兼容旧参数：已不使用（facts 流水线）",
    )
    parser.add_argument(
        "--qa-concurrency",
        type=int,
        default=100,
        help="兼容旧参数：若未设置 --chunk-max-concurrency，则用该值作为 chunk_max_concurrency",
    )
    parser.add_argument(
        "--chunk-max-concurrency",
        type=int,
        default=8,
        help="同一 context 内 chunk 级并发（默认 8；建议与 --max-workers 共同控制避免线程爆炸）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="并行评测线程数（默认 50，仍会被 BENCH_JUDGE_MAX_CONCURRENCY 截断）",
    )
    parser.add_argument(
        "--gen-api-key",
        type=str,
        help="生成侧 API Key（默认读取环境变量 BENCH_GEN_API_KEY，未设置则回退 judge 配置）",
    )
    parser.add_argument(
        "--gen-base-url",
        type=str,
        help="生成侧 Base URL（默认读取 BENCH_GEN_BASE_URL，未设置则回退 judge 配置）",
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        help="生成侧模型名称（默认读取 BENCH_GEN_MODEL，未设置则回退 deepseek-reasoner）",
    )
    parser.add_argument(
        "--save-generated",
        type=str,
        help="可选：将每个 context 生成的全部 QA 写入指定 JSONL，便于排查对齐问题",
    )
    parser.add_argument(
        "--debug-file",
        type=str,
        default="",
        help="可选：写入一步式生成的 chunk 级调试 JSONL（记录 LLM 原始输出/解析结果）",
    )
    parser.add_argument(
        "--few-shot-examples",
        type=int,
        default=4,
        help="自动从数据集前 N 条构造 few-shot 示例（默认 4，0 表示禁用自动示例）",
    )
    parser.add_argument(
        "--align-threshold",
        type=float,
        default=ALIGN_THRESHOLD,
        help=f"对齐阈值（默认 {ALIGN_THRESHOLD}；低于则视为失败并默认跳过该样本）",
    )
    parser.add_argument(
        "--align-weight-question",
        type=float,
        default=float(ALIGN_WEIGHTS.get("question", 0.65)),
        help="对齐权重：question（默认 0.65）",
    )
    parser.add_argument(
        "--align-weight-answer",
        type=float,
        default=float(ALIGN_WEIGHTS.get("answer", 0.35)),
        help="对齐权重：answer（默认 0.35）",
    )
    parser.add_argument(
        "--keep-below-threshold",
        action="store_true",
        help="保留未达对齐阈值的样本（仍会输出，但标记 below_threshold=True）",
    )
    return parser.parse_args()


def _build_auto_few_shot(triples: List[Dict[str, Any]], max_examples: int = 4) -> List[Dict[str, str]]:
    """
    从数据集样本中自动抽取 few-shot 示例，避免手动维护。
    默认取前 max_examples 条带 ref_question/ref_answer 的记录。
    """
    examples: List[Dict[str, str]] = []
    for t in triples:
        if len(examples) >= max_examples:
            break
        q = t.get("ref_question") or t.get("question")
        a = t.get("ref_answer") or t.get("answer")
        if q and a:
            examples.append({"question": str(q), "answer": str(a)})
    return examples


def _build_generation_client(cfg: Dict[str, Any]) -> OpenAI:
    """
    生成侧 client：默认优先使用 BENCH_GEN_* / CLI 传入的配置。
    """
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def _generate_synthetic_qa_for_context(
    client: OpenAI,
    context: str,
    qa_per_context: int,
    gen_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    使用完整 QA pipeline facade：
    - text -> split chunks
    - each chunk -> LLM generate {"items":[...]} once
    - flatten items -> candidate QA list
    """
    qa_per_chunk = int(gen_config.get("qa_per_chunk") or 0)
    if qa_per_chunk <= 0:
        qa_per_chunk = max(1, int(qa_per_context or 1))

    items = process_text_to_qa_one_step(
        client,
        context,
        {
            "chunk_size": int(gen_config.get("chunk_size") or 600),
            "qa_per_chunk": qa_per_chunk,
            "qa_detail_mode": str(gen_config.get("qa_detail_mode") or "point"),
            "prompt_language": str(gen_config.get("prompt_language") or "auto"),
            "chunk_max_concurrency": int(gen_config.get("chunk_max_concurrency") or 8),
            "model": str(gen_config.get("model") or ""),
            "request_timeout": int(gen_config.get("request_timeout") or 120),
            "question_type_mode": str(gen_config.get("question_type_mode") or "fixed"),
            "question_types": gen_config.get("question_types"),
            "question_type_weights": gen_config.get("question_type_weights"),
            "few_shot_examples": gen_config.get("few_shot_examples"),
            "debug_file": gen_config.get("debug_file"),
        },
        original_filename="benchmark",
    )

    qa_data: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        q = str(it.get("question", "") or "").strip()
        a = str(it.get("answer", "") or "").strip()
        if not q or not a:
            continue
        it["question"] = q
        it["answer"] = a
        qa_data.append(it)

    if not qa_data:
        return []

    # 稳定排序：避免 chunk 并发导致候选截断不稳定
    qa_data.sort(key=lambda x: (str(x.get("question", "")), str(x.get("answer", ""))))

    max_candidates = int(gen_config.get("max_candidates_per_context") or 0)
    if max_candidates > 0:
        qa_data = qa_data[:max_candidates]
    return qa_data


def _build_gen_config_from_env(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    从评审配置派生一个生成配置，优先使用 BENCH_GEN_* / CLI 提供的配置。
    """
    env_api_key = os.environ.get("BENCH_GEN_API_KEY")
    env_base_url = os.environ.get("BENCH_GEN_BASE_URL")
    env_model = os.environ.get("BENCH_GEN_MODEL")
    return {
        "api_key": env_api_key or base_cfg.get("api_key") or "",
        "base_url": env_base_url or base_cfg.get("base_url") or "https://api.deepseek.com/v1",
        "model": env_model or base_cfg.get("model") or "deepseek-chat",
        "max_retries": base_cfg.get("max_retries", 2),
        "request_timeout": base_cfg.get("request_timeout", 120),
        # 一步式生成（与端点 8 保持一致）
        "chunk_size": 600,
        "qa_per_chunk": 1,
        "qa_detail_mode": "point",
        "prompt_language": "auto",
        "chunk_max_concurrency": 8,
        # 默认生成简答题，以贴近抽取式/简答题数据集
        "question_type_mode": "fixed",
        "question_types": ["简答题"],
        "question_type_weights": None,
        # 对齐阶段候选上限，避免过多噪声
        "max_candidates_per_context": 50,
        "debug_file": None,
    }


def _cosine_similarity(vec1: Any, vec2: Any) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    if v1.ndim != 1:
        v1 = v1.reshape(-1)
    if v2.ndim != 1:
        v2 = v2.reshape(-1)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _pick_best_aligned_qa_with_cached_candidates(
    *,
    ref_q_emb: Any,
    ref_a_emb: Any | None,
    cand_q_embs: List[Any],
    cand_a_embs: List[Any] | None,
    synthetic_qas: List[Dict[str, Any]],
    align_weights: Dict[str, float],
    align_threshold: float,
    keep_below_threshold: bool,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if not synthetic_qas:
        return {}, {
            "question_similarity": 0.0,
            "answer_similarity": 0.0,
            "alignment_score": 0.0,
        }

    has_ref_answer = ref_a_emb is not None and bool(cand_a_embs)

    best_idx = 0
    best_score = -1.0
    best_q_sim = 0.0
    best_a_sim = 0.0

    for idx, _qa in enumerate(synthetic_qas):
        if idx >= len(cand_q_embs):
            break
        q_sim = _cosine_similarity(ref_q_emb, cand_q_embs[idx])
        if has_ref_answer and cand_a_embs is not None and idx < len(cand_a_embs):
            a_sim = _cosine_similarity(ref_a_emb, cand_a_embs[idx])  # type: ignore[arg-type]
            align_score = (
                align_weights.get("question", 0.7) * q_sim
                + align_weights.get("answer", 0.3) * a_sim
            )
        else:
            a_sim = 0.0
            align_score = q_sim

        if align_score > best_score:
            best_score = align_score
            best_idx = idx
            best_q_sim = q_sim
            best_a_sim = a_sim

    below = best_score < align_threshold
    best_item = synthetic_qas[best_idx] if (keep_below_threshold or not below) else {}
    info: Dict[str, float] = {
        "question_similarity": float(best_q_sim),
        "answer_similarity": float(best_a_sim),
        "alignment_score": float(best_score),
    }
    if below:
        info["below_threshold"] = True  # type: ignore[assignment]
    return best_item, info


def _evaluate_context_group(
    judge_client: OpenAI,
    gen_client: OpenAI,
    gen_config: Dict[str, Any],
    context: str,
    triples: List[Dict[str, Any]],
    qa_per_context: int,
    aligner: QAEvaluator,
    wall_start: float,
    align_weights: Dict[str, float],
    align_threshold: float,
    keep_below_threshold: bool,
    save_generated_path: str | None = None,
    save_lock: threading.Lock | None = None,
) -> List[Dict[str, Any]]:
    """
    按 context 维度生成一次合成 QA，然后对该 context 下的所有 (question, answer)
    逐条做嵌入对齐 + LLM 评审。

    返回：该 context 下所有带 scores 的记录列表。
    """
    # 仅对该 context 生成一次合成 QA
    gen_start = time.perf_counter()
    synthetic_qas = _generate_synthetic_qa_for_context(
        gen_client,
        context,
        qa_per_context=qa_per_context,
        gen_config=gen_config,
    )
    gen_duration = time.perf_counter() - gen_start
    wall_duration = time.perf_counter() - wall_start

    # 可选：将该 context 下生成的全部 QA 记录下来，便于排查对齐问题
    if save_generated_path:
        os.makedirs(os.path.dirname(save_generated_path) or ".", exist_ok=True)
        record = {
            "context_id": triples[0].get("id") if triples else None,
            "dataset": triples[0].get("dataset") if triples else None,
            "context_hash": hashlib.sha1(context.encode("utf-8")).hexdigest(),
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
        if save_lock:
            with save_lock:
                with open(save_generated_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with open(save_generated_path, "a", encoding="utf-8") as out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    results: List[Dict[str, Any]] = []

    if not synthetic_qas:
        # 生成失败时，该 context 下所有样本统一记为“未能生成合成问答”
        for triple in triples:
            ref_question = triple.get("question", "")
            ref_answer = triple.get("answer", "")
            scored = {
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
                    "correctness": {"score": 0.0, "reasons": "未能生成合成问答"},
                    "semantic_equivalence": {"score": 0.0, "reasons": "未能生成合成问答"},
                    "style_similarity": {"score": 0.0, "reasons": "未能生成合成问答"},
                },
            }
            results.append(scored)
        return results

    # 预计算候选向量（同一 context 下复用），避免每条 ref 重复 encode 候选列表
    cand_questions = [str(qa.get("question", "") or "") for qa in synthetic_qas]
    cand_q_embs = list(aligner.st_model.encode(cand_questions, convert_to_tensor=False))
    cand_answers = [str(qa.get("answer", "") or "") for qa in synthetic_qas]
    cand_a_embs = list(aligner.st_model.encode(cand_answers, convert_to_tensor=False)) if cand_answers else []

    # 对该 context 下的每个参考 QA，基于同一批 synthetic_qas 做嵌入对齐 + 评审
    for triple in triples:
        ref_question = triple.get("question", "")
        ref_answer = triple.get("answer", "")

        ref_q_emb = list(aligner.st_model.encode([str(ref_question or "")], convert_to_tensor=False))[0]
        has_ref_answer = bool(ref_answer and str(ref_answer).strip())
        ref_a_emb = (
            list(aligner.st_model.encode([str(ref_answer)], convert_to_tensor=False))[0]
            if has_ref_answer
            else None
        )
        best_qa, align_info = _pick_best_aligned_qa_with_cached_candidates(
            ref_q_emb=ref_q_emb,
            ref_a_emb=ref_a_emb,
            cand_q_embs=cand_q_embs,
            cand_a_embs=cand_a_embs if has_ref_answer else None,
            synthetic_qas=synthetic_qas,
            align_weights=align_weights,
            align_threshold=align_threshold,
            keep_below_threshold=keep_below_threshold,
        )
        # 未达阈值则直接跳过保存，避免同一事实的多个低质量问答进入结果
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

        scored = {
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
        results.append(scored)

    return results


def main() -> None:
    """
    主流程（骨架版）：
    - 从 CMRC2018 / CJRC 的 triples JSONL 中抽取若干条样本；
    - 调用现有生成流水线合成 QA；
    - 用 LLM-as-a-judge 在三个维度上打分；
    - 输出逐条 JSONL，后续可以再写一个统计脚本做汇总。
    """
    args = _parse_args()
    cfg = _load_config_from_env()
    judge_client = build_judge_client(cfg)
    gen_config = _build_gen_config_from_env(cfg)
    # CLI 覆盖生成配置
    if args.gen_api_key:
        gen_config["api_key"] = args.gen_api_key
    if args.gen_base_url:
        gen_config["base_url"] = args.gen_base_url
    if args.gen_model:
        gen_config["model"] = args.gen_model
    gen_client = _build_generation_client(gen_config)

    qa_per_chunk = int(args.qa_per_chunk or 0)
    if qa_per_chunk <= 0:
        qa_per_chunk = int(args.qa_per_context or 0) or int(args.qa_per_fact or 0) or 1
    gen_config["qa_per_chunk"] = max(1, qa_per_chunk)
    gen_config["chunk_size"] = max(200, int(args.chunk_size))
    gen_config["max_candidates_per_context"] = max(0, int(args.max_candidates_per_context))
    gen_config["qa_detail_mode"] = str(args.qa_detail_mode or "point").strip().lower()
    gen_config["prompt_language"] = str(args.prompt_language or "auto").strip().lower()

    chunk_max_concurrency = int(args.chunk_max_concurrency or 0)
    if chunk_max_concurrency <= 0:
        chunk_max_concurrency = int(args.qa_concurrency or 0) or 8
    gen_config["chunk_max_concurrency"] = max(1, chunk_max_concurrency)

    gen_config["question_type_mode"] = str(args.question_type_mode or "fixed").strip().lower()
    raw_qtypes = str(args.question_types or "").strip()
    if raw_qtypes:
        normalized = raw_qtypes.replace("，", ",").replace("、", ",")
        gen_config["question_types"] = [s.strip() for s in normalized.split(",") if s.strip()]
    raw_weights = str(args.question_type_weights or "").strip()
    if raw_weights:
        gen_config["question_type_weights"] = raw_weights
    if str(args.debug_file or "").strip():
        gen_config["debug_file"] = str(args.debug_file).strip()

    align_weights = {
        "question": float(args.align_weight_question),
        "answer": float(args.align_weight_answer),
    }
    align_threshold = float(args.align_threshold)

    # 嵌入对齐模型（使用 qa_evaluation 中的 BGE-M3）
    try:
        from qa.qa_evaluation.qa_quality_evaluator import QAEvaluator  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "对齐依赖未安装或导入失败：请先安装 requirements.txt（需要 sentence-transformers 等）。"
        ) from exc
    try:
        aligner = QAEvaluator(use_local_models=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"对齐所需本地模型未就绪，请先下载 qa_evaluation 所需模型: {exc}")

    if not os.path.exists(args.triples_file):
        raise SystemExit(f"未找到三元组文件: {args.triples_file}")
    triples_iter = list(_iter_triples(args.triples_file, args.max_samples))
    if not triples_iter:
        raise SystemExit(f"未在 {args.triples_file} 中读取到有效样本")

    # 自动从数据集提取 few-shot 示例（可通过参数控制数量或禁用）
    if args.few_shot_examples > 0:
        auto_examples = _build_auto_few_shot(triples_iter, max_examples=args.few_shot_examples)
    else:
        auto_examples = []
    if auto_examples:
        gen_config["few_shot_examples"] = auto_examples

    # 按 context 聚合：对同一 context 只生成一次合成 QA
    context_groups: Dict[str, List[Dict[str, Any]]] = {}
    for triple in triples_iter:
        ctx = triple.get("context", "") or ""
        context_groups.setdefault(ctx, []).append(triple)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = len(context_groups)
    processed = 0
    wall_start = time.perf_counter()
    judge_max_concurrency = max(1, int(os.environ.get("BENCH_JUDGE_MAX_CONCURRENCY", "50")))
    worker_count = max(1, min(args.max_workers, judge_max_concurrency))
    save_lock = threading.Lock() if args.save_generated else None
    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_context_group,
                    judge_client,
                    gen_client,
                    gen_config,
                    context,
                    triples,
                    args.qa_per_context,
                    aligner,
                    time.perf_counter(),
                    align_weights,
                    align_threshold,
                    bool(args.keep_below_threshold),
                    args.save_generated,
                    save_lock,
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
