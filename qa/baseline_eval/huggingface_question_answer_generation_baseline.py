# 文件作用：运行 HuggingFace 问答生成模型基线评测。
# 关联说明：复用 benchmark_synthetic_qa 的对齐和 synthetic_qa_judge 的评审，替换生成策略为 HF QAG。

from __future__ import annotations

import argparse
import json
import os
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QAG 基线：使用 HuggingFace QAG 模型（context→question+answer），再复用对齐+评审。",
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
        default="iarfmoose/t5-base-question-generator",
        help="QAG 模型名称或本地路径（默认 iarfmoose/t5-base-question-generator）。",
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
        default=96,
        help="生成最大 token 数（默认 96）。",
    )
    parser.add_argument(
        "--save-generated",
        type=str,
        help="可选：将每个 context 的候选问答写入指定 JSONL 便于排查。",
    )
    return parser.parse_args()


def _build_qag_pipeline(model_name: str, device: str):
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


def _parse_qag_output(text: str) -> Tuple[str, str]:
    """
    尝试从模型输出中拆分 question / answer。
    常见格式：'question ? <sep> answer' 或 'question ? answer'.
    """
    if not text:
        return "", ""
    raw = text.strip()
    # 优先 <sep>
    if "<sep>" in raw:
        parts = raw.split("<sep>", 1)
        return parts[0].strip(), parts[1].strip()
    # 尝试 Question: / Answer:
    if "Answer:" in raw and "Question:" in raw:
        q_part = raw.split("Question:", 1)[1]
        if "Answer:" in q_part:
            q_text, a_text = q_part.split("Answer:", 1)
            return q_text.strip(" ?：:").strip(), a_text.strip()
    # 按第一个问号截断
    q_mark = raw.find("?")
    if q_mark != -1:
        q_text = raw[: q_mark + 1].strip()
        a_text = raw[q_mark + 1 :].strip("：: ").strip()
        return q_text, a_text
    return raw, ""


def _generate_qas_for_context(
    pipe: Any,
    context: str,
    qa_per_context: int,
    num_beams: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    prompt = f"generate qa: {context}"
    try:
        outputs = pipe(
            prompt,
            num_beams=num_beams,
            max_new_tokens=max_length,
            num_return_sequences=qa_per_context,
        )
    except Exception:
        outputs = []
    qas: List[Dict[str, Any]] = []
    if isinstance(outputs, list):
        for item in outputs:
            text = item.get("generated_text") if isinstance(item, dict) else None
            if not text:
                continue
            q, a = _parse_qag_output(text)
            if q:
                qas.append({"question": q, "answer": a or context[:80]})
    return qas


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
    """
    对单个 context 生成一批 QA，再对每条参考 QA 做对齐+评审。
    """
    gen_start = time.perf_counter()
    synthetic_qas = _generate_qas_for_context(
        pipe,
        context,
        qa_per_context=qa_per_context,
        num_beams=gen_params["num_beams"],
        max_length=gen_params["max_length"],
    )
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

    pipe = _build_qag_pipeline(args.model_name, args.device)
    gen_params = {
        "num_beams": max(1, args.num_beams),
        "max_length": max(8, args.max_length),
    }

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

    with open(args.output, "w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _evaluate_context_group,
                    pipe,
                    context,
                    triples,
                    args.qa_per_context,
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
