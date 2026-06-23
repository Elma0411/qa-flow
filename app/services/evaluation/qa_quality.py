# 文件作用：调度本地评价、LLM 评价和阈值过滤逻辑。
# 关联说明：桥接 qa.qa_evaluation 的评价器和 app 路由层的任务执行。

import asyncio
import json
import os
import tempfile
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from qa.qa_evaluation.llm_quality_evaluator import evaluate_qa_pairs
from qa.qa_evaluation.auto_metrics_evaluator import evaluate_qa_pairs_auto_metrics

from app.core.config import (
    AUTO_EVAL_MAX_ITEMS_PER_REQUEST,
    EVAL_BATCH_SIZE,
    LLM_EVALUATION_METRICS,
    LOCAL_EVALUATION_AVG_METRICS,
    LOCAL_EVALUATION_METRICS,
)
from app.services.gpu import clear_cuda_runtime_for_device, gpu_stage

try:
    from qa.qa_evaluation.qa_quality_evaluator import QAEvaluator

    QA_EVALUATION_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - optional dependency
    QAEvaluator = None  # type: ignore
    QA_EVALUATION_AVAILABLE = False
    QA_EVALUATION_IMPORT_ERROR = exc


def execute_llm_evaluation_blocking(
    qa_data: List[Dict[str, Any]],
    criteria_list: List[str],
    max_eval_concurrency: int = 8,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not qa_data:
        return None
    temp_qa_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    try:
        json.dump(qa_data, temp_qa_file, ensure_ascii=False, indent=2)
    finally:
        temp_qa_file.close()
    try:
        evaluation_results = evaluate_qa_pairs(
            temp_qa_file.name,
            criteria_list,
            max_concurrency=max_eval_concurrency,
            llm_config=llm_config,
        )
        return evaluation_results
    finally:
        try:
            os.unlink(temp_qa_file.name)
        except OSError:
            pass


def execute_local_evaluation_blocking(
    qa_data: List[Dict[str, Any]],
    use_local_models: bool,
    *,
    bertscore_device: Optional[str] = None,
    gpu_job_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Local evaluation is defined as "automatic metrics evaluation" (EM/Token_F1/ROUGE_L_F1/BLEU/BERTScore).

    Notes:
    - `use_local_models` is kept for backward compatibility but is no longer used.
    - When reference text is missing, metrics will be 0 and `missing_reference=1`.
    """
    if not qa_data:
        return None
    if len(qa_data) > AUTO_EVAL_MAX_ITEMS_PER_REQUEST:
        raise ValueError(
            f"QA条数过多({len(qa_data)})，请拆分后再评估，或使用离线脚本批量评估（上限 {AUTO_EVAL_MAX_ITEMS_PER_REQUEST}）"
        )
    if gpu_job_id:
        with gpu_stage(str(gpu_job_id), "local_bertscore") as leased_device:
            effective_device = leased_device or bertscore_device
            try:
                return evaluate_qa_pairs_auto_metrics(qa_data, bertscore_device=effective_device)
            finally:
                clear_cuda_runtime_for_device(effective_device)
    return evaluate_qa_pairs_auto_metrics(qa_data, bertscore_device=bertscore_device)


def filter_qa_pairs_by_threshold(
    evaluation_results: Optional[Dict[str, Any]],
    evaluation_method: str,
    score_threshold: float,
    criteria_list: List[str],
    local_evaluation_results: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not evaluation_results:
        return [], None
    filtered_qa_pairs: List[Dict[str, Any]] = []
    original_qa_data = evaluation_results.get("results", [])
    local_metrics = LOCAL_EVALUATION_METRICS
    for qa_result in original_qa_data:
        evaluation = qa_result.get("evaluation", {})
        avg_score = qa_result.get("average_score", 0.0)
        if evaluation_method == "llm":
            scores = [evaluation.get(metric, {}).get("score") for metric in criteria_list if metric in evaluation]
            scores = [s for s in scores if isinstance(s, (int, float))]
            avg_score = float(sum(scores) / len(scores)) if scores else 0.0
        elif evaluation_method == "local":
            avg_score = float(avg_score or 0.0)
        qa_result["average_score"] = avg_score
        if avg_score < score_threshold:
            continue
        qa_item = {
            "question": qa_result.get("question", ""),
            "answer": qa_result.get("answer", ""),
            "theme": qa_result.get("theme", ""),
            "average_score": avg_score,
            "evaluation_scores": {},
        }
        if evaluation_method == "llm":
            for metric in criteria_list:
                metric_entry = evaluation.get(metric)
                if isinstance(metric_entry, dict) and "score" in metric_entry:
                    qa_item["evaluation_scores"][metric] = metric_entry["score"]
        elif evaluation_method == "local":
            for metric in local_metrics:
                metric_entry = evaluation.get(metric)
                if isinstance(metric_entry, dict) and "score" in metric_entry:
                    qa_item["evaluation_scores"][metric] = metric_entry["score"]
        filtered_qa_pairs.append(qa_item)
    filter_info = {
        "threshold": score_threshold,
        "original_count": len(original_qa_data),
        "filtered_count": len(filtered_qa_pairs),
        "removed_count": len(original_qa_data) - len(filtered_qa_pairs),
    }
    return filtered_qa_pairs, filter_info


def chunked_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def run_llm_evaluation_batches(
    qa_data: List[Dict[str, Any]],
    criteria_list: List[str],
    progress_callback: Callable[[str], Awaitable[Any]],
    max_eval_concurrency: int = 8,
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    batches = chunked_list(qa_data, EVAL_BATCH_SIZE)
    if not batches:
        return {"method": "llm_batch", "results": [], "batch_count": 0}
    tasks = {
        asyncio.create_task(
            asyncio.to_thread(
                execute_llm_evaluation_blocking,
                batch,
                criteria_list,
                max_eval_concurrency,
                llm_config,
            )
        ): idx + 1
        for idx, batch in enumerate(batches)
    }
    completed = 0
    results = []
    total_batches = len(batches)
    for fut in asyncio.as_completed(tasks):
        completed += 1
        res = await fut
        await progress_callback(f"LLM 评估批次 {completed}/{total_batches}")
        if res:
            results.append(res)
    combined_results = []
    for res in results:
        combined_results.extend(res.get("results", []))
    return {
        "method": "llm_batch",
        "results": combined_results,
        "batch_count": total_batches,
    }


__all__ = [
    "QA_EVALUATION_AVAILABLE",
    "chunked_list",
    "execute_llm_evaluation_blocking",
    "execute_local_evaluation_blocking",
    "filter_qa_pairs_by_threshold",
    "run_llm_evaluation_batches",
]
