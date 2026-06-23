# 文件作用：合并多个单文件 consolidated payload 为批量汇总结果。
# 关联说明：由 pipeline_execution 调用，和 consolidation.py 的单文件结果构造形成上下游关系。

import time
from collections import Counter
from typing import Any, Dict, List

from .score_summary import _compute_unsupervised_scores_from_items

def merge_consolidated_entries(
    task_id: str,
    entries: List[Dict[str, Any]],
    chunk_size: int,
    qa_per_chunk: int,
    qa_detail_mode: str,
    prompt_language: str,
    include_evaluation: bool,
    evaluation_method: str,
    filter_by_threshold: bool,
    score_threshold: float,
    llm_model: str,
    include_unsupervised_evaluation: bool = False,
) -> Dict[str, Any]:
    """
    Merge multiple single-file consolidated payloads into one summary payload.
    Used by batch endpoint 8B.
    """
    created_ts = int(time.time())

    combined_items: List[Dict[str, Any]] = []
    combined_facts: List[Dict[str, Any]] = []
    category_counter: Counter = Counter()

    generation_seconds_total = 0.0
    generation_qa_total = 0
    ocr_seconds_total = 0.0
    evaluation_seconds_total = 0.0
    evaluation_qa_total = 0
    unsupervised_seconds_total = 0.0
    unsupervised_qa_total = 0
    total_facts = 0
    total_qa = 0
    total_filtered = 0
    filter_details: List[Dict[str, Any]] = []
    original_files: List[str] = []

    for entry in entries:
        payload = entry["payload"]
        original_files.append(entry["filename"])

        combined_items.extend(payload.get("items", []))
        combined_facts.extend(payload.get("facts", []))

        category_counter.update(
            payload.get("category_distribution")
            or payload.get("theme_distribution")
            or {}
        )

        counts = payload.get("counts", {}) or {}
        total_facts += counts.get("facts", 0)
        total_qa += counts.get("qa_pairs", 0)
        total_filtered += counts.get("filtered_qa_pairs", 0)

        timing = payload.get("timing") or {}
        if timing.get("ocr_seconds") is not None:
            ocr_seconds_total += float(timing.get("ocr_seconds") or 0.0)
        if timing.get("generation_seconds"):
            generation_seconds_total += float(timing.get("generation_seconds") or 0.0)
        generation_qa_total += timing.get("qa_generated") or 0
        if timing.get("evaluation_seconds"):
            evaluation_seconds_total += float(timing.get("evaluation_seconds") or 0.0)
        evaluation_qa_total += timing.get("qa_evaluated") or 0
        if timing.get("unsupervised_seconds") is not None:
            unsupervised_seconds_total += float(timing.get("unsupervised_seconds") or 0.0)
        unsupervised_qa_total += timing.get("unsupervised_qa_scored") or 0

        if payload.get("filter_info"):
            filter_details.append(
                {"filename": entry["filename"], **payload["filter_info"]}
            )

    generation_avg = (
        (generation_seconds_total / generation_qa_total)
        if generation_qa_total
        else None
    )
    evaluation_avg = (
        (evaluation_seconds_total / evaluation_qa_total)
        if evaluation_qa_total
        else None
    )
    unsupervised_avg = (
        (unsupervised_seconds_total / unsupervised_qa_total)
        if unsupervised_qa_total
        else None
    )

    filter_basis = (
        evaluation_method
        if (filter_by_threshold and evaluation_method in ("llm", "local", "faithfulness", "answerability", "unsupervised_f1"))
        else None
    )

    params: Dict[str, Any] = {
        "chunk_size": chunk_size,
        "qa_per_chunk": qa_per_chunk,
        "qa_detail_mode": qa_detail_mode,
        "prompt_language": prompt_language,
        "include_evaluation": include_evaluation,
        "include_unsupervised_evaluation": include_unsupervised_evaluation,
        "evaluation_method": evaluation_method,
        "filter_by_threshold": filter_by_threshold,
        "score_threshold": score_threshold if filter_by_threshold else None,
    }

    counts: Dict[str, Any] = {
        "facts": total_facts,
        "qa_pairs": total_qa,
        "filtered_qa_pairs": total_filtered,
    }

    merged_payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "task": {
            "task_id": task_id,
            "original_filename": original_files,
            "created_at": created_ts,
            "params": params,
        },
        "model": {
            "llm_model": llm_model,
            # 合并结果主要用于下载/检查，这里用一个固定占位即可
            "embed_model": "bge-m3",
            "embed_dim": 1024,
            "distance": "cosine",
        },
        "counts": counts,
        "category_distribution": dict(category_counter),
        "theme_distribution": dict(category_counter),
        "filter_info": {
            "threshold": score_threshold,
            "entries": filter_details,
        }
        if filter_by_threshold and filter_details
        else None,
        "filter_basis": filter_basis,
        "timing": {
            "ocr_seconds": ocr_seconds_total,
            "generation_seconds": generation_seconds_total
            if generation_qa_total
            else None,
            "generation_avg_seconds_per_qa": generation_avg,
            "qa_generated": generation_qa_total,
            "evaluation_seconds": evaluation_seconds_total
            if evaluation_qa_total
            else None,
            "evaluation_avg_seconds_per_qa": evaluation_avg,
            "qa_evaluated": evaluation_qa_total,
            "unsupervised_seconds": unsupervised_seconds_total
            if unsupervised_qa_total
            else None,
            "unsupervised_avg_seconds_per_qa": unsupervised_avg,
            "unsupervised_qa_scored": unsupervised_qa_total,
        },
        "facts": combined_facts,
        "items": combined_items,
    }
    if include_unsupervised_evaluation:
        merged_payload["unsupervised_scores"] = _compute_unsupervised_scores_from_items(combined_items)
    return merged_payload

__all__ = ["merge_consolidated_entries"]
