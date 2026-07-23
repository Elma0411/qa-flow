# 文件作用：执行评测数据集生成、评审和结果写出流程。
# 关联说明：串联 dataset、qa pipeline 和 judge，实际执行评测作业。

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from app.core.config import AUTO_EVAL_MAX_ITEMS_PER_REQUEST, CONFIG, LOCAL_EVALUATION_METRICS
from app.core.logger import logger
from .dataset import (
    extract_canonical_rows,
    load_dataset_rows_from_path,
    resolve_row_range,
)
from .common import (
    _chunk,
    _safe_float,
    _unsupervised_scores_from_suite_summary,
    _write_json,
    _write_jsonl,
)
from app.services.evaluation import execute_local_evaluation_blocking
from app.services.unsupervised_evaluation import (
    UNSUPERVISED_EVALUATION_AVAILABLE,
    execute_unsupervised_suite_blocking,
    resolve_evaluation_model_path,
)


def _extract_local_scores(local_eval_row: Dict[str, Any]) -> Dict[str, float]:
    evaluation = local_eval_row.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return {}
    scores: Dict[str, float] = {}
    for metric in LOCAL_EVALUATION_METRICS:
        entry = evaluation.get(metric)
        if isinstance(entry, dict) and isinstance(entry.get("score"), (int, float)):
            scores[metric] = float(entry["score"])
    return scores


def evaluate_dataset_job(
    *,
    job_id: str,
    input_files: List[Dict[str, Any]],
    dataset_name: str,
    question_field: str,
    answer_field: str,
    context_field: str,
    ref_answer_field: Optional[str],
    id_field: Optional[str],
    original_filename_field: Optional[str],
    input_format: str,
    encoding: Optional[str],
    delimiter: str,
    sheet_name: Optional[str],
    unsupervised_batch_size: Optional[int] = None,
    faithfulness_nli_model: Optional[str] = None,
    answerability_qa_model: Optional[str] = None,
    coverage_embedding_model: Optional[str] = None,
    faithfulness_hypothesis_mode: Optional[str] = None,
    faithfulness_hypothesis_timeout: Optional[int] = None,
    faithfulness_hypothesis_max_retries: Optional[int] = None,
    faithfulness_hypothesis_max_concurrency: Optional[int] = None,
    gpu_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    started_at = time.time()
    outputs_dir = CONFIG["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)
    task_id = f"eval_job_{job_id}"

    normalized_inputs: List[Dict[str, Any]] = []
    for file_index, item in enumerate(input_files or []):
        if not isinstance(item, dict):
            raise ValueError(f"input_files[{file_index}] 必须是对象")
        input_path = str(item.get("input_path") or "").strip()
        if not input_path:
            raise ValueError(f"input_files[{file_index}].input_path 不能为空")
        upload_filename = (
            os.path.basename(str(item.get("upload_filename") or "").strip())
            or os.path.basename(input_path)
            or f"file_{file_index}"
        )
        normalized_inputs.append(
            {
                "file_index": int(item.get("file_index", file_index)),
                "input_path": input_path,
                "upload_filename": upload_filename,
                "row_start": item.get("row_start"),
                "row_end": item.get("row_end"),
            }
        )
    if not normalized_inputs:
        raise ValueError("至少需要一个输入文件")

    qa_rows: List[Dict[str, Any]] = []
    file_summaries: List[Dict[str, Any]] = []
    baseline_columns: Optional[List[str]] = None
    baseline_column_set: Optional[set[str]] = None
    shared_column_set: Optional[set[str]] = None

    for input_item in normalized_inputs:
        detected_format, columns, raw_rows = load_dataset_rows_from_path(
            input_item["input_path"],
            input_format=input_format,
            encoding=encoding,
            delimiter=delimiter,
            sheet_name=sheet_name,
        )
        column_list = [str(col) for col in (columns or [])]
        column_set = set(column_list)
        if baseline_columns is None:
            baseline_columns = list(column_list)
            baseline_column_set = set(column_set)
            shared_column_set = set(column_set)
        else:
            shared_column_set = set(shared_column_set or set()).intersection(column_set)
            missing_columns = sorted(set(baseline_column_set or set()) - column_set)
            extra_columns = sorted(column_set - set(baseline_column_set or set()))
            if missing_columns or extra_columns:
                raise ValueError(
                    "批量评测要求所有文件字段完全一致："
                    f"{input_item['upload_filename']} 缺少 {missing_columns or '[]'}，"
                    f"多出 {extra_columns or '[]'}"
                )

        row_range = resolve_row_range(
            total_rows=len(raw_rows),
            row_start=input_item.get("row_start"),
            row_end=input_item.get("row_end"),
        )
        range_zero = row_range.get("resolved_zero_based") if isinstance(row_range, dict) else {}
        start_0 = range_zero.get("start")
        end_0 = range_zero.get("end")

        canonical_rows_all = extract_canonical_rows(
            raw_rows,
            dataset_name=dataset_name,
            task_id=task_id,
            original_filename_default=input_item["upload_filename"],
            question_field=question_field,
            answer_field=answer_field,
            context_field=context_field,
            ref_answer_field=ref_answer_field,
            id_field=id_field,
            original_filename_field=original_filename_field,
        )
        canonical_rows = [
            row
            for row in canonical_rows_all
            if isinstance(row, dict)
            and isinstance(row.get("_row_index"), int)
            and start_0 is not None
            and end_0 is not None
            and start_0 <= int(row.get("_row_index")) <= end_0
        ]

        file_summaries.append(
            {
                "file_index": input_item["file_index"],
                "upload_filename": input_item["upload_filename"],
                "detected_format": detected_format,
                "columns": column_list,
                "row_count_raw": len(raw_rows),
                "row_count_selected": len(canonical_rows),
                "row_range": row_range,
            }
        )

        for row in canonical_rows:
            qa_rows.append(
                {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "task_id": row["task_id"],
                    "original_filename": row["original_filename"],
                    "context": row["context"],
                    "source_fact_text": row["context"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "ref_answer": row.get("ref_answer"),
                    "meta": row.get("meta") or {},
                    "is_primary": True,
                    "is_augmented": False,
                }
            )

    parse_seconds = time.time() - started_at

    unsup_started = time.time()
    unsup_summary: Dict[str, Any] = {"method": "unsupervised_suite_v1", "scores": {}}
    if not UNSUPERVISED_EVALUATION_AVAILABLE:
        unsup_summary["available"] = False
        unsup_summary["error"] = "unsupervised evaluation dependencies missing"
    else:
        unsup_cfg = CONFIG.get("unsupervised") if isinstance(CONFIG.get("unsupervised"), dict) else {}
        request_timeout = (
            faithfulness_hypothesis_timeout
            if faithfulness_hypothesis_timeout is not None
            else unsup_cfg.get("hypothesis_timeout")
        )
        max_retries = (
            faithfulness_hypothesis_max_retries
            if faithfulness_hypothesis_max_retries is not None
            else unsup_cfg.get("hypothesis_max_retries")
        )
        max_concurrency = (
            faithfulness_hypothesis_max_concurrency
            if faithfulness_hypothesis_max_concurrency is not None
            else unsup_cfg.get("hypothesis_max_concurrency")
        )
        faith_model_path = resolve_evaluation_model_path(
            faithfulness_nli_model,
            kind="faithfulness_nli",
        )
        qa_model_path = resolve_evaluation_model_path(
            answerability_qa_model,
            kind="answerability_qa",
        )
        coverage_model_path = resolve_evaluation_model_path(
            coverage_embedding_model,
            kind="coverage_embedding",
        )
        hypothesis_mode = faithfulness_hypothesis_mode or unsup_cfg.get("hypothesis_mode")
        llm_api_key = unsup_cfg.get("hypothesis_api_key") or CONFIG.get("api_key")
        llm_base_url = unsup_cfg.get("hypothesis_base_url") or CONFIG.get("base_url")
        llm_model = unsup_cfg.get("hypothesis_model") or CONFIG.get("model")
        batch_size_override: Optional[int] = None
        if unsupervised_batch_size is not None:
            try:
                batch_size_override = int(unsupervised_batch_size)
            except Exception:
                batch_size_override = None
            if batch_size_override is not None:
                batch_size_override = max(1, min(512, int(batch_size_override)))
        try:
            unsup_summary = execute_unsupervised_suite_blocking(
                qa_rows,
                only_primary=True,
                prune_item_details=False,
                faith_model_path=faith_model_path,
                qa_model_path=qa_model_path,
                coverage_embed_model_path=coverage_model_path,
                hypothesis_mode=hypothesis_mode,
                llm_api_key=llm_api_key,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                llm_request_timeout=int(request_timeout) if request_timeout is not None else None,
                llm_max_retries=int(max_retries) if max_retries is not None else None,
                llm_max_concurrency=int(max_concurrency) if max_concurrency is not None else None,
                faith_batch_size=batch_size_override,
                qa_batch_size=batch_size_override,
                coverage_embed_batch_size=batch_size_override,
                fluency_batch_size=batch_size_override,
                gpu_job_id=gpu_job_id or job_id,
            )
        except Exception as exc:
            logger.exception("unsupervised suite failed")
            unsup_summary = {
                "method": "unsupervised_suite_v1",
                "scores": {},
                "error": str(exc)[:800],
            }
    unsup_seconds = time.time() - unsup_started

    sup_started = time.time()
    local_summary: Dict[str, Any] = {"method": "local_auto_metrics", "computed": 0}
    with_ref = [
        row
        for row in qa_rows
        if isinstance(row.get("ref_answer"), str) and str(row["ref_answer"]).strip()
    ]

    local_by_id: Dict[str, Dict[str, float]] = {}
    if with_ref:
        local_inputs: List[Dict[str, Any]] = []
        for row in with_ref:
            local_inputs.append(
                {
                    "id": row["id"],
                    "question": row.get("question") or "",
                    "answer": row.get("answer") or "",
                    "source_fact_text": str(row.get("ref_answer") or ""),
                }
            )

        computed = 0
        for index, part in enumerate(_chunk(local_inputs, int(AUTO_EVAL_MAX_ITEMS_PER_REQUEST))):
            try:
                res = execute_local_evaluation_blocking(
                    part,
                    use_local_models=True,
                    gpu_job_id=gpu_job_id or job_id,
                )
            except Exception as exc:
                logger.exception("local evaluation failed: chunk %s", index + 1)
                local_summary.setdefault("errors", []).append(str(exc)[:800])
                continue
            if not isinstance(res, dict) or not isinstance(res.get("results"), list):
                continue
            for row in res.get("results") or []:
                if not isinstance(row, dict):
                    continue
                rid = str(row.get("id") or "").strip()
                if not rid:
                    continue
                local_by_id[rid] = _extract_local_scores(row)
                computed += 1
        local_summary["computed"] = computed
    local_seconds = time.time() - sup_started

    for row in qa_rows:
        rid = str(row.get("id") or "")
        if rid in local_by_id:
            row["evaluation"] = {"llm": None, "local": {"scores": local_by_id[rid]}}
        else:
            row["evaluation"] = {"llm": None, "local": None}

        ue = row.get("unsupervised_evaluation") or {}
        scores = ue.get("scores") if isinstance(ue, dict) else {}
        uf1 = scores.get("unsupervised_f1") if isinstance(scores, dict) else None
        row["average_score"] = _safe_float(uf1, 0.0)
        row["evaluation_method"] = "unsupervised_f1"

    scored_path = os.path.join(outputs_dir, f"eval_job_{job_id}_scored.jsonl")
    summary_path = os.path.join(outputs_dir, f"eval_job_{job_id}_summary.json")

    unsup_scores = _unsupervised_scores_from_suite_summary(unsup_summary)
    minimal_summary_line = {
        "mode": "unsupervised",
        "scores": {
            "faithfulness": float(unsup_scores["faithfulness"]),
            "answerability": float(unsup_scores["answerability"]),
            "coverage_recall_soft": float(unsup_scores["coverage_recall_soft"]),
            "coverage_self": float(unsup_scores["coverage_self"]),
            "coverage_score": float(unsup_scores["coverage_score"]),
            "unsupervised_f1": float(unsup_scores["unsupervised_f1"]),
        },
    }

    persisted_rows: List[Dict[str, Any]] = []
    for row in qa_rows:
        out = dict(row)
        if out.get("ref_answer") is None:
            out.pop("ref_answer", None)
        persisted_rows.append(out)
    _write_jsonl(scored_path, [*persisted_rows, {"id": "__SUMMARY__", "summary": minimal_summary_line}])

    supervised_avg: Dict[str, float] = {}
    if local_by_id:
        metric_vals: Dict[str, List[float]] = {metric: [] for metric in LOCAL_EVALUATION_METRICS}
        for scores in local_by_id.values():
            for metric in LOCAL_EVALUATION_METRICS:
                value = scores.get(metric)
                if isinstance(value, (int, float)):
                    metric_vals[metric].append(float(value))
        for metric, values in metric_vals.items():
            supervised_avg[metric] = float(sum(values) / len(values)) if values else 0.0

    summary_payload: Dict[str, Any] = {
        "job_id": job_id,
        "dataset_name": dataset_name,
        "task_id": task_id,
        "input": {
            "input_files_count": len(normalized_inputs),
            "schema_consistent": True,
            "shared_columns": [
                column
                for column in (baseline_columns or [])
                if column in set(shared_column_set or set())
            ],
            "files": file_summaries,
        },
        "mapping": {
            "question_field": question_field,
            "answer_field": answer_field,
            "context_field": context_field,
            "ref_answer_field": ref_answer_field,
            "id_field": id_field,
            "original_filename_field": original_filename_field,
        },
        "counts": {
            "total": len(qa_rows),
            "with_ref_answer": len(with_ref),
            "input_files": len(normalized_inputs),
        },
        "performance": {
            "unsupervised_batch_size": unsupervised_batch_size,
        },
        "unsupervised": {
            "models": {
                "faithfulness_nli_model": str(faithfulness_nli_model or "") or "auto",
                "answerability_qa_model": str(answerability_qa_model or "") or "auto",
                "coverage_embedding_model": str(coverage_embedding_model or "") or "auto",
            },
            "scores": minimal_summary_line["scores"],
            "details": unsup_summary,
        },
        "supervised": {
            "computed": local_summary.get("computed", 0),
            "scores_avg": supervised_avg,
            "details": local_summary,
        },
        "files": {
            "scored_jsonl": os.path.relpath(scored_path, start=".").replace("\\", "/"),
            "summary_json": os.path.relpath(summary_path, start=".").replace("\\", "/"),
        },
        "timing": {
            "parse_seconds": float(parse_seconds),
            "unsupervised_seconds": float(unsup_seconds),
            "supervised_seconds": float(local_seconds),
            "total_seconds": float(time.time() - started_at),
        },
    }
    _write_json(summary_path, summary_payload)
    return summary_payload


__all__ = ["evaluate_dataset_job"]
