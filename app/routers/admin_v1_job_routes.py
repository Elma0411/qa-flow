# 文件作用：提供管理端评测作业、无监督评测作业和作业状态接口。
# 关联说明：依赖 admin_v1_common 和 eval/unsupervised 服务，专注后台评测作业管理。

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from app.core.config import (
    AUTO_EVAL_MAX_ITEMS_PER_REQUEST,
    LLM_EVALUATION_METRICS,
    LOCAL_EVALUATION_METRICS,
)
from app.core.logger import logger
from app.routers.admin_v1_common import (
    AutoFilterConfig,
    EvaluationJobRequest,
    ExportRequest,
    UnsupervisedEvaluationJobRequest,
    _compute_llm_average,
    _ensure_unsupervised_fields_available,
    _extract_llm_scores_reasons,
    _extract_local_scores,
)
from app.services import admin as admin_qa_service
from app.services.admin import (
    cancel_job,
    complete_job,
    create_job,
    fail_job,
    get_job,
    set_task,
    start_job,
    update_job,
)
from app.services.admin import get_meta_map
from app.services.evaluation import (
    chunked_list,
    execute_local_evaluation_blocking,
    run_llm_evaluation_batches,
)
from app.services.gpu import admit_gpu_job, release_gpu_job
from app.services.unsupervised_evaluation import (
    UNSUPERVISED_EVALUATION_AVAILABLE,
    execute_unsupervised_suite_blocking,
)

router = APIRouter()

@router.post("/unsupervised-evaluation-jobs")
async def create_unsupervised_evaluation_job(payload: UnsupervisedEvaluationJobRequest) -> Dict[str, Any]:
    ids = [str(x) for x in (payload.selection.ids or []) if x]
    if not ids:
        raise HTTPException(status_code=400, detail="selection.ids 不能为空")
    if not UNSUPERVISED_EVALUATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="无监督评价不可用：缺少 transformers/torch 依赖")

    _ensure_unsupervised_fields_available()

    job = create_job(
        "unsupervised_evaluation",
        total=len(ids),
        message="queued",
        params={
            "selection_total": len(ids),
            "force": payload.force,
            "write_back": payload.write_back,
            "include_inactive": payload.include_inactive,
        },
    )
    admit_info = admit_gpu_job(job.job_id, job_type="admin_unsupervised_evaluation")
    if not bool(admit_info.get("accepted")):
        fail_job(job.job_id, error=str(admit_info.get("reason") or "gpu_queue_full"))
        raise HTTPException(status_code=429, detail="GPU 任务排队已满，请稍后重试")

    async def _runner() -> None:
        try:
            start_job(job.job_id, total=len(ids), message="running")
            update_job(job.job_id, append_log=f"selected ids: {len(ids)}")

            selected_ids = list(ids)
            excluded_inactive: List[str] = []
            if not payload.include_inactive:
                meta_map = get_meta_map(selected_ids)
                selected_ids = [
                    qa_id
                    for qa_id in selected_ids
                    if (not meta_map.get(qa_id)) or bool(meta_map[qa_id].is_active)
                ]
                excluded_inactive = [x for x in ids if x not in selected_ids]
                if excluded_inactive:
                    update_job(job.job_id, append_log=f"excluded inactive: {len(excluded_inactive)}")

            records = admin_qa_service.fetch_records_by_ids(selected_ids)
            missing_ids = [x for x in selected_ids if x not in records]
            if missing_ids:
                update_job(job.job_id, append_log=f"missing ids: {len(missing_ids)}")

            to_eval: List[str] = []
            skipped_existing: List[str] = []
            for qa_id in selected_ids:
                rec = records.get(qa_id)
                if not rec:
                    continue
                already = False
                current_unsup_method = str(rec.get("unsupervised_method") or "").strip()
                if current_unsup_method == "unsupervised_suite_v1":
                    raw_scores = rec.get("unsupervised_scores")
                    parsed_scores: Dict[str, Any] = {}
                    if isinstance(raw_scores, dict):
                        parsed_scores = raw_scores
                    elif isinstance(raw_scores, str) and raw_scores.strip():
                        try:
                            parsed_scores = json.loads(raw_scores)
                        except Exception:
                            parsed_scores = {}
                    required_keys = {"faithfulness", "answerability", "coverage_recall_soft", "coverage_self", "coverage_score", "unsupervised_f1"}
                    already = bool(required_keys.issubset(set(parsed_scores.keys())))
                if already and not payload.force:
                    skipped_existing.append(qa_id)
                else:
                    to_eval.append(qa_id)

            update_job(
                job.job_id,
                processed=len(skipped_existing),
                append_log=f"skipped existing: {len(skipped_existing)}; to_eval: {len(to_eval)}",
            )

            updated_records: List[Dict[str, Any]] = []
            updated_ids: List[str] = []
            artifact_rows: List[Dict[str, Any]] = []

            chunks = chunked_list(to_eval, 64)
            for idx, chunk in enumerate(chunks):
                update_job(job.job_id, append_log=f"unsupervised eval chunk {idx + 1}/{len(chunks)} ({len(chunk)})")
                qa_for_eval: List[Dict[str, Any]] = []
                for qa_id in chunk:
                    rec = records.get(qa_id) or {}
                    qa_for_eval.append(
                        {
                            "id": qa_id,
                            "question": rec.get("question") or "",
                            "answer": rec.get("answer") or "",
                            "source_fact_text": rec.get("source_fact_text") or "",
                            "question_type": rec.get("question_type") or "",
                            "is_augmented": bool(rec.get("is_augmented", False)),
                        }
                    )

                if qa_for_eval:
                    summary = await asyncio.to_thread(
                        execute_unsupervised_suite_blocking,
                        qa_for_eval,
                        only_primary=False,
                        prune_item_details=False,
                        gpu_job_id=job.job_id,
                    )
                    update_job(job.job_id, append_log=f"chunk summary: {summary}")

                for row in qa_for_eval:
                    qa_id = str(row.get("id") or "")
                    ue = row.get("unsupervised_evaluation")
                    if not qa_id or not isinstance(ue, dict):
                        continue
                    rec = records.get(qa_id)
                    if not rec:
                        continue

                    method = str(ue.get("method") or "")
                    scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
                    meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}
                    faith_raw = (scores or {}).get("faithfulness")
                    faithfulness = -1.0
                    if isinstance(faith_raw, (int, float, str)):
                        try:
                            faithfulness = float(faith_raw)
                        except Exception:
                            faithfulness = -1.0

                    rec["unsupervised_method"] = method
                    rec["unsupervised_scores"] = json.dumps(scores or {}, ensure_ascii=False)
                    rec["unsupervised_meta"] = json.dumps(meta or {}, ensure_ascii=False)
                    rec["faithfulness"] = faithfulness
                    current_method = str(rec.get("evaluation_method") or "").strip().lower()
                    if not current_method or current_method in {
                        "faithfulness",
                        "answerability",
                        "unsupervised",
                        "unsupervised_f1",
                    }:
                        score_key = "unsupervised_f1"
                        score_value = (scores or {}).get(score_key)
                        method_name = "unsupervised_f1"
                        if not isinstance(score_value, (int, float, str)):
                            score_key = "faithfulness"
                            score_value = (scores or {}).get(score_key)
                            method_name = "faithfulness"
                        if not isinstance(score_value, (int, float, str)):
                            score_key = "answerability"
                            score_value = (scores or {}).get(score_key)
                            method_name = "answerability"
                        try:
                            rec["average_score"] = float(score_value) if score_value is not None else -1.0
                        except Exception:
                            rec["average_score"] = -1.0
                        rec["evaluation_method"] = method_name

                    updated_records.append(rec)
                    updated_ids.append(qa_id)
                    artifact_rows.append(row)

                update_job(job.job_id, processed=len(skipped_existing) + len(updated_ids))

            if payload.write_back and updated_records:
                update_job(job.job_id, append_log=f"write-back replace records: {len(updated_records)}")
                await asyncio.to_thread(admin_qa_service.replace_records, updated_records)

            artifact_path: Optional[str] = None
            try:
                if artifact_rows:
                    artifact_path = admin_qa_service.export_items_to_json(
                        artifact_rows, prefix=f"unsupervised_job_{job.job_id}"
                    )
            except Exception as exc:
                update_job(job.job_id, append_log=f"artifact export skipped: {exc}")

            processed = len(skipped_existing) + len(updated_ids) + len(missing_ids)
            update_job(job.job_id, processed=processed)

            result = {
                "selection_total": len(ids),
                "excluded_inactive": excluded_inactive,
                "missing_ids": missing_ids,
                "skipped_existing": skipped_existing,
                "evaluated": updated_ids if payload.write_back else [r.get("id") for r in artifact_rows if r.get("id")],
                "write_back": payload.write_back,
                "artifact_path": artifact_path,
            }
            complete_job(job.job_id, result=result, message="done")
        except asyncio.CancelledError:
            logger.warning("admin unsupervised evaluation job canceled: %s", job.job_id)
            update_job(job.job_id, status="canceled", message="canceled")
            raise
        except Exception as exc:
            logger.exception("admin unsupervised evaluation job failed")
            fail_job(job.job_id, error=str(exc))
        finally:
            release_gpu_job(job.job_id)

    task = asyncio.create_task(_runner())
    set_task(job.job_id, task)
    return {"job_id": job.job_id}



@router.post("/evaluation-jobs")
async def create_evaluation_job(payload: EvaluationJobRequest) -> Dict[str, Any]:
    ids = [str(x) for x in (payload.selection.ids or []) if x]
    if not ids:
        raise HTTPException(status_code=400, detail="selection.ids 不能为空")

    criteria_list = payload.criteria_list or list(LLM_EVALUATION_METRICS)
    criteria_list = [c for c in criteria_list if isinstance(c, str) and c.strip()]
    if payload.evaluation_method == "llm" and not criteria_list:
        raise HTTPException(status_code=400, detail="criteria_list 不能为空（llm）")

    auto_filter_cfg = payload.auto_filter or AutoFilterConfig(enabled=False, score_threshold=0.7)
    threshold = float(auto_filter_cfg.score_threshold or 0.0)
    if payload.evaluation_method == "llm":
        criteria_desc = ",".join(criteria_list) if criteria_list else ""
        msg = f"queued: method=llm; criteria={criteria_desc}; auto_filter={auto_filter_cfg.enabled}; threshold={threshold}"
    else:
        msg = f"queued: method=local; metrics=fixed; auto_filter={auto_filter_cfg.enabled}; threshold={threshold}"
    job = create_job(
        "evaluation",
        total=len(ids),
        message=msg,
        params={
            "selection_total": len(ids),
            "evaluation_method": payload.evaluation_method,
            "criteria_list": criteria_list if payload.evaluation_method == "llm" else None,
            "local_metrics": list(LOCAL_EVALUATION_METRICS) if payload.evaluation_method == "local" else None,
            "force": payload.force,
            "write_back": payload.write_back,
            "include_inactive": payload.include_inactive,
            "auto_filter": {"enabled": auto_filter_cfg.enabled, "score_threshold": threshold},
        },
    )
    if payload.evaluation_method == "local":
        admit_info = admit_gpu_job(job.job_id, job_type="admin_local_evaluation")
        if not bool(admit_info.get("accepted")):
            fail_job(job.job_id, error=str(admit_info.get("reason") or "gpu_queue_full"))
            raise HTTPException(status_code=429, detail="GPU 任务排队已满，请稍后重试")

    async def _runner() -> None:
        try:
            start_job(job.job_id, total=len(ids), message=job.message.replace("queued:", "running:"))
            update_job(job.job_id, append_log=f"selected ids: {len(ids)}")

            selected_ids = list(ids)
            excluded_inactive: List[str] = []
            if not payload.include_inactive:
                meta_map = get_meta_map(selected_ids)
                active_ids: List[str] = []
                for qa_id in selected_ids:
                    meta = meta_map.get(qa_id)
                    is_active = True if not meta else bool(meta.is_active)
                    if is_active:
                        active_ids.append(qa_id)
                    else:
                        excluded_inactive.append(qa_id)
                selected_ids = active_ids
                update_job(job.job_id, append_log=f"excluded inactive: {len(excluded_inactive)}")

            records = admin_qa_service.fetch_records_by_ids(selected_ids)
            missing_ids = [qa_id for qa_id in selected_ids if qa_id not in records]
            if missing_ids:
                update_job(job.job_id, append_log=f"missing ids: {len(missing_ids)}")

            to_consider = [qa_id for qa_id in selected_ids if qa_id in records]
            to_eval: List[str] = []
            skipped_evaluated: List[str] = []
            for qa_id in to_consider:
                rec = records[qa_id]
                already = bool((rec.get("evaluation_method") or "").strip())
                if already and not payload.force:
                    skipped_evaluated.append(qa_id)
                else:
                    to_eval.append(qa_id)

            update_job(
                job.job_id,
                total=len(to_consider) + len(missing_ids),
                processed=len(skipped_evaluated),
                append_log=f"skipped evaluated: {len(skipped_evaluated)}; to_eval: {len(to_eval)}",
            )

            eval_results: List[Dict[str, Any]] = []
            if to_eval:
                qa_for_eval: List[Dict[str, Any]] = []
                for qa_id in to_eval:
                    rec = records[qa_id]
                    qa_for_eval.append(
                        {
                            "id": qa_id,
                            "question": rec.get("question") or "",
                            "answer": rec.get("answer") or "",
                            "source_fact_text": rec.get("source_fact_text") or "",
                            "source": rec.get("source") or rec.get("source_id") or "",
                        }
                    )

                if payload.evaluation_method == "llm":

                    async def _progress(msg: str) -> Any:
                        update_job(job.job_id, append_log=msg)
                        return None

                    llm_res = await run_llm_evaluation_batches(
                        qa_for_eval, criteria_list, _progress
                    )
                    eval_results = llm_res.get("results") or []
                else:
                    # local auto-metrics evaluation; chunk to avoid huge single request
                    chunks = chunked_list(qa_for_eval, AUTO_EVAL_MAX_ITEMS_PER_REQUEST)
                    for idx, chunk in enumerate(chunks):
                        update_job(job.job_id, append_log=f"local eval chunk {idx + 1}/{len(chunks)} ({len(chunk)})")
                        res = await asyncio.to_thread(
                            execute_local_evaluation_blocking,
                            chunk,
                            True,
                            gpu_job_id=job.job_id,
                        )
                        if res and res.get("results"):
                            eval_results.extend(res["results"])

            # write-back
            updated_ids: List[str] = []
            if eval_results and payload.write_back:
                updated_records: List[Dict[str, Any]] = []
                by_id: Dict[str, Dict[str, Any]] = {}
                for r in eval_results:
                    qa_id = str(r.get("id") or "")
                    if qa_id:
                        by_id[qa_id] = r

                for qa_id in to_eval:
                    rec = records.get(qa_id)
                    if not rec:
                        continue
                    ev_row = by_id.get(qa_id)
                    if not ev_row:
                        continue
                    evaluation = ev_row.get("evaluation") or {}
                    if not isinstance(evaluation, dict):
                        evaluation = {}

                    if payload.evaluation_method == "llm":
                        scores, reasons = _extract_llm_scores_reasons(evaluation, criteria_list)
                        avg_score = _compute_llm_average(evaluation, criteria_list)
                        rec["evaluation_method"] = "llm"
                        rec["llm_scores"] = json.dumps(scores, ensure_ascii=False)
                        rec["llm_reasons"] = json.dumps(reasons, ensure_ascii=False)
                        rec["average_score"] = float(avg_score)
                        if auto_filter_cfg.enabled:
                            rec["filtered"] = bool(avg_score >= threshold)
                            rec["filter_basis"] = "llm"
                    else:
                        scores = _extract_local_scores(evaluation)
                        avg_score = float(ev_row.get("average_score") or 0.0)
                        rec["evaluation_method"] = "local"
                        rec["local_scores"] = json.dumps(scores, ensure_ascii=False)
                        rec["average_score"] = float(avg_score)
                        if auto_filter_cfg.enabled:
                            rec["filtered"] = bool(avg_score >= threshold)
                            rec["filter_basis"] = "local"

                    updated_records.append(rec)
                    updated_ids.append(qa_id)

                update_job(job.job_id, append_log=f"write-back replace records: {len(updated_records)}")
                await asyncio.to_thread(admin_qa_service.replace_records, updated_records)
            elif payload.write_back and auto_filter_cfg.enabled and skipped_evaluated:
                # Apply auto_filter to already-evaluated items even when force=false (no re-eval),
                # so users can toggle filtering without paying for another eval run.
                updated_records = []
                for qa_id in skipped_evaluated:
                    rec = records.get(qa_id)
                    if not rec:
                        continue
                    raw_avg = rec.get("average_score")
                    try:
                        avg_score = float(raw_avg) if raw_avg is not None else 0.0
                    except Exception:
                        avg_score = 0.0
                    rec["filtered"] = bool(avg_score >= threshold)
                    method = str(rec.get("evaluation_method") or "").strip().lower()
                    if method in {"llm", "local"}:
                        rec["filter_basis"] = method
                    updated_records.append(rec)
                if updated_records:
                    update_job(
                        job.job_id,
                        append_log=f"auto_filter write-back for skipped_evaluated: {len(updated_records)}",
                    )
                    await asyncio.to_thread(admin_qa_service.replace_records, updated_records)

            processed = len(skipped_evaluated) + len(to_eval) + len(missing_ids)
            update_job(job.job_id, processed=processed)

            artifact_path: Optional[str] = None
            try:
                if eval_results:
                    artifact_path = admin_qa_service.export_items_to_json(
                        eval_results,
                        prefix=f"evaluation_job_{job.job_id}",
                        meta={
                            "job_id": job.job_id,
                            "job_type": job.job_type,
                            "params": job.params or {},
                        },
                    )
            except Exception as exc:
                update_job(job.job_id, append_log=f"artifact export skipped: {exc}")

            result = {
                "selection_total": len(ids),
                "excluded_inactive": excluded_inactive,
                "missing_ids": missing_ids,
                "skipped_evaluated": skipped_evaluated,
                "evaluated": updated_ids if payload.write_back else [str(r.get("id")) for r in eval_results if r.get("id")],
                "write_back": payload.write_back,
                "auto_filter": {"enabled": auto_filter_cfg.enabled, "score_threshold": threshold},
                "artifact_path": artifact_path,
            }
            complete_job(job.job_id, result=result, message="done")
        except asyncio.CancelledError:
            logger.warning("admin evaluation job canceled: %s", job.job_id)
            update_job(job.job_id, status="canceled", message="canceled")
            raise
        except Exception as exc:
            logger.exception("admin evaluation job failed")
            fail_job(job.job_id, error=str(exc))
        finally:
            if payload.evaluation_method == "local":
                release_gpu_job(job.job_id)

    task = asyncio.create_task(_runner())
    set_task(job.job_id, task)
    return {"job_id": job.job_id}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str) -> Dict[str, Any]:
    ok = cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {"success": True, "job_id": job_id}


@router.post("/exports")
async def create_export_job(payload: ExportRequest) -> Dict[str, Any]:
    ids = [str(x) for x in payload.ids if x]
    if not ids:
        raise HTTPException(status_code=400, detail="ids 不能为空")
    job = create_job(
        "export",
        total=len(ids),
        message="queued",
        params={
            "selection_total": len(ids),
            "include_inactive": payload.include_inactive,
        },
    )

    async def _runner() -> None:
        try:
            start_job(job.job_id, total=len(ids), message="running")
            selected_ids = list(ids)
            if not payload.include_inactive:
                meta_map = get_meta_map(selected_ids)
                selected_ids = [
                    qa_id
                    for qa_id in selected_ids
                    if (not meta_map.get(qa_id)) or bool(meta_map[qa_id].is_active)
                ]
            records = admin_qa_service.fetch_records_by_ids(selected_ids)
            meta_map = get_meta_map(selected_ids)
            items: List[Dict[str, Any]] = []
            for qa_id, rec in records.items():
                rec = dict(rec)
                rec.pop("embedding_vector", None)
                meta = meta_map.get(qa_id)
                rec["admin"] = meta.to_dict() if meta else {
                    "id": qa_id,
                    "is_active": True,
                    "review_status": None,
                    "review_note": None,
                    "updated_at": None,
                }
                items.append(rec)
            path = admin_qa_service.export_items_to_json(items, prefix=f"export_{job.job_id}")
            update_job(job.job_id, processed=len(items))
            complete_job(job.job_id, result={"path": path, "count": len(items)}, message="done")
        except asyncio.CancelledError:
            update_job(job.job_id, status="canceled", message="canceled")
            raise
        except Exception as exc:
            fail_job(job.job_id, error=str(exc))

    task = asyncio.create_task(_runner())
    set_task(job.job_id, task)
    return {"job_id": job.job_id}

__all__ = ['router']
