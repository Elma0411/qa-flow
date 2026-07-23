# 文件作用：提供评测数据集任务的提交、查询和结果入库接口。
# 关联说明：对接 app.services.eval_jobs，和 pipeline_evaluation_routes 分别服务数据集评测与单次评价。

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import CONFIG
from app.core.logger import logger
from app.services.admin import (
    cancel_job,
    complete_job,
    create_job,
    delete_job,
    fail_job,
    get_job,
    list_jobs,
    set_task,
    start_job,
    update_job,
)
from app.services.artifacts import (
    delete_artifacts_now,
    delete_paths_now,
    get_owner_artifact_expire_at,
    register_temporary_artifacts,
)
from app.services.eval_jobs import (
    mapping_check,
    parse_file_ranges_json,
    preview_datasets,
)
from app.services.eval_jobs import evaluate_dataset_job, ingest_scored_items_to_milvus, read_scored_items_page
from app.services.gpu import admit_gpu_job, release_gpu_job
from app.services.unsupervised_evaluation import validate_evaluation_model_name


router = APIRouter(prefix="/eval", tags=["eval"])
_ARTIFACT_TTL_SECONDS = 24 * 60 * 60


def _save_upload_to_outputs(upload_file: UploadFile, *, job_id: str, file_index: int) -> Dict[str, Any]:
    outputs_dir = CONFIG["outputs_dir"]
    os.makedirs(outputs_dir, exist_ok=True)
    filename = str(upload_file.filename or "")
    ext = os.path.splitext(filename)[1].lower()
    if not ext or len(ext) > 10:
        ext = ".bin"
    path = os.path.join(outputs_dir, f"eval_job_{job_id}_input_{int(file_index):03d}{ext}")
    try:
        data = upload_file.file.read()
    finally:
        try:
            upload_file.file.seek(0)
        except Exception:
            pass
    with open(path, "wb") as f:
        f.write(data)
    return {
        "file_index": int(file_index),
        "upload_filename": filename or os.path.basename(path),
        "input_path": path,
    }


def _rel_output(path: Optional[str]) -> Optional[str]:
    raw = str(path or "").strip()
    if not raw:
        return None
    return os.path.relpath(raw, start=".").replace("\\", "/")


def _build_eval_artifact_result(summary: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(summary or {})
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    artifact_paths = [files.get("scored_jsonl"), files.get("summary_json")]
    abs_paths = [
        os.path.join(CONFIG["outputs_dir"], os.path.basename(str(path)))
        for path in artifact_paths
        if str(path or "").strip()
    ]
    register_temporary_artifacts(
        owner_kind="eval_job",
        owner_id=str(result.get("job_id") or ""),
        artifact_kind="eval_result",
        paths=abs_paths,
        ttl_seconds=_ARTIFACT_TTL_SECONDS,
    )
    result["history_source"] = "artifacts"
    result["milvus_task_id"] = None
    result["artifacts_deleted"] = False
    result["artifacts_expire_at"] = get_owner_artifact_expire_at("eval_job", str(result.get("job_id") or ""))
    return result


def _collect_eval_artifact_paths(result: Dict[str, Any]) -> List[str]:
    files = result.get("files") if isinstance(result.get("files"), dict) else {}
    paths: List[str] = []
    for key in ("scored_jsonl", "summary_json"):
        raw = str(files.get(key) or "").strip()
        if raw:
            paths.append(os.path.join(CONFIG["outputs_dir"], os.path.basename(raw)))
    return paths


@router.post("/preview")
async def preview_endpoint(
    files: List[UploadFile] = File(...),
    input_format: str = Form("auto"),
    encoding: Optional[str] = Form(None),
    delimiter: str = Form(","),
    sheet_name: Optional[str] = Form(None),
    sample_size: int = Form(5),
    file_ranges_json: Optional[str] = Form(None),
    # mapping (optional, for validation)
    question_field: Optional[str] = Form(None),
    answer_field: Optional[str] = Form(None),
    context_field: Optional[str] = Form(None),
    ref_answer_field: Optional[str] = Form(None),
    id_field: Optional[str] = Form(None),
    original_filename_field: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        file_ranges = parse_file_ranges_json(file_ranges_json, files_count=len(files or []))
        preview = preview_datasets(
            files,
            input_format=input_format,
            encoding=encoding,
            delimiter=delimiter,
            sheet_name=sheet_name,
            sample_size=max(1, int(sample_size or 5)),
            file_ranges_by_index=file_ranges,
            question_field=question_field,
            answer_field=answer_field,
            context_field=context_field,
            ref_answer_field=ref_answer_field,
            id_field=id_field,
            original_filename_field=original_filename_field,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    shared_columns = list(preview.shared_columns or [])
    return {
        "files_count": len(preview.files),
        "schema_consistent": bool(preview.schema_consistent),
        "columns": shared_columns,
        "shared_columns": shared_columns,
        "files": preview.files,
        "warnings": preview.warnings,
        "mapping_check": mapping_check(
            shared_columns,
            question_field=question_field,
            answer_field=answer_field,
            context_field=context_field,
            ref_answer_field=ref_answer_field,
            id_field=id_field,
            original_filename_field=original_filename_field,
        ),
    }


@router.get("/jobs")
async def list_eval_jobs(limit: int = Query(50, ge=1, le=200)) -> Dict[str, Any]:
    jobs = [j.to_dict() for j in list_jobs(int(limit)) if str(j.job_type) in {"eval", "eval_ingest"}]
    return {"jobs": jobs}


@router.post("/jobs")
async def create_eval_job(
    files: List[UploadFile] = File(...),
    dataset_name: str = Form(...),
    question_field: str = Form(...),
    answer_field: str = Form(...),
    context_field: str = Form(...),
    ref_answer_field: Optional[str] = Form(None),
    id_field: Optional[str] = Form(None),
    original_filename_field: Optional[str] = Form(None),
    input_format: str = Form("auto"),
    encoding: Optional[str] = Form(None),
    delimiter: str = Form(","),
    sheet_name: Optional[str] = Form(None),
    file_ranges_json: Optional[str] = Form(None),
    unsupervised_batch_size: Optional[int] = Form(None),
    faithfulness_nli_model: Optional[str] = Form(None),
    answerability_qa_model: Optional[str] = Form(None),
    coverage_embedding_model: Optional[str] = Form(None),
    # faithfulness hypothesis overrides (optional)
    faithfulness_hypothesis_mode: Optional[str] = Form(None),
    faithfulness_hypothesis_timeout: Optional[int] = Form(None),
    faithfulness_hypothesis_max_retries: Optional[int] = Form(None),
    faithfulness_hypothesis_max_concurrency: Optional[int] = Form(None),
) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="至少需要上传一个文件")
    ds = str(dataset_name or "").strip()
    if not ds:
        raise HTTPException(status_code=400, detail="dataset_name 不能为空")

    try:
        file_ranges = parse_file_ranges_json(file_ranges_json, files_count=len(files))
        preview = preview_datasets(
            files,
            input_format=input_format,
            encoding=encoding,
            delimiter=delimiter,
            sheet_name=sheet_name,
            sample_size=1,
            file_ranges_by_index=file_ranges,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not preview.schema_consistent:
        mismatch_files = [
            {
                "file_index": item.get("file_index"),
                "filename": item.get("filename"),
                "missing_columns": item.get("missing_columns") or [],
                "extra_columns": item.get("extra_columns") or [],
            }
            for item in preview.files
            if str(item.get("schema_status") or "") != "ok"
        ]
        raise HTTPException(
            status_code=400,
            detail=f"批量评测要求所有文件字段完全一致: {json.dumps(mismatch_files, ensure_ascii=False)}",
        )

    mapping_status = mapping_check(
        preview.shared_columns,
        question_field=question_field,
        answer_field=answer_field,
        context_field=context_field,
        ref_answer_field=ref_answer_field,
        id_field=id_field,
        original_filename_field=original_filename_field,
    )
    missing_required = [
        info.get("value")
        for info in mapping_status.values()
        if isinstance(info, dict) and info.get("required") and info.get("provided") and not info.get("exists")
    ]
    if missing_required:
        raise HTTPException(
            status_code=400,
            detail=f"映射字段不存在于共享列中: {missing_required}",
        )

    try:
        resolved_nli_model = validate_evaluation_model_name(
            faithfulness_nli_model,
            kind="faithfulness_nli",
        )
        resolved_qa_model = validate_evaluation_model_name(
            answerability_qa_model,
            kind="answerability_qa",
        )
        resolved_coverage_model = validate_evaluation_model_name(
            coverage_embedding_model,
            kind="coverage_embedding",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = create_job(
        "eval",
        total=0,
        message="queued",
        params={
            "dataset_name": ds,
            "input_format": input_format,
            "encoding": encoding,
            "delimiter": delimiter,
            "sheet_name": sheet_name,
            "question_field": question_field,
            "answer_field": answer_field,
            "context_field": context_field,
            "ref_answer_field": ref_answer_field,
            "id_field": id_field,
            "original_filename_field": original_filename_field,
            "input_files_count": len(files),
            "input_filenames": [str(f.filename or f"file_{idx}") for idx, f in enumerate(files)],
            "file_ranges": [
                {
                    "file_index": int(idx),
                    "row_start": cfg.get("row_start"),
                    "row_end": cfg.get("row_end"),
                }
                for idx, cfg in sorted(file_ranges.items(), key=lambda item: item[0])
            ],
            "unsupervised_batch_size": unsupervised_batch_size,
            "faithfulness_nli_model": resolved_nli_model,
            "answerability_qa_model": resolved_qa_model,
            "coverage_embedding_model": resolved_coverage_model,
            "faithfulness_hypothesis_mode": faithfulness_hypothesis_mode,
            "faithfulness_hypothesis_timeout": faithfulness_hypothesis_timeout,
            "faithfulness_hypothesis_max_retries": faithfulness_hypothesis_max_retries,
            "faithfulness_hypothesis_max_concurrency": faithfulness_hypothesis_max_concurrency,
        },
    )
    admit_info = admit_gpu_job(job.job_id, job_type="eval")
    if not bool(admit_info.get("accepted")):
        fail_job(job.job_id, error=str(admit_info.get("reason") or "gpu_queue_full"))
        raise HTTPException(
            status_code=429,
            detail="GPU 任务排队已满，请稍后重试",
        )

    try:
        saved_inputs = [
            {
                **_save_upload_to_outputs(upload_file, job_id=job.job_id, file_index=file_index),
                "row_start": (file_ranges.get(file_index) or {}).get("row_start"),
                "row_end": (file_ranges.get(file_index) or {}).get("row_end"),
            }
            for file_index, upload_file in enumerate(files)
        ]
    except Exception as exc:
        cancel_job(job.job_id)
        release_gpu_job(job.job_id)
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {exc}")

    async def _runner() -> None:
        try:
            start_job(job.job_id, message="running")
            for input_item in saved_inputs:
                update_job(
                    job.job_id,
                    append_log=f"saved input[{input_item['file_index']}]: {os.path.basename(input_item['input_path'])}",
                )
            summary = await asyncio.to_thread(
                evaluate_dataset_job,
                job_id=job.job_id,
                input_files=saved_inputs,
                dataset_name=ds,
                question_field=question_field,
                answer_field=answer_field,
                context_field=context_field,
                ref_answer_field=ref_answer_field,
                id_field=id_field,
                original_filename_field=original_filename_field,
                input_format=input_format,
                encoding=encoding,
                delimiter=delimiter,
                sheet_name=sheet_name,
                unsupervised_batch_size=unsupervised_batch_size,
                faithfulness_nli_model=resolved_nli_model,
                answerability_qa_model=resolved_qa_model,
                coverage_embedding_model=resolved_coverage_model,
                faithfulness_hypothesis_mode=faithfulness_hypothesis_mode,
                faithfulness_hypothesis_timeout=faithfulness_hypothesis_timeout,
                faithfulness_hypothesis_max_retries=faithfulness_hypothesis_max_retries,
                faithfulness_hypothesis_max_concurrency=faithfulness_hypothesis_max_concurrency,
                gpu_job_id=job.job_id,
            )
            update_job(job.job_id, processed=int((summary.get("counts") or {}).get("total") or 0))
            complete_job(job.job_id, result=_build_eval_artifact_result(summary), message="done")
        except asyncio.CancelledError:
            logger.warning("eval job canceled: %s", job.job_id)
            update_job(job.job_id, status="canceled", message="canceled")
            raise
        except Exception as exc:
            logger.exception("eval job failed: %s", job.job_id)
            fail_job(job.job_id, error=str(exc))
        finally:
            delete_paths_now(
                [item.get("input_path") for item in saved_inputs if isinstance(item, dict)],
                reason="eval_input_terminal",
            )
            release_gpu_job(job.job_id)

    task = asyncio.create_task(_runner())
    set_task(job.job_id, task)

    return {"job_id": job.job_id, "status_url": f"/eval/jobs/{job.job_id}"}


@router.get("/jobs/{job_id}")
async def get_eval_job(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job or str(job.job_type) not in {"eval", "eval_ingest"}:
        raise HTTPException(status_code=404, detail="job 不存在")
    return job.to_dict()


@router.delete("/jobs/{job_id}")
async def cancel_eval_job(job_id: str) -> Dict[str, Any]:
    ok = cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {"success": True, "job_id": job_id}


@router.delete("/jobs/{job_id}/history")
async def delete_eval_job_history(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job or str(job.job_type) not in {"eval", "eval_ingest"}:
        raise HTTPException(status_code=404, detail="job 不存在")
    if str(job.status) not in {"succeeded", "failed", "canceled"}:
        raise HTTPException(status_code=409, detail="任务仍在运行，不能删除历史记录")
    result = job.result if isinstance(job.result, dict) else {}
    artifact_paths = _collect_eval_artifact_paths(result)
    delete_artifacts_now(
        owner_kind="eval_job",
        owner_id=job_id,
        reason="history_deleted",
    )
    deleted_paths = delete_paths_now(artifact_paths, reason="eval_history_deleted")
    ok = delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job 不存在")
    return {"success": True, "job_id": job_id, "deleted_artifacts": len(deleted_paths)}


@router.get("/jobs/{job_id}/download")
async def get_eval_job_download_links(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job or not isinstance(job.result, dict):
        raise HTTPException(status_code=404, detail="job 不存在或尚未完成")
    files = job.result.get("files") if isinstance(job.result.get("files"), dict) else {}
    scored = files.get("scored_jsonl")
    summary = files.get("summary_json")
    scored_name = os.path.basename(str(scored)) if scored else None
    summary_name = os.path.basename(str(summary)) if summary else None
    return {
        "job_id": job_id,
        "history_source": job.result.get("history_source"),
        "milvus_task_id": job.result.get("milvus_task_id"),
        "files": files,
        "download": {
            "scored_jsonl": f"/download/outputs/{scored_name}" if scored_name else None,
            "summary_json": f"/download/outputs/{summary_name}" if summary_name else None,
        },
    }


@router.get("/jobs/{job_id}/result")
async def get_eval_job_result(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    include_details: bool = Query(False),
) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job or not isinstance(job.result, dict):
        raise HTTPException(status_code=404, detail="job 不存在或尚未完成")
    summary = dict(job.result)
    files = summary.get("files") if isinstance(summary.get("files"), dict) else {}
    scored = files.get("scored_jsonl")
    if not scored:
        history_source = str(summary.get("history_source") or "").strip().lower()
        milvus_task_id = str(summary.get("milvus_task_id") or "").strip()
        if history_source == "milvus" and milvus_task_id:
            if not include_details:
                if isinstance(summary.get("unsupervised"), dict):
                    summary["unsupervised"] = {k: v for k, v in summary["unsupervised"].items() if k != "details"}
                if isinstance(summary.get("supervised"), dict):
                    summary["supervised"] = {k: v for k, v in summary["supervised"].items() if k != "details"}
            return {
                "job_id": job_id,
                "summary": summary,
                "total": 0,
                "items": [],
                "history_redirect": {"source": "milvus", "task_id": milvus_task_id},
            }
        raise HTTPException(status_code=404, detail="scored 文件不存在")
    scored_path = os.path.join(CONFIG["outputs_dir"], os.path.basename(scored))
    if not os.path.exists(scored_path):
        history_source = str(summary.get("history_source") or "").strip().lower()
        milvus_task_id = str(summary.get("milvus_task_id") or "").strip()
        if history_source == "milvus" and milvus_task_id:
            if not include_details:
                if isinstance(summary.get("unsupervised"), dict):
                    summary["unsupervised"] = {k: v for k, v in summary["unsupervised"].items() if k != "details"}
                if isinstance(summary.get("supervised"), dict):
                    summary["supervised"] = {k: v for k, v in summary["supervised"].items() if k != "details"}
            return {
                "job_id": job_id,
                "summary": summary,
                "total": 0,
                "items": [],
                "history_redirect": {"source": "milvus", "task_id": milvus_task_id},
            }
        raise HTTPException(status_code=404, detail=f"scored 文件不存在: {scored}")

    items, total = await asyncio.to_thread(
        read_scored_items_page,
        scored_path,
        offset=offset,
        limit=limit,
        threshold=threshold,
    )

    if not include_details:
        if isinstance(summary.get("unsupervised"), dict):
            summary["unsupervised"] = {k: v for k, v in summary["unsupervised"].items() if k != "details"}
        if isinstance(summary.get("supervised"), dict):
            summary["supervised"] = {k: v for k, v in summary["supervised"].items() if k != "details"}
    return {"job_id": job_id, "summary": summary, "total": total, "items": items}


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(..., description="数据集名称（用于 task_id 与 ID 命名空间）")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="unsupervised_f1 阈值")
    enable_vector_storage: bool = Field(True, description="是否写入 Milvus 向量字段")


@router.post("/jobs/{job_id}/ingest")
async def ingest_eval_job(job_id: str, payload: IngestRequest) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job or not isinstance(job.result, dict):
        raise HTTPException(status_code=404, detail="job 不存在或尚未完成")
    if str(job.result.get("history_source") or "").strip().lower() == "milvus":
        raise HTTPException(status_code=409, detail="该评测结果已入库；如需更换阈值，请重跑评测")
    files = job.result.get("files") if isinstance(job.result.get("files"), dict) else {}
    scored = files.get("scored_jsonl")
    if not scored:
        raise HTTPException(status_code=404, detail="scored 文件不存在")
    scored_path = os.path.join(CONFIG["outputs_dir"], os.path.basename(scored))
    if not os.path.exists(scored_path):
        raise HTTPException(status_code=404, detail=f"scored 文件不存在: {scored}")

    res = await asyncio.to_thread(
        ingest_scored_items_to_milvus,
        scored_path,
        dataset_name=payload.dataset_name,
        threshold=payload.threshold,
        enable_vector_storage=payload.enable_vector_storage,
        job_id=job_id,
    )
    milvus_info = res.get("milvus") if isinstance(res.get("milvus"), dict) else {}
    consolidated_file = res.get("consolidated_file")
    consolidated_abs = (
        os.path.join(CONFIG["outputs_dir"], os.path.basename(str(consolidated_file)))
        if str(consolidated_file or "").strip()
        else None
    )
    if bool(milvus_info.get("success")):
        delete_paths_now(
            [
                scored_path,
                os.path.join(CONFIG["outputs_dir"], os.path.basename(str(files.get("summary_json") or "")))
                if str(files.get("summary_json") or "").strip()
                else None,
                consolidated_abs,
            ],
            reason="milvus_ingested",
        )
        updated_result = dict(job.result)
        updated_result["history_source"] = "milvus"
        updated_result["milvus_task_id"] = str(res.get("milvus_task_id") or res.get("task_id") or "")
        updated_result["artifacts_deleted"] = True
        updated_result["artifacts_expire_at"] = None
        updated_result["files"] = {"scored_jsonl": None, "summary_json": None}
        update_job(job_id, result=updated_result, append_log=f"milvus_task_id={updated_result['milvus_task_id']}")
        res["history_source"] = "milvus"
        res["artifacts_deleted"] = True
    else:
        register_temporary_artifacts(
            owner_kind="eval_ingest",
            owner_id=job_id,
            artifact_kind="eval_ingest_consolidated",
            paths=[consolidated_abs] if consolidated_abs else [],
            ttl_seconds=_ARTIFACT_TTL_SECONDS,
        )
        res["history_source"] = "artifacts"
        res["artifacts_deleted"] = False
    return res
