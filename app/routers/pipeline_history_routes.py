# 文件作用：提供流水线任务状态、历史记录、产物下载和清理接口。
# 关联说明：复用 pipeline_state、storage、artifacts 服务，管理 pipeline 任务和产物。

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import ACTIVE_BATCH_JOBS, CONFIG
from app.core.time_utils import now_server_local_iso
from app.routers.pipeline_common import _collect_pipeline_output_artifact_paths
from app.services.artifacts import (
    delete_artifacts_now,
    delete_paths_now,
    get_owner_artifact_expire_at,
)
from app.services.milvus import store_qa_payload_to_milvus
from app.services.pipeline_state import (
    delete_pipeline_task_status,
    get_pipeline_store_path,
    get_pipeline_task_status,
    list_pipeline_task_statuses,
    upsert_pipeline_task_status,
)
from app.services.storage import sanitize_filename

router = APIRouter()


class IngestSelectedQARequest(BaseModel):
    source_file: Optional[str] = None
    selected_ids: List[str] = Field(default_factory=list)
    select_all_task: bool = False


def _outputs_dir() -> str:
    return os.path.abspath(str(CONFIG["outputs_dir"]))


def _registered_output_path(raw_path: Any) -> Optional[str]:
    basename = os.path.basename(str(raw_path or "").strip())
    if not basename:
        return None
    base = _outputs_dir()
    full_path = os.path.abspath(os.path.join(base, basename))
    if os.path.commonpath([base, full_path]) != base:
        return None
    return full_path


def _artifact_not_expired(task_id: str) -> Tuple[bool, Optional[int]]:
    expire_at = get_owner_artifact_expire_at("pipeline_task", task_id)
    if expire_at is not None and int(expire_at) <= int(time.time()):
        return False, int(expire_at)
    return True, int(expire_at) if expire_at is not None else None


def _iter_task_outputs(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = status.get("outputs") if isinstance(status.get("outputs"), list) else []
    return [item for item in outputs if isinstance(item, dict)]


def _source_matches(output: Dict[str, Any], source_file: Optional[str]) -> bool:
    wanted = str(source_file or "").strip()
    if not wanted:
        return True
    candidates = [
        output.get("source_file"),
        output.get("core_file"),
        output.get("filename"),
    ]
    values = output.get("source_files")
    if isinstance(values, list):
        candidates.extend(values)
    return any(str(value or "").strip() == wanted for value in candidates)


def _registered_debug_paths(status: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    seen = set()
    for output in _iter_task_outputs(status):
        for raw in [output.get("debug_jsonl")]:
            path = _registered_output_path(raw)
            if path and path not in seen:
                seen.add(path)
                paths.append(path)
        values = output.get("debug_json_files")
        if isinstance(values, list):
            for raw in values:
                path = _registered_output_path(raw)
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)
    return paths


def _find_consolidated_output(
    status: Dict[str, Any],
    source_file: Optional[str] = None,
) -> Tuple[int, Dict[str, Any], str]:
    fallback: Optional[Tuple[int, Dict[str, Any], str]] = None
    for idx, output in enumerate(_iter_task_outputs(status)):
        path = _registered_output_path(output.get("consolidated_json"))
        if not path:
            continue
        current = (idx, output, path)
        if fallback is None:
            fallback = current
        if _source_matches(output, source_file):
            return current
    if fallback and not str(source_file or "").strip():
        return fallback
    raise HTTPException(
        status_code=410,
        detail="当前任务的临时合并 JSON 不存在或已过期，无法人工入库；请重新生成任务。",
    )


def _load_registered_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise HTTPException(
            status_code=410,
            detail="临时合并 JSON 已被清理，无法人工入库；请重新生成任务。",
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取合并 JSON 失败：{str(exc)}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="合并 JSON 格式无效，无法入库。")
    return payload


def _update_manual_ingest_status(
    *,
    status: Dict[str, Any],
    task_id: str,
    output_index: int,
    vector_result: Dict[str, Any],
    selected_count: int,
    select_all_task: bool,
) -> Dict[str, Any]:
    updated = dict(status or {})
    outputs = [dict(item) for item in _iter_task_outputs(status)]
    if 0 <= output_index < len(outputs):
        record = dict(outputs[output_index])
        record["history_source"] = "milvus"
        record["milvus_task_id"] = task_id
        record["vector_storage_result"] = dict(vector_result or {})
        record["manual_ingest"] = True
        record["manual_ingest_selected_count"] = int(selected_count)
        record["manual_ingest_select_all"] = bool(select_all_task)
        record["manual_ingested_at"] = now_server_local_iso()
        record["artifacts_deleted"] = False
        record["artifacts_expire_at"] = get_owner_artifact_expire_at("pipeline_task", task_id)
        outputs[output_index] = record
    updated["outputs"] = outputs
    updated["history_source"] = "milvus"
    updated["milvus_task_id"] = task_id
    updated["artifacts_deleted"] = False
    updated["artifacts_expire_at"] = get_owner_artifact_expire_at("pipeline_task", task_id)
    updated["updated_at"] = now_server_local_iso()
    upsert_pipeline_task_status(task_id, updated)
    return updated


@router.get("/task-status/{task_id}")
async def task_status(task_id: str):
    """
    查询完整流水线任务状态（单文件或批量）。
    """
    status = get_pipeline_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return status


@router.get("/pipeline-tasks/{task_id}/debug-jsonl")
async def read_pipeline_task_debug_jsonl(
    task_id: str,
    chunk_index: Optional[int] = Query(default=None),
    event: Optional[str] = Query(default=None),
):
    status = get_pipeline_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    is_valid, expire_at = _artifact_not_expired(task_id)
    if not is_valid:
        raise HTTPException(
            status_code=410,
            detail="模型原始响应调试文件已过期并等待清理，请重新生成任务。",
        )

    debug_paths = _registered_debug_paths(status)
    existing_paths = [path for path in debug_paths if os.path.exists(path)]
    if not existing_paths:
        raise HTTPException(
            status_code=404,
            detail="当前任务没有可读取的模型原始响应文件，可能尚未生成或已被清理。",
        )

    event_filter = str(event or "").strip()
    records: List[Dict[str, Any]] = []
    skipped_lines = 0
    for path in existing_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except Exception:
                        skipped_lines += 1
                        continue
                    if not isinstance(record, dict):
                        skipped_lines += 1
                        continue
                    if chunk_index is not None:
                        try:
                            record_chunk = int(record.get("chunk_index"))
                        except Exception:
                            continue
                        if record_chunk != int(chunk_index):
                            continue
                    if event_filter and str(record.get("event") or "") != event_filter:
                        continue
                    record["_debug_file"] = os.path.basename(path)
                    records.append(record)
        except FileNotFoundError:
            continue
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"读取调试日志失败：{str(exc)}")

    return {
        "task_id": task_id,
        "artifacts_expire_at": expire_at,
        "debug_files": [os.path.basename(path) for path in existing_paths],
        "filters": {"chunk_index": chunk_index, "event": event_filter or None},
        "count": len(records),
        "skipped_lines": skipped_lines,
        "records": records,
    }


@router.post("/pipeline-tasks/{task_id}/ingest-selected-qa")
async def ingest_selected_pipeline_qa(task_id: str, body: IngestSelectedQARequest):
    status = get_pipeline_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    is_valid, expire_at = _artifact_not_expired(task_id)
    if not is_valid:
        raise HTTPException(
            status_code=410,
            detail="临时合并 JSON 已过期，无法人工入库；请重新生成任务。",
        )

    output_index, output_record, json_path = _find_consolidated_output(status, body.source_file)
    payload = _load_registered_json(json_path)
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise HTTPException(status_code=400, detail="合并 JSON 中没有可入库的 QA。")

    if body.select_all_task:
        selected_items = [item for item in raw_items if isinstance(item, dict)]
        missing_count = 0
    else:
        selected_ids = {str(item).strip() for item in (body.selected_ids or []) if str(item).strip()}
        if not selected_ids:
            raise HTTPException(status_code=400, detail="请先勾选要入库的 QA，或使用一键全选当前任务。")
        selected_items = [
            item
            for item in raw_items
            if isinstance(item, dict) and str(item.get("id") or "").strip() in selected_ids
        ]
        missing_count = max(0, len(selected_ids) - len(selected_items))

    if not selected_items:
        raise HTTPException(status_code=400, detail="没有匹配到可入库的 QA，请刷新任务结果后重试。")

    selected_payload = dict(payload)
    selected_payload["items"] = selected_items
    counts = dict(selected_payload.get("counts") or {})
    counts["qa_pairs"] = len(selected_items)
    counts["filtered_qa_pairs"] = sum(1 for item in selected_items if bool(item.get("filtered")))
    selected_payload["counts"] = counts

    vector_result = store_qa_payload_to_milvus(selected_payload, enable_vector_storage=True)
    if not vector_result.get("success"):
        raise HTTPException(
            status_code=500,
            detail=f"人工入库失败：{vector_result.get('message') or '未知错误'}",
        )

    updated_status = _update_manual_ingest_status(
        status=status,
        task_id=task_id,
        output_index=output_index,
        vector_result=vector_result,
        selected_count=len(selected_items),
        select_all_task=body.select_all_task,
    )
    return {
        "success": True,
        "task_id": task_id,
        "source_file": body.source_file or output_record.get("source_file") or "",
        "selected_count": len(selected_items),
        "stored_count": int(vector_result.get("stored_count") or 0),
        "skipped_count": missing_count,
        "vector_storage_result": vector_result,
        "artifacts_expire_at": expire_at,
        "task": {
            "task_id": updated_status.get("task_id"),
            "history_source": updated_status.get("history_source"),
            "milvus_task_id": updated_status.get("milvus_task_id"),
            "artifacts_expire_at": updated_status.get("artifacts_expire_at"),
        },
    }


@router.get("/task-file-csv/{task_id}")
async def download_task_file_csv(task_id: str, original_filename: str):
    """
    根据 task_id 和原始文件名下载对应的合并 CSV。
    - 仅在当前统一完整流水线入口中，以“分文件保存”模式有效；
    - 若找不到该文件对应的 CSV，则回退到任务级最新 CSV。
    """
    outputs_dir = CONFIG["outputs_dir"]
    try:
        if not os.path.exists(outputs_dir):
            raise HTTPException(status_code=404, detail="输出目录不存在")

        safe_name = sanitize_filename(original_filename)
        prefix = f"{task_id}_{safe_name}_consolidated"
        candidates = [
            f
            for f in os.listdir(outputs_dir)
            if f.startswith(prefix) and f.endswith(".csv")
        ]

        # 若按文件名未找到，则回退为该任务下的任意 CSV（与 /task-csv 行为一致）
        if not candidates:
            candidates = [
                f
                for f in os.listdir(outputs_dir)
                if f.startswith(task_id) and f.endswith(".csv")
            ]
            if not candidates:
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到任务{task_id} 对应的 CSV 文件",
                )

        csv_file = sorted(candidates)[-1]
        full_path = os.path.join(outputs_dir, csv_file)
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404, detail=f"CSV 文件不存在: {csv_file}"
            )
        return FileResponse(path=full_path, filename=csv_file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"下载按文件 CSV 失败: {str(exc)}"
        )


@router.get("/task-csv/{task_id}")
async def download_task_csv(task_id: str):
    """
    根据 task_id 在 outputs 目录中查找对应的 CSV 文件（最新的一份）并返回下载。

    - 适用于当前统一完整流水线入口生成的合并结果，文件名形如：
      {task_id}_*_consolidated_*.csv 或 {task_id}_consolidated_batch_*.csv
    """
    try:
        outputs_dir = CONFIG["outputs_dir"]
        if not os.path.exists(outputs_dir):
            raise HTTPException(status_code=404, detail="输出目录不存在")
        candidates = [
            f
            for f in os.listdir(outputs_dir)
            if f.startswith(task_id) and f.endswith(".csv")
        ]
        if not candidates:
            raise HTTPException(
                status_code=404, detail=f"未找到任务 {task_id} 对应的 CSV 文件"
            )
        csv_file = sorted(candidates)[-1]
        full_path = os.path.join(outputs_dir, csv_file)
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404, detail=f"CSV 文件不存在: {csv_file}"
            )
        return FileResponse(path=full_path, filename=csv_file)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"下载任务 CSV 失败: {str(exc)}")


@router.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """
    显式中断正在进行的批量或单文件流水线任务。

    - 如果任务不存在或已结束，返回 404
    - 如果任务本身已完成，返回 400
    - 如果取消命令已发出，返回 cancel_requested
    """
    task = ACTIVE_BATCH_JOBS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")

    # 如果任务已经标记完成/失败，则不再尝试取消
    if task.done():
        ACTIVE_BATCH_JOBS.pop(task_id, None)
        raise HTTPException(status_code=400, detail="任务已完成或已结束")

    task.cancel()

    # 等待短时间，给异步任务机会正常收尾
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # 超时或取消异常对于调用方来说可以忽略
        pass

    # 主动更新一次状态文件，标记为已取消
    try:
        status_data = get_pipeline_task_status(task_id)
        if status_data:
            status_data["status"] = "canceled"
            status_data["message"] = "Task was canceled by user"
            status_data["updated_at"] = now_server_local_iso()
            upsert_pipeline_task_status(task_id, status_data)
    except Exception:
        # 这里的失败不影响取消本身
        pass

    return {"status": "cancel_requested", "task_id": task_id}


@router.get("/pipeline/jobs")
async def list_pipeline_jobs(limit: int = 50):
    jobs = list_pipeline_task_statuses(limit=max(1, min(int(limit or 50), 200)))
    return {"jobs": jobs, "store_path": get_pipeline_store_path()}


@router.delete("/pipeline/jobs/{task_id}")
async def delete_pipeline_job_history(task_id: str):
    status = get_pipeline_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="任务不存在")

    active_task = ACTIVE_BATCH_JOBS.get(task_id)
    if active_task is not None and not active_task.done():
        raise HTTPException(status_code=409, detail="任务仍在运行，不能删除历史记录")

    artifact_paths = _collect_pipeline_output_artifact_paths(status)
    delete_artifacts_now(
        owner_kind="pipeline_task",
        owner_id=task_id,
        reason="history_deleted",
    )
    deleted_paths = delete_paths_now(artifact_paths, reason="pipeline_history_deleted")
    ok = delete_pipeline_task_status(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {
        "success": True,
        "task_id": task_id,
        "deleted_artifacts": len(deleted_paths),
        "store_path": get_pipeline_store_path(),
    }


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """
    下载 outputs 目录下的生成结果文件。
    """
    if file_path.startswith("outputs/"):
        full_path = os.path.join(CONFIG["outputs_dir"], os.path.basename(file_path))
    else:
        full_path = os.path.join(CONFIG["outputs_dir"], file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(path=full_path, filename=os.path.basename(full_path))


@router.get("/list-files")
async def list_files():
    """
    列出 outputs 目录下所有文件。
    """
    os.makedirs(CONFIG["outputs_dir"], exist_ok=True)
    outputs = [
        f
        for f in os.listdir(CONFIG["outputs_dir"])
        if os.path.isfile(os.path.join(CONFIG["outputs_dir"], f))
    ]
    return {"outputs": outputs}



__all__ = ["router"]
