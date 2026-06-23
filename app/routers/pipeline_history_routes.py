# 文件作用：提供流水线任务状态、历史记录、产物下载和清理接口。
# 关联说明：复用 pipeline_state、storage、artifacts 服务，管理 pipeline 任务和产物。

import asyncio
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import ACTIVE_BATCH_JOBS, CONFIG
from app.core.time_utils import now_server_local_iso
from app.routers.pipeline_common import _collect_pipeline_output_artifact_paths
from app.services.artifacts import delete_artifacts_now, delete_paths_now
from app.services.pipeline_state import (
    delete_pipeline_task_status,
    get_pipeline_store_path,
    get_pipeline_task_status,
    list_pipeline_task_statuses,
    upsert_pipeline_task_status,
)
from app.services.storage import sanitize_filename

router = APIRouter()

@router.get("/task-status/{task_id}")
async def task_status(task_id: str):
    """
    查询完整流水线任务状态（单文件或批量）。
    """
    status = get_pipeline_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return status


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
