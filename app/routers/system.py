# 文件作用：提供服务健康检查、环境检查和根路由接口。
# 关联说明：对接 app.services.system 和 core 配置，提供健康检查与环境诊断。

import os
import time
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from app.core.clients import get_default_openai_client
from app.core.config import CONFIG
from app.services.system import run_environment_check, test_api_connection
from app.services.pipeline_state import get_pipeline_task_status


router = APIRouter()


@router.get("/")
def root_index():
    return RedirectResponse(url="/ui/index.html")


@router.get("/test-connection")
async def test_connection():
    try:
        success, result = test_api_connection(get_default_openai_client(), CONFIG["model"])
        if success:
            return {"status": "success", "message": f"API连接测试: {result}"}
        raise HTTPException(status_code=500, detail=f"API连接失败: {result}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"API连接失败: {str(exc)}")


@router.get("/task-status/{task_id}")
async def task_status(task_id: str):
    status = get_pipeline_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return status


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    if file_path.startswith("outputs/"):
        full_path = os.path.join(CONFIG["outputs_dir"], os.path.basename(file_path))
    else:
        full_path = os.path.join(CONFIG["outputs_dir"], file_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(path=full_path, filename=os.path.basename(full_path))


@router.get("/list-files")
async def list_files():
    os.makedirs(CONFIG["outputs_dir"], exist_ok=True)
    outputs = [
        f
        for f in os.listdir(CONFIG["outputs_dir"])
        if os.path.isfile(os.path.join(CONFIG["outputs_dir"], f))
    ]
    return {"outputs": outputs}


@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@router.get("/environment-check")
async def environment_check():
    return await asyncio.to_thread(run_environment_check)
