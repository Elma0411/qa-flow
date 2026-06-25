# 文件作用：提供大模型配置档案的增删改查与激活接口。
# 关联说明：对接 app.services.llm_config，配置结果会被 pipeline、evaluation、debug 路由使用。

from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import llm_config as llm_config_service

router = APIRouter()


class LLMConfigPayload(BaseModel):
    name: str
    api_key: str
    base_url: str
    model: str
    api_type: Optional[str] = "openai"
    model_version: Optional[str] = ""


@router.get("/llm-configs")
async def list_llm_configs() -> Dict[str, object]:
    return llm_config_service.list_profiles()


@router.post("/llm-configs")
async def upsert_llm_config(payload: LLMConfigPayload) -> Dict[str, object]:
    try:
        return llm_config_service.upsert_profile(
            payload.name,
            payload.api_key,
            payload.base_url,
            payload.model,
            payload.api_type or "openai",
            payload.model_version or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/llm-configs/{name}/activate")
async def activate_llm_config(name: str) -> Dict[str, object]:
    try:
        return llm_config_service.activate_profile(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/llm-configs/{name}")
async def delete_llm_config(name: str) -> Dict[str, object]:
    try:
        return llm_config_service.delete_profile(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


__all__ = ["router"]
