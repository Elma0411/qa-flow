# 文件作用：提供 OCR 服务配置档案的管理与测试接口。
# 关联说明：对接 app.services.ocr 的配置部分，pipeline_batch_routes 使用 OCR 解析能力。

from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import ocr as ocr_config_service

router = APIRouter()


class OcrRequestConfig(BaseModel):
    batch_field: str = Field("files", description="批量上传字段名（batch_ocr 使用）")
    file_field: str = Field("file", description="单文件上传字段名（process_api 使用）")
    extra_form_fields: Dict[str, str] = Field(default_factory=dict, description="额外表单字段（键值对）")


class OcrResponseConfig(BaseModel):
    mode: str = Field("structured_json", description="structured_json|text|file")


class OcrConfigPayload(BaseModel):
    name: str = Field(..., description="配置名称")
    provider: str = Field(..., description="batch_ocr|process_api")
    post_url: str = Field(..., description="完整 POST 地址（包含 path）")
    timeout_seconds: int = Field(600, ge=1, le=3600, description="默认超时（秒）")
    request: OcrRequestConfig = Field(default_factory=OcrRequestConfig)
    response: Optional[OcrResponseConfig] = None


@router.get("/ocr-configs")
async def list_ocr_configs() -> Dict[str, Any]:
    return ocr_config_service.list_profiles()


@router.post("/ocr-configs")
async def upsert_ocr_config(payload: OcrConfigPayload) -> Dict[str, Any]:
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()  # type: ignore
        if not data.get("response"):
            data["response"] = {"mode": "structured_json" if payload.provider == "batch_ocr" else "text"}
        return ocr_config_service.upsert_profile(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/ocr-configs/{name}/activate")
async def activate_ocr_config(name: str) -> Dict[str, Any]:
    try:
        return ocr_config_service.activate_profile(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/ocr-configs/{name}")
async def delete_ocr_config(name: str) -> Dict[str, Any]:
    try:
        return ocr_config_service.delete_profile(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/ocr-configs/{name}/test")
async def test_ocr_config(name: str) -> Dict[str, Any]:
    profile = ocr_config_service.get_profile(name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"配置不存在: {name}")
    url = str(profile.get("post_url") or "").strip()
    timeout = int(profile.get("timeout_seconds") or 10)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, timeout=httpx.Timeout(timeout))
        return {
            "ok": True,
            "status_code": resp.status_code,
            "content_type": resp.headers.get("content-type", ""),
            "note": "可达性测试：该请求不带文件，返回 4xx/5xx 也可能是正常的（说明服务可达但需要文件/参数）",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


__all__ = ["router"]

