# 文件作用：提供大模型连通性和对话调试接口。
# 关联说明：复用 core 客户端和 qa.common 响应解析，用于排查 LLM 配置是否可用。

import asyncio
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.clients import get_default_openai_client
from app.core.config import CONFIG
from app.services import llm_config as llm_config_service
from qa.common import extract_first_choice_content, safe_response_dump

router = APIRouter()


class LlmDebugChatRequest(BaseModel):
    prompt: str = Field(default="请回复 OK", description="用户输入（user message）")
    system_prompt: str = Field(default="You are a helpful assistant.", description="系统提示（system message）")
    response_format: str = Field(
        default="json_object",
        description="text | json_object（与流水线一致建议用 json_object）",
    )
    timeout_seconds: int = Field(default=30, ge=1, le=600)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


@router.post("/llm-debug/chat")
async def llm_debug_chat(payload: LlmDebugChatRequest) -> Dict[str, Any]:
    store = llm_config_service.list_profiles()
    active_profile = store.get("active") if isinstance(store, dict) else None

    api_key_present = bool(str(CONFIG.get("api_key") or "").strip())
    base_url = str(CONFIG.get("base_url") or "")
    model = str(CONFIG.get("model") or "")

    request: Dict[str, Any] = {
        "model": model,
        "timeout_seconds": payload.timeout_seconds,
        "max_tokens": payload.max_tokens,
        "temperature": payload.temperature,
        "response_format": payload.response_format,
        "messages": [
            {"role": "system", "content": payload.system_prompt},
            {"role": "user", "content": payload.prompt},
        ],
    }

    if not api_key_present or not base_url or not model:
        return {
            "ok": False,
            "active_profile": active_profile,
            "config": {
                "api_key_present": api_key_present,
                "base_url": base_url,
                "model": model,
            },
            "request": request,
            "error": {
                "type": "LLMConfigError",
                "message": "LLM 未配置：请先在 /ui/index.html 的「LLM 配置管理」保存并激活配置，或设置环境变量 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL。",
            },
            "elapsed_ms": 0,
        }

    client = get_default_openai_client()

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": request["messages"],
        "timeout": int(payload.timeout_seconds),
        "max_tokens": int(payload.max_tokens),
    }
    if payload.temperature is not None:
        kwargs["temperature"] = float(payload.temperature)
    if str(payload.response_format or "").strip().lower() == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    started = time.time()
    try:
        resp = await asyncio.to_thread(lambda: client.chat.completions.create(**kwargs))
        content = extract_first_choice_content(resp)
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            "ok": True,
            "active_profile": active_profile,
            "config": {
                "api_key_present": api_key_present,
                "base_url": base_url,
                "model": model,
            },
            "request": request,
            "response": {
                "response_type": type(resp).__name__,
                "content": content,
                "response_dump": safe_response_dump(resp),
            },
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        elapsed_ms = int((time.time() - started) * 1000)
        return {
            "ok": False,
            "active_profile": active_profile,
            "config": {
                "api_key_present": api_key_present,
                "base_url": base_url,
                "model": model,
            },
            "request": request,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "elapsed_ms": elapsed_ms,
        }


__all__ = ["router"]
