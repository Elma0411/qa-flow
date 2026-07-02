# 文件作用：提供外部服务客户端的创建、缓存与复用入口。
# 关联说明：依赖 config 中的连接参数，被 routers 和 services 调用来复用外部客户端。
# 现已统一通过 app.services.llm 池系统管理所有 LLM 客户端。

from __future__ import annotations

from typing import Optional

from app.core.config import CONFIG
from app.services.llm import LLMClientConfig, LLMClientProtocol, get_llm_client_pool


def build_llm_client_config(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_type: Optional[str] = None,
    model_version: Optional[str] = None,
    max_concurrent_requests: Optional[int] = None,
) -> LLMClientConfig:
    return LLMClientConfig(
        api_base=base_url or CONFIG.get("base_url"),
        model_name=model or CONFIG.get("model"),
        api_key=api_key or CONFIG.get("api_key"),
        api_type=api_type if api_type is not None else CONFIG.get("api_type"),
        model_version=(
            model_version if model_version is not None else CONFIG.get("model_version")
        ),
        timeout_seconds=float(CONFIG.get("request_timeout", 120) or 120),
        max_concurrent_requests=max_concurrent_requests,
    )


def get_default_openai_client() -> LLMClientProtocol:
    """返回默认的大模型客户端（来自共享连接池）。"""
    return get_llm_client_pool().get_client(build_llm_client_config())


def clear_default_client_cache() -> None:
    """清空连接池中所有已缓存的客户端。"""
    try:
        get_llm_client_pool().close_all()
    except Exception:
        pass


def build_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMClientProtocol:
    """用给定的凭证创建一个新的池客户端（从池中获取或新建）。"""
    return get_llm_client_pool().get_client(
        build_llm_client_config(api_key=api_key, base_url=base_url)
    )


__all__ = [
    "build_llm_client_config",
    "build_openai_client",
    "get_default_openai_client",
    "clear_default_client_cache",
]
