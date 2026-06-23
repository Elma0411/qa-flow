# 文件作用：提供外部服务客户端的创建、缓存与复用入口。
# 关联说明：依赖 config 中的连接参数，被 routers 和 services 调用来复用外部客户端。

from functools import lru_cache
from typing import Optional

from openai import OpenAI

from app.core.config import CONFIG


@lru_cache()
def get_default_openai_client() -> OpenAI:
    return OpenAI(
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"],
    )


def clear_default_client_cache() -> None:
    try:
        get_default_openai_client.cache_clear()
    except Exception:
        pass


def build_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    return OpenAI(
        api_key=api_key or CONFIG["api_key"],
        base_url=base_url or CONFIG["base_url"],
    )


__all__ = ["build_openai_client", "get_default_openai_client", "clear_default_client_cache"]
