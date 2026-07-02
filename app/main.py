# 文件作用：组装 FastAPI 应用、中间件、静态资源挂载与路由注册。
# 关联说明：作为应用工厂层，只负责装配，不承载业务逻辑。

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.logger import logger
from app.routers import router as api_router
from app.services.artifacts import (
    initialize_artifact_lifecycle,
    start_artifact_cleanup_loop,
    stop_artifact_cleanup_loop,
)
from app.services.gpu import get_scheduler_snapshot
from app.services.llm import get_llm_client_pool
from app.services.llm_config import activate_profile as activate_llm_profile
from app.services.llm_config import list_profiles as list_llm_profiles
from app.services.milvus import ensure_milvus_initialized


def _load_active_llm_profile() -> None:
    try:
        store = list_llm_profiles()
        active = str(store.get("active") or "").strip()
        if not active:
            return
        activate_llm_profile(active)
        logger.info("Loaded active LLM profile into runtime config: %s", active)
    except Exception as exc:
        logger.warning("Unable to load active LLM profile into runtime config: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_active_llm_profile()
    ensure_milvus_initialized()
    initialize_artifact_lifecycle()
    start_artifact_cleanup_loop()
    try:
        logger.info("GPU scheduler snapshot: %s", get_scheduler_snapshot())
    except Exception:
        logger.info("GPU scheduler snapshot unavailable")
    yield
    get_llm_client_pool().close_all()
    await stop_artifact_cleanup_loop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="问答生成API",
        description="基于大语言模型的问答生成系统API，支持LaTeX公式和混合内容处理",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    try:
        app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
    except Exception:
        pass
    app.include_router(api_router)
    return app


app = create_app()
