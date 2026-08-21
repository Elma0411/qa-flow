# 文件作用：组装 FastAPI 应用、中间件、静态资源挂载与路由注册。
# 关联说明：作为应用工厂层，只负责装配，不承载业务逻辑。

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
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
from qa.retrieval import get_reranker_service


UI_BUILD_ID = "2026-08-21-2"


def _set_ui_cache_headers(response) -> None:
    """Prevent browser/VPN gateways from mixing UI builds across deployments."""

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0, s-maxage=0"
    )
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Surrogate-Control"] = "no-store"
    response.headers["CDN-Cache-Control"] = "no-store"
    response.headers["X-Accel-Expires"] = "0"
    response.headers["X-QA-UI-Build"] = UI_BUILD_ID


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
    get_reranker_service().close()
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

    @app.middleware("http")
    async def disable_ui_html_cache(request, call_next):
        response = await call_next(request)
        path = str(request.url.path or "")
        # The directory entry `/ui/` is the URL most commonly bookmarked and
        # rewritten by SSLVPN/WebVPN. It serves index.html through StaticFiles
        # but does not end in `.html`, so matching only HTML files leaves the
        # gateway free to cache the old workbench indefinitely. Apply the same
        # policy to the entry path and every asset under the UI mount.
        if path == "/" or path == "/ui" or path.startswith("/ui/"):
            _set_ui_cache_headers(response)
        return response

    try:
        @app.get("/ui", include_in_schema=False)
        async def redirect_ui_entry():
            # StaticFiles normally emits an absolute redirect here. A relative
            # target is required when the browser reached us through a VPN
            # prefix that the backend itself cannot see.
            return RedirectResponse(url="./ui/", status_code=307)

        @app.get("/ui/", include_in_schema=False)
        async def serve_ui_index():
            # Keep the canonical directory URL, but bypass StaticFiles' index
            # fallback so `/ui/` receives the same explicit cache policy as
            # `/ui/index.html` under every proxy.
            return FileResponse("static/index.html", media_type="text/html")

        app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
    except Exception:
        pass
    app.include_router(api_router)
    return app


app = create_app()
