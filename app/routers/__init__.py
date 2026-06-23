# 文件作用：聚合并导出应用的各类 API 路由模块。
# 关联说明：作为 app/main.py 的唯一路由 facade；同目录各文件分别承载一个业务 API 面。

from fastapi import APIRouter

from . import (
    admin_v1,
    doc_chunks,
    eval_v1,
    knowledge_tagging,
    llm_config,
    llm_debug,
    milvus_admin,
    ocr_compat_routes,
    ocr_config,
    pipeline,
    search,
    system,
)

router = APIRouter()
router.include_router(system.router)
router.include_router(admin_v1.router)
router.include_router(doc_chunks.router)
router.include_router(eval_v1.router)
router.include_router(knowledge_tagging.router)
router.include_router(pipeline.router)
router.include_router(search.router)
router.include_router(milvus_admin.router)
router.include_router(llm_config.router)
router.include_router(llm_debug.router)
router.include_router(ocr_config.router)
router.include_router(ocr_compat_routes.router)

__all__ = [
    "router",
    "system",
    "admin_v1",
    "doc_chunks",
    "eval_v1",
    "knowledge_tagging",
    "pipeline",
    "search",
    "milvus_admin",
    "llm_config",
    "llm_debug",
    "ocr_compat_routes",
    "ocr_config",
]
