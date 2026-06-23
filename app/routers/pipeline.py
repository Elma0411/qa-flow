# 文件作用：聚合完整流水线相关路由模块。
# 关联说明：聚合 batch、integrated、evaluation、history 路由。

from fastapi import APIRouter

from app.routers.pipeline_batch_routes import (
    router as batch_router,
    run_batch_complete_pipeline_async,
)
from app.routers.pipeline_evaluation_routes import router as evaluation_router
from app.routers.pipeline_history_routes import router as history_router
from app.routers.pipeline_integrated_routes import router as integrated_router

router = APIRouter()
router.include_router(evaluation_router)
router.include_router(batch_router)
router.include_router(integrated_router)
router.include_router(history_router)

__all__ = ["router", "run_batch_complete_pipeline_async"]
