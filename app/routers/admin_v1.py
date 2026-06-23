# 文件作用：聚合管理端 v1 路由并统一挂载子路由。
# 关联说明：聚合 admin_v1_item_routes 和 admin_v1_job_routes，对外形成管理端路由入口。

from fastapi import APIRouter

from app.routers.admin_v1_item_routes import router as item_router
from app.routers.admin_v1_job_routes import router as job_router

router = APIRouter(prefix='/admin/v1', tags=['admin'])
router.include_router(item_router)
router.include_router(job_router)

__all__ = ['router']
