# 文件作用：作为临时产物生命周期服务的公共 facade。
# 关联说明：聚合同目录 lifecycle.py，供 pipeline_history_routes 和 batch 路由使用。

"""Public facade for artifact lifecycle services."""

from .lifecycle import (
    ARTIFACT_LIFECYCLE,
    ArtifactLifecycleService,
    cleanup_expired_artifacts,
    delete_artifacts_now,
    delete_paths_now,
    get_owner_artifact_expire_at,
    initialize_artifact_lifecycle,
    register_temporary_artifacts,
    start_artifact_cleanup_loop,
    stop_artifact_cleanup_loop,
)

__all__ = [
    "ARTIFACT_LIFECYCLE",
    "ArtifactLifecycleService",
    "cleanup_expired_artifacts",
    "delete_artifacts_now",
    "delete_paths_now",
    "get_owner_artifact_expire_at",
    "initialize_artifact_lifecycle",
    "register_temporary_artifacts",
    "start_artifact_cleanup_loop",
    "stop_artifact_cleanup_loop",
]
