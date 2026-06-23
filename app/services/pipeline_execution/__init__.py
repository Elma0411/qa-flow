# 文件作用：作为 pipeline execution 服务的公共 facade。
# 关联说明：向 pipeline_batch_routes 暴露完整流水线执行函数，具体实现位于 service.py。

from .service import run_batch_complete_pipeline_async

__all__ = ["run_batch_complete_pipeline_async"]
