# 文件作用：作为 GPU 调度服务的公共 facade。
# 关联说明：聚合同目录 scheduler.py，供 pipeline 和模型服务做 GPU 准入控制。

"""Public facade for GPU scheduling services."""

from .scheduler import *

