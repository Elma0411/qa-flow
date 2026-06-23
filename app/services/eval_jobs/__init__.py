# 文件作用：作为评测作业服务的公共 facade。
# 关联说明：聚合 dataset、run、result、service，供 eval_v1 路由调用。

"""Public facade for evaluation dataset jobs."""

from .dataset import *
from .service import *

