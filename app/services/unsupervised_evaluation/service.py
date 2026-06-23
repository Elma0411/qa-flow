# 文件作用：聚合无监督评估执行、过滤和结果写回服务。
# 关联说明：聚合 suite/aggregation/runners 输出，向路由提供完整服务接口。

from .aggregation import (
    _attach_suite_aggregates,
    _compute_suite_four_scores,
    _upgrade_faithfulness_to_suite,
)
from .common import _context_group_id, _safe_float
from .runtime import UNSUPERVISED_EVALUATION_RUNTIME as _rt

UNSUPERVISED_EVALUATION_RUNTIME = _rt
from .runners import (
    execute_unsupervised_answerability_blocking,
    execute_unsupervised_coverage_recall_blocking,
    execute_unsupervised_faithfulness_blocking,
    execute_unsupervised_fluency_ppl_blocking,
)
from .suite import execute_unsupervised_suite_blocking

__all__ = [
    "UNSUPERVISED_EVALUATION_RUNTIME",
    "execute_unsupervised_faithfulness_blocking",
    "execute_unsupervised_answerability_blocking",
    "execute_unsupervised_coverage_recall_blocking",
    "execute_unsupervised_fluency_ppl_blocking",
    "execute_unsupervised_suite_blocking",
]
