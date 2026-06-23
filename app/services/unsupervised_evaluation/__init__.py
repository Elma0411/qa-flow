# 文件作用：作为无监督问答评估服务的公共 facade。
# 关联说明：聚合 runtime、runners、aggregation、suite、service 给路由调用。

"""Public facade for unsupervised QA evaluation services."""

from .runtime import (
    UNSUPERVISED_EVALUATION_RUNTIME,
    UnsupervisedEvaluationRuntime,
)
from .service import (
    execute_unsupervised_answerability_blocking,
    execute_unsupervised_coverage_recall_blocking,
    execute_unsupervised_faithfulness_blocking,
    execute_unsupervised_fluency_ppl_blocking,
    execute_unsupervised_suite_blocking,
)

_DYNAMIC_ATTRS = {
    "UNSUPERVISED_ANSWERABILITY_AVAILABLE",
    "UNSUPERVISED_ANSWERABILITY_IMPORT_ERROR",
    "UNSUPERVISED_COVERAGE_AVAILABLE",
    "UNSUPERVISED_COVERAGE_IMPORT_ERROR",
    "UNSUPERVISED_EVALUATION_AVAILABLE",
    "UNSUPERVISED_FAITHFULNESS_AVAILABLE",
    "UNSUPERVISED_FAITHFULNESS_IMPORT_ERROR",
    "UNSUPERVISED_FLUENCY_PPL_AVAILABLE",
    "UNSUPERVISED_FLUENCY_PPL_IMPORT_ERROR",
    "attach_answerability",
    "attach_coverage_recall",
    "attach_faithfulness",
    "attach_fluency_ppl",
    "release_answerability_device_cache",
    "release_coverage_device_cache",
    "release_fluency_device_cache",
    "release_nli_device_cache",
}


def __getattr__(name):
    if name in _DYNAMIC_ATTRS:
        return getattr(UNSUPERVISED_EVALUATION_RUNTIME, name)
    raise AttributeError(name)


__all__ = [
    "UNSUPERVISED_EVALUATION_RUNTIME",
    "UnsupervisedEvaluationRuntime",
    "execute_unsupervised_answerability_blocking",
    "execute_unsupervised_coverage_recall_blocking",
    "execute_unsupervised_faithfulness_blocking",
    "execute_unsupervised_fluency_ppl_blocking",
    "execute_unsupervised_suite_blocking",
    "UNSUPERVISED_ANSWERABILITY_AVAILABLE",
    "UNSUPERVISED_ANSWERABILITY_IMPORT_ERROR",
    "UNSUPERVISED_COVERAGE_AVAILABLE",
    "UNSUPERVISED_COVERAGE_IMPORT_ERROR",
    "UNSUPERVISED_EVALUATION_AVAILABLE",
    "UNSUPERVISED_FAITHFULNESS_AVAILABLE",
    "UNSUPERVISED_FAITHFULNESS_IMPORT_ERROR",
    "UNSUPERVISED_FLUENCY_PPL_AVAILABLE",
    "UNSUPERVISED_FLUENCY_PPL_IMPORT_ERROR",
    "attach_answerability",
    "attach_coverage_recall",
    "attach_faithfulness",
    "attach_fluency_ppl",
    "release_answerability_device_cache",
    "release_coverage_device_cache",
    "release_fluency_device_cache",
    "release_nli_device_cache",
]
