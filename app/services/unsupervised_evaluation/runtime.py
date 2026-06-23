# 文件作用：维护无监督评估的运行态、可用性和延迟加载对象。
# 关联说明：被 runners/suite/service 共享，集中保存评估器导入、释放和可用性状态。

from __future__ import annotations

from typing import Any


class UnsupervisedEvaluationRuntime:
    """Container for unsupervised evaluation runtime state."""

    def __init__(self) -> None:
        self.attach_faithfulness = None
        self.release_nli_device_cache = None
        self.UNSUPERVISED_FAITHFULNESS_AVAILABLE = False
        self.UNSUPERVISED_FAITHFULNESS_IMPORT_ERROR: Any = None

        self.attach_answerability = None
        self.release_answerability_device_cache = None
        self.UNSUPERVISED_ANSWERABILITY_AVAILABLE = False
        self.UNSUPERVISED_ANSWERABILITY_IMPORT_ERROR: Any = None

        self.attach_coverage_recall = None
        self.release_coverage_device_cache = None
        self.UNSUPERVISED_COVERAGE_AVAILABLE = False
        self.UNSUPERVISED_COVERAGE_IMPORT_ERROR: Any = None

        self.attach_fluency_ppl = None
        self.release_fluency_device_cache = None
        self.UNSUPERVISED_FLUENCY_PPL_AVAILABLE = False
        self.UNSUPERVISED_FLUENCY_PPL_IMPORT_ERROR: Any = None

        self.UNSUPERVISED_EVALUATION_AVAILABLE = False

    def refresh(self) -> None:
        try:
            from qa.qa_evaluation.unsupervised_faithfulness import (
                TRANSFORMERS_AVAILABLE,
                attach_faithfulness,
                release_nli_device_cache,
            )

            self.attach_faithfulness = attach_faithfulness
            self.release_nli_device_cache = release_nli_device_cache
            self.UNSUPERVISED_FAITHFULNESS_AVAILABLE = bool(TRANSFORMERS_AVAILABLE)
            self.UNSUPERVISED_FAITHFULNESS_IMPORT_ERROR = None
        except Exception as exc:  # pragma: no cover - optional dependency
            self.attach_faithfulness = None
            self.release_nli_device_cache = None
            self.UNSUPERVISED_FAITHFULNESS_AVAILABLE = False
            self.UNSUPERVISED_FAITHFULNESS_IMPORT_ERROR = exc

        try:
            from qa.qa_evaluation.unsupervised_answerability import (
                TRANSFORMERS_AVAILABLE as QA_TRANSFORMERS_AVAILABLE,
                attach_answerability,
                release_answerability_device_cache,
            )

            self.attach_answerability = attach_answerability
            self.release_answerability_device_cache = release_answerability_device_cache
            self.UNSUPERVISED_ANSWERABILITY_AVAILABLE = bool(QA_TRANSFORMERS_AVAILABLE)
            self.UNSUPERVISED_ANSWERABILITY_IMPORT_ERROR = None
        except Exception as exc:  # pragma: no cover - optional dependency
            self.attach_answerability = None
            self.release_answerability_device_cache = None
            self.UNSUPERVISED_ANSWERABILITY_AVAILABLE = False
            self.UNSUPERVISED_ANSWERABILITY_IMPORT_ERROR = exc

        try:
            from qa.qa_evaluation.unsupervised_coverage_recall import (
                SENTENCE_TRANSFORMERS_AVAILABLE,
                attach_coverage_recall,
                release_coverage_device_cache,
            )

            self.attach_coverage_recall = attach_coverage_recall
            self.release_coverage_device_cache = release_coverage_device_cache
            self.UNSUPERVISED_COVERAGE_AVAILABLE = bool(SENTENCE_TRANSFORMERS_AVAILABLE)
            self.UNSUPERVISED_COVERAGE_IMPORT_ERROR = None
        except Exception as exc:  # pragma: no cover - optional dependency
            self.attach_coverage_recall = None
            self.release_coverage_device_cache = None
            self.UNSUPERVISED_COVERAGE_AVAILABLE = False
            self.UNSUPERVISED_COVERAGE_IMPORT_ERROR = exc

        try:
            from qa.qa_evaluation.unsupervised_fluency_ppl import (
                TRANSFORMERS_AVAILABLE as FLUENCY_TRANSFORMERS_AVAILABLE,
                attach_fluency_ppl,
                release_fluency_device_cache,
            )

            self.attach_fluency_ppl = attach_fluency_ppl
            self.release_fluency_device_cache = release_fluency_device_cache
            self.UNSUPERVISED_FLUENCY_PPL_AVAILABLE = bool(FLUENCY_TRANSFORMERS_AVAILABLE)
            self.UNSUPERVISED_FLUENCY_PPL_IMPORT_ERROR = None
        except Exception as exc:  # pragma: no cover - optional dependency
            self.attach_fluency_ppl = None
            self.release_fluency_device_cache = None
            self.UNSUPERVISED_FLUENCY_PPL_AVAILABLE = False
            self.UNSUPERVISED_FLUENCY_PPL_IMPORT_ERROR = exc

        self.UNSUPERVISED_EVALUATION_AVAILABLE = bool(
            self.UNSUPERVISED_FAITHFULNESS_AVAILABLE
            or self.UNSUPERVISED_ANSWERABILITY_AVAILABLE
            or self.UNSUPERVISED_COVERAGE_AVAILABLE
            or self.UNSUPERVISED_FLUENCY_PPL_AVAILABLE
        )


UNSUPERVISED_EVALUATION_RUNTIME = UnsupervisedEvaluationRuntime()
UNSUPERVISED_EVALUATION_RUNTIME.refresh()


def __getattr__(name: str) -> Any:
    if hasattr(UNSUPERVISED_EVALUATION_RUNTIME, name):
        return getattr(UNSUPERVISED_EVALUATION_RUNTIME, name)
    raise AttributeError(name)


__all__ = [
    "UNSUPERVISED_EVALUATION_RUNTIME",
    "UnsupervisedEvaluationRuntime",
]
