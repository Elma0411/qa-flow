# 文件作用：作为问答质量评估模块的公共入口。
# 关联说明：聚合本地评价器和 LLM 评价器，供 app.services.evaluation 使用。

"""
QA质量评估模块

本模块提供基于本地深度学习模型的问答对质量评估功能，包括：
- 语义相似度评估（使用BGE-M3模型）
- 覆盖度评估
- 准确性评估
- 流畅度评估（使用BERT模型）
- 重叠度评估

主要类：
- QAEvaluator: 主要的评估器类
- MaskedBert: BERT模型包装器（来自language_models模块）

主要函数：
- load_data: 数据加载函数
"""

try:
    from .qa_quality_evaluator import QAEvaluator, load_data  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    QAEvaluator = None  # type: ignore
    load_data = None  # type: ignore

try:
    from .language_models import MaskedBert  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MaskedBert = None  # type: ignore

try:
    from .llm_quality_evaluator import (  # type: ignore
        DEFAULT_CONFIG as LLM_QUALITY_DEFAULT_CONFIG,
        evaluate_qa_pairs,
    )
except Exception:  # pragma: no cover - optional dependency
    LLM_QUALITY_DEFAULT_CONFIG = None  # type: ignore
    evaluate_qa_pairs = None  # type: ignore

__version__ = "1.0.0"
__author__ = "QA Evaluation Team"

__all__ = [
    "QAEvaluator",
    "load_data",
    "MaskedBert",
    "LLM_QUALITY_DEFAULT_CONFIG",
    "evaluate_qa_pairs",
]
