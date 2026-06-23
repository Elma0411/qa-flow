# 文件作用：实现本地模型驱动的问答质量综合评估器。
# 关联说明：核心评估器本体；模型加载、评分算法和 CLI 已拆到同包独立模块。

from __future__ import annotations

from app.core.runtime_paths import (
    DEFAULT_COVERAGE_EMBED_MODEL_NAME,
    DEFAULT_FLUENCY_MODEL_NAME,
    resolve_model_reference,
)

from .model_loading import load_fluency_model, load_grammar_tool, load_semantic_model
from .runtime import select_device
from .scoring import (
    score_accuracy,
    score_coverage,
    score_fluency,
    score_overlap,
    score_relevance,
)


class QAEvaluator:
    def __init__(self, use_local_models: bool = True):
        from sentence_transformers import SentenceTransformer

        self.device = select_device()
        self.st_model = load_semantic_model(
            resolve_model_reference=resolve_model_reference,
            default_coverage_model=DEFAULT_COVERAGE_EMBED_MODEL_NAME,
            use_local_models=use_local_models,
            sentence_transformer_cls=SentenceTransformer,
        )
        self.ppl_model = load_fluency_model(
            resolve_model_reference=resolve_model_reference,
            default_fluency_model=DEFAULT_FLUENCY_MODEL_NAME,
        )
        self.grammar_tool, self.grammar_available = load_grammar_tool()

    def relevance(self, question: str, answer: str) -> float:
        return float(score_relevance(self.st_model, question, answer))

    def coverage(self, question: str, answer: str) -> float:
        return score_coverage(self.st_model, question, answer)

    def overlap(self, answer: str, source: str) -> float:
        return score_overlap(self.st_model, answer, source)

    def accuracy(self, answer: str, source: str) -> float:
        return score_accuracy(self.st_model, answer, source)

    def qa_fluency(self, question: str, answer: str) -> float:
        return score_fluency(self.ppl_model, self.grammar_tool, self.grammar_available, question, answer)


def load_data(filepath: str):
    import json
    import pandas as pd

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)
