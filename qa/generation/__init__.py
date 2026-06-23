# 文件作用：作为问答生成能力的公共 facade。
# 关联说明：聚合 qa_generation_flow、evidence_units、text_quality_filters 给 pipeline 使用。

"""Public facade for one-step QA generation capabilities."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "apply_question_type_plan": ".qa_generation_flow",
    "build_document_chunks": ".evidence_units",
    "build_question_type_plan": ".qa_generation_flow",
    "call_candidate_question_llm": ".qa_generation_flow",
    "call_evidence_answer_llm": ".qa_generation_flow",
    "contains_ambiguous_reference": ".text_quality_filters",
    "DEFAULT_MAX_UNIT_CHARS": ".evidence_units",
    "DEFAULT_SEMANTIC_TOP_K": ".evidence_units",
    "normalize_question_type_mode": ".qa_generation_flow",
    "normalize_question_type_weights": ".qa_generation_flow",
    "normalize_question_types": ".qa_generation_flow",
    "QADocumentEvidenceIndex": ".evidence_units",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
