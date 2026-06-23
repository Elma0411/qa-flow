# 文件作用：作为问答条目校验能力的公共 facade。
# 关联说明：聚合 qa_item.py，供 text_to_qa_pipeline 和 generation 使用。

"""Public facade for QA item validation and normalization."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "normalize_difficulty_level": ".qa_item",
    "normalize_question_type": ".qa_item",
    "validate_and_normalize_item": ".qa_item",
    "validate_and_normalize_item_with_reason": ".qa_item",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
