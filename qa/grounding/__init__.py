# 文件作用：作为事实溯源校验能力的公共 facade。
# 关联说明：聚合 source_fact_grounding，供 text_to_qa_pipeline 调用。

"""Public facade for QA source-grounding checks."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "normalize_grounding_text": ".source_fact_grounding",
    "validate_source_fact_grounding": ".source_fact_grounding",
    "validate_source_fact_text_detail_mode": ".source_fact_grounding",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
