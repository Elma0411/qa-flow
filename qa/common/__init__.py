# 文件作用：作为 QA 公共工具的公共 facade。
# 关联说明：聚合 language 和 llm_response，供 generation、evaluation、augmentation 共用。

"""Public facade for shared QA helpers."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "build_language_instruction": ".language",
    "detect_language": ".language",
    "extract_first_choice_content": ".llm_response",
    "safe_response_dump": ".llm_response",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
