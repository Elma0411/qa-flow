# 文件作用：作为问答增广能力的公共 facade。
# 关联说明：聚合同目录 llm_qa_augmentation.py，供 pipeline_batch_routes 使用。

"""Public facade for QA augmentation capabilities."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "augment_qa_pairs": ".llm_qa_augmentation",
    "call_api": ".llm_qa_augmentation",
    "construct_prompt": ".llm_qa_augmentation",
    "extract_qa_pairs": ".llm_qa_augmentation",
    "main": ".llm_qa_augmentation",
    "read_csv": ".llm_qa_augmentation",
    "save_to_csv": ".llm_qa_augmentation",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
