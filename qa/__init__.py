# 文件作用：作为完整 QA pipeline 的公共 facade。
# 关联说明：向 app 层暴露完整 pipeline；具体实现分布在 text_to_qa_pipeline 和 pipeline_runtime。

"""Public facade for full QA pipeline entry points."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "OneStepPipelineRuntime": ".pipeline_runtime",
    "parse_one_step_pipeline_runtime": ".pipeline_runtime",
    "process_text_to_qa_one_step": ".text_to_qa_pipeline",
    "resolve_one_step_chunks": ".pipeline_runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
