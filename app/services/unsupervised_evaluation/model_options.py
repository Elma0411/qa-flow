# 文件作用：集中定义无监督评估可选模型并解析本地模型路径。
# 关联说明：供评测路由和流水线服务共享，避免前后端模型名校验分散。

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from app.core.runtime_paths import resolve_model_reference


EVALUATION_MODEL_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "faithfulness_nli": (
        "mdeberta_v3_base_xnli_nli_2mil7",
        "erlangshen_roberta_110m_nli",
        "xlm_roberta_large_xnli",
    ),
    "answerability_qa": (
        "deepset_xlm_roberta_base_squad2",
        "deepset_xlm_roberta_large_squad2",
    ),
    "coverage_embedding": (
        "bge-m3",
        "qwen3_embedding_0_6b",
        "qwen3_embedding_4b",
    ),
}


def normalize_evaluation_model_name(model_name: Optional[str], *, kind: str) -> str:
    """Validate a UI model name and normalize auto/default to an empty override."""
    model_kind = str(kind or "").strip()
    allowed = EVALUATION_MODEL_OPTIONS.get(model_kind)
    if allowed is None:
        raise ValueError(f"未知评估模型类型: {model_kind}")

    normalized = str(model_name or "").strip()
    if not normalized or normalized.lower() in {"auto", "default"}:
        return ""
    if normalized not in allowed:
        raise ValueError(f"{model_kind} 必须为空/auto 或以下之一: {list(allowed)}")
    return normalized


def resolve_evaluation_model_path(model_name: Optional[str], *, kind: str) -> Optional[str]:
    """Resolve a selected model name and require its local directory to exist."""
    normalized = normalize_evaluation_model_name(model_name, kind=kind)
    if not normalized:
        return None
    resolved_path = resolve_model_reference(normalized)
    if not os.path.isdir(resolved_path):
        raise ValueError(f"{kind} 模型目录不存在: {resolved_path}")
    return resolved_path


def validate_evaluation_model_name(model_name: Optional[str], *, kind: str) -> str:
    normalized = normalize_evaluation_model_name(model_name, kind=kind)
    if normalized:
        resolve_evaluation_model_path(normalized, kind=kind)
    return normalized


__all__ = [
    "EVALUATION_MODEL_OPTIONS",
    "normalize_evaluation_model_name",
    "resolve_evaluation_model_path",
    "validate_evaluation_model_name",
]
