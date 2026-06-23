# 文件作用：提供 pipeline 服务层可复用的纯工具函数和常量。
# 关联说明：供 pipeline_execution 与相关 router 共同使用，避免服务层反向依赖 app.routers。

import json
from typing import Any, Dict, List, Optional

_ARTIFACT_TTL_SECONDS = 24 * 60 * 60


def parse_few_shot_examples(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """
    容错解析 few_shot_examples。
    支持 list、dict、JSON 字符串；非法输入返回 None。
    只用于学习问答长度、语气、格式，不复述示例答案或事实。
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [ex for ex in raw if isinstance(ex, dict)] or None
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [ex for ex in parsed if isinstance(ex, dict)] or None
        except Exception:
            return None
    return None


def _normalize_artifact_path(path: Optional[str]) -> Optional[str]:
    raw = str(path or "").strip()
    return raw or None


def _compute_average_scores_for_result(
    qa_result: Dict[str, Any],
    criteria_list: List[str],
) -> float:
    evaluation = qa_result.get("evaluation", {})
    scores = [
        evaluation.get(metric, {}).get("score")
        for metric in criteria_list
        if metric in evaluation
    ]
    scores = [s for s in scores if isinstance(s, (int, float))]
    return float(sum(scores) / len(scores)) if scores else 0.0


def _parent_key(item: Dict[str, Any]) -> str:
    """
    Build a stable key for matching primary/augmented QA pairs.
    Defined at module scope so it is always available even when augment_per_qa=0.
    """
    return f"{item.get('question','')}|||{item.get('answer','')}"


__all__ = [
    "_ARTIFACT_TTL_SECONDS",
    "_compute_average_scores_for_result",
    "_normalize_artifact_path",
    "_parent_key",
    "parse_few_shot_examples",
]
