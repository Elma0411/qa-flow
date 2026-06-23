# 文件作用：定义管理端路由共用请求模型、筛选参数和评分提取工具。
# 关联说明：被 admin_v1_item_routes 和 admin_v1_job_routes 复用，避免管理端模型和评分逻辑重复。

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import LOCAL_EVALUATION_METRICS
from app.services import milvus as milvus_service


class TriState(str, Enum):
    true = "true"
    false = "false"
    all = "all"


def _tristate_to_optional_bool(value: TriState) -> Optional[bool]:
    if value == TriState.all:
        return None
    return value == TriState.true


class QATagPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    knowledge_category: Optional[str] = None
    knowledge_category_reason: Optional[str] = None
    knowledge_category_confidence: Optional[float] = None
    question_type: Optional[str] = None
    question_type_reason: Optional[str] = None
    difficulty_level: Optional[str] = None
    difficulty_score: Optional[float] = None
    filtered: Optional[bool] = None


class AdminMetaPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_active: Optional[bool] = None
    review_status: Optional[str] = None
    review_note: Optional[str] = None


class IngestConsolidatedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_file: str = Field(
        ...,
        description="要入库的 consolidated JSON 文件名或路径（仅允许读取 outputs 目录下文件）",
    )



class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(default_factory=list)


class AutoFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    score_threshold: float = 0.7


class EvaluationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: Selection
    evaluation_method: Literal["llm", "local"]
    criteria_list: Optional[List[str]] = None
    force: bool = False
    write_back: bool = True
    include_inactive: bool = False
    auto_filter: Optional[AutoFilterConfig] = None


class UnsupervisedEvaluationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selection: Selection
    force: bool = False
    write_back: bool = True
    include_inactive: bool = False


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(default_factory=list)
    include_inactive: bool = False


def _ensure_unsupervised_fields_available() -> None:
    if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
        raise HTTPException(status_code=503, detail="Milvus服务不可用，请先启动并连接向量库")
    fields = {f.name for f in milvus_service.milvus_client.schema.fields}
    required = {
        "faithfulness",
        "unsupervised_method",
        "unsupervised_scores",
        "unsupervised_meta",
    }
    missing = sorted(required - fields)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Milvus schema 缺少字段 {missing}，无法写入无监督评价结果；"
                "请运行 `python scripts/rebuild_milvus_collection.py` 重建集合后重试"
            ),
        )



def _compute_llm_average(ev: Dict[str, Any], criteria_list: List[str]) -> float:
    values: List[float] = []
    for metric in criteria_list:
        entry = ev.get(metric)
        if isinstance(entry, dict) and isinstance(entry.get("score"), (int, float)):
            values.append(float(entry["score"]))
    return float(sum(values) / len(values)) if values else 0.0


def _extract_llm_scores_reasons(
    ev: Dict[str, Any], criteria_list: List[str]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    scores: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    for metric in criteria_list:
        entry = ev.get(metric)
        if not isinstance(entry, dict):
            continue
        score_val = entry.get("score")
        if isinstance(score_val, (int, float)):
            scores[metric] = float(score_val)
        reason_val = entry.get("reasons") or entry.get("reason") or entry.get("explanation")
        if reason_val is not None:
            reasons[metric] = str(reason_val)
    return scores, reasons


def _extract_local_scores(ev: Dict[str, Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for metric in LOCAL_EVALUATION_METRICS:
        entry = ev.get(metric)
        if isinstance(entry, dict) and isinstance(entry.get("score"), (int, float)):
            scores[metric] = float(entry["score"])
    return scores

__all__ = [
    'AdminMetaPatch',
    'AutoFilterConfig',
    'EvaluationJobRequest',
    'ExportRequest',
    'IngestConsolidatedRequest',
    'QATagPatch',
    'Selection',
    'TriState',
    'UnsupervisedEvaluationJobRequest',
    '_compute_llm_average',
    '_ensure_unsupervised_fields_available',
    '_extract_llm_scores_reasons',
    '_extract_local_scores',
    '_tristate_to_optional_bool',
]
