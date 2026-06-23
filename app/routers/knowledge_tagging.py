# 文件作用：提供文档知识类别预测接口。
# 关联说明：对接 app.services.knowledge_tagging，供 pipeline 或外部调用知识分类能力。

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.knowledge_tagging import classify_document_text

router = APIRouter(prefix="/knowledge-tagging", tags=["knowledge-tagging"])


class KnowledgeTaggingRequest(BaseModel):
    text: str = Field(..., description="待分类文本（建议：OCR 后全文或前几段）")
    filename: Optional[str] = Field(None, description="可选：文件名（有助于命中标准号等特征）")
    classifier_mode: Optional[str] = Field(
        None,
        description="分类器: doc_level3_rule=新规则分类器, legacy_model=旧本地模型；不填默认 doc_level3_rule",
    )


@router.post("/predict")
async def predict_knowledge_tag(payload: KnowledgeTaggingRequest) -> Dict[str, Any]:
    """
    Predict the finest (三级) single label path for an input text.
    """
    try:
        result = classify_document_text(
            payload.text,
            filename=payload.filename or "",
            classifier_mode=payload.classifier_mode,
        )
        return {
            "knowledge_category": result.knowledge_category,
            "knowledge_category_confidence": result.knowledge_category_confidence,
            "knowledge_category_reason": result.knowledge_category_reason,
            "knowledge_category_source": result.knowledge_category_source,
            "knowledge_category_detail": result.knowledge_category_detail,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
