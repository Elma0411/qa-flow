# 文件作用：提供 Milvus 初始化、状态检查和管理接口。
# 关联说明：对接 app.services.milvus，和 search/admin 路由共同使用同一向量库服务。

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import CONFIG
from app.services import milvus as milvus_service

router = APIRouter()


class UpdatePayload(BaseModel):
    id: str
    knowledge_category: Optional[str] = None
    knowledge_category_reason: Optional[str] = None
    knowledge_category_confidence: Optional[float] = None
    question_type: Optional[str] = None
    question_type_reason: Optional[str] = None
    difficulty_level: Optional[str] = None
    difficulty_score: Optional[float] = None
    filtered: Optional[bool] = None
    average_score: Optional[float] = None


@router.post("/milvus/update", deprecated=True)
async def milvus_update(payload: UpdatePayload):
    """
    根据 id 更新部分字段。

    Milvus 不支持原地 update，这里采用「查询→删除→重新插入」策略，
    主要用于微调知识类别、题型、难度及过滤/平均分等语义字段。
    """
    try:
        if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
            raise HTTPException(status_code=503, detail="Milvus服务不可用")
        milvus_service.milvus_client.load()
        source_field = milvus_service._resolve_source_field_name()
        rows = milvus_service.milvus_client.query(
            expr=f'id == "{payload.id}"',
            output_fields=[
                "id",
                "task_id",
                "original_filename",
                source_field,
                "source_fact_text",
                "question",
                "answer",
                "question_type",
                "question_type_reason",
                "answer_explanation",
                "knowledge_category",
                "knowledge_category_reason",
                "knowledge_category_confidence",
                "difficulty_level",
                "difficulty_score",
                "llm_model",
                "embed_model",
                "embed_dim",
                "filtered",
                "average_score",
                "evaluation_method",
                "llm_scores",
                "llm_reasons",
                "local_scores",
                "created_at",
                "filter_basis",
                "embedding_vector",
            ],
            limit=1,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="记录不存在")
        record = rows[0]
        updated = dict(record)
        if payload.knowledge_category is not None:
            updated["knowledge_category"] = payload.knowledge_category
        if payload.knowledge_category_reason is not None:
            updated["knowledge_category_reason"] = payload.knowledge_category_reason
        if payload.knowledge_category_confidence is not None:
            updated["knowledge_category_confidence"] = float(payload.knowledge_category_confidence)
        if payload.question_type is not None:
            updated["question_type"] = payload.question_type
        if payload.question_type_reason is not None:
            updated["question_type_reason"] = payload.question_type_reason
        if payload.difficulty_level is not None:
            updated["difficulty_level"] = payload.difficulty_level
        if payload.difficulty_score is not None:
            updated["difficulty_score"] = float(payload.difficulty_score)
        if payload.filtered is not None:
            updated["filtered"] = payload.filtered
        if payload.average_score is not None:
            updated["average_score"] = float(payload.average_score)
        milvus_service.milvus_client.delete(expr=f'id == "{payload.id}"')
        milvus_service.milvus_client.flush()
        milvus_service.milvus_client.insert([updated])
        milvus_service.milvus_client.flush()
        return {"success": True, "id": payload.id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"更新失败: {str(exc)}")


class DeletePayload(BaseModel):
    ids: Optional[List[str]] = None
    task_id: Optional[str] = None
    original_filename: Optional[str] = None


@router.post("/milvus/delete", deprecated=True)
async def milvus_delete(payload: DeletePayload):
    """按 ids / task_id / original_filename 批量删除记录（至少提供一个条件）"""
    try:
        if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
            raise HTTPException(status_code=503, detail="Milvus服务不可用")
        exprs = []
        if payload.ids:
            quoted = ",".join([f'"{i}"' for i in payload.ids])
            exprs.append(f'id in [{quoted}]')
        if payload.task_id:
            exprs.append(f'task_id == "{payload.task_id}"')
        if payload.original_filename:
            exprs.append(f'original_filename == "{payload.original_filename}"')
        if not exprs:
            raise HTTPException(status_code=400, detail="请至少提供 ids、task_id 或 original_filename 之一")
        expr = " or ".join(exprs)
        res = milvus_service.milvus_client.delete(expr=expr)
        milvus_service.milvus_client.flush()
        return {"success": True, "expr": expr, "result": str(res)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(exc)}")


@router.get("/milvus-status")
async def get_milvus_status():
    """获取 Milvus 连接状态和基本统计信息"""
    try:
        status_info = {
            "milvus_available": milvus_service.MILVUS_AVAILABLE,
            "connected": bool(milvus_service.milvus_client),
            "embedding_model_loaded": bool(milvus_service.embedding_model),
            "collection_name": CONFIG["milvus"]["collection_name"]
            if milvus_service.MILVUS_AVAILABLE
            else None,
            "config": CONFIG["milvus"] if milvus_service.MILVUS_AVAILABLE else None,
            "milvus_lite_available": milvus_service.MILVUS_LITE_AVAILABLE,
            "using_milvus_lite": bool(
                CONFIG.get("milvus", {}).get("enable_milvus_lite", False)
            )
            and milvus_service.MILVUS_LITE_AVAILABLE,
        }
        if milvus_service.MILVUS_AVAILABLE and milvus_service.milvus_client:
            try:
                milvus_service.milvus_client.load()
                num_entities = milvus_service.milvus_client.num_entities
                status_info["collection_stats"] = {
                    "total_entities": num_entities,
                    "schema": [
                        field.name for field in milvus_service.milvus_client.schema.fields
                    ],
                }
            except Exception as exc:
                status_info["collection_error"] = str(exc)
        return status_info
    except Exception as exc:
        return {"error": f"获取Milvus状态失败: {str(exc)}"}


@router.post("/init-milvus")
async def initialize_milvus():
    """手动初始化 Milvus 连接与集合"""
    try:
        if not milvus_service.MILVUS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Milvus相关库未安装")
        success, message = milvus_service.init_milvus()
        if success:
            return {
                "success": True,
                "message": message,
                "collection_name": CONFIG["milvus"]["collection_name"],
            }
        raise HTTPException(status_code=503, detail=f"Milvus初始化失败: {message}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(exc)}")
