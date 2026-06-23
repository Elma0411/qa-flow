# 文件作用：提供管理端问答条目的查询、更新、删除和导出接口。
# 关联说明：依赖 admin_v1_common 和 app.services.admin，专注问答条目管理。

import asyncio
import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import CONFIG
from app.routers.admin_v1_common import (
    AdminMetaPatch,
    IngestConsolidatedRequest,
    QATagPatch,
    TriState,
    _tristate_to_optional_bool,
)
from app.services import admin as admin_qa_service
from app.services import milvus as milvus_service
from app.services.admin import batch_upsert, delete_meta, get_meta_map, upsert_meta

router = APIRouter()

@router.post("/ingest-consolidated")
async def ingest_consolidated(payload: IngestConsolidatedRequest) -> Dict[str, Any]:
    """
    Re-ingest an existing consolidated JSON into Milvus.

    Useful for backfilling augmented queries into Milvus after upgrading the storage logic.
    The file is resolved by basename under CONFIG['outputs_dir'].
    """
    if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
        raise HTTPException(status_code=503, detail="Milvus服务不可用，请先启动并连接向量库")
    raw = str(payload.output_file or "")
    name = os.path.basename(raw.replace("\\", "/"))
    if not name:
        raise HTTPException(status_code=400, detail="output_file 不能为空")
    full_path = os.path.join(CONFIG["outputs_dir"], name)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {name}")
    try:
        result = await asyncio.to_thread(
            milvus_service.store_qa_pairs_to_milvus, full_path, True
        )
        return {"success": True, "output_file": name, "full_path": full_path, "milvus": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"入库失败: {exc}")


@router.get("/qa-items")
async def list_qa_items(
    task_id: Optional[str] = Query(None),
    original_filename: Optional[str] = Query(None),
    knowledge_category: Optional[List[str]] = Query(None),
    question_type: Optional[List[str]] = Query(None),
    difficulty_level: Optional[List[str]] = Query(None),
    filtered: TriState = Query(TriState.all),
    evaluated: TriState = Query(TriState.all),
    is_active: TriState = Query(TriState.true),
    review_status: Optional[str] = Query(None),
    min_avg_score: Optional[float] = Query(None),
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
) -> Dict[str, Any]:
    try:
        return admin_qa_service.list_qa_items(
            task_id=task_id,
            original_filename=original_filename,
            knowledge_categories=knowledge_category,
            question_types=question_type,
            difficulty_levels=difficulty_level,
            filtered=_tristate_to_optional_bool(filtered),
            evaluated=_tristate_to_optional_bool(evaluated),
            is_active=_tristate_to_optional_bool(is_active),
            review_status=review_status,
            min_avg_score=min_avg_score,
            q=q,
            page=page,
            page_size=page_size,
        )
    except admin_qa_service.AdminMilvusError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询失败: {exc}")


@router.get("/qa-items/{qa_id}")
async def get_qa_item(qa_id: str) -> Dict[str, Any]:
    try:
        return admin_qa_service.get_qa_item(qa_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="记录不存在")
    except admin_qa_service.AdminMilvusError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取详情失败: {exc}")


@router.patch("/qa-items/{qa_id}")
async def patch_qa_item(qa_id: str, payload: QATagPatch) -> Dict[str, Any]:
    patch = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not patch:
        raise HTTPException(status_code=400, detail="没有任何可更新字段")
    try:
        return admin_qa_service.update_qa_item_fields(qa_id, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail="记录不存在")
    except admin_qa_service.AdminMilvusError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"更新失败: {exc}")


@router.patch("/qa-items/{qa_id}/admin-meta")
async def patch_admin_meta(qa_id: str, payload: AdminMetaPatch) -> Dict[str, Any]:
    patch = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not patch:
        raise HTTPException(status_code=400, detail="没有任何可更新字段")
    try:
        meta = upsert_meta(
            qa_id,
            is_active=patch.get("is_active"),
            review_status=patch.get("review_status"),
            review_note=patch.get("review_note"),
        )
        return {"id": qa_id, "admin": meta.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"更新治理元数据失败: {exc}")


class BatchUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(default_factory=list)
    patch: QATagPatch


@router.post("/qa-items/batch-update")
async def batch_update(payload: BatchUpdateRequest) -> Dict[str, Any]:
    ids = [str(x) for x in payload.ids if x]
    patch = payload.patch.model_dump(exclude_unset=True, exclude_none=True)
    if not ids:
        raise HTTPException(status_code=400, detail="ids 不能为空")
    if not patch:
        raise HTTPException(status_code=400, detail="patch 不能为空")
    try:
        result = admin_qa_service.batch_update_fields(ids, patch)
        return {"success": True, **result}
    except admin_qa_service.AdminMilvusError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"批量更新失败: {exc}")


class BatchAdminUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(default_factory=list)
    patch: AdminMetaPatch


@router.post("/qa-items/batch-admin-update")
async def batch_admin_update(payload: BatchAdminUpdateRequest) -> Dict[str, Any]:
    ids = [str(x) for x in payload.ids if x]
    patch = payload.patch.model_dump(exclude_unset=True, exclude_none=True)
    if not ids:
        raise HTTPException(status_code=400, detail="ids 不能为空")
    if not patch:
        raise HTTPException(status_code=400, detail="patch 不能为空")
    try:
        meta_map = batch_upsert(
            ids,
            is_active=patch.get("is_active"),
            review_status=patch.get("review_status"),
            review_note=patch.get("review_note"),
        )
        return {
            "success": True,
            "updated": [str(k) for k in meta_map.keys()],
            "admin": {k: v.to_dict() for k, v in meta_map.items()},
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"批量治理更新失败: {exc}")


class BatchDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: List[str] = Field(default_factory=list)
    mode: Literal["soft", "hard"] = "soft"
    review_note: Optional[str] = None
    backup_enabled: bool = True


@router.post("/qa-items/batch-delete")
async def batch_delete(payload: BatchDeleteRequest) -> Dict[str, Any]:
    ids = [str(x) for x in payload.ids if x]
    if not ids:
        raise HTTPException(status_code=400, detail="ids 不能为空")
    if payload.mode == "soft":
        meta_map = batch_upsert(
            ids,
            is_active=False,
            review_status="deleted",
            review_note=payload.review_note,
        )
        return {"success": True, "mode": "soft", "updated": list(meta_map.keys())}

    # hard delete (optional backup)
    backup_path: Optional[str] = None
    if payload.backup_enabled:
        try:
            records = admin_qa_service.fetch_records_by_ids(ids)
            meta_map = get_meta_map(ids)
            items: List[Dict[str, Any]] = []
            for qa_id, rec in records.items():
                rec = dict(rec)
                rec.pop("embedding_vector", None)
                meta = meta_map.get(qa_id)
                rec["admin"] = meta.to_dict() if meta else {
                    "id": qa_id,
                    "is_active": True,
                    "review_status": None,
                    "review_note": None,
                    "updated_at": None,
                }
                items.append(rec)
            backup_path = admin_qa_service.export_items_to_json(items, prefix="hard_delete_backup")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"备份失败，已中止硬删: {exc}")

    try:
        delete_res = admin_qa_service.hard_delete(ids)
        removed_meta = delete_meta(ids)
        return {
            "success": True,
            "mode": "hard",
            "backup_path": backup_path,
            "milvus": delete_res,
            "admin_meta_removed": removed_meta,
        }
    except admin_qa_service.AdminMilvusError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"硬删失败: {exc}")


class QASearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_text: str
    top_k: int = 10
    task_id: Optional[str] = None
    filtered: TriState = TriState.all
    min_avg_score: Optional[float] = None
    knowledge_category: Optional[List[str]] = None
    question_type: Optional[List[str]] = None
    difficulty_level: Optional[List[str]] = None
    is_active: TriState = TriState.true


@router.post("/qa-search")
async def qa_search(payload: QASearchRequest) -> Dict[str, Any]:
    if not payload.query_text or not payload.query_text.strip():
        raise HTTPException(status_code=400, detail="query_text 不能为空")
    try:
        only_filtered = _tristate_to_optional_bool(payload.filtered)
        result = milvus_service.search_qa_pairs_in_milvus(
            query_text=payload.query_text,
            top_k=max(1, min(200, int(payload.top_k))),
            task_id=payload.task_id,
            only_filtered=only_filtered,
            min_avg_score=payload.min_avg_score,
            categories=payload.knowledge_category,
            question_types=payload.question_type,
            difficulty_levels=payload.difficulty_level,
        )
        if not result.get("success"):
            raise HTTPException(status_code=503, detail=result.get("message") or "搜索失败")
        rows = result.get("results") or []
        ids = [str(r.get("id")) for r in rows if r.get("id")]
        meta_map = get_meta_map(ids)
        active_filter = _tristate_to_optional_bool(payload.is_active)
        enriched: List[Dict[str, Any]] = []
        for r in rows:
            qa_id = str(r.get("id") or "")
            if not qa_id:
                continue
            meta = meta_map.get(qa_id)
            if not meta:
                meta = {
                    "id": qa_id,
                    "is_active": True,
                    "review_status": None,
                    "review_note": None,
                    "updated_at": None,
                }
            else:
                meta = meta.to_dict()
            if active_filter is not None and bool(meta.get("is_active", True)) != bool(active_filter):
                continue
            evaluated_flag = bool(r.get("evaluation_method")) or bool(r.get("llm_scores")) or bool(r.get("local_scores"))
            rr = dict(r)
            rr["evaluated"] = evaluated_flag
            rr["admin"] = meta
            enriched.append(rr)
        result["results"] = enriched
        result["returned"] = len(enriched)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"语义搜索失败: {exc}")

__all__ = ['router']
