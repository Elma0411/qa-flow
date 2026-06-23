# 文件作用：提供文档块树、单块详情以及块关联问答的查询接口。
# 关联说明：对接 app.services.doc_chunks，与 pipeline 路由生成的文档块产物配套查询。

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.logger import logger
from app.services import milvus as milvus_service
from app.services import admin as admin_qa_service
from app.services.debug import load_chunk_qa_items_from_artifacts
from app.services.doc_chunks import (
    fetch_chunks_by_doc_id,
    get_chunk_by_id,
    list_docs_by_task,
)

router = APIRouter(prefix="/doc-chunks", tags=["doc-chunks"])


def _resolve_milvus_source_field() -> str:
    try:
        if not milvus_service.milvus_client:
            return "source"
        names = {f.name for f in milvus_service.milvus_client.schema.fields}
        if "source" in names:
            return "source"
        if "source_id" in names:
            return "source_id"
    except Exception:
        return "source"
    return "source"


def _safe_split_title_path(title_path: str) -> List[str]:
    raw = str(title_path or "").strip()
    if not raw:
        return []
    if ">" in raw:
        return [p.strip() for p in raw.split(">") if p.strip()]
    if "/" in raw:
        return [p.strip() for p in raw.split("/") if p.strip()]
    return [raw]


def _build_tree_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a hierarchical tree purely from leaf chunks.
    - Nodes are keyed by `index_path` prefixes (split by '.').
    - Leaf chunks are attached to their `index_path` node.
    """

    node_map: Dict[str, Dict[str, Any]] = {}

    def get_node(index_path: str) -> Dict[str, Any]:
        if index_path in node_map:
            return node_map[index_path]
        node = {
            "index_path": index_path,
            "title": "",
            "children": [],
            "chunks": [],
        }
        node_map[index_path] = node
        return node

    root = get_node("")
    root["title"] = "ROOT"

    for ch in chunks:
        index_path = str(ch.get("index_path") or "").strip()
        if not index_path:
            index_path = str(ch.get("chunk_index") or "").strip() or "0"
        index_parts = [p for p in index_path.split(".") if p.strip()]
        title_parts = _safe_split_title_path(str(ch.get("title_path") or ""))
        if len(title_parts) >= len(index_parts) and index_parts:
            title_parts_aligned = title_parts[-len(index_parts) :]
        else:
            title_parts_aligned = title_parts

        parent_path = ""
        for depth, part in enumerate(index_parts, start=1):
            cur_path = part if not parent_path else f"{parent_path}.{part}"
            parent_node = get_node(parent_path)
            cur_node = get_node(cur_path)
            if cur_node not in parent_node["children"]:
                parent_node["children"].append(cur_node)

            if not cur_node.get("title"):
                idx = depth - 1
                if idx < len(title_parts_aligned):
                    cur_node["title"] = title_parts_aligned[idx]
                else:
                    cur_node["title"] = part
            parent_path = cur_path

        leaf_node = get_node(index_path)
        if not leaf_node.get("title") and title_parts:
            leaf_node["title"] = title_parts[-1]
        leaf_node["chunks"].append(
            {
                "chunk_id": ch.get("chunk_id") or ch.get("id"),
                "chunk_index": ch.get("chunk_index"),
                "title_path": ch.get("title_path"),
                "index_path": index_path,
            }
        )

    def sort_node(node: Dict[str, Any]) -> None:
        node["chunks"].sort(key=lambda x: int(x.get("chunk_index") or 0))

        def sort_key(n: Dict[str, Any]) -> Any:
            p = str(n.get("index_path") or "")
            last = p.split(".")[-1] if p else ""
            try:
                return (0, int(last))
            except Exception:
                return (1, last)

        node["children"].sort(key=sort_key)
        for child in node["children"]:
            sort_node(child)

    sort_node(root)
    return root


def _resolve_chunk_task_id(chunk_id: str) -> Optional[str]:
    try:
        result = get_chunk_by_id(chunk_id)
    except Exception:
        return None
    if not result.get("success"):
        return None
    chunk = result.get("chunk") if isinstance(result.get("chunk"), dict) else {}
    task_id = str((chunk or {}).get("task_id") or "").strip()
    return task_id or None


@router.get("/by-task/{task_id}")
async def docs_by_task(task_id: str) -> Dict[str, Any]:
    return list_docs_by_task(task_id)


@router.get("/tree")
async def doc_tree(
    doc_id: str = Query(..., description="doc_id"),
    task_id: Optional[str] = Query(None, description="task_id（可选：指定则按任务隔离）"),
) -> Dict[str, Any]:
    result = fetch_chunks_by_doc_id(doc_id, task_id=task_id, include_text=False)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=str(result.get("message") or "query_failed"))

    chunks = result.get("chunks") or []
    tree = _build_tree_from_chunks(chunks)
    return {
        "success": True,
        "task_id": task_id,
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "tree": tree,
    }


@router.get("/{chunk_id}")
async def chunk_detail(chunk_id: str) -> Dict[str, Any]:
    result = get_chunk_by_id(chunk_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=str(result.get("message") or "chunk_not_found"))
    return result


@router.get("/{chunk_id}/qa")
async def qa_by_chunk(
    chunk_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    only_filtered: bool = Query(False, description="是否只返回过滤后的问答对"),
) -> Dict[str, Any]:
    source_field = _resolve_milvus_source_field()
    milvus_items: List[Dict[str, Any]] = []
    milvus_total = 0
    milvus_error: Optional[Exception] = None

    if milvus_service.MILVUS_AVAILABLE and milvus_service.milvus_client:
        filter_expr = f"{source_field} == {json.dumps(str(chunk_id))}"
        if only_filtered:
            filter_expr += " and filtered == true"

        try:
            milvus_service.milvus_client.load()
        except Exception:
            pass

        offset = (page - 1) * page_size
        output_fields = [
            "id",
            "task_id",
            "original_filename",
            source_field,
            "question",
            "answer",
            "question_type",
            "answer_explanation",
            "knowledge_category",
            "knowledge_category_reason",
            "knowledge_category_confidence",
            "filtered",
            "average_score",
            "evaluation_method",
            "created_at",
            "is_primary",
            "is_augmented",
            "variant_of",
        ]
        try:
            rows = milvus_service.milvus_client.query(
                expr=filter_expr,
                output_fields=output_fields,
                offset=offset,
                limit=page_size,
            )
            milvus_items = [row for row in rows or [] if isinstance(row, dict)]
            try:
                count_rows = milvus_service.milvus_client.query(
                    expr=filter_expr,
                    output_fields=["id"],
                    limit=16384,
                )
                milvus_total = len(count_rows)
            except Exception:
                milvus_total = len(milvus_items)
        except Exception as exc:
            milvus_error = exc
            logger.exception("query qa by chunk failed: %s", exc)

    if milvus_items or milvus_total > 0:
        return {
            "success": True,
            "chunk_id": chunk_id,
            "source": "milvus",
            "source_field": source_field,
            "page": page,
            "page_size": page_size,
            "total": milvus_total,
            "items": milvus_items,
        }

    task_id = _resolve_chunk_task_id(chunk_id)
    if task_id:
        artifact_result = load_chunk_qa_items_from_artifacts(
            task_id=task_id,
            chunk_id=chunk_id,
            only_filtered=only_filtered,
            page=page,
            page_size=page_size,
        )
        if artifact_result.get("success"):
            return {
                "success": True,
                "chunk_id": chunk_id,
                "task_id": task_id,
                "source": "artifacts",
                "source_field": "source_chunk_id",
                "page": artifact_result.get("page"),
                "page_size": artifact_result.get("page_size"),
                "total": artifact_result.get("total", 0),
                "items": artifact_result.get("items") or [],
            }

    if milvus_error is not None:
        raise HTTPException(status_code=500, detail=f"查询失败: {milvus_error}")

    return {
        "success": True,
        "chunk_id": chunk_id,
        "task_id": task_id,
        "source": "milvus" if milvus_service.MILVUS_AVAILABLE and milvus_service.milvus_client else "artifacts",
        "source_field": source_field if milvus_service.MILVUS_AVAILABLE and milvus_service.milvus_client else "source_chunk_id",
        "page": page,
        "page_size": page_size,
        "total": 0,
        "items": [],
    }


@router.get("/{chunk_id}/debug")
async def chunk_debug(
    chunk_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    only_filtered: bool = Query(False, description="是否只返回过滤后的问答对"),
) -> Dict[str, Any]:
    detail = await chunk_detail(chunk_id)
    qa_result = await qa_by_chunk(
        chunk_id=chunk_id,
        page=page,
        page_size=page_size,
        only_filtered=only_filtered,
    )
    qa_source = str(qa_result.get("source") or "").strip().lower()
    qa_items = qa_result.get("items") if isinstance(qa_result, dict) else []
    enriched_items: List[Dict[str, Any]] = []
    for item in qa_items or []:
        if not isinstance(item, dict):
            continue
        if qa_source != "milvus":
            enriched_items.append(dict(item))
            continue
        qa_id = str(item.get("id") or "").strip()
        if not qa_id:
            enriched_items.append(item)
            continue
        try:
            debug_detail = admin_qa_service.get_qa_item(qa_id)
        except Exception as exc:
            logger.warning("load qa debug detail failed qa_id=%s err=%s", qa_id, exc)
            enriched_items.append(item)
            continue
        merged = dict(item)
        merged.update(
            {
                "source_anchor_text": debug_detail.get("source_anchor_text"),
                "source_chunk_id": debug_detail.get("source_chunk_id"),
                "source_chunk_index": debug_detail.get("source_chunk_index"),
                "source_chunk_title_path": debug_detail.get("source_chunk_title_path"),
                "evidence_chunk_ids": debug_detail.get("evidence_chunk_ids") or [],
                "qa_generation_unit_id": debug_detail.get("qa_generation_unit_id"),
                "qa_generation_unit_text": debug_detail.get("qa_generation_unit_text"),
                "retrieval_trace": debug_detail.get("retrieval_trace") or {},
                "source_fact_text": debug_detail.get("source_fact_text") or item.get("source_fact_text"),
                "question_type_reason": debug_detail.get("question_type_reason"),
                "answer_explanation": debug_detail.get("answer_explanation") or item.get("answer_explanation"),
                "difficulty_level": debug_detail.get("difficulty_level"),
                "difficulty_score": debug_detail.get("difficulty_score"),
                "evaluation": debug_detail.get("evaluation"),
                "unsupervised_evaluation": debug_detail.get("unsupervised_evaluation"),
                "similar_questions": debug_detail.get("similar_questions") or [],
            }
        )
        enriched_items.append(merged)

    qa_view_items = sorted(
        enriched_items,
        key=lambda row: (
            0 if row.get("is_primary") else 1,
            int(row.get("created_at") or 0),
            str(row.get("id") or ""),
        ),
        reverse=False,
    )
    return {
        "success": True,
        "chunk": detail.get("chunk"),
        "qa": {
            "chunk_id": chunk_id,
            "source": qa_result.get("source"),
            "source_field": qa_result.get("source_field"),
            "page": qa_result.get("page"),
            "page_size": qa_result.get("page_size"),
            "total": qa_result.get("total"),
            "items": qa_view_items,
        },
    }
