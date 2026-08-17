# 文件作用：提供文档块树、单块详情以及块关联问答的查询接口。
# 关联说明：对接 app.services.doc_chunks，与 pipeline 路由生成的文档块产物配套查询。

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core.logger import logger
from app.services import milvus as milvus_service
from app.services import admin as admin_qa_service
from app.services.debug import (
    get_debug_items_by_source_chunk_id,
    get_debug_map,
    load_chunk_qa_items_from_artifacts,
)
from app.services.doc_chunks import (
    DOC_TREE_CHUNKS_COLLECTION,
    DOC_TREE_CHUNKS_SCHEMA_VERSION,
    build_doc_id as build_tree_doc_id,
    fetch_chunks_by_doc_id,
    get_chunk_by_id,
    list_docs_by_task,
    rebuild_doc_tree_chunks,
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
    Build the section tree from v2 section paths and attach independent content chunks.
    """

    node_map: Dict[str, Dict[str, Any]] = {}

    def get_node(section_path: str) -> Dict[str, Any]:
        if section_path in node_map:
            return node_map[section_path]
        node = {
            "section_path": section_path,
            "title": "",
            "children": [],
            "chunks": [],
        }
        node_map[section_path] = node
        return node

    root = get_node("")
    root["title"] = "ROOT"

    for ch in chunks:
        section_path = str(ch.get("section_path") or "").strip()
        if not section_path:
            raise ValueError("chunk v2 missing section_path")
        index_parts = [p for p in section_path.split(".") if p.strip()]
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

        leaf_node = get_node(section_path)
        if not leaf_node.get("title") and title_parts:
            leaf_node["title"] = title_parts[-1]
        leaf_node["chunks"].append(
            {
                "chunk_id": ch.get("chunk_id") or ch.get("id"),
                "chunk_index": ch.get("chunk_index"),
                "title_path": ch.get("title_path"),
                "section_path": section_path,
                "section_chunk_index": ch.get("section_chunk_index"),
                "fragment_group_id": ch.get("fragment_group_id"),
                "fragment_index": ch.get("fragment_index"),
                "fragment_count": ch.get("fragment_count"),
                "content_kind": ch.get("content_kind"),
            }
        )

    def sort_node(node: Dict[str, Any]) -> None:
        node["chunks"].sort(key=lambda x: int(x.get("chunk_index") or 0))

        def sort_key(n: Dict[str, Any]) -> Any:
            p = str(n.get("section_path") or "")
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


class ChunkRebuildRequest(BaseModel):
    task_id: str
    text: str
    original_filename: str
    chunk_size: int = 600
    split_type: Optional[str] = "markdown"
    text_split_min_length: Optional[int] = None
    text_split_max_length: Optional[int] = None
    chunk_overlap: Optional[int] = None
    prefix_max_depth: int = 4


@router.post("/rebuild")
async def rebuild_chunks(payload: ChunkRebuildRequest) -> Dict[str, Any]:
    """Re-chunk source text and explicitly replace one task's v2 rows."""
    try:
        from qa.chunking import build_tree_chunks

        task_id = str(payload.task_id or "").strip()
        original_filename = str(payload.original_filename or "").strip()
        text = str(payload.text or "")
        if not task_id:
            raise ValueError("task_id is required for explicit chunk rebuild")
        if not original_filename:
            raise ValueError("original_filename is required for explicit chunk rebuild")
        if not text.strip():
            raise ValueError("text is required for explicit chunk rebuild")
        doc_id = build_tree_doc_id(original_filename, text)
        _chunks_for_llm, chunks_meta, chunking_report = build_tree_chunks(
            text,
            chunk_size=max(1, int(payload.chunk_size)),
            original_filename=original_filename,
            task_id=task_id,
            doc_id=doc_id,
            prefix_max_depth=max(0, int(payload.prefix_max_depth)),
            split_type=payload.split_type,
            text_split_min_length=payload.text_split_min_length,
            text_split_max_length=payload.text_split_max_length,
            chunk_overlap=payload.chunk_overlap,
        )
        result = rebuild_doc_tree_chunks(chunks_meta, task_id=task_id)
        if not result.get("success"):
            raise HTTPException(
                status_code=503,
                detail=str(result.get("message") or "v2 chunk rebuild failed"),
            )
        result.update(
            {
                "doc_id": doc_id,
                "original_filename": original_filename,
                "chunking_report": chunking_report,
            }
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("explicit v2 chunk rebuild failed for task_id=%s", payload.task_id)
        raise HTTPException(status_code=500, detail=f"v2 chunk rebuild failed: {exc}")


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


@router.get("/by-doc/assets")
async def get_document_assets(
    doc_id: Optional[str] = Query(None, description="文档ID（doc_id 或 task_id 至少提供一个）"),
    task_id: Optional[str] = Query(None, description="任务ID。配合 original_filename 可精确锁定文档"),
    original_filename: Optional[str] = Query(None, description="原始文件名。配合 task_id 做精确过滤，不填则取第一个匹配文档"),
    include_full_text: bool = Query(True, description="返回一体化流程产生的完整纯文本文档（从 doc_content_chunks_v2 按序拼接）"),
    include_qas: bool = Query(True, description="返回该文档产生的全部 QA 对（标量精确查询，不丢失数据）"),
    include_chunks: bool = Query(False, description="返回所有文本块列表（含 chunk_id、标题路径、文本等元数据）"),
    qa_only_active: bool = Query(True, description="仅返回活跃状态的 QA（is_active=true）。设为 false 返回全部含已停用"),
    qa_page_size: int = Query(200, ge=1, le=200, description="QA 内部分页大小，不影响最终返回结果"),
) -> Dict[str, Any]:
    """
    一次获取指定文档在 qa-flow 中产生的全部数据资产。

    适用场景：RAG 模块在 qa-flow 流水线完成后，获取某个文档的完整纯文本和全部 QA 对，
    用于构建本地知识库、RAG 上下文或下游 LLM 应用。

    核心特性：
    - 纯 Milvus 标量查询，不走向量相似检索，确保精确完整
    - 自选返回内容：全文 / QA 对 / 文本块可按需组合
    - 全文由 doc_content_chunks_v2 按 chunk_index 排序拼接还原，等价于一体化流程输出
    """
    # ── 1. 解析标识符 ──
    resolved_doc_id = str(doc_id or "").strip() or None
    resolved_task_id = str(task_id or "").strip() or None
    resolved_filename = str(original_filename or "").strip() or None

    if not resolved_doc_id and not resolved_task_id:
        raise HTTPException(
            status_code=400,
            detail="至少需要提供 doc_id 或 task_id 之一",
        )

    # 只给了 task_id 时，通过 list_docs_by_task 解析 doc_id 和 filename
    if not resolved_doc_id and resolved_task_id:
        docs_result = list_docs_by_task(resolved_task_id)
        if not docs_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=str(docs_result.get("message") or "查询任务文档列表失败"),
            )
        docs: List[Dict[str, Any]] = docs_result.get("docs") or []
        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"未找到 task_id={resolved_task_id} 关联的文档（可能 pipeline 尚未完成或 chunk 入库失败）",
            )
        if resolved_filename:
            matched = [d for d in docs if d.get("original_filename") == resolved_filename]
            if not matched:
                available = [d.get("original_filename") for d in docs]
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到文件 '{resolved_filename}'。task_id={resolved_task_id} 下的文件: {available}",
                )
            doc_info = matched[0]
        else:
            doc_info = docs[0]
        resolved_doc_id = str(doc_info.get("doc_id") or "").strip()
        resolved_filename = resolved_filename or str(doc_info.get("original_filename") or "").strip()

    if not resolved_doc_id:
        raise HTTPException(
            status_code=400,
            detail="无法解析 doc_id，请直接提供 doc_id 参数",
        )

    # ── 2. 获取文本块（需要全文或块列表时） ──
    chunks: List[Dict[str, Any]] = []
    need_chunks = include_full_text or include_chunks
    if need_chunks:
        chunks_result = fetch_chunks_by_doc_id(
            resolved_doc_id,
            task_id=resolved_task_id,
            include_text=include_full_text or include_chunks,
        )
        if not chunks_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=str(chunks_result.get("message") or "查询文档块失败"),
            )
        chunks = chunks_result.get("chunks") or []
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=f"未找到 doc_id={resolved_doc_id} 的文档块（Milvus 中无此文档的 chunk 记录）",
            )
        # 从 chunks 补全元信息
        if not resolved_filename:
            resolved_filename = str(chunks[0].get("original_filename") or "").strip()
        if not resolved_task_id:
            resolved_task_id = str(chunks[0].get("task_id") or "").strip()

    # 仅需 QA 时也必须补全 task_id + filename，避免同名文件跨任务串数据。
    if include_qas and (not resolved_filename or not resolved_task_id) and not chunks:
        chunks_result = fetch_chunks_by_doc_id(
            resolved_doc_id,
            task_id=resolved_task_id,
            include_text=False,
        )
        if chunks_result.get("success"):
            temp_chunks = chunks_result.get("chunks") or []
            if temp_chunks:
                resolved_filename = str(temp_chunks[0].get("original_filename") or "").strip()
                if not resolved_task_id:
                    resolved_task_id = str(temp_chunks[0].get("task_id") or "").strip()

    # ── 3. 构建全文 ──
    full_text: Optional[str] = None
    if include_full_text:
        sorted_chunks = sorted(chunks, key=lambda c: int(c.get("chunk_index") or 0))
        full_text = "\n\n".join(
            str(c.get("text") or "") for c in sorted_chunks
        )

    # ── 4. 获取全部 QA ──
    qas: List[Dict[str, Any]] = []
    total_qas = 0
    if include_qas:
        if not resolved_filename:
            raise HTTPException(
                status_code=400,
                detail="无法确定文件名，无法查询 QA 对。请同时提供 original_filename 参数",
            )
        is_active_filter: Optional[bool] = True if qa_only_active else None
        page = 1
        while True:
            try:
                qa_result = admin_qa_service.list_qa_items(
                    task_id=resolved_task_id,
                    original_filename=resolved_filename,
                    is_active=is_active_filter,
                    page=page,
                    page_size=qa_page_size,
                )
            except admin_qa_service.AdminMilvusError as exc:
                raise HTTPException(status_code=503, detail=f"Milvus 查询 QA 失败: {exc}")
            except Exception as exc:
                logger.exception("list_qa_items failed for doc_id=%s", resolved_doc_id)
                raise HTTPException(status_code=500, detail=f"查询 QA 失败: {exc}")

            items = qa_result.get("items") or []
            qa_ids = [
                str(item.get("id") or "").strip()
                for item in items
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            ]
            debug_map = get_debug_map(qa_ids)
            for item in items:
                if not isinstance(item, dict):
                    continue
                qa_id = str(item.get("id") or "").strip()
                debug_detail = debug_map.get(qa_id) or {}
                enriched = dict(debug_detail)
                enriched.update(item)
                qas.append(enriched)
            pagination = qa_result.get("pagination") if isinstance(qa_result.get("pagination"), dict) else {}
            total_qas = int(pagination.get("total_items") or len(qas))
            total_pages = int(pagination.get("total_pages") or 0)
            if not items or page >= total_pages:
                break
            page += 1

    # ── 5. 组装响应 ──
    response: Dict[str, Any] = {
        "success": True,
        "doc_id": resolved_doc_id,
        "task_id": resolved_task_id,
        "original_filename": resolved_filename,
        "collection_name": DOC_TREE_CHUNKS_COLLECTION,
        "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
    }

    if include_full_text:
        response["full_text"] = full_text
        response["full_text_chars"] = len(full_text) if full_text else 0

    if include_qas:
        response["qas"] = qas
        response["total_qas"] = len(qas)
        response["reported_total_qas"] = total_qas

    if include_chunks:
        response["chunks"] = chunks
        response["total_chunks"] = len(chunks)

    return response


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
        "collection_name": DOC_TREE_CHUNKS_COLLECTION,
        "schema_version": DOC_TREE_CHUNKS_SCHEMA_VERSION,
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
    only_filtered_value = (
        only_filtered
        if isinstance(only_filtered, bool)
        else bool(getattr(only_filtered, "default", False))
    )
    source_field = _resolve_milvus_source_field()
    milvus_items: List[Dict[str, Any]] = []
    milvus_total = 0
    milvus_error: Optional[Exception] = None
    chunk_context: Dict[str, Any] = {}

    try:
        chunk_lookup = get_chunk_by_id(chunk_id)
        if chunk_lookup.get("success") and isinstance(chunk_lookup.get("chunk"), dict):
            chunk_context = dict(chunk_lookup["chunk"])
    except Exception:
        chunk_context = {}

    debug_items = get_debug_items_by_source_chunk_id(
        chunk_id,
        task_id=str(chunk_context.get("task_id") or "").strip() or None,
        original_filename=str(chunk_context.get("original_filename") or "").strip() or None,
        only_filtered=only_filtered_value,
    )

    if milvus_service.MILVUS_AVAILABLE and milvus_service.milvus_client:
        filter_expr = f"{source_field} == {json.dumps(str(chunk_id))}"
        if only_filtered_value:
            filter_expr += " and filtered == true"

        try:
            milvus_service.milvus_client.load()
        except Exception:
            pass

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
                offset=0,
                limit=16384,
            )
            milvus_items = [row for row in rows or [] if isinstance(row, dict)]
        except Exception as exc:
            milvus_error = exc
            logger.exception("query qa by chunk failed: %s", exc)

    merged_by_id: Dict[str, Dict[str, Any]] = {}
    for item in milvus_items:
        item_id = str(item.get("id") or "").strip()
        if item_id:
            merged_by_id[item_id] = item
    for item in debug_items:
        item_id = str(item.get("id") or "").strip()
        if item_id and item_id not in merged_by_id:
            merged_by_id[item_id] = item

    if merged_by_id:
        merged_items = sorted(
            merged_by_id.values(),
            key=lambda row: (
                0 if row.get("is_primary") else 1,
                int(row.get("created_at") or 0),
                str(row.get("id") or ""),
            ),
        )
        milvus_total = len(merged_items)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "success": True,
            "chunk_id": chunk_id,
            "source": "milvus" if milvus_items else "debug",
            "source_field": source_field if milvus_items else "source_chunk_id",
            "page": page,
            "page_size": page_size,
            "total": milvus_total,
            "items": merged_items[start:end],
        }

    task_id = str(chunk_context.get("task_id") or "").strip() or _resolve_chunk_task_id(chunk_id)
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
                "source_chunk_id": debug_detail.get("source_chunk_id"),
                "source_chunk_index": debug_detail.get("source_chunk_index"),
                "source_chunk_title_path": debug_detail.get("source_chunk_title_path"),
                "source_chunk_ids": debug_detail.get("source_chunk_ids") or [],
                "source_chunk_indexes": debug_detail.get("source_chunk_indexes") or [],
                "source_chunk_title_paths": debug_detail.get("source_chunk_title_paths") or [],
                "evidence_chunk_ids": debug_detail.get("evidence_chunk_ids") or [],
                "qa_generation_unit_id": debug_detail.get("qa_generation_unit_id"),
                "qa_generation_unit_text": debug_detail.get("qa_generation_unit_text"),
                "qa_generation_unit_index": debug_detail.get("qa_generation_unit_index"),
                "qa_generation_unit_type": debug_detail.get("qa_generation_unit_type"),
                "qa_generation_unit_mode": debug_detail.get("qa_generation_unit_mode"),
                "qa_generation_scenario_intent": debug_detail.get(
                    "qa_generation_scenario_intent"
                ),
                "qa_generation_reader_need": debug_detail.get("qa_generation_reader_need"),
                "qa_generation_material_ids": debug_detail.get(
                    "qa_generation_material_ids"
                )
                or [],
                "qa_generation_required_material_ids": debug_detail.get(
                    "qa_generation_required_material_ids"
                )
                or [],
                "qa_generation_optional_material_ids": debug_detail.get(
                    "qa_generation_optional_material_ids"
                )
                or [],
                "qa_generation_subject_label": debug_detail.get(
                    "qa_generation_subject_label"
                ),
                "evidence_mode": debug_detail.get("evidence_mode") or "text",
                "required_image_refs": debug_detail.get("required_image_refs") or [],
                "qa_generation_unit_source_chunk_indexes": debug_detail.get(
                    "qa_generation_unit_source_chunk_indexes"
                )
                or [],
                "qa_generation_unit_section_path": debug_detail.get(
                    "qa_generation_unit_section_path"
                ),
                "qa_generation_unit_quality_child_coverage": debug_detail.get(
                    "qa_generation_unit_quality_child_coverage"
                ),
                "evidence_hits": debug_detail.get("evidence_hits") or [],
                "retrieval_trace": debug_detail.get("retrieval_trace") or {},
                "source_fact_text": debug_detail.get("source_fact_text") or item.get("source_fact_text"),
                "answer_explanation": debug_detail.get("answer_explanation") or item.get("answer_explanation"),
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
