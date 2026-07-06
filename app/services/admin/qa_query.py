# 文件作用：封装管理端问答条目的 Milvus 查询与分页读取。
# 关联说明：读取 Milvus 和 meta 数据；写入操作放在 qa_write.py。

from typing import Any, Dict, List, Optional

from app.services import milvus as milvus_service
from .meta import get_meta_map
from .qa_common import (
    _default_admin_meta,
    _escape_expr_value,
    _expr_in,
    _fetch_rows,
    _get_allowed_fields,
    _is_evaluated,
    _parse_json_field,
    _resolve_source_field,
)
from app.services.debug import get_debug_map


def _build_milvus_expr(
    *,
    task_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    original_filenames: Optional[List[str]] = None,
    knowledge_categories: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    filtered: Optional[bool] = None,
    evaluated: Optional[bool] = None,
    min_avg_score: Optional[float] = None,
) -> str:
    parts: List[str] = []
    if task_id:
        parts.append(f'task_id == "{_escape_expr_value(task_id)}"')
    names: List[str] = []
    if original_filename:
        names.append(str(original_filename))
    if original_filenames:
        names.extend([str(name) for name in original_filenames if name])
    names = list(dict.fromkeys([name for name in names if name]))
    if names:
        if len(names) == 1:
            parts.append(f'original_filename == "{_escape_expr_value(names[0])}"')
        else:
            parts.append(_expr_in("original_filename", names))
    if knowledge_categories:
        parts.append(_expr_in("knowledge_category", knowledge_categories))
    if question_types:
        parts.append(_expr_in("question_type", question_types))
    if difficulty_levels:
        parts.append(_expr_in("difficulty_level", difficulty_levels))
    if filtered is not None:
        parts.append(f"filtered == {str(bool(filtered)).lower()}")
    if evaluated is not None:
        parts.append('evaluation_method != ""' if evaluated else 'evaluation_method == ""')
    if min_avg_score is not None:
        parts.append(f"average_score >= {float(min_avg_score)}")
    return " and ".join(parts) if parts else 'id != ""'


def list_qa_items(
    *,
    task_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    original_filenames: Optional[List[str]] = None,
    knowledge_categories: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    filtered: Optional[bool] = None,
    evaluated: Optional[bool] = None,
    min_avg_score: Optional[float] = None,
    is_active: Optional[bool] = True,
    review_status: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> Dict[str, Any]:
    expr = _build_milvus_expr(
        task_id=task_id,
        original_filename=original_filename,
        original_filenames=original_filenames,
        knowledge_categories=knowledge_categories,
        question_types=question_types,
        difficulty_levels=difficulty_levels,
        filtered=filtered,
        evaluated=evaluated,
        min_avg_score=min_avg_score,
    )
    source_field = _resolve_source_field()
    allowed_fields = set(_get_allowed_fields())
    output_fields = [
        "id",
        "task_id",
        "original_filename",
        source_field,
        "source_fact_text",
        "question",
        "answer",
        "knowledge_category",
        "question_type",
        "difficulty_level",
        "filtered",
        "average_score",
        "faithfulness",
        "evaluation_method",
        "unsupervised_method",
        "unsupervised_scores",
        "created_at",
        "filter_basis",
        "is_primary",
        "is_augmented",
        "variant_of",
    ]
    output_fields = [field for field in output_fields if field in allowed_fields]
    rows = _fetch_rows(expr, output_fields)
    ids = [str(row.get("id")) for row in rows if row.get("id")]
    meta_map = get_meta_map(ids)

    needle = (q or "").strip().lower()
    filtered_items: List[Dict[str, Any]] = []
    for row in rows:
        qa_id = str(row.get("id") or "")
        if not qa_id:
            continue
        meta = meta_map.get(qa_id) or _default_admin_meta(qa_id)
        if is_active is not None and bool(meta.is_active) != bool(is_active):
            continue
        if review_status is not None and (meta.review_status or "") != str(review_status):
            continue
        if needle:
            haystack = " ".join(
                [
                    str(row.get("question") or ""),
                    str(row.get("answer") or ""),
                    str(row.get("source_fact_text") or ""),
                    str(row.get(source_field) or ""),
                ]
            ).lower()
            if needle not in haystack:
                continue
        filtered_items.append(
            {
                "id": qa_id,
                "task_id": row.get("task_id"),
                "original_filename": row.get("original_filename"),
                "source": row.get(source_field),
                "source_fact_text": row.get("source_fact_text"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "knowledge_category": row.get("knowledge_category"),
                "question_type": row.get("question_type"),
                "difficulty_level": row.get("difficulty_level"),
                "filtered": row.get("filtered"),
                "average_score": row.get("average_score"),
                "faithfulness": row.get("faithfulness"),
                "evaluation_method": row.get("evaluation_method"),
                "unsupervised_method": row.get("unsupervised_method"),
                "unsupervised_scores": _parse_json_field(row.get("unsupervised_scores")),
                "evaluated": _is_evaluated(row),
                "created_at": row.get("created_at"),
                "filter_basis": row.get("filter_basis"),
                "is_primary": row.get("is_primary"),
                "is_augmented": row.get("is_augmented"),
                "variant_of": row.get("variant_of"),
                "admin": meta.to_dict(),
            }
        )

    filtered_items.sort(
        key=lambda item: (item.get("created_at") or 0, item.get("id") or ""),
        reverse=True,
    )

    safe_page = max(1, int(page))
    safe_size = max(1, min(200, int(page_size)))
    start = (safe_page - 1) * safe_size
    end = start + safe_size
    total = len(filtered_items)
    return {
        "filters": {
            "task_id": task_id,
            "original_filename": original_filename,
            "original_filenames": original_filenames,
            "knowledge_categories": knowledge_categories,
            "question_types": question_types,
            "difficulty_levels": difficulty_levels,
            "filtered": filtered,
            "evaluated": evaluated,
            "min_avg_score": min_avg_score,
            "is_active": is_active,
            "review_status": review_status,
            "q": q,
        },
        "pagination": {
            "page": safe_page,
            "page_size": safe_size,
            "total_items": total,
            "total_pages": (total + safe_size - 1) // safe_size if total else 0,
        },
        "items": filtered_items[start:end],
    }


def get_qa_item(qa_id: str) -> Dict[str, Any]:
    source_field = _resolve_source_field()
    allowed_fields = set(_get_allowed_fields())
    output_fields = [
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
        "faithfulness",
        "evaluation_method",
        "llm_scores",
        "llm_reasons",
        "local_scores",
        "unsupervised_method",
        "unsupervised_scores",
        "unsupervised_meta",
        "created_at",
        "filter_basis",
        "is_primary",
        "is_augmented",
        "variant_of",
    ]
    output_fields = [field for field in output_fields if field in allowed_fields]
    rows = milvus_service.milvus_client.query(  # type: ignore[union-attr]
        expr=f'id == "{_escape_expr_value(str(qa_id))}"',
        output_fields=output_fields,
        limit=1,
    )
    if not rows:
        raise KeyError("记录不存在")

    row = rows[0]
    qa_id = str(row.get("id"))
    meta_map = get_meta_map([qa_id])
    meta = meta_map.get(qa_id) or _default_admin_meta(qa_id)

    llm_scores = _parse_json_field(row.get("llm_scores"))
    llm_reasons = _parse_json_field(row.get("llm_reasons"))
    local_scores = _parse_json_field(row.get("local_scores"))
    evaluation: Dict[str, Any] = {}
    if llm_scores or llm_reasons:
        evaluation["llm"] = {"scores": llm_scores, "reasons": llm_reasons}
    if local_scores:
        evaluation["local"] = {"scores": local_scores}

    unsup_method = str(row.get("unsupervised_method") or "")
    unsup_scores = _parse_json_field(row.get("unsupervised_scores"))
    unsup_meta = _parse_json_field(row.get("unsupervised_meta"))
    unsupervised_evaluation: Optional[Dict[str, Any]] = None
    if unsup_method or unsup_scores or unsup_meta:
        unsupervised_evaluation = {
            "method": unsup_method or None,
            "scores": unsup_scores,
            "meta": unsup_meta,
        }

    similar_questions: List[Dict[str, Any]] = []
    try:
        is_primary_flag = bool(row.get("is_primary")) and not bool(row.get("is_augmented"))
        if is_primary_flag and "variant_of" in allowed_fields:
            variant_fields = [
                "id",
                "question",
                "answer",
                "question_type",
                "answer_explanation",
                "average_score",
                "filtered",
                "evaluation_method",
                "created_at",
                "is_augmented",
                "variant_of",
            ]
            variant_fields = [field for field in variant_fields if field in allowed_fields]
            variants = milvus_service.milvus_client.query(  # type: ignore[union-attr]
                expr=f'variant_of == "{_escape_expr_value(qa_id)}" and is_augmented == true',
                output_fields=variant_fields,
                limit=200,
            )
            for variant in variants or []:
                similar_questions.append(
                    {
                        "id": variant.get("id"),
                        "question": variant.get("question"),
                        "answer": variant.get("answer"),
                        "question_type": variant.get("question_type"),
                        "answer_explanation": variant.get("answer_explanation"),
                        "score": variant.get("average_score"),
                        "average_score": variant.get("average_score"),
                        "filtered": variant.get("filtered"),
                        "evaluation_method": variant.get("evaluation_method"),
                        "created_at": variant.get("created_at"),
                        "is_augmented": variant.get("is_augmented"),
                        "variant_of": variant.get("variant_of"),
                    }
                )
            similar_questions.sort(
                key=lambda item: (item.get("created_at") or 0, item.get("id") or ""),
                reverse=True,
            )
    except Exception:
        similar_questions = []

    debug_map = get_debug_map([qa_id])
    debug_payload = debug_map.get(qa_id) or {}

    return {
        "id": qa_id,
        "task_id": row.get("task_id"),
        "original_filename": row.get("original_filename"),
        "source": row.get(source_field),
        "source_fact_text": row.get("source_fact_text"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "question_type": row.get("question_type"),
        "question_type_reason": row.get("question_type_reason"),
        "answer_explanation": row.get("answer_explanation"),
        "knowledge_category": row.get("knowledge_category"),
        "knowledge_category_reason": row.get("knowledge_category_reason"),
        "knowledge_category_confidence": row.get("knowledge_category_confidence"),
        "difficulty_level": row.get("difficulty_level"),
        "difficulty_score": row.get("difficulty_score"),
        "filtered": row.get("filtered"),
        "average_score": row.get("average_score"),
        "faithfulness": row.get("faithfulness"),
        "evaluation_method": row.get("evaluation_method"),
        "evaluated": _is_evaluated(row),
        "evaluation": evaluation,
        "unsupervised_evaluation": unsupervised_evaluation,
        "llm_model": row.get("llm_model"),
        "embed_model": row.get("embed_model"),
        "embed_dim": row.get("embed_dim"),
        "created_at": row.get("created_at"),
        "filter_basis": row.get("filter_basis"),
        "is_primary": row.get("is_primary"),
        "is_augmented": row.get("is_augmented"),
        "variant_of": row.get("variant_of"),
        "similar_questions": similar_questions,
        "source_anchor_text": debug_payload.get("source_anchor_text"),
        "source_chunk_id": debug_payload.get("source_chunk_id"),
        "source_chunk_index": debug_payload.get("source_chunk_index"),
        "source_chunk_title_path": debug_payload.get("source_chunk_title_path"),
        "evidence_chunk_ids": debug_payload.get("evidence_chunk_ids") or [],
        "qa_generation_unit_id": debug_payload.get("qa_generation_unit_id"),
        "qa_generation_unit_text": debug_payload.get("qa_generation_unit_text"),
        "retrieval_query": debug_payload.get("retrieval_query"),
        "must_have_terms": debug_payload.get("must_have_terms") or [],
        "answer_scope_hint": debug_payload.get("answer_scope_hint"),
        "answer_scope": debug_payload.get("answer_scope"),
        "effective_answer_scope": debug_payload.get("effective_answer_scope")
        or debug_payload.get("answer_scope"),
        "answer_scope_decision": debug_payload.get("answer_scope_decision") or {},
        "evidence_usage": debug_payload.get("evidence_usage") or [],
        "retrieval_trace": debug_payload.get("retrieval_trace") or {},
        "admin": meta.to_dict(),
    }


__all__ = ["get_qa_item", "list_qa_items"]
