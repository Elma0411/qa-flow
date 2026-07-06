# 文件作用：合并问答、评估和文件信息为统一结果产物。
# 关联说明：负责单文件结果构造；多文件合并与 CSV 导出分别下沉到 merge/csv_export。

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from .score_summary import _compute_unsupervised_scores_from_items

from app.core.config import (
    CONFIG,
    LLM_EVALUATION_METRICS,
    LOCAL_EVALUATION_METRICS,
    LOCAL_EVALUATION_AVG_METRICS,
)


def _build_fact_maps(
    task_id: str,
    original_filename: str,
    categorized_facts: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """
    Build:
    - fact_category_map: atomic_fact -> metadata (id, category, source, language, ...).
    - facts_compact: list of fact records for the consolidated JSON.
    - category_counts: frequency of each knowledge_category.
    """
    fact_category_map: Dict[str, Dict[str, Any]] = {}
    facts_compact: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    for cf in categorized_facts:
        atomic_fact = (
            cf.get("atomic_fact") or cf.get("fact") or cf.get("source_fact")
        )
        if not atomic_fact:
            continue

        knowledge_category = cf.get("knowledge_category") or cf.get("theme")
        knowledge_category_reason = (
            cf.get("knowledge_category_reason") or cf.get("theme_reason")
        )
        knowledge_category_confidence = (
            cf.get("knowledge_category_confidence") or cf.get("theme_confidence")
        )
        source = cf.get("source") or cf.get("source_id")

        fact_id = hashlib.sha1(
            (task_id + original_filename + str(atomic_fact)).encode("utf-8")
        ).hexdigest()

        category_key = knowledge_category or "未分类"
        category_counts[category_key] = category_counts.get(category_key, 0) + 1

        meta = {
            "fact_id": fact_id,
            "knowledge_category": knowledge_category,
            "knowledge_category_reason": knowledge_category_reason,
            "knowledge_category_confidence": knowledge_category_confidence,
            "source": source,
            "language": cf.get("language"),
        }
        fact_category_map[str(atomic_fact)] = meta

        facts_compact.append(
            {
                **meta,
                "atomic_fact": atomic_fact,
            }
        )

    return fact_category_map, facts_compact, category_counts


def _build_evaluation_maps(
    evaluation_results: Optional[Dict[str, Any]],
    include_evaluation: bool,
    evaluation_method: str,
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
]:
    """
    Build lookup maps:
    - llm_map[(question, answer)] -> {"scores": {...}, "reasons": {...}}
    - local_map[(question, answer)] -> {"scores": {...}}
    """
    llm_metrics = LLM_EVALUATION_METRICS
    local_metrics = LOCAL_EVALUATION_METRICS
    llm_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    local_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if not (include_evaluation and evaluation_results):
        return llm_map, local_map

    if evaluation_method == "llm" and evaluation_results.get("results"):
        for r in evaluation_results["results"]:
            key = (r.get("question", ""), r.get("answer", ""))
            scores: Dict[str, Any] = {}
            reasons: Dict[str, Any] = {}
            ev = r.get("evaluation", {}) or {}
            for metric in llm_metrics:
                if metric in ev and isinstance(ev[metric], dict):
                    scores[metric] = ev[metric].get("score")
                    reason_val = (
                        ev[metric].get("reasons")
                        or ev[metric].get("reason")
                        or ev[metric].get("explanation")
                        or "未提供原因"
                    )
                    reasons[metric] = reason_val
            llm_map[key] = {"scores": scores, "reasons": reasons}
    elif evaluation_method == "local" and evaluation_results.get("results"):
        for r in evaluation_results["results"]:
            key = (r.get("question", ""), r.get("answer", ""))
            ev = r.get("evaluation", {}) or {}
            scores: Dict[str, Any] = {}
            for metric in local_metrics:
                if metric in ev and isinstance(ev[metric], dict):
                    scores[metric] = ev[metric].get("score")
            local_map[key] = {"scores": scores}

    return llm_map, local_map


def build_consolidated_entry(
    task_id: str,
    original_filename: str,
    facts: List[Dict[str, Any]],
    categorized_facts: List[Dict[str, Any]],
    qa_data: List[Dict[str, Any]],
    evaluation_results: Optional[Dict[str, Any]],
    filtered_qa_data: Optional[List[Dict[str, Any]]],
    include_evaluation: bool,
    evaluation_method: str,
    filter_by_threshold: bool,
    score_threshold: float,
    chunk_size: int,
    qa_per_chunk: int,
    qa_detail_mode: str,
    prompt_language: str,
    llm_model: str,
    include_unsupervised_evaluation: bool = False,
    ocr_seconds: Optional[float] = None,
    generation_seconds: Optional[float] = None,
    unsupervised_seconds: Optional[float] = None,
    evaluation_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a consolidated JSON payload for a single input file.

    Key goals:
    - 保留 fact 级别的 source。
    - 为每条 QA item 写入 source/source_fact_*。
    - 在 item 级别增加 filter_basis 字段。
    """
    del facts  # currently unused, but kept for signature compatibility

    created_ts = int(time.time())

    # ---------- Fact-level aggregation ----------
    fact_category_map, facts_compact, category_counts = _build_fact_maps(
        task_id, original_filename, categorized_facts
    )
    sorted_categories = sorted(
        category_counts.items(), key=lambda x: x[1], reverse=True
    )

    # ---------- Evaluation lookup maps ----------
    llm_map, local_map = _build_evaluation_maps(
        evaluation_results=evaluation_results,
        include_evaluation=include_evaluation,
        evaluation_method=evaluation_method,
    )
    llm_metrics = LLM_EVALUATION_METRICS
    local_metrics = LOCAL_EVALUATION_METRICS

    # Original QA lookup (full fields) for merging with filtered_qa_data
    qa_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for qa in qa_data:
        key = (qa.get("question", ""), qa.get("answer", ""))
        qa_lookup[key] = qa

    # 主问答 id 映射（用于变体关联）
    primary_id_map: Dict[str, str] = {}
    for qa in qa_data:
        if qa.get("is_augmented"):
            continue
        key = f"{qa.get('question','')}|||{qa.get('answer','')}"
        item_id = hashlib.sha1(
            (task_id + original_filename + (qa.get("question", "") or "") + (qa.get("answer", "") or "")).encode("utf-8")
        ).hexdigest()
        primary_id_map[key] = item_id

    # 仅当过滤结果已实际生成（filtered_qa_data != None）时才启用过滤；
    # 同时保持：filtered_qa_data=[] 表示“全部被阈值淘汰”。
    filter_applied = bool(filter_by_threshold and filtered_qa_data is not None)
    if filter_applied:
        qa_source = filtered_qa_data or []
        use_filtered = True
    else:
        qa_source = qa_data
        use_filtered = False

    # filter_basis at run level
    filter_basis = (
        evaluation_method
        if (filter_applied and evaluation_method in ("llm", "local", "faithfulness", "answerability", "unsupervised_f1"))
        else None
    )

    created_items: List[Dict[str, Any]] = []

    for qa in qa_source:
        q = qa.get("question", "")
        a = qa.get("answer", "")
        key = (q, a)

        base = qa_lookup.get(key, {})
        merged: Dict[str, Any] = {}
        merged.update(base)
        merged.update(qa)

        # Category information
        kc = (
            merged.get("knowledge_category")
            or (base.get("knowledge_category") if base else None)
            or merged.get("theme")
            or "未分类"
        )
        kc_reason = (
            merged.get("knowledge_category_reason")
            or (base.get("knowledge_category_reason") if base else None)
            or merged.get("theme_reason")
        )

        # Map back to atomic fact text first (one-step generation should always provide source_fact_text)
        source_fact_text = merged.get("source_fact_text") or merged.get("source_fact") or ""
        fact_info = fact_category_map.get(str(source_fact_text), {})

        kc_conf = (
            merged.get("knowledge_category_confidence")
            or merged.get("theme_confidence")
            or fact_info.get("knowledge_category_confidence")
        )

        # Evaluation data
        llm_scores = (llm_map.get(key) or {}).get("scores")
        llm_reasons = (llm_map.get(key) or {}).get("reasons")
        local_scores = (local_map.get(key) or {}).get("scores")

        avg_score = merged.get("average_score")
        if avg_score is None:
            if evaluation_method == "llm" and llm_scores:
                vals = [
                    v
                    for m in llm_metrics
                    if isinstance(llm_scores.get(m), (int, float))
                    for v in [llm_scores.get(m)]
                ]
                avg_score = (sum(vals) / len(vals)) if vals else None
            elif evaluation_method == "local" and local_scores:
                vals = [
                    v
                    for m in LOCAL_EVALUATION_AVG_METRICS
                    if isinstance(local_scores.get(m), (int, float))
                    for v in [local_scores.get(m)]
                ]
                avg_score = (sum(vals) / len(vals)) if vals else None
            elif evaluation_method in ("faithfulness", "answerability", "unsupervised_f1"):
                ue = merged.get("unsupervised_evaluation") or {}
                ue_scores = ue.get("scores") if isinstance(ue, dict) else {}
                key_map = {
                    "faithfulness": "faithfulness",
                    "answerability": "answerability",
                    "unsupervised_f1": "unsupervised_f1",
                }
                score_key = key_map.get(evaluation_method, "faithfulness")
                raw = ue_scores.get(score_key) if isinstance(ue_scores, dict) else None
                try:
                    avg_score = float(raw) if raw is not None else None
                except Exception:
                    avg_score = None

        # Derive answer_explanation (优先用 merged 自带，其次用 LLM reasons 里比较有用的一个)
        answer_explanation = merged.get("answer_explanation")
        if not answer_explanation and llm_reasons:
            for metric in ("accuracy", "completeness", "relevance"):
                if llm_reasons.get(metric):
                    answer_explanation = llm_reasons[metric]
                    break

        item_id = hashlib.sha1(
            (task_id + original_filename + q + a).encode("utf-8")
        ).hexdigest()

        source = merged.get("source") or fact_info.get("source")

        is_primary = bool(merged.get("is_primary", not merged.get("is_augmented")))
        is_augmented = bool(merged.get("is_augmented", False))
        variant_of = merged.get("variant_of")
        if not variant_of:
            parent_key = merged.get("variant_of_key")
            if parent_key and parent_key in primary_id_map:
                variant_of = primary_id_map.get(parent_key)

        # 单选题的选项与正确选项（如果有）
        raw_options = merged.get("options") or merged.get("choices")
        options: Optional[List[str]]
        if isinstance(raw_options, list):
            options = [str(o) for o in raw_options]
        else:
            options = None

        correct_option = merged.get("correct_option")
        if isinstance(correct_option, str) and correct_option.strip():
            correct_option = correct_option.strip()
        else:
            correct_option = None

        filtered_flag = use_filtered and not merged.get("is_augmented", False)

        created_item: Dict[str, Any] = {
            "id": item_id,
            "task_id": task_id,
            "original_filename": original_filename,
            "knowledge_category": kc,
            "knowledge_category_confidence": kc_conf,
            "knowledge_category_reason": fact_info.get("knowledge_category_reason") or kc_reason,
            "question_type": merged.get("question_type", "简答题"),
            "question_type_reason": merged.get("question_type_reason"),
            "options": options,
            "correct_option": correct_option,
            "answer_explanation": answer_explanation,
            "difficulty_level": merged.get("difficulty_level") or "中等",
            "difficulty_score": merged.get("difficulty_score"),
            "llm_model": llm_model,
            "embed_model": CONFIG["milvus"]["embedding_model"],
            "embed_dim": CONFIG["milvus"]["vector_dim"],
            "question": q,
            "answer": a,
            "source": source,
            "source_fact_id": fact_info.get("fact_id"),
            "source_fact_text": source_fact_text,
            "source_anchor_text": merged.get("source_anchor_text"),
            "source_chunk_id": merged.get("source_chunk_id"),
            "source_chunk_index": merged.get("source_chunk_index"),
            "source_chunk_title_path": merged.get("source_chunk_title_path"),
            "evidence_chunk_ids": merged.get("evidence_chunk_ids"),
            "qa_generation_unit_id": merged.get("qa_generation_unit_id"),
            "qa_generation_unit_text": merged.get("qa_generation_unit_text"),
            "retrieval_query": merged.get("retrieval_query"),
            "must_have_terms": merged.get("must_have_terms") or [],
            "answer_scope_hint": merged.get("answer_scope_hint"),
            "answer_scope": merged.get("answer_scope"),
            "effective_answer_scope": merged.get("effective_answer_scope")
            or merged.get("answer_scope"),
            "answer_scope_decision": merged.get("answer_scope_decision") or {},
            "evidence_usage": merged.get("evidence_usage") or [],
            "retrieval_trace": merged.get("retrieval_trace"),
            "filtered": filtered_flag,
            "average_score": avg_score,
            "evaluation_method": evaluation_method if include_evaluation else None,
            "evaluation": {
                "llm": {"scores": llm_scores, "reasons": llm_reasons} if llm_scores else None,
                "local": {"scores": local_scores} if local_scores else None,
            },
            "unsupervised_evaluation": merged.get("unsupervised_evaluation"),
            "text_for_embedding": merged.get("text_for_embedding") or f"{q} [SEP] {a}",
            "created_at": created_ts,
            "filter_basis": filter_basis,
            "is_primary": is_primary,
            "is_augmented": is_augmented,
            "variant_of": variant_of,
            # 保留增广后的相似问向，供导出 CSV/JSON 时使用
            "similar_questions": merged.get("similar_questions"),
        }

        created_items.append(created_item)

    filter_info = (
        (evaluation_results or {}).get("filter_info")
        if (filter_applied and evaluation_results)
        else None
    )

    qa_generated = len(qa_data)
    qa_evaluated = qa_generated if include_evaluation else 0
    generation_avg = (
        (generation_seconds / qa_generated)
        if generation_seconds and qa_generated
        else None
    )
    evaluation_avg = (
        (evaluation_seconds / qa_evaluated)
        if evaluation_seconds and qa_evaluated
        else None
    )

    unsupervised_qa_scored = sum(
        1
        for it in created_items
        if isinstance(it.get("unsupervised_evaluation"), dict)
        and isinstance((it.get("unsupervised_evaluation") or {}).get("scores"), dict)
    )
    unsupervised_avg = (
        (unsupervised_seconds / unsupervised_qa_scored)
        if unsupervised_seconds and unsupervised_qa_scored
        else None
    )

    params: Dict[str, Any] = {
        "chunk_size": chunk_size,
        "qa_per_chunk": qa_per_chunk,
        "qa_detail_mode": qa_detail_mode,
        "prompt_language": prompt_language,
        "include_evaluation": include_evaluation,
        "include_unsupervised_evaluation": include_unsupervised_evaluation,
        "evaluation_method": evaluation_method,
        "filter_by_threshold": filter_by_threshold,
        "score_threshold": score_threshold if filter_by_threshold else None,
    }
    counts: Dict[str, Any] = {
        "facts": len(facts_compact),
        "qa_pairs": len(qa_data),
        "filtered_qa_pairs": len(qa_source) if use_filtered else 0,
    }

    consolidated_payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "task": {
            "task_id": task_id,
            "original_filename": original_filename,
            "created_at": created_ts,
            "params": params,
        },
        "model": {
            "llm_model": llm_model,
            "embed_model": CONFIG["milvus"]["embedding_model"],
            "embed_dim": CONFIG["milvus"]["vector_dim"],
            "distance": CONFIG["milvus"].get("metric_type", "IP"),
        },
        "counts": counts,
        "category_distribution": dict(sorted_categories),
        "theme_distribution": dict(
            sorted_categories
        ),  # backward compatibility for legacy consumers
        "filter_info": filter_info,
        "filter_basis": filter_basis,
        "facts": facts_compact,
        "timing": {
            "ocr_seconds": ocr_seconds,
            "generation_seconds": generation_seconds,
            "generation_avg_seconds_per_qa": generation_avg,
            "qa_generated": qa_generated,
            "unsupervised_seconds": unsupervised_seconds,
            "unsupervised_avg_seconds_per_qa": unsupervised_avg,
            "unsupervised_qa_scored": unsupervised_qa_scored,
            "evaluation_seconds": evaluation_seconds,
            "evaluation_avg_seconds_per_qa": evaluation_avg,
            "qa_evaluated": qa_evaluated,
        },
        "items": created_items,
    }
    if include_unsupervised_evaluation:
        consolidated_payload["unsupervised_scores"] = _compute_unsupervised_scores_from_items(created_items)
    return {"filename": original_filename, "payload": consolidated_payload}





__all__ = [
    'build_consolidated_entry',
]
