# 文件作用：调用大模型生成候选问题和最终问答条目。
# 关联说明：依赖 prompts、evidence_units、text_quality_filters，是生成阶段的 LLM 调用层。

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.common import (
    build_language_instruction,
    detect_language,
    safe_response_dump,
)
from qa.grounding import normalize_grounding_text, validate_source_fact_grounding
from qa.prompts.qa_generation_prompts import (
    build_candidate_question_system_prompt,
    build_evidence_answer_system_prompt,
)
from qa.prompts.category_templates import resolve_category_prompt_template_key
from qa.generation import contains_ambiguous_reference
from qa.validation import normalize_difficulty_level, normalize_question_type

ALLOWED_QUESTION_TYPES = {"简答题", "单选题", "判断题", "计算题"}


def normalize_question_types(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        cleaned = [str(x).strip() for x in raw if str(x).strip()]
    else:
        cleaned = [s.strip() for s in str(raw).split(",") if s.strip()]
    return cleaned or None


def normalize_question_type_mode(raw: Any) -> str:
    mode = (str(raw or "mixed")).strip().lower()
    return mode if mode in {"fixed", "mixed"} else "mixed"


def normalize_question_type_weights(raw: Any) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        weights = raw
    elif isinstance(raw, str):
        try:
            weights = json.loads(raw)
        except Exception:
            return None
    else:
        return None
    if not isinstance(weights, dict):
        return None
    normalized: Dict[str, float] = {}
    for key, val in weights.items():
        k = str(key).strip()
        try:
            f = float(val)
        except Exception:
            continue
        if not k:
            continue
        normalized[k] = f
    return normalized or None


def build_question_type_plan(
    *,
    question_type_mode: str,
    question_types: Optional[List[str]],
    question_type_weights: Optional[Dict[str, float]],
    qa_per_chunk: int,
    seed_text: str,
) -> List[str]:
    """
    Build a deterministic per-item question_type plan to make the model follow
    the requested types.

    - If question_types is empty -> all "简答题"
    - fixed -> all first type
    - mixed -> weighted allocation (if provided) else round-robin cycling
    """
    desired = max(1, int(qa_per_chunk or 1))
    types = [t for t in (question_types or []) if t in ALLOWED_QUESTION_TYPES]
    if not types:
        return ["简答题"] * desired

    mode = (str(question_type_mode or "mixed")).strip().lower()
    if mode not in {"fixed", "mixed"}:
        mode = "mixed"

    if mode == "fixed":
        return [types[0]] * desired

    weights = question_type_weights or None
    if isinstance(weights, dict) and weights:
        filtered: List[Tuple[str, float]] = []
        for question_type in types:
            try:
                weight = float(weights.get(question_type, 0.0))
            except Exception:
                weight = 0.0
            if weight > 0:
                filtered.append((question_type, weight))
        if filtered:
            total_weight = sum(weight for _, weight in filtered)
            if total_weight > 0:
                exact = [
                    (question_type, (weight / total_weight) * desired)
                    for question_type, weight in filtered
                ]
                counts: Dict[str, int] = {
                    question_type: int(math.floor(value))
                    for question_type, value in exact
                }
                remaining = desired - sum(counts.values())
                remainders = sorted(
                    [
                        (question_type, value - counts[question_type])
                        for question_type, value in exact
                    ],
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                for index in range(max(0, remaining)):
                    question_type = remainders[index % len(remainders)][0]
                    counts[question_type] = counts.get(question_type, 0) + 1

                plan: List[str] = []
                for question_type, _ in filtered:
                    plan.extend([question_type] * counts.get(question_type, 0))
                if len(plan) < desired:
                    plan.extend([filtered[0][0]] * (desired - len(plan)))
                plan = plan[:desired]

                seed = hashlib.sha1((seed_text or "").encode("utf-8")).hexdigest()
                rnd = random.Random(seed)
                rnd.shuffle(plan)
                return plan

    plan: List[str] = []
    while len(plan) < desired:
        plan.extend(types)
    return plan[:desired]


def apply_question_type_plan(
    items: List[Dict[str, Any]],
    plan: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """
    Reorder/select items to follow the planned question types.
    - Greedy matching: for each planned type, pick the first remaining item with that type.
    - If a planned type cannot be satisfied, skip it.
    """
    if not plan:
        return items
    remaining = [it for it in items if isinstance(it, dict)]
    planned: List[Dict[str, Any]] = []
    for question_type in plan:
        matched_index = None
        for index, item in enumerate(remaining):
            if item.get("question_type") == question_type:
                matched_index = index
                break
        if matched_index is None:
            continue
        planned.append(remaining.pop(matched_index))
    return planned


def _parse_json_items(raw: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads((raw or "").strip()) if raw else None
    except Exception:
        return []
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        item = parsed.get("item")
        if isinstance(item, dict):
            return [item]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _normalize_candidate_question(
    item: Dict[str, Any],
    *,
    language_code: str,
    expected_question_type: Optional[str],
    source_chunk_text: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    question = str(item.get("question") or item.get("q") or "").strip()
    source_anchor_text = str(
        item.get("source_anchor_text")
        or item.get("anchor_text")
        or item.get("source_fact_text")
        or ""
    ).strip()
    if not question:
        return None, "missing_question"
    if not source_anchor_text:
        return None, "missing_source_anchor_text"
    if contains_ambiguous_reference(question, language_code=language_code):
        return None, "ambiguous_reference_question"

    grounded, grounding_reason = validate_source_fact_grounding(
        source_anchor_text,
        chunk_text=source_chunk_text,
        qa_detail_mode="summary",
        language_code=language_code,
    )
    if not grounded:
        return None, grounding_reason

    question_type = normalize_question_type(
        item.get("question_type") or item.get("type"),
        expected=expected_question_type,
    )
    difficulty_level = normalize_difficulty_level(item.get("difficulty_level"))
    try:
        difficulty_score = (
            float(item.get("difficulty_score"))
            if item.get("difficulty_score") is not None
            else None
        )
    except Exception:
        difficulty_score = None
    if difficulty_score is not None:
        difficulty_score = max(0.0, min(1.0, difficulty_score))
    retrieval_query = str(item.get("retrieval_query") or "").strip()
    raw_terms = item.get("must_have_terms")
    if isinstance(raw_terms, list):
        must_have_terms = [str(term).strip() for term in raw_terms if str(term).strip()]
    else:
        must_have_terms = [
            term.strip()
            for term in str(raw_terms or "").replace("，", ",").split(",")
            if term.strip()
        ]
    answer_scope_hint = str(
        item.get("answer_scope_hint") or item.get("answer_scope") or "source_primary"
    ).strip().lower()
    if answer_scope_hint not in {"source_primary", "same_section", "cross_chunk"}:
        answer_scope_hint = "source_primary"

    return (
        {
            "question": question,
            "source_anchor_text": source_anchor_text,
            "retrieval_query": retrieval_query,
            "must_have_terms": must_have_terms[:8],
            "answer_scope_hint": answer_scope_hint,
            # Backward-compatible alias. Downstream code replaces answer_scope
            # with the system-approved effective scope before answer generation.
            "answer_scope": answer_scope_hint,
            "question_type": question_type,
            "question_type_reason": str(item.get("question_type_reason") or "").strip(),
            "difficulty_level": difficulty_level,
            "difficulty_score": difficulty_score,
        },
        "ok",
    )


def _is_source_anchored(
    *,
    source_fact_text: str,
    source_anchor_text: str,
    source_chunk_text: str,
) -> bool:
    fact = normalize_grounding_text(source_fact_text)
    anchor = normalize_grounding_text(source_anchor_text)
    source = normalize_grounding_text(source_chunk_text)
    if not fact or not source:
        return False
    if fact in source:
        return True
    if anchor and (anchor in fact or fact in anchor):
        return True
    if anchor and len(anchor) >= 8:
        grams = [anchor[index : index + 3] for index in range(max(0, len(anchor) - 2))]
        if grams:
            matched = sum(1 for gram in grams if gram in fact)
            return (matched / len(grams)) >= 0.55
    return False


def _resolve_generation_language(prompt_language: str, text: str) -> Tuple[str, str]:
    lang = (prompt_language or "auto").strip().lower()
    if lang == "auto":
        detected = detect_language(text)
        lang = detected if detected in {"zh", "en"} else "zh"
    if lang not in {"zh", "en"}:
        lang = "zh"
    return lang, build_language_instruction(lang)


def call_candidate_question_llm(
    *,
    client: Any,
    model: str,
    source_chunk_text: str,
    source_chunk_meta: Dict[str, Any],
    candidate_count: int,
    prompt_language: str,
    question_type_plan: List[str],
    few_shot_examples: Optional[List[Dict[str, Any]]],
    request_timeout: int,
    knowledge_category: Optional[str] = None,
    chunk_index: Optional[int] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        source_chunk_text,
    )
    system_prompt = build_candidate_question_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        candidate_count=candidate_count,
        question_type_plan=question_type_plan,
        few_shot_examples=few_shot_examples,
        knowledge_category=knowledge_category,
    )
    prompt_template_key = resolve_category_prompt_template_key(knowledge_category)
    title_path = str(source_chunk_meta.get("title_path") or "").strip()
    user_content = (
        "主来源块信息：\n"
        f"chunk_id: {source_chunk_meta.get('chunk_id') or ''}\n"
        f"title_path: {title_path}\n\n"
        f"knowledge_category: {knowledge_category or ''}\n\n"
        "主来源块正文：\n"
        f"{source_chunk_text}"
    )

    response_type: Optional[str] = None
    response_dump: Any = None
    raw = ""
    parse_error: Optional[str] = None
    try:
        raw = client.create_chat_completion_text(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=float(request_timeout),
        ).strip()
        response_type = "str"
        response_dump = safe_response_dump(raw)
        raw_items = _parse_json_items(raw)
    except Exception as exc:
        parse_error = str(exc)
        raw_items = []
        if debug_writer:
            debug_writer(
                {
                    "event": "candidate_question_llm_call",
                    "chunk_index": chunk_index,
                    "model": model,
                    "knowledge_category": knowledge_category,
                    "prompt_template_key": prompt_template_key,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "response_type": response_type,
                    "response_dump": response_dump,
                    "raw_response": raw,
                    "parse_error": parse_error,
                }
            )
        raise

    normalized: List[Dict[str, Any]] = []
    dropped_reasons: Dict[str, int] = {}
    seen_questions: set[str] = set()
    for index, item in enumerate(raw_items):
        expected = question_type_plan[index] if index < len(question_type_plan) else None
        candidate, reason = _normalize_candidate_question(
            item,
            language_code=language_code,
            expected_question_type=expected,
            source_chunk_text=source_chunk_text,
        )
        if not candidate:
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            continue
        key = normalize_grounding_text(str(candidate.get("question") or ""))
        if key in seen_questions:
            dropped_reasons["duplicate_question"] = dropped_reasons.get("duplicate_question", 0) + 1
            continue
        seen_questions.add(key)
        normalized.append(candidate)

    if debug_writer:
        debug_writer(
            {
                "event": "candidate_question_llm_call",
                "chunk_index": chunk_index,
                "model": model,
                "candidate_count": candidate_count,
                "knowledge_category": knowledge_category,
                "prompt_template_key": prompt_template_key,
                "question_type_plan": question_type_plan,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response_type": response_type,
                "response_dump": response_dump,
                "raw_response": raw,
                "parse_error": parse_error,
                "items_raw_count": len(raw_items),
                "items_validated_count": len(normalized),
                "dropped_validation_reasons": dropped_reasons,
            }
        )
    return normalized[: max(1, int(candidate_count))]


def call_evidence_answer_llm(
    *,
    client: Any,
    model: str,
    candidate: Dict[str, Any],
    generation_unit: Dict[str, Any],
    qa_detail_mode: str,
    prompt_language: str,
    request_timeout: int,
    item_normalizer_with_reason: Callable[..., Tuple[Optional[Dict[str, Any]], str]],
    source_fact_detail_validator: Callable[..., Tuple[bool, str]],
    source_fact_grounding_validator: Callable[..., Tuple[bool, str]],
    source_override_handler: Callable[..., None],
    fixed_knowledge_category: Optional[str] = None,
    fixed_knowledge_category_confidence: Optional[float] = None,
    fixed_knowledge_category_reason: str = "",
    use_category_prompt_templates: bool = True,
    chunk_index: Optional[int] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    source_chunk = generation_unit.get("source_chunk") or {}
    source_chunk_text = str(source_chunk.get("text") or "").strip()
    unit_text = str(generation_unit.get("qa_generation_unit_text") or "").strip()
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        unit_text or source_chunk_text,
    )
    use_fixed_knowledge_category = bool(str(fixed_knowledge_category or "").strip())
    prompt_template_category = (
        fixed_knowledge_category if use_category_prompt_templates else None
    )
    system_prompt = build_evidence_answer_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        qa_detail_mode=qa_detail_mode,
        include_knowledge_category_fields=not use_fixed_knowledge_category,
        knowledge_category=prompt_template_category,
    )
    prompt_template_key = resolve_category_prompt_template_key(prompt_template_category)
    candidate_question = str(candidate.get("question") or "").strip()
    source_anchor_text = str(candidate.get("source_anchor_text") or "").strip()
    retrieval_query = str(candidate.get("retrieval_query") or "").strip()
    must_have_terms = candidate.get("must_have_terms") if isinstance(candidate.get("must_have_terms"), list) else []
    answer_scope = str(candidate.get("answer_scope") or "source_primary").strip().lower()
    if answer_scope not in {"source_primary", "same_section", "cross_chunk"}:
        answer_scope = "source_primary"
    answer_scope_hint = str(candidate.get("answer_scope_hint") or answer_scope).strip().lower()
    if answer_scope_hint not in {"source_primary", "same_section", "cross_chunk"}:
        answer_scope_hint = "source_primary"
    question_type = str(candidate.get("question_type") or "简答题").strip() or "简答题"
    user_content = (
        f"candidate_question: {candidate_question}\n"
        f"source_anchor_text: {source_anchor_text}\n"
        f"retrieval_query: {retrieval_query}\n"
        f"must_have_terms: {json.dumps(must_have_terms, ensure_ascii=False)}\n"
        f"answer_scope: {answer_scope}\n"
        f"question_type: {question_type}\n"
        f"question_type_reason: {candidate.get('question_type_reason') or ''}\n"
        f"difficulty_level: {candidate.get('difficulty_level') or '中等'}\n"
        f"difficulty_score: {candidate.get('difficulty_score') if candidate.get('difficulty_score') is not None else ''}\n\n"
        f"knowledge_category: {prompt_template_category or ''}\n\n"
        "qa_generation_unit_text:\n"
        f"{unit_text}"
    )

    response_type: Optional[str] = None
    response_dump: Any = None
    raw = ""
    parse_error: Optional[str] = None
    dropped_reason = ""
    try:
        raw = client.create_chat_completion_text(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=float(request_timeout),
        ).strip()
        response_type = "str"
        response_dump = safe_response_dump(raw)
        raw_items = _parse_json_items(raw)
    except Exception as exc:
        parse_error = str(exc)
        raw_items = []
        if debug_writer:
            debug_writer(
                {
                    "event": "evidence_answer_llm_call",
                    "chunk_index": chunk_index,
                    "candidate": candidate,
                    "generation_unit": {
                        "qa_generation_unit_id": generation_unit.get("qa_generation_unit_id"),
                        "evidence_chunk_ids": generation_unit.get("evidence_chunk_ids"),
                    },
                    "knowledge_category": fixed_knowledge_category,
                    "use_category_prompt_templates": use_category_prompt_templates,
                    "prompt_template_category": prompt_template_category,
                    "prompt_template_key": prompt_template_key,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "response_type": response_type,
                    "response_dump": response_dump,
                    "raw_response": raw,
                    "parse_error": parse_error,
                }
            )
        raise

    normalized_item: Optional[Dict[str, Any]] = None
    raw_item: Optional[Dict[str, Any]] = raw_items[0] if raw_items else None
    if not raw_items:
        dropped_reason = "missing_items"
    else:
        normalized_item, dropped_reason = item_normalizer_with_reason(
            raw_items[0],
            language_code=language_code,
            expected_question_type=question_type,
            fixed_knowledge_category=fixed_knowledge_category,
            fixed_knowledge_category_confidence=fixed_knowledge_category_confidence,
            fixed_knowledge_category_reason=fixed_knowledge_category_reason,
        )
        if normalized_item and raw_item:
            evidence_usage = raw_item.get("evidence_usage")
            if isinstance(evidence_usage, list):
                normalized_item["evidence_usage"] = [
                    entry for entry in evidence_usage if isinstance(entry, dict)
                ][:12]
    if normalized_item:
        if str(normalized_item.get("question") or "").strip() != candidate_question:
            dropped_reason = "question_mismatch"
            normalized_item = None
    if normalized_item:
        ok, detail_reason = source_fact_detail_validator(
            str(normalized_item.get("source_fact_text") or ""),
            qa_detail_mode=qa_detail_mode,
            language_code=language_code,
        )
        if not ok:
            dropped_reason = detail_reason
            normalized_item = None
    if normalized_item:
        grounded, grounding_reason = source_fact_grounding_validator(
            str(normalized_item.get("source_fact_text") or ""),
            chunk_text=unit_text,
            qa_detail_mode=qa_detail_mode,
            language_code=language_code,
        )
        if not grounded:
            dropped_reason = grounding_reason
            normalized_item = None
    if normalized_item and not _is_source_anchored(
        source_fact_text=str(normalized_item.get("source_fact_text") or ""),
        source_anchor_text=source_anchor_text,
        source_chunk_text=source_chunk_text,
    ):
        dropped_reason = "source_fact_not_anchored_to_source_chunk"
        normalized_item = None

    if normalized_item:
        source_override_handler(
            normalized_item,
            chunk_text=source_chunk_text,
            language_code=language_code,
        )
        normalized_item["question"] = candidate_question
        normalized_item["source_anchor_text"] = source_anchor_text
        normalized_item["retrieval_query"] = retrieval_query
        normalized_item["must_have_terms"] = must_have_terms
        normalized_item["answer_scope_hint"] = answer_scope_hint
        normalized_item["answer_scope"] = answer_scope
        normalized_item["effective_answer_scope"] = answer_scope
        normalized_item["answer_scope_decision"] = candidate.get("answer_scope_decision") or {}
        normalized_item["source_chunk_id"] = source_chunk.get("chunk_id")
        normalized_item["source_chunk_index"] = source_chunk.get("chunk_index")
        normalized_item["source_chunk_title_path"] = source_chunk.get("title_path")
        normalized_item["evidence_chunk_ids"] = generation_unit.get("evidence_chunk_ids") or []
        normalized_item["qa_generation_unit_id"] = generation_unit.get("qa_generation_unit_id")
        normalized_item["qa_generation_unit_text"] = unit_text
        normalized_item["retrieval_trace"] = generation_unit.get("retrieval_trace") or {}
        normalized_item["source"] = source_chunk.get("chunk_id") or normalized_item.get("source")
        normalized_item["text_for_embedding"] = (
            f"{candidate_question} [SEP] {normalized_item.get('answer') or ''}"
        )

    if debug_writer:
        debug_writer(
            {
                "event": "evidence_answer_llm_call",
                "chunk_index": chunk_index,
                "candidate": candidate,
                "generation_unit": {
                    "qa_generation_unit_id": generation_unit.get("qa_generation_unit_id"),
                    "evidence_chunk_ids": generation_unit.get("evidence_chunk_ids"),
                    "retrieval_trace": generation_unit.get("retrieval_trace"),
                },
                "knowledge_category": fixed_knowledge_category,
                "use_category_prompt_templates": use_category_prompt_templates,
                "prompt_template_category": prompt_template_category,
                "prompt_template_key": prompt_template_key,
                "model": model,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response_type": response_type,
                "response_dump": response_dump,
                "raw_response": raw,
                "parse_error": parse_error,
                "items_raw_count": len(raw_items),
                "items_validated_count": 1 if normalized_item else 0,
                "dropped_reason": "" if normalized_item else dropped_reason,
            }
        )
    return normalized_item, "ok" if normalized_item else dropped_reason



__all__ = [
    "apply_question_type_plan",
    "build_question_type_plan",
    "call_candidate_question_llm",
    "call_evidence_answer_llm",
    "normalize_question_type_mode",
    "normalize_question_type_weights",
    "normalize_question_types",
]
