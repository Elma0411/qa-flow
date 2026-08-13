# 文件作用：调用大模型生成候选问题和最终问答条目。
# 关联说明：依赖 prompts、evidence_units 和 validation，是生成阶段的 LLM 调用层。

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
from qa.prompts.qa_generation_prompts import (
    build_candidate_question_system_prompt,
    build_evidence_answer_system_prompt,
    build_question_editor_system_prompt,
    build_scenario_planner_system_prompt,
)
from qa.prompts.category_templates import resolve_category_prompt_template_key
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
    expected_question_type: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    question = str(item.get("question") or item.get("q") or "").strip()
    if not question:
        return None, "missing_question"

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
    return (
        {
            "question": question,
            "question_type": question_type,
            "question_type_reason": str(item.get("question_type_reason") or "").strip(),
            "difficulty_level": difficulty_level,
            "difficulty_score": difficulty_score,
        },
        "ok",
    )


def _restore_evidence_usage_ids(
    raw_entries: Any,
    ref_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert model-facing evidence references into persisted chunk IDs."""
    if not isinstance(raw_entries, list):
        return []
    restored: List[Dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        ref = str(raw_entry.get("evidence_ref") or "").strip()
        mapped = ref_map.get(ref) if ref else None
        if not isinstance(mapped, dict) or not str(mapped.get("chunk_id") or "").strip():
            continue
        entry = {
            key: value
            for key, value in raw_entry.items()
            if key not in {"evidence_ref", "chunk_id"}
        }
        entry["chunk_id"] = mapped["chunk_id"]
        entry["role"] = str(mapped.get("role") or "evidence")
        entry["evidence_ref"] = ref
        restored.append(entry)
    return restored


def _primary_usage_covers_bound_materials(
    primary_ids: List[str],
    *,
    source_unit: Dict[str, Any],
    source_chunks: List[Dict[str, Any]],
) -> bool:
    material_ids = [str(value) for value in source_unit.get("material_ids") or [] if str(value)]
    if len(material_ids) <= 1:
        return True
    raw_mapping = source_unit.get("material_source_chunk_indexes")
    if not isinstance(raw_mapping, dict):
        return True
    chunk_id_by_index = {
        int(chunk.get("chunk_index") or 0): str(chunk.get("chunk_id") or "").strip()
        for chunk in source_chunks
        if isinstance(chunk, dict)
    }
    cited = set(primary_ids)
    for material_id in material_ids:
        indexes = raw_mapping.get(material_id)
        if not isinstance(indexes, list):
            return False
        material_chunk_ids = {
            chunk_id_by_index.get(int(index or 0), "")
            for index in indexes
        }
        material_chunk_ids.discard("")
        if not material_chunk_ids or cited.isdisjoint(material_chunk_ids):
            return False
    return True


def _resolve_generation_language(prompt_language: str, text: str) -> Tuple[str, str]:
    lang = (prompt_language or "auto").strip().lower()
    if lang == "auto":
        detected = detect_language(text)
        lang = detected if detected in {"zh", "en"} else "zh"
    if lang not in {"zh", "en"}:
        lang = "zh"
    return lang, build_language_instruction(lang)


def call_scenario_planner_llm(
    *,
    client: Any,
    model: str,
    section_materials: List[Any],
    requested_count: int,
    qa_detail_mode: str,
    prompt_language: str,
    request_timeout: int,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Plan typed scenarios whose material IDs can be validated deterministically."""
    if requested_count <= 0 or not section_materials:
        return []
    readable_materials = [material.to_prompt_dict() for material in section_materials]
    joined_text = "\n\n".join(str(item.get("content") or "") for item in readable_materials)
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        joined_text,
    )
    system_prompt = build_scenario_planner_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        requested_count=requested_count,
        qa_detail_mode=qa_detail_mode,
    )
    user_content = json.dumps({"materials": readable_materials}, ensure_ascii=False)
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
    except Exception as exc:
        if debug_writer:
            debug_writer(
                {
                    "event": "scenario_planner_llm_call",
                    "model": model,
                    "qa_detail_mode": qa_detail_mode,
                    "requested_count": requested_count,
                    "material_count": len(readable_materials),
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "parse_error": str(exc),
                }
            )
        raise
    raw_items = _parse_json_items(raw)
    allowed_ids = {str(material.material_id) for material in section_materials}
    expected_scenario_type = str(qa_detail_mode or "").strip().lower()
    normalized: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    for item in raw_items:
        scenario_type = str(item.get("scenario_type") or item.get("type") or "").strip().lower()
        if scenario_type not in {"point", "summary"}:
            dropped["invalid_scenario_type"] = dropped.get("invalid_scenario_type", 0) + 1
            continue
        if expected_scenario_type in {"point", "summary"} and scenario_type != expected_scenario_type:
            dropped["scenario_type_mismatch"] = dropped.get("scenario_type_mismatch", 0) + 1
            continue
        intent = str(item.get("intent") or "").strip()
        reader_need = str(item.get("reader_need") or "").strip()
        material_ids: List[str] = []
        for value in item.get("material_ids") or []:
            material_id = str(value or "").strip()
            if material_id in allowed_ids and material_id not in material_ids:
                material_ids.append(material_id)
        if not intent or not reader_need or not material_ids:
            dropped["missing_required_field"] = dropped.get("missing_required_field", 0) + 1
            continue
        if scenario_type == "point" and len(material_ids) != 1:
            dropped["point_requires_one_material"] = dropped.get("point_requires_one_material", 0) + 1
            continue
        normalized.append(
            {
                "scenario_type": scenario_type,
                "intent": intent,
                "reader_need": reader_need,
                "material_ids": material_ids,
            }
        )
    if debug_writer:
        debug_writer(
            {
                "event": "scenario_planner_llm_call",
                "model": model,
                "qa_detail_mode": qa_detail_mode,
                "requested_count": requested_count,
                "material_count": len(readable_materials),
                "system_prompt": system_prompt,
                "user_content": user_content,
                "raw_response": raw,
                "items_raw_count": len(raw_items),
                "items_validated_count": len(normalized),
                "dropped_validation_reasons": dropped,
            }
        )
    return normalized


def call_question_editor_llm(
    *,
    client: Any,
    model: str,
    candidate: Dict[str, Any],
    source_material: str,
    scenario_intent: str,
    reader_need: str,
    qa_detail_mode: str,
    prompt_language: str,
    request_timeout: int,
    chunk_index: Optional[int] = None,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Run one semantic editor decision; code validates only its output shape."""
    original_question = str(candidate.get("question") or "").strip()
    if not original_question:
        return None, "missing_question"
    language_code, language_instruction = _resolve_generation_language(
        prompt_language,
        source_material or original_question,
    )
    system_prompt = build_question_editor_system_prompt(
        language_code=language_code,
        language_instruction=language_instruction,
        qa_detail_mode=qa_detail_mode,
    )
    payload = {
        "scenario_type": qa_detail_mode,
        "scenario_intent": scenario_intent,
        "reader_need": reader_need,
        "question_type": candidate.get("question_type"),
        "original_question": original_question,
        "source_material": source_material,
    }
    user_content = json.dumps(payload, ensure_ascii=False)
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
    except Exception as exc:
        if debug_writer:
            debug_writer(
                {
                    "event": "question_editor_llm_call",
                    "chunk_index": chunk_index,
                    "model": model,
                    "qa_detail_mode": qa_detail_mode,
                    "scenario_intent": scenario_intent,
                    "reader_need": reader_need,
                    "original_question": original_question,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "parse_error": str(exc),
                }
            )
        raise
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    decision = str(parsed.get("decision") or "").strip().lower()
    edited_question = str(parsed.get("question") or "").strip()
    reason = str(parsed.get("reason") or "").strip()
    result: Optional[Dict[str, Any]] = None
    status = decision or "invalid_editor_response"
    if decision == "keep":
        result = dict(candidate)
        result["question"] = original_question
        status = "keep"
    elif decision == "rewrite" and edited_question:
        result = dict(candidate)
        result["question"] = edited_question
        status = "rewrite"
    elif decision == "drop":
        status = "drop"
    else:
        status = "invalid_editor_response"
    if debug_writer:
        debug_writer(
            {
                "event": "question_editor_llm_call",
                "chunk_index": chunk_index,
                "model": model,
                "qa_detail_mode": qa_detail_mode,
                "scenario_intent": scenario_intent,
                "reader_need": reader_need,
                "original_question": original_question,
                "editor_decision": decision,
                "edited_question": edited_question,
                "editor_reason": reason,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "raw_response": raw,
                "result_status": status,
            }
        )
    return result, status


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
    qa_detail_mode: str = "point",
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
        qa_detail_mode=qa_detail_mode,
        question_type_plan=question_type_plan,
        few_shot_examples=few_shot_examples,
        knowledge_category=knowledge_category,
    )
    prompt_template_key = resolve_category_prompt_template_key(knowledge_category)
    # Keep the candidate prompt readable. Chunk IDs, paths, and category
    # metadata stay in the trace; they are not useful for writing a natural
    # reader question and tend to leak into the wording.
    scenario_intent = str(source_chunk_meta.get("qa_generation_unit_scenario_intent") or "").strip()
    reader_need = str(source_chunk_meta.get("qa_generation_unit_reader_need") or "").strip()
    user_content = (
        f"scenario_intent: {scenario_intent}\n"
        f"reader_need: {reader_need}\n\n"
        "主来源材料：\n"
        + str(source_chunk_text or "").strip()
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
                    "qa_detail_mode": qa_detail_mode,
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
            expected_question_type=expected,
        )
        if not candidate:
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            continue
        key = " ".join(str(candidate.get("question") or "").split()).casefold()
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
                "qa_detail_mode": qa_detail_mode,
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
    source_unit_text = str(generation_unit.get("source_unit_text") or "").strip()
    unit_text = str(generation_unit.get("qa_generation_unit_text") or "").strip()
    evidence_ref_map = generation_unit.get("llm_evidence_ref_map")
    if not isinstance(evidence_ref_map, dict):
        evidence_ref_map = {}
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
    question_type = str(candidate.get("question_type") or "简答题").strip() or "简答题"
    user_content = (
        f"candidate_question: {candidate_question}\n"
        f"question_type: {question_type}\n"
        "\n"
        "可读证据材料（仅使用这些正文）：\n"
        f"{unit_text}\n\n"
        "evidence_ref 可选值：\n"
        f"{json.dumps(list(evidence_ref_map.keys()), ensure_ascii=False)}"
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
                normalized_item["evidence_usage"] = _restore_evidence_usage_ids(
                    evidence_usage,
                    evidence_ref_map,
                )[:12]
            restored_usage = normalized_item.get("evidence_usage") or []
            if not any(
                isinstance(entry, dict) and entry.get("role") == "primary_source"
                for entry in restored_usage
            ):
                normalized_item = None
                dropped_reason = "missing_primary_evidence_usage"
    if normalized_item:
        source_override_handler(
            normalized_item,
            chunk_text=source_unit_text or source_chunk_text,
            language_code=language_code,
        )
        normalized_item["question"] = candidate_question
        evidence_usage = normalized_item.get("evidence_usage")
        primary_ids: List[str] = []
        if isinstance(evidence_usage, list):
            for entry in evidence_usage:
                if not isinstance(entry, dict) or entry.get("role") != "primary_source":
                    continue
                chunk_id = str(entry.get("chunk_id") or "").strip()
                if chunk_id and chunk_id not in primary_ids:
                    primary_ids.append(chunk_id)
        source_chunks = generation_unit.get("source_chunks") or []
        chunks_by_id = {
            str(chunk.get("chunk_id") or "").strip(): chunk
            for chunk in source_chunks
            if isinstance(chunk, dict) and str(chunk.get("chunk_id") or "").strip()
        }
        primary_chunks = [chunks_by_id[chunk_id] for chunk_id in primary_ids if chunk_id in chunks_by_id]
        source_unit_payload = generation_unit.get("source_unit")
        if not isinstance(source_unit_payload, dict):
            source_unit_payload = {}
        if qa_detail_mode == "summary" and not _primary_usage_covers_bound_materials(
            primary_ids,
            source_unit=source_unit_payload,
            source_chunks=[chunk for chunk in source_chunks if isinstance(chunk, dict)],
        ):
            normalized_item = None
            dropped_reason = "incomplete_summary_primary_coverage"
        if normalized_item and not primary_chunks:
            primary_chunks = [source_chunk] if isinstance(source_chunk, dict) else []
            primary_ids = [
                str(source_chunk.get("chunk_id") or "").strip()
            ] if source_chunk else []
        primary_source = primary_chunks[0] if primary_chunks else source_chunk
        if normalized_item:
            normalized_item["source_chunk_id"] = primary_source.get("chunk_id")
            normalized_item["source_chunk_index"] = primary_source.get("chunk_index")
            normalized_item["source_chunk_title_path"] = primary_source.get("title_path")
            normalized_item["source_chunk_ids"] = primary_ids
            normalized_item["source_chunk_indexes"] = [
                chunk.get("chunk_index") for chunk in primary_chunks if chunk.get("chunk_index") is not None
            ]
            normalized_item["source_chunk_title_paths"] = [
                chunk.get("title_path") for chunk in primary_chunks if str(chunk.get("title_path") or "").strip()
            ]
            normalized_item["evidence_chunk_ids"] = generation_unit.get("evidence_chunk_ids") or []
            normalized_item["qa_generation_unit_id"] = generation_unit.get("qa_generation_unit_id")
            normalized_item["qa_generation_unit_text"] = unit_text
            normalized_item["retrieval_trace"] = generation_unit.get("retrieval_trace") or {}
            normalized_item["source"] = primary_source.get("chunk_id") or normalized_item.get("source")
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
    "call_question_editor_llm",
    "call_scenario_planner_llm",
    "normalize_question_type_mode",
    "normalize_question_type_weights",
    "normalize_question_types",
]
