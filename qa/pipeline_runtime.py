# 文件作用：解析完整 QA pipeline 的运行配置并执行 generation unit worker。
# 关联说明：被 text_to_qa_pipeline 调用，负责运行配置和 unit/chunk worker 兼容层。

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.generation import (
    DEFAULT_EVIDENCE_TOKEN_BUDGET,
    DEFAULT_FINAL_EVIDENCE_K,
    DEFAULT_SCENARIO_PLANNING_BATCH_CHARS,
    GenerationUnit,
    QADocumentEvidenceIndex,
    build_question_type_plan,
    call_candidate_question_llm,
    call_evidence_answer_llm,
    call_question_editor_llm,
    normalize_question_type_mode,
    normalize_question_type_weights,
    normalize_question_types,
)


@dataclass
class OneStepPipelineRuntime:
    chunk_size: int
    qa_per_chunk: int
    qa_total_limit: Optional[int]
    qa_total_limit_scope: str
    qa_detail_mode: str
    prompt_language: str
    text_model_concurrency: int
    question_type_mode: str
    question_types: Optional[List[str]]
    question_type_weights: Optional[Dict[str, float]]
    few_shot_examples: Optional[List[Dict[str, Any]]]
    debug_file: Optional[str]
    request_timeout: int
    model: str
    include_chunk_index: bool
    fixed_knowledge_category: Optional[str]
    fixed_knowledge_category_confidence: Optional[float]
    fixed_knowledge_category_reason: str
    use_category_prompt_templates: bool
    strict_max_attempts: int
    pre_split_chunks: Optional[List[str]]
    pre_split_chunk_meta: Optional[List[Dict[str, Any]]]
    final_evidence_k: int
    evidence_token_budget: int
    scenario_planning_batch_chars: int
    # Planner calls use the same text-model budget as generation units.


def parse_one_step_pipeline_runtime(config: Dict[str, Any]) -> OneStepPipelineRuntime:
    chunk_size = int(config.get("chunk_size") or 600)
    qa_per_chunk = int(config.get("qa_per_chunk") or 1)
    raw_qa_total_limit = config.get("qa_total_limit")
    qa_total_limit: Optional[int]
    if raw_qa_total_limit is None or str(raw_qa_total_limit).strip() == "":
        qa_total_limit = None
    else:
        try:
            qa_total_limit = max(0, int(raw_qa_total_limit))
        except Exception:
            qa_total_limit = None
    qa_total_limit_scope = str(config.get("qa_total_limit_scope") or "per_file").strip().lower()
    if qa_total_limit_scope not in {"per_file", "batch"}:
        qa_total_limit_scope = "per_file"
    qa_detail_mode = str(config.get("qa_detail_mode") or "point").strip().lower()
    if qa_detail_mode not in {"point", "summary", "auto"}:
        qa_detail_mode = "point"
    prompt_language = str(config.get("prompt_language") or "auto")
    try:
        text_model_concurrency = max(
            1,
            int(
                config.get("text_model_concurrency")
                or os.environ.get("TEXT_MODEL_CONCURRENCY")
                or 8
            ),
        )
    except (TypeError, ValueError):
        text_model_concurrency = 8
    question_type_mode = normalize_question_type_mode(config.get("question_type_mode"))
    question_types = normalize_question_types(config.get("question_types"))
    question_type_weights = normalize_question_type_weights(
        config.get("question_type_weights")
    )

    few_shot_examples = config.get("few_shot_examples")
    if not isinstance(few_shot_examples, list):
        few_shot_examples = None

    debug_file = str(config.get("debug_file") or "").strip() or None
    request_timeout = int(config.get("request_timeout") or 120)
    model = str(config.get("model") or "")
    include_chunk_index = bool(config.get("include_chunk_index", False))

    fixed_knowledge_category_raw = str(
        config.get("fixed_knowledge_category") or ""
    ).strip()
    fixed_knowledge_category = (
        fixed_knowledge_category_raw if fixed_knowledge_category_raw else None
    )
    fixed_knowledge_category_reason = str(
        config.get("fixed_knowledge_category_reason") or ""
    ).strip()

    fixed_confidence_raw = config.get("fixed_knowledge_category_confidence")
    try:
        fixed_knowledge_category_confidence = (
            float(fixed_confidence_raw)
            if fixed_confidence_raw is not None
            and str(fixed_confidence_raw).strip() != ""
            else None
        )
    except Exception:
        fixed_knowledge_category_confidence = None
    if fixed_knowledge_category_confidence is not None:
        fixed_knowledge_category_confidence = max(
            0.0,
            min(1.0, float(fixed_knowledge_category_confidence)),
        )
    use_category_prompt_templates_raw = config.get("use_category_prompt_templates", True)
    if isinstance(use_category_prompt_templates_raw, str):
        use_category_prompt_templates = use_category_prompt_templates_raw.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    else:
        use_category_prompt_templates = bool(use_category_prompt_templates_raw)

    strict_max_attempts = int(config.get("strict_max_attempts") or 2)

    raw_pre_split_chunks = config.get("pre_split_chunks")
    pre_split_chunks: Optional[List[str]] = None
    if isinstance(raw_pre_split_chunks, list) and all(
        isinstance(chunk, str) for chunk in raw_pre_split_chunks
    ):
        pre_split_chunks = [
            str(chunk).strip() for chunk in raw_pre_split_chunks if str(chunk).strip()
        ]

    raw_pre_split_chunk_meta = config.get("pre_split_chunk_meta")
    pre_split_chunk_meta: Optional[List[Dict[str, Any]]] = None
    if isinstance(raw_pre_split_chunk_meta, list):
        pre_split_chunk_meta = [
            dict(item) for item in raw_pre_split_chunk_meta if isinstance(item, dict)
        ]

    final_evidence_k_value = config.get("final_evidence_k")
    final_evidence_k = max(
        0,
        int(
            DEFAULT_FINAL_EVIDENCE_K
            if final_evidence_k_value is None
            else final_evidence_k_value
        ),
    )
    evidence_token_budget_value = config.get("evidence_token_budget")
    evidence_token_budget = max(
        256,
        int(
            DEFAULT_EVIDENCE_TOKEN_BUDGET
            if evidence_token_budget_value is None
            else evidence_token_budget_value
        ),
    )
    scenario_planning_batch_chars = max(
        1000,
        int(
            DEFAULT_SCENARIO_PLANNING_BATCH_CHARS
            if config.get("scenario_planning_batch_chars") is None
            else config.get("scenario_planning_batch_chars")
        ),
    )

    return OneStepPipelineRuntime(
        chunk_size=chunk_size,
        qa_per_chunk=qa_per_chunk,
        qa_total_limit=qa_total_limit,
        qa_total_limit_scope=qa_total_limit_scope,
        qa_detail_mode=qa_detail_mode,
        prompt_language=prompt_language,
        text_model_concurrency=text_model_concurrency,
        question_type_mode=question_type_mode,
        question_types=question_types,
        question_type_weights=question_type_weights,
        few_shot_examples=few_shot_examples,
        debug_file=debug_file,
        request_timeout=request_timeout,
        model=model,
        include_chunk_index=include_chunk_index,
        fixed_knowledge_category=fixed_knowledge_category,
        fixed_knowledge_category_confidence=fixed_knowledge_category_confidence,
        fixed_knowledge_category_reason=fixed_knowledge_category_reason,
        use_category_prompt_templates=use_category_prompt_templates,
        strict_max_attempts=strict_max_attempts,
        pre_split_chunks=pre_split_chunks,
        pre_split_chunk_meta=pre_split_chunk_meta,
        final_evidence_k=final_evidence_k,
        evidence_token_budget=evidence_token_budget,
        scenario_planning_batch_chars=scenario_planning_batch_chars,
    )


def resolve_one_step_chunks(text: str, runtime: OneStepPipelineRuntime) -> List[str]:
    if runtime.pre_split_chunks:
        return list(runtime.pre_split_chunks)
    from qa.chunking import build_tree_chunks

    inline_doc_id = hashlib.sha1(str(text or "").encode("utf-8")).hexdigest()
    chunks, meta, _report = build_tree_chunks(
        text,
        chunk_size=max(1, int(runtime.chunk_size)),
        original_filename="inline.txt",
        task_id=f"inline-{inline_doc_id[:16]}",
        doc_id=inline_doc_id,
        split_type="text",
    )
    runtime.pre_split_chunk_meta = meta
    return chunks


def _dedup_chunk_pool(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("question") or ""), str(item.get("answer") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _append_wall_interval(
    intervals: List[Dict[str, Any]],
    stage: str,
    started_at: float,
    ended_at: Optional[float] = None,
) -> None:
    end = time.perf_counter() if ended_at is None else float(ended_at)
    start = float(started_at)
    if end <= start:
        return
    intervals.append(
        {
            "stage": stage,
            "start": start,
            "end": end,
            "seconds": end - start,
        }
    )


def _effective_qa_detail_mode(mode: str) -> str:
    normalized = str(mode or "point").strip().lower()
    return normalized if normalized in {"point", "summary"} else "point"


def run_one_step_chunk_worker(
    *,
    chunk_index: int,
    chunk_text: str,
    source_chunk_meta: Dict[str, Any],
    evidence_index: QADocumentEvidenceIndex,
    runtime: OneStepPipelineRuntime,
    client: Any,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]],
    item_normalizer_with_reason: Callable[..., Tuple[Optional[Dict[str, Any]], str]],
    source_override_handler: Callable[..., None],
) -> Dict[str, Any]:
    target = max(1, int(runtime.qa_per_chunk))
    effective_qa_detail_mode = _effective_qa_detail_mode(runtime.qa_detail_mode)
    max_attempts = max(1, int(runtime.strict_max_attempts))
    # A planned scenario is already the semantic candidate. One LLM draft plus
    # one editor decision per attempt prevents the model from mining several
    # questions from the same scenario merely because an old multiplier is set.
    candidate_count = 1
    plan_full = build_question_type_plan(
        question_type_mode=runtime.question_type_mode,
        question_types=runtime.question_types,
        question_type_weights=runtime.question_type_weights,
        qa_per_chunk=candidate_count,
        seed_text=chunk_text,
    )

    attempt_used_total = 0
    items_final: List[Dict[str, Any]] = []
    dropped_answer_reasons: Dict[str, int] = {}
    seen_questions: set[str] = set()
    chunk_started_at = time.perf_counter()
    wall_intervals: List[Dict[str, Any]] = []
    candidate_question_seconds = 0.0
    question_editor_seconds = 0.0
    retrieval_embedding_seconds = 0.0
    retrieval_ranking_seconds = 0.0
    retrieval_unit_seconds = 0.0
    answer_generation_seconds = 0.0
    candidate_questions_total = 0
    candidates_considered = 0
    skipped_empty_or_duplicate = 0
    candidate_retry_used = False
    for attempt_index in range(1, max_attempts + 1):
        attempt_used_total = attempt_index
        candidate_started_at = time.perf_counter()
        candidate_generation_error = ""
        try:
            candidates = call_candidate_question_llm(
                client=client,
                model=runtime.model,
                source_chunk_text=chunk_text,
                source_chunk_meta=source_chunk_meta,
                candidate_count=candidate_count,
                prompt_language=runtime.prompt_language,
                question_type_plan=plan_full,
                few_shot_examples=runtime.few_shot_examples,
                request_timeout=runtime.request_timeout,
                qa_detail_mode=effective_qa_detail_mode,
                knowledge_category=(
                    runtime.fixed_knowledge_category
                    if runtime.use_category_prompt_templates
                    else None
                ),
                chunk_index=chunk_index,
                debug_writer=debug_writer,
            )
        except Exception as exc:
            candidates = []
            candidate_generation_error = str(exc)
            dropped_answer_reasons["candidate_generation_error"] = (
                dropped_answer_reasons.get("candidate_generation_error", 0) + 1
            )
        candidate_ended_at = time.perf_counter()
        candidate_question_seconds += candidate_ended_at - candidate_started_at
        _append_wall_interval(
            wall_intervals,
            "candidate_question",
            candidate_started_at,
            candidate_ended_at,
        )
        candidate_questions_total += len(candidates)
        if candidate_generation_error:
            if not candidate_retry_used and attempt_index < max_attempts:
                candidate_retry_used = True
                continue
            break
        if not candidates:
            dropped_answer_reasons["candidate_empty"] = (
                dropped_answer_reasons.get("candidate_empty", 0) + 1
            )
            if not candidate_retry_used and attempt_index < max_attempts:
                candidate_retry_used = True
                continue
            break
        edited_candidates: List[Dict[str, Any]] = []
        editor_drop_reasons: Dict[str, int] = {}
        source_unit_payload = (
            source_chunk_meta.get("_qa_source_unit")
            if isinstance(source_chunk_meta.get("_qa_source_unit"), dict)
            else {}
        )
        for candidate in candidates:
            editor_started_at = time.perf_counter()
            editor_error = ""
            try:
                edited_candidate, editor_status = call_question_editor_llm(
                    client=client,
                    model=runtime.model,
                    candidate=candidate,
                    source_material=chunk_text,
                    scenario_intent=str(source_unit_payload.get("scenario_intent") or ""),
                    reader_need=str(source_unit_payload.get("reader_need") or ""),
                    qa_detail_mode=effective_qa_detail_mode,
                    prompt_language=runtime.prompt_language,
                    request_timeout=runtime.request_timeout,
                    chunk_index=chunk_index,
                    source_material_path=str(
                        source_chunk_meta.get("qa_generation_unit_title_path")
                        or source_chunk_meta.get("title_path")
                        or ""
                    ).strip()
                    or None,
                    source_material_paths=(
                        list(source_chunk_meta.get("qa_generation_unit_material_paths") or [])
                        if isinstance(source_chunk_meta.get("qa_generation_unit_material_paths"), list)
                        else None
                    ),
                    debug_writer=debug_writer,
                )
            except Exception as exc:
                edited_candidate, editor_status = None, "question_editor_error"
                editor_error = str(exc)
            editor_ended_at = time.perf_counter()
            question_editor_seconds += editor_ended_at - editor_started_at
            _append_wall_interval(
                wall_intervals,
                "question_editor",
                editor_started_at,
                editor_ended_at,
            )
            if edited_candidate:
                edited_candidates.append(edited_candidate)
            else:
                editor_drop_reasons[editor_status] = editor_drop_reasons.get(editor_status, 0) + 1
                if editor_error and debug_writer:
                    debug_writer(
                        {
                            "event": "question_editor_llm_call",
                            "chunk_index": chunk_index,
                            "candidate": candidate,
                            "result_status": editor_status,
                            "parse_error": editor_error,
                        }
                    )
        candidates = edited_candidates
        for reason, count in editor_drop_reasons.items():
            dropped_answer_reasons[f"question_editor_{reason}"] = (
                dropped_answer_reasons.get(f"question_editor_{reason}", 0) + count
            )
        if not candidates:
            if not candidate_retry_used and attempt_index < max_attempts:
                candidate_retry_used = True
                continue
            break
        retrieval_timing: Dict[str, float] = {}
        retrieval_started_at = time.perf_counter()
        source_indexes = [
            int(value)
            for value in source_unit_payload.get("source_chunk_indexes") or [chunk_index]
        ]
        source_chunk_ids = [
            str(evidence_index.get_chunk(index).get("chunk_id") or "")
            for index in source_indexes
        ]
        retrieval_questions = [
            str(candidate.get("question") or "").strip()
            for candidate in candidates
            if str(candidate.get("question") or "").strip()
        ]
        try:
            retrieval_map = evidence_index.retrieve_many(
                retrieval_questions,
                source_chunk_ids=source_chunk_ids,
                final_evidence_k=runtime.final_evidence_k,
                evidence_token_budget=runtime.evidence_token_budget,
                timing=retrieval_timing,
            )
        except Exception as exc:
            dropped_answer_reasons["retrieval_error"] = (
                dropped_answer_reasons.get("retrieval_error", 0) + 1
            )
            if debug_writer:
                debug_writer(
                    {
                        "event": "retrieval_error",
                        "chunk_index": chunk_index,
                        "questions": retrieval_questions,
                        "error": str(exc),
                    }
                )
            continue
        retrieval_ended_at = time.perf_counter()
        _append_wall_interval(
            wall_intervals,
            "retrieval",
            retrieval_started_at,
            retrieval_ended_at,
        )
        retrieval_embedding_seconds += float(retrieval_timing.get("embedding_seconds") or 0.0)
        retrieval_ranking_seconds += float(retrieval_timing.get("ranking_seconds") or 0.0)
        for candidate in candidates:
            validation_started_at = time.perf_counter()
            question_key = str(candidate.get("question") or "").strip()
            if not question_key or question_key in seen_questions:
                skipped_empty_or_duplicate += 1
                _append_wall_interval(
                    wall_intervals,
                    "validation_and_bookkeeping",
                    validation_started_at,
                )
                continue
            seen_questions.add(question_key)
            candidates_considered += 1
            candidate_for_answer = dict(candidate)
            retrieval_result = retrieval_map.get(question_key) or {
                "selected_windows": [],
                "selected_chunk_ids": [],
                "trace": {},
            }
            _append_wall_interval(
                wall_intervals,
                "validation_and_bookkeeping",
                validation_started_at,
            )
            unit_started_at = time.perf_counter()
            generation_unit = evidence_index.build_generation_unit(
                source_chunk_index=chunk_index,
                source_unit=source_unit_payload,
                question=question_key,
                retrieval_result=retrieval_result,
                final_evidence_k=runtime.final_evidence_k,
                evidence_token_budget=runtime.evidence_token_budget,
            )
            unit_ended_at = time.perf_counter()
            retrieval_unit_seconds += unit_ended_at - unit_started_at
            _append_wall_interval(
                wall_intervals,
                "retrieval",
                unit_started_at,
                unit_ended_at,
            )
            answer_started_at = time.perf_counter()
            try:
                item, reason = call_evidence_answer_llm(
                    client=client,
                    model=runtime.model,
                    candidate=candidate_for_answer,
                    generation_unit=generation_unit,
                    qa_detail_mode=effective_qa_detail_mode,
                    prompt_language=runtime.prompt_language,
                    request_timeout=runtime.request_timeout,
                    item_normalizer_with_reason=item_normalizer_with_reason,
                    source_override_handler=source_override_handler,
                    fixed_knowledge_category=runtime.fixed_knowledge_category,
                    fixed_knowledge_category_confidence=runtime.fixed_knowledge_category_confidence,
                    fixed_knowledge_category_reason=runtime.fixed_knowledge_category_reason,
                    use_category_prompt_templates=runtime.use_category_prompt_templates,
                    chunk_index=chunk_index,
                    debug_writer=debug_writer,
                )
            except Exception as exc:
                item, reason = None, "answer_generation_error"
                dropped_answer_reasons[reason] = dropped_answer_reasons.get(reason, 0) + 1
                if debug_writer:
                    debug_writer(
                        {
                            "event": "answer_generation_error",
                            "chunk_index": chunk_index,
                            "candidate": candidate_for_answer,
                            "error": str(exc),
                        }
                    )
            answer_ended_at = time.perf_counter()
            answer_generation_seconds += answer_ended_at - answer_started_at
            _append_wall_interval(
                wall_intervals,
                "answer_generation",
                answer_started_at,
                answer_ended_at,
            )
            validation_started_at = time.perf_counter()
            if item:
                items_final.append(item)
                if len(items_final) >= target:
                    _append_wall_interval(
                        wall_intervals,
                        "validation_and_bookkeeping",
                        validation_started_at,
                    )
                    break
            else:
                dropped_answer_reasons[reason] = dropped_answer_reasons.get(reason, 0) + 1
            _append_wall_interval(
                wall_intervals,
                "validation_and_bookkeeping",
                validation_started_at,
            )
        if len(items_final) >= target:
            break

    validation_started_at = time.perf_counter()
    items_final = _dedup_chunk_pool(items_final)[:target]
    chunk_finished_at = time.perf_counter()
    _append_wall_interval(
        wall_intervals,
        "validation_and_bookkeeping",
        validation_started_at,
        chunk_finished_at,
    )
    chunk_total_seconds = chunk_finished_at - chunk_started_at
    retrieval_seconds = (
        retrieval_embedding_seconds + retrieval_ranking_seconds + retrieval_unit_seconds
    )
    measured_seconds = (
        candidate_question_seconds
        + question_editor_seconds
        + retrieval_seconds
        + answer_generation_seconds
    )
    validation_and_bookkeeping_seconds = max(0.0, chunk_total_seconds - measured_seconds)
    dropped_reason_stats = dict(dropped_answer_reasons)
    if skipped_empty_or_duplicate:
        dropped_reason_stats["empty_or_duplicate_question"] = skipped_empty_or_duplicate

    if runtime.include_chunk_index:
        for item in items_final:
            if isinstance(item, dict):
                item["chunk_index"] = chunk_index

    selected_evidence_window_count = 0
    selected_evidence_chunk_count = 0
    for item in items_final:
        trace = item.get("retrieval_trace") if isinstance(item, dict) else None
        if not isinstance(trace, dict):
            continue
        selected_windows = trace.get("selected_windows")
        selected_chunks = trace.get("selected_evidence_chunk_ids")
        selected_evidence_window_count += int(
            trace.get("selected_evidence_window_count")
            if trace.get("selected_evidence_window_count") is not None
            else len(selected_windows) if isinstance(selected_windows, list) else 0
        )
        selected_evidence_chunk_count += int(
            trace.get("selected_evidence_chunk_count")
            if trace.get("selected_evidence_chunk_count") is not None
            else len(selected_chunks) if isinstance(selected_chunks, list) else 0
        )

    return {
        "chunk_index": chunk_index,
        "attempt_used": attempt_used_total,
        "items": items_final,
        "dropped_answer_reasons": dropped_answer_reasons,
        "candidate_questions": candidate_questions_total,
        "candidates_considered": candidates_considered,
        "valid_items": len(items_final),
        "selected_evidence_window_count": selected_evidence_window_count,
        "selected_evidence_chunk_count": selected_evidence_chunk_count,
        "dropped_reason_stats": dropped_reason_stats,
        "wall_intervals": wall_intervals,
        "timing": {
            "chunk_total_seconds": chunk_total_seconds,
            "candidate_question_seconds": candidate_question_seconds,
            "question_editor_seconds": question_editor_seconds,
            "retrieval_seconds": retrieval_seconds,
            "retrieval_embedding_seconds": retrieval_embedding_seconds,
            "retrieval_ranking_seconds": retrieval_ranking_seconds,
            "retrieval_unit_seconds": retrieval_unit_seconds,
            "answer_generation_seconds": answer_generation_seconds,
            "validation_and_bookkeeping_seconds": validation_and_bookkeeping_seconds,
        },
    }


def run_one_step_unit_worker(
    *,
    unit: GenerationUnit,
    evidence_index: QADocumentEvidenceIndex,
    runtime: OneStepPipelineRuntime,
    client: Any,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]],
    item_normalizer_with_reason: Callable[..., Tuple[Optional[Dict[str, Any]], str]],
    source_override_handler: Callable[..., None],
) -> Dict[str, Any]:
    if int(unit.qa_budget or 0) <= 0:
        return {
            "unit_index": unit.unit_index,
            "unit_id": unit.unit_id,
            "unit_type": unit.unit_type,
            "qa_mode": unit.qa_mode,
            "chunk_index": unit.anchor_chunk_index,
            "anchor_chunk_index": unit.anchor_chunk_index,
            "source_chunk_indexes": list(unit.source_chunk_indexes),
            "attempt_used": 0,
            "items": [],
            "valid_items": 0,
            "skip_reason": "zero_unit_budget",
            "timing": {},
        }

    unit_runtime = replace(
        runtime,
        qa_per_chunk=1,
        qa_detail_mode=_effective_qa_detail_mode(unit.qa_mode),
    )
    source_chunk_meta = dict(unit.source_chunk_meta)
    source_chunk_meta["_qa_source_unit"] = unit.to_source_unit()
    payload = run_one_step_chunk_worker(
        chunk_index=int(unit.anchor_chunk_index),
        chunk_text=unit.unit_text,
        source_chunk_meta=source_chunk_meta,
        evidence_index=evidence_index,
        runtime=unit_runtime,
        client=client,
        debug_writer=debug_writer,
        item_normalizer_with_reason=item_normalizer_with_reason,
        source_override_handler=source_override_handler,
    )
    payload.update(
        {
            "unit_index": unit.unit_index,
            "unit_id": unit.unit_id,
            "unit_type": unit.unit_type,
            "qa_mode": unit.qa_mode,
            "anchor_chunk_index": unit.anchor_chunk_index,
            "source_chunk_indexes": list(unit.source_chunk_indexes),
            "section_path": unit.section_path,
            "quality_child_coverage": unit.quality_child_coverage,
            "unit_debug": unit.to_debug_dict(),
        }
    )
    items = payload.get("items") if isinstance(payload, dict) else None
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            item["qa_generation_unit_id"] = unit.unit_id
            item["qa_generation_unit_index"] = unit.unit_index
            item["qa_generation_unit_type"] = unit.unit_type
            item["qa_generation_unit_mode"] = unit.qa_mode
            item["qa_generation_scenario_intent"] = unit.scenario_intent
            item["qa_generation_reader_need"] = unit.reader_need
            item["qa_generation_material_ids"] = list(unit.material_ids)
            item["qa_generation_unit_source_chunk_indexes"] = list(unit.source_chunk_indexes)
            item["qa_generation_unit_section_path"] = unit.section_path
            item["qa_generation_unit_quality_child_coverage"] = unit.quality_child_coverage
    return payload


__all__ = [
    "OneStepPipelineRuntime",
    "parse_one_step_pipeline_runtime",
    "resolve_one_step_chunks",
    "run_one_step_chunk_worker",
    "run_one_step_unit_worker",
]
