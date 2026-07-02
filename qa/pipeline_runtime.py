# 文件作用：解析完整 QA pipeline 的运行配置并执行 chunk 级 worker。
# 关联说明：被 text_to_qa_pipeline 调用，负责运行配置和单 chunk worker。

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.generation import (
    DEFAULT_MAX_UNIT_CHARS,
    DEFAULT_SEMANTIC_TOP_K,
    QADocumentEvidenceIndex,
    build_question_type_plan,
    call_candidate_question_llm,
    call_evidence_answer_llm,
    normalize_question_type_mode,
    normalize_question_type_weights,
    normalize_question_types,
)
from qa.chunking import split_text


@dataclass
class OneStepPipelineRuntime:
    chunk_size: int
    qa_per_chunk: int
    qa_detail_mode: str
    prompt_language: str
    chunk_max_concurrency: int
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
    candidate_multiplier: int
    semantic_top_k: int
    max_unit_chars: int


def parse_one_step_pipeline_runtime(config: Dict[str, Any]) -> OneStepPipelineRuntime:
    chunk_size = int(config.get("chunk_size") or 600)
    qa_per_chunk = int(config.get("qa_per_chunk") or 1)
    qa_detail_mode = str(config.get("qa_detail_mode") or "point")
    prompt_language = str(config.get("prompt_language") or "auto")
    chunk_max_concurrency = int(config.get("chunk_max_concurrency") or 8)
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

    candidate_multiplier = max(1, int(config.get("candidate_multiplier") or 2))
    semantic_top_k = max(0, int(config.get("semantic_top_k") or DEFAULT_SEMANTIC_TOP_K))
    max_unit_chars = max(1000, int(config.get("max_unit_chars") or DEFAULT_MAX_UNIT_CHARS))

    return OneStepPipelineRuntime(
        chunk_size=chunk_size,
        qa_per_chunk=qa_per_chunk,
        qa_detail_mode=qa_detail_mode,
        prompt_language=prompt_language,
        chunk_max_concurrency=chunk_max_concurrency,
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
        candidate_multiplier=candidate_multiplier,
        semantic_top_k=semantic_top_k,
        max_unit_chars=max_unit_chars,
    )


def resolve_one_step_chunks(text: str, runtime: OneStepPipelineRuntime) -> List[str]:
    if runtime.pre_split_chunks:
        return list(runtime.pre_split_chunks)
    return split_text(text, chunk_size=max(1, int(runtime.chunk_size)))


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
    source_fact_detail_validator: Callable[..., Tuple[bool, str]],
    source_fact_grounding_validator: Callable[..., Tuple[bool, str]],
    source_override_handler: Callable[..., None],
) -> Dict[str, Any]:
    target = max(1, int(runtime.qa_per_chunk))
    max_attempts = max(1, int(runtime.strict_max_attempts))
    candidate_count = max(target, target * max(1, int(runtime.candidate_multiplier)))
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
    candidate_question_seconds = 0.0
    retrieval_embedding_seconds = 0.0
    retrieval_ranking_seconds = 0.0
    retrieval_unit_seconds = 0.0
    answer_generation_seconds = 0.0
    candidate_questions_total = 0
    candidates_considered = 0
    skipped_empty_or_duplicate = 0
    for attempt_index in range(1, max_attempts + 1):
        attempt_used_total = attempt_index
        candidate_started_at = time.perf_counter()
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
            knowledge_category=(
                runtime.fixed_knowledge_category
                if runtime.use_category_prompt_templates
                else None
            ),
            chunk_index=chunk_index,
            debug_writer=debug_writer,
        )
        candidate_question_seconds += time.perf_counter() - candidate_started_at
        candidate_questions_total += len(candidates)
        retrieval_timing: Dict[str, float] = {}
        retrieval_map = evidence_index.retrieve_many(
            [str(candidate.get("question") or "") for candidate in candidates],
            source_chunk_index=chunk_index,
            top_k=runtime.semantic_top_k,
            timing=retrieval_timing,
        )
        retrieval_embedding_seconds += float(retrieval_timing.get("embedding_seconds") or 0.0)
        retrieval_ranking_seconds += float(retrieval_timing.get("ranking_seconds") or 0.0)
        for candidate in candidates:
            question_key = str(candidate.get("question") or "").strip()
            if not question_key or question_key in seen_questions:
                skipped_empty_or_duplicate += 1
                continue
            seen_questions.add(question_key)
            candidates_considered += 1
            semantic_hits, raw_semantic_trace = retrieval_map.get(question_key, (None, None))
            unit_started_at = time.perf_counter()
            generation_unit = evidence_index.build_generation_unit(
                source_chunk_index=chunk_index,
                question=question_key,
                source_anchor_text=str(candidate.get("source_anchor_text") or ""),
                semantic_top_k=runtime.semantic_top_k,
                max_unit_chars=runtime.max_unit_chars,
                semantic_hits=semantic_hits,
                raw_semantic_trace=raw_semantic_trace,
            )
            retrieval_unit_seconds += time.perf_counter() - unit_started_at
            answer_started_at = time.perf_counter()
            item, reason = call_evidence_answer_llm(
                client=client,
                model=runtime.model,
                candidate=candidate,
                generation_unit=generation_unit,
                qa_detail_mode=runtime.qa_detail_mode,
                prompt_language=runtime.prompt_language,
                request_timeout=runtime.request_timeout,
                item_normalizer_with_reason=item_normalizer_with_reason,
                source_fact_detail_validator=source_fact_detail_validator,
                source_fact_grounding_validator=source_fact_grounding_validator,
                source_override_handler=source_override_handler,
                fixed_knowledge_category=runtime.fixed_knowledge_category,
                fixed_knowledge_category_confidence=runtime.fixed_knowledge_category_confidence,
                fixed_knowledge_category_reason=runtime.fixed_knowledge_category_reason,
                use_category_prompt_templates=runtime.use_category_prompt_templates,
                chunk_index=chunk_index,
                debug_writer=debug_writer,
            )
            answer_generation_seconds += time.perf_counter() - answer_started_at
            if item:
                items_final.append(item)
                if len(items_final) >= target:
                    break
            else:
                dropped_answer_reasons[reason] = dropped_answer_reasons.get(reason, 0) + 1
        if len(items_final) >= target:
            break

    items_final = _dedup_chunk_pool(items_final)[:target]
    chunk_total_seconds = time.perf_counter() - chunk_started_at
    retrieval_seconds = (
        retrieval_embedding_seconds + retrieval_ranking_seconds + retrieval_unit_seconds
    )
    measured_seconds = (
        candidate_question_seconds + retrieval_seconds + answer_generation_seconds
    )
    validation_and_bookkeeping_seconds = max(0.0, chunk_total_seconds - measured_seconds)
    dropped_reason_stats = dict(dropped_answer_reasons)
    if skipped_empty_or_duplicate:
        dropped_reason_stats["empty_or_duplicate_question"] = skipped_empty_or_duplicate

    if runtime.include_chunk_index:
        for item in items_final:
            if isinstance(item, dict):
                item["chunk_index"] = chunk_index

    return {
        "chunk_index": chunk_index,
        "attempt_used": attempt_used_total,
        "items": items_final,
        "dropped_answer_reasons": dropped_answer_reasons,
        "candidate_questions": candidate_questions_total,
        "candidates_considered": candidates_considered,
        "valid_items": len(items_final),
        "dropped_reason_stats": dropped_reason_stats,
        "timing": {
            "chunk_total_seconds": chunk_total_seconds,
            "candidate_question_seconds": candidate_question_seconds,
            "retrieval_seconds": retrieval_seconds,
            "retrieval_embedding_seconds": retrieval_embedding_seconds,
            "retrieval_ranking_seconds": retrieval_ranking_seconds,
            "retrieval_unit_seconds": retrieval_unit_seconds,
            "answer_generation_seconds": answer_generation_seconds,
            "validation_and_bookkeeping_seconds": validation_and_bookkeeping_seconds,
        },
    }


__all__ = [
    "OneStepPipelineRuntime",
    "parse_one_step_pipeline_runtime",
    "resolve_one_step_chunks",
    "run_one_step_chunk_worker",
]
