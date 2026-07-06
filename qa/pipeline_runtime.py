# 文件作用：解析完整 QA pipeline 的运行配置并执行 chunk 级 worker。
# 关联说明：被 text_to_qa_pipeline 调用，负责运行配置和单 chunk worker。

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.generation import (
    DEFAULT_HYBRID_WEIGHT_DENSE,
    DEFAULT_HYBRID_WEIGHT_LEXICAL,
    DEFAULT_MAX_UNIT_CHARS,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_SEMANTIC_TOP_K,
    DEFAULT_STRUCTURE_WEIGHT,
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
    retrieval_mode: str
    hybrid_weight_dense: float
    hybrid_weight_lexical: float
    retrieval_structure_weight: float
    rerank_top_n: int
    answer_scope_policy: str


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
    retrieval_mode = str(config.get("retrieval_mode") or DEFAULT_RETRIEVAL_MODE).strip().lower()
    if retrieval_mode not in {"semantic", "hybrid"}:
        retrieval_mode = DEFAULT_RETRIEVAL_MODE
    try:
        hybrid_weight_dense = float(config.get("hybrid_weight_dense", DEFAULT_HYBRID_WEIGHT_DENSE))
    except Exception:
        hybrid_weight_dense = DEFAULT_HYBRID_WEIGHT_DENSE
    try:
        hybrid_weight_lexical = float(config.get("hybrid_weight_lexical", DEFAULT_HYBRID_WEIGHT_LEXICAL))
    except Exception:
        hybrid_weight_lexical = DEFAULT_HYBRID_WEIGHT_LEXICAL
    try:
        retrieval_structure_weight = float(config.get("retrieval_structure_weight", DEFAULT_STRUCTURE_WEIGHT))
    except Exception:
        retrieval_structure_weight = DEFAULT_STRUCTURE_WEIGHT
    hybrid_weight_dense = max(0.0, min(1.0, hybrid_weight_dense))
    hybrid_weight_lexical = max(0.0, min(1.0, hybrid_weight_lexical))
    retrieval_structure_weight = max(0.0, min(0.5, retrieval_structure_weight))
    rerank_top_n = max(1, int(config.get("rerank_top_n") or DEFAULT_RERANK_TOP_N))
    answer_scope_policy = str(config.get("answer_scope_policy") or "source_primary").strip().lower()
    if answer_scope_policy not in {"source_primary", "same_section", "cross_chunk"}:
        answer_scope_policy = "source_primary"

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
        retrieval_mode=retrieval_mode,
        hybrid_weight_dense=hybrid_weight_dense,
        hybrid_weight_lexical=hybrid_weight_lexical,
        retrieval_structure_weight=retrieval_structure_weight,
        rerank_top_n=rerank_top_n,
        answer_scope_policy=answer_scope_policy,
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


def _normalize_answer_scope(raw: Any) -> str:
    scope = str(raw or "").strip().lower()
    return scope if scope in {"source_primary", "same_section", "cross_chunk"} else "source_primary"


def _candidate_answer_scope_hint(candidate: Dict[str, Any]) -> str:
    return _normalize_answer_scope(
        candidate.get("answer_scope_hint") or candidate.get("answer_scope")
    )


def _retrieval_scope_for_hint(candidate_scope_hint: Any, policy: str) -> str:
    requested = _normalize_answer_scope(candidate_scope_hint)
    policy_scope = _normalize_answer_scope(policy)
    if policy_scope == "source_primary":
        return "source_primary"
    if policy_scope == "same_section" and requested == "cross_chunk":
        return "same_section"
    if policy_scope == "same_section":
        return requested if requested in {"source_primary", "same_section"} else "source_primary"
    return requested


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _scope_trace_summary(trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    return {
        "chunk_index": trace.get("chunk_index"),
        "title_path": trace.get("title_path"),
        "score": trace.get("score"),
        "dense_score": trace.get("dense_score"),
        "lexical_score": trace.get("lexical_score"),
        "structure_score": trace.get("structure_score"),
        "must_term_hits": trace.get("must_term_hits"),
        "must_term_total": trace.get("must_term_total"),
        "must_term_coverage": trace.get("must_term_coverage"),
        "score_gap_top1_top2": trace.get("score_gap_top1_top2"),
        "same_parent": bool(trace.get("same_parent")),
        "adjacent": bool(trace.get("adjacent")),
        "title_overlap": trace.get("title_overlap"),
        "rerank_score": trace.get("rerank_score")
        if trace.get("rerank_score") is not None
        else trace.get("cross_encoder_score"),
    }


def _term_evidence_ok(trace: Dict[str, Any]) -> bool:
    total = int(_safe_float(trace.get("must_term_total"), 0.0))
    if total <= 0:
        return _safe_float(trace.get("lexical_score"), 0.0) >= 0.08
    return (
        _safe_float(trace.get("must_term_coverage"), 0.0) >= 0.34
        or _safe_float(trace.get("lexical_score"), 0.0) >= 0.18
    )


def _gap_evidence_ok(trace: Dict[str, Any]) -> bool:
    gap = trace.get("score_gap_top1_top2")
    if gap is None:
        return True
    return (
        _safe_float(gap, 0.0) >= 0.008
        or _safe_float(trace.get("lexical_score"), 0.0) >= 0.24
        or _safe_float(trace.get("structure_score"), 0.0) >= 0.45
    )


def _same_section_evidence_ok(trace: Dict[str, Any]) -> bool:
    relation_ok = bool(trace.get("same_parent") or trace.get("adjacent"))
    score_ok = (
        _safe_float(trace.get("score"), 0.0) >= 0.38
        or _safe_float(trace.get("structure_score"), 0.0) >= 0.45
    )
    return relation_ok and score_ok and _term_evidence_ok(trace) and _gap_evidence_ok(trace)


def _cross_chunk_evidence_ok(trace: Dict[str, Any]) -> bool:
    score_ok = _safe_float(trace.get("score"), 0.0) >= 0.42
    rerank_score = trace.get("rerank_score")
    rerank_ok = rerank_score is None or _safe_float(rerank_score, 0.0) >= 0.0
    return score_ok and rerank_ok and _term_evidence_ok(trace) and _gap_evidence_ok(trace)


def _decide_effective_answer_scope(
    *,
    candidate_scope_hint: Any,
    policy: str,
    raw_semantic_trace: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    hint = _normalize_answer_scope(candidate_scope_hint)
    policy_scope = _normalize_answer_scope(policy)
    non_source = [
        trace
        for trace in (raw_semantic_trace or [])
        if isinstance(trace, dict) and not trace.get("is_source_chunk")
    ]
    same_section_candidates = [
        trace for trace in non_source if trace.get("same_parent") or trace.get("adjacent")
    ]
    best_same = same_section_candidates[0] if same_section_candidates else None
    best_any = non_source[0] if non_source else None

    decision: Dict[str, Any] = {
        "answer_scope_hint": hint,
        "answer_scope_policy": policy_scope,
        "effective_answer_scope": "source_primary",
        "reason_code": "default_source_primary",
        "reason": "默认只使用主来源块，避免补充证据漂移。",
        "evidence": {
            "best_same_section": _scope_trace_summary(best_same),
            "best_any": _scope_trace_summary(best_any),
        },
        "checks": {
            "same_section_evidence_ok": _same_section_evidence_ok(best_same)
            if best_same
            else False,
            "cross_chunk_evidence_ok": _cross_chunk_evidence_ok(best_any)
            if best_any
            else False,
        },
    }

    if policy_scope == "source_primary":
        decision.update(
            {
                "reason_code": "policy_source_primary",
                "reason": "前端策略限制为只使用主来源块，系统未放宽证据范围。",
            }
        )
        return decision

    if hint == "source_primary":
        decision.update(
            {
                "reason_code": "hint_source_primary",
                "reason": "模型建议主来源块已足够，系统保持主来源块范围。",
            }
        )
        return decision

    same_ok = bool(decision["checks"]["same_section_evidence_ok"])
    cross_ok = bool(decision["checks"]["cross_chunk_evidence_ok"])

    if policy_scope == "cross_chunk" and hint == "cross_chunk" and best_any and cross_ok:
        decision.update(
            {
                "effective_answer_scope": "cross_chunk",
                "reason_code": "cross_chunk_evidence_approved",
                "reason": "前端允许跨 chunk，且召回证据的综合分、关键词覆盖和分数间隔达到阈值。",
            }
        )
        return decision

    if hint in {"same_section", "cross_chunk"} and best_same and same_ok:
        decision.update(
            {
                "effective_answer_scope": "same_section",
                "reason_code": "same_section_evidence_approved",
                "reason": "前端允许同章节补充，且召回证据来自同章节或相邻块，质量达到阈值。",
            }
        )
        return decision

    decision.update(
        {
            "reason_code": "evidence_not_confident",
            "reason": "补充证据的章节关系、综合分、关键词覆盖或 top1/top2 分差不足，系统收窄到主来源块。",
        }
    )
    return decision


def _candidate_retrieval_query(candidate: Dict[str, Any], source_chunk_meta: Dict[str, Any]) -> str:
    explicit = str(candidate.get("retrieval_query") or "").strip()
    if explicit:
        return explicit
    parts = [
        str(candidate.get("question") or "").strip(),
        str(candidate.get("source_anchor_text") or "").strip(),
        str(source_chunk_meta.get("title_path") or "").strip(),
    ]
    terms = candidate.get("must_have_terms")
    if isinstance(terms, list):
        parts.extend(str(term).strip() for term in terms if str(term).strip())
    return "\n".join(part for part in parts if part)


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
    wall_intervals: List[Dict[str, Any]] = []
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
        candidate_ended_at = time.perf_counter()
        candidate_question_seconds += candidate_ended_at - candidate_started_at
        _append_wall_interval(
            wall_intervals,
            "candidate_question",
            candidate_started_at,
            candidate_ended_at,
        )
        candidate_questions_total += len(candidates)
        retrieval_timing: Dict[str, float] = {}
        retrieval_started_at = time.perf_counter()
        retrieval_payloads = [
            {
                "key": str(candidate.get("question") or "").strip(),
                "query": _candidate_retrieval_query(candidate, source_chunk_meta),
                "must_have_terms": candidate.get("must_have_terms") or [],
                "answer_scope": _retrieval_scope_for_hint(
                    _candidate_answer_scope_hint(candidate),
                    runtime.answer_scope_policy,
                ),
            }
            for candidate in candidates
            if str(candidate.get("question") or "").strip()
        ]
        retrieval_map = evidence_index.retrieve_many(
            retrieval_payloads,
            source_chunk_index=chunk_index,
            top_k=runtime.semantic_top_k,
            timing=retrieval_timing,
            retrieval_mode=runtime.retrieval_mode,
            hybrid_weight_dense=runtime.hybrid_weight_dense,
            hybrid_weight_lexical=runtime.hybrid_weight_lexical,
            structure_weight=runtime.retrieval_structure_weight,
            rerank_top_n=runtime.rerank_top_n,
        )
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
            retrieval_query = _candidate_retrieval_query(candidate, source_chunk_meta)
            answer_scope_hint = _candidate_answer_scope_hint(candidate)
            candidate_for_answer = dict(candidate)
            candidate_for_answer["retrieval_query"] = retrieval_query
            candidate_for_answer["answer_scope_hint"] = answer_scope_hint
            candidate_for_answer["must_have_terms"] = candidate.get("must_have_terms") or []
            semantic_hits, raw_semantic_trace = retrieval_map.get(question_key, (None, None))
            scope_decision = _decide_effective_answer_scope(
                candidate_scope_hint=answer_scope_hint,
                policy=runtime.answer_scope_policy,
                raw_semantic_trace=raw_semantic_trace,
            )
            effective_answer_scope = str(
                scope_decision.get("effective_answer_scope") or "source_primary"
            )
            candidate_for_answer["answer_scope"] = effective_answer_scope
            candidate_for_answer["effective_answer_scope"] = effective_answer_scope
            candidate_for_answer["answer_scope_decision"] = scope_decision
            _append_wall_interval(
                wall_intervals,
                "validation_and_bookkeeping",
                validation_started_at,
            )
            unit_started_at = time.perf_counter()
            generation_unit = evidence_index.build_generation_unit(
                source_chunk_index=chunk_index,
                question=question_key,
                source_anchor_text=str(candidate_for_answer.get("source_anchor_text") or ""),
                retrieval_query=retrieval_query,
                must_have_terms=candidate_for_answer.get("must_have_terms") or [],
                answer_scope=effective_answer_scope,
                semantic_top_k=runtime.semantic_top_k,
                max_unit_chars=runtime.max_unit_chars,
                retrieval_mode=runtime.retrieval_mode,
                hybrid_weight_dense=runtime.hybrid_weight_dense,
                hybrid_weight_lexical=runtime.hybrid_weight_lexical,
                structure_weight=runtime.retrieval_structure_weight,
                rerank_top_n=runtime.rerank_top_n,
                semantic_hits=semantic_hits,
                raw_semantic_trace=raw_semantic_trace,
                answer_scope_hint=answer_scope_hint,
                answer_scope_policy=runtime.answer_scope_policy,
                answer_scope_decision=scope_decision,
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
            item, reason = call_evidence_answer_llm(
                client=client,
                model=runtime.model,
                candidate=candidate_for_answer,
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
        "wall_intervals": wall_intervals,
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
