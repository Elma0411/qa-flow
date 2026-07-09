# 文件作用：编排从输入文本到问答条目的完整生成流程。
# 关联说明：调用 chunking、generation、grounding、validation，是 QA 目录的完整流程编排层。

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from qa.grounding import (
    validate_source_fact_grounding as _validate_source_fact_grounding,
    validate_source_fact_text_detail_mode as _validate_source_fact_text_detail_mode,
)
from qa.validation import (
    validate_and_normalize_item_with_reason as _validate_and_normalize_item_with_reason,
)
from qa.pipeline_runtime import (
    parse_one_step_pipeline_runtime,
    resolve_one_step_chunks,
    run_one_step_unit_worker,
)
from qa.generation import QADocumentEvidenceIndex, build_document_chunks, plan_generation_units
from qa.chunking import split_text

DEFAULT_SOURCE_BY_LANGUAGE = {"zh": "文本内容", "en": "text content"}
GENERIC_SOURCE_LABELS_BY_LANGUAGE = {
    "zh": {"文本内容", "全文", "全文内容", "全篇", "本文"},
    "en": {"text content", "full text", "the passage", "passage", "entire text"},
}


def _extract_source_labels(chunk_text: str, language_code: str) -> List[Tuple[str, int]]:
    """
    Extract lightweight source labels (best-effort) from the chunk itself, so we can avoid
    always falling back to "文本内容"/"text content".
    """
    text = chunk_text or ""
    labels: List[Tuple[str, int]] = []
    seen: set[str] = set()

    if language_code == "zh":
        # Common patterns: 第一条/第二条/第3条/第2章/第1段...
        for m in re.finditer(r"(第[一二三四五六七八九十百千万0-9]+[条章节段])", text):
            label = m.group(1)
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append((label, m.start(1)))

        # Section-style headings: 一、... 二、... (keep a short, stable label)
        for m in re.finditer(r"(?m)^([一二三四五六七八九十百千万]+、[^\n]{0,40})", text):
            label = m.group(1).strip()
            if not label:
                continue
            label = re.sub(r"\s+", "", label)
            if len(label) > 24:
                label = label[:24]
            if label in seen:
                continue
            seen.add(label)
            labels.append((label, m.start(1)))

        # Attachments: 附件1 / 附件 2
        for m in re.finditer(r"(?m)^(附件\s*\d+)", text):
            label = re.sub(r"\s+", "", m.group(1).strip())
            if not label or label in seen:
                continue
            seen.add(label)
            labels.append((label, m.start(1)))
        return labels

    # Best-effort EN: "Article 1", "Section 2", "Chapter 3", "Paragraph 4"
    for m in re.finditer(
        r"\b(Article|Section|Chapter|Paragraph)\s+(\d+)\b", text, flags=re.IGNORECASE
    ):
        label = f"{m.group(1).title()} {m.group(2)}"
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append((label, m.start(1)))

    # Best-effort EN: "RULE ONE", "RULE 1"
    for m in re.finditer(r"(?m)^(RULE\s+\w+)", text, flags=re.IGNORECASE):
        label = m.group(1).strip().upper()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append((label, m.start(1)))
    return labels


def _infer_source_from_chunk(
    *, chunk_text: str, source_fact_text: str, language_code: str
) -> Optional[str]:
    labels = _extract_source_labels(chunk_text, language_code=language_code)
    if not labels:
        return None

    fact = (source_fact_text or "").strip()
    if fact:
        for label, _pos in labels:
            if label and label in fact:
                return label

        idx = (chunk_text or "").find(fact)
        if idx >= 0:
            chosen = None
            for label, pos in labels:
                if pos <= idx:
                    chosen = label
                else:
                    break
            if chosen:
                return chosen

    # If only one label exists and the chunk starts with it, use it as a fallback.
    if len(labels) == 1:
        only = labels[0][0]
        if (chunk_text or "").lstrip().startswith(only):
            return only
        return only
    return None


def _maybe_override_source(
    item: Dict[str, Any], *, chunk_text: str, language_code: str
) -> None:
    default_source = DEFAULT_SOURCE_BY_LANGUAGE.get(language_code, "文本内容")
    generic = GENERIC_SOURCE_LABELS_BY_LANGUAGE.get(language_code, {default_source})
    current = str(item.get("source") or "").strip()
    if current and current not in generic:
        return
    inferred = _infer_source_from_chunk(
        chunk_text=chunk_text,
        source_fact_text=str(item.get("source_fact_text") or ""),
        language_code=language_code,
    )
    if inferred:
        item["source"] = inferred


def _build_jsonl_debug_writer(path: Optional[str]) -> Optional[Callable[[Dict[str, Any]], None]]:
    if not path:
        return None
    lock = threading.Lock()

    def _write(record: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            # Best-effort debug logging: never break the pipeline
            return

    return _write


_GENERATION_WALL_STAGE_TO_FIELD = {
    "candidate_question": "candidate_question_seconds",
    "retrieval": "retrieval_seconds",
    "answer_generation": "answer_generation_seconds",
    "validation_and_bookkeeping": "validation_and_bookkeeping_seconds",
}

_GENERATION_WALL_FIELDS = (
    "candidate_question_seconds",
    "retrieval_seconds",
    "answer_generation_seconds",
    "validation_and_bookkeeping_seconds",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if number >= 0.0 else None


def _attribute_generation_wall_detail(
    intervals: List[Dict[str, Any]],
    *,
    document_started_at: float,
    document_finished_at: float,
) -> Dict[str, float]:
    document_total_seconds = max(0.0, document_finished_at - document_started_at)
    detail: Dict[str, float] = {field: 0.0 for field in _GENERATION_WALL_FIELDS}
    normalized: List[Tuple[float, float, str]] = []

    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        field = _GENERATION_WALL_STAGE_TO_FIELD.get(str(interval.get("stage") or ""))
        if not field:
            continue
        start = _safe_float(interval.get("start"))
        end = _safe_float(interval.get("end"))
        if start is None or end is None:
            continue
        start = max(document_started_at, start)
        end = min(document_finished_at, end)
        if end <= start:
            continue
        normalized.append((start, end, field))

    if normalized and document_total_seconds > 0:
        boundaries = {document_started_at, document_finished_at}
        for start, end, _field in normalized:
            boundaries.add(start)
            boundaries.add(end)
        ordered = sorted(boundaries)
        for idx in range(len(ordered) - 1):
            left = ordered[idx]
            right = ordered[idx + 1]
            if right <= left:
                continue
            active_fields = [
                field for start, end, field in normalized if start < right and end > left
            ]
            if not active_fields:
                continue
            share = (right - left) / len(active_fields)
            for field in active_fields:
                detail[field] += share

    attributed_seconds = sum(detail[field] for field in _GENERATION_WALL_FIELDS)
    detail["scheduler_gap_seconds"] = max(0.0, document_total_seconds - attributed_seconds)
    detail["document_total_seconds"] = document_total_seconds
    return detail


def _generation_timing_with_metadata(
    base: Dict[str, float],
    *,
    index_seconds: float,
    total_chunks: int,
    chunks_completed: int,
    total_generation_units: int,
    generation_units_completed: int,
    qa_generated: int,
    qa_total_limit: Optional[int],
    qa_total_limit_scope: str,
) -> Dict[str, Any]:
    return {
        **base,
        "index_build_seconds": index_seconds,
        "chunks_total": total_chunks,
        "chunks_completed": chunks_completed,
        "generation_units_total": total_generation_units,
        "generation_units_completed": generation_units_completed,
        "qa_generated": qa_generated,
        "qa_total_limit": qa_total_limit,
        "qa_total_limit_scope": qa_total_limit_scope,
    }


def process_text_to_qa_one_step(
    client: Any,
    text: str,
    config: Dict[str, Any],
    original_filename: str = "",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieval-augmented QA generation for Pipeline 8/8B:
    - Split the input text into source chunks.
    - Plan generation units from source chunks and lightweight structure.
    - Generate candidate questions from each generation unit.
    - Retrieve same-document evidence chunks by candidate-question semantics.
    - Generate final QA from the source unit plus organized evidence context.
    """
    document_started_at = time.perf_counter()
    runtime = parse_one_step_pipeline_runtime(config)
    debug_writer = _build_jsonl_debug_writer(runtime.debug_file)

    if not runtime.model:
        raise ValueError("model is required for process_text_to_qa_one_step")

    raw_chunks = resolve_one_step_chunks(text, runtime)
    if not raw_chunks:
        return []
    document_chunks = build_document_chunks(
        raw_chunks,
        runtime.pre_split_chunk_meta,
    )
    if not document_chunks:
        return []
    unit_plan = plan_generation_units(
        document_chunks,
        qa_total_limit=runtime.qa_total_limit,
        qa_per_chunk=runtime.qa_per_chunk,
        qa_detail_mode=runtime.qa_detail_mode,
        chunk_size=runtime.chunk_size,
        max_unit_chars=runtime.max_unit_chars,
    )
    generation_units = list(unit_plan.units)
    total_chunks = len(document_chunks)
    total_generation_units = len(generation_units)
    if not generation_units:
        if progress_callback:
            try:
                progress_callback(
                    {
                        "event": "start",
                        "generation_mode": "same_document_semantic_evidence",
                        "original_filename": original_filename,
                        "total_chunks": total_chunks,
                        "total_generation_units": 0,
                        "qa_total_limit": runtime.qa_total_limit,
                        "qa_total_limit_scope": runtime.qa_total_limit_scope,
                        "unit_plan_summary": unit_plan.summary(),
                        "chunk_quality_details": [
                            quality.to_dict()
                            for _, quality in sorted(unit_plan.chunk_quality.items())
                        ],
                        "timing": {"index_build_seconds": 0.0},
                    }
                )
                progress_callback(
                    {
                        "event": "done",
                        "generation_mode": "same_document_semantic_evidence",
                        "original_filename": original_filename,
                        "total_chunks": total_chunks,
                        "total_generation_units": 0,
                        "total_items": 0,
                        "unit_plan_summary": unit_plan.summary(),
                        "chunk_quality_details": [
                            quality.to_dict()
                            for _, quality in sorted(unit_plan.chunk_quality.items())
                        ],
                        "generation_unit_details": [],
                        "chunk_details": [],
                        "timing": _generation_timing_with_metadata(
                            {"document_total_seconds": time.perf_counter() - document_started_at},
                            index_seconds=0.0,
                            total_chunks=total_chunks,
                            chunks_completed=0,
                            total_generation_units=0,
                            generation_units_completed=0,
                            qa_generated=0,
                            qa_total_limit=runtime.qa_total_limit,
                            qa_total_limit_scope=runtime.qa_total_limit_scope,
                        ),
                    }
                )
            except Exception:
                pass
        return []

    index_started_at = time.perf_counter()
    evidence_index = QADocumentEvidenceIndex.build(document_chunks)
    index_seconds = time.perf_counter() - index_started_at
    if progress_callback:
        try:
            progress_callback(
                {
                    "event": "start",
                    "generation_mode": "same_document_semantic_evidence",
                    "original_filename": original_filename,
                    "total_chunks": total_chunks,
                    "total_generation_units": total_generation_units,
                    "qa_total_limit": runtime.qa_total_limit,
                    "qa_total_limit_scope": runtime.qa_total_limit_scope,
                    "unit_plan_summary": unit_plan.summary(),
                    "chunk_quality_details": [
                        quality.to_dict()
                        for _, quality in sorted(unit_plan.chunk_quality.items())
                    ],
                    "generation_unit_plan": [
                        unit.to_debug_dict() for unit in generation_units
                    ],
                    "timing": {
                        "index_build_seconds": index_seconds,
                    },
                }
            )
        except Exception:
            pass

    max_workers = max(1, min(int(runtime.chunk_max_concurrency), len(generation_units)))

    results: List[Dict[str, Any]] = []
    unit_items_by_index: Dict[int, List[Dict[str, Any]]] = {}
    unit_errors: List[str] = []
    generation_unit_debug_details: List[Dict[str, Any]] = []
    generation_wall_intervals: List[Dict[str, Any]] = []
    generation_accumulator: Dict[str, float] = {
        "candidate_question_seconds": 0.0,
        "retrieval_seconds": 0.0,
        "retrieval_embedding_seconds": 0.0,
        "retrieval_ranking_seconds": 0.0,
        "retrieval_unit_seconds": 0.0,
        "answer_generation_seconds": 0.0,
        "validation_and_bookkeeping_seconds": 0.0,
        "chunk_total_seconds": 0.0,
    }
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                run_one_step_unit_worker,
                unit=unit,
                evidence_index=evidence_index,
                runtime=runtime,
                client=client,
                debug_writer=debug_writer,
                item_normalizer_with_reason=_validate_and_normalize_item_with_reason,
                source_fact_detail_validator=_validate_source_fact_text_detail_mode,
                source_fact_grounding_validator=_validate_source_fact_grounding,
                source_override_handler=_maybe_override_source,
            ): unit
            for unit in generation_units
        }
        completed_units = 0
        for future in as_completed(future_map):
            completed_units += 1
            unit = future_map.get(future)
            payload: Dict[str, Any]
            try:
                payload = future.result()
            except Exception as exc:
                unit_errors.append(f"unit {getattr(unit, 'unit_index', '?')}: {exc}")
                payload = {
                    "unit_index": getattr(unit, "unit_index", None),
                    "unit_id": getattr(unit, "unit_id", None),
                    "unit_type": getattr(unit, "unit_type", None),
                    "qa_mode": getattr(unit, "qa_mode", None),
                    "chunk_index": getattr(unit, "anchor_chunk_index", None),
                    "anchor_chunk_index": getattr(unit, "anchor_chunk_index", None),
                    "source_chunk_indexes": getattr(unit, "source_chunk_indexes", []),
                    "attempt_used": 0,
                    "items": [],
                    "error": str(exc),
                    "timing": {},
                }

            items = payload.get("items") if isinstance(payload, dict) else None
            items_list = items if isinstance(items, list) else []
            unit_index = payload.get("unit_index") if isinstance(payload, dict) else None
            try:
                unit_index_int = int(unit_index)
            except Exception:
                unit_index_int = None
            if unit_index_int and unit_index_int > 0:
                unit_items_by_index[unit_index_int] = [
                    it for it in items_list if isinstance(it, dict)
                ]
                timing = payload.get("timing") if isinstance(payload, dict) else {}
                if not isinstance(timing, dict):
                    timing = {}
                for key in generation_accumulator:
                    try:
                        generation_accumulator[key] += float(timing.get(key) or 0.0)
                    except Exception:
                        pass
                raw_wall_intervals = payload.get("wall_intervals")
                if isinstance(raw_wall_intervals, list):
                    generation_wall_intervals.extend(
                        interval
                        for interval in raw_wall_intervals
                        if isinstance(interval, dict)
                    )
                unit_detail = {
                    "unit_index": unit_index_int,
                    "unit_id": payload.get("unit_id"),
                    "unit_type": payload.get("unit_type"),
                    "qa_mode": payload.get("qa_mode"),
                    "anchor_chunk_index": payload.get("anchor_chunk_index") or payload.get("chunk_index"),
                    "source_chunk_indexes": payload.get("source_chunk_indexes") or [],
                    "parent_index_path": payload.get("parent_index_path"),
                    "quality_child_coverage": payload.get("quality_child_coverage"),
                    "attempt_used": payload.get("attempt_used"),
                    "candidate_questions": payload.get("candidate_questions", 0),
                    "candidates_considered": payload.get("candidates_considered", 0),
                    "valid_items": len(items_list),
                    "dropped_reason_stats": payload.get("dropped_reason_stats")
                    or payload.get("dropped_answer_reasons")
                    or {},
                    "timing": timing,
                }
                if isinstance(payload.get("unit_debug"), dict):
                    unit_detail["unit_debug"] = payload.get("unit_debug")
                if payload.get("error"):
                    unit_detail["error"] = payload.get("error")
                generation_unit_debug_details.append(unit_detail)

            if progress_callback and isinstance(payload, dict):
                try:
                    progress_callback(
                        {
                            "event": "generation_unit_completed",
                            "generation_mode": "same_document_semantic_evidence",
                            "original_filename": original_filename,
                            "unit_index": payload.get("unit_index"),
                            "unit_id": payload.get("unit_id"),
                            "unit_type": payload.get("unit_type"),
                            "qa_mode": payload.get("qa_mode"),
                            "chunk_index": payload.get("chunk_index"),
                            "anchor_chunk_index": payload.get("anchor_chunk_index") or payload.get("chunk_index"),
                            "source_chunk_indexes": payload.get("source_chunk_indexes") or [],
                            "completed_units": completed_units,
                            "total_generation_units": total_generation_units,
                            "completed_chunks": completed_units,
                            "total_chunks": total_chunks,
                            "valid_items": len(items_list),
                            "attempt_used": payload.get("attempt_used"),
                            "error": payload.get("error"),
                            "skip_reason": payload.get("skip_reason"),
                            "dropped_answer_reasons": payload.get("dropped_answer_reasons"),
                            "dropped_reason_stats": payload.get("dropped_reason_stats"),
                            "candidate_questions": payload.get("candidate_questions"),
                            "candidates_considered": payload.get("candidates_considered"),
                            "timing": payload.get("timing"),
                            "unit_debug": payload.get("unit_debug"),
                        }
                    )
                except Exception:
                    pass

    # Non-strict mode: do not fail the whole file if some chunks produce 0 items.
    # Keep chunk_errors for debugging only.

    _ = original_filename  # reserved for future logging/telemetry
    for idx in range(1, total_generation_units + 1):
        items_list = unit_items_by_index.get(idx) or []
        for item in items_list:
            if isinstance(item, dict):
                results.append(item)
    if runtime.qa_total_limit is not None:
        results = results[: max(0, int(runtime.qa_total_limit))]
    if progress_callback:
        try:
            document_finished_at = time.perf_counter()
            document_total_seconds = max(0.0, document_finished_at - document_started_at)
            generation_cumulative_detail = _generation_timing_with_metadata(
                {
                    **generation_accumulator,
                    "document_total_seconds": document_total_seconds,
                },
                index_seconds=index_seconds,
                total_chunks=total_chunks,
                chunks_completed=len(generation_unit_debug_details),
                total_generation_units=total_generation_units,
                generation_units_completed=len(generation_unit_debug_details),
                qa_generated=len(results),
                qa_total_limit=runtime.qa_total_limit,
                qa_total_limit_scope=runtime.qa_total_limit_scope,
            )
            generation_wall_detail = _generation_timing_with_metadata(
                _attribute_generation_wall_detail(
                    generation_wall_intervals,
                    document_started_at=document_started_at,
                    document_finished_at=document_finished_at,
                ),
                index_seconds=index_seconds,
                total_chunks=total_chunks,
                chunks_completed=len(generation_unit_debug_details),
                total_generation_units=total_generation_units,
                generation_units_completed=len(generation_unit_debug_details),
                qa_generated=len(results),
                qa_total_limit=runtime.qa_total_limit,
                qa_total_limit_scope=runtime.qa_total_limit_scope,
            )
            generation_summary = {
                **generation_wall_detail,
                "generation_wall_detail": generation_wall_detail,
                "generation_cumulative_detail": generation_cumulative_detail,
                "unit_plan_summary": unit_plan.summary(),
                "chunk_quality_details": [
                    quality.to_dict()
                    for _, quality in sorted(unit_plan.chunk_quality.items())
                ],
            }
            progress_callback(
                {
                    "event": "done",
                    "generation_mode": "same_document_semantic_evidence",
                    "original_filename": original_filename,
                    "total_chunks": total_chunks,
                    "total_generation_units": total_generation_units,
                    "total_items": len(results),
                    "qa_total_limit": runtime.qa_total_limit,
                    "qa_total_limit_scope": runtime.qa_total_limit_scope,
                    "unit_plan_summary": unit_plan.summary(),
                    "chunk_quality_details": [
                        quality.to_dict()
                        for _, quality in sorted(unit_plan.chunk_quality.items())
                    ],
                    "timing": generation_summary,
                    "generation_unit_details": sorted(
                        generation_unit_debug_details,
                        key=lambda item: int(item.get("unit_index") or 0),
                    ),
                    "chunk_details": sorted(
                        generation_unit_debug_details,
                        key=lambda item: int(item.get("anchor_chunk_index") or 0),
                    ),
                }
            )
        except Exception:
            pass
    return results


__all__ = ["process_text_to_qa_one_step", "split_text"]
