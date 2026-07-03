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
    run_one_step_chunk_worker,
)
from qa.generation import QADocumentEvidenceIndex, build_document_chunks
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
    qa_generated: int,
) -> Dict[str, Any]:
    return {
        **base,
        "index_build_seconds": index_seconds,
        "chunks_total": total_chunks,
        "chunks_completed": chunks_completed,
        "qa_generated": qa_generated,
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
    - Generate candidate questions from each source chunk.
    - Retrieve same-document evidence chunks by candidate-question semantics.
    - Generate final QA from the source chunk plus organized evidence context.
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
    index_started_at = time.perf_counter()
    evidence_index = QADocumentEvidenceIndex.build(document_chunks)
    index_seconds = time.perf_counter() - index_started_at
    chunks = [str(chunk.get("text") or "").strip() for chunk in document_chunks]

    total_chunks = len(chunks)
    if progress_callback:
        try:
            progress_callback(
                {
                    "event": "start",
                    "generation_mode": "same_document_semantic_evidence",
                    "original_filename": original_filename,
                    "total_chunks": total_chunks,
                    "timing": {
                        "index_build_seconds": index_seconds,
                    },
                }
            )
        except Exception:
            pass

    max_workers = max(1, min(int(runtime.chunk_max_concurrency), len(chunks)))

    results: List[Dict[str, Any]] = []
    chunk_items_by_index: Dict[int, List[Dict[str, Any]]] = {}
    chunk_errors: List[str] = []
    chunk_debug_details: List[Dict[str, Any]] = []
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
                run_one_step_chunk_worker,
                chunk_index=idx,
                chunk_text=chunk_text,
                source_chunk_meta=document_chunks[idx - 1],
                evidence_index=evidence_index,
                runtime=runtime,
                client=client,
                debug_writer=debug_writer,
                item_normalizer_with_reason=_validate_and_normalize_item_with_reason,
                source_fact_detail_validator=_validate_source_fact_text_detail_mode,
                source_fact_grounding_validator=_validate_source_fact_grounding,
                source_override_handler=_maybe_override_source,
            ): idx
            for idx, chunk_text in enumerate(chunks, start=1)
        }
        completed_chunks = 0
        for future in as_completed(future_map):
            completed_chunks += 1
            payload: Dict[str, Any]
            try:
                payload = future.result()
            except Exception as exc:
                chunk_errors.append(f"chunk {future_map.get(future)}: {exc}")
                payload = {
                    "chunk_index": future_map.get(future),
                    "attempt_used": 0,
                    "items": [],
                    "error": str(exc),
                    "timing": {},
                }

            items = payload.get("items") if isinstance(payload, dict) else None
            items_list = items if isinstance(items, list) else []
            chunk_index = payload.get("chunk_index") if isinstance(payload, dict) else None
            try:
                chunk_index_int = int(chunk_index)
            except Exception:
                chunk_index_int = None
            if chunk_index_int and chunk_index_int > 0:
                chunk_items_by_index[chunk_index_int] = [
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
                chunk_detail = {
                    "chunk_index": chunk_index_int,
                    "attempt_used": payload.get("attempt_used"),
                    "candidate_questions": payload.get("candidate_questions", 0),
                    "candidates_considered": payload.get("candidates_considered", 0),
                    "valid_items": len(items_list),
                    "dropped_reason_stats": payload.get("dropped_reason_stats")
                    or payload.get("dropped_answer_reasons")
                    or {},
                    "timing": timing,
                }
                if payload.get("error"):
                    chunk_detail["error"] = payload.get("error")
                chunk_debug_details.append(chunk_detail)

            if progress_callback and isinstance(payload, dict):
                try:
                    progress_callback(
                        {
                            "event": "chunk_completed",
                            "generation_mode": "same_document_semantic_evidence",
                            "original_filename": original_filename,
                            "chunk_index": payload.get("chunk_index"),
                            "completed_chunks": completed_chunks,
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
                        }
                    )
                except Exception:
                    pass

    # Non-strict mode: do not fail the whole file if some chunks produce 0 items.
    # Keep chunk_errors for debugging only.

    _ = original_filename  # reserved for future logging/telemetry
    for idx in range(1, total_chunks + 1):
        items_list = chunk_items_by_index.get(idx) or []
        for item in items_list:
            if isinstance(item, dict):
                results.append(item)
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
                chunks_completed=len(chunk_debug_details),
                qa_generated=len(results),
            )
            generation_wall_detail = _generation_timing_with_metadata(
                _attribute_generation_wall_detail(
                    generation_wall_intervals,
                    document_started_at=document_started_at,
                    document_finished_at=document_finished_at,
                ),
                index_seconds=index_seconds,
                total_chunks=total_chunks,
                chunks_completed=len(chunk_debug_details),
                qa_generated=len(results),
            )
            generation_summary = {
                **generation_wall_detail,
                "generation_wall_detail": generation_wall_detail,
                "generation_cumulative_detail": generation_cumulative_detail,
            }
            progress_callback(
                {
                    "event": "done",
                    "generation_mode": "same_document_semantic_evidence",
                    "original_filename": original_filename,
                    "total_chunks": total_chunks,
                    "total_items": len(results),
                    "timing": generation_summary,
                    "chunk_details": sorted(
                        chunk_debug_details,
                        key=lambda item: int(item.get("chunk_index") or 0),
                    ),
                }
            )
        except Exception:
            pass
    return results


__all__ = ["process_text_to_qa_one_step", "split_text"]
