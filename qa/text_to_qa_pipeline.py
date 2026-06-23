# 文件作用：编排从输入文本到问答条目的完整生成流程。
# 关联说明：调用 chunking、generation、grounding、validation，是 QA 目录的完整流程编排层。

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

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



def process_text_to_qa_one_step(
    client: OpenAI,
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
    evidence_index = QADocumentEvidenceIndex.build(document_chunks)
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
                }
            )
        except Exception:
            pass

    max_workers = max(1, min(int(runtime.chunk_max_concurrency), len(chunks)))

    results: List[Dict[str, Any]] = []
    chunk_items_by_index: Dict[int, List[Dict[str, Any]]] = {}
    chunk_errors: List[str] = []
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
            progress_callback(
                {
                    "event": "done",
                    "generation_mode": "same_document_semantic_evidence",
                    "original_filename": original_filename,
                    "total_chunks": total_chunks,
                    "total_items": len(results),
                }
            )
        except Exception:
            pass
    return results


__all__ = ["process_text_to_qa_one_step", "split_text"]
