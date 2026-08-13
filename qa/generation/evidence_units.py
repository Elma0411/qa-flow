"""Same-document evidence index backed by the fixed retrieval pipeline."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence

from qa.retrieval import EvidenceChunk, EvidenceRetrievalPipeline


DEFAULT_FINAL_EVIDENCE_K = 5
DEFAULT_EVIDENCE_TOKEN_BUDGET = 4000
DEFAULT_MAX_UNIT_CHARS = 12000


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def build_document_chunks(
    pre_split_chunks: Sequence[str],
    chunk_meta_list: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    meta_by_index: Dict[int, Dict[str, Any]] = {}
    for index, raw_meta in enumerate(chunk_meta_list or [], start=1):
        if not isinstance(raw_meta, dict):
            continue
        chunk_index = max(1, _safe_int(raw_meta.get("chunk_index"), index))
        meta_by_index[chunk_index] = dict(raw_meta)

    chunks: List[Dict[str, Any]] = []
    for index, raw_text in enumerate(pre_split_chunks or [], start=1):
        meta = dict(meta_by_index.get(index) or {})
        text = _safe_text(meta.get("text")) or _safe_text(raw_text)
        if not text:
            continue
        title_path = _safe_text(meta.get("title_path"))
        text_for_embedding = _safe_text(meta.get("text_for_embedding")) or text
        retrieval_text = text_for_embedding
        if title_path and title_path not in text_for_embedding:
            retrieval_text = f"标题路径：{title_path}\n{text_for_embedding}".strip()
        chunk_id = _safe_text(meta.get("chunk_id")) or hashlib.sha1(
            f"{index}|||{title_path}|||{text}".encode("utf-8")
        ).hexdigest()
        section_path = _safe_text(meta.get("section_path"))
        if not section_path:
            raise ValueError(
                "pre_split_chunk_meta must provide v2 section_path for every chunk"
            )
        section_parent_path = _safe_text(meta.get("section_parent_path"))
        source_asset_ids = meta.get("source_asset_ids") or []
        if not isinstance(source_asset_ids, list):
            source_asset_ids = []
        chunks.append(
            {
                **meta,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "section_chunk_index": max(1, _safe_int(meta.get("section_chunk_index"), 1)),
                "section_path": section_path,
                "section_parent_path": section_parent_path,
                "section_level": max(1, _safe_int(meta.get("section_level"), 1)),
                "section_is_leaf": bool(meta.get("section_is_leaf", True)),
                "title_path": title_path,
                "fragment_group_id": _safe_text(meta.get("fragment_group_id")) or chunk_id,
                "fragment_index": max(1, _safe_int(meta.get("fragment_index"), 1)),
                "fragment_count": max(1, _safe_int(meta.get("fragment_count"), 1)),
                "content_kind": _safe_text(meta.get("content_kind")) or "text",
                "source_asset_ids": source_asset_ids,
                "text": text,
                "text_for_embedding": text_for_embedding,
                "retrieval_text": retrieval_text,
            }
        )
    return chunks


class QADocumentEvidenceIndex:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks:
            raise ValueError("QA evidence index requires at least one content chunk")
        if len(chunks) != len(embeddings):
            raise ValueError("content chunk and embedding counts do not match")
        self.chunks = chunks
        self.embeddings = embeddings
        self._chunks_by_index = {
            int(chunk.get("chunk_index") or 0): chunk for chunk in chunks
        }
        self._chunks_by_id = {
            _safe_text(chunk.get("chunk_id")): chunk for chunk in chunks
        }
        typed_chunks = [EvidenceChunk.from_dict(chunk) for chunk in chunks]
        self.pipeline = EvidenceRetrievalPipeline(typed_chunks, embeddings)

    @classmethod
    def build(cls, chunks: List[Dict[str, Any]]) -> "QADocumentEvidenceIndex":
        from app.services.milvus import generate_embeddings

        retrieval_texts = [_safe_text(chunk.get("retrieval_text")) for chunk in chunks]
        if not all(retrieval_texts):
            raise ValueError("QA evidence index found an empty retrieval_text")
        embeddings = generate_embeddings(retrieval_texts)
        return cls(chunks=chunks, embeddings=embeddings)

    def get_chunk(self, chunk_index: int) -> Dict[str, Any]:
        chunk = self._chunks_by_index.get(int(chunk_index))
        if not chunk:
            raise ValueError(f"source chunk not found: {chunk_index}")
        return chunk

    def retrieve_many(
        self,
        questions: Sequence[str],
        *,
        source_chunk_ids: Sequence[str],
        final_evidence_k: int,
        evidence_token_budget: int,
        timing: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        from app.services.milvus import generate_embeddings

        clean_questions: List[str] = []
        seen: set[str] = set()
        for question in questions:
            value = _safe_text(question)
            if value and value not in seen:
                seen.add(value)
                clean_questions.append(value)
        if not clean_questions:
            return {}
        embedding_started = time.perf_counter()
        query_embeddings = generate_embeddings(clean_questions)
        if timing is not None:
            timing["embedding_seconds"] = timing.get("embedding_seconds", 0.0) + time.perf_counter() - embedding_started
        results: Dict[str, Dict[str, Any]] = {}
        for question, embedding in zip(clean_questions, query_embeddings):
            results[question] = self.pipeline.retrieve(
                question,
                embedding,
                final_evidence_k=final_evidence_k,
                evidence_token_budget=evidence_token_budget,
                source_chunk_ids=source_chunk_ids,
                timing=timing,
            )
        return results

    def build_generation_unit(
        self,
        *,
        source_chunk_index: int,
        source_unit: Optional[Dict[str, Any]],
        question: str,
        retrieval_result: Dict[str, Any],
        final_evidence_k: int,
        evidence_token_budget: int,
    ) -> Dict[str, Any]:
        source_chunk = self.get_chunk(source_chunk_index)
        source_unit_payload = dict(source_unit or {})
        source_indexes = [
            _safe_int(value)
            for value in source_unit_payload.get("source_chunk_indexes") or []
            if _safe_int(value) in self._chunks_by_index
        ] or [source_chunk_index]
        source_chunks = [self.get_chunk(index) for index in source_indexes]
        source_ids = [_safe_text(chunk.get("chunk_id")) for chunk in source_chunks]
        windows = list(retrieval_result.get("selected_windows") or [])

        evidence_chunks: List[Dict[str, Any]] = []
        seen_ids = set(source_ids)
        for window in windows:
            for chunk_id in window.chunk_ids:
                if chunk_id in seen_ids or chunk_id not in self._chunks_by_id:
                    continue
                seen_ids.add(chunk_id)
                evidence_chunks.append(self._chunks_by_id[chunk_id])

        sections: List[str] = ["【主来源材料】"]
        ref_map: Dict[str, Dict[str, Any]] = {}
        for position, chunk in enumerate(source_chunks, start=1):
            label = f"主材料-{position}"
            sections.append(f"{label}\n{_safe_text(chunk.get('text'))}")
            ref_map[label] = {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "role": "primary_source",
            }
        if evidence_chunks:
            sections.append("【检索证据】")
            for position, chunk in enumerate(evidence_chunks, start=1):
                label = f"检索证据-{position}"
                title_path = _safe_text(chunk.get("title_path"))
                rendered = _safe_text(chunk.get("text"))
                if title_path:
                    rendered = f"标题路径：{title_path}\n{rendered}"
                sections.append(f"{label}\n{rendered}")
                ref_map[label] = {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "role": "retrieved_evidence",
                }

        evidence_ids = [_safe_text(chunk.get("chunk_id")) for chunk in evidence_chunks]
        unit_id = hashlib.sha1(
            ("|||".join(source_ids + [_safe_text(question)] + evidence_ids)).encode("utf-8")
        ).hexdigest()
        trace = dict(retrieval_result.get("trace") or {})
        trace["query"] = _safe_text(question)
        trace["source_chunk_ids"] = source_ids
        trace["selected_evidence_chunk_ids"] = evidence_ids
        trace["final_evidence_k"] = int(final_evidence_k)
        trace["evidence_token_budget"] = int(evidence_token_budget)
        return {
            "qa_generation_unit_id": unit_id,
            "source_chunk": source_chunk,
            "source_unit": source_unit_payload,
            "source_chunks": source_chunks,
            "source_chunk_ids": source_ids,
            "source_unit_text": _safe_text(source_unit_payload.get("unit_text")) or "\n\n".join(_safe_text(chunk.get("text")) for chunk in source_chunks),
            "evidence_hits": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "title_path": chunk.get("title_path"),
                    "role": "retrieved_evidence",
                }
                for chunk in evidence_chunks
            ],
            "evidence_chunk_ids": evidence_ids,
            "qa_generation_unit_text": "\n\n".join(sections).strip(),
            "llm_evidence_ref_map": ref_map,
            "retrieval_trace": trace,
        }


__all__ = [
    "DEFAULT_EVIDENCE_TOKEN_BUDGET",
    "DEFAULT_FINAL_EVIDENCE_K",
    "DEFAULT_MAX_UNIT_CHARS",
    "QADocumentEvidenceIndex",
    "build_document_chunks",
]
