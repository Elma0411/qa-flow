# 文件作用：构建同文档证据索引和生成单元。
# 关联说明：被 qa_generation_flow 和 text_to_qa_pipeline 调用，负责证据索引与生成上下文。

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SEMANTIC_TOP_K = 3
DEFAULT_MAX_UNIT_CHARS = 6000


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _dot_score(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    limit = min(len(left), len(right))
    return float(sum(float(left[index]) * float(right[index]) for index in range(limit)))


def _split_title_path(title_path: str) -> List[str]:
    normalized = _safe_text(title_path).replace("＞", ">")
    return [part.strip() for part in normalized.split(">") if part.strip()]


def _parent_title_path(title_path: str) -> str:
    parts = _split_title_path(title_path)
    if len(parts) <= 1:
        return ""
    return " > ".join(parts[:-1])


def _format_chunk_for_unit(chunk: Dict[str, Any]) -> str:
    title_path = _safe_text(chunk.get("title_path"))
    text = _safe_text(chunk.get("text")) or _safe_text(chunk.get("text_for_embedding"))
    if title_path:
        return f"标题路径：{title_path}\n内容：{text}".strip()
    return f"内容：{text}".strip()


def build_document_chunks(
    pre_split_chunks: Sequence[str],
    chunk_meta_list: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    meta_by_index: Dict[int, Dict[str, Any]] = {}
    for index, raw_meta in enumerate(chunk_meta_list or [], start=1):
        if not isinstance(raw_meta, dict):
            continue
        chunk_index = _safe_int(raw_meta.get("chunk_index"), index)
        if chunk_index <= 0:
            chunk_index = index
        meta_by_index[chunk_index] = dict(raw_meta)

    chunks: List[Dict[str, Any]] = []
    for index, raw_text in enumerate(pre_split_chunks or [], start=1):
        meta = dict(meta_by_index.get(index) or {})
        text = _safe_text(meta.get("text")) or _safe_text(raw_text)
        text_for_embedding = _safe_text(meta.get("text_for_embedding")) or text
        title_path = _safe_text(meta.get("title_path"))
        if title_path and title_path not in text_for_embedding:
            retrieval_text = f"标题路径：{title_path}\n{text_for_embedding}".strip()
        else:
            retrieval_text = text_for_embedding

        chunk_id = _safe_text(meta.get("chunk_id"))
        if not chunk_id:
            chunk_id = hashlib.sha1(f"{index}|||{title_path}|||{text}".encode("utf-8")).hexdigest()

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": index,
                "index_path": _safe_text(meta.get("index_path")) or str(index),
                "title_path": title_path,
                "parent_index_path": _safe_text(meta.get("parent_index_path")),
                "root_index_path": _safe_text(meta.get("root_index_path")),
                "level": _safe_int(meta.get("level"), 1),
                "text": text,
                "text_for_embedding": text_for_embedding,
                "retrieval_text": retrieval_text,
                "path_summary": _safe_text(meta.get("path_summary")),
                "split_type": _safe_text(meta.get("split_type")),
            }
        )
    return [chunk for chunk in chunks if _safe_text(chunk.get("text"))]


@dataclass(frozen=True)
class EvidenceHit:
    chunk_id: str
    chunk_index: int
    score: float
    title_path: str
    parent_index_path: str
    role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "title_path": self.title_path,
            "parent_index_path": self.parent_index_path,
            "role": self.role,
        }


class QADocumentEvidenceIndex:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks:
            raise ValueError("qa_generation_units requires at least one chunk")
        if len(chunks) != len(embeddings):
            raise ValueError("chunk embedding count does not match chunk count")
        self.chunks = chunks
        self.embeddings = embeddings
        self._chunks_by_index = {
            int(chunk.get("chunk_index") or 0): chunk
            for chunk in chunks
            if int(chunk.get("chunk_index") or 0) > 0
        }

    @classmethod
    def build(cls, chunks: List[Dict[str, Any]]) -> "QADocumentEvidenceIndex":
        from app.services.milvus import generate_embeddings

        retrieval_texts = [_safe_text(chunk.get("retrieval_text")) for chunk in chunks]
        if not all(retrieval_texts):
            raise ValueError("qa_generation_units found empty retrieval_text")
        embeddings = generate_embeddings(retrieval_texts)
        return cls(chunks=chunks, embeddings=embeddings)

    def get_chunk(self, chunk_index: int) -> Dict[str, Any]:
        chunk = self._chunks_by_index.get(int(chunk_index))
        if not chunk:
            raise ValueError(f"source chunk not found: {chunk_index}")
        return chunk

    def retrieve(
        self,
        query: str,
        *,
        source_chunk_index: int,
        top_k: int = DEFAULT_SEMANTIC_TOP_K,
    ) -> Tuple[List[EvidenceHit], List[Dict[str, Any]]]:
        from app.services.milvus import generate_embeddings

        clean_query = _safe_text(query)
        if not clean_query:
            return [], []
        query_embedding = generate_embeddings([clean_query])[0]
        return self._rank_with_query_embedding(
            query_embedding,
            source_chunk_index=source_chunk_index,
            top_k=top_k,
        )

    def retrieve_many(
        self,
        queries: Sequence[str],
        *,
        source_chunk_index: int,
        top_k: int = DEFAULT_SEMANTIC_TOP_K,
    ) -> Dict[str, Tuple[List[EvidenceHit], List[Dict[str, Any]]]]:
        from app.services.milvus import generate_embeddings

        clean_queries = [_safe_text(query) for query in queries if _safe_text(query)]
        if not clean_queries:
            return {}
        query_embeddings = generate_embeddings(clean_queries)
        results: Dict[str, Tuple[List[EvidenceHit], List[Dict[str, Any]]]] = {}
        for query, query_embedding in zip(clean_queries, query_embeddings):
            results[query] = self._rank_with_query_embedding(
                query_embedding,
                source_chunk_index=source_chunk_index,
                top_k=top_k,
            )
        return results

    def _rank_with_query_embedding(
        self,
        query_embedding: Sequence[float],
        *,
        source_chunk_index: int,
        top_k: int,
    ) -> Tuple[List[EvidenceHit], List[Dict[str, Any]]]:
        scored: List[Tuple[Dict[str, Any], float]] = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            scored.append((chunk, _dot_score(query_embedding, embedding)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        source_index = int(source_chunk_index)
        raw_trace = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "title_path": chunk.get("title_path"),
                "parent_index_path": chunk.get("parent_index_path"),
                "score": score,
                "is_source_chunk": int(chunk.get("chunk_index") or 0) == source_index,
            }
            for chunk, score in scored[: max(10, int(top_k) + 3)]
        ]

        hits: List[EvidenceHit] = []
        for chunk, score in scored:
            chunk_index = int(chunk.get("chunk_index") or 0)
            if chunk_index <= 0 or chunk_index == source_index:
                continue
            hits.append(
                EvidenceHit(
                    chunk_id=_safe_text(chunk.get("chunk_id")),
                    chunk_index=chunk_index,
                    score=score,
                    title_path=_safe_text(chunk.get("title_path")),
                    parent_index_path=_safe_text(chunk.get("parent_index_path")),
                    role="semantic_hit",
                )
            )
            if len(hits) >= max(0, int(top_k)):
                break
        return hits, raw_trace

    def build_generation_unit(
        self,
        *,
        source_chunk_index: int,
        question: str,
        source_anchor_text: str,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
        semantic_hits: Optional[List[EvidenceHit]] = None,
        raw_semantic_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_chunk = self.get_chunk(source_chunk_index)
        if semantic_hits is None or raw_semantic_trace is None:
            hits, raw_trace = self.retrieve(
                question,
                source_chunk_index=source_chunk_index,
                top_k=semantic_top_k,
            )
        else:
            hits, raw_trace = semantic_hits, raw_semantic_trace
        source_parent = _safe_text(source_chunk.get("parent_index_path"))

        selected_hits: List[EvidenceHit] = []
        remaining_budget = max(1000, int(max_unit_chars)) - len(_format_chunk_for_unit(source_chunk))
        for hit in hits:
            chunk = self.get_chunk(hit.chunk_index)
            hit_text = _format_chunk_for_unit(chunk)
            if remaining_budget - len(hit_text) < 0:
                continue
            role = "same_section_context" if hit.parent_index_path and hit.parent_index_path == source_parent else "related_context"
            selected_hits.append(
                EvidenceHit(
                    chunk_id=hit.chunk_id,
                    chunk_index=hit.chunk_index,
                    score=hit.score,
                    title_path=hit.title_path,
                    parent_index_path=hit.parent_index_path,
                    role=role,
                )
            )
            remaining_budget -= len(hit_text)

        unit_text = self._render_unit_text(
            source_chunk=source_chunk,
            hits=selected_hits,
        )
        evidence_chunk_ids = [hit.chunk_id for hit in selected_hits if hit.chunk_id]
        unit_id = hashlib.sha1(
            (
                _safe_text(source_chunk.get("chunk_id"))
                + "|||"
                + _safe_text(question)
                + "|||"
                + "|||".join(evidence_chunk_ids)
            ).encode("utf-8")
        ).hexdigest()
        return {
            "qa_generation_unit_id": unit_id,
            "source_chunk": source_chunk,
            "source_anchor_text": _safe_text(source_anchor_text),
            "evidence_hits": [hit.to_dict() for hit in selected_hits],
            "evidence_chunk_ids": evidence_chunk_ids,
            "qa_generation_unit_text": unit_text,
            "retrieval_trace": {
                "query": _safe_text(question),
                "semantic_top_k": int(semantic_top_k),
                "max_unit_chars": int(max_unit_chars),
                "raw_semantic_hits": raw_trace,
                "selected_evidence": [hit.to_dict() for hit in selected_hits],
            },
        }

    def _render_unit_text(
        self,
        *,
        source_chunk: Dict[str, Any],
        hits: Sequence[EvidenceHit],
    ) -> str:
        sections: List[str] = [
            "【主来源块】\n"
            + f"chunk_id：{_safe_text(source_chunk.get('chunk_id'))}\n"
            + _format_chunk_for_unit(source_chunk)
        ]

        same_section_hits = [hit for hit in hits if hit.role == "same_section_context"]
        related_hits = [hit for hit in hits if hit.role != "same_section_context"]

        if same_section_hits:
            parent_title = _parent_title_path(_safe_text(source_chunk.get("title_path"))) or "同章节"
            parts = [f"【同章节上下文：{parent_title}】"]
            for hit in same_section_hits:
                chunk = self.get_chunk(hit.chunk_index)
                parts.append(f"chunk_id：{hit.chunk_id}\n{_format_chunk_for_unit(chunk)}")
            sections.append("\n\n".join(parts))

        if related_hits:
            parts = ["【相关补充】"]
            for hit in related_hits:
                chunk = self.get_chunk(hit.chunk_index)
                parent_title = _parent_title_path(_safe_text(chunk.get("title_path")))
                prefix = f"父级章节：{parent_title}\n" if parent_title else ""
                parts.append(
                    f"chunk_id：{hit.chunk_id}\n"
                    + prefix
                    + _format_chunk_for_unit(chunk)
                )
            sections.append("\n\n".join(parts))

        return "\n\n".join(section for section in sections if section.strip()).strip()


__all__ = [
    "DEFAULT_MAX_UNIT_CHARS",
    "DEFAULT_SEMANTIC_TOP_K",
    "QADocumentEvidenceIndex",
    "build_document_chunks",
]
