# 文件作用：构建同文档证据索引和生成单元。
# 关联说明：被 qa_generation_flow 和 text_to_qa_pipeline 调用，负责证据索引与生成上下文。

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SEMANTIC_TOP_K = 3
DEFAULT_MAX_UNIT_CHARS = 6000
DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_HYBRID_WEIGHT_DENSE = 0.68
DEFAULT_HYBRID_WEIGHT_LEXICAL = 0.24
DEFAULT_STRUCTURE_WEIGHT = 0.08
DEFAULT_RERANK_TOP_N = 12


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


def _dense_to_unit_interval(score: float) -> float:
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def _lexical_tokens(text: str) -> List[str]:
    raw = _safe_text(text).lower()
    if not raw:
        return []
    tokens: List[str] = []
    for segment in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", raw):
        if re.fullmatch(r"[a-z0-9_]+", segment):
            if len(segment) >= 2:
                tokens.append(segment)
            continue
        if len(segment) == 1:
            tokens.append(segment)
            continue
        tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        if len(segment) >= 3:
            tokens.extend(segment[index : index + 3] for index in range(len(segment) - 2))
    return tokens


def _normalize_terms(raw_terms: Any) -> List[str]:
    if raw_terms is None:
        return []
    if isinstance(raw_terms, list):
        values = raw_terms
    else:
        values = re.split(r"[,，;；\n]+", str(raw_terms))
    terms: List[str] = []
    seen: set[str] = set()
    for value in values:
        term = _safe_text(value)
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _lexical_query_tokens(query: str, must_have_terms: Sequence[str]) -> List[str]:
    tokens = _lexical_tokens(query)
    for term in must_have_terms:
        term_tokens = _lexical_tokens(term)
        tokens.extend(term_tokens)
        tokens.extend(term_tokens)
    return tokens


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
    dense_score: float = 0.0
    lexical_score: float = 0.0
    structure_score: float = 0.0
    fused_score: float = 0.0
    retrieval_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "title_path": self.title_path,
            "parent_index_path": self.parent_index_path,
            "role": self.role,
            "dense_score": self.dense_score,
            "lexical_score": self.lexical_score,
            "structure_score": self.structure_score,
            "fused_score": self.fused_score,
            "retrieval_rank": self.retrieval_rank,
        }


class QADocumentEvidenceIndex:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks:
            raise ValueError("qa_generation_units requires at least one chunk")
        if len(chunks) != len(embeddings):
            raise ValueError("chunk embedding count does not match chunk count")
        self.chunks = chunks
        self.embeddings = embeddings
        self._chunk_lexical_tokens = [
            _lexical_tokens(
                "\n".join(
                    [
                        _safe_text(chunk.get("title_path")),
                        _safe_text(chunk.get("path_summary")),
                        _safe_text(chunk.get("retrieval_text")),
                    ]
                )
            )
            for chunk in chunks
        ]
        token_document_frequency: Counter[str] = Counter()
        for tokens in self._chunk_lexical_tokens:
            token_document_frequency.update(set(tokens))
        self._token_idf = {
            token: math.log((len(chunks) + 1) / (count + 0.5)) + 1.0
            for token, count in token_document_frequency.items()
        }
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
        must_have_terms: Optional[Sequence[str]] = None,
        answer_scope: str = "source_primary",
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        hybrid_weight_dense: float = DEFAULT_HYBRID_WEIGHT_DENSE,
        hybrid_weight_lexical: float = DEFAULT_HYBRID_WEIGHT_LEXICAL,
        structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> Tuple[List[EvidenceHit], List[Dict[str, Any]]]:
        from app.services.milvus import generate_embeddings

        clean_query = _safe_text(query)
        if not clean_query:
            return [], []
        query_embedding = generate_embeddings([clean_query])[0]
        return self._rank_with_query_embedding(
            clean_query,
            query_embedding,
            source_chunk_index=source_chunk_index,
            top_k=top_k,
            must_have_terms=must_have_terms,
            answer_scope=answer_scope,
            retrieval_mode=retrieval_mode,
            hybrid_weight_dense=hybrid_weight_dense,
            hybrid_weight_lexical=hybrid_weight_lexical,
            structure_weight=structure_weight,
            rerank_top_n=rerank_top_n,
        )

    def retrieve_many(
        self,
        queries: Sequence[Any],
        *,
        source_chunk_index: int,
        top_k: int = DEFAULT_SEMANTIC_TOP_K,
        timing: Optional[Dict[str, float]] = None,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        hybrid_weight_dense: float = DEFAULT_HYBRID_WEIGHT_DENSE,
        hybrid_weight_lexical: float = DEFAULT_HYBRID_WEIGHT_LEXICAL,
        structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> Dict[str, Tuple[List[EvidenceHit], List[Dict[str, Any]]]]:
        from app.services.milvus import generate_embeddings

        payloads: List[Dict[str, Any]] = []
        for raw_query in queries:
            if isinstance(raw_query, dict):
                key = _safe_text(raw_query.get("key") or raw_query.get("question") or raw_query.get("query"))
                query = _safe_text(raw_query.get("query") or raw_query.get("question"))
                must_have_terms = _normalize_terms(raw_query.get("must_have_terms"))
                answer_scope = _safe_text(raw_query.get("answer_scope")) or "source_primary"
            else:
                key = _safe_text(raw_query)
                query = key
                must_have_terms = []
                answer_scope = "source_primary"
            if not key or not query:
                continue
            payloads.append(
                {
                    "key": key,
                    "query": query,
                    "must_have_terms": must_have_terms,
                    "answer_scope": answer_scope,
                }
            )
        if not payloads:
            return {}
        embedding_start = time.perf_counter()
        query_embeddings = generate_embeddings([payload["query"] for payload in payloads])
        embedding_seconds = time.perf_counter() - embedding_start
        if timing is not None:
            timing["embedding_seconds"] = timing.get("embedding_seconds", 0.0) + embedding_seconds
        results: Dict[str, Tuple[List[EvidenceHit], List[Dict[str, Any]]]] = {}
        rank_start = time.perf_counter()
        for payload, query_embedding in zip(payloads, query_embeddings):
            results[payload["key"]] = self._rank_with_query_embedding(
                payload["query"],
                query_embedding,
                source_chunk_index=source_chunk_index,
                top_k=top_k,
                must_have_terms=payload.get("must_have_terms"),
                answer_scope=payload.get("answer_scope") or "source_primary",
                retrieval_mode=retrieval_mode,
                hybrid_weight_dense=hybrid_weight_dense,
                hybrid_weight_lexical=hybrid_weight_lexical,
                structure_weight=structure_weight,
                rerank_top_n=rerank_top_n,
            )
        rank_seconds = time.perf_counter() - rank_start
        if timing is not None:
            timing["ranking_seconds"] = timing.get("ranking_seconds", 0.0) + rank_seconds
        return results

    def _rank_with_query_embedding(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        source_chunk_index: int,
        top_k: int,
        must_have_terms: Optional[Sequence[str]] = None,
        answer_scope: str = "source_primary",
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        hybrid_weight_dense: float = DEFAULT_HYBRID_WEIGHT_DENSE,
        hybrid_weight_lexical: float = DEFAULT_HYBRID_WEIGHT_LEXICAL,
        structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> Tuple[List[EvidenceHit], List[Dict[str, Any]]]:
        source_index = int(source_chunk_index)
        source_chunk = self._chunks_by_index.get(source_index) or {}
        source_parent = _safe_text(source_chunk.get("parent_index_path"))
        source_title_parts = set(_split_title_path(_safe_text(source_chunk.get("title_path"))))
        mode = _safe_text(retrieval_mode).lower() or DEFAULT_RETRIEVAL_MODE
        if mode not in {"semantic", "hybrid"}:
            mode = DEFAULT_RETRIEVAL_MODE
        dense_weight = max(0.0, _safe_float(hybrid_weight_dense, DEFAULT_HYBRID_WEIGHT_DENSE))
        lexical_weight = max(0.0, _safe_float(hybrid_weight_lexical, DEFAULT_HYBRID_WEIGHT_LEXICAL))
        if mode == "semantic":
            lexical_weight = 0.0
            structure_weight = 0.0
        if dense_weight <= 0 and lexical_weight <= 0:
            dense_weight = 1.0
        weight_total = dense_weight + lexical_weight
        dense_weight = dense_weight / weight_total
        lexical_weight = lexical_weight / weight_total
        structure_weight = max(0.0, min(0.5, _safe_float(structure_weight, DEFAULT_STRUCTURE_WEIGHT)))
        rerank_top_n = max(int(top_k), int(rerank_top_n or DEFAULT_RERANK_TOP_N), 1)
        query_terms = _normalize_terms(must_have_terms)
        query_tokens = _lexical_query_tokens(query, query_terms)
        query_token_weight = sum(self._token_idf.get(token, 1.0) for token in set(query_tokens)) or 1.0

        scored: List[Dict[str, Any]] = []
        for position, (chunk, embedding) in enumerate(zip(self.chunks, self.embeddings)):
            dense_score = _dot_score(query_embedding, embedding)
            chunk_tokens = self._chunk_lexical_tokens[position] if position < len(self._chunk_lexical_tokens) else []
            chunk_token_counts = Counter(chunk_tokens)
            lexical_overlap = 0.0
            if query_tokens and chunk_token_counts:
                lexical_overlap = sum(
                    self._token_idf.get(token, 1.0)
                    for token in set(query_tokens)
                    if chunk_token_counts.get(token, 0) > 0
                ) / query_token_weight
            must_term_hits = 0
            for term in query_terms:
                term_text = _safe_text(term).lower()
                target_text = (
                    _safe_text(chunk.get("title_path"))
                    + "\n"
                    + _safe_text(chunk.get("retrieval_text"))
                ).lower()
                if term_text and term_text in target_text:
                    must_term_hits += 1
            must_term_coverage = (must_term_hits / len(query_terms)) if query_terms else 0.0
            lexical_score = max(0.0, min(1.0, 0.75 * lexical_overlap + 0.25 * must_term_coverage))

            chunk_index = int(chunk.get("chunk_index") or 0)
            chunk_parent = _safe_text(chunk.get("parent_index_path"))
            chunk_title_parts = set(_split_title_path(_safe_text(chunk.get("title_path"))))
            same_parent = bool(source_parent and chunk_parent and source_parent == chunk_parent)
            adjacent = bool(chunk_index > 0 and abs(chunk_index - source_index) == 1)
            title_overlap = (
                len(source_title_parts & chunk_title_parts) / max(1, len(source_title_parts | chunk_title_parts))
                if source_title_parts or chunk_title_parts
                else 0.0
            )
            scope = _safe_text(answer_scope).lower() or "source_primary"
            scope_same_section_boost = 0.15 if scope in {"same_section", "cross_chunk"} and same_parent else 0.0
            scope_cross_boost = 0.08 if scope == "cross_chunk" and lexical_score > 0 else 0.0
            structure_score = max(
                0.0,
                min(
                    1.0,
                    (0.55 if same_parent else 0.0)
                    + (0.25 if adjacent else 0.0)
                    + (0.20 * title_overlap)
                    + scope_same_section_boost
                    + scope_cross_boost,
                ),
            )
            fused_score = (
                dense_weight * _dense_to_unit_interval(dense_score)
                + lexical_weight * lexical_score
                + structure_weight * structure_score
            )
            scored.append(
                {
                    "chunk": chunk,
                    "dense_score": dense_score,
                    "lexical_score": lexical_score,
                    "structure_score": structure_score,
                    "fused_score": fused_score,
                    "same_parent": same_parent,
                    "adjacent": adjacent,
                    "title_overlap": title_overlap,
                }
            )

        scored.sort(key=lambda item: item["dense_score"], reverse=True)
        dense_rank = {
            int(item["chunk"].get("chunk_index") or 0): rank
            for rank, item in enumerate(scored, start=1)
        }
        lexical_sorted = sorted(scored, key=lambda item: item["lexical_score"], reverse=True)
        lexical_rank = {
            int(item["chunk"].get("chunk_index") or 0): rank
            for rank, item in enumerate(lexical_sorted, start=1)
        }
        pool_size = max(rerank_top_n, int(top_k) + 3)
        pool_by_index: Dict[int, Dict[str, Any]] = {}
        for item in scored[:pool_size]:
            pool_by_index[int(item["chunk"].get("chunk_index") or 0)] = item
        for item in lexical_sorted[:pool_size]:
            pool_by_index[int(item["chunk"].get("chunk_index") or 0)] = item
        final_pool = sorted(
            pool_by_index.values(),
            key=lambda item: item["fused_score"],
            reverse=True,
        )
        if mode == "semantic":
            final_pool = scored
        final_scores = [float(item["fused_score"]) for item in final_pool]
        score_gap_top1_top2 = (
            final_scores[0] - final_scores[1] if len(final_scores) >= 2 else None
        )
        score_gap_top1_topk = (
            final_scores[0] - final_scores[min(len(final_scores), max(1, int(top_k))) - 1]
            if final_scores and int(top_k) > 1
            else None
        )
        raw_trace = [
            {
                "chunk_id": item["chunk"].get("chunk_id"),
                "chunk_index": item["chunk"].get("chunk_index"),
                "title_path": item["chunk"].get("title_path"),
                "parent_index_path": item["chunk"].get("parent_index_path"),
                "score": item["fused_score"],
                "dense_score": item["dense_score"],
                "lexical_score": item["lexical_score"],
                "structure_score": item["structure_score"],
                "dense_rank": dense_rank.get(int(item["chunk"].get("chunk_index") or 0)),
                "lexical_rank": lexical_rank.get(int(item["chunk"].get("chunk_index") or 0)),
                "final_rank": rank,
                "same_parent": item["same_parent"],
                "adjacent": item["adjacent"],
                "title_overlap": item["title_overlap"],
                "is_source_chunk": int(item["chunk"].get("chunk_index") or 0) == source_index,
                "score_gap_top1_top2": score_gap_top1_top2,
                "score_gap_top1_topk": score_gap_top1_topk,
            }
            for rank, item in enumerate(final_pool[: max(10, int(top_k) + 3)], start=1)
        ]

        hits: List[EvidenceHit] = []
        for rank, item in enumerate(final_pool, start=1):
            chunk = item["chunk"]
            chunk_index = int(chunk.get("chunk_index") or 0)
            if chunk_index <= 0 or chunk_index == source_index:
                continue
            hits.append(
                EvidenceHit(
                    chunk_id=_safe_text(chunk.get("chunk_id")),
                    chunk_index=chunk_index,
                    score=float(item["fused_score"]),
                    title_path=_safe_text(chunk.get("title_path")),
                    parent_index_path=_safe_text(chunk.get("parent_index_path")),
                    role="semantic_hit",
                    dense_score=float(item["dense_score"]),
                    lexical_score=float(item["lexical_score"]),
                    structure_score=float(item["structure_score"]),
                    fused_score=float(item["fused_score"]),
                    retrieval_rank=rank,
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
        retrieval_query: str = "",
        must_have_terms: Optional[Sequence[str]] = None,
        answer_scope: str = "source_primary",
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
        retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
        hybrid_weight_dense: float = DEFAULT_HYBRID_WEIGHT_DENSE,
        hybrid_weight_lexical: float = DEFAULT_HYBRID_WEIGHT_LEXICAL,
        structure_weight: float = DEFAULT_STRUCTURE_WEIGHT,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        semantic_hits: Optional[List[EvidenceHit]] = None,
        raw_semantic_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        source_chunk = self.get_chunk(source_chunk_index)
        clean_retrieval_query = _safe_text(retrieval_query) or _safe_text(question)
        normalized_terms = _normalize_terms(must_have_terms)
        normalized_scope = _safe_text(answer_scope).lower() or "source_primary"
        if semantic_hits is None or raw_semantic_trace is None:
            hits, raw_trace = self.retrieve(
                clean_retrieval_query,
                source_chunk_index=source_chunk_index,
                top_k=semantic_top_k,
                must_have_terms=normalized_terms,
                answer_scope=normalized_scope,
                retrieval_mode=retrieval_mode,
                hybrid_weight_dense=hybrid_weight_dense,
                hybrid_weight_lexical=hybrid_weight_lexical,
                structure_weight=structure_weight,
                rerank_top_n=rerank_top_n,
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
                    dense_score=hit.dense_score,
                    lexical_score=hit.lexical_score,
                    structure_score=hit.structure_score,
                    fused_score=hit.fused_score,
                    retrieval_rank=hit.retrieval_rank,
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
                "retrieval_query": clean_retrieval_query,
                "must_have_terms": normalized_terms,
                "answer_scope": normalized_scope,
                "retrieval_mode": _safe_text(retrieval_mode) or DEFAULT_RETRIEVAL_MODE,
                "hybrid_weight_dense": _safe_float(hybrid_weight_dense, DEFAULT_HYBRID_WEIGHT_DENSE),
                "hybrid_weight_lexical": _safe_float(hybrid_weight_lexical, DEFAULT_HYBRID_WEIGHT_LEXICAL),
                "structure_weight": _safe_float(structure_weight, DEFAULT_STRUCTURE_WEIGHT),
                "rerank_top_n": int(rerank_top_n or DEFAULT_RERANK_TOP_N),
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
    "DEFAULT_HYBRID_WEIGHT_DENSE",
    "DEFAULT_HYBRID_WEIGHT_LEXICAL",
    "DEFAULT_MAX_UNIT_CHARS",
    "DEFAULT_RETRIEVAL_MODE",
    "DEFAULT_RERANK_TOP_N",
    "DEFAULT_SEMANTIC_TOP_K",
    "DEFAULT_STRUCTURE_WEIGHT",
    "QADocumentEvidenceIndex",
    "build_document_chunks",
]
