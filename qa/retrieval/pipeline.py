"""Fixed dense + BM25 + RRF + BGE + structure-window retrieval."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .reranker import RerankerService, get_reranker_service
from .types import EvidenceChunk, EvidenceWindow, RankedChunk
from .windows import EvidenceWindowBuilder


DEFAULT_DENSE_RECALL_K = 24
DEFAULT_BM25_RECALL_K = 24
DEFAULT_RRF_CANDIDATE_K = 32
DEFAULT_ATOMIC_RERANK_K = 12
CALIBRATED_RELEVANCE_MIN_LOGIT = -1.0
ATOMIC_RELEVANCE_MAX_LOGIT_DROP = 8.0
WINDOW_RELEVANCE_MAX_LOGIT_DROP = 4.0
ATOMIC_PRIMARY_SOURCE_MAX_LOGIT_DROP = 1.0
WINDOW_PRIMARY_SOURCE_MAX_LOGIT_DROP = 2.0


def normalize_retrieval_query(question: str) -> str:
    return " ".join(str(question or "").replace("\u3000", " ").split()).strip()


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def admit_relevant_ranked(
    ranked: Sequence[tuple[str, float]],
    *,
    minimum_logit: float = CALIBRATED_RELEVANCE_MIN_LOGIT,
    maximum_top_drop: float,
    reference_score: Optional[float] = None,
    maximum_reference_drop: Optional[float] = None,
) -> tuple[List[tuple[str, float]], Dict[str, str]]:
    """Apply the calibrated absolute and relative BGE relevance gates."""
    values = [(str(identifier), float(score)) for identifier, score in ranked]
    if not values:
        return [], {}
    top_score = max(score for _identifier, score in values)
    admitted: List[tuple[str, float]] = []
    rejected: Dict[str, str] = {}
    for identifier, score in values:
        if score < float(minimum_logit):
            rejected[identifier] = "below_calibrated_minimum"
            continue
        if top_score - score > float(maximum_top_drop):
            rejected[identifier] = "outside_top_score_band"
            continue
        if (
            reference_score is not None
            and maximum_reference_drop is not None
            and float(reference_score) - score > float(maximum_reference_drop)
        ):
            rejected[identifier] = "outside_primary_source_band"
            continue
        admitted.append((identifier, score))
    return admitted, rejected


class EvidenceRetrievalPipeline:
    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        reranker: Optional[RerankerService] = None,
    ) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("evidence retrieval requires at least one content chunk")
        if len(self.chunks) != len(embeddings):
            raise ValueError("content chunk and embedding counts do not match")
        self.embeddings = [tuple(float(value) for value in embedding) for embedding in embeddings]
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("evidence retrieval requires unique chunk_id values")
        self.bm25 = BM25Index(self.chunks)
        self.windows = EvidenceWindowBuilder(self.chunks)
        self.reranker = reranker or get_reranker_service()

    def _dense_search(self, query_embedding: Sequence[float], top_k: int) -> List[RankedChunk]:
        values = sorted(
            (
                (chunk.chunk_id, _dot(query_embedding, embedding), chunk.chunk_index)
                for chunk, embedding in zip(self.chunks, self.embeddings)
            ),
            key=lambda item: (-item[1], item[2], item[0]),
        )
        return [
            RankedChunk(chunk_id=chunk_id, score=score, rank=rank, source="dense")
            for rank, (chunk_id, score, _chunk_index) in enumerate(values[:top_k], start=1)
        ]

    def retrieve(
        self,
        question: str,
        query_embedding: Sequence[float],
        *,
        final_evidence_k: int,
        evidence_token_budget: int,
        source_chunk_ids: Optional[Sequence[str]] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        ranking_started = time.perf_counter()
        query = normalize_retrieval_query(question)
        if not query:
            raise ValueError("evidence retrieval question cannot be empty")
        source_ids = {str(value) for value in source_chunk_ids or [] if str(value)}

        recall_started = time.perf_counter()
        dense = self._dense_search(query_embedding, DEFAULT_DENSE_RECALL_K)
        lexical = self.bm25.search(query, top_k=DEFAULT_BM25_RECALL_K)
        fused = reciprocal_rank_fusion([dense, lexical])[:DEFAULT_RRF_CANDIDATE_K]
        candidates = [item for item in fused if item.chunk_id in self.by_id]
        if timing is not None:
            timing["recall_seconds"] = timing.get("recall_seconds", 0.0) + time.perf_counter() - recall_started

        atomic_started = time.perf_counter()
        atomic_pairs = [
            (item.chunk_id, self.by_id[item.chunk_id].retrieval_text)
            for item in candidates
        ]
        atomic_ranked = self.reranker.rank(query, atomic_pairs)
        source_reference_pairs = [
            (chunk_id, self.by_id[chunk_id].retrieval_text)
            for chunk_id in sorted(source_ids)
            if chunk_id in self.by_id
        ]
        source_reference_ranked = (
            self.reranker.rank(query, source_reference_pairs)
            if source_reference_pairs
            else []
        )
        source_reference_top_score = (
            max(score for _chunk_id, score in source_reference_ranked)
            if source_reference_ranked
            else None
        )
        atomic_non_source = [
            (chunk_id, score)
            for chunk_id, score in atomic_ranked
            if chunk_id not in source_ids
        ]
        atomic_admitted, atomic_rejected = admit_relevant_ranked(
            atomic_non_source,
            maximum_top_drop=ATOMIC_RELEVANCE_MAX_LOGIT_DROP,
            reference_score=source_reference_top_score,
            maximum_reference_drop=ATOMIC_PRIMARY_SOURCE_MAX_LOGIT_DROP,
        )
        atomic_ranked_ids = [chunk_id for chunk_id, _score in atomic_admitted[:DEFAULT_ATOMIC_RERANK_K]]
        atomic_scores = {chunk_id: score for chunk_id, score in atomic_ranked}
        if timing is not None:
            timing["atomic_rerank_seconds"] = timing.get("atomic_rerank_seconds", 0.0) + time.perf_counter() - atomic_started

        window_started = time.perf_counter()
        windows = self.windows.build(query=query, ranked_chunk_ids=atomic_ranked_ids)
        windows_by_id = {window.window_id: window for window in windows}
        raw_window_ranked = self.reranker.rank(
            query,
            [(window.window_id, window.text) for window in windows],
        ) if windows else []
        raw_window_scores = {window_id: score for window_id, score in raw_window_ranked}
        window_ranked = sorted(
            raw_window_ranked,
            key=lambda item: (
                -item[1],
                len(windows_by_id[item[0]].text) if item[0] in windows_by_id else 0,
                len(windows_by_id[item[0]].chunk_ids) if item[0] in windows_by_id else 0,
                item[0],
            ),
        )
        window_scores = raw_window_scores
        if window_ranked:
            window_ranked, window_rejected = admit_relevant_ranked(
                window_ranked,
                maximum_top_drop=WINDOW_RELEVANCE_MAX_LOGIT_DROP,
                reference_score=source_reference_top_score,
                maximum_reference_drop=WINDOW_PRIMARY_SOURCE_MAX_LOGIT_DROP,
            )
        else:
            window_rejected = {
                window_id: "no_atomic_anchor_admitted" for window_id in windows_by_id
            }

        # Structural alternatives with the same atomic anchors are mutually
        # exclusive (for example sibling hits with or without a parent body).
        deduped_window_ranked: List[tuple[str, float]] = []
        ranked_anchor_groups: set[tuple[str, ...]] = set()
        for window_id, score in window_ranked:
            window = windows_by_id[window_id]
            anchor_group = tuple(sorted(window.anchor_chunk_ids))
            if anchor_group and anchor_group in ranked_anchor_groups:
                continue
            if anchor_group:
                ranked_anchor_groups.add(anchor_group)
            deduped_window_ranked.append((window_id, score))
        window_ranked = deduped_window_ranked

        selected: List[EvidenceWindow] = []
        selected_chunk_ids: set[str] = set(source_ids)
        selected_evidence_chunk_ids: set[str] = set()
        tokens_used = 0
        for window_id, _score in window_ranked if int(final_evidence_k) > 0 else []:
            window = windows_by_id[window_id]
            if all(chunk_id in selected_chunk_ids for chunk_id in window.chunk_ids):
                continue
            if selected_evidence_chunk_ids.intersection(window.chunk_ids):
                continue
            incremental_chunks = [
                self.by_id[chunk_id]
                for chunk_id in window.chunk_ids
                if chunk_id not in selected_chunk_ids and chunk_id in self.by_id
            ]
            if not incremental_chunks:
                continue
            incremental_tokens = max(
                1,
                sum(max(1, (len(chunk.text) + 2) // 3) for chunk in incremental_chunks),
            )
            if tokens_used + incremental_tokens > max(1, int(evidence_token_budget)):
                continue
            selected.append(window)
            selected_chunk_ids.update(window.chunk_ids)
            selected_evidence_chunk_ids.update(
                chunk_id for chunk_id in window.chunk_ids if chunk_id not in source_ids
            )
            tokens_used += incremental_tokens
            if len(selected) >= max(0, int(final_evidence_k)):
                break
        if timing is not None:
            timing["window_rerank_seconds"] = timing.get("window_rerank_seconds", 0.0) + time.perf_counter() - window_started
            timing["ranking_seconds"] = timing.get("ranking_seconds", 0.0) + time.perf_counter() - ranking_started

        unique_selected_chunk_ids: List[str] = []
        emitted_chunk_ids: set[str] = set(source_ids)
        for window in selected:
            for chunk_id in window.chunk_ids:
                if chunk_id in emitted_chunk_ids:
                    continue
                emitted_chunk_ids.add(chunk_id)
                unique_selected_chunk_ids.append(chunk_id)
        return {
            "query": query,
            "selected_windows": selected,
            "selected_chunk_ids": unique_selected_chunk_ids,
            "trace": {
                "pipeline": "bm25_dense_rrf_bge_admission_structure_v2",
                "dense_hits": [item.__dict__ for item in dense],
                "bm25_hits": [item.__dict__ for item in lexical],
                "rrf_hits": [
                    {
                        "chunk_id": item.chunk_id,
                        "rrf_score": item.score,
                        "source_count": item.source_count,
                        "source_ranks": item.source_ranks,
                    }
                    for item in fused
                ],
                "atomic_rerank": [
                    {
                        "chunk_id": chunk_id,
                        "rerank_score": score,
                        "admitted": chunk_id not in atomic_rejected and chunk_id not in source_ids,
                        "rejection_reason": atomic_rejected.get(chunk_id),
                    }
                    for chunk_id, score in atomic_ranked
                ],
                "primary_source_rerank": [
                    {"chunk_id": chunk_id, "rerank_score": score}
                    for chunk_id, score in source_reference_ranked
                ],
                "window_candidates": [
                    {
                        "window_id": window.window_id,
                        "chunk_ids": list(window.chunk_ids),
                        "reason": window.reason,
                        "includes_parent_body": window.includes_parent_body,
                        "rerank_score": window_scores.get(window.window_id),
                        "admitted": window.window_id not in window_rejected,
                        "rejection_reason": window_rejected.get(window.window_id),
                    }
                    for window in windows
                ],
                "selected_windows": [
                    {
                        "window_id": window.window_id,
                        "chunk_ids": list(window.chunk_ids),
                        "reason": window.reason,
                        "rerank_score": window_scores.get(window.window_id),
                    }
                    for window in selected
                ],
                "atomic_scores": atomic_scores,
                "selected_evidence_window_count": len(selected),
                "selected_evidence_chunk_count": len(unique_selected_chunk_ids),
                "final_evidence_k": int(final_evidence_k),
                "evidence_token_budget": int(evidence_token_budget),
                "evidence_tokens_estimated": tokens_used,
                "relevance_admission": {
                    "minimum_logit": CALIBRATED_RELEVANCE_MIN_LOGIT,
                    "atomic_maximum_top_drop": ATOMIC_RELEVANCE_MAX_LOGIT_DROP,
                    "window_maximum_top_drop": WINDOW_RELEVANCE_MAX_LOGIT_DROP,
                    "primary_source_top_score": source_reference_top_score,
                    "atomic_primary_source_maximum_drop": ATOMIC_PRIMARY_SOURCE_MAX_LOGIT_DROP,
                    "window_primary_source_maximum_drop": WINDOW_PRIMARY_SOURCE_MAX_LOGIT_DROP,
                    "atomic_admitted_count": len(atomic_admitted),
                    "window_admitted_count": len(window_ranked),
                },
            },
        }


__all__ = [
    "ATOMIC_PRIMARY_SOURCE_MAX_LOGIT_DROP",
    "ATOMIC_RELEVANCE_MAX_LOGIT_DROP",
    "CALIBRATED_RELEVANCE_MIN_LOGIT",
    "DEFAULT_ATOMIC_RERANK_K",
    "DEFAULT_BM25_RECALL_K",
    "DEFAULT_DENSE_RECALL_K",
    "DEFAULT_RRF_CANDIDATE_K",
    "EvidenceRetrievalPipeline",
    "WINDOW_RELEVANCE_MAX_LOGIT_DROP",
    "WINDOW_PRIMARY_SOURCE_MAX_LOGIT_DROP",
    "admit_relevant_ranked",
    "normalize_retrieval_query",
]
