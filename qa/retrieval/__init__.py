"""Public facade for fixed QA evidence retrieval."""

from .bm25 import BM25Index, tokenize
from .fusion import reciprocal_rank_fusion
from .reranker import RerankerService, get_reranker_service
from .pipeline import EvidenceRetrievalPipeline, normalize_retrieval_query
from .types import EvidenceChunk, EvidenceWindow, FusedChunk, RankedChunk
from .windows import EvidenceWindowBuilder

__all__ = [
    "BM25Index",
    "EvidenceChunk",
    "EvidenceWindow",
    "EvidenceWindowBuilder",
    "EvidenceRetrievalPipeline",
    "FusedChunk",
    "RankedChunk",
    "RerankerService",
    "get_reranker_service",
    "reciprocal_rank_fusion",
    "normalize_retrieval_query",
    "tokenize",
]
