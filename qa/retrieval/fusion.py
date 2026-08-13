"""Rank-based retrieval fusion."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .types import FusedChunk


def _chunk_id(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(getattr(value, "chunk_id", "") or "").strip()


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]],
    *,
    rank_constant: int = 60,
) -> List[FusedChunk]:
    scores: Dict[str, float] = {}
    source_ranks: Dict[str, Dict[str, int]] = {}
    for source_index, ranking in enumerate(rankings, start=1):
        source_name = f"source_{source_index}"
        seen: set[str] = set()
        for rank, value in enumerate(ranking, start=1):
            chunk_id = _chunk_id(value)
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (int(rank_constant) + rank)
            source_ranks.setdefault(chunk_id, {})[source_name] = rank
    fused = [
        FusedChunk(
            chunk_id=chunk_id,
            score=score,
            source_count=len(source_ranks.get(chunk_id) or {}),
            source_ranks=source_ranks.get(chunk_id) or {},
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda item: (-item.score, item.chunk_id))
    return fused


__all__ = ["reciprocal_rank_fusion"]
