"""Standard Okapi BM25 over document content chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from .types import EvidenceChunk, RankedChunk


_SEGMENT_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in _SEGMENT_RE.finditer(str(text or "")):
        segment = match.group(0).lower()
        if re.fullmatch(r"[a-z0-9_]+", segment):
            tokens.append(segment)
            continue
        tokens.extend(segment)
        tokens.extend(segment[index : index + 2] for index in range(max(0, len(segment) - 1)))
    return tokens


class BM25Index:
    def __init__(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = float(k1)
        self.b = float(b)
        self._term_counts = [Counter(tokenize(chunk.retrieval_text)) for chunk in self.chunks]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = sum(self._lengths) / max(1, len(self._lengths))
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        document_count = len(self.chunks)
        self._idf: Dict[str, float] = {
            token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int) -> List[RankedChunk]:
        query_counts = Counter(tokenize(query))
        if not query_counts or int(top_k) <= 0:
            return []
        scored: List[tuple[EvidenceChunk, float]] = []
        for chunk, term_counts, document_length in zip(
            self.chunks,
            self._term_counts,
            self._lengths,
        ):
            score = 0.0
            normalization = self.k1 * (
                1.0 - self.b + self.b * document_length / max(1.0, self._average_length)
            )
            for term, query_frequency in query_counts.items():
                term_frequency = term_counts.get(term, 0)
                if term_frequency <= 0:
                    continue
                score += (
                    self._idf.get(term, 0.0)
                    * (term_frequency * (self.k1 + 1.0))
                    / (term_frequency + normalization)
                    * query_frequency
                )
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: (-item[1], item[0].chunk_index, item[0].chunk_id))
        return [
            RankedChunk(chunk_id=chunk.chunk_id, score=score, rank=rank, source="bm25")
            for rank, (chunk, score) in enumerate(scored[: max(0, int(top_k))], start=1)
        ]


__all__ = ["BM25Index", "tokenize"]
