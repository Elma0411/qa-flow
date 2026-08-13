"""Typed values shared by the QA evidence retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    chunk_index: int
    section_path: str
    section_parent_path: str
    section_level: int
    section_is_leaf: bool
    section_chunk_index: int
    title_path: str
    fragment_group_id: str
    fragment_index: int
    fragment_count: int
    content_kind: str
    source_asset_ids: Tuple[str, ...]
    text: str
    retrieval_text: str
    path_summary: str = ""
    embedding: Tuple[float, ...] = ()

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "EvidenceChunk":
        section_path = str(value.get("section_path") or "").strip()
        section_parent_path = str(value.get("section_parent_path") or "").strip()
        if not section_path:
            raise ValueError("evidence chunk is missing section_path")
        chunk_id = str(value.get("chunk_id") or value.get("id") or "").strip()
        if not chunk_id:
            raise ValueError("evidence chunk is missing chunk_id")
        assets = value.get("source_asset_ids") or []
        if not isinstance(assets, (list, tuple)):
            assets = []
        return cls(
            chunk_id=chunk_id,
            chunk_index=int(value.get("chunk_index") or 0),
            section_path=section_path,
            section_parent_path=section_parent_path,
            section_level=int(value.get("section_level") or 1),
            section_is_leaf=bool(value.get("section_is_leaf", True)),
            section_chunk_index=int(value.get("section_chunk_index") or 1),
            title_path=str(value.get("title_path") or "").strip(),
            fragment_group_id=str(
                value.get("fragment_group_id") or value.get("chunk_id") or value.get("id") or ""
            ).strip(),
            fragment_index=int(value.get("fragment_index") or 1),
            fragment_count=int(value.get("fragment_count") or 1),
            content_kind=str(value.get("content_kind") or "text").strip() or "text",
            source_asset_ids=tuple(str(item).strip() for item in assets if str(item).strip()),
            text=str(value.get("text") or "").strip(),
            retrieval_text=str(value.get("retrieval_text") or value.get("text_for_embedding") or value.get("text") or "").strip(),
            path_summary=str(value.get("path_summary") or "").strip(),
            embedding=tuple(float(item) for item in value.get("embedding") or ()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "section_path": self.section_path,
            "section_parent_path": self.section_parent_path,
            "section_level": self.section_level,
            "section_is_leaf": self.section_is_leaf,
            "section_chunk_index": self.section_chunk_index,
            "title_path": self.title_path,
            "fragment_group_id": self.fragment_group_id,
            "fragment_index": self.fragment_index,
            "fragment_count": self.fragment_count,
            "content_kind": self.content_kind,
            "source_asset_ids": list(self.source_asset_ids),
            "text": self.text,
            "retrieval_text": self.retrieval_text,
            "path_summary": self.path_summary,
        }


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    score: float
    rank: int
    source: str


@dataclass(frozen=True)
class FusedChunk:
    chunk_id: str
    score: float
    source_count: int
    source_ranks: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceWindow:
    window_id: str
    chunk_ids: Tuple[str, ...]
    reason: str
    text: str
    title_path: str
    anchor_chunk_ids: Tuple[str, ...]
    includes_parent_body: bool = False


__all__ = [
    "EvidenceChunk",
    "EvidenceWindow",
    "FusedChunk",
    "RankedChunk",
]
