"""Data contracts for the integrated document pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ImageMarker:
    image_id: str
    marker: str
    div_tag: str


@dataclass
class ChunkContext:
    chunk_index: int
    chunk_id: str
    text: str
    title_path: str = ""
    path_summary: str = ""
    summary: str = ""


@dataclass
class ImageAnchorContext:
    image_id: str
    marker: str
    image_path: Path
    chunk: ChunkContext
    original_context_before: str = ""
    original_context_after: str = ""
    context_before: str = ""
    context_after: str = ""


@dataclass
class PlacementDecision:
    image_id: str
    accepted: bool
    score: float
    reason: str
    raw_response: str = ""
    error: str = ""


@dataclass
class IntegratedDocumentResult:
    filename: str
    content: str
    markdown_content: str
    plain_text: Optional[str]
    pre_split_chunks: List[str]
    pre_split_chunk_meta: List[Dict[str, Any]]
    chunking_report: Dict[str, Any]
    ocr_raw_entry: Dict[str, Any]
    ocr_seconds: float

