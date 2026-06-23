"""Image marker helpers for OCR markdown."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from app.services.document_processing.ocr_processor.ocr_models import ImageInfo

from .models import ImageMarker

_DIV_PATTERN = re.compile(r"<div[^>]*>.*?</div>", re.DOTALL | re.IGNORECASE)
_SRC_PATTERN = re.compile(r'src="([^"]+)"', re.IGNORECASE)
_MARKER_RE = re.compile(r"\[\[IMAGE_REF:([^\]\s]+)\]\]")


def marker_for_image(image_id: str) -> str:
    return f"[[IMAGE_REF:{image_id}]]"


def extract_marker_ids(text: str) -> List[str]:
    return [match.group(1) for match in _MARKER_RE.finditer(text or "")]


def locate_markers_in_chunks(chunks_meta: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    locations: Dict[str, int] = {}
    for index, meta in enumerate(chunks_meta or [], start=1):
        if not isinstance(meta, Mapping):
            continue
        try:
            chunk_index = int(meta.get("chunk_index") or index)
        except Exception:
            chunk_index = index
        for image_id in extract_marker_ids(str(meta.get("text") or "")):
            locations.setdefault(image_id, chunk_index)
    return locations


def replace_image_divs_with_markers(
    markdown_content: str,
    images_info: Iterable[ImageInfo],
) -> Tuple[str, List[ImageMarker]]:
    current = str(markdown_content or "")
    markers: List[ImageMarker] = []
    used_ids: set[str] = set()

    for image in images_info:
        image_id = str(image.image_id or "").strip()
        if not image_id or image_id in used_ids:
            continue
        marker = marker_for_image(image_id)
        div_tag = str(image.div_tag or "").strip()
        replaced = False
        if div_tag and div_tag in current:
            current = current.replace(div_tag, marker, 1)
            replaced = True
        else:
            for match in list(_DIV_PATTERN.finditer(current)):
                div = match.group(0)
                src_match = _SRC_PATTERN.search(div)
                src_value = src_match.group(1) if src_match else ""
                filename = src_value.rsplit("/", 1)[-1]
                stem = filename.rsplit(".", 1)[0] if "." in filename else filename
                if stem == image_id:
                    start, end = match.span()
                    current = current[:start] + marker + current[end:]
                    div_tag = div
                    replaced = True
                    break
        if replaced:
            markers.append(ImageMarker(image_id=image_id, marker=marker, div_tag=div_tag))
            used_ids.add(image_id)
    return current, markers


def restore_markers_in_text(text: str, replacements: Dict[str, str], *, remove_missing: bool = True) -> str:
    current = str(text or "")
    for image_id in extract_marker_ids(current):
        marker = marker_for_image(image_id)
        replacement = replacements.get(image_id)
        if replacement:
            current = current.replace(marker, f"\n\n【图片描述：{replacement.strip()}】\n\n")
        elif remove_missing:
            current = current.replace(marker, "")
    return current


def with_context(image: ImageInfo, *, before: str, after: str) -> ImageInfo:
    return replace(image, context_before=before, context_after=after)
