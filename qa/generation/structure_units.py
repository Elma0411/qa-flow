"""Plan LLM-backed QA scenarios from logical document sections."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


QUALITY_STATUS_USABLE = "usable"
QUALITY_STATUS_CONTEXT_ONLY = "context_only"
QUALITY_STATUS_DROP = "drop"

UNIT_TYPE_POINT_SCENARIO = "point_scenario"
UNIT_TYPE_SUMMARY_SCENARIO = "summary_scenario"
SCENARIO_TYPE_POINT = "point"
SCENARIO_TYPE_SUMMARY = "summary"
DEFAULT_AUTO_SUMMARY_RATIO = 0.35
DEFAULT_SCENARIO_PLANNING_BATCH_CHARS = 24000


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _planning_material_ref(position: int) -> str:
    """Return a model-facing alias with no internal numeric meaning."""
    number = max(1, int(position))
    letters: List[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "主材料-" + "".join(reversed(letters))


def _planning_image_ref(position: int) -> str:
    number = max(1, int(position))
    letters: List[str] = []
    while number:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "图片-" + "".join(reversed(letters))


def _parent_node_path(title_path: str) -> str:
    value = _safe_text(title_path).replace("＞", ">")
    if ">" not in value:
        return ""
    return value.rsplit(">", 1)[0].strip(" >")


def _subject_label(title_path: str) -> str:
    """Derive a concise standalone subject from the human-readable path."""
    value = _safe_text(title_path).replace("＞", ">")
    first = next((part.strip() for part in value.split(">") if part.strip()), "")
    first = re.sub(r"^#{1,6}\s*", "", first).strip()
    first = re.sub(r"\.(?:pdf|docx?|txt|md)$", "", first, flags=re.I).strip()
    if not first:
        return ""
    if first.startswith("《") and first.endswith("》"):
        return first
    if re.search(r"(条例|办法|规定|细则|规则|规范|标准|指南|手册|方案|通知|意见|决定|法律|法)$", first):
        return f"《{first}》"
    return first


def _invoke_scenario_planner(
    scenario_planner: Callable[..., Sequence[Dict[str, Any]]],
    batch: Sequence["SectionMaterial"],
    requested_count: int,
    scenario_type: str,
    *,
    batch_index: int,
    batch_count: int,
) -> Sequence[Dict[str, Any]]:
    """Call old three-argument planners and new contextual planners alike."""
    try:
        parameters = inspect.signature(scenario_planner).parameters
        accepts_context = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ) or "planning_batch_index" in parameters
    except (TypeError, ValueError):
        accepts_context = False
    if accepts_context:
        return scenario_planner(
            batch,
            requested_count,
            scenario_type,
            planning_batch_index=batch_index,
            planning_batch_count=batch_count,
        )
    return scenario_planner(batch, requested_count, scenario_type)


def _collapse_text(text: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(text))


def _text_for_quality(chunk: Dict[str, Any]) -> str:
    return _safe_text(chunk.get("text")) or _safe_text(chunk.get("text_for_embedding"))


def _token_set(text: str) -> set[str]:
    normalized = _collapse_text(text).lower()
    tokens: set[str] = set()
    for segment in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized):
        if re.fullmatch(r"[a-z0-9_]+", segment):
            if len(segment) >= 2:
                tokens.add(segment)
        elif len(segment) <= 2:
            tokens.add(segment)
        else:
            tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _line_texts(text: str) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _has_sentence_signal(text: str) -> bool:
    return bool(re.search(r"[。！？!?；;:：]", text)) or bool(
        re.search(r"\b(is|are|means|includes|shall|must|should|requires?)\b", text, re.I)
    )


def _has_list_signal(text: str) -> bool:
    return bool(
        re.search(
            r"(?m)^\s*(?:[-*•]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*\S+",
            text,
        )
    )


def _has_fact_signal(text: str) -> bool:
    if not text:
        return False
    if _has_sentence_signal(text) or _has_list_signal(text):
        return True
    return bool(
        re.search(
            r"(定义为|是指|包括|包含|适用于|应当|必须|不得|要求|标准|条件|流程|步骤|范围|"
            r"\bdefine[sd]?\b|\binclude[sd]?\b|\brequire[sd]?\b|applies to)",
            text,
            re.I,
        )
    )


def _looks_placeholder(text: str) -> bool:
    clean = _collapse_text(text)
    if not clean:
        return True
    hit = bool(
        re.search(
            r"(图片|图像|图示|截图|附件|二维码|扫描件|占位|见图|见附件|image|figure|attachment|placeholder)",
            clean,
            re.I,
        )
    )
    return hit and len(clean) <= 160 and not _has_fact_signal(clean)


def _looks_title_only(text: str, title_path: str) -> bool:
    clean = _collapse_text(text)
    if not clean:
        return True
    lines = _line_texts(text)
    if len(lines) > 2:
        return False
    title_tail = title_path.replace("＞", ">").split(">")[-1].strip() if title_path else ""
    if title_tail and clean == title_tail:
        return True
    if len(clean) <= 48 and not _has_sentence_signal(clean) and not _has_list_signal(clean):
        return True
    return bool(
        re.fullmatch(
            r"(第?[一二三四五六七八九十0-9]+[章节条部分篇].{0,40}|[0-9.、\s]{1,12}\S{0,40})",
            clean,
        )
    )


def _looks_table_fragment(text: str) -> bool:
    lines = _line_texts(text)
    if not lines:
        return False
    tableish = sum(
        1
        for line in lines
        if "|" in line
        or "\t" in line
        or (len(re.split(r"\s{2,}|,|，", line)) >= 3 and not _has_sentence_signal(line))
    )
    return tableish >= max(1, len(lines) // 2) and not _has_fact_signal(text)


def _symbol_digit_ratio(text: str) -> float:
    clean = _collapse_text(text)
    if not clean:
        return 1.0
    noisy = sum(
        1
        for ch in clean
        if not ("\u4e00" <= ch <= "\u9fff") and not ch.isalpha() and not ch.isspace()
    )
    return noisy / max(1, len(clean))


def _structure_signal(text: str) -> bool:
    return bool(
        re.search(
            r"(流程|步骤|条件|材料|范围|规则|标准|要求|对比|分类|组成|清单|目录|适用|定义|"
            r"process|step|condition|rule|standard|requirement|compare|category|definition)",
            text,
            re.I,
        )
    )


@dataclass(frozen=True)
class ChunkQuality:
    chunk_index: int
    status: str
    score: float
    reasons: List[str]
    char_count: int
    duplicate_ratio: float
    has_title_path: bool
    has_structure_signal: bool
    has_fact_signal: bool
    symbol_digit_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "status": self.status,
            "score": round(float(self.score), 4),
            "reasons": list(self.reasons),
            "char_count": self.char_count,
            "duplicate_ratio": round(float(self.duplicate_ratio), 4),
            "has_title_path": self.has_title_path,
            "has_structure_signal": self.has_structure_signal,
            "has_fact_signal": self.has_fact_signal,
            "symbol_digit_ratio": round(float(self.symbol_digit_ratio), 4),
        }


@dataclass(frozen=True)
class StructureGraph:
    chunk_count: int
    children_by_parent: Dict[str, List[int]]
    previous_by_index: Dict[int, Optional[int]]
    next_by_index: Dict[int, Optional[int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_count": self.chunk_count,
            "parent_group_count": len(self.children_by_parent),
            "parent_groups": {key: list(value) for key, value in self.children_by_parent.items() if key},
        }


@dataclass(frozen=True)
class ImageMaterial:
    image_id: str
    description: str
    context_before: str
    context_after: str
    source_chunk_index: int

    def to_prompt_dict(self, image_ref: str) -> Dict[str, Any]:
        return {
            "image_ref": image_ref,
            "description": self.description,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_chunk_index": self.source_chunk_index,
            "description_char_count": len(self.description),
            "context_before_char_count": len(self.context_before),
            "context_after_char_count": len(self.context_after),
        }


@dataclass(frozen=True)
class SectionMaterial:
    material_id: str
    material_index: int
    section_path: str
    section_parent_path: str
    section_level: int
    title_path: str
    source_chunk_indexes: List[int]
    source_chunk_ids: List[str]
    content_kinds: List[str]
    source_asset_ids: List[str]
    material_text: str
    text_content: str
    image_materials: List[ImageMaterial]
    subject_label: str
    usable: bool
    quality_score: float

    def to_prompt_dict(
        self,
        material_ref: Optional[str] = None,
        image_refs_by_id: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Render a material for the planner without exposing internal IDs."""
        node_path = self.title_path or "未标注章节"
        image_ref_map = image_refs_by_id or {
            image.image_id: _planning_image_ref(index)
            for index, image in enumerate(self.image_materials, start=1)
        }
        return {
            "material_ref": material_ref or _planning_material_ref(self.material_index),
            "node_path": node_path,
            "parent_node_path": _parent_node_path(node_path),
            "subject_label": self.subject_label,
            "text_content": self.text_content,
            "image_materials": [
                image.to_prompt_dict(image_ref_map[image.image_id])
                for image in self.image_materials
                if image.image_id in image_ref_map
            ],
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "material_index": self.material_index,
            "section_path": self.section_path,
            "section_parent_path": self.section_parent_path,
            "section_level": self.section_level,
            "title_path": self.title_path,
            "source_chunk_indexes": list(self.source_chunk_indexes),
            "source_chunk_ids": list(self.source_chunk_ids),
            "content_kinds": list(self.content_kinds),
            "source_asset_ids": list(self.source_asset_ids),
            "material_char_count": len(self.material_text),
            "text_char_count": len(self.text_content),
            "subject_label": self.subject_label,
            "image_materials": [image.to_debug_dict() for image in self.image_materials],
            "usable": self.usable,
            "quality_score": round(float(self.quality_score), 4),
        }


@dataclass(frozen=True)
class GenerationUnit:
    unit_id: str
    unit_index: int
    unit_type: str
    qa_mode: str
    scenario_intent: str
    reader_need: str
    material_ids: List[str]
    required_material_ids: List[str]
    optional_material_ids: List[str]
    evidence_mode: str
    required_image_ids: List[str]
    subject_label: str
    prompt_materials: List[Dict[str, Any]]
    material_ref_map: Dict[str, str]
    image_ref_map: Dict[str, str]
    material_source_chunk_indexes: Dict[str, List[int]]
    anchor_chunk_index: int
    source_chunk_indexes: List[int]
    section_path: str
    title_path: str
    unit_text: str
    qa_budget: int
    child_count: int
    usable_child_count: int
    quality_child_coverage: float
    debug: Dict[str, Any]
    source_chunk_meta: Dict[str, Any]

    def with_index_and_budget(self, unit_index: int, qa_budget: int) -> "GenerationUnit":
        return replace(self, unit_index=unit_index, qa_budget=max(0, int(qa_budget)))

    def to_source_unit(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_index": self.unit_index,
            "unit_type": self.unit_type,
            "qa_mode": self.qa_mode,
            "scenario_intent": self.scenario_intent,
            "reader_need": self.reader_need,
            "material_ids": list(self.material_ids),
            "required_material_ids": list(self.required_material_ids),
            "optional_material_ids": list(self.optional_material_ids),
            "evidence_mode": self.evidence_mode,
            "required_image_ids": list(self.required_image_ids),
            "subject_label": self.subject_label,
            "prompt_materials": [dict(value) for value in self.prompt_materials],
            "material_ref_map": dict(self.material_ref_map),
            "image_ref_map": dict(self.image_ref_map),
            "material_paths": list(self.debug.get("material_paths") or []),
            "material_source_chunk_indexes": {
                key: list(value)
                for key, value in self.material_source_chunk_indexes.items()
            },
            "anchor_chunk_index": self.anchor_chunk_index,
            "source_chunk_indexes": list(self.source_chunk_indexes),
            "section_path": self.section_path,
            "title_path": self.title_path,
            "unit_text": self.unit_text,
            "qa_budget": self.qa_budget,
            "debug": dict(self.debug),
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            **self.to_source_unit(),
            "child_count": self.child_count,
            "usable_child_count": self.usable_child_count,
            "quality_child_coverage": round(float(self.quality_child_coverage), 4),
            "unit_char_count": len(self.unit_text),
        }


@dataclass(frozen=True)
class GenerationUnitPlan:
    units: List[GenerationUnit]
    reserve_units: List[GenerationUnit]
    section_materials: List[SectionMaterial]
    chunk_quality: Dict[int, ChunkQuality]
    graph: StructureGraph
    requested_total_qa: int
    effective_total_qa: int
    qa_total_limit: Optional[int]
    qa_detail_mode: str
    qa_per_chunk_fallback: int
    dropped_unit_count_by_budget: int
    scenario_candidates_by_type: Dict[str, int]
    scenario_selected_by_type: Dict[str, int]
    scenario_planner_calls_by_type: Dict[str, int]
    scenario_planner_batches_by_type: Dict[str, List[int]]
    scenario_planner_batch_details: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    scenario_planning_batch_chars: int = DEFAULT_SCENARIO_PLANNING_BATCH_CHARS
    text_model_concurrency: int = 1

    def summary(self) -> Dict[str, Any]:
        quality_counts: Dict[str, int] = defaultdict(int)
        for quality in self.chunk_quality.values():
            quality_counts[quality.status] += 1
        return {
            "chunks_total": self.graph.chunk_count,
            "section_materials_total": len(self.section_materials),
            "generation_units_total": len(self.units),
            "reserve_generation_units_total": len(self.reserve_units),
            "requested_total_qa": self.requested_total_qa,
            "effective_total_qa": self.effective_total_qa,
            "qa_total_limit": self.qa_total_limit,
            "qa_detail_mode": self.qa_detail_mode,
            "qa_per_chunk_fallback": self.qa_per_chunk_fallback,
            "dropped_unit_count_by_budget": self.dropped_unit_count_by_budget,
            "quality_counts": dict(quality_counts),
            "scenario_candidates_by_type": dict(self.scenario_candidates_by_type),
            "scenario_selected_by_type": dict(self.scenario_selected_by_type),
            "scenario_planner_calls_by_type": dict(self.scenario_planner_calls_by_type),
            "scenario_planner_batches_by_type": {
                key: list(value)
                for key, value in self.scenario_planner_batches_by_type.items()
            },
            "scenario_planner_batch_details": {
                key: [dict(item) for item in value]
                for key, value in self.scenario_planner_batch_details.items()
            },
            "scenario_planning_batch_chars": self.scenario_planning_batch_chars,
            "text_model_concurrency": self.text_model_concurrency,
        }


def build_structure_graph(document_chunks: Sequence[Dict[str, Any]]) -> StructureGraph:
    ordered = [int(chunk.get("chunk_index") or index) for index, chunk in enumerate(document_chunks, 1)]
    previous_by_index: Dict[int, Optional[int]] = {}
    next_by_index: Dict[int, Optional[int]] = {}
    for position, chunk_index in enumerate(ordered):
        previous_by_index[chunk_index] = ordered[position - 1] if position > 0 else None
        next_by_index[chunk_index] = ordered[position + 1] if position + 1 < len(ordered) else None
    children_by_parent: Dict[str, List[int]] = defaultdict(list)
    for chunk in document_chunks:
        parent = _safe_text(chunk.get("section_parent_path"))
        if parent:
            children_by_parent[parent].append(int(chunk.get("chunk_index") or 0))
    return StructureGraph(
        chunk_count=len(document_chunks),
        children_by_parent=dict(children_by_parent),
        previous_by_index=previous_by_index,
        next_by_index=next_by_index,
    )


def evaluate_chunk_quality(
    chunk: Dict[str, Any],
    *,
    previous_chunk: Optional[Dict[str, Any]] = None,
    next_chunk: Optional[Dict[str, Any]] = None,
) -> ChunkQuality:
    text = _text_for_quality(chunk)
    clean = _collapse_text(text)
    tokens = _token_set(text)
    duplicate_ratio = max(
        [
            _jaccard(tokens, _token_set(_text_for_quality(neighbor)))
            for neighbor in (previous_chunk, next_chunk)
            if neighbor
        ]
        or [0.0]
    )
    title_path = _safe_text(chunk.get("title_path"))
    has_fact_signal = _has_fact_signal(text)
    has_structure_signal = _structure_signal(text) or _structure_signal(title_path)
    title_only = _looks_title_only(text, title_path)
    placeholder = _looks_placeholder(text)
    table_fragment = _looks_table_fragment(text)
    noisy_ratio = _symbol_digit_ratio(text)
    score = 1.0
    reasons: List[str] = []
    if len(clean) < 80 and not title_path and not _has_list_signal(text):
        score -= 0.35
        reasons.append("short_without_structure")
    if title_only:
        score -= 0.30
        reasons.append("title_only")
    if table_fragment:
        score -= 0.25
        reasons.append("table_fragment_without_fact_sentence")
    if placeholder:
        score -= 0.25
        reasons.append("placeholder_without_qa_text")
    if duplicate_ratio >= 0.84:
        score -= 0.20
        reasons.append("high_adjacent_duplicate")
    if noisy_ratio >= 0.48 and len(clean) < 320:
        score -= 0.15
        reasons.append("symbol_digit_ratio_abnormal")
    if title_path:
        score += 0.10
    if has_structure_signal or has_fact_signal:
        score += 0.10
    score = max(0.0, min(1.0, score))
    if not clean or placeholder or (title_only and not has_fact_signal):
        status = QUALITY_STATUS_DROP
    elif table_fragment or score < 0.52 or (duplicate_ratio >= 0.90 and not has_fact_signal):
        status = QUALITY_STATUS_CONTEXT_ONLY
    else:
        status = QUALITY_STATUS_USABLE
    return ChunkQuality(
        chunk_index=int(chunk.get("chunk_index") or 0),
        status=status,
        score=score,
        reasons=reasons,
        char_count=len(clean),
        duplicate_ratio=duplicate_ratio,
        has_title_path=bool(title_path),
        has_structure_signal=has_structure_signal,
        has_fact_signal=has_fact_signal,
        symbol_digit_ratio=noisy_ratio,
    )


_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_IMAGE_DESCRIPTION_RE = re.compile(r"\s*【图片描述：(.*?)】\s*", re.DOTALL)


def _strip_repeated_fragment_heading(text: str) -> str:
    """Remove the splitter-repeated heading from non-first physical fragments."""
    lines = str(text or "").splitlines()
    first_content_index = next(
        (index for index, line in enumerate(lines) if line.strip()),
        None,
    )
    if first_content_index is None or not _MARKDOWN_HEADING_RE.match(lines[first_content_index]):
        return str(text or "").strip()
    return "\n".join(lines[first_content_index + 1 :]).strip()


def _render_material_text(
    chunks: Sequence[Dict[str, Any]],
    *,
    exclude_image_descriptions: bool = False,
) -> str:
    pieces: List[str] = []
    for chunk in sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0)):
        text = _text_for_quality(chunk).strip()
        if exclude_image_descriptions:
            text = _IMAGE_DESCRIPTION_RE.sub("\n", text).strip()
        if int(chunk.get("fragment_index") or 1) > 1:
            text = _strip_repeated_fragment_heading(text)
        if not text:
            continue
        if pieces and text == pieces[-1]:
            continue
        pieces.append(text)
    return "\n\n".join(pieces).strip()


def _build_image_materials(chunks: Sequence[Dict[str, Any]]) -> List[ImageMaterial]:
    """Restore typed image blocks, including old integrated chunk metadata."""
    restored: List[ImageMaterial] = []
    seen: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0)):
        chunk_index = int(chunk.get("chunk_index") or 0)
        raw_materials = chunk.get("image_materials")
        if isinstance(raw_materials, list):
            for raw in raw_materials:
                if not isinstance(raw, dict):
                    continue
                image_id = _safe_text(raw.get("image_id") or raw.get("source_asset_id"))
                description = _safe_text(raw.get("description"))
                if not image_id or not description or image_id in seen:
                    continue
                seen.add(image_id)
                restored.append(
                    ImageMaterial(
                        image_id=image_id,
                        description=description,
                        context_before=_safe_text(raw.get("context_before")),
                        context_after=_safe_text(raw.get("context_after")),
                        source_chunk_index=chunk_index,
                    )
                )
        accepted_ids = [
            _safe_text(value)
            for value in (chunk.get("image_replacements") or {}).get("accepted_ids", [])
            if _safe_text(value)
        ] if isinstance(chunk.get("image_replacements"), dict) else []
        descriptions = [match.strip() for match in _IMAGE_DESCRIPTION_RE.findall(_text_for_quality(chunk))]
        for image_id, description in zip(accepted_ids, descriptions):
            if not image_id or not description or image_id in seen:
                continue
            seen.add(image_id)
            restored.append(
                ImageMaterial(
                    image_id=image_id,
                    description=description,
                    context_before="",
                    context_after="",
                    source_chunk_index=chunk_index,
                )
            )
    return restored


def _batch_section_materials(
    materials: Sequence[SectionMaterial],
    *,
    max_batch_chars: int,
    preserve_parent_neighborhood: bool,
) -> List[List[SectionMaterial]]:
    """Pack complete logical sections without truncating or splitting a material."""
    ordered = sorted(materials, key=lambda material: material.material_index)
    if not ordered:
        return []
    budget = max(1000, int(max_batch_chars))
    groups: List[List[SectionMaterial]] = []
    if preserve_parent_neighborhood:
        current_group: List[SectionMaterial] = []
        current_key: Optional[Tuple[str, str]] = None
        for material in ordered:
            group_key = (
                material.section_parent_path or material.section_path,
                material.section_path.split(".")[0],
            )
            if current_group and group_key != current_key:
                groups.append(current_group)
                current_group = []
            current_group.append(material)
            current_key = group_key
        if current_group:
            groups.append(current_group)
    else:
        groups = [[material] for material in ordered]

    batches: List[List[SectionMaterial]] = []
    current: List[SectionMaterial] = []
    current_chars = 0
    for group in groups:
        group_chars = sum(len(material.material_text) + len(material.title_path) + 160 for material in group)
        if current and current_chars + group_chars > budget:
            batches.append(current)
            current = []
            current_chars = 0
        if group_chars <= budget:
            current.extend(group)
            current_chars += group_chars
            continue
        for material in group:
            material_chars = len(material.material_text) + len(material.title_path) + 160
            if current and current_chars + material_chars > budget:
                batches.append(current)
                current = []
                current_chars = 0
            current.append(material)
            current_chars += material_chars
    if current:
        batches.append(current)
    return batches


def _allocate_planning_counts(
    batches: Sequence[Sequence[SectionMaterial]],
    requested_count: int,
) -> List[int]:
    """Allocate an integer scenario cap across batches without exceeding the total."""
    total = max(0, int(requested_count))
    if total == 0 or not batches:
        return [0] * len(batches)
    weights = [max(1, len(batch)) for batch in batches]
    weight_total = sum(weights)
    exact = [total * weight / weight_total for weight in weights]
    counts = [int(value) for value in exact]
    for index in sorted(
        range(len(batches)),
        key=lambda item: (-(exact[item] - counts[item]), item),
    )[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _plan_scenario_pool(
    materials: Sequence[SectionMaterial],
    *,
    scenario_type: str,
    requested_count: int,
    scenario_planner: Callable[..., Sequence[Dict[str, Any]]],
    max_batch_chars: int,
    max_concurrency: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    batches = _batch_section_materials(
        materials,
        max_batch_chars=max_batch_chars,
        preserve_parent_neighborhood=scenario_type == SCENARIO_TYPE_SUMMARY,
    )
    counts = _allocate_planning_counts(batches, requested_count)
    active_batches = [
        (batch_index, batch, count)
        for batch_index, (batch, count) in enumerate(zip(batches, counts), start=1)
        if count > 0
    ]
    if not active_batches:
        return [], []

    def run_one(
        batch_index: int,
        batch: Sequence[SectionMaterial],
        count: int,
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
        raw_items = _invoke_scenario_planner(
            scenario_planner,
            batch,
            count,
            scenario_type,
            batch_index=batch_index,
            batch_count=len(batches),
        )
        metadata = getattr(raw_items, "debug_metadata", {})
        return (
            batch_index,
            [item for item in raw_items if isinstance(item, dict)],
            dict(metadata) if isinstance(metadata, dict) else {},
        )

    results_by_batch: Dict[int, List[Dict[str, Any]]] = {}
    metadata_by_batch: Dict[int, Dict[str, Any]] = {}
    errors_by_batch: Dict[int, str] = {}
    worker_count = max(1, min(int(max_concurrency or 1), len(active_batches)))
    if worker_count == 1:
        for batch_index, batch, count in active_batches:
            try:
                index, result, metadata = run_one(batch_index, batch, count)
                results_by_batch[index] = result
                metadata_by_batch[index] = metadata
            except Exception as exc:
                errors_by_batch[batch_index] = f"{type(exc).__name__}: {exc}"
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(run_one, batch_index, batch, count): batch_index
                for batch_index, batch, count in active_batches
            }
            for future in as_completed(future_map):
                batch_index = future_map[future]
                try:
                    index, result, metadata = future.result()
                    results_by_batch[index] = result
                    metadata_by_batch[index] = metadata
                except Exception as exc:
                    errors_by_batch[batch_index] = f"{type(exc).__name__}: {exc}"

    planned: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    materials_by_id = {material.material_id: material for material in materials}
    for batch_index, batch, count in active_batches:
        batch_items = results_by_batch.get(batch_index, [])
        for item in batch_items:
            item_copy = dict(item)
            item_copy.setdefault("_planning_batch_index", batch_index)
            item_copy.setdefault("_planning_batch_count", len(batches))
            planned.append(item_copy)
        scenarios: List[Dict[str, Any]] = []
        for item in batch_items:
            required_ids = _normalize_material_id_list(
                item.get("required_material_ids"),
                materials_by_id,
            )
            optional_ids = _normalize_material_id_list(
                item.get("optional_material_ids"), materials_by_id
            )
            paths = [
                materials_by_id[material_id].title_path
                for material_id in [*required_ids, *optional_ids]
                if material_id in materials_by_id and materials_by_id[material_id].title_path
            ]
            scenarios.append(
                {
                    "scenario_type": item.get("scenario_type") or item.get("type"),
                    "scenario_intent": item.get("intent"),
                    "reader_need": item.get("reader_need"),
                    "required_material_paths": [
                        materials_by_id[material_id].title_path
                        for material_id in required_ids
                        if material_id in materials_by_id and materials_by_id[material_id].title_path
                    ],
                    "optional_material_paths": [
                        materials_by_id[material_id].title_path
                        for material_id in optional_ids
                        if material_id in materials_by_id and materials_by_id[material_id].title_path
                    ],
                    "material_paths": paths,
                }
            )
        details.append(
            {
                "batch_index": batch_index,
                "batch_count": len(batches),
                "scenario_type": scenario_type,
                "requested_count": count,
                "returned_count": len(batch_items),
                "validated_count": int(
                    metadata_by_batch.get(batch_index, {}).get(
                        "items_validated_count", len(batch_items)
                    )
                    or 0
                ),
                "dropped_reasons": dict(
                    metadata_by_batch.get(batch_index, {}).get(
                        "dropped_validation_reasons", {}
                    )
                    or {}
                ),
                "raw_response_available": bool(
                    metadata_by_batch.get(batch_index, {}).get("raw_response")
                ),
                "error": errors_by_batch.get(batch_index),
                "material_count": len(batch),
                "material_paths": [material.title_path for material in batch if material.title_path],
                "scenarios": scenarios,
            }
        )
    return planned, details


def _merge_cross_batch_summary_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    materials_by_id: Dict[str, SectionMaterial],
) -> List[Dict[str, Any]]:
    """Create conservative cross-batch Summary candidates from local plans.

    Long documents can split one parent section across planner calls. This
    merge only combines candidates with a shared parent path or a strong
    intent overlap, and never merges Point candidates. The full source text is
    still fetched later by material ID, so the merge call itself has no extra
    document-sized prompt or LLM request.
    """
    summary_items = [
        dict(item)
        for item in candidates
        if isinstance(item, dict)
        and _normalize_scenario_type(item.get("scenario_type") or item.get("type"))
        == SCENARIO_TYPE_SUMMARY
    ]
    merged: List[Dict[str, Any]] = []
    for left_index, left in enumerate(summary_items):
        left_required = _normalize_material_id_list(
            left.get("required_material_ids"), materials_by_id
        )
        left_optional = _normalize_material_id_list(
            left.get("optional_material_ids"), materials_by_id
        )
        left_batch = left.get("_planning_batch_index")
        for right in summary_items[left_index + 1 :]:
            if right.get("_planning_batch_index") == left_batch:
                continue
            right_required = _normalize_material_id_list(
                right.get("required_material_ids"), materials_by_id
            )
            right_optional = _normalize_material_id_list(
                right.get("optional_material_ids"), materials_by_id
            )
            if not left_required or not right_required:
                continue
            left_parents = {
                material.section_parent_path
                for material_id in [*left_required, *left_optional]
                if (material := materials_by_id.get(material_id)) and material.section_parent_path
            }
            right_parents = {
                material.section_parent_path
                for material_id in [*right_required, *right_optional]
                if (material := materials_by_id.get(material_id)) and material.section_parent_path
            }
            intent_overlap = _jaccard(
                _token_set(_safe_text(left.get("intent"))),
                _token_set(_safe_text(right.get("intent"))),
            )
            need_overlap = _jaccard(
                _token_set(_safe_text(left.get("reader_need"))),
                _token_set(_safe_text(right.get("reader_need"))),
            )
            if not (left_parents & right_parents or max(intent_overlap, need_overlap) >= 0.42):
                continue
            required_ids: List[str] = []
            optional_ids: List[str] = []
            for material_id in [*left_required, *right_required]:
                if material_id not in required_ids:
                    required_ids.append(material_id)
            for material_id in [*left_optional, *right_optional]:
                if material_id not in required_ids and material_id not in optional_ids:
                    optional_ids.append(material_id)
            if len(required_ids) < 2:
                continue
            required_image_ids = list(dict.fromkeys([
                *[str(value) for value in left.get("required_image_ids") or [] if str(value)],
                *[str(value) for value in right.get("required_image_ids") or [] if str(value)],
            ]))
            merged.append(
                {
                    "scenario_type": SCENARIO_TYPE_SUMMARY,
                    "intent": _safe_text(left.get("intent")) or _safe_text(right.get("intent")),
                    "reader_need": _safe_text(left.get("reader_need"))
                    or _safe_text(right.get("reader_need")),
                    "required_material_ids": required_ids,
                    "optional_material_ids": optional_ids,
                    "evidence_mode": "mixed" if required_image_ids else "text",
                    "required_image_ids": required_image_ids,
                    "cross_batch_merge": True,
                    "cross_batch_merge_from": [left_batch, right.get("_planning_batch_index")],
                }
            )
    return merged


def build_section_materials(
    document_chunks: Sequence[Dict[str, Any]],
    chunk_quality: Dict[int, ChunkQuality],
) -> List[SectionMaterial]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in document_chunks:
        section_path = _safe_text(chunk.get("section_path"))
        if section_path:
            grouped[section_path].append(dict(chunk))
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: min(int(chunk.get("chunk_index") or 0) for chunk in item[1]),
    )
    materials: List[SectionMaterial] = []
    for material_index, (section_path, chunks) in enumerate(ordered_groups, start=1):
        chunks = sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0))
        retained = [
            chunk
            for chunk in chunks
            if chunk_quality.get(int(chunk.get("chunk_index") or 0), None)
            and chunk_quality[int(chunk.get("chunk_index") or 0)].status != QUALITY_STATUS_DROP
        ]
        if not retained:
            continue
        indexes = [int(chunk.get("chunk_index") or 0) for chunk in retained]
        ids = [_safe_text(chunk.get("chunk_id")) for chunk in retained]
        usable_count = sum(
            chunk_quality[index].status == QUALITY_STATUS_USABLE for index in indexes
        )
        assets: List[str] = []
        content_kinds: List[str] = []
        for chunk in retained:
            kind = _safe_text(chunk.get("content_kind")) or "text"
            if kind not in content_kinds:
                content_kinds.append(kind)
            for asset in chunk.get("source_asset_ids") or []:
                value = _safe_text(asset)
                if value and value not in assets:
                    assets.append(value)
        material_id = f"section-{material_index}"
        image_materials = _build_image_materials(retained)
        materials.append(
            SectionMaterial(
                material_id=material_id,
                material_index=material_index,
                section_path=section_path,
                section_parent_path=_safe_text(retained[0].get("section_parent_path")),
                section_level=max(1, int(retained[0].get("section_level") or 1)),
                title_path=_safe_text(retained[0].get("title_path")),
                source_chunk_indexes=indexes,
                source_chunk_ids=ids,
                content_kinds=content_kinds,
                source_asset_ids=assets,
                material_text=_render_material_text(retained),
                text_content=_render_material_text(
                    retained,
                    exclude_image_descriptions=True,
                ),
                image_materials=image_materials,
                subject_label=_subject_label(_safe_text(retained[0].get("title_path"))),
                usable=usable_count > 0,
                quality_score=sum(chunk_quality[index].score for index in indexes) / max(1, len(indexes)),
            )
        )
    return materials


def _normalize_scenario_type(value: Any) -> str:
    normalized = _safe_text(value).lower()
    if normalized in {"point", "point only", "single", "single_hop", "单点", "单点题"}:
        return SCENARIO_TYPE_POINT
    if normalized in {"summary", "summary only", "multi", "multi_hop", "总结", "总结题"}:
        return SCENARIO_TYPE_SUMMARY
    return ""


def _normalize_evidence_mode(value: Any, *, has_required_images: bool) -> str:
    normalized = _safe_text(value).lower()
    if normalized not in {"text", "visual", "mixed"}:
        normalized = "mixed" if has_required_images else "text"
    if has_required_images and normalized == "text":
        return "mixed"
    if not has_required_images and normalized in {"visual", "mixed"}:
        return "text"
    return normalized


def _normalize_material_id_list(raw: Any, materials_by_id: Dict[str, SectionMaterial]) -> List[str]:
    values = raw if isinstance(raw, list) else []
    result: List[str] = []
    for value in values:
        material_id = _safe_text(value)
        if material_id in materials_by_id and material_id not in result:
            result.append(material_id)
    return result


def _build_generation_unit(
    raw: Dict[str, Any],
    *,
    materials_by_id: Dict[str, SectionMaterial],
    chunks_by_index: Dict[int, Dict[str, Any]],
) -> Optional[GenerationUnit]:
    scenario_type = _normalize_scenario_type(raw.get("scenario_type") or raw.get("type"))
    intent = _safe_text(raw.get("intent"))
    reader_need = _safe_text(raw.get("reader_need")) or intent
    required_material_ids = _normalize_material_id_list(
        raw.get("required_material_ids"), materials_by_id
    )
    optional_material_ids = _normalize_material_id_list(
        raw.get("optional_material_ids"), materials_by_id
    )
    material_ids: List[str] = []
    for material_id in [*required_material_ids, *optional_material_ids]:
        if material_id not in material_ids:
            material_ids.append(material_id)
    if not scenario_type or not intent or not reader_need or not required_material_ids:
        return None
    if scenario_type == SCENARIO_TYPE_POINT and (
        len(required_material_ids) != 1 or optional_material_ids
    ):
        return None
    if scenario_type == SCENARIO_TYPE_SUMMARY and len(required_material_ids) > 3:
        return None
    materials = [materials_by_id[material_id] for material_id in material_ids]
    required_materials = [materials_by_id[material_id] for material_id in required_material_ids]
    available_image_ids = {
        image.image_id
        for material in materials
        for image in material.image_materials
    }
    required_image_ids: List[str] = []
    for value in raw.get("required_image_ids") or []:
        image_id = _safe_text(value)
        if image_id in available_image_ids and image_id not in required_image_ids:
            required_image_ids.append(image_id)
    for material in materials:
        if not any(image.image_id in required_image_ids for image in material.image_materials):
            continue
        if material.material_id not in required_material_ids:
            required_material_ids.append(material.material_id)
        optional_material_ids = [
            material_id for material_id in optional_material_ids
            if material_id != material.material_id
        ]
    material_ids = []
    for material_id in [*required_material_ids, *optional_material_ids]:
        if material_id not in material_ids:
            material_ids.append(material_id)
    if scenario_type == SCENARIO_TYPE_POINT and (
        len(required_material_ids) != 1 or optional_material_ids
    ):
        return None
    if scenario_type == SCENARIO_TYPE_SUMMARY and len(required_material_ids) > 3:
        return None
    required_materials = [materials_by_id[material_id] for material_id in required_material_ids]
    evidence_mode = _normalize_evidence_mode(
        raw.get("evidence_mode"),
        has_required_images=bool(required_image_ids),
    )
    if scenario_type == SCENARIO_TYPE_SUMMARY and len(required_materials) == 1:
        material = required_materials[0]
        image_supports_summary = len(material.image_materials) >= 2 or any(
            _has_list_signal(image.description)
            or (
                _structure_signal(image.description)
                and len(re.findall(r"[。！？!?；;]", image.description)) >= 2
            )
            for image in material.image_materials
        )
        if (
            len(material.source_chunk_indexes) < 2
            and not _has_list_signal(material.material_text)
            and not image_supports_summary
        ):
            return None
    source_indexes: List[int] = []
    for material in materials:
        for index in material.source_chunk_indexes:
            if index not in source_indexes:
                source_indexes.append(index)
    source_indexes.sort()
    if not source_indexes:
        return None
    anchor_chunk = chunks_by_index[source_indexes[0]]
    unit_type = (
        UNIT_TYPE_POINT_SCENARIO
        if scenario_type == SCENARIO_TYPE_POINT
        else UNIT_TYPE_SUMMARY_SCENARIO
    )
    raw_id = (
        f"{scenario_type}|||{intent}|||{'|'.join(material_ids)}|||"
        f"{evidence_mode}|||{'|'.join(required_image_ids)}"
    )
    unit_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
    material_ref_map = {
        _planning_material_ref(index): material.material_id
        for index, material in enumerate(materials, start=1)
    }
    image_ref_map: Dict[str, str] = {}
    for material in materials:
        for image in material.image_materials:
            if image.image_id not in image_ref_map.values():
                image_ref_map[_planning_image_ref(len(image_ref_map) + 1)] = image.image_id
    image_alias_by_id = {image_id: ref for ref, image_id in image_ref_map.items()}
    prompt_materials = [
        material.to_prompt_dict(
            material_ref=next(
                ref for ref, material_id in material_ref_map.items()
                if material_id == material.material_id
            ),
            image_refs_by_id=image_alias_by_id,
        )
        for material in materials
    ]
    unit_sections: List[str] = []
    for material in materials:
        is_required = material.material_id in required_material_ids
        role = "主材料" if is_required else "可选主材料"
        node_path = material.title_path or "未标注章节"
        unit_sections.append(
            f"【{role}节点路径】{node_path}\n"
            f"【{role}正文】\n{material.material_text}"
        )
    unit_text = "\n\n".join(unit_sections).strip()
    section_paths = {material.section_path for material in materials}
    section_path = materials[0].section_path if len(section_paths) == 1 else ""
    source_meta = dict(anchor_chunk)
    source_meta.update(
        {
            "qa_generation_unit_id": unit_id,
            "qa_generation_unit_type": unit_type,
            "qa_generation_unit_mode": scenario_type,
            "qa_generation_unit_source_chunk_indexes": source_indexes,
            "qa_generation_unit_material_ids": material_ids,
            "qa_generation_unit_required_material_ids": required_material_ids,
            "qa_generation_unit_optional_material_ids": optional_material_ids,
            "qa_generation_unit_evidence_mode": evidence_mode,
            "qa_generation_unit_required_image_ids": required_image_ids,
            "qa_generation_unit_subject_label": materials[0].subject_label,
            "qa_generation_unit_prompt_materials": prompt_materials,
            "qa_generation_unit_material_ref_map": material_ref_map,
            "qa_generation_unit_image_ref_map": image_ref_map,
            "qa_generation_unit_material_source_chunk_indexes": {
                material.material_id: list(material.source_chunk_indexes)
                for material in materials
            },
            "qa_generation_unit_scenario_intent": intent,
            "qa_generation_unit_reader_need": reader_need,
            "qa_generation_unit_title_path": materials[0].title_path,
            "qa_generation_unit_material_paths": [
                material.title_path for material in materials if material.title_path
            ],
            "qa_generation_unit_required_material_paths": [
                material.title_path
                for material in required_materials
                if material.title_path
            ],
            "qa_generation_unit_optional_material_paths": [
                material.title_path
                for material in materials
                if material.material_id in optional_material_ids and material.title_path
            ],
            "qa_generation_unit_text": unit_text,
        }
    )
    return GenerationUnit(
        unit_id=unit_id,
        unit_index=0,
        unit_type=unit_type,
        qa_mode=scenario_type,
        scenario_intent=intent,
        reader_need=reader_need,
        material_ids=material_ids,
        required_material_ids=required_material_ids,
        optional_material_ids=optional_material_ids,
        evidence_mode=evidence_mode,
        required_image_ids=required_image_ids,
        subject_label=materials[0].subject_label,
        prompt_materials=prompt_materials,
        material_ref_map=material_ref_map,
        image_ref_map=image_ref_map,
        material_source_chunk_indexes={
            material.material_id: list(material.source_chunk_indexes)
            for material in materials
        },
        anchor_chunk_index=source_indexes[0],
        source_chunk_indexes=source_indexes,
        section_path=section_path,
        title_path=materials[0].title_path,
        unit_text=unit_text,
        qa_budget=1,
        child_count=len(materials),
        usable_child_count=sum(material.usable for material in required_materials),
        quality_child_coverage=(
            sum(material.usable for material in required_materials) / len(required_materials)
        ),
        debug={
            "planner_reason": "llm_scenario",
            "raw_scenario": dict(raw),
            "material_paths": [material.title_path for material in materials if material.title_path],
            "required_material_paths": [
                material.title_path for material in required_materials if material.title_path
            ],
            "optional_material_paths": [
                material.title_path
                for material in materials
                if material.material_id in optional_material_ids and material.title_path
            ],
            "evidence_mode": evidence_mode,
            "required_image_ids": list(required_image_ids),
            "subject_label": materials[0].subject_label,
        },
        source_chunk_meta=source_meta,
    )


def _fallback_point_units(
    materials: Sequence[SectionMaterial],
    *,
    chunks_by_index: Dict[int, Dict[str, Any]],
    existing_units: Sequence[GenerationUnit],
    requested_count: int,
) -> List[GenerationUnit]:
    """Create evidence-bound point intents when the planner underfills its pool."""
    if requested_count <= 0:
        return []
    existing_material_ids = {
        unit.material_ids[0]
        for unit in existing_units
        if unit.qa_mode == SCENARIO_TYPE_POINT and len(unit.material_ids) == 1
    }
    fallback: List[GenerationUnit] = []
    for material in materials:
        if material.material_id in existing_material_ids:
            continue
        reader_need = f"了解{material.title_path or '本节'}的一个具体事实"
        raw = {
            "scenario_type": SCENARIO_TYPE_POINT,
            "intent": reader_need,
            "reader_need": reader_need,
            "required_material_ids": [material.material_id],
            "optional_material_ids": [],
            "evidence_mode": "text",
            "required_image_ids": [],
            "fallback_reason": "llm_point_pool_underfilled",
        }
        unit = _build_generation_unit(
            raw,
            materials_by_id={material.material_id: material},
            chunks_by_index=chunks_by_index,
        )
        if unit is not None:
            fallback.append(unit)
        if len(fallback) >= requested_count:
            break
    return fallback


def _required_visual_text(unit: GenerationUnit) -> str:
    required_ids = set(unit.required_image_ids)
    if not required_ids:
        return ""
    descriptions: List[str] = []
    for material in unit.prompt_materials:
        if not isinstance(material, dict):
            continue
        for image in material.get("image_materials") or []:
            if not isinstance(image, dict):
                continue
            image_ref = _safe_text(image.get("image_ref"))
            image_id = _safe_text(unit.image_ref_map.get(image_ref))
            if image_id in required_ids:
                descriptions.append(_safe_text(image.get("description")))
    return _collapse_text(" ".join(value for value in descriptions if value)).casefold()


def _select_scenarios(
    candidates: Sequence[GenerationUnit],
    *,
    requested_mode: str,
    requested_total: int,
    auto_summary_ratio: float,
) -> Tuple[List[GenerationUnit], List[GenerationUnit], Dict[str, int], Dict[str, int], int]:
    deduped: List[GenerationUnit] = []
    seen: set[Tuple[str, str, Tuple[str, ...]]] = set()
    for unit in candidates:
        key = (
            unit.qa_mode,
            _collapse_text(unit.scenario_intent).casefold(),
            tuple(unit.material_ids),
        )
        if key in seen:
            continue
        unit_text = _collapse_text(
            f"{unit.scenario_intent} {unit.reader_need}"
        ).casefold()
        unit_tokens = _token_set(unit_text)
        unit_visual_text = _required_visual_text(unit)
        semantically_duplicated = False
        for existing in deduped:
            existing_visual_text = _required_visual_text(existing)
            visual_duplicate = False
            if unit_visual_text and existing_visual_text:
                visual_containment = (
                    min(len(unit_visual_text), len(existing_visual_text)) >= 24
                    and (
                        unit_visual_text in existing_visual_text
                        or existing_visual_text in unit_visual_text
                    )
                )
                visual_duplicate = visual_containment or (
                    _jaccard(_token_set(unit_visual_text), _token_set(existing_visual_text)) >= 0.74
                )
            if not set(unit.material_ids).intersection(existing.material_ids) and not visual_duplicate:
                continue
            existing_text = _collapse_text(
                f"{existing.scenario_intent} {existing.reader_need}"
            ).casefold()
            containment = (
                min(len(unit_text), len(existing_text)) >= 10
                and (unit_text in existing_text or existing_text in unit_text)
            )
            if visual_duplicate or containment or _jaccard(unit_tokens, _token_set(existing_text)) >= 0.78:
                semantically_duplicated = True
                break
        if semantically_duplicated:
            continue
        seen.add(key)
        deduped.append(unit)
    pools = {
        SCENARIO_TYPE_POINT: [unit for unit in deduped if unit.qa_mode == SCENARIO_TYPE_POINT],
        SCENARIO_TYPE_SUMMARY: [unit for unit in deduped if unit.qa_mode == SCENARIO_TYPE_SUMMARY],
    }
    candidate_counts = {key: len(value) for key, value in pools.items()}
    total = max(0, int(requested_total))
    mode = _safe_text(requested_mode).lower() or "auto"
    if mode == SCENARIO_TYPE_POINT:
        selected = pools[SCENARIO_TYPE_POINT][:total]
    elif mode == SCENARIO_TYPE_SUMMARY:
        selected = pools[SCENARIO_TYPE_SUMMARY][:total]
    else:
        summary_target = min(total, max(0, round(total * max(0.0, min(1.0, auto_summary_ratio)))))
        point_target = max(0, total - summary_target)
        selected_summary = pools[SCENARIO_TYPE_SUMMARY][:summary_target]
        selected_point = pools[SCENARIO_TYPE_POINT][:point_target]
        remaining = total - len(selected_summary) - len(selected_point)
        if remaining > 0:
            selected_point.extend(pools[SCENARIO_TYPE_POINT][len(selected_point) : len(selected_point) + remaining])
            remaining = total - len(selected_summary) - len(selected_point)
        if remaining > 0:
            selected_summary.extend(
                pools[SCENARIO_TYPE_SUMMARY][len(selected_summary) : len(selected_summary) + remaining]
            )
        selected = selected_point + selected_summary
    selected.sort(key=lambda unit: (unit.anchor_chunk_index, unit.qa_mode, unit.unit_id))
    selected = [unit.with_index_and_budget(index, 1) for index, unit in enumerate(selected, 1)]
    selected_ids = {unit.unit_id for unit in selected}
    reserves = [unit for unit in deduped if unit.unit_id not in selected_ids]
    selected_counts = {
        SCENARIO_TYPE_POINT: sum(unit.qa_mode == SCENARIO_TYPE_POINT for unit in selected),
        SCENARIO_TYPE_SUMMARY: sum(unit.qa_mode == SCENARIO_TYPE_SUMMARY for unit in selected),
    }
    return selected, reserves, candidate_counts, selected_counts, max(0, len(deduped) - len(selected))


def plan_generation_units(
    document_chunks: Sequence[Dict[str, Any]],
    *,
    qa_total_limit: Optional[int],
    qa_per_chunk: int,
    qa_detail_mode: str,
    chunk_size: int,
    scenario_planner: Callable[..., Sequence[Dict[str, Any]]],
    auto_summary_ratio: float = DEFAULT_AUTO_SUMMARY_RATIO,
    scenario_planning_batch_chars: int = DEFAULT_SCENARIO_PLANNING_BATCH_CHARS,
    text_model_concurrency: int = 1,
) -> GenerationUnitPlan:
    chunks = [dict(chunk) for chunk in document_chunks if _text_for_quality(chunk)]
    graph = build_structure_graph(chunks)
    chunks_by_index = {int(chunk.get("chunk_index") or 0): chunk for chunk in chunks}
    quality: Dict[int, ChunkQuality] = {}
    for chunk in chunks:
        index = int(chunk.get("chunk_index") or 0)
        quality[index] = evaluate_chunk_quality(
            chunk,
            previous_chunk=chunks_by_index.get(graph.previous_by_index.get(index) or 0),
            next_chunk=chunks_by_index.get(graph.next_by_index.get(index) or 0),
        )
    materials = build_section_materials(chunks, quality)
    usable_materials = [material for material in materials if material.usable and material.material_text]
    if qa_total_limit is None:
        requested_total = max(1, int(qa_per_chunk or 1)) * len(usable_materials)
    else:
        requested_total = max(0, int(qa_total_limit))
    mode = _safe_text(qa_detail_mode).lower() or "auto"
    if mode == SCENARIO_TYPE_POINT:
        point_planning_count = requested_total
        summary_planning_count = 0
    elif mode == SCENARIO_TYPE_SUMMARY:
        point_planning_count = 0
        summary_planning_count = requested_total
    else:
        # Plan enough point capacity to absorb every missing summary slot. The
        # allocator still applies the target mix globally after both pools exist.
        point_planning_count = requested_total
        summary_planning_count = (
            max(1, round(requested_total * max(0.0, min(1.0, auto_summary_ratio))))
            if requested_total > 0
            else 0
        )
    configured_batch_chars = int(scenario_planning_batch_chars or 0)
    if configured_batch_chars == DEFAULT_SCENARIO_PLANNING_BATCH_CHARS and chunk_size:
        # Keep the default bounded for larger chunk targets while preserving
        # explicitly small test/developer budgets.
        effective_batch_chars = min(
            DEFAULT_SCENARIO_PLANNING_BATCH_CHARS,
            max(12000, int(chunk_size) * 24),
        )
    else:
        effective_batch_chars = max(1000, configured_batch_chars or DEFAULT_SCENARIO_PLANNING_BATCH_CHARS)
    planning_max_concurrency = max(1, int(text_model_concurrency or 1))

    planner_batches = {
        SCENARIO_TYPE_POINT: _batch_section_materials(
            usable_materials,
            max_batch_chars=effective_batch_chars,
            preserve_parent_neighborhood=False,
        ) if point_planning_count > 0 else [],
        SCENARIO_TYPE_SUMMARY: _batch_section_materials(
            usable_materials,
            max_batch_chars=effective_batch_chars,
            preserve_parent_neighborhood=True,
        ) if summary_planning_count > 0 else [],
    }

    # Point and Summary pools are independent planning jobs.  Run the two
    # pools together; the shared text-model client gate remains the single
    # authority for actual outbound request concurrency.
    raw_point_scenarios: List[Dict[str, Any]] = []
    point_batch_details: List[Dict[str, Any]] = []
    raw_summary_scenarios: List[Dict[str, Any]] = []
    summary_batch_details: List[Dict[str, Any]] = []

    def _run_planner_pool(scenario_type: str, requested_count: int):
        return _plan_scenario_pool(
            usable_materials,
            scenario_type=scenario_type,
            requested_count=requested_count,
            scenario_planner=scenario_planner,
            max_batch_chars=effective_batch_chars,
            max_concurrency=planning_max_concurrency,
        )

    planning_jobs = []
    if point_planning_count > 0:
        planning_jobs.append((SCENARIO_TYPE_POINT, point_planning_count))
    if summary_planning_count > 0:
        planning_jobs.append((SCENARIO_TYPE_SUMMARY, summary_planning_count))
    if len(planning_jobs) == 1:
        scenario_type, requested_count = planning_jobs[0]
        planned, details = _run_planner_pool(scenario_type, requested_count)
        if scenario_type == SCENARIO_TYPE_POINT:
            raw_point_scenarios, point_batch_details = planned, details
        else:
            raw_summary_scenarios, summary_batch_details = planned, details
    elif planning_jobs:
        with ThreadPoolExecutor(max_workers=len(planning_jobs)) as planner_executor:
            future_map = {
                planner_executor.submit(_run_planner_pool, scenario_type, requested_count): scenario_type
                for scenario_type, requested_count in planning_jobs
            }
            for future in as_completed(future_map):
                scenario_type = future_map[future]
                planned, details = future.result()
                if scenario_type == SCENARIO_TYPE_POINT:
                    raw_point_scenarios, point_batch_details = planned, details
                else:
                    raw_summary_scenarios, summary_batch_details = planned, details
    materials_by_id = {material.material_id: material for material in usable_materials}
    cross_batch_summary_scenarios = _merge_cross_batch_summary_candidates(
        raw_summary_scenarios,
        materials_by_id=materials_by_id,
    )
    raw_scenarios = [
        *raw_point_scenarios,
        *cross_batch_summary_scenarios,
        *raw_summary_scenarios,
    ]
    if cross_batch_summary_scenarios:
        summary_batch_details.append(
            {
                "batch_index": "global-merge",
                "batch_count": len(summary_batch_details),
                "scenario_type": SCENARIO_TYPE_SUMMARY,
                "requested_count": len(cross_batch_summary_scenarios),
                "returned_count": len(cross_batch_summary_scenarios),
                "material_count": len({
                    material_id
                    for item in cross_batch_summary_scenarios
                    for material_id in item.get("required_material_ids") or []
                }),
                "material_paths": [],
                "scenarios": [
                    {
                        "scenario_type": SCENARIO_TYPE_SUMMARY,
                        "scenario_intent": item.get("intent"),
                        "reader_need": item.get("reader_need"),
                        "required_material_paths": [
                            materials_by_id[material_id].title_path
                            for material_id in item.get("required_material_ids") or []
                            if material_id in materials_by_id and materials_by_id[material_id].title_path
                        ],
                        "optional_material_paths": [],
                        "material_paths": [
                            materials_by_id[material_id].title_path
                            for material_id in item.get("required_material_ids") or []
                            if material_id in materials_by_id and materials_by_id[material_id].title_path
                        ],
                    }
                    for item in cross_batch_summary_scenarios
                ],
            }
        )
    candidates = [
        unit
        for raw in raw_scenarios
        if isinstance(raw, dict)
        for unit in [_build_generation_unit(raw, materials_by_id=materials_by_id, chunks_by_index=chunks_by_index)]
        if unit is not None
    ]
    if mode in {"auto", SCENARIO_TYPE_POINT}:
        existing_point_count = sum(
            unit.qa_mode == SCENARIO_TYPE_POINT for unit in candidates
        )
        candidates.extend(
            _fallback_point_units(
                usable_materials,
                chunks_by_index=chunks_by_index,
                existing_units=candidates,
                requested_count=max(0, point_planning_count - existing_point_count),
            )
        )
    units, reserve_candidates, candidate_counts, selected_counts, dropped = _select_scenarios(
        candidates,
        requested_mode=qa_detail_mode,
        requested_total=requested_total,
        auto_summary_ratio=auto_summary_ratio,
    )
    reserve_target = min(
        requested_total,
        max(2, (requested_total + 4) // 5),
    ) if requested_total > 0 else 0
    reserve_units = list(reserve_candidates[:reserve_target])
    if mode in {"auto", SCENARIO_TYPE_POINT} and len(reserve_units) < reserve_target:
        reserve_units.extend(
            _fallback_point_units(
                usable_materials,
                chunks_by_index=chunks_by_index,
                existing_units=[*units, *reserve_units],
                requested_count=reserve_target - len(reserve_units),
            )
        )
    reserve_units = [
        unit.with_index_and_budget(len(units) + index, 1)
        for index, unit in enumerate(reserve_units[:reserve_target], start=1)
    ]
    return GenerationUnitPlan(
        units=units,
        reserve_units=reserve_units,
        section_materials=materials,
        chunk_quality=quality,
        graph=graph,
        requested_total_qa=requested_total,
        effective_total_qa=len(units),
        qa_total_limit=qa_total_limit,
        qa_detail_mode=_safe_text(qa_detail_mode).lower() or "auto",
        qa_per_chunk_fallback=max(1, int(qa_per_chunk or 1)),
        dropped_unit_count_by_budget=dropped,
        scenario_candidates_by_type=candidate_counts,
        scenario_selected_by_type=selected_counts,
        scenario_planner_calls_by_type={
            key: sum(
                count > 0
                for count in _allocate_planning_counts(
                    value,
                    point_planning_count
                    if key == SCENARIO_TYPE_POINT
                    else summary_planning_count,
                )
            )
            for key, value in planner_batches.items()
        },
        scenario_planner_batches_by_type={
            key: [len(batch) for batch in value]
            for key, value in planner_batches.items()
        },
        scenario_planner_batch_details={
            SCENARIO_TYPE_POINT: point_batch_details,
            SCENARIO_TYPE_SUMMARY: summary_batch_details,
        },
        scenario_planning_batch_chars=effective_batch_chars,
        text_model_concurrency=planning_max_concurrency,
    )


__all__ = [
    "ChunkQuality",
    "DEFAULT_AUTO_SUMMARY_RATIO",
    "DEFAULT_SCENARIO_PLANNING_BATCH_CHARS",
    "GenerationUnit",
    "GenerationUnitPlan",
    "ImageMaterial",
    "QUALITY_STATUS_CONTEXT_ONLY",
    "QUALITY_STATUS_DROP",
    "QUALITY_STATUS_USABLE",
    "SCENARIO_TYPE_POINT",
    "SCENARIO_TYPE_SUMMARY",
    "SectionMaterial",
    "StructureGraph",
    "UNIT_TYPE_POINT_SCENARIO",
    "UNIT_TYPE_SUMMARY_SCENARIO",
    "build_section_materials",
    "build_structure_graph",
    "evaluate_chunk_quality",
    "plan_generation_units",
]
