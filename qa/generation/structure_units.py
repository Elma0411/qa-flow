"""Plan LLM-backed QA scenarios from logical document sections."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, replace
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
    usable: bool
    quality_score: float

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "title": self.title_path,
            "content": self.material_text,
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

    def summary(self) -> Dict[str, Any]:
        quality_counts: Dict[str, int] = defaultdict(int)
        for quality in self.chunk_quality.values():
            quality_counts[quality.status] += 1
        return {
            "chunks_total": self.graph.chunk_count,
            "section_materials_total": len(self.section_materials),
            "generation_units_total": len(self.units),
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


def _render_material_text(chunks: Sequence[Dict[str, Any]]) -> str:
    pieces: List[str] = []
    for chunk in sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0)):
        text = _text_for_quality(chunk).strip()
        if int(chunk.get("fragment_index") or 1) > 1:
            text = _strip_repeated_fragment_heading(text)
        if not text:
            continue
        if pieces and text == pieces[-1]:
            continue
        pieces.append(text)
    return "\n\n".join(pieces).strip()


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
    scenario_planner: Callable[[Sequence[SectionMaterial], int, str], Sequence[Dict[str, Any]]],
    max_batch_chars: int,
) -> List[Dict[str, Any]]:
    batches = _batch_section_materials(
        materials,
        max_batch_chars=max_batch_chars,
        preserve_parent_neighborhood=scenario_type == SCENARIO_TYPE_SUMMARY,
    )
    counts = _allocate_planning_counts(batches, requested_count)
    planned: List[Dict[str, Any]] = []
    for batch, count in zip(batches, counts):
        if count <= 0:
            continue
        planned.extend(
            item
            for item in scenario_planner(batch, count, scenario_type)
            if isinstance(item, dict)
        )
    return planned


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
                usable=usable_count > 0,
                quality_score=sum(chunk_quality[index].score for index in indexes) / max(1, len(indexes)),
            )
        )
    return materials


def _normalize_scenario_type(value: Any) -> str:
    normalized = _safe_text(value).lower()
    if normalized in {"point", "single", "single_hop", "单点", "单点题"}:
        return SCENARIO_TYPE_POINT
    if normalized in {"summary", "multi", "multi_hop", "总结", "总结题"}:
        return SCENARIO_TYPE_SUMMARY
    return ""


def _build_generation_unit(
    raw: Dict[str, Any],
    *,
    materials_by_id: Dict[str, SectionMaterial],
    chunks_by_index: Dict[int, Dict[str, Any]],
) -> Optional[GenerationUnit]:
    scenario_type = _normalize_scenario_type(raw.get("scenario_type") or raw.get("type"))
    intent = _safe_text(raw.get("intent"))
    reader_need = _safe_text(raw.get("reader_need")) or intent
    material_ids = []
    for raw_id in raw.get("material_ids") or []:
        material_id = _safe_text(raw_id)
        if material_id in materials_by_id and material_id not in material_ids:
            material_ids.append(material_id)
    if not scenario_type or not intent or not reader_need or not material_ids:
        return None
    if scenario_type == SCENARIO_TYPE_POINT and len(material_ids) != 1:
        return None
    materials = [materials_by_id[material_id] for material_id in material_ids]
    if scenario_type == SCENARIO_TYPE_SUMMARY and len(materials) == 1:
        material = materials[0]
        if len(material.source_chunk_indexes) < 2 and not _has_list_signal(material.material_text):
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
    raw_id = f"{scenario_type}|||{intent}|||{'|'.join(material_ids)}"
    unit_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
    unit_text = "\n\n".join(material.material_text for material in materials).strip()
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
            "qa_generation_unit_material_source_chunk_indexes": {
                material.material_id: list(material.source_chunk_indexes)
                for material in materials
            },
            "qa_generation_unit_scenario_intent": intent,
            "qa_generation_unit_reader_need": reader_need,
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
        usable_child_count=sum(material.usable for material in materials),
        quality_child_coverage=sum(material.usable for material in materials) / len(materials),
        debug={"planner_reason": "llm_scenario", "raw_scenario": dict(raw)},
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
            "material_ids": [material.material_id],
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


def _select_scenarios(
    candidates: Sequence[GenerationUnit],
    *,
    requested_mode: str,
    requested_total: int,
    auto_summary_ratio: float,
) -> Tuple[List[GenerationUnit], Dict[str, int], Dict[str, int], int]:
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
    selected_counts = {
        SCENARIO_TYPE_POINT: sum(unit.qa_mode == SCENARIO_TYPE_POINT for unit in selected),
        SCENARIO_TYPE_SUMMARY: sum(unit.qa_mode == SCENARIO_TYPE_SUMMARY for unit in selected),
    }
    return selected, candidate_counts, selected_counts, max(0, len(deduped) - len(selected))


def plan_generation_units(
    document_chunks: Sequence[Dict[str, Any]],
    *,
    qa_total_limit: Optional[int],
    qa_per_chunk: int,
    qa_detail_mode: str,
    chunk_size: int,
    scenario_planner: Callable[[Sequence[SectionMaterial], int, str], Sequence[Dict[str, Any]]],
    auto_summary_ratio: float = DEFAULT_AUTO_SUMMARY_RATIO,
    scenario_planning_batch_chars: int = DEFAULT_SCENARIO_PLANNING_BATCH_CHARS,
) -> GenerationUnitPlan:
    del chunk_size
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
    raw_scenarios = _plan_scenario_pool(
        usable_materials,
        scenario_type=SCENARIO_TYPE_POINT,
        requested_count=point_planning_count,
        scenario_planner=scenario_planner,
        max_batch_chars=scenario_planning_batch_chars,
    )
    raw_scenarios.extend(
        _plan_scenario_pool(
            usable_materials,
            scenario_type=SCENARIO_TYPE_SUMMARY,
            requested_count=summary_planning_count,
            scenario_planner=scenario_planner,
            max_batch_chars=scenario_planning_batch_chars,
        )
    )
    planner_batches = {
        SCENARIO_TYPE_POINT: _batch_section_materials(
            usable_materials,
            max_batch_chars=scenario_planning_batch_chars,
            preserve_parent_neighborhood=False,
        ) if point_planning_count > 0 else [],
        SCENARIO_TYPE_SUMMARY: _batch_section_materials(
            usable_materials,
            max_batch_chars=scenario_planning_batch_chars,
            preserve_parent_neighborhood=True,
        ) if summary_planning_count > 0 else [],
    }
    materials_by_id = {material.material_id: material for material in usable_materials}
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
    units, candidate_counts, selected_counts, dropped = _select_scenarios(
        candidates,
        requested_mode=qa_detail_mode,
        requested_total=requested_total,
        auto_summary_ratio=auto_summary_ratio,
    )
    return GenerationUnitPlan(
        units=units,
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
    )


__all__ = [
    "ChunkQuality",
    "DEFAULT_AUTO_SUMMARY_RATIO",
    "DEFAULT_SCENARIO_PLANNING_BATCH_CHARS",
    "GenerationUnit",
    "GenerationUnitPlan",
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
