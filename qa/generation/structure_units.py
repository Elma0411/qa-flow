# 文件作用：基于文档结构规划 QA generation unit，并执行轻量 chunk 质量门控。
# 关联说明：被 text_to_qa_pipeline 调用，把 leaf chunk 转成更适合出题的 generation unit。

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


QUALITY_STATUS_USABLE = "usable"
QUALITY_STATUS_CONTEXT_ONLY = "context_only"
QUALITY_STATUS_DROP = "drop"

UNIT_TYPE_LEAF = "leaf"
UNIT_TYPE_SECTION = "section"
UNIT_TYPE_VIRTUAL_PARENT = "virtual_parent"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


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
            continue
        if len(segment) <= 2:
            tokens.add(segment)
            continue
        tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _line_texts(text: str) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _has_sentence_signal(text: str) -> bool:
    if re.search(r"[。！？!?；;:：]", text):
        return True
    return bool(re.search(r"\b(is|are|means|includes|shall|must|should|requires?)\b", text, re.I))


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
    if re.search(r"\d{4}[-年/]\d{1,2}|[0-9]+(?:\.[0-9]+)?\s*(?:%|元|万|小时|天|kg|km|m|GB|MB)", text, re.I):
        return True
    if re.search(r"(定义为|是指|包括|包含|适用于|应当|必须|不得|要求|标准|条件|流程|步骤|范围)", text):
        return True
    return bool(re.search(r"\b(define[sd]?|include[sd]?|require[sd]?|applies to|condition|process|step|standard)\b", text, re.I))


def _looks_placeholder(text: str) -> bool:
    clean = _collapse_text(text)
    if not clean:
        return True
    placeholder_hit = bool(
        re.search(
            r"(图片|图像|图示|截图|附件|二维码|扫描件|占位|见图|见附件|image|figure|attachment|placeholder)",
            clean,
            re.I,
        )
    )
    return placeholder_hit and len(clean) <= 160 and not _has_fact_signal(clean)


def _looks_title_only(text: str, title_path: str) -> bool:
    clean = _collapse_text(text)
    if not clean:
        return True
    lines = _line_texts(text)
    if len(lines) > 2:
        return False
    title_tail = ""
    if title_path:
        title_tail = title_path.replace("＞", ">").split(">")[-1].strip()
    if title_tail and clean == title_tail:
        return True
    if len(clean) <= 48 and not _has_sentence_signal(clean) and not _has_list_signal(clean):
        return True
    return bool(re.fullmatch(r"(第?[一二三四五六七八九十0-9]+[章节条部分篇].{0,40}|[0-9.、\s]{1,12}\S{0,40})", clean))


def _looks_table_fragment(text: str) -> bool:
    lines = _line_texts(text)
    if not lines:
        return False
    tableish_lines = 0
    for line in lines:
        if "|" in line or "\t" in line:
            tableish_lines += 1
            continue
        if len(re.split(r"\s{2,}|,|，", line)) >= 3 and not _has_sentence_signal(line):
            tableish_lines += 1
    if tableish_lines < max(1, len(lines) // 2):
        return False
    return not _has_fact_signal(text)


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
            r"(流程|步骤|条件|材料|范围|规则|标准|要求|对比|分类|组成|清单|目录|适用|定义|process|step|condition|rule|standard|requirement|compare|category|definition)",
            text,
            re.I,
        )
    )


def _adjacent_duplicate_ratio(
    chunk: Dict[str, Any],
    previous_chunk: Optional[Dict[str, Any]],
    next_chunk: Optional[Dict[str, Any]],
) -> float:
    tokens = _token_set(_text_for_quality(chunk))
    if not tokens:
        return 0.0
    ratios = []
    for neighbor in (previous_chunk, next_chunk):
        if not neighbor:
            continue
        ratios.append(_jaccard(tokens, _token_set(_text_for_quality(neighbor))))
    return max(ratios or [0.0])


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
            "parent_groups": {
                key: list(value)
                for key, value in self.children_by_parent.items()
                if key
            },
        }


@dataclass(frozen=True)
class GenerationUnit:
    unit_id: str
    unit_index: int
    unit_type: str
    qa_mode: str
    anchor_chunk_index: int
    source_chunk_indexes: List[int]
    parent_index_path: str
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
            "anchor_chunk_index": self.anchor_chunk_index,
            "source_chunk_indexes": list(self.source_chunk_indexes),
            "parent_index_path": self.parent_index_path,
            "title_path": self.title_path,
            "unit_text": self.unit_text,
            "qa_budget": self.qa_budget,
            "child_count": self.child_count,
            "usable_child_count": self.usable_child_count,
            "quality_child_coverage": self.quality_child_coverage,
            "debug": dict(self.debug),
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "unit_index": self.unit_index,
            "unit_type": self.unit_type,
            "qa_mode": self.qa_mode,
            "anchor_chunk_index": self.anchor_chunk_index,
            "source_chunk_indexes": list(self.source_chunk_indexes),
            "parent_index_path": self.parent_index_path,
            "title_path": self.title_path,
            "qa_budget": self.qa_budget,
            "child_count": self.child_count,
            "usable_child_count": self.usable_child_count,
            "quality_child_coverage": round(float(self.quality_child_coverage), 4),
            "unit_char_count": len(self.unit_text),
            "debug": dict(self.debug),
        }


@dataclass(frozen=True)
class GenerationUnitPlan:
    units: List[GenerationUnit]
    chunk_quality: Dict[int, ChunkQuality]
    graph: StructureGraph
    requested_total_qa: int
    effective_total_qa: int
    qa_total_limit: Optional[int]
    qa_detail_mode: str
    qa_per_chunk_fallback: int
    dropped_unit_count_by_budget: int

    def summary(self) -> Dict[str, Any]:
        quality_counts: Dict[str, int] = defaultdict(int)
        for quality in self.chunk_quality.values():
            quality_counts[quality.status] += 1
        unit_type_counts: Dict[str, int] = defaultdict(int)
        mode_counts: Dict[str, int] = defaultdict(int)
        for unit in self.units:
            unit_type_counts[unit.unit_type] += 1
            mode_counts[unit.qa_mode] += 1
        return {
            "chunks_total": self.graph.chunk_count,
            "generation_units_total": len(self.units),
            "requested_total_qa": self.requested_total_qa,
            "effective_total_qa": self.effective_total_qa,
            "qa_total_limit": self.qa_total_limit,
            "qa_detail_mode": self.qa_detail_mode,
            "qa_per_chunk_fallback": self.qa_per_chunk_fallback,
            "dropped_unit_count_by_budget": self.dropped_unit_count_by_budget,
            "quality_counts": dict(quality_counts),
            "unit_type_counts": dict(unit_type_counts),
            "mode_counts": dict(mode_counts),
        }


def build_structure_graph(document_chunks: Sequence[Dict[str, Any]]) -> StructureGraph:
    ordered = [
        int(chunk.get("chunk_index") or index)
        for index, chunk in enumerate(document_chunks, start=1)
    ]
    previous_by_index: Dict[int, Optional[int]] = {}
    next_by_index: Dict[int, Optional[int]] = {}
    for pos, chunk_index in enumerate(ordered):
        previous_by_index[chunk_index] = ordered[pos - 1] if pos > 0 else None
        next_by_index[chunk_index] = ordered[pos + 1] if pos + 1 < len(ordered) else None

    children_by_parent: Dict[str, List[int]] = defaultdict(list)
    for chunk in document_chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        parent_key = _safe_text(chunk.get("parent_index_path"))
        if not parent_key:
            continue
        children_by_parent[parent_key].append(chunk_index)
    return StructureGraph(
        chunk_count=len(document_chunks),
        children_by_parent={key: value for key, value in children_by_parent.items()},
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
    char_count = len(clean)
    title_path = _safe_text(chunk.get("title_path"))
    has_title_path = bool(title_path)
    has_fact_signal = _has_fact_signal(text)
    has_structure_signal = _structure_signal(text) or _structure_signal(title_path)
    title_only = _looks_title_only(text, title_path)
    placeholder = _looks_placeholder(text)
    table_fragment = _looks_table_fragment(text)
    duplicate_ratio = _adjacent_duplicate_ratio(chunk, previous_chunk, next_chunk)
    noisy_ratio = _symbol_digit_ratio(text)

    score = 1.0
    reasons: List[str] = []
    if char_count < 80 and not has_title_path and not _has_list_signal(text):
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
    if noisy_ratio >= 0.48 and char_count < 320:
        score -= 0.15
        reasons.append("symbol_digit_ratio_abnormal")
    if has_title_path:
        score += 0.10
    if has_structure_signal or has_fact_signal:
        score += 0.10
    if 0.18 <= duplicate_ratio <= 0.72 and has_title_path:
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
        char_count=char_count,
        duplicate_ratio=duplicate_ratio,
        has_title_path=has_title_path,
        has_structure_signal=has_structure_signal,
        has_fact_signal=has_fact_signal,
        symbol_digit_ratio=noisy_ratio,
    )


def _render_unit_text(chunks: Sequence[Dict[str, Any]], *, max_chars: int) -> str:
    parts: List[str] = []
    remaining = max(1000, int(max_chars))
    for chunk in chunks:
        title_path = _safe_text(chunk.get("title_path"))
        text = _text_for_quality(chunk)
        chunk_id = _safe_text(chunk.get("chunk_id"))
        chunk_index = int(chunk.get("chunk_index") or 0)
        header = f"chunk_index：{chunk_index}\nchunk_id：{chunk_id}"
        if title_path:
            header += f"\ntitle_path：{title_path}"
        rendered = f"{header}\n内容：{text}".strip()
        if len(rendered) > remaining and not parts:
            parts.append(rendered[:remaining].rstrip())
            break
        if len(rendered) > remaining:
            break
        parts.append(rendered)
        remaining -= len(rendered)
    return "\n\n".join(parts).strip()


def _make_unit_id(unit_type: str, chunks: Sequence[Dict[str, Any]]) -> str:
    raw = unit_type + "|||" + "|||".join(
        f"{chunk.get('chunk_id') or ''}:{chunk.get('chunk_index') or ''}"
        for chunk in chunks
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _resolve_unit_mode(requested_mode: str, unit_type: str) -> str:
    mode = _safe_text(requested_mode).lower() or "point"
    if mode in {"point", "summary"}:
        return mode
    if unit_type in {UNIT_TYPE_SECTION, UNIT_TYPE_VIRTUAL_PARENT}:
        return "summary"
    return "point"


def _source_meta_for_unit(
    anchor_chunk: Dict[str, Any],
    *,
    unit_id: str,
    unit_type: str,
    qa_mode: str,
    source_chunk_indexes: Sequence[int],
    unit_text: str,
) -> Dict[str, Any]:
    meta = dict(anchor_chunk)
    meta["qa_generation_unit_id"] = unit_id
    meta["qa_generation_unit_type"] = unit_type
    meta["qa_generation_unit_mode"] = qa_mode
    meta["qa_generation_unit_source_chunk_indexes"] = list(source_chunk_indexes)
    meta["qa_generation_unit_text"] = unit_text
    return meta


def _build_generation_unit(
    *,
    unit_type: str,
    chunks: Sequence[Dict[str, Any]],
    chunk_quality: Dict[int, ChunkQuality],
    requested_mode: str,
    max_unit_chars: int,
    reason: str,
    virtual_child_count: Optional[int] = None,
) -> GenerationUnit:
    ordered_chunks = sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0))
    anchor_chunk = next(
        (
            chunk
            for chunk in ordered_chunks
            if chunk_quality.get(int(chunk.get("chunk_index") or 0), None)
            and chunk_quality[int(chunk.get("chunk_index") or 0)].status == QUALITY_STATUS_USABLE
        ),
        ordered_chunks[0],
    )
    source_indexes = [int(chunk.get("chunk_index") or 0) for chunk in ordered_chunks]
    unit_id = _make_unit_id(unit_type, ordered_chunks)
    qa_mode = _resolve_unit_mode(requested_mode, unit_type)
    unit_text = _render_unit_text(ordered_chunks, max_chars=max_unit_chars)
    usable_child_count = sum(
        1
        for chunk in ordered_chunks
        if chunk_quality.get(int(chunk.get("chunk_index") or 0))
        and chunk_quality[int(chunk.get("chunk_index") or 0)].status == QUALITY_STATUS_USABLE
    )
    child_count = int(virtual_child_count or len(ordered_chunks))
    quality_child_coverage = usable_child_count / max(1, len(ordered_chunks))
    title_path = _safe_text(anchor_chunk.get("title_path"))
    parent_index_path = _safe_text(anchor_chunk.get("parent_index_path"))
    source_meta = _source_meta_for_unit(
        anchor_chunk,
        unit_id=unit_id,
        unit_type=unit_type,
        qa_mode=qa_mode,
        source_chunk_indexes=source_indexes,
        unit_text=unit_text,
    )
    return GenerationUnit(
        unit_id=unit_id,
        unit_index=0,
        unit_type=unit_type,
        qa_mode=qa_mode,
        anchor_chunk_index=int(anchor_chunk.get("chunk_index") or 0),
        source_chunk_indexes=source_indexes,
        parent_index_path=parent_index_path,
        title_path=title_path,
        unit_text=unit_text,
        qa_budget=0,
        child_count=child_count,
        usable_child_count=usable_child_count,
        quality_child_coverage=quality_child_coverage,
        debug={
            "planner_reason": reason,
            "source_chunk_quality": {
                int(chunk.get("chunk_index") or 0): chunk_quality[int(chunk.get("chunk_index") or 0)].to_dict()
                for chunk in ordered_chunks
                if int(chunk.get("chunk_index") or 0) in chunk_quality
            },
        },
        source_chunk_meta=source_meta,
    )


def _count_virtual_children(text: str) -> int:
    lines = _line_texts(text)
    structured = [
        line
        for line in lines
        if re.match(r"^\s*(?:[-*•]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*\S+", line)
    ]
    paragraph_count = len([part for part in re.split(r"\n\s*\n", text) if part.strip()])
    return max(len(structured), paragraph_count)


def _unit_potential(unit: GenerationUnit) -> int:
    if unit.unit_type == UNIT_TYPE_SECTION:
        return max(1, min(3, unit.usable_child_count or unit.child_count))
    if unit.unit_type == UNIT_TYPE_VIRTUAL_PARENT:
        return 2 if unit.child_count >= 2 else 1
    return 1


def _allocate_budget(
    units: Sequence[GenerationUnit],
    requested_total: int,
) -> Tuple[List[GenerationUnit], int, int]:
    if requested_total <= 0 or not units:
        return [], 0, len(units)

    ordered_units = sorted(units, key=lambda unit: (unit.anchor_chunk_index, unit.unit_type))
    selected = list(ordered_units[:requested_total])
    dropped = max(0, len(ordered_units) - len(selected))
    budgets = {unit.unit_id: 1 for unit in selected}
    remaining = max(0, requested_total - len(selected))

    while remaining > 0:
        changed = False
        for unit in sorted(
            selected,
            key=lambda item: (
                0 if item.unit_type in {UNIT_TYPE_SECTION, UNIT_TYPE_VIRTUAL_PARENT} else 1,
                item.anchor_chunk_index,
            ),
        ):
            if budgets[unit.unit_id] >= _unit_potential(unit):
                continue
            budgets[unit.unit_id] += 1
            remaining -= 1
            changed = True
            if remaining <= 0:
                break
        if not changed:
            break

    planned = [
        unit.with_index_and_budget(index, budgets.get(unit.unit_id, 0))
        for index, unit in enumerate(selected, start=1)
    ]
    effective_total = sum(max(0, int(unit.qa_budget)) for unit in planned)
    return planned, effective_total, dropped


def plan_generation_units(
    document_chunks: Sequence[Dict[str, Any]],
    *,
    qa_total_limit: Optional[int],
    qa_per_chunk: int,
    qa_detail_mode: str,
    chunk_size: int,
    max_unit_chars: int,
) -> GenerationUnitPlan:
    chunks = [dict(chunk) for chunk in document_chunks if _text_for_quality(chunk)]
    graph = build_structure_graph(chunks)
    chunks_by_index = {
        int(chunk.get("chunk_index") or 0): chunk
        for chunk in chunks
        if int(chunk.get("chunk_index") or 0) > 0
    }

    quality: Dict[int, ChunkQuality] = {}
    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        previous_chunk = chunks_by_index.get(graph.previous_by_index.get(chunk_index) or 0)
        next_chunk = chunks_by_index.get(graph.next_by_index.get(chunk_index) or 0)
        quality[chunk_index] = evaluate_chunk_quality(
            chunk,
            previous_chunk=previous_chunk,
            next_chunk=next_chunk,
        )

    usable_leaf_count = sum(
        1 for item in quality.values() if item.status == QUALITY_STATUS_USABLE
    )
    if qa_total_limit is None:
        requested_total = max(1, int(qa_per_chunk or 1)) * max(1, usable_leaf_count or len(chunks))
    else:
        requested_total = max(0, int(qa_total_limit))

    candidates: List[GenerationUnit] = []
    covered: set[int] = set()
    for parent_key, indexes in sorted(
        graph.children_by_parent.items(),
        key=lambda item: min(item[1]) if item[1] else 10**9,
    ):
        if len(indexes) < 2:
            continue
        group_chunks = [
            chunks_by_index[index]
            for index in indexes
            if index in chunks_by_index
            and quality.get(index)
            and quality[index].status != QUALITY_STATUS_DROP
        ]
        usable_count = sum(
            1
            for chunk in group_chunks
            if quality[int(chunk.get("chunk_index") or 0)].status == QUALITY_STATUS_USABLE
        )
        if len(group_chunks) < 2 or usable_count < 2:
            continue
        group_chars = sum(len(_collapse_text(_text_for_quality(chunk))) for chunk in group_chunks)
        if group_chars > max(1000, int(max_unit_chars)):
            continue
        candidates.append(
            _build_generation_unit(
                unit_type=UNIT_TYPE_SECTION,
                chunks=group_chunks,
                chunk_quality=quality,
                requested_mode=qa_detail_mode,
                max_unit_chars=max_unit_chars,
                reason=f"same_parent_group:{parent_key}",
            )
        )
        covered.update(int(chunk.get("chunk_index") or 0) for chunk in group_chunks)

    long_threshold = max(900, int(chunk_size or 600) * 2)
    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        if chunk_index in covered:
            continue
        if quality.get(chunk_index) and quality[chunk_index].status != QUALITY_STATUS_USABLE:
            continue
        text = _text_for_quality(chunk)
        virtual_children = _count_virtual_children(text)
        if len(_collapse_text(text)) < long_threshold or virtual_children < 2:
            continue
        candidates.append(
            _build_generation_unit(
                unit_type=UNIT_TYPE_VIRTUAL_PARENT,
                chunks=[chunk],
                chunk_quality=quality,
                requested_mode=qa_detail_mode,
                max_unit_chars=max_unit_chars,
                reason="long_structured_chunk",
                virtual_child_count=virtual_children,
            )
        )
        covered.add(chunk_index)

    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index") or 0)
        if chunk_index in covered:
            continue
        if quality.get(chunk_index) and quality[chunk_index].status != QUALITY_STATUS_USABLE:
            continue
        candidates.append(
            _build_generation_unit(
                unit_type=UNIT_TYPE_LEAF,
                chunks=[chunk],
                chunk_quality=quality,
                requested_mode=qa_detail_mode,
                max_unit_chars=max_unit_chars,
                reason="usable_leaf_chunk",
            )
        )

    planned_units, effective_total, dropped = _allocate_budget(candidates, requested_total)
    return GenerationUnitPlan(
        units=planned_units,
        chunk_quality=quality,
        graph=graph,
        requested_total_qa=requested_total,
        effective_total_qa=effective_total,
        qa_total_limit=qa_total_limit,
        qa_detail_mode=_safe_text(qa_detail_mode).lower() or "point",
        qa_per_chunk_fallback=max(1, int(qa_per_chunk or 1)),
        dropped_unit_count_by_budget=dropped,
    )


__all__ = [
    "ChunkQuality",
    "GenerationUnit",
    "GenerationUnitPlan",
    "QUALITY_STATUS_CONTEXT_ONLY",
    "QUALITY_STATUS_DROP",
    "QUALITY_STATUS_USABLE",
    "StructureGraph",
    "UNIT_TYPE_LEAF",
    "UNIT_TYPE_SECTION",
    "UNIT_TYPE_VIRTUAL_PARENT",
    "build_structure_graph",
    "evaluate_chunk_quality",
    "plan_generation_units",
]
