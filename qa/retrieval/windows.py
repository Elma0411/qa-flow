"""Deterministic structure-window construction after atomic reranking."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import EvidenceChunk, EvidenceWindow


_LEFT_CONTEXT_RE = re.compile(
    r"(?:上述|前述|以上|该(?:项|条|款|章|节|办法|规定|要求|材料|流程|条件|情况)|"
    r"此(?:项|条|款|章|节|办法|规定|要求|材料|流程|条件|情况)|"
    r"其(?:中|他|余)|前者|后者|除外|例外|补充|继续|另外|此外|同时)"
)
_RIGHT_CONTEXT_RE = re.compile(
    r"(?:如下|下列|包括|包含|分为|分别为|具体为|见下|详见|例如|即|：|:)$"
)


class EvidenceWindowBuilder:
    def __init__(self, chunks: Sequence[EvidenceChunk]) -> None:
        self.chunks = sorted(chunks, key=lambda chunk: (chunk.chunk_index, chunk.chunk_id))
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.by_fragment_group: Dict[str, List[EvidenceChunk]] = defaultdict(list)
        self.by_section: Dict[str, List[EvidenceChunk]] = defaultdict(list)
        for chunk in self.chunks:
            self.by_fragment_group[chunk.fragment_group_id].append(chunk)
            self.by_section[chunk.section_path].append(chunk)
        for values in self.by_fragment_group.values():
            values.sort(key=lambda chunk: (chunk.fragment_index, chunk.chunk_index))
        for values in self.by_section.values():
            values.sort(key=lambda chunk: (chunk.section_chunk_index, chunk.chunk_index))

    @staticmethod
    def _is_contiguous(left: EvidenceChunk, right: EvidenceChunk) -> bool:
        return right.section_chunk_index == left.section_chunk_index + 1

    @staticmethod
    def _dependency_directions(chunk: EvidenceChunk) -> Tuple[bool, bool]:
        text = str(chunk.text or "").strip()
        needs_left = bool(_LEFT_CONTEXT_RE.search(text))
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        needs_right = bool(_RIGHT_CONTEXT_RE.search(text) or _RIGHT_CONTEXT_RE.search(first_line))
        return needs_left, needs_right

    def _expand_fragment_groups(self, chunks: Sequence[EvidenceChunk]) -> List[EvidenceChunk]:
        expanded: List[EvidenceChunk] = []
        seen_groups: set[str] = set()
        for chunk in chunks:
            group_id = chunk.fragment_group_id
            if group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            expanded.extend(self.by_fragment_group.get(group_id) or [chunk])
        return expanded

    def _contiguous_anchor_runs(
        self,
        section_anchors: Sequence[EvidenceChunk],
    ) -> List[List[EvidenceChunk]]:
        ordered = sorted(
            section_anchors,
            key=lambda chunk: (chunk.section_chunk_index, chunk.chunk_index, chunk.chunk_id),
        )
        runs: List[List[EvidenceChunk]] = []
        for anchor in ordered:
            if not runs or not self._is_contiguous(runs[-1][-1], anchor):
                runs.append([anchor])
            else:
                runs[-1].append(anchor)
        return runs

    @staticmethod
    def _window_id(chunk_ids: Iterable[str], reason: str) -> str:
        payload = reason + "|||" + "|||".join(chunk_ids)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _make_window(
        self,
        chunks: Sequence[EvidenceChunk],
        *,
        reason: str,
        anchor_chunk_ids: Sequence[str],
        includes_parent_body: bool = False,
    ) -> EvidenceWindow:
        ordered: List[EvidenceChunk] = []
        seen: set[str] = set()
        for chunk in sorted(chunks, key=lambda item: (item.chunk_index, item.chunk_id)):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ordered.append(chunk)
        chunk_ids = tuple(chunk.chunk_id for chunk in ordered)
        text_parts: List[str] = []
        for chunk in ordered:
            if chunk.title_path and (not text_parts or chunk.title_path != ordered[0].title_path):
                text_parts.append(f"标题路径：{chunk.title_path}\n{chunk.text}".strip())
            else:
                text_parts.append(chunk.text)
        return EvidenceWindow(
            window_id=self._window_id(chunk_ids, reason),
            chunk_ids=chunk_ids,
            reason=reason,
            text="\n\n".join(part for part in text_parts if part).strip(),
            title_path=ordered[0].title_path if ordered else "",
            anchor_chunk_ids=tuple(anchor_chunk_ids),
            includes_parent_body=includes_parent_body,
        )

    def build(self, *, query: str, ranked_chunk_ids: Sequence[str]) -> List[EvidenceWindow]:
        del query  # structure construction is deterministic; reranking uses the query later
        anchors = [self.by_id[chunk_id] for chunk_id in ranked_chunk_ids if chunk_id in self.by_id]
        windows: List[EvidenceWindow] = []
        emitted: set[Tuple[str, ...]] = set()

        def emit(chunks: Sequence[EvidenceChunk], *, reason: str, anchor_ids: Sequence[str], parent: bool = False) -> None:
            window = self._make_window(
                chunks,
                reason=reason,
                anchor_chunk_ids=anchor_ids,
                includes_parent_body=parent,
            )
            if not window.chunk_ids or window.chunk_ids in emitted:
                return
            emitted.add(window.chunk_ids)
            windows.append(window)

        # A physical split is indivisible at evidence time.
        for anchor in anchors:
            fragments = self.by_fragment_group.get(anchor.fragment_group_id) or [anchor]
            emit(
                fragments,
                reason="fragment_group" if len(fragments) > 1 else "atomic",
                anchor_ids=[anchor.chunk_id],
            )

        # A single atomic block only pulls a neighbor when its wording explicitly
        # depends on preceding or following content. Fragment groups are already
        # complete and never receive a speculative neighbor.
        for anchor in anchors:
            fragments = self.by_fragment_group.get(anchor.fragment_group_id) or [anchor]
            if len(fragments) > 1:
                continue
            needs_left, needs_right = self._dependency_directions(anchor)
            if not needs_left and not needs_right:
                continue
            section_chunks = self.by_section.get(anchor.section_path) or []
            try:
                position = section_chunks.index(anchor)
            except ValueError:
                continue
            contextual: List[EvidenceChunk] = [anchor]
            if needs_left and position > 0:
                contextual = self._expand_fragment_groups([section_chunks[position - 1]]) + contextual
            if needs_right and position + 1 < len(section_chunks):
                contextual += self._expand_fragment_groups([section_chunks[position + 1]])
            if len(contextual) > 1:
                emit(
                    contextual,
                    reason="dependency_context",
                    anchor_ids=[anchor.chunk_id],
                )

        # Only consecutive atomic hits within one logical section are merged.
        anchors_by_section: Dict[str, List[EvidenceChunk]] = defaultdict(list)
        for anchor in anchors:
            anchors_by_section[anchor.section_path].append(anchor)
        for section_anchors in anchors_by_section.values():
            for run in self._contiguous_anchor_runs(section_anchors):
                if len(run) < 2:
                    continue
                emit(
                    self._expand_fragment_groups(run),
                    reason="same_section_contiguous",
                    anchor_ids=[anchor.chunk_id for anchor in run],
                )

        # Hits in multiple child sections share a real parent. Only hit children are combined.
        anchors_by_parent: Dict[str, List[EvidenceChunk]] = defaultdict(list)
        for anchor in anchors:
            if anchor.section_parent_path:
                anchors_by_parent[anchor.section_parent_path].append(anchor)
        for parent_path, child_anchors in anchors_by_parent.items():
            child_sections = {anchor.section_path for anchor in child_anchors}
            if len(child_sections) < 2:
                continue
            expanded = self._expand_fragment_groups(child_anchors)
            anchor_ids = [anchor.chunk_id for anchor in child_anchors]
            emit(expanded, reason="sibling_hits", anchor_ids=anchor_ids)
            parent_body = self.by_section.get(parent_path) or []
            if parent_body:
                emit(
                    list(parent_body) + expanded,
                    reason="sibling_hits_with_parent_body",
                    anchor_ids=anchor_ids,
                    parent=True,
                )

        return windows


__all__ = ["EvidenceWindowBuilder"]
