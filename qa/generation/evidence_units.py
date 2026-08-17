"""Same-document evidence index backed by the fixed retrieval pipeline."""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from qa.retrieval import EvidenceChunk, EvidenceRetrievalPipeline


DEFAULT_FINAL_EVIDENCE_K = 5
DEFAULT_EVIDENCE_TOKEN_BUDGET = 4000


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _prompt_title_path(chunk: Dict[str, Any]) -> str:
    return _safe_text(chunk.get("title_path")) or "未标注章节"


def _render_text_only(value: Any) -> str:
    """Keep retrieval text intact in the index, but avoid duplicating typed images in LLM evidence."""
    text = _safe_text(value)
    text = re.sub(r"【图片描述[：:].*?】", "", text, flags=re.S)
    text = re.sub(r"\[图片描述[：:].*?\]", "", text, flags=re.S)
    return text.strip()


def build_document_chunks(
    pre_split_chunks: Sequence[str],
    chunk_meta_list: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    meta_by_index: Dict[int, Dict[str, Any]] = {}
    for index, raw_meta in enumerate(chunk_meta_list or [], start=1):
        if not isinstance(raw_meta, dict):
            continue
        chunk_index = max(1, _safe_int(raw_meta.get("chunk_index"), index))
        meta_by_index[chunk_index] = dict(raw_meta)

    chunks: List[Dict[str, Any]] = []
    for index, raw_text in enumerate(pre_split_chunks or [], start=1):
        meta = dict(meta_by_index.get(index) or {})
        text = _safe_text(meta.get("text")) or _safe_text(raw_text)
        if not text:
            continue
        title_path = _safe_text(meta.get("title_path"))
        text_for_embedding = _safe_text(meta.get("text_for_embedding")) or text
        retrieval_text = text_for_embedding
        if title_path and title_path not in text_for_embedding:
            retrieval_text = f"标题路径：{title_path}\n{text_for_embedding}".strip()
        chunk_id = _safe_text(meta.get("chunk_id")) or hashlib.sha1(
            f"{index}|||{title_path}|||{text}".encode("utf-8")
        ).hexdigest()
        section_path = _safe_text(meta.get("section_path"))
        if not section_path:
            raise ValueError(
                "pre_split_chunk_meta must provide v2 section_path for every chunk"
            )
        section_parent_path = _safe_text(meta.get("section_parent_path"))
        source_asset_ids = meta.get("source_asset_ids") or []
        if not isinstance(source_asset_ids, list):
            source_asset_ids = []
        chunks.append(
            {
                **meta,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "section_chunk_index": max(1, _safe_int(meta.get("section_chunk_index"), 1)),
                "section_path": section_path,
                "section_parent_path": section_parent_path,
                "section_level": max(1, _safe_int(meta.get("section_level"), 1)),
                "section_is_leaf": bool(meta.get("section_is_leaf", True)),
                "title_path": title_path,
                "fragment_group_id": _safe_text(meta.get("fragment_group_id")) or chunk_id,
                "fragment_index": max(1, _safe_int(meta.get("fragment_index"), 1)),
                "fragment_count": max(1, _safe_int(meta.get("fragment_count"), 1)),
                "content_kind": _safe_text(meta.get("content_kind")) or "text",
                "source_asset_ids": source_asset_ids,
                "text": text,
                "text_for_embedding": text_for_embedding,
                "retrieval_text": retrieval_text,
            }
        )
    return chunks


class QADocumentEvidenceIndex:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not chunks:
            raise ValueError("QA evidence index requires at least one content chunk")
        if len(chunks) != len(embeddings):
            raise ValueError("content chunk and embedding counts do not match")
        self.chunks = chunks
        self.embeddings = embeddings
        self._chunks_by_index = {
            int(chunk.get("chunk_index") or 0): chunk for chunk in chunks
        }
        self._chunks_by_id = {
            _safe_text(chunk.get("chunk_id")): chunk for chunk in chunks
        }
        typed_chunks = [EvidenceChunk.from_dict(chunk) for chunk in chunks]
        self.pipeline = EvidenceRetrievalPipeline(typed_chunks, embeddings)

    @classmethod
    def build(cls, chunks: List[Dict[str, Any]]) -> "QADocumentEvidenceIndex":
        from app.services.milvus import generate_embeddings

        retrieval_texts = [_safe_text(chunk.get("retrieval_text")) for chunk in chunks]
        if not all(retrieval_texts):
            raise ValueError("QA evidence index found an empty retrieval_text")
        embeddings = generate_embeddings(retrieval_texts)
        return cls(chunks=chunks, embeddings=embeddings)

    def get_chunk(self, chunk_index: int) -> Dict[str, Any]:
        chunk = self._chunks_by_index.get(int(chunk_index))
        if not chunk:
            raise ValueError(f"source chunk not found: {chunk_index}")
        return chunk

    def retrieve_many(
        self,
        questions: Sequence[str],
        *,
        source_chunk_ids: Sequence[str],
        final_evidence_k: int,
        evidence_token_budget: int,
        timing: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        from app.services.milvus import generate_embeddings

        clean_questions: List[str] = []
        seen: set[str] = set()
        for question in questions:
            value = _safe_text(question)
            if value and value not in seen:
                seen.add(value)
                clean_questions.append(value)
        if not clean_questions:
            return {}
        embedding_started = time.perf_counter()
        query_embeddings = generate_embeddings(clean_questions)
        if timing is not None:
            timing["embedding_seconds"] = timing.get("embedding_seconds", 0.0) + time.perf_counter() - embedding_started
        results: Dict[str, Dict[str, Any]] = {}
        for question, embedding in zip(clean_questions, query_embeddings):
            results[question] = self.pipeline.retrieve(
                question,
                embedding,
                final_evidence_k=final_evidence_k,
                evidence_token_budget=evidence_token_budget,
                source_chunk_ids=source_chunk_ids,
                timing=timing,
            )
        return results

    def build_generation_unit(
        self,
        *,
        source_chunk_index: int,
        source_unit: Optional[Dict[str, Any]],
        question: str,
        retrieval_result: Dict[str, Any],
        final_evidence_k: int,
        evidence_token_budget: int,
    ) -> Dict[str, Any]:
        source_chunk = self.get_chunk(source_chunk_index)
        source_unit_payload = dict(source_unit or {})
        source_indexes = [
            _safe_int(value)
            for value in source_unit_payload.get("source_chunk_indexes") or []
            if _safe_int(value) in self._chunks_by_index
        ] or [source_chunk_index]
        source_chunks = [self.get_chunk(index) for index in source_indexes]
        source_ids = [_safe_text(chunk.get("chunk_id")) for chunk in source_chunks]

        # The planner binds materials, not individual chunks. Keep that
        # distinction visible to the answer model: required materials are the
        # only sources that can satisfy Summary coverage; optional materials
        # are context and must not turn an otherwise valid answer into a drop.
        material_chunk_indexes = source_unit_payload.get(
            "material_source_chunk_indexes"
        )
        if not isinstance(material_chunk_indexes, dict):
            material_chunk_indexes = {}
        required_material_ids = [
            _safe_text(value)
            for value in source_unit_payload.get("required_material_ids")
            or []
            if _safe_text(value)
        ]
        optional_material_ids = [
            _safe_text(value)
            for value in source_unit_payload.get("optional_material_ids") or []
            if _safe_text(value)
        ]
        required_indexes = {
            _safe_int(index)
            for material_id in required_material_ids
            for index in (material_chunk_indexes.get(material_id) or [])
            if _safe_int(index) in self._chunks_by_index
        }
        optional_indexes = {
            _safe_int(index)
            for material_id in optional_material_ids
            for index in (material_chunk_indexes.get(material_id) or [])
            if _safe_int(index) in self._chunks_by_index
        }
        # Legacy callers do not carry material-level mappings. Their complete
        # source unit remains required so old persisted/debug payloads keep the
        # same evidence behavior.
        if not required_indexes:
            required_indexes = set(source_indexes)
        optional_indexes -= required_indexes
        required_chunks = [
            chunk for chunk in source_chunks
            if _safe_int(chunk.get("chunk_index")) in required_indexes
        ]
        optional_chunks = [
            chunk for chunk in source_chunks
            if _safe_int(chunk.get("chunk_index")) in optional_indexes
        ]
        windows = list(retrieval_result.get("selected_windows") or [])

        evidence_chunks: List[Dict[str, Any]] = []
        seen_ids = set(source_ids)
        for window in windows:
            for chunk_id in window.chunk_ids:
                if chunk_id in seen_ids or chunk_id not in self._chunks_by_id:
                    continue
                seen_ids.add(chunk_id)
                evidence_chunks.append(self._chunks_by_id[chunk_id])

        prompt_materials = source_unit_payload.get("prompt_materials")
        prompt_materials = prompt_materials if isinstance(prompt_materials, list) else []
        material_ref_map = source_unit_payload.get("material_ref_map")
        material_ref_map = material_ref_map if isinstance(material_ref_map, dict) else {}
        image_ref_map = source_unit_payload.get("image_ref_map")
        image_ref_map = image_ref_map if isinstance(image_ref_map, dict) else {}
        required_image_ids = {
            _safe_text(value)
            for value in source_unit_payload.get("required_image_ids") or []
            if _safe_text(value)
        }
        material_by_id: Dict[str, Dict[str, Any]] = {}
        for material in prompt_materials:
            if not isinstance(material, dict):
                continue
            material_id = _safe_text(material_ref_map.get(_safe_text(material.get("material_ref"))))
            if material_id:
                material_by_id[material_id] = material
        chunks_by_index = {
            _safe_int(chunk.get("chunk_index")): chunk
            for chunk in source_chunks
            if isinstance(chunk, dict)
        }

        def material_pointer(material_id: str) -> Dict[str, Any]:
            for index in material_chunk_indexes.get(material_id) or []:
                chunk = chunks_by_index.get(_safe_int(index))
                if chunk:
                    return chunk
            return source_chunk

        sections: List[str] = ["【必需正文证据】"]
        ref_map: Dict[str, Dict[str, Any]] = {}
        rendered_required_ids: List[str] = []
        for position, material_id in enumerate(required_material_ids, start=1):
            material = material_by_id.get(material_id)
            chunk = material_pointer(material_id)
            label = f"正文证据-{position}"
            text = _safe_text(material.get("text_content")) if material else _render_text_only(chunk.get("text"))
            path = _safe_text(material.get("node_path")) if material else _prompt_title_path(chunk)
            if not text:
                text = _render_text_only(chunk.get("text"))
            sections.append(f"{label}\n节点路径：{path or '未标注章节'}\n正文：{text}")
            ref_map[label] = {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "title_path": chunk.get("title_path"),
                "material_id": material_id,
                "role": "primary_source",
            }
            rendered_required_ids.append(material_id)
        if not rendered_required_ids:
            for position, chunk in enumerate(required_chunks, start=1):
                label = f"正文证据-{position}"
                sections.append(
                    f"{label}\n节点路径：{_prompt_title_path(chunk)}\n"
                    f"正文：{_render_text_only(chunk.get('text'))}"
                )
                ref_map[label] = {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "title_path": chunk.get("title_path"),
                    "role": "primary_source",
                }
        if optional_material_ids:
            sections.append("【可选正文证据】")
        for position, material_id in enumerate(optional_material_ids, start=1):
            material = material_by_id.get(material_id)
            chunk = material_pointer(material_id)
            label = f"可选正文证据-{position}"
            text = _safe_text(material.get("text_content")) if material else _render_text_only(chunk.get("text"))
            path = _safe_text(material.get("node_path")) if material else _prompt_title_path(chunk)
            if not text:
                continue
            sections.append(f"{label}\n节点路径：{path or '未标注章节'}\n正文：{text}")
            ref_map[label] = {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "title_path": chunk.get("title_path"),
                "material_id": material_id,
                "role": "optional_source",
            }
        required_images: List[tuple[str, str, str, Dict[str, Any]]] = []
        for material_id, material in material_by_id.items():
            for image in material.get("image_materials") or []:
                if not isinstance(image, dict):
                    continue
                image_ref = _safe_text(image.get("image_ref"))
                image_id = _safe_text(image_ref_map.get(image_ref))
                if image_id and image_id in required_image_ids:
                    required_images.append((material_id, image_id, image_ref, image))
        if required_images:
            sections.append("【必需图片证据】")
        for position, (material_id, image_id, _image_ref, image) in enumerate(required_images, start=1):
            chunk = material_pointer(material_id)
            label = f"图片证据-{position}"
            sections.append(
                f"{label}\n节点路径：{_prompt_title_path(chunk)}\n"
                f"图片事实：{_safe_text(image.get('description'))}"
            )
            ref_map[label] = {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "title_path": chunk.get("title_path"),
                "material_id": material_id,
                "image_id": image_id,
                "role": "primary_visual",
            }
        if evidence_chunks:
            sections.append("【补充正文证据】")
            for position, chunk in enumerate(evidence_chunks, start=1):
                label = f"补充正文证据-{position}"
                title_path = _safe_text(chunk.get("title_path"))
                rendered = _render_text_only(chunk.get("text"))
                rendered = (
                    f"节点路径：{title_path or '未标注章节'}\n正文：{rendered}"
                )
                sections.append(f"{label}\n{rendered}")
                ref_map[label] = {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "title_path": chunk.get("title_path"),
                    "role": "retrieved_evidence",
                }

        evidence_ids = [_safe_text(chunk.get("chunk_id")) for chunk in evidence_chunks]
        unit_id = hashlib.sha1(
            ("|||".join(source_ids + [_safe_text(question)] + evidence_ids)).encode("utf-8")
        ).hexdigest()
        trace = dict(retrieval_result.get("trace") or {})
        trace["query"] = _safe_text(question)
        trace["source_chunk_ids"] = source_ids
        trace["selected_evidence_chunk_ids"] = evidence_ids
        trace["final_evidence_k"] = int(final_evidence_k)
        trace["evidence_token_budget"] = int(evidence_token_budget)
        return {
            "qa_generation_unit_id": unit_id,
            "source_chunk": source_chunk,
            "source_unit": source_unit_payload,
            "source_chunks": source_chunks,
            "source_chunk_ids": source_ids,
            "source_unit_text": _safe_text(source_unit_payload.get("unit_text")) or "\n\n".join(_safe_text(chunk.get("text")) for chunk in source_chunks),
            "evidence_hits": [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "title_path": chunk.get("title_path"),
                    "role": "retrieved_evidence",
                }
                for chunk in evidence_chunks
            ],
            "required_image_ids": list(required_image_ids),
            "evidence_chunk_ids": evidence_ids,
            "qa_generation_unit_text": "\n\n".join(sections).strip(),
            "llm_evidence_ref_map": ref_map,
            "retrieval_trace": trace,
        }


__all__ = [
    "DEFAULT_EVIDENCE_TOKEN_BUDGET",
    "DEFAULT_FINAL_EVIDENCE_K",
    "QADocumentEvidenceIndex",
    "build_document_chunks",
]
