"""Integrated OCR -> image understanding -> QA handoff preprocessor."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import UploadFile
except ImportError:  # pragma: no cover - only needed for runtime type checks
    UploadFile = Any

from app.core.config import CONFIG
from app.services.doc_chunks import build_doc_id
from app.services.image_understanding import analyze_images_simple
from app.services.image_understanding.image_models import AnalysisResult
from app.services.llm import LLMClientConfig

from .markers import (
    extract_marker_ids,
    locate_markers_in_chunks,
    replace_image_divs_with_markers,
    restore_markers_in_text,
    with_context,
)
from .models import ChunkContext, ImageAnchorContext, IntegratedDocumentResult
from .ocr_worker import get_ocr_worker_manager, resolve_ocr_use_gpu
from .placement import ImagePlacementJudge, normalize_fit_score
from .summary import ChunkSummaryService, normalize_summary_mode


def _safe_upload_filename(filename: Optional[str]) -> str:
    name = (filename or "upload.bin").replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name or "upload.bin"


def _guess_content_format(filename: Optional[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    return "markdown" if suffix in {".md", ".markdown"} else "text"


def _build_file_content_record(
    *,
    filename: str,
    content: Optional[str],
    status: str,
    ocr_seconds: Optional[float],
    error: Optional[str] = None,
    content_format: str = "markdown",
    markdown_content: Optional[str] = None,
    plain_text: Optional[str] = None,
    ocr_raw_entry: Optional[Dict[str, Any]] = None,
    pre_split_chunks: Optional[List[str]] = None,
    pre_split_chunk_meta: Optional[List[Dict[str, Any]]] = None,
    chunking_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "content": content,
        "size": len(content) if isinstance(content, str) else 0,
        "status": status,
        "error": error,
        "ocr_seconds": ocr_seconds,
        "content_format": content_format,
        "markdown_content": markdown_content if markdown_content else (content if content_format == "markdown" else None),
        "plain_text": plain_text if plain_text else (content if content_format == "text" else None),
        "ocr_pages": [],
        "ocr_raw_entry": ocr_raw_entry,
        "pre_split_chunks": pre_split_chunks or None,
        "pre_split_chunk_meta": pre_split_chunk_meta or None,
        "chunking_report": chunking_report or None,
    }


def _build_llm_config() -> LLMClientConfig:
    return LLMClientConfig(
        api_base=CONFIG.get("base_url"),
        model_name=CONFIG.get("model"),
        api_key=CONFIG.get("api_key"),
        api_type="openai",
        timeout_seconds=float(CONFIG.get("request_timeout", 120) or 120),
    )


def _chunk_contexts(chunks_meta: List[Dict[str, Any]]) -> List[ChunkContext]:
    contexts: List[ChunkContext] = []
    for index, meta in enumerate(chunks_meta or [], start=1):
        if not isinstance(meta, dict):
            continue
        contexts.append(
            ChunkContext(
                chunk_index=int(meta.get("chunk_index") or index),
                chunk_id=str(meta.get("chunk_id") or ""),
                text=str(meta.get("text") or ""),
                title_path=str(meta.get("title_path") or ""),
                path_summary=str(meta.get("path_summary") or ""),
            )
        )
    return contexts


def _local_context_around_marker(text: str, marker: str, limit: int = 900) -> Tuple[str, str]:
    source = str(text or "")
    pos = source.find(marker)
    if pos < 0:
        return "", ""
    before = source[max(0, pos - limit) : pos].strip()
    after = source[pos + len(marker) : pos + len(marker) + limit].strip()
    return before, after


class IntegratedPipelineRunner:
    def __init__(
        self,
        *,
        task_id: str,
        chunk_size: int,
        chunking_prefix_max_depth: int = 4,
        chunking_split_type: Optional[str] = None,
        chunking_text_split_min_length: Optional[int] = None,
        chunking_text_split_max_length: Optional[int] = None,
        chunking_chunk_overlap: Optional[int] = None,
        chunking_separator: Optional[str] = None,
        chunking_separators: Optional[List[str]] = None,
        chunking_split_language: Optional[str] = None,
        chunking_custom_separator: Optional[str] = None,
        chunking_manual_split_points: Optional[List[Dict[str, Any]]] = None,
        chunking_markdown_heading_correction_enabled: bool = True,
        image_context_summary_mode: str = "lightweight",
        image_fit_check_enabled: bool = True,
        image_fit_min_score: float = 0.65,
        remove_watermark: bool = False,
        watermark_dpi: int = 200,
        replace_images: bool = True,
        docx_strategy: str = "pdf",
        use_gpu: Optional[bool] = None,
    ) -> None:
        self.task_id = task_id
        self.chunk_size = max(1, int(chunk_size or 600))
        self.chunking_prefix_max_depth = max(0, min(12, int(chunking_prefix_max_depth or 4)))
        self.chunking_split_type = chunking_split_type
        self.chunking_text_split_min_length = chunking_text_split_min_length
        self.chunking_text_split_max_length = chunking_text_split_max_length
        self.chunking_chunk_overlap = chunking_chunk_overlap
        self.chunking_separator = chunking_separator
        self.chunking_separators = chunking_separators
        self.chunking_split_language = chunking_split_language
        self.chunking_custom_separator = chunking_custom_separator
        self.chunking_manual_split_points = chunking_manual_split_points
        self.chunking_markdown_heading_correction_enabled = bool(chunking_markdown_heading_correction_enabled)
        self.image_context_summary_mode = normalize_summary_mode(image_context_summary_mode)
        self.image_fit_check_enabled = bool(image_fit_check_enabled)
        self.image_fit_min_score = normalize_fit_score(image_fit_min_score)
        self.remove_watermark = bool(remove_watermark)
        self.watermark_dpi = int(watermark_dpi or 200)
        self.replace_images = bool(replace_images)
        self.docx_strategy = str(docx_strategy or "pdf")
        self.use_gpu = resolve_ocr_use_gpu(default=True) if use_gpu is None else bool(use_gpu)
        self.llm_config = _build_llm_config()

    def process_text_file(self, *, filename: str, content: str) -> IntegratedDocumentResult:
        doc_id = build_doc_id(filename, content)
        chunks_for_llm, chunks_meta, chunking_report = self._build_chunks(
            text=content,
            original_filename=filename,
            doc_id=doc_id,
            content_format=_guess_content_format(filename),
            debug_writer=None,
        )
        final_content = "\n\n".join(str(meta.get("text") or chunk) for chunk, meta in zip(chunks_for_llm, chunks_meta))
        return IntegratedDocumentResult(
            filename=filename,
            content=final_content,
            markdown_content=final_content if _guess_content_format(filename) == "markdown" else "",
            plain_text=final_content if _guess_content_format(filename) == "text" else None,
            pre_split_chunks=chunks_for_llm,
            pre_split_chunk_meta=chunks_meta,
            chunking_report=chunking_report,
            ocr_raw_entry={"status": "skipped", "reason": "text_input"},
            ocr_seconds=0.0,
        )

    def process_ocr_file(self, *, file_path: str, filename: str, output_dir: str) -> IntegratedDocumentResult:
        started = time.perf_counter()
        ocr_result = get_ocr_worker_manager().extract(
            file_path=file_path,
            output_dir=output_dir,
            docx_strategy=self.docx_strategy,
            remove_watermark=self.remove_watermark,
            watermark_dpi=self.watermark_dpi,
            replace_images=self.replace_images,
            use_gpu=self.use_gpu,
        )
        ocr_seconds = time.perf_counter() - started
        marked_markdown, image_markers = replace_image_divs_with_markers(
            ocr_result.markdown_content,
            ocr_result.images_info,
        )
        doc_id = build_doc_id(filename, marked_markdown)
        chunks_for_llm, chunks_meta, chunking_report = self._build_chunks(
            text=marked_markdown,
            original_filename=filename,
            doc_id=doc_id,
            content_format="markdown",
            debug_writer=None,
        )
        contexts = ChunkSummaryService(
            mode=self.image_context_summary_mode,
            llm_config=self.llm_config,
        ).summarize(_chunk_contexts(chunks_meta))
        context_by_index = {ctx.chunk_index: ctx for ctx in contexts}
        for meta in chunks_meta:
            ctx = context_by_index.get(int(meta.get("chunk_index") or 0))
            if ctx:
                meta["image_context_summary"] = ctx.summary

        image_info_by_id = {str(info.image_id): info for info in ocr_result.images_info}
        marker_ids = {marker.image_id for marker in image_markers}
        marker_chunk_indices = locate_markers_in_chunks(chunks_meta)
        anchors: Dict[str, ImageAnchorContext] = {}
        for meta in chunks_meta:
            chunk_index = int(meta.get("chunk_index") or 0)
            ctx = context_by_index.get(chunk_index)
            if not ctx:
                continue
            text = str(meta.get("text") or "")
            for image_id in extract_marker_ids(text):
                if marker_chunk_indices.get(image_id) != chunk_index:
                    continue
                if image_id not in marker_ids:
                    continue
                info = image_info_by_id.get(image_id)
                if info is None:
                    continue
                marker = f"[[IMAGE_REF:{image_id}]]"
                local_before, local_after = _local_context_around_marker(text, marker)
                raw_path = Path(str(info.file_path))
                image_path = raw_path if raw_path.is_absolute() else Path(ocr_result.output_dir) / raw_path
                anchors[image_id] = ImageAnchorContext(
                    image_id=image_id,
                    marker=marker,
                    image_path=image_path,
                    chunk=ctx,
                    original_context_before=str(info.context_before or ""),
                    original_context_after=str(info.context_after or ""),
                    context_before=local_before,
                    context_after=local_after,
                )

        replacements: Dict[str, str] = {}
        placement_details: List[Dict[str, Any]] = []
        if anchors:
            enhanced_images = []
            for image_id, anchor in anchors.items():
                info = image_info_by_id[image_id]
                before = (
                    f"图片所在 chunk 摘要：\n{anchor.chunk.summary}\n\n"
                    f"图片前局部上下文：\n{anchor.context_before}\n\n"
                    f"OCR 原上文：\n{anchor.original_context_before}"
                )
                after = (
                    f"图片后局部上下文：\n{anchor.context_after}\n\n"
                    f"OCR 原下文：\n{anchor.original_context_after}"
                )
                enhanced_images.append(with_context(info, before=before, after=after))
            enhanced_ocr_result = type(ocr_result)(
                pdf_name=ocr_result.pdf_name,
                total_pages=ocr_result.total_pages,
                markdown_content=marked_markdown,
                images_info=enhanced_images,
                figure_titles=ocr_result.figure_titles,
                processing_time=ocr_result.processing_time,
                output_dir=ocr_result.output_dir,
            )
            analysis_result = analyze_images_simple(
                ocr_result=enhanced_ocr_result,
                output_dir=str(Path(output_dir) / "image_analysis"),
                use_api=True,
                enable_classification=False,
            )
            judge = ImagePlacementJudge(
                enabled=self.image_fit_check_enabled,
                min_score=self.image_fit_min_score,
                llm_config=self.llm_config,
            )
            for desc in analysis_result.descriptions:
                if desc.status != "success":
                    placement_details.append(
                        {"image_id": desc.image_id, "accepted": False, "reason": desc.error_message}
                    )
                    continue
                anchor = anchors.get(desc.image_id)
                if anchor is None:
                    continue
                decision = judge.judge(anchor=anchor, description=desc.description)
                placement_details.append(
                    {
                        "image_id": decision.image_id,
                        "accepted": decision.accepted,
                        "score": decision.score,
                        "reason": decision.reason,
                        "error": decision.error,
                    }
                )
                if decision.accepted:
                    replacements[desc.image_id] = desc.description
        else:
            analysis_result = AnalysisResult(
                pdf_name=ocr_result.pdf_name,
                total_images=0,
                analyzed_images=0,
                descriptions=[],
                processing_time=0.0,
                output_dir=Path(output_dir) / "image_analysis",
            )

        final_chunks: List[str] = []
        for meta in chunks_meta:
            meta["text"] = restore_markers_in_text(str(meta.get("text") or ""), replacements)
            meta["text_for_embedding"] = restore_markers_in_text(str(meta.get("text_for_embedding") or ""), replacements)
            meta["image_replacements"] = {
                "accepted_ids": sorted(replacements),
                "placement_details": placement_details,
            }
            final_chunks.append(str(meta.get("text_for_embedding") or meta.get("text") or ""))

        final_markdown = "\n\n".join(str(meta.get("text") or "") for meta in chunks_meta if str(meta.get("text") or "").strip())
        ocr_raw_entry = ocr_result.to_dict()
        ocr_raw_entry["integrated_pipeline"] = {
            "replace_images": self.replace_images,
            "image_markers": [marker.__dict__ for marker in image_markers],
            "image_analysis": {
                "total_images": analysis_result.total_images,
                "analyzed_images": analysis_result.analyzed_images,
            },
            "placement_details": placement_details,
            "accepted_image_ids": sorted(replacements),
        }
        return IntegratedDocumentResult(
            filename=filename,
            content=final_markdown,
            markdown_content=final_markdown,
            plain_text=None,
            pre_split_chunks=final_chunks,
            pre_split_chunk_meta=chunks_meta,
            chunking_report=chunking_report,
            ocr_raw_entry=ocr_raw_entry,
            ocr_seconds=ocr_seconds,
        )

    def _build_chunks(
        self,
        *,
        text: str,
        original_filename: str,
        doc_id: str,
        content_format: str,
        debug_writer: Any,
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
        from qa.chunking import build_tree_chunks

        return build_tree_chunks(
            text,
            chunk_size=self.chunk_size,
            original_filename=original_filename,
            task_id=self.task_id,
            doc_id=doc_id,
            prefix_max_depth=self.chunking_prefix_max_depth,
            debug_writer=debug_writer,
            split_type=self.chunking_split_type,
            text_split_min_length=self.chunking_text_split_min_length,
            text_split_max_length=self.chunking_text_split_max_length,
            chunk_overlap=self.chunking_chunk_overlap,
            separator=self.chunking_separator,
            separators=self.chunking_separators,
            split_language=self.chunking_split_language,
            custom_separator=self.chunking_custom_separator,
            manual_split_points=self.chunking_manual_split_points,
            force_heading_correction=(
                content_format == "markdown" and self.chunking_markdown_heading_correction_enabled
            ),
        )


async def _save_upload_to_path(upload_file: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    upload_file.file.seek(0)
    content = await upload_file.read()
    path.write_bytes(content)
    upload_file.file.seek(0)


async def resolve_uploaded_files_with_integrated_processing(
    upload_files: List[UploadFile],
    *,
    task_id: str,
    chunk_size: int,
    ocr_enabled: bool = True,
    ocr_fail_fast: bool = False,
    image_context_summary_mode: str = "lightweight",
    image_fit_check_enabled: bool = True,
    image_fit_min_score: float = 0.65,
    remove_watermark: bool = False,
    watermark_dpi: int = 200,
    replace_images: bool = True,
    docx_strategy: str = "pdf",
    chunking_prefix_max_depth: int = 4,
    chunking_split_type: Optional[str] = None,
    chunking_text_split_min_length: Optional[int] = None,
    chunking_text_split_max_length: Optional[int] = None,
    chunking_chunk_overlap: Optional[int] = None,
    chunking_separator: Optional[str] = None,
    chunking_separators: Optional[List[str]] = None,
    chunking_split_language: Optional[str] = None,
    chunking_custom_separator: Optional[str] = None,
    chunking_manual_split_points: Optional[List[Dict[str, Any]]] = None,
    chunking_markdown_heading_correction_enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from app.services.ocr import file_requires_ocr
    from app.services.storage.uploads import read_uploaded_file_content

    runner = IntegratedPipelineRunner(
        task_id=task_id,
        chunk_size=chunk_size,
        chunking_prefix_max_depth=chunking_prefix_max_depth,
        chunking_split_type=chunking_split_type,
        chunking_text_split_min_length=chunking_text_split_min_length,
        chunking_text_split_max_length=chunking_text_split_max_length,
        chunking_chunk_overlap=chunking_chunk_overlap,
        chunking_separator=chunking_separator,
        chunking_separators=chunking_separators,
        chunking_split_language=chunking_split_language,
        chunking_custom_separator=chunking_custom_separator,
        chunking_manual_split_points=chunking_manual_split_points,
        chunking_markdown_heading_correction_enabled=chunking_markdown_heading_correction_enabled,
        image_context_summary_mode=image_context_summary_mode,
        image_fit_check_enabled=image_fit_check_enabled,
        image_fit_min_score=image_fit_min_score,
        remove_watermark=remove_watermark,
        watermark_dpi=watermark_dpi,
        replace_images=replace_images,
        docx_strategy=docx_strategy,
    )
    base_dir = Path(CONFIG["outputs_dir"]) / "integrated_pipeline" / task_id
    input_dir = base_dir / "uploads"
    output_dir = base_dir / "documents"
    file_contents: List[Dict[str, Any]] = []
    ocr_summary: List[Dict[str, Any]] = []

    for index, upload_file in enumerate(upload_files):
        filename = _safe_upload_filename(upload_file.filename)
        requires_ocr = file_requires_ocr(upload_file)
        try:
            if requires_ocr and not ocr_enabled:
                raise RuntimeError("OCR is disabled but file requires OCR")
            if requires_ocr:
                input_path = input_dir / f"{index:04d}_{filename}"
                await _save_upload_to_path(upload_file, input_path)
                doc_output_dir = output_dir / f"{index:04d}_{Path(filename).stem}"
                result = await asyncio.to_thread(
                    runner.process_ocr_file,
                    file_path=str(input_path),
                    filename=filename,
                    output_dir=str(doc_output_dir),
                )
                content_format = "markdown"
            else:
                text_content = read_uploaded_file_content(upload_file)
                result = await asyncio.to_thread(
                    runner.process_text_file,
                    filename=filename,
                    content=text_content,
                )
                content_format = _guess_content_format(filename)
            file_contents.append(
                _build_file_content_record(
                    filename=filename,
                    content=result.content,
                    status="success",
                    ocr_seconds=result.ocr_seconds,
                    content_format=content_format,
                    markdown_content=result.markdown_content if content_format == "markdown" else None,
                    plain_text=result.plain_text if content_format == "text" else None,
                    ocr_raw_entry=result.ocr_raw_entry,
                    pre_split_chunks=result.pre_split_chunks,
                    pre_split_chunk_meta=result.pre_split_chunk_meta,
                    chunking_report=result.chunking_report,
                )
            )
            ocr_summary.append(
                {
                    "filename": filename,
                    "status": "success",
                    "integrated_pipeline": True,
                    "content_format": content_format,
                    "has_pre_split_chunks": True,
                    "chunks": len(result.pre_split_chunk_meta),
                    "ocr_seconds": result.ocr_seconds,
                    "replace_images": runner.replace_images,
                }
            )
        except Exception as exc:
            error_message = str(exc)
            file_contents.append(
                _build_file_content_record(
                    filename=filename,
                    content=None,
                    status="error",
                    error=error_message,
                    ocr_seconds=0.0,
                    content_format="text",
                )
            )
            ocr_summary.append(
                {
                    "filename": filename,
                    "status": "error",
                    "integrated_pipeline": True,
                    "error": error_message,
                    "ocr_seconds": 0.0,
                    "replace_images": runner.replace_images,
                }
            )
            if ocr_fail_fast:
                raise RuntimeError(error_message) from exc

    return file_contents, ocr_summary
