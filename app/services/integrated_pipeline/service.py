"""Integrated OCR -> image understanding -> QA handoff preprocessor."""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from fastapi import UploadFile
except ImportError:  # pragma: no cover - only needed for runtime type checks
    UploadFile = Any

from app.core.clients import build_llm_client_config
from app.core.config import CONFIG
from app.services.doc_chunks import build_doc_id
from app.services.image_understanding import analyze_images_simple
from app.services.image_understanding.image_models import AnalysisResult
from app.services.llm import LLMClientConfig, get_llm_client_pool

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


def _resolve_positive_int(
    value: Optional[int],
    *,
    env_name: str,
    default: int,
    minimum: int = 1,
    maximum: int = 128,
) -> int:
    raw: Any = value
    if raw is None:
        raw = str(os.getenv(env_name) or "").strip()
    try:
        resolved = int(raw)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(minimum, min(maximum, resolved))


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
    return build_llm_client_config()


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


def _chunk_context_around_marker(text: str, marker: str) -> Tuple[str, str]:
    source = str(text or "")
    pos = source.find(marker)
    if pos < 0:
        return "", ""
    before = restore_markers_in_text(source[:pos], {}, remove_missing=True).strip()
    after = restore_markers_in_text(source[pos + len(marker) :], {}, remove_missing=True).strip()
    return before, after


def _format_image_side_context(label: str, text: str) -> str:
    value = str(text or "").strip() or "（无）"
    return f"{label}：\n{value}"


def _format_image_before_context(summary: str, before_text: str) -> str:
    parts: List[str] = []
    summary_value = str(summary or "").strip()
    if summary_value:
        parts.append(f"图片所在 chunk 摘要：\n{summary_value}")
    parts.append(_format_image_side_context("图片之前内容", before_text))
    return "\n\n".join(parts)


def _map_image_analysis_progress_stage(sub_stage: Any) -> str:
    return "image_classification" if str(sub_stage or "").strip() == "image_classification" else "doc_image_analysis"


def _normalize_pdf_docx_strategy(value: str) -> str:
    return "pdf"


def _emit_integrated_progress(
    progress_callback: Optional[Callable[[str, str, str, str, Optional[Dict[str, Any]]], None]],
    *,
    filename: str,
    stage: str,
    state: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(filename, stage, state, message, extra or {})
    except Exception:
        # Progress reporting should never break the preprocessing path.
        pass


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
        image_analysis_enabled: bool = True,
        image_analysis_use_api: bool = True,
        image_analysis_vlm_api_base: Optional[str] = None,
        image_analysis_vlm_model_name: Optional[str] = None,
        image_analysis_vlm_api_key: Optional[str] = None,
        image_analysis_vlm_api_type: Optional[str] = None,
        image_analysis_vlm_model_version: Optional[str] = None,
        image_analysis_enable_classification: bool = False,
        image_analysis_classification_confidence_threshold: float = 0.0,
        image_analysis_max_concurrency: Optional[int] = None,
        image_fit_max_concurrency: Optional[int] = None,
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
        self.docx_strategy = _normalize_pdf_docx_strategy(docx_strategy)
        self.use_gpu = resolve_ocr_use_gpu(default=True) if use_gpu is None else bool(use_gpu)
        self.llm_config = _build_llm_config()
        self.image_analysis_enabled = bool(image_analysis_enabled)
        self.image_analysis_use_api = bool(image_analysis_use_api)
        self.image_analysis_vlm_api_base = image_analysis_vlm_api_base or self.llm_config.api_base
        self.image_analysis_vlm_model_name = image_analysis_vlm_model_name or self.llm_config.model_name
        self.image_analysis_vlm_api_key = image_analysis_vlm_api_key or self.llm_config.api_key
        self.image_analysis_vlm_api_type = image_analysis_vlm_api_type or self.llm_config.api_type or "openai"
        self.image_analysis_vlm_model_version = image_analysis_vlm_model_version or self.llm_config.model_version
        self.image_analysis_enable_classification = bool(image_analysis_enable_classification)
        self.image_analysis_classification_confidence_threshold = max(
            0.0,
            min(1.0, float(image_analysis_classification_confidence_threshold or 0.0)),
        )
        self.image_analysis_max_concurrency = _resolve_positive_int(
            image_analysis_max_concurrency,
            env_name="IMAGE_ANALYSIS_MAX_CONCURRENCY",
            default=1,
            maximum=64,
        )
        self.image_fit_max_concurrency = _resolve_positive_int(
            image_fit_max_concurrency,
            env_name="IMAGE_FIT_MAX_CONCURRENCY",
            default=1,
            maximum=64,
        )

    def process_text_file(
        self,
        *,
        filename: str,
        content: str,
        progress_callback: Optional[Callable[[str, str, str, str, Optional[Dict[str, Any]]], None]] = None,
    ) -> IntegratedDocumentResult:
        doc_id = build_doc_id(filename, content)
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_pre_chunking",
            state="processing",
            message="文本切块中",
            extra={"content_chars": len(content or "")},
        )
        chunks_for_llm, chunks_meta, chunking_report = self._build_chunks(
            text=content,
            original_filename=filename,
            doc_id=doc_id,
            content_format=_guess_content_format(filename),
            debug_writer=None,
        )
        final_content = "\n\n".join(str(meta.get("text") or chunk) for chunk, meta in zip(chunks_for_llm, chunks_meta))
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_pre_chunking",
            state="completed",
            message=f"文本切块完成：{len(chunks_meta)} 个 chunk",
            extra={"chunks": len(chunks_meta), "chunking_report": chunking_report},
        )
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_handoff",
            state="completed",
            message="文档预处理完成，交给问答流水线",
            extra={"chunks": len(chunks_meta), "content_format": _guess_content_format(filename)},
        )
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

    def process_ocr_file(
        self,
        *,
        file_path: str,
        filename: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str, str, str, str, Optional[Dict[str, Any]]], None]] = None,
    ) -> IntegratedDocumentResult:
        started = time.perf_counter()

        def ocr_progress(event: Dict[str, Any]) -> None:
            if not isinstance(event, dict):
                return
            extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_ocr",
                state=str(event.get("state") or "processing"),
                message=str(event.get("message") or "OCR 处理中"),
                extra={"sub_stage": event.get("stage"), **extra},
            )

        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_ocr",
            state="processing",
            message="dw OCR/文档抽取开始",
            extra={
                "docx_strategy": self.docx_strategy,
                "remove_watermark": self.remove_watermark,
                "replace_images": self.replace_images,
            },
        )
        ocr_result = get_ocr_worker_manager().extract(
            file_path=file_path,
            output_dir=output_dir,
            docx_strategy=self.docx_strategy,
            remove_watermark=self.remove_watermark,
            watermark_dpi=self.watermark_dpi,
            replace_images=self.replace_images,
            use_gpu=self.use_gpu,
            progress_callback=ocr_progress,
        )
        ocr_seconds = time.perf_counter() - started
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_ocr",
            state="completed",
            message=f"dw OCR/文档抽取完成：{ocr_result.total_pages} 页，{len(ocr_result.images_info)} 张图片",
            extra={
                "total_pages": ocr_result.total_pages,
                "total_images": len(ocr_result.images_info),
                "ocr_seconds": ocr_seconds,
            },
        )
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_marker",
            state="processing",
            message="图片占位标记替换中",
        )
        marked_markdown, image_markers = replace_image_divs_with_markers(
            ocr_result.markdown_content,
            ocr_result.images_info,
        )
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_marker",
            state="completed",
            message=f"图片占位标记完成：{len(image_markers)} 个 marker",
            extra={"image_markers": len(image_markers), "markdown_chars": len(marked_markdown)},
        )
        doc_id = build_doc_id(filename, marked_markdown)
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_pre_chunking",
            state="processing",
            message="OCR Markdown 切块中",
            extra={"markdown_chars": len(marked_markdown)},
        )
        chunks_for_llm, chunks_meta, chunking_report = self._build_chunks(
            text=marked_markdown,
            original_filename=filename,
            doc_id=doc_id,
            content_format="markdown",
            debug_writer=None,
        )
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_pre_chunking",
            state="completed",
            message=f"OCR Markdown 切块完成：{len(chunks_meta)} 个 chunk",
            extra={"chunks": len(chunks_meta), "chunking_report": chunking_report},
        )
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_chunk_summary",
            state="processing",
            message="图片所在 chunk 摘要生成中",
            extra={"chunks": len(chunks_meta), "mode": self.image_context_summary_mode},
        )
        contexts = ChunkSummaryService(
            mode=self.image_context_summary_mode,
            llm_config=self.llm_config,
        ).summarize(_chunk_contexts(chunks_meta))
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_chunk_summary",
            state="completed",
            message=f"chunk 摘要完成：{len(contexts)} 个 chunk",
            extra={"chunks": len(contexts), "mode": self.image_context_summary_mode},
        )
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
            prompt_context_text = str(meta.get("text_for_embedding") or text)
            for image_id in extract_marker_ids(text):
                if marker_chunk_indices.get(image_id) != chunk_index:
                    continue
                if image_id not in marker_ids:
                    continue
                info = image_info_by_id.get(image_id)
                if info is None:
                    continue
                marker = f"[[IMAGE_REF:{image_id}]]"
                local_before, local_after = _chunk_context_around_marker(
                    prompt_context_text,
                    marker,
                )
                if not local_before and not local_after:
                    local_before, local_after = _chunk_context_around_marker(text, marker)
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
        if anchors and self.image_analysis_enabled:
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_image_analysis",
                state="processing",
                message=f"图片理解准备中：{len(anchors)} 张图片",
                extra={
                    "total_images": len(anchors),
                    "anchors": len(anchors),
                    "use_api": self.image_analysis_use_api,
                    "vlm_api_type": self.image_analysis_vlm_api_type,
                    "vlm_model_name": self.image_analysis_vlm_model_name,
                    "classification_enabled": self.image_analysis_enable_classification,
                },
            )
            enhanced_images = []
            for image_id, anchor in anchors.items():
                info = image_info_by_id[image_id]
                before = _format_image_before_context(anchor.chunk.summary, anchor.context_before)
                after = _format_image_side_context("图片之后内容", anchor.context_after)
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

            def image_progress(event: Dict[str, Any]) -> None:
                if not isinstance(event, dict):
                    return
                extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
                sub_stage = str(event.get("stage") or "").strip()
                stage = _map_image_analysis_progress_stage(sub_stage)
                _emit_integrated_progress(
                    progress_callback,
                    filename=filename,
                    stage=stage,
                    state=str(event.get("state") or "processing"),
                    message=str(event.get("message") or "图片理解中"),
                    extra={"sub_stage": sub_stage or event.get("stage"), **extra},
                )

            vlm_client = None
            if self.image_analysis_use_api:
                vlm_client = get_llm_client_pool().get_client(
                    build_llm_client_config(
                        base_url=self.image_analysis_vlm_api_base,
                        model=self.image_analysis_vlm_model_name,
                        api_key=self.image_analysis_vlm_api_key,
                        api_type=self.image_analysis_vlm_api_type,
                        model_version=self.image_analysis_vlm_model_version,
                    )
                )

            analysis_result = analyze_images_simple(
                ocr_result=enhanced_ocr_result,
                output_dir=str(Path(output_dir) / "image_analysis"),
                use_api=self.image_analysis_use_api,
                api_base=self.image_analysis_vlm_api_base,
                model_name=self.image_analysis_vlm_model_name,
                api_key=self.image_analysis_vlm_api_key,
                vlm_api_type=self.image_analysis_vlm_api_type,
                model_version=self.image_analysis_vlm_model_version,
                vlm_client=vlm_client,
                enable_classification=self.image_analysis_enable_classification,
                classification_confidence_threshold=self.image_analysis_classification_confidence_threshold,
                max_concurrency=self.image_analysis_max_concurrency,
                progress_callback=image_progress,
            )
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_image_analysis",
                state="completed",
                message=f"图片理解完成：{analysis_result.analyzed_images}/{analysis_result.total_images}",
                extra={
                    "total_images": analysis_result.total_images,
                    "analyzed_images": analysis_result.analyzed_images,
                    "failed_images": analysis_result.total_images - analysis_result.analyzed_images,
                    "image_analysis_seconds": analysis_result.processing_time,
                },
            )
            judge = ImagePlacementJudge(
                enabled=self.image_fit_check_enabled,
                min_score=self.image_fit_min_score,
                llm_config=self.llm_config,
            )
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_placement",
                state="processing",
                message="图片结果契合度判断中",
                extra={"enabled": self.image_fit_check_enabled, "min_score": self.image_fit_min_score},
            )
            def judge_description(desc: Any) -> Tuple[Any, Dict[str, Any], bool]:
                if desc.status != "success":
                    return (
                        desc.image_id,
                        {"image_id": desc.image_id, "accepted": False, "reason": desc.error_message},
                        False,
                    )
                anchor = anchors.get(desc.image_id)
                if anchor is None:
                    return (
                        desc.image_id,
                        {"image_id": desc.image_id, "accepted": False, "reason": "missing image anchor"},
                        False,
                    )
                decision = judge.judge(anchor=anchor, description=desc.description)
                detail = {
                        "image_id": decision.image_id,
                        "accepted": decision.accepted,
                        "score": decision.score,
                        "reason": decision.reason,
                        "error": decision.error,
                }
                return desc.image_id, detail, bool(decision.accepted)

            successful_descriptions = list(analysis_result.descriptions)
            max_fit_workers = min(self.image_fit_max_concurrency, max(1, len(successful_descriptions)))
            if max_fit_workers <= 1 or len(successful_descriptions) <= 1:
                judge_rows = [judge_description(desc) for desc in successful_descriptions]
            else:
                indexed_rows: List[Tuple[int, Any, Dict[str, Any], bool]] = []
                with ThreadPoolExecutor(max_workers=max_fit_workers) as executor:
                    future_map = {
                        executor.submit(judge_description, desc): index
                        for index, desc in enumerate(successful_descriptions)
                    }
                    for future in as_completed(future_map):
                        index = future_map[future]
                        try:
                            image_id, detail, accepted = future.result()
                        except Exception as exc:
                            image_id = str(getattr(successful_descriptions[index], "image_id", ""))
                            detail = {"image_id": image_id, "accepted": False, "reason": str(exc), "error": str(exc)}
                            accepted = False
                        indexed_rows.append((index, image_id, detail, accepted))
                indexed_rows.sort(key=lambda item: item[0])
                judge_rows = [(image_id, detail, accepted) for _index, image_id, detail, accepted in indexed_rows]

            for image_id, detail, accepted in judge_rows:
                placement_details.append(detail)
                if accepted:
                    matched_desc = next(
                        (desc for desc in analysis_result.descriptions if desc.image_id == image_id),
                        None,
                    )
                    if matched_desc is not None:
                        replacements[image_id] = matched_desc.description
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_placement",
                state="completed",
                message=f"图片回填判断完成：接受 {len(replacements)}/{len(placement_details)}",
                extra={
                    "accepted_images": len(replacements),
                    "checked_images": len(placement_details),
                    "enabled": self.image_fit_check_enabled,
                    "min_score": self.image_fit_min_score,
                },
            )
        else:
            analysis_result = AnalysisResult(
                pdf_name=ocr_result.pdf_name,
                total_images=len(anchors),
                analyzed_images=0,
                descriptions=[],
                processing_time=0.0,
                output_dir=Path(output_dir) / "image_analysis",
            )
            skip_message = "图片理解跳过：未找到可分析图片"
            if anchors and not self.image_analysis_enabled:
                skip_message = f"图片理解跳过：已禁用，保留 {len(anchors)} 个图片占位不回填"
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_image_analysis",
                state="completed",
                message=skip_message,
                extra={
                    "total_images": len(anchors),
                    "analyzed_images": 0,
                    "enabled": self.image_analysis_enabled,
                },
            )
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_placement",
                state="completed",
                message="图片回填判断跳过：无图片",
                extra={"accepted_images": 0, "checked_images": 0},
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
                "enabled": self.image_analysis_enabled,
                "use_api": self.image_analysis_use_api,
                "vlm_api_type": self.image_analysis_vlm_api_type,
                "model_name": self.image_analysis_vlm_model_name,
                "classification_enabled": self.image_analysis_enable_classification,
                "classification_confidence_threshold": self.image_analysis_classification_confidence_threshold,
                "total_images": analysis_result.total_images,
                "analyzed_images": analysis_result.analyzed_images,
            },
            "placement_details": placement_details,
            "accepted_image_ids": sorted(replacements),
        }
        _emit_integrated_progress(
            progress_callback,
            filename=filename,
            stage="doc_handoff",
            state="completed",
            message="文档预处理完成，交给问答流水线",
            extra={
                "chunks": len(chunks_meta),
                "accepted_images": len(replacements),
                "total_images": analysis_result.total_images,
                "analyzed_images": analysis_result.analyzed_images,
            },
        )
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
    image_analysis_enabled: bool = True,
    image_analysis_use_api: bool = True,
    image_analysis_vlm_api_base: Optional[str] = None,
    image_analysis_vlm_model_name: Optional[str] = None,
    image_analysis_vlm_api_key: Optional[str] = None,
    image_analysis_vlm_api_type: Optional[str] = None,
    image_analysis_vlm_model_version: Optional[str] = None,
    image_analysis_enable_classification: bool = False,
    image_analysis_classification_confidence_threshold: float = 0.0,
    doc_max_concurrency: Optional[int] = None,
    ocr_max_concurrency: Optional[int] = None,
    image_analysis_max_concurrency: Optional[int] = None,
    image_fit_max_concurrency: Optional[int] = None,
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
    progress_callback: Optional[Callable[[str, str, str, str, Optional[Dict[str, Any]]], None]] = None,
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
        image_analysis_enabled=image_analysis_enabled,
        image_analysis_use_api=image_analysis_use_api,
        image_analysis_vlm_api_base=image_analysis_vlm_api_base,
        image_analysis_vlm_model_name=image_analysis_vlm_model_name,
        image_analysis_vlm_api_key=image_analysis_vlm_api_key,
        image_analysis_vlm_api_type=image_analysis_vlm_api_type,
        image_analysis_vlm_model_version=image_analysis_vlm_model_version,
        image_analysis_enable_classification=image_analysis_enable_classification,
        image_analysis_classification_confidence_threshold=image_analysis_classification_confidence_threshold,
        image_analysis_max_concurrency=image_analysis_max_concurrency,
        image_fit_max_concurrency=image_fit_max_concurrency,
    )
    base_dir = Path(CONFIG["outputs_dir"]) / "integrated_pipeline" / task_id
    input_dir = base_dir / "uploads"
    output_dir = base_dir / "documents"
    file_contents: List[Dict[str, Any]] = []
    ocr_summary: List[Dict[str, Any]] = []

    resolved_doc_max_concurrency = _resolve_positive_int(
        doc_max_concurrency,
        env_name="DOC_MAX_CONCURRENCY",
        default=1,
        maximum=64,
    )
    resolved_ocr_max_concurrency = _resolve_positive_int(
        ocr_max_concurrency,
        env_name="OCR_MAX_CONCURRENCY",
        default=1,
        maximum=64,
    )
    doc_semaphore = asyncio.Semaphore(resolved_doc_max_concurrency)
    ocr_semaphore = asyncio.Semaphore(resolved_ocr_max_concurrency)
    file_records: List[Optional[Dict[str, Any]]] = [None] * len(upload_files)
    ocr_records: List[Optional[Dict[str, Any]]] = [None] * len(upload_files)

    async def process_one(index: int, upload_file: UploadFile) -> None:
        filename = _safe_upload_filename(upload_file.filename)
        requires_ocr = file_requires_ocr(upload_file)
        async with doc_semaphore:
            _emit_integrated_progress(
                progress_callback,
                filename=filename,
                stage="doc_input",
                state="processing",
                message="准备文档输入文件",
                extra={"file_index": index + 1, "requires_ocr": requires_ocr},
            )
            try:
                if requires_ocr and not ocr_enabled:
                    raise RuntimeError("OCR is disabled but file requires OCR")
                if requires_ocr:
                    input_path = input_dir / f"{index:04d}_{filename}"
                    await _save_upload_to_path(upload_file, input_path)
                    _emit_integrated_progress(
                        progress_callback,
                        filename=filename,
                        stage="doc_input",
                        state="completed",
                        message="文档输入文件已保存",
                        extra={"file_index": index + 1, "input_path": str(input_path), "requires_ocr": True},
                    )
                    doc_output_dir = output_dir / f"{index:04d}_{Path(filename).stem}"
                    async with ocr_semaphore:
                        result = await asyncio.to_thread(
                            runner.process_ocr_file,
                            file_path=str(input_path),
                            filename=filename,
                            output_dir=str(doc_output_dir),
                            progress_callback=progress_callback,
                        )
                    content_format = "markdown"
                else:
                    _emit_integrated_progress(
                        progress_callback,
                        filename=filename,
                        stage="doc_input",
                        state="completed",
                        message="文本输入就绪",
                        extra={"file_index": index + 1, "requires_ocr": False},
                    )
                    _emit_integrated_progress(
                        progress_callback,
                        filename=filename,
                        stage="doc_text_read",
                        state="processing",
                        message="读取文本输入",
                        extra={"file_index": index + 1},
                    )
                    text_content = read_uploaded_file_content(upload_file)
                    _emit_integrated_progress(
                        progress_callback,
                        filename=filename,
                        stage="doc_text_read",
                        state="completed",
                        message="文本输入读取完成",
                        extra={"file_index": index + 1, "content_chars": len(text_content or "")},
                    )
                    result = await asyncio.to_thread(
                        runner.process_text_file,
                        filename=filename,
                        content=text_content,
                        progress_callback=progress_callback,
                    )
                    content_format = _guess_content_format(filename)
                file_records[index] = _build_file_content_record(
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
                ocr_records[index] = {
                    "filename": filename,
                    "status": "success",
                    "integrated_pipeline": True,
                    "content_format": content_format,
                    "has_pre_split_chunks": True,
                    "chunks": len(result.pre_split_chunk_meta),
                    "ocr_seconds": result.ocr_seconds,
                    "replace_images": runner.replace_images,
                }
            except Exception as exc:
                error_message = str(exc)
                _emit_integrated_progress(
                    progress_callback,
                    filename=filename,
                    stage="doc_error",
                    state="failed",
                    message=error_message,
                    extra={"file_index": index + 1, "error": error_message},
                )
                file_records[index] = _build_file_content_record(
                    filename=filename,
                    content=None,
                    status="error",
                    error=error_message,
                    ocr_seconds=0.0,
                    content_format="text",
                )
                ocr_records[index] = {
                    "filename": filename,
                    "status": "error",
                    "integrated_pipeline": True,
                    "error": error_message,
                    "ocr_seconds": 0.0,
                    "replace_images": runner.replace_images,
                }
                if ocr_fail_fast:
                    raise RuntimeError(error_message) from exc

    tasks = [
        asyncio.create_task(process_one(index, upload_file))
        for index, upload_file in enumerate(upload_files)
    ]
    if ocr_fail_fast:
        try:
            await asyncio.gather(*tasks)
        except Exception as exc:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise RuntimeError(str(exc)) from exc
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(str(failures[0])) from failures[0]

    file_contents.extend(record for record in file_records if record is not None)
    ocr_summary.extend(record for record in ocr_records if record is not None)

    return file_contents, ocr_summary
