"""dw-compatible OCR processing route."""

from __future__ import annotations

import asyncio
import math
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.core.config import CONFIG
from app.services.document_processing import normalize_output_format, resolve_result_output_file
from app.services.integrated_pipeline.ocr_worker import (
    resolve_ocr_replace_images,
    resolve_ocr_use_gpu,
)
from app.services.llm import LLMClientConfig, get_llm_client_pool, normalize_vlm_api_type

router = APIRouter()

SUPPORTED_UPLOAD_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".ofd",
    ".docx",
    ".doc",
}


@router.get("/ocr-health")
async def ocr_health():
    return {"status": "ok", "service": "ocr_compat"}


def _request_id() -> str:
    return f"{int(time.time() * 1000)}_{uuid4().hex[:8]}"


def _normalize_threshold(raw_value: float) -> float:
    try:
        threshold = float(raw_value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="classification_confidence_threshold must be a number") from exc
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="classification_confidence_threshold must be between 0.0 and 1.0")
    return threshold


def _cleanup_paths(*paths: Optional[Path]) -> None:
    for path in paths:
        if path is None:
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except Exception:
            pass


async def _save_upload(upload: UploadFile, request_dir: Path) -> Path:
    request_dir.mkdir(parents=True, exist_ok=True)
    filename = (upload.filename or "upload.bin").replace("\\", "/").rsplit("/", 1)[-1] or "upload.bin"
    path = request_dir / filename
    content = await upload.read()
    path.write_bytes(content)
    return path


@router.post("/process")
async def process_document(
    file: UploadFile = File(..., description="PDF, image, OFD, DOCX, or DOC file to process"),
    enable_image_analysis: bool = Form(True, description="Whether to run image analysis"),
    enable_classification: bool = Form(False, description="Whether to classify images before choosing prompts"),
    classification_confidence_threshold: float = Form(0.9),
    remove_watermark: bool = Form(False),
    watermark_dpi: int = Form(200),
    replace_images: Optional[bool] = Form(None),
    use_api: bool = Form(True),
    vlm_api_base: Optional[str] = Form(None),
    vlm_model_name: Optional[str] = Form(None),
    vlm_api_key: Optional[str] = Form(None),
    vlm_api_type: Optional[str] = Form(None),
    vlm_model_version: Optional[str] = Form(None),
    output_format: str = Form("text", description="Output format: text | markdown | ocr_markdown"),
    docx_strategy: str = Form("pdf", description="DOCX processing strategy: auto | native | pdf"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    resolved_output_format = normalize_output_format(output_format)
    threshold = _normalize_threshold(classification_confidence_threshold)
    resolved_vlm_api_type = normalize_vlm_api_type(vlm_api_type) if use_api else "openai"
    resolved_replace_images = resolve_ocr_replace_images(replace_images, default=True)

    base_dir = Path(CONFIG["outputs_dir"]) / "ocr_compat"
    request_id = _request_id()
    request_dir = base_dir / "uploads" / request_id
    output_dir = base_dir / "outputs" / request_id

    try:
        input_path = await _save_upload(file, request_dir)

        from app.services.document_processing.document_pipeline import DocumentPipeline
        from app.services.document_processing.ocr_processor.ocr_processor import resolve_model_base_dir
        from app.services.document_processing.pipeline import PDFPipeline

        vlm_client = None
        if use_api and enable_image_analysis:
            vlm_client = get_llm_client_pool().get_client(
                LLMClientConfig(
                    api_base=vlm_api_base,
                    model_name=vlm_model_name,
                    api_key=vlm_api_key,
                    api_type=resolved_vlm_api_type,
                    model_version=vlm_model_version,
                    timeout_seconds=180.0,
                )
            )

        model_base_dir = resolve_model_base_dir(str(Path(CONFIG["models_dir"]) / "ocr"))
        pdf_pipeline = PDFPipeline(
            model_base_dir=str(model_base_dir),
            use_gpu=resolve_ocr_use_gpu(default=True),
            output_base_dir=None,
            use_api=use_api,
            replace_images=resolved_replace_images,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            vlm_api_base=getattr(getattr(vlm_client, "config", None), "base_url", vlm_api_base),
            vlm_model_name=getattr(getattr(vlm_client, "config", None), "model_name", vlm_model_name),
            vlm_api_key=getattr(getattr(vlm_client, "config", None), "api_key", vlm_api_key),
            vlm_api_type=resolved_vlm_api_type,
            vlm_model_version=getattr(getattr(vlm_client, "config", None), "model_version", vlm_model_version),
            vlm_client=vlm_client,
            classification_confidence_threshold=threshold,
        )
        document_pipeline = DocumentPipeline(pdf_pipeline)
        result = await asyncio.to_thread(
            document_pipeline.process_file,
            file_path=str(input_path),
            custom_output_dir=str(output_dir),
            enable_image_analysis=enable_image_analysis,
            enable_classification=enable_classification,
            classification_confidence_threshold=threshold,
            remove_watermark=remove_watermark,
            watermark_dpi=watermark_dpi,
            docx_strategy=docx_strategy,
        )
        output_path, media_type = resolve_result_output_file(result, resolved_output_format)
        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Processing completed but output file missing")
        return FileResponse(
            path=output_path,
            media_type=media_type,
            filename=output_path.name,
            background=BackgroundTask(_cleanup_paths, request_dir, output_dir),
        )
    except HTTPException:
        _cleanup_paths(request_dir, output_dir)
        raise
    except Exception as exc:
        _cleanup_paths(request_dir, output_dir)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
