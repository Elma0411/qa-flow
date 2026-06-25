"""Async job manager for standalone document processing."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import CONFIG
from app.core.time_utils import now_server_local_iso


class DocumentProcessingJobManager:
    """Owns document-processing job status, persisted history, and active tasks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store_path = Path(str(CONFIG["outputs_dir"])) / "document_processing_jobs_store.json"
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._load()

    def store_path(self) -> str:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        return str(self._store_path)

    def job_base_dir(self, job_id: str) -> Path:
        return Path(str(CONFIG["outputs_dir"])) / "document_processing_jobs" / job_id

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return
        jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(jobs, list):
            return
        with self._lock:
            self._jobs = {
                str(item.get("job_id")): dict(item)
                for item in jobs
                if isinstance(item, dict) and str(item.get("job_id") or "").strip()
            }

    def _persist_locked(self) -> None:
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        items = sorted(
            self._jobs.values(),
            key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
            reverse=True,
        )
        tmp_path = self._store_path.with_suffix(self._store_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 1,
                    "updated_at": now_server_local_iso(),
                    "jobs": items,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp_path, self._store_path)

    def create_job(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = now_server_local_iso()
        job = {
            "job_id": job_id,
            "status": "queued",
            "message": "等待开始文档解析",
            "created_at": now,
            "updated_at": now,
            "file_progress": {},
            **dict(payload or {}),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._persist_locked()
            return dict(job)

    def update_job(self, job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, dict):
                return None
            job.update(updates)
            job["updated_at"] = now_server_local_iso()
            self._jobs[job_id] = job
            self._persist_locked()
            return dict(job)

    def update_stage(
        self,
        job_id: str,
        *,
        filename: str,
        stage: str,
        state: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, dict):
                return None
            file_progress = job.setdefault("file_progress", {})
            file_entry = file_progress.setdefault(
                filename or "upload",
                {
                    "status": "queued",
                    "message": "",
                    "stages": {},
                },
            )
            stages = file_entry.setdefault("stages", {})
            stage_entry = stages.setdefault(stage, {})
            stage_entry.update(
                {
                    "state": state,
                    "message": message,
                    "updated_at": now_server_local_iso(),
                }
            )
            if extra:
                stage_entry.setdefault("extra", {}).update(dict(extra))
            if state == "failed":
                file_entry["status"] = "failed"
            elif state == "completed" and stage in {"completed", "document_output"}:
                file_entry["status"] = "completed"
            else:
                file_entry["status"] = "processing"
            file_entry["message"] = message
            job["message"] = message
            job["updated_at"] = now_server_local_iso()
            self._jobs[job_id] = job
            self._persist_locked()
            return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if isinstance(job, dict) else None

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._jobs.values(),
                key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
                reverse=True,
            )
            return [dict(item) for item in items[: max(1, int(limit))]]

    def set_active_task(self, job_id: str, task: asyncio.Task) -> None:
        with self._lock:
            self._active_tasks[job_id] = task

    def pop_active_task(self, job_id: str) -> None:
        with self._lock:
            self._active_tasks.pop(job_id, None)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            task = self._active_tasks.get(job_id)
            if task is None or task.done():
                return False
            task.cancel()
            return True


DOCUMENT_PROCESSING_JOBS = DocumentProcessingJobManager()


async def run_document_processing_job(
    *,
    job_id: str,
    input_path: str,
    original_filename: str,
    params: Dict[str, Any],
) -> None:
    manager = DOCUMENT_PROCESSING_JOBS
    started = time.perf_counter()
    filename = original_filename or Path(input_path).name
    output_dir = manager.job_base_dir(job_id) / "output"

    def progress_callback(event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        manager.update_stage(
            job_id,
            filename=filename,
            stage=str(event.get("stage") or "processing"),
            state=str(event.get("state") or "processing"),
            message=str(event.get("message") or ""),
            extra=event.get("extra") if isinstance(event.get("extra"), dict) else None,
        )

    try:
        manager.update_job(
            job_id,
            status="running",
            message="文档解析任务已开始",
            started_at=now_server_local_iso(),
        )
        progress_callback(
            {
                "stage": "input",
                "state": "completed",
                "message": "上传文件已保存",
                "extra": {"input_path": input_path},
            }
        )

        from app.services.document_processing.document_pipeline import DocumentPipeline
        from app.services.document_processing.ocr_processor.ocr_processor import resolve_model_base_dir
        from app.services.document_processing.pipeline import PDFPipeline
        from app.services.integrated_pipeline.ocr_worker import resolve_ocr_use_gpu
        from app.core.clients import build_llm_client_config
        from app.services.llm import get_llm_client_pool, normalize_vlm_api_type

        use_api = bool(params.get("use_api", True))
        enable_image_analysis = bool(params.get("enable_image_analysis", True))
        resolved_vlm_api_type = normalize_vlm_api_type(params.get("vlm_api_type")) if use_api else "openai"
        vlm_client = None
        if use_api and enable_image_analysis:
            vlm_client = get_llm_client_pool().get_client(
                build_llm_client_config(
                    base_url=params.get("vlm_api_base"),
                    model=params.get("vlm_model_name"),
                    api_key=params.get("vlm_api_key"),
                    api_type=resolved_vlm_api_type,
                    model_version=params.get("vlm_model_version"),
                )
            )

        model_base_dir = resolve_model_base_dir(str(Path(CONFIG["models_dir"]) / "ocr"))
        pdf_pipeline = PDFPipeline(
            model_base_dir=str(model_base_dir),
            use_gpu=resolve_ocr_use_gpu(default=True),
            output_base_dir=None,
            use_api=use_api,
            replace_images=bool(params.get("replace_images", True)),
            remove_watermark=bool(params.get("remove_watermark", False)),
            watermark_dpi=int(params.get("watermark_dpi") or 200),
            vlm_api_base=getattr(getattr(vlm_client, "config", None), "base_url", params.get("vlm_api_base")),
            vlm_model_name=getattr(getattr(vlm_client, "config", None), "model_name", params.get("vlm_model_name")),
            vlm_api_key=getattr(getattr(vlm_client, "config", None), "api_key", params.get("vlm_api_key")),
            vlm_api_type=resolved_vlm_api_type,
            vlm_model_version=getattr(getattr(vlm_client, "config", None), "model_version", params.get("vlm_model_version")),
            vlm_client=vlm_client,
            classification_confidence_threshold=float(params.get("classification_confidence_threshold") or 0.0),
        )
        document_pipeline = DocumentPipeline(pdf_pipeline)
        result = await asyncio.to_thread(
            document_pipeline.process_file,
            file_path=input_path,
            custom_output_dir=str(output_dir),
            enable_image_analysis=enable_image_analysis,
            enable_classification=bool(params.get("enable_classification", False)),
            classification_confidence_threshold=float(params.get("classification_confidence_threshold") or 0.0),
            remove_watermark=bool(params.get("remove_watermark", False)),
            watermark_dpi=int(params.get("watermark_dpi") or 200),
            docx_strategy=str(params.get("docx_strategy") or "pdf"),
            progress_callback=progress_callback,
        )
        elapsed = time.perf_counter() - started
        candidate_files = {
            "text": result.get("output_text_file") or result.get("output_file"),
            "markdown": result.get("output_markdown_file"),
            "ocr_markdown": result.get("ocr_markdown_file"),
            "summary": str(Path(str(result.get("output_dir") or output_dir)) / "ocr_output" / "ocr_summary.json"),
            "image_analysis_summary": str(Path(str(result.get("output_dir") or output_dir)) / "image_analysis" / "analysis_summary.json"),
        }
        files = {
            key: str(path)
            for key, path in candidate_files.items()
            if path and Path(str(path)).exists()
        }
        progress_callback(
            {
                "stage": "completed",
                "state": "completed",
                "message": "文档解析完成",
                "extra": {"elapsed_seconds": round(elapsed, 2)},
            }
        )
        manager.update_job(
            job_id,
            status="completed",
            message="文档解析完成",
            finished_at=now_server_local_iso(),
            result=result,
            files=files,
            elapsed_seconds=round(elapsed, 2),
        )
    except asyncio.CancelledError:
        manager.update_job(
            job_id,
            status="canceled",
            message="任务已取消",
            finished_at=now_server_local_iso(),
        )
        raise
    except Exception as exc:
        progress_callback(
            {
                "stage": "error",
                "state": "failed",
                "message": str(exc),
                "extra": {"error": str(exc)},
            }
        )
        manager.update_job(
            job_id,
            status="failed",
            message=str(exc),
            finished_at=now_server_local_iso(),
            error=str(exc),
        )


__all__ = [
    "DOCUMENT_PROCESSING_JOBS",
    "DocumentProcessingJobManager",
    "run_document_processing_job",
]
