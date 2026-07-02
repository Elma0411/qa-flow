# 文件作用：执行批量完整 QA pipeline 的服务层编排。
# 关联说明：由 pipeline_batch_routes 调用，内部串联 qa、evaluation、storage、milvus、artifact 等服务。

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

from app.services.llm import LLMClientConfig, get_llm_client_pool

from qa import process_text_to_qa_one_step
from qa.augmentation import augment_qa_pairs
from qa.chunking import ENGINE_VERSION, build_tree_chunks
from qa.common import detect_language

from app.core.config import CONFIG
from app.core.logger import logger
from app.core.time_utils import now_server_local_iso
from app.services.pipeline_common import (
    _ARTIFACT_TTL_SECONDS,
    _compute_average_scores_for_result,
    _normalize_artifact_path,
    _parent_key,
    parse_few_shot_examples,
)
from app.services.artifacts import (
    delete_artifacts_now,
    delete_paths_now,
    get_owner_artifact_expire_at,
    register_temporary_artifacts,
)
from app.services.doc_chunks import (
    build_doc_id as build_tree_doc_id,
    store_doc_tree_chunks,
)
from app.services.evaluation import (
    execute_local_evaluation_blocking,
    filter_qa_pairs_by_threshold,
    run_llm_evaluation_batches,
)
from app.services.gpu import (
    clear_cuda_runtime_for_device,
    gpu_stage,
    release_gpu_job,
)
from app.services.knowledge_tagging import (
    OLD_MODEL_CLASSIFIER,
    classify_document_text,
    normalize_knowledge_classifier,
    release_knowledge_tagger_device_cache,
)
from app.services.milvus import MILVUS_AVAILABLE, store_qa_pairs_to_milvus
from app.services.pipeline_state import upsert_pipeline_task_status
from app.services.storage import (
    build_consolidated_entry,
    get_output_path,
    merge_consolidated_entries,
    sanitize_filename,
    write_consolidated_csv,
)
from app.services.unsupervised_evaluation import (
    UNSUPERVISED_EVALUATION_AVAILABLE,
    execute_unsupervised_suite_blocking,
)

async def run_batch_complete_pipeline_async(job_context: Dict[str, Any]) -> None:
    """
    核心异步流水线：针对一个或多个文件执行
    文本 -> (按 chunk) 直接生成问答对+来源事实 -> (可选) 评估 -> (可选) 向量存储。
    """
    task_id: str = job_context["task_id"]
    file_contents: List[Dict[str, Any]] = job_context["file_contents"]
    chunk_size: int = job_context["chunk_size"]
    qa_per_chunk: int = job_context["qa_per_chunk"]
    qa_detail_mode: str = job_context.get("qa_detail_mode") or "point"
    prompt_language: str = job_context.get("prompt_language") or "auto"
    enable_chunk_storage: bool = bool(job_context.get("enable_chunk_storage", True))
    chunk_storage_fail_fast: bool = bool(job_context.get("chunk_storage_fail_fast", False))
    try:
        chunking_prefix_max_depth = int(job_context.get("chunking_prefix_max_depth") or 4)
    except Exception:
        chunking_prefix_max_depth = 4
    chunking_prefix_max_depth = max(0, min(12, chunking_prefix_max_depth))
    chunking_split_type = str(job_context.get("chunking_split_type") or "").strip().lower() or None
    try:
        chunking_text_split_min_length = (
            max(1, int(job_context["chunking_text_split_min_length"]))
            if job_context.get("chunking_text_split_min_length") is not None
            else None
        )
    except Exception:
        chunking_text_split_min_length = None
    try:
        chunking_text_split_max_length = (
            max(1, int(job_context["chunking_text_split_max_length"]))
            if job_context.get("chunking_text_split_max_length") is not None
            else None
        )
    except Exception:
        chunking_text_split_max_length = None
    try:
        chunking_chunk_overlap = (
            max(0, int(job_context["chunking_chunk_overlap"]))
            if job_context.get("chunking_chunk_overlap") is not None
            else None
        )
    except Exception:
        chunking_chunk_overlap = None
    chunking_separator = (
        str(job_context.get("chunking_separator"))
        if job_context.get("chunking_separator") is not None
        else None
    )
    chunking_separators = job_context.get("chunking_separators")
    if not isinstance(chunking_separators, list):
        chunking_separators = None
    else:
        chunking_separators = [str(item) for item in chunking_separators if str(item).strip()]
    chunking_split_language = (
        str(job_context.get("chunking_split_language")).strip()
        if job_context.get("chunking_split_language") is not None
        else None
    )
    chunking_custom_separator = (
        str(job_context.get("chunking_custom_separator"))
        if job_context.get("chunking_custom_separator") is not None
        else None
    )
    chunking_manual_split_points = job_context.get("chunking_manual_split_points")
    if not isinstance(chunking_manual_split_points, list):
        chunking_manual_split_points = None
    chunking_markdown_heading_correction_enabled = bool(
        job_context.get("chunking_markdown_heading_correction_enabled", True)
    )
    include_evaluation: bool = job_context["include_evaluation"]
    include_unsupervised_evaluation: bool = bool(
        job_context.get("include_unsupervised_evaluation", False)
    )
    evaluation_method: str = job_context["evaluation_method"]
    if include_evaluation and evaluation_method in ("faithfulness", "answerability", "unsupervised_f1"):
        include_unsupervised_evaluation = True
    faithfulness_hypothesis_mode: str = str(
        job_context.get("faithfulness_hypothesis_mode") or "llm"
    ).strip().lower()
    try:
        faithfulness_hypothesis_max_concurrency = int(
            job_context.get("faithfulness_hypothesis_max_concurrency") or 8
        )
    except Exception:
        faithfulness_hypothesis_max_concurrency = 8
    faithfulness_hypothesis_max_concurrency = max(1, faithfulness_hypothesis_max_concurrency)
    filter_by_threshold: bool = job_context["filter_by_threshold"]
    score_threshold: float = job_context["score_threshold"]
    save_mode: str = job_context["save_mode"]
    enable_vector_storage: bool = job_context["enable_vector_storage"]
    status_data: Dict[str, Any] = job_context["status_data"]
    criteria_list: List[str] = job_context["criteria_list"]
    llm_config: Dict[str, Any] = job_context["llm_config"]
    llm_max_concurrent_requests = job_context.get("llm_max_concurrent_requests")
    max_concurrency: int = job_context["max_concurrency"]
    chunk_max_concurrency: int = job_context.get("chunk_max_concurrency", 8)
    chunk_max_attempts: int = max(1, int(job_context.get("chunk_max_attempts") or 2))
    augment_max_concurrency: int = job_context.get("augment_max_concurrency", 8)
    eval_max_concurrency: int = job_context["eval_max_concurrency"]
    question_type_mode: str = job_context.get("question_type_mode") or "mixed"
    question_types = job_context.get("question_types")
    question_type_weights = job_context.get("question_type_weights")
    few_shot_examples = parse_few_shot_examples(job_context.get("few_shot_examples"))
    knowledge_classifier = normalize_knowledge_classifier(job_context.get("knowledge_classifier"))
    use_category_prompt_templates = bool(job_context.get("use_category_prompt_templates", True))
    parsed_qt_weights = None
    if isinstance(question_type_weights, dict):
        parsed_qt_weights = question_type_weights
    elif isinstance(question_type_weights, str):
        try:
            parsed_qt_weights = json.loads(question_type_weights)
        except Exception:
            parsed_qt_weights = None
    augment_per_qa: int = job_context.get("augment_per_qa", 0)

    def build_categorized_facts_from_qa(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a categorized_facts list compatible with build_consolidated_entry()
        from one-step generated QA items (each item contains source_fact_text + source + category fields).
        """
        fact_map: Dict[str, Dict[str, Any]] = {}
        for qa in qa_items:
            atomic_fact = str(qa.get("source_fact_text") or "").strip()
            if not atomic_fact:
                continue
            entry = {
                "atomic_fact": atomic_fact,
                "source": str(qa.get("source") or "").strip() or None,
                "knowledge_category": str(qa.get("knowledge_category") or "").strip() or None,
                "knowledge_category_reason": str(qa.get("knowledge_category_reason") or "").strip() or "",
                "knowledge_category_confidence": qa.get("knowledge_category_confidence"),
                "language": detect_language(atomic_fact),
            }
            existing = fact_map.get(atomic_fact)
            if not existing:
                fact_map[atomic_fact] = entry
                continue
            try:
                if float(entry.get("knowledge_category_confidence") or 0.0) > float(
                    existing.get("knowledge_category_confidence") or 0.0
                ):
                    fact_map[atomic_fact] = entry
            except Exception:
                continue
        return list(fact_map.values())

    os.makedirs(CONFIG["outputs_dir"], exist_ok=True)

    progress_lock = asyncio.Lock()

    async def update_job_fields(**updates: Any) -> None:
        async with progress_lock:
            status_data.update(updates)
            status_data["updated_at"] = now_server_local_iso()
            upsert_pipeline_task_status(task_id, status_data)

    async def log_progress(message: str) -> None:
        logger.info("[batch %s] %s", task_id, message)
        await update_job_fields(message=message)

    async def update_file_progress(
        filename: str,
        stage: str,
        state: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        terminal: bool = False,
    ) -> None:
        async with progress_lock:
            files_progress: Dict[str, Any] = status_data.setdefault("file_progress", {})
            file_entry = files_progress.setdefault(
                filename,
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
                stage_entry.setdefault("extra", {}).update(extra)
            file_entry["status"] = state
            file_entry["message"] = message
            if terminal:
                file_entry["completed_at"] = now_server_local_iso()
            completed = sum(
                1 for f in files_progress.values() if f.get("status") == "completed"
            )
            failed = sum(
                1 for f in files_progress.values() if f.get("status") == "failed"
            )
            status_data["completed_files"] = completed
            status_data["failed_files"] = failed
            status_data["updated_at"] = now_server_local_iso()
            upsert_pipeline_task_status(task_id, status_data)

    await update_job_fields(
        status="processing",
        message="Batch started",
        started_at=now_server_local_iso(),
        total_files=len(file_contents),
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_single_file(file_info: Dict[str, Any]) -> Dict[str, Any]:
        filename = file_info["filename"]
        await update_file_progress(filename, "queued", "waiting", "等待进入流水线")

        if file_info["status"] != "success":
            error_msg = file_info.get("error", "文件读取失败")
            await update_file_progress(filename, "upload_validation", "failed", error_msg, terminal=True)
            return {
                "filename": filename,
                "status": "error",
                "error": error_msg,
                "processing_result": {
                    "filename": filename,
                    "status": "error",
                    "error": error_msg,
                },
            }

        async with semaphore:
            try:
                content = file_info.get("content", "") or ""
                content_format = str(file_info.get("content_format") or "").strip().lower()
                content_preview = (content[:8000] if isinstance(content, str) else "") or ""
                content_len = len(content) if isinstance(content, str) else 0
                debug_file: Optional[str] = None
                await update_file_progress(
                    filename,
                    "knowledge_tagging",
                    "processing",
                    "文档知识分类中",
                    extra={
                        "source_text_chars_total": content_len,
                        "source_text_excerpt": content_preview,
                    },
                )
                try:
                    def _run_knowledge_tagging() -> Any:
                        if knowledge_classifier == OLD_MODEL_CLASSIFIER:
                            with gpu_stage(task_id, "knowledge_tagging") as leased_device:
                                try:
                                    return classify_document_text(
                                        content,
                                        filename=filename,
                                        device_override=leased_device,
                                        classifier_mode=knowledge_classifier,
                                    )
                                finally:
                                    release_knowledge_tagger_device_cache(leased_device)
                                    clear_cuda_runtime_for_device(leased_device)
                        return classify_document_text(
                            content,
                            filename=filename,
                            classifier_mode=knowledge_classifier,
                        )

                    doc_tag = await asyncio.to_thread(_run_knowledge_tagging)
                except Exception as exc:
                    await update_file_progress(
                        filename,
                        "knowledge_tagging",
                        "failed",
                        f"知识分类失败: {str(exc)}",
                        terminal=True,
                    )
                    return {
                        "filename": filename,
                        "status": "error",
                        "error": f"knowledge_tagging_failed: {str(exc)}",
                        "processing_result": {
                            "filename": filename,
                            "status": "error",
                            "error": f"知识分类失败: {str(exc)}",
                        },
                        "debug_file": debug_file,
                    }

                await update_file_progress(
                    filename,
                    "knowledge_tagging",
                    "completed",
                    f"知识分类完成：{doc_tag.knowledge_category}",
                    extra={
                        "knowledge_category": doc_tag.knowledge_category,
                        "knowledge_category_confidence": doc_tag.knowledge_category_confidence,
                        "knowledge_category_reason": doc_tag.knowledge_category_reason,
                        "knowledge_category_source": doc_tag.knowledge_category_source,
                        "knowledge_classifier": knowledge_classifier,
                    },
                )
                safe_name = sanitize_filename(filename)
                debug_file = get_output_path(f"{task_id}_{safe_name}_one_step_debug", ".jsonl")
                await update_file_progress(
                    filename,
                    "qa_generation",
                    "processing",
                    "一步式生成问答对中",
                    extra={"debug_file": debug_file},
                )
                client = get_llm_client_pool().get_client(
                    LLMClientConfig(
                        api_base=llm_config.get("base_url"),
                        model_name=llm_config.get("model"),
                        api_key=llm_config.get("api_key"),
                        api_type=llm_config.get("api_type"),
                        model_version=llm_config.get("model_version"),
                        max_concurrent_requests=llm_max_concurrent_requests,
                    )
                )
                loop = asyncio.get_running_loop()
                gen_last_update = 0.0
                generation_chunk_details: List[Dict[str, Any]] = []
                generation_timing_summary: Dict[str, Any] = {}

                def _on_generation_progress(info: Dict[str, Any]) -> None:
                    nonlocal gen_last_update, generation_timing_summary
                    try:
                        event = str((info or {}).get("event") or "")
                        now = time.time()
                        if event == "start":
                            total_chunks = int((info or {}).get("total_chunks") or 0)
                            msg = f"一步式生成问答对中：共 {total_chunks} 个 chunk"
                            asyncio.run_coroutine_threadsafe(
                                update_file_progress(
                                    filename,
                                    "qa_generation",
                                    "processing",
                                    msg,
                                    extra={
                                        "chunks_total": total_chunks,
                                        "chunks_done": 0,
                                        "generation_timing": (info or {}).get("timing") or {},
                                        "llm_max_concurrent_requests": llm_max_concurrent_requests,
                                    },
                                ),
                                loop,
                            )
                            return

                        if event == "done":
                            generation_timing_summary = dict((info or {}).get("timing") or {})
                            raw_chunk_details = (info or {}).get("chunk_details")
                            if isinstance(raw_chunk_details, list):
                                generation_chunk_details[:] = [
                                    dict(item) for item in raw_chunk_details if isinstance(item, dict)
                                ]
                            asyncio.run_coroutine_threadsafe(
                                update_file_progress(
                                    filename,
                                    "qa_generation",
                                    "processing",
                                    "一步式生成问答对收尾中",
                                    extra={
                                        "chunks_total": int((info or {}).get("total_chunks") or 0),
                                        "chunks_done": int((info or {}).get("total_chunks") or 0),
                                        "total_items": int((info or {}).get("total_items") or 0),
                                        "generation_timing": generation_timing_summary,
                                        "generation_chunk_details": generation_chunk_details,
                                    },
                                ),
                                loop,
                            )
                            return

                        if event != "chunk_completed":
                            return

                        total_chunks = int((info or {}).get("total_chunks") or 0)
                        completed_chunks = int((info or {}).get("completed_chunks") or 0)
                        valid_items = int((info or {}).get("valid_items") or 0)
                        attempt_used = (info or {}).get("attempt_used")
                        chunk_index = (info or {}).get("chunk_index")
                        last_error = (info or {}).get("error")
                        chunk_timing = (info or {}).get("timing") or {}
                        chunk_detail = {
                            "chunk_index": chunk_index,
                            "attempt_used": attempt_used,
                            "candidate_questions": (info or {}).get("candidate_questions"),
                            "candidates_considered": (info or {}).get("candidates_considered"),
                            "valid_items": valid_items,
                            "dropped_reason_stats": (info or {}).get("dropped_reason_stats")
                            or (info or {}).get("dropped_answer_reasons")
                            or {},
                            "timing": chunk_timing if isinstance(chunk_timing, dict) else {},
                        }
                        if last_error:
                            chunk_detail["error"] = str(last_error)[:800]
                        generation_chunk_details.append(chunk_detail)
                        generation_chunk_details.sort(
                            key=lambda item: int(item.get("chunk_index") or 0)
                        )

                        if total_chunks <= 0:
                            return
                        # throttle: avoid excessive status writes
                        if completed_chunks != total_chunks and (now - gen_last_update) < 0.8:
                            return
                        gen_last_update = now

                        suffix = f"（本 chunk 有效 {valid_items} 条"
                        if attempt_used:
                            suffix += f"，尝试 {attempt_used}"
                        suffix += "）"
                        msg = f"一步式生成中：chunk {completed_chunks}/{total_chunks}{suffix}"
                        extra_payload = {
                            "chunks_total": total_chunks,
                            "chunks_done": completed_chunks,
                            "last_chunk_index": chunk_index,
                            "last_chunk_items": valid_items,
                            "last_chunk_timing": chunk_timing if isinstance(chunk_timing, dict) else {},
                            "generation_chunk_details": generation_chunk_details,
                        }
                        if last_error:
                            extra_payload["last_error"] = str(last_error)[:800]
                        asyncio.run_coroutine_threadsafe(
                            update_file_progress(
                                filename,
                                "qa_generation",
                                "processing",
                                msg,
                                extra=extra_payload,
                            ),
                            loop,
                        )
                    except Exception:
                        return

                def _write_debug_event(payload: Dict[str, Any]) -> None:
                    try:
                        with open(debug_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    except Exception:
                        return

                doc_id = build_tree_doc_id(filename, content)
                await update_file_progress(
                    filename,
                    "chunking",
                    "processing",
                    "Easy Dataset Python 切分中",
                    extra={
                        "doc_id": doc_id,
                        "engine_version": ENGINE_VERSION,
                        "split_type": chunking_split_type or "markdown",
                        "prefix_max_depth": chunking_prefix_max_depth,
                    },
                )
                try:
                    pre_split_chunks = file_info.get("pre_split_chunks")
                    pre_split_chunk_meta = file_info.get("pre_split_chunk_meta")
                    if (
                        isinstance(pre_split_chunks, list)
                        and pre_split_chunks
                        and all(isinstance(chunk, str) for chunk in pre_split_chunks)
                        and isinstance(pre_split_chunk_meta, list)
                        and pre_split_chunk_meta
                    ):
                        chunks_for_llm = [str(chunk) for chunk in pre_split_chunks]
                        chunks_meta = [
                            dict(item) for item in pre_split_chunk_meta if isinstance(item, dict)
                        ]
                        chunking_report = dict(file_info.get("chunking_report") or {})
                        chunking_report.setdefault("mode", "pre_split_integrated")
                        chunking_report.setdefault("effective_split_type", chunking_split_type or "markdown")
                        chunking_report.setdefault("chunks", len(chunks_meta))
                        chunking_report.setdefault("text_chars_total", len(content))
                    else:
                        chunks_for_llm, chunks_meta, chunking_report = await asyncio.to_thread(
                            build_tree_chunks,
                            content,
                            chunk_size=max(1, int(chunk_size)),
                            original_filename=filename,
                            task_id=task_id,
                            doc_id=doc_id,
                            prefix_max_depth=chunking_prefix_max_depth,
                            debug_writer=_write_debug_event,
                            split_type=chunking_split_type,
                            text_split_min_length=chunking_text_split_min_length,
                            text_split_max_length=chunking_text_split_max_length,
                            chunk_overlap=chunking_chunk_overlap,
                            separator=chunking_separator,
                            separators=chunking_separators,
                            split_language=chunking_split_language,
                            custom_separator=chunking_custom_separator,
                            manual_split_points=chunking_manual_split_points,
                            force_heading_correction=(
                                content_format == "markdown"
                                and bool(chunking_markdown_heading_correction_enabled)
                            ),
                        )
                except Exception as exc:
                    exc_code = str(getattr(exc, "code", "chunking_failed") or "chunking_failed")
                    exc_message = str(getattr(exc, "message", str(exc)) or str(exc))
                    await update_file_progress(
                        filename,
                        "chunking",
                        "failed",
                        f"Chunking 失败: {exc_message}",
                        extra={
                            "error_code": exc_code,
                            "error": exc_message[:1200],
                        },
                        terminal=True,
                    )
                    return {
                        "filename": filename,
                        "status": "error",
                        "error": f"{exc_code}: {exc_message}",
                        "processing_result": {
                            "filename": filename,
                            "status": "error",
                            "error": f"{exc_code}: {exc_message}",
                        },
                        "debug_file": debug_file,
                    }
                for meta in chunks_meta or []:
                    if not isinstance(meta, dict):
                        continue
                    meta["doc_id"] = doc_id
                    meta["task_id"] = task_id
                    meta["original_filename"] = filename
                await update_file_progress(
                    filename,
                    "chunking",
                    "completed",
                    f"切分完成：{len(chunks_meta)} 个 chunk",
                    extra={
                        "chunking_report": {
                            "mode": (chunking_report or {}).get("mode"),
                            "effective_split_type": (chunking_report or {}).get("effective_split_type"),
                            "chunks": (chunking_report or {}).get("chunks"),
                            "text_chars_total": (chunking_report or {}).get("text_chars_total"),
                            "toc_items": (chunking_report or {}).get("toc_items"),
                            "heading_correction": (chunking_report or {}).get("heading_correction"),
                        },
                    },
                )

                await update_file_progress(
                    filename,
                    "chunk_storage",
                    "processing",
                    "chunk 入库中（doc_tree_chunks）",
                    extra={"enable_chunk_storage": enable_chunk_storage},
                )
                try:
                    chunk_storage_result = await asyncio.to_thread(
                        store_doc_tree_chunks,
                        chunks_meta,
                        enable=enable_chunk_storage,
                    )
                except Exception as exc:
                    chunk_storage_result = {
                        "success": False,
                        "message": f"chunk 入库异常: {exc}",
                        "stored_count": 0,
                    }
                chunk_storage_success = (not enable_chunk_storage) or bool(
                    (chunk_storage_result or {}).get("success")
                )
                await update_file_progress(
                    filename,
                    "chunk_storage",
                    "completed" if chunk_storage_success else "failed",
                    (chunk_storage_result or {}).get("message") or ("chunk 入库成功" if chunk_storage_success else "chunk 入库失败"),
                    extra={
                        "stored_count": (chunk_storage_result or {}).get("stored_count"),
                        "collection_name": (chunk_storage_result or {}).get("collection_name"),
                    },
                )
                if enable_chunk_storage and chunk_storage_fail_fast and not chunk_storage_success:
                    await update_file_progress(
                        filename,
                        "chunk_storage",
                        "failed",
                        "chunk 入库失败（fail_fast=True）",
                        terminal=True,
                    )
                    return {
                        "filename": filename,
                        "status": "error",
                        "error": (chunk_storage_result or {}).get("message") or "chunk_storage_failed",
                        "processing_result": {
                            "filename": filename,
                            "status": "error",
                            "error": (chunk_storage_result or {}).get("message") or "chunk_storage_failed",
                            "doc_id": doc_id,
                            "chunk_storage_result": chunk_storage_result,
                        },
                    }

                generation_start = time.time()
                qa_data = await asyncio.to_thread(
                    process_text_to_qa_one_step,
                    client,
                    content,
                    {
                        "chunk_size": chunk_size,
                        "qa_per_chunk": qa_per_chunk,
                        "qa_detail_mode": qa_detail_mode,
                        "prompt_language": prompt_language,
                        "chunk_max_concurrency": chunk_max_concurrency,
                        "strict_max_attempts": chunk_max_attempts,
                        "model": llm_config["model"],
                        "request_timeout": CONFIG.get("request_timeout", 120),
                        "question_type_mode": question_type_mode,
                        "question_types": question_types,
                        "question_type_weights": parsed_qt_weights,
                        "fixed_knowledge_category": doc_tag.knowledge_category,
                        "fixed_knowledge_category_confidence": doc_tag.knowledge_category_confidence,
                        "fixed_knowledge_category_reason": doc_tag.knowledge_category_reason,
                        "use_category_prompt_templates": use_category_prompt_templates,
                        "few_shot_examples": few_shot_examples,
                        "debug_file": debug_file,
                        "include_chunk_index": True,
                        "pre_split_chunks": chunks_for_llm,
                        "pre_split_chunk_meta": chunks_meta,
                    },
                    original_filename=filename,
                    progress_callback=_on_generation_progress,
                )
                # Replace model-provided `source` with stable chunk_id for traceability.
                chunk_id_map = {
                    int(m.get("chunk_index") or 0): str(m.get("chunk_id") or "")
                    for m in (chunks_meta or [])
                    if isinstance(m, dict) and int(m.get("chunk_index") or 0) > 0 and str(m.get("chunk_id") or "").strip()
                }
                if chunk_id_map:
                    for qa in qa_data or []:
                        if not isinstance(qa, dict):
                            continue
                        try:
                            idx = int(qa.get("chunk_index") or 0)
                        except Exception:
                            continue
                        if idx > 0 and idx in chunk_id_map:
                            qa["source"] = chunk_id_map[idx]
                generation_duration = time.time() - generation_start
                if not qa_data:
                    raise ValueError(
                        f"一步式问答生成失败：未生成任何有效 items（请下载调试日志查看 LLM 原始响应：{os.path.basename(debug_file)}）"
                    )

                categorized_facts = build_categorized_facts_from_qa(qa_data)
                facts: List[Dict[str, Any]] = []
                primary_count = len(qa_data)
                for qa in qa_data:
                    qa.setdefault("is_primary", True)
                    qa.setdefault("is_augmented", False)
                augmented_qas: List[Dict[str, Any]] = []
                if augment_per_qa > 0 and qa_data:
                    await update_file_progress(
                        filename,
                        "qa_augmentation",
                        "processing",
                        "问答增广中",
                        extra={"augment_total": len(qa_data), "augment_done": 0},
                    )

                    aug_last_update = 0.0

                    def _on_augment_progress(info: Dict[str, Any]) -> None:
                        nonlocal aug_last_update
                        try:
                            event = str((info or {}).get("event") or "")
                            now = time.time()
                            total = int((info or {}).get("total") or 0)
                            completed = int((info or {}).get("completed") or 0)
                            total_augmented = int((info or {}).get("total_augmented") or 0)

                            if event == "start":
                                msg = f"问答增广中：0/{total}（每题增广 {augment_per_qa} 条）"
                                asyncio.run_coroutine_threadsafe(
                                    update_file_progress(
                                        filename,
                                        "qa_augmentation",
                                        "processing",
                                        msg,
                                        extra={
                                            "augment_total": total,
                                            "augment_done": 0,
                                            "total_augmented": 0,
                                        },
                                    ),
                                    loop,
                                )
                                return

                            if event != "item_completed":
                                return
                            if total <= 0:
                                return
                            if completed != total and (now - aug_last_update) < 0.8:
                                return
                            aug_last_update = now

                            msg = f"问答增广中：{completed}/{total}（累计生成 {total_augmented} 条）"
                            asyncio.run_coroutine_threadsafe(
                                update_file_progress(
                                    filename,
                                    "qa_augmentation",
                                    "processing",
                                    msg,
                                    extra={
                                        "augment_total": total,
                                        "augment_done": completed,
                                        "total_augmented": total_augmented,
                                    },
                                ),
                                loop,
                            )
                        except Exception:
                            return

                    augmented_qas = await asyncio.to_thread(
                        augment_qa_pairs,
                        qa_data,
                        augment_per_qa=augment_per_qa,
                        client=client,
                        model=llm_config["model"],
                        max_workers=augment_max_concurrency,
                        progress_callback=_on_augment_progress,
                    )
                    await update_file_progress(
                        filename,
                        "qa_augmentation",
                        "completed",
                        f"问答增广完成：生成 {len(augmented_qas)} 条",
                        extra={
                            "augment_total": len(qa_data),
                            "augment_done": len(qa_data),
                            "total_augmented": len(augmented_qas),
                        },
                    )
                    # 将增广问句附加到主问答的 similar_questions，而不再作为独立条目存储
                    primary_map = {_parent_key(p): p for p in qa_data}
                    for aug in augmented_qas:
                        aug.setdefault("is_primary", False)
                        aug["is_augmented"] = True
                        parent_key = aug.get("variant_of_key") or _parent_key(aug)
                        primary = primary_map.get(parent_key)
                        if not primary:
                            continue
                        sims = primary.setdefault("similar_questions", [])
                        sims.append(
                            {
                                "question": aug.get("question", ""),
                                "answer": aug.get("answer", ""),
                                "question_type": aug.get("question_type"),
                                "answer_explanation": aug.get("answer_explanation", ""),
                                "is_augmented": True,
                            }
                        )
                all_qas = qa_data
                await update_file_progress(
                    filename,
                    "qa_generation",
                    "completed",
                    f"生成 {len(qa_data)} 条主问答，增广 {len(augmented_qas)} 条（已附加为 similar_questions）",
                    extra={
                        "generation_seconds": generation_duration,
                        "generation_timing": generation_timing_summary,
                        "generation_chunk_details": generation_chunk_details,
                        "qa_generated": len(qa_data),
                    },
                )

                unsupervised_duration: Optional[float] = None
                unsupervised_summary: Optional[Dict[str, Any]] = None
                if include_unsupervised_evaluation and qa_data:
                    if not UNSUPERVISED_EVALUATION_AVAILABLE:
                        await update_file_progress(
                            filename,
                            "unsupervised_evaluation",
                            "completed",
                            "无监督评价跳过：缺少 transformers/torch 或 sentence-transformers 依赖",
                            extra={"available": False},
                        )
                    else:
                        await update_file_progress(
                            filename,
                            "unsupervised_evaluation",
                            "processing",
                            "无监督评价中（Faithfulness/Answerability/Coverage/F1）",
                        )
                        try:
                            unsup_start = time.time()
                            unsupervised_summary = await asyncio.to_thread(
                                execute_unsupervised_suite_blocking,
                                qa_data,
                                only_primary=True,
                                hypothesis_mode=faithfulness_hypothesis_mode,
                                llm_api_key=llm_config.get("api_key"),
                                llm_base_url=llm_config.get("base_url"),
                                llm_model=llm_config.get("model"),
                                llm_max_retries=llm_config.get("max_retries"),
                                llm_max_concurrency=faithfulness_hypothesis_max_concurrency,
                                gpu_job_id=task_id,
                            )
                            unsupervised_duration = time.time() - unsup_start
                            await update_file_progress(
                                filename,
                                "unsupervised_evaluation",
                                "completed",
                                "无监督评价完成",
                                extra={
                                    **(unsupervised_summary or {}),
                                    "unsupervised_seconds": unsupervised_duration,
                                },
                            )
                        except Exception as unsup_exc:
                            await update_file_progress(
                                filename,
                                "unsupervised_evaluation",
                                "completed",
                                f"无监督评价失败，已跳过：{unsup_exc}",
                                extra={"error": str(unsup_exc)[:800]},
                            )

                evaluation_results: Optional[Dict[str, Any]] = None
                filtered_qa_data: Optional[List[Dict[str, Any]]] = None
                local_evaluation_results: Optional[Dict[str, Any]] = None
                evaluation_duration: Optional[float] = None
                evaluation_file: Optional[str] = None

                if include_evaluation and qa_data:
                    current_stage = "evaluation"
                    # NOTE: Unsupervised methods compute during the dedicated `unsupervised_evaluation` stage.
                    # Keep `evaluation_seconds` for LLM/local evaluation only, to avoid confusing 0.00s in UI.
                    timed_evaluation = evaluation_method in ("llm", "local")
                    eval_start = time.time() if timed_evaluation else 0.0

                    async def evaluation_progress(message: str) -> None:
                        await update_file_progress(
                            filename,
                            current_stage,
                            "processing",
                            f"{filename}: {message}",
                        )

                    await update_file_progress(
                        filename,
                        current_stage,
                        "processing",
                        f"准备执行 {evaluation_method} 评估",
                    )

                    eval_input = qa_data + augmented_qas
                    llm_evaluation_results: Optional[Dict[str, Any]] = None
                    if evaluation_method == "llm":
                        llm_evaluation_results = await run_llm_evaluation_batches(
                            eval_input,
                            criteria_list,
                            evaluation_progress,
                            max_eval_concurrency=eval_max_concurrency,
                            llm_config=llm_config,
                        )

                    if evaluation_method == "local":
                        local_evaluation_results = await asyncio.to_thread(
                            execute_local_evaluation_blocking,
                            eval_input,
                            _LOCAL_EVAL_COMPAT_FLAG,
                            gpu_job_id=task_id,
                        )
                        if local_evaluation_results:
                            await evaluation_progress("本地评估完成")

                    if evaluation_method == "llm":
                        evaluation_results = llm_evaluation_results
                    elif evaluation_method == "local":
                        evaluation_results = local_evaluation_results
                    elif evaluation_method in ("faithfulness", "answerability", "unsupervised_f1", "unsupervised"):
                        # Unsupervised methods use scores written into `unsupervised_evaluation`.
                        if not UNSUPERVISED_EVALUATION_AVAILABLE:
                            await evaluation_progress(
                                "无监督评价不可用：缺少 transformers/torch 或 sentence-transformers 依赖"
                            )
                            evaluation_results = None
                        else:
                            evaluation_results = {"method": evaluation_method, "results": []}
                            await evaluation_progress("无监督评价完成（已写入 unsupervised_evaluation）")

                    # 为后续过滤/存储补全 average_score（LLM 评估默认不自带）
                    if (
                        evaluation_method == "llm"
                        and evaluation_results
                        and evaluation_results.get("results")
                    ):
                        for res in evaluation_results["results"]:
                            try:
                                res["average_score"] = _compute_average_scores_for_result(res, criteria_list)
                            except Exception:
                                res["average_score"] = 0.0

                    # 将增广结果根据评估附加到 similar_questions，可选过滤
                    eval_scores_map: Dict[str, float] = {}
                    evaluation_scores_available = False
                    if evaluation_results and evaluation_results.get("results"):
                        for res in evaluation_results.get("results", []):
                            key = f"{res.get('question','')}|||{res.get('answer','')}"
                            try:
                                eval_scores_map[key] = float(res.get("average_score", 0.0) or 0.0)
                            except Exception:
                                eval_scores_map[key] = 0.0
                        evaluation_scores_available = True
                    elif evaluation_method in ("faithfulness", "answerability", "unsupervised_f1", "unsupervised"):
                        method_key = evaluation_method
                        if method_key == "unsupervised":
                            method_key = "unsupervised_f1"
                        score_key = {
                            "faithfulness": "faithfulness",
                            "answerability": "answerability",
                            "unsupervised_f1": "unsupervised_f1",
                        }.get(method_key, "faithfulness")
                        for primary in qa_data:
                            ue = primary.get("unsupervised_evaluation") or {}
                            scores = ue.get("scores") if isinstance(ue, dict) else {}
                            raw = scores.get(score_key) if isinstance(scores, dict) else None
                            try:
                                val = float(raw) if raw is not None else 0.0
                            except Exception:
                                val = 0.0
                            primary["average_score"] = val
                            primary["evaluation_method"] = method_key
                            eval_scores_map[_parent_key(primary)] = val
                        evaluation_scores_available = bool(eval_scores_map)

                    filtered_primary: List[Dict[str, Any]] = []
                    for primary in qa_data:
                        sims = []
                        for aug in augmented_qas:
                            parent_key = aug.get("variant_of_key") or _parent_key(aug)
                            if parent_key != _parent_key(primary):
                                continue
                            aug_score = eval_scores_map.get(_parent_key(aug), 0.0)
                            if filter_by_threshold and evaluation_scores_available and aug_score < score_threshold:
                                continue
                            sims.append(
                                {
                                    "question": aug.get("question", ""),
                                    "answer": aug.get("answer", ""),
                                    "question_type": aug.get("question_type"),
                                    "answer_explanation": aug.get("answer_explanation", ""),
                                    "score": aug_score if evaluation_scores_available else None,
                                    "is_augmented": True,
                                }
                            )
                        primary["similar_questions"] = sims
                        p_score = eval_scores_map.get(_parent_key(primary), 0.0)
                        if filter_by_threshold and evaluation_scores_available and p_score < score_threshold:
                            continue
                        filtered_primary.append(primary)

                    if filter_by_threshold and evaluation_scores_available:
                        filter_info = {
                            "threshold": score_threshold,
                            "original_count": len(qa_data),
                            "filtered_count": len(filtered_primary),
                            "removed_count": len(qa_data) - len(filtered_primary),
                        }
                        qa_data = filtered_primary
                        filtered_qa_data = filtered_primary
                        if evaluation_results:
                            evaluation_results["filter_info"] = filter_info
                    else:
                        filtered_qa_data = None

                    evaluation_duration = (time.time() - eval_start) if timed_evaluation else None
                    # 将完整评估结果（含原始 LLM 响应）落盘，便于排查
                    if evaluation_results:
                        eval_safe_name = sanitize_filename(filename or "evaluation")
                        evaluation_file = get_output_path(f"{task_id}_{eval_safe_name}_evaluation", ".json")
                        with open(evaluation_file, "w", encoding="utf-8") as f:
                            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
                    await update_file_progress(
                        filename,
                        current_stage,
                        "completed",
                        "问答评估完成",
                        {"filtered": len(filtered_qa_data or qa_data)},
                    )
                else:
                    # 无评估时也附加相似问句
                    for primary in qa_data:
                        sims = []
                        for aug in augmented_qas:
                            parent_key = aug.get("variant_of_key") or _parent_key(aug)
                            if parent_key != _parent_key(primary):
                                continue
                            sims.append(
                                {
                                    "question": aug.get("question", ""),
                                    "answer": aug.get("answer", ""),
                                    "question_type": aug.get("question_type"),
                                    "answer_explanation": aug.get("answer_explanation", ""),
                                    "is_augmented": True,
                                }
                            )
                        primary["similar_questions"] = sims
                    filtered_qa_data = None

                final_qa_len = len(qa_data)
                await update_file_progress(
                    filename,
                    "completed",
                    "completed",
                    "流水线完成",
                    {"qa_pairs": final_qa_len},
                    terminal=True,
                )

                file_result = {
                    "filename": filename,
                    "status": "success",
                    "doc_id": doc_id,
                    "counts": {
                        "facts": len(categorized_facts),
                        "categorized": len(categorized_facts),
                        "qa_pairs": final_qa_len,
                        "chunks": len(chunks_meta or []),
                    },
                    "chunking_report": chunking_report,
                    "chunk_storage_result": chunk_storage_result,
                }

                return {
                    "filename": filename,
                    "status": "success",
                    "doc_id": doc_id,
                    "chunking_report": chunking_report,
                    "chunk_storage_result": chunk_storage_result,
                    "facts": facts,
                    "categorized_facts": categorized_facts,
                    "qa_data": qa_data,
                    "source_text_excerpt": content_preview,
                    "source_text_chars_total": content_len,
                    "unsupervised_scores": (unsupervised_summary or {}).get("scores")
                    if include_unsupervised_evaluation
                    else None,
                    "evaluation_results": evaluation_results,
                    "evaluation_file": evaluation_file,
                    "debug_file": debug_file,
                    "filtered_qa_data": filtered_qa_data,
                    "processing_result": file_result,
                    "timing": {
                        "ocr_seconds": float(file_info.get("ocr_seconds") or 0.0)
                        if isinstance(file_info.get("ocr_seconds"), (int, float))
                        else 0.0,
                        "generation_seconds": generation_duration,
                        "generation_avg_seconds_per_qa": (generation_duration / primary_count)
                        if primary_count
                        else None,
                        "generation_detail": generation_timing_summary,
                        "generation_chunk_details": generation_chunk_details,
                        "qa_generated": len(qa_data),
                        "unsupervised_seconds": unsupervised_duration,
                        "unsupervised_qa_scored": (unsupervised_summary or {}).get("computed", 0)
                        if include_unsupervised_evaluation
                        else 0,
                        "evaluation_seconds": evaluation_duration,
                        "evaluation_avg_seconds_per_qa": (evaluation_duration / primary_count)
                        if evaluation_duration and primary_count
                        else None,
                        "qa_evaluated": primary_count if include_evaluation else 0,
                    },
                }
            except Exception as exc:
                logger.exception("批量流水线处理文件失败: %s", filename)
                await update_file_progress(filename, "error", "failed", str(exc), terminal=True)
                return {
                    "filename": filename,
                    "status": "error",
                    "error": str(exc),
                    "processing_result": {
                        "filename": filename,
                        "status": "error",
                        "error": str(exc),
                    },
                    "debug_file": debug_file if "debug_file" in locals() else None,
                }

    await log_progress(f"dispatching {len(file_contents)} file(s)")
    file_tasks = [asyncio.create_task(process_single_file(file_info)) for file_info in file_contents]
    results = await asyncio.gather(*file_tasks, return_exceptions=True)

    successful_files: List[Dict[str, Any]] = []
    failed_messages: List[str] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error("[batch %s] file task exception: %s", task_id, str(result))
            failed_messages.append(str(result))
        elif isinstance(result, dict):
            if result.get("status") == "success":
                successful_files.append(result)
            else:
                failed_messages.append(result.get("error") or "Unknown error")
        else:
            failed_messages.append(str(result))

    consolidated_entries: List[Dict[str, Any]] = []
    evaluation_file_map: Dict[str, Optional[str]] = {}
    debug_file_map: Dict[str, Optional[str]] = {}
    orphan_artifacts: List[str] = []
    if successful_files:
        for file_result in successful_files:
            try:
                evaluation_file_map[file_result.get("filename")] = file_result.get("evaluation_file")
                debug_file_map[file_result.get("filename")] = file_result.get("debug_file")
                timing = file_result.get("timing") or {}
                entry = build_consolidated_entry(
                    task_id=task_id,
                    original_filename=file_result.get("filename"),
                    facts=file_result.get("facts", []),
                    categorized_facts=file_result.get("categorized_facts", []),
                    qa_data=file_result.get("qa_data", []),
                    evaluation_results=file_result.get("evaluation_results"),
                    filtered_qa_data=file_result.get("filtered_qa_data"),
                    include_evaluation=include_evaluation,
                    include_unsupervised_evaluation=include_unsupervised_evaluation,
                    evaluation_method=evaluation_method,
                    filter_by_threshold=filter_by_threshold,
                    score_threshold=score_threshold,
                    chunk_size=chunk_size,
                    qa_per_chunk=qa_per_chunk,
                    qa_detail_mode=qa_detail_mode,
                    prompt_language=prompt_language,
                    llm_model=llm_config["model"],
                    ocr_seconds=timing.get("ocr_seconds"),
                    generation_seconds=timing.get("generation_seconds"),
                    unsupervised_seconds=timing.get("unsupervised_seconds"),
                    evaluation_seconds=timing.get("evaluation_seconds"),
                )
                if isinstance(entry.get("payload"), dict):
                    entry_timing = entry["payload"].setdefault("timing", {})
                    if isinstance(entry_timing, dict):
                        for detail_key in ("generation_detail", "generation_chunk_details"):
                            if detail_key in timing:
                                entry_timing[detail_key] = timing.get(detail_key)
                excerpt = file_result.get("source_text_excerpt")
                if isinstance(excerpt, str) and excerpt:
                    entry.setdefault("payload", {})["source_text_excerpt"] = excerpt
                    entry["payload"]["source_text_chars_total"] = int(file_result.get("source_text_chars_total") or 0)
                consolidated_entries.append(entry)
            except Exception as build_err:
                logger.exception(
                    "[batch %s] consolidate file %s failed: %s",
                    task_id,
                    file_result.get("filename"),
                    build_err,
                )
                failed_messages.append(f"Consolidation failed for {file_result.get('filename')}: {build_err}")

    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("status") == "success":
            continue
        for key in ("debug_file", "evaluation_file"):
            candidate = _normalize_artifact_path(result.get(key))
            if candidate:
                orphan_artifacts.append(candidate)

    final_status = "completed" if consolidated_entries else "failed"
    final_message = f"Batch finished: {len(consolidated_entries)} success, {len(failed_messages)} failed"
    if final_status == "completed" and failed_messages:
        final_message = f"Batch finished with partial failures ({len(failed_messages)} files)"

    def _finalize_pipeline_artifacts(record: Dict[str, Any], artifact_paths: List[Optional[str]]) -> Dict[str, Any]:
        normalized_paths = [path for path in [_normalize_artifact_path(p) for p in artifact_paths] if path]
        milvus_result = record.get("vector_storage_result") if isinstance(record.get("vector_storage_result"), dict) else {}
        milvus_success = bool(milvus_result.get("success"))
        if milvus_success:
            delete_paths_now(normalized_paths, reason="milvus_ingested")
            record["history_source"] = "milvus"
            record["milvus_task_id"] = task_id
            record["artifacts_deleted"] = True
            record["artifacts_expire_at"] = None
            for key in (
                "consolidated_json",
                "consolidated_csv",
                "evaluation_json",
                "debug_jsonl",
                "evaluation_json_files",
                "debug_json_files",
            ):
                if key in record:
                    record[key] = None
            return record
        register_temporary_artifacts(
            owner_kind="pipeline_task",
            owner_id=task_id,
            artifact_kind="pipeline_artifact",
            paths=normalized_paths,
            ttl_seconds=_ARTIFACT_TTL_SECONDS,
        )
        record["history_source"] = "artifacts"
        record["milvus_task_id"] = None
        record["artifacts_deleted"] = False
        record["artifacts_expire_at"] = get_owner_artifact_expire_at("pipeline_task", task_id)
        return record

    output_records: List[Dict[str, Any]] = []
    if consolidated_entries:
        if save_mode == "separate":
            for entry in consolidated_entries:
                safe_name = sanitize_filename(entry["filename"] or "file")
                json_path = get_output_path(f"{task_id}_{safe_name}_consolidated", ".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(entry["payload"], f, ensure_ascii=False, indent=2)
                csv_path = get_output_path(f"{task_id}_{safe_name}_consolidated", ".csv")
                write_consolidated_csv(entry["payload"].get("items", []), csv_path)
                evaluation_json = evaluation_file_map.get(entry["filename"])
                record: Dict[str, Any] = {
                    "mode": "separate",
                    "source_file": entry["filename"],
                    "consolidated_json": json_path,
                    "consolidated_csv": csv_path,
                    "qa_pairs": entry["payload"].get("counts", {}).get("qa_pairs", 0),
                    "filtered_pairs": entry["payload"].get("counts", {}).get("filtered_qa_pairs", 0),
                    "timing": entry["payload"].get("timing") or {},
                }
                if evaluation_json:
                    record["evaluation_json"] = evaluation_json
                debug_json = debug_file_map.get(entry["filename"])
                if debug_json:
                    record["debug_jsonl"] = debug_json
                if enable_vector_storage and MILVUS_AVAILABLE:
                    milvus_result = store_qa_pairs_to_milvus(json_path, enable_vector_storage)
                    record["vector_storage_result"] = milvus_result
                output_records.append(
                    _finalize_pipeline_artifacts(
                        record,
                        [json_path, csv_path, evaluation_json, debug_json],
                    )
                )
        else:
            combined_payload = merge_consolidated_entries(
                task_id=task_id,
                entries=consolidated_entries,
                chunk_size=chunk_size,
                qa_per_chunk=qa_per_chunk,
                qa_detail_mode=qa_detail_mode,
                prompt_language=prompt_language,
                include_evaluation=include_evaluation,
                include_unsupervised_evaluation=include_unsupervised_evaluation,
                evaluation_method=evaluation_method,
                filter_by_threshold=filter_by_threshold,
                score_threshold=score_threshold,
                llm_model=llm_config["model"],
            )
            json_path = get_output_path(f"{task_id}_consolidated_batch", ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(combined_payload, f, ensure_ascii=False, indent=2)
            csv_path = get_output_path(f"{task_id}_consolidated_batch", ".csv")
            write_consolidated_csv(combined_payload.get("items", []), csv_path)
            record = {
                "mode": "unified",
                "source_files": [entry["filename"] for entry in consolidated_entries],
                "consolidated_json": json_path,
                "consolidated_csv": csv_path,
                "qa_pairs": combined_payload.get("counts", {}).get("qa_pairs", 0),
                "filtered_pairs": combined_payload.get("counts", {}).get("filtered_qa_pairs", 0),
                "timing": combined_payload.get("timing") or {},
            }
            eval_files = [f.get("evaluation_file") for f in successful_files if f.get("evaluation_file")]
            if eval_files:
                record["evaluation_json_files"] = eval_files
            debug_files = [f.get("debug_file") for f in successful_files if f.get("debug_file")]
            if debug_files:
                record["debug_json_files"] = debug_files
            if enable_vector_storage and MILVUS_AVAILABLE:
                milvus_result = store_qa_pairs_to_milvus(json_path, enable_vector_storage)
                record["vector_storage_result"] = milvus_result
            output_records.append(
                _finalize_pipeline_artifacts(
                    record,
                    [json_path, csv_path, *(eval_files or []), *(debug_files or [])],
                )
            )

        if orphan_artifacts:
            register_temporary_artifacts(
                owner_kind="pipeline_task",
                owner_id=task_id,
                artifact_kind="pipeline_orphan_artifact",
                paths=orphan_artifacts,
                ttl_seconds=_ARTIFACT_TTL_SECONDS,
            )

        artifacts_deleted = bool(output_records) and all(bool(item.get("artifacts_deleted")) for item in output_records) and not orphan_artifacts
        artifacts_expire_at = get_owner_artifact_expire_at("pipeline_task", task_id)
        task_history_source = (
            "milvus"
            if output_records and all(str(item.get("history_source") or "") == "milvus" for item in output_records) and not orphan_artifacts
            else "artifacts"
        )
        task_milvus_id = task_id if any(str(item.get("history_source") or "") == "milvus" for item in output_records) else None

        await update_job_fields(
            status=final_status,
            message=final_message,
            outputs=output_records,
            history_source=task_history_source,
            milvus_task_id=task_milvus_id,
            artifacts_deleted=artifacts_deleted,
            artifacts_expire_at=(artifacts_expire_at or None),
            details={"failed_reasons": failed_messages} if failed_messages else {},
        )
        await log_progress(f"batch done: {final_message}")
    elif orphan_artifacts:
        register_temporary_artifacts(
            owner_kind="pipeline_task",
            owner_id=task_id,
            artifact_kind="pipeline_orphan_artifact",
            paths=orphan_artifacts,
            ttl_seconds=_ARTIFACT_TTL_SECONDS,
        )
        await update_job_fields(
            status=final_status,
            message=final_message,
            outputs=[],
            history_source="artifacts",
            milvus_task_id=None,
            artifacts_deleted=False,
            artifacts_expire_at=get_owner_artifact_expire_at("pipeline_task", task_id),
            details={"failed_reasons": failed_messages} if failed_messages else {},
        )
        await log_progress(f"batch done: {final_message}")
    else:
        await update_job_fields(
            status=final_status,
            message=final_message,
            outputs=[],
            history_source="artifacts",
            milvus_task_id=None,
            artifacts_deleted=False,
            artifacts_expire_at=None,
            details={"failed_reasons": failed_messages} if failed_messages else {},
        )
        await log_progress(f"batch done: {final_message}")

__all__ = ["run_batch_complete_pipeline_async"]
