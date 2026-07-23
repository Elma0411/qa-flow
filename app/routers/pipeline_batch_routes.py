# 文件作用：承载批量上传完整流水线接口的参数解析和任务调度。
# 关联说明：实际 pipeline 执行已下沉到 app.services.pipeline_execution，router 只保留 API 契约。

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from qa.chunking import SUPPORTED_SPLIT_TYPES

from app.core.config import ACTIVE_BATCH_JOBS, CONFIG, LLM_EVALUATION_METRICS
from app.core.logger import logger
from app.core.time_utils import now_server_local_iso
from app.services.pipeline_common import parse_few_shot_examples
from app.services.gpu import admit_gpu_job, release_gpu_job
from app.services.knowledge_tagging import normalize_knowledge_classifier
from app.services.ocr import resolve_uploaded_files_with_auto_ocr
from app.services.pipeline_execution import run_batch_complete_pipeline_async
from app.services.pipeline_state import (
    get_pipeline_store_path,
    get_pipeline_task_status,
    upsert_pipeline_task_status,
)
from app.services.storage import resolve_batch_concurrency
from app.services.unsupervised_evaluation import validate_evaluation_model_name

router = APIRouter()
_LOCAL_EVAL_COMPAT_FLAG = True


def _decode_chunking_form_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = str(value)
    if not raw.strip():
        return None
    return raw.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")


def _parse_chunking_separators(value: Optional[str]) -> Optional[List[str]]:
    decoded = _decode_chunking_form_text(value)
    if not decoded:
        return None
    stripped = decoded.strip()
    if not stripped:
        return None
    items: List[str]
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="chunking_separators 不是合法的 JSON 数组") from exc
        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="chunking_separators 必须是字符串数组")
        items = []
        for raw_item in payload:
            normalized = _decode_chunking_form_text(str(raw_item))
            if normalized:
                items.append(normalized)
    else:
        items = []
        normalized_source = stripped.replace("，", ",").replace("\r", "\n")
        for line in normalized_source.split("\n"):
            for part in line.split(","):
                normalized = _decode_chunking_form_text(part)
                if normalized:
                    items.append(normalized)
    deduped: List[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped or None


def _parse_manual_split_points(value: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    raw = str(value or "").strip()
    if not raw:
        return None
    source_items: List[Any]
    if raw.startswith("["):
        try:
            payload = json.loads(raw)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="chunking_manual_split_points 不是合法的 JSON 数组",
            ) from exc
        if not isinstance(payload, list):
            raise HTTPException(
                status_code=400,
                detail="chunking_manual_split_points 必须是数字数组或对象数组",
            )
        source_items = payload
    else:
        source_items = []
        normalized_source = raw.replace("，", ",").replace("；", "\n").replace(";", "\n").replace("\r", "\n")
        for line in normalized_source.split("\n"):
            for part in line.split(","):
                stripped = str(part or "").strip()
                if stripped:
                    source_items.append(stripped)

    parsed: List[Dict[str, Any]] = []
    seen_positions: set[int] = set()
    for index, item in enumerate(source_items, start=1):
        if isinstance(item, dict):
            raw_position = item.get("position")
            point_id = item.get("id") or index
            preview = str(item.get("preview") or "")
        else:
            raw_position = item
            point_id = index
            preview = ""
        try:
            position = int(raw_position)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"chunking_manual_split_points 第 {index} 个位置不是合法整数",
            ) from exc
        if position < 0:
            raise HTTPException(
                status_code=400,
                detail=f"chunking_manual_split_points 第 {index} 个位置必须 >= 0",
            )
        if position in seen_positions:
            raise HTTPException(
                status_code=400,
                detail=f"chunking_manual_split_points 存在重复位置：{position}",
            )
        seen_positions.add(position)
        parsed.append(
            {
                "id": point_id,
                "position": position,
                "preview": preview,
            }
        )
    return sorted(parsed, key=lambda item: int(item["position"]))

@router.post("/batch-upload-complete-pipeline-with-evaluation")
async def batch_upload_complete_pipeline_with_evaluation(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(600),
    qa_per_chunk: Optional[int] = Form(None, description="兼容旧参数：未设置 qa_total_limit 时按 chunk 数估算题量"),
    qa_total_limit: Optional[int] = Form(None, description="主问答总数上限；可按文件或按批次生效"),
    qa_total_limit_scope: str = Form("per_file", description="题数上限范围：per_file / batch"),
    qa_detail_mode: str = Form(
        "auto",
        description="问答粒度: 'auto'=按 generation unit 自动选择, 'point'=单点事实直答, 'summary'=总结/对比/推理",
    ),
    prompt_language: str = Form(
        "auto",
        description="提示词语言: 'auto'=自动识别, 'zh'=中文, 'en'=英文",
    ),
    question_type_mode: str = Form(
        "mixed",
        description="题型模式: 'fixed'=仅使用 question_types 列表首个题型, 'mixed'=在题型集合内混合（默认轮询，可配权重）",
    ),
    question_types: Optional[str] = Form(
        None,
        description="题型列表(逗号分隔): 允许 '简答题','单选题','判断题','计算题'，例如 '简答题,判断题,计算题'",
    ),
    question_type_weights: Optional[str] = Form(
        None,
        description='题型权重（仅 mixed），JSON，如 {"简答题":0.6,"单选题":0.2,"判断题":0.2}',
    ),
    few_shot_examples: Optional[str] = Form(
        None,
        description="few-shot 示例（JSON），仅用于学习问法/语气/长度，不会复述答案或事实",
    ),
    augment_per_qa: int = Form(0, description="每条主问答增广条数，0 表示不增广"),
    include_evaluation: bool = Form(False, description="是否执行问答评估"),
    include_unsupervised_evaluation: bool = Form(
        False, description="是否执行无监督评价（Faithfulness/Answerability/Coverage/F1）"
    ),
    evaluation_method: str = Form(
        "llm",
        description=(
            "评估方式: 'llm'(远程模型) / 'local'(自动指标评估) / "
            "'faithfulness'(忠实度) / 'answerability'(可回答性) / 'unsupervised_f1'(无监督F1)"
        ),
    ),
    unsupervised_batch_size: Optional[int] = Form(
        None,
        description="无监督评估本地模型批量；Qwen3-Embedding-4B 在 11GB 显存建议填 1",
    ),
    faithfulness_nli_model: Optional[str] = Form(
        None,
        description="忠实度 NLI 模型名；为空使用服务默认",
    ),
    answerability_qa_model: Optional[str] = Form(
        None,
        description="可回答性抽取式 QA 模型名；为空使用服务默认",
    ),
    coverage_embedding_model: Optional[str] = Form(
        None,
        description="覆盖度 Embedding 模型名；为空使用服务默认",
    ),
    faithfulness_hypothesis_mode: str = Form(
        "llm",
        description="忠实度评估(QA→陈述句)生成方式: 'llm'=用大模型改写（仅 faithfulness 生效）",
    ),
    faithfulness_hypothesis_max_concurrency: int = Form(
        8,
        description="忠实度评估(QA→陈述句)的大模型改写并发数（仅 faithfulness 生效）",
    ),
    filter_by_threshold: bool = Form(False, description="是否按平均分阈值过滤问答对"),
    score_threshold: float = Form(0.7, description="平均分阈值"),
    enable_vector_storage: bool = Form(True, description="是否自动入库 QA 到向量库"),
    enable_chunk_storage: bool = Form(True, description="是否保存 chunk 溯源索引 doc_tree_chunks"),
    chunk_storage_fail_fast: bool = Form(False, description="溯源索引失败时是否终止任务（默认 False）"),
    chunking_prefix_max_depth: int = Form(4, description="前缀使用的上级标题层数（0~12）"),
    chunking_split_type: Optional[str] = Form(
        None,
        description="切分方式：markdown / text / token / recursive / code / custom；不填则默认 markdown",
    ),
    chunking_text_split_min_length: Optional[int] = Form(
        None,
        description="markdown 模式最小切块长度",
    ),
    chunking_text_split_max_length: Optional[int] = Form(
        None,
        description="markdown 模式最大切块长度",
    ),
    chunking_chunk_overlap: Optional[int] = Form(
        None,
        description="text/token/recursive/code 模式的重叠长度",
    ),
    chunking_separator: Optional[str] = Form(
        None,
        description="text 模式分隔符，可写 \\n 表示换行",
    ),
    chunking_separators: Optional[str] = Form(
        None,
        description="recursive 模式分隔符列表；支持逗号、换行或 JSON 数组",
    ),
    chunking_split_language: Optional[str] = Form(
        None,
        description="code 模式语言，如 js / python / java / markdown",
    ),
    chunking_custom_separator: Optional[str] = Form(
        None,
        description="custom 模式分隔符，可写 \\n 表示换行",
    ),
    chunking_manual_split_points: Optional[str] = Form(
        None,
        description="手工切分点；支持逗号、换行或 JSON 数组，按字符位置切块",
    ),
    chunking_markdown_heading_correction_enabled: bool = Form(
        True,
        description="是否启用 OCR Markdown 标题层级校正（默认 True）",
    ),
    knowledge_classifier: str = Form(
        "doc_level3_rule",
        description="知识分类器：doc_level3_rule=新规则分类器，legacy_model=旧本地模型",
    ),
    use_category_prompt_templates: bool = Form(
        True,
        description="是否按知识分类标签启用专用出题/答案提示词模板；False=统一使用通用模板",
    ),
    ocr_enabled: bool = Form(True, description="是否启用自动 OCR/抽取（默认 True）"),
    ocr_timeout_seconds: Optional[int] = Form(None, description="转发 OCR 的超时时间（秒）"),
    ocr_fail_fast: bool = Form(False, description="True=任一文件 OCR 失败则整体失败"),
    save_mode: str = Form(
        "separate",
        description="'unified' 合并输出, 'separate' 单文件输出",
    ),
    sync_mode: bool = Form(False, description="True=等待任务完成后返回"),
    max_concurrency: Optional[int] = Form(
        None,
        description="最大文件并发处理数（默认 3）",
    ),
    eval_max_concurrency: Optional[int] = Form(
        None,
        description="评估阶段最大并发数（默认 8）",
    ),
    chunk_max_concurrency: Optional[int] = Form(
        None,
        description="同一文件内 chunk 级 LLM 并发数（默认 8）",
    ),
    chunk_max_attempts: Optional[int] = Form(
        None,
        description="每个 chunk 生成最大尝试次数（含首次，默认 2=最多重试 1 次）",
    ),
    retrieval_mode: str = Form(
        "hybrid",
        description="检索排序模式：hybrid=向量+词项+结构加权，semantic=仅向量语义排序",
    ),
    semantic_top_k: Optional[int] = Form(
        None,
        description="每个候选问题选入答案生成上下文的 evidence chunk 数（默认 3）",
    ),
    rerank_top_n: Optional[int] = Form(
        None,
        description="先参与轻量重排的候选 chunk 数（默认 12）",
    ),
    hybrid_weight_dense: Optional[float] = Form(
        None,
        description="hybrid 检索中 dense 向量权重（默认 0.68）",
    ),
    hybrid_weight_lexical: Optional[float] = Form(
        None,
        description="hybrid 检索中词项匹配权重（默认 0.24）",
    ),
    retrieval_structure_weight: Optional[float] = Form(
        None,
        description="同章节/相邻 chunk/title_path 结构加权（默认 0.08）",
    ),
    answer_scope_policy: str = Form(
        "source_primary",
        description="答案证据范围：source_primary / same_section / cross_chunk",
    ),
    llm_max_concurrent_requests: Optional[int] = Form(
        None,
        description="当前任务内同一 LLM/VLM client 同时外发 API 请求数；不填使用 VLM_API_MAX_CONCURRENT_REQUESTS",
    ),
    augment_max_concurrency: Optional[int] = Form(
        None,
        description="问答增广并发数（默认 8）",
    ),
):
    """
    端点 8B：批量执行完整流水线（一步式生成问答+来源事实），可选评估与 Milvus 写入。
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        qa_detail_mode = (qa_detail_mode or "auto").strip().lower()
        if qa_detail_mode not in ("point", "summary", "auto"):
            qa_detail_mode = "auto"
        try:
            qa_per_chunk = max(1, int(qa_per_chunk or 1))
        except Exception:
            qa_per_chunk = 1
        if qa_total_limit is not None:
            qa_total_limit = max(0, int(qa_total_limit))
        qa_total_limit_scope = str(qa_total_limit_scope or "per_file").strip().lower()
        if qa_total_limit_scope not in {"per_file", "batch"}:
            qa_total_limit_scope = "per_file"
        prompt_language = (prompt_language or "auto").strip().lower()
        if prompt_language not in ("auto", "zh", "en"):
            prompt_language = "auto"
        chunking_split_type = str(chunking_split_type or "").strip().lower() or None
        if chunking_split_type == "manual":
            raise HTTPException(
                status_code=400,
                detail="chunking_split_type 不支持 manual；请改用 chunking_manual_split_points",
            )
        if chunking_split_type and chunking_split_type not in SUPPORTED_SPLIT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"chunking_split_type 仅支持 {', '.join(SUPPORTED_SPLIT_TYPES)}",
            )
        if chunking_text_split_min_length is not None:
            chunking_text_split_min_length = max(1, int(chunking_text_split_min_length))
        if chunking_text_split_max_length is not None:
            chunking_text_split_max_length = max(1, int(chunking_text_split_max_length))
        if (
            chunking_text_split_min_length is not None
            and chunking_text_split_max_length is not None
            and chunking_text_split_max_length < chunking_text_split_min_length
        ):
            raise HTTPException(
                status_code=400,
                detail="chunking_text_split_max_length 不能小于 chunking_text_split_min_length",
            )
        if chunking_chunk_overlap is not None:
            chunking_chunk_overlap = max(0, int(chunking_chunk_overlap))
        chunking_separator = _decode_chunking_form_text(chunking_separator)
        parsed_chunking_separators = _parse_chunking_separators(chunking_separators)
        chunking_split_language = str(chunking_split_language or "").strip() or None
        chunking_custom_separator = _decode_chunking_form_text(chunking_custom_separator)
        parsed_manual_split_points = _parse_manual_split_points(chunking_manual_split_points)
        knowledge_classifier = normalize_knowledge_classifier(knowledge_classifier)
        evaluation_method = (evaluation_method or "llm").strip().lower()
        if evaluation_method == "unsupervised":
            evaluation_method = "unsupervised_f1"
        if evaluation_method not in ("llm", "local", "faithfulness", "answerability", "unsupervised_f1"):
            raise HTTPException(
                status_code=400,
                detail="evaluation_method 仅支持 llm / local / faithfulness / answerability / unsupervised_f1",
            )
        faithfulness_hypothesis_mode = (faithfulness_hypothesis_mode or "llm").strip().lower()
        if faithfulness_hypothesis_mode != "llm":
            faithfulness_hypothesis_mode = "llm"
        faithfulness_hypothesis_max_concurrency = max(
            1, int(faithfulness_hypothesis_max_concurrency or 1)
        )
        try:
            resolved_nli_model = validate_evaluation_model_name(
                faithfulness_nli_model,
                kind="faithfulness_nli",
            )
            resolved_qa_model = validate_evaluation_model_name(
                answerability_qa_model,
                kind="answerability_qa",
            )
            resolved_coverage_model = validate_evaluation_model_name(
                coverage_embedding_model,
                kind="coverage_embedding",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if unsupervised_batch_size is not None:
            unsupervised_batch_size = max(1, min(512, int(unsupervised_batch_size)))
        parsed_few_shot_examples = parse_few_shot_examples(few_shot_examples)
        if include_evaluation and evaluation_method in ("faithfulness", "answerability", "unsupervised_f1"):
            include_unsupervised_evaluation = True
        if not include_evaluation:
            filter_by_threshold = False

        batch_task_id = f"batch_complete_task_{int(time.time())}"
        admit_info = admit_gpu_job(batch_task_id, job_type="pipeline")
        if not bool(admit_info.get("accepted")):
            raise HTTPException(status_code=429, detail="GPU 任务排队已满，请稍后重试")
        try:
            file_contents, ocr_summary = await resolve_uploaded_files_with_auto_ocr(
                files,
                ocr_enabled=ocr_enabled,
                ocr_timeout_seconds=ocr_timeout_seconds,
                ocr_fail_fast=ocr_fail_fast,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OCR failed: {str(exc)}")
        successful_sources = [f for f in file_contents if f["status"] == "success"]
        if not successful_sources:
            raise HTTPException(status_code=400, detail="All uploaded files failed to read")

        llm_config = {
            "api_key": CONFIG["api_key"],
            "base_url": CONFIG["base_url"],
            "model": CONFIG["model"],
            "api_type": CONFIG.get("api_type") or "openai",
            "model_version": CONFIG.get("model_version") or "",
            "max_retries": CONFIG["max_retries"],
        }

        concurrency_limit = resolve_batch_concurrency(max_concurrency)
        eval_concurrency = eval_max_concurrency or 8
        chunk_concurrency = chunk_max_concurrency or 8
        chunk_attempts = max(1, int(chunk_max_attempts or 2))
        retrieval_mode = str(retrieval_mode or "hybrid").strip().lower()
        if retrieval_mode not in {"semantic", "hybrid"}:
            retrieval_mode = "hybrid"
        semantic_top_k = max(0, int(semantic_top_k or 3))
        rerank_top_n = max(1, int(rerank_top_n or 12))
        hybrid_weight_dense = max(0.0, min(1.0, float(hybrid_weight_dense if hybrid_weight_dense is not None else 0.68)))
        hybrid_weight_lexical = max(0.0, min(1.0, float(hybrid_weight_lexical if hybrid_weight_lexical is not None else 0.24)))
        retrieval_structure_weight = max(0.0, min(0.5, float(retrieval_structure_weight if retrieval_structure_weight is not None else 0.08)))
        answer_scope_policy = str(answer_scope_policy or "source_primary").strip().lower()
        if answer_scope_policy not in {"source_primary", "same_section", "cross_chunk"}:
            answer_scope_policy = "source_primary"
        llm_request_concurrency = (
            max(1, int(llm_max_concurrent_requests))
            if llm_max_concurrent_requests is not None
            else None
        )
        augment_concurrency = augment_max_concurrency or 8

        now = now_server_local_iso()
        status_data = {
            "status": "queued",
            "batch_mode": True,
            "task_id": batch_task_id,
            "save_mode": save_mode,
            "total_files": len(files),
            "completed_files": 0,
            "failed_files": 0,
            "include_evaluation": include_evaluation,
            "include_unsupervised_evaluation": include_unsupervised_evaluation,
            "evaluation_method": evaluation_method,
            "unsupervised_batch_size": unsupervised_batch_size,
            "faithfulness_nli_model": resolved_nli_model,
            "answerability_qa_model": resolved_qa_model,
            "coverage_embedding_model": resolved_coverage_model,
            "faithfulness_hypothesis_mode": faithfulness_hypothesis_mode,
            "faithfulness_hypothesis_max_concurrency": faithfulness_hypothesis_max_concurrency,
            "filter_by_threshold": filter_by_threshold,
            "score_threshold": score_threshold if filter_by_threshold else None,
            "vector_storage_enabled": enable_vector_storage,
            "chunk_storage_enabled": enable_chunk_storage,
            "chunk_storage_fail_fast": chunk_storage_fail_fast,
            "knowledge_classifier": knowledge_classifier,
            "use_category_prompt_templates": bool(use_category_prompt_templates),
            "chunking_prefix_max_depth": chunking_prefix_max_depth,
            "chunking_config": {
                "split_type": chunking_split_type or "markdown",
                "prefix_max_depth": chunking_prefix_max_depth,
                "markdown_heading_correction_enabled": bool(chunking_markdown_heading_correction_enabled),
                "text_split_min_length": chunking_text_split_min_length,
                "text_split_max_length": chunking_text_split_max_length,
                "chunk_overlap": chunking_chunk_overlap,
                "separator": chunking_separator,
                "separators": parsed_chunking_separators,
                "split_language": chunking_split_language,
                "custom_separator": chunking_custom_separator,
                "manual_split_points_count": len(parsed_manual_split_points or []),
                "manual_split_points_preview": [
                    int(item.get("position") or 0)
                    for item in (parsed_manual_split_points or [])[:10]
                ],
            },
            "augment_per_qa": augment_per_qa,
            "input_filenames": [f.filename for f in files],
            "ocr_enabled": ocr_enabled,
            "ocr_timeout_seconds": ocr_timeout_seconds,
            "ocr_fail_fast": ocr_fail_fast,
            "ocr_summary": ocr_summary,
            "concurrency": concurrency_limit,
            "chunk_concurrency": chunk_concurrency,
            "chunk_max_attempts": chunk_attempts,
            "retrieval_config": {
                "retrieval_mode": retrieval_mode,
                "semantic_top_k": semantic_top_k,
                "rerank_top_n": rerank_top_n,
                "hybrid_weight_dense": hybrid_weight_dense,
                "hybrid_weight_lexical": hybrid_weight_lexical,
                "retrieval_structure_weight": retrieval_structure_weight,
                "answer_scope_policy": answer_scope_policy,
            },
            "llm_max_concurrent_requests": llm_request_concurrency,
            "augment_concurrency": augment_concurrency,
            "evaluation_concurrency": eval_concurrency,
            "file_progress": {},
            "message": "Task queued, waiting for workers",
            "history_source": "artifacts",
            "milvus_task_id": None,
            "artifacts_deleted": False,
            "artifacts_expire_at": None,
            "created_at": now,
            "updated_at": now,
            "qa_per_chunk": qa_per_chunk,
            "qa_total_limit": qa_total_limit,
            "qa_total_limit_scope": qa_total_limit_scope,
            "qa_detail_mode": qa_detail_mode,
            "prompt_language": prompt_language,
            "question_type_mode": question_type_mode,
            "question_types": question_types,
            "question_type_weights": question_type_weights,
            "few_shot_examples": parsed_few_shot_examples,
        }
        upsert_pipeline_task_status(batch_task_id, status_data)

        job_context = {
            "task_id": batch_task_id,
            "file_contents": file_contents,
            "chunk_size": chunk_size,
            "qa_per_chunk": qa_per_chunk,
            "qa_total_limit": qa_total_limit,
            "qa_total_limit_scope": qa_total_limit_scope,
            "qa_detail_mode": qa_detail_mode,
            "prompt_language": prompt_language,
            "include_evaluation": include_evaluation,
            "include_unsupervised_evaluation": include_unsupervised_evaluation,
            "evaluation_method": evaluation_method,
            "unsupervised_batch_size": unsupervised_batch_size,
            "faithfulness_nli_model": resolved_nli_model,
            "answerability_qa_model": resolved_qa_model,
            "coverage_embedding_model": resolved_coverage_model,
            "faithfulness_hypothesis_mode": faithfulness_hypothesis_mode,
            "faithfulness_hypothesis_max_concurrency": faithfulness_hypothesis_max_concurrency,
            "filter_by_threshold": filter_by_threshold,
            "score_threshold": score_threshold,
            "save_mode": save_mode,
            "enable_vector_storage": enable_vector_storage,
            "enable_chunk_storage": enable_chunk_storage,
            "chunk_storage_fail_fast": chunk_storage_fail_fast,
            "knowledge_classifier": knowledge_classifier,
            "use_category_prompt_templates": bool(use_category_prompt_templates),
            "chunking_prefix_max_depth": chunking_prefix_max_depth,
            "chunking_split_type": chunking_split_type,
            "chunking_markdown_heading_correction_enabled": bool(chunking_markdown_heading_correction_enabled),
            "chunking_text_split_min_length": chunking_text_split_min_length,
            "chunking_text_split_max_length": chunking_text_split_max_length,
            "chunking_chunk_overlap": chunking_chunk_overlap,
            "chunking_separator": chunking_separator,
            "chunking_separators": parsed_chunking_separators,
            "chunking_split_language": chunking_split_language,
            "chunking_custom_separator": chunking_custom_separator,
            "chunking_manual_split_points": parsed_manual_split_points,
            "status_data": status_data,
            "criteria_list": LLM_EVALUATION_METRICS,
            "llm_config": llm_config,
            "llm_max_concurrent_requests": llm_request_concurrency,
            "max_concurrency": concurrency_limit,
            "chunk_max_concurrency": chunk_concurrency,
            "chunk_max_attempts": chunk_attempts,
            "retrieval_mode": retrieval_mode,
            "semantic_top_k": semantic_top_k,
            "rerank_top_n": rerank_top_n,
            "hybrid_weight_dense": hybrid_weight_dense,
            "hybrid_weight_lexical": hybrid_weight_lexical,
            "retrieval_structure_weight": retrieval_structure_weight,
            "answer_scope_policy": answer_scope_policy,
            "augment_max_concurrency": augment_concurrency,
            "eval_max_concurrency": eval_concurrency,
            "question_type_mode": question_type_mode,
            "question_types": question_types,
            "question_type_weights": question_type_weights,
            "few_shot_examples": parsed_few_shot_examples,
            "augment_per_qa": augment_per_qa,
        }

        if sync_mode:
            try:
                await run_batch_complete_pipeline_async(job_context)
                return get_pipeline_task_status(batch_task_id) or {
                    "task_id": batch_task_id,
                    "status": "failed",
                    "message": "pipeline status missing after sync run",
                }
            finally:
                release_gpu_job(batch_task_id)

        task = asyncio.create_task(run_batch_complete_pipeline_async(job_context))
        ACTIVE_BATCH_JOBS[batch_task_id] = task

        def _cleanup(_task: asyncio.Task) -> None:
            ACTIVE_BATCH_JOBS.pop(batch_task_id, None)
            release_gpu_job(batch_task_id)

        task.add_done_callback(_cleanup)
        logger.info("[batch %s] job scheduled (%d files)", batch_task_id, len(file_contents))
        return {
            "status": "processing",
            "batch_mode": True,
            "task_id": batch_task_id,
            "message": "Batch job scheduled, use task-status to poll progress",
            "status_store": get_pipeline_store_path(),
            "total_files": len(files),
            "save_mode": save_mode,
            "include_evaluation": include_evaluation,
            "include_unsupervised_evaluation": include_unsupervised_evaluation,
            "evaluation_method": evaluation_method,
            "filter_by_threshold": filter_by_threshold,
            "knowledge_classifier": knowledge_classifier,
            "concurrency": concurrency_limit,
            "chunking_config": status_data["chunking_config"],
            "llm_max_concurrent_requests": llm_request_concurrency,
        }
    except HTTPException:
        try:
            release_gpu_job(batch_task_id)  # type: ignore[name-defined]
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            release_gpu_job(batch_task_id)  # type: ignore[name-defined]
        except Exception:
            pass
        logger.exception("Batch pipeline failed")
        raise HTTPException(status_code=500, detail=f"Batch pipeline failed: {str(exc)}")



__all__ = ["router", "run_batch_complete_pipeline_async"]
