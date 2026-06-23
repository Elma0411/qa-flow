# 文件作用：提供本地结果文件与向量库问答搜索接口。
# 关联说明：连接 storage 本地结果和 milvus 搜索能力，补充管理端查询入口。

import glob
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, HTTPException, Query

from app.core.config import CONFIG
from app.core.logger import logger
from app.services import milvus as milvus_service

router = APIRouter()


def _resolve_milvus_source_field() -> str:
    """
    Prefer unified `source`. If the connected collection still uses legacy `source_id`,
    fall back to it.
    """
    try:
        if not milvus_service.milvus_client:
            return "source"
        names = {f.name for f in milvus_service.milvus_client.schema.fields}
        if "source" in names:
            return "source"
        if "source_id" in names:
            return "source_id"
    except Exception:
        return "source"
    return "source"


@router.post("/query-qa", deprecated=True)
async def query_qa(
    query_text: str = Form(..., description="查询文本"),
    task_id: Optional[str] = Form(None, description="任务ID过滤"),
    top_k: int = Form(10, description="返回结果数量"),
    only_filtered: bool = Form(True, description="是否只返回过滤后的高质量问答对"),
    min_avg_score: float = Form(0.0, description="最低平均分"),
    include_raw_responses: bool = Form(False, description="是否包含原始LLM响应"),
) -> Dict[str, Any]:
    """
    基于本地 consolidated JSON 的简单文本相似度查询（演示用）。

    实际线上检索推荐使用基于 Milvus 的 /task-qa 和 /file-qa。
    """
    try:
        outputs_dir = CONFIG["outputs_dir"]
        pattern = os.path.join(outputs_dir, "*_consolidated_*.json")
        consolidated_files = glob.glob(pattern)
        if not consolidated_files:
            return {"message": "未找到任何合并的JSON文件", "results": [], "total": 0}
        all_items: List[Dict[str, Any]] = []
        for file_path in consolidated_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if task_id and data.get("task", {}).get("task_id") != task_id:
                    continue
                items = data.get("items", [])
                for item in items:
                    if only_filtered and not item.get("filtered", False):
                        continue
                    avg_score = item.get("average_score")
                    if avg_score is not None and avg_score < min_avg_score:
                        continue
                    item_text = item.get("text_for_embedding", "") or (
                        (item.get("question") or "") + " " + (item.get("answer") or "")
                    )
                    query_lower = query_text.lower()
                    text_lower = item_text.lower()
                    if query_lower in text_lower:
                        overlap = [
                            w for w in query_lower.split() if w and w in text_lower
                        ]
                        similarity = len(overlap) / max(len(query_lower.split()), 1)
                    else:
                        similarity = 0.0
                    result_item = dict(item)
                    result_item["similarity"] = similarity
                    result_item["source_file"] = os.path.basename(file_path)
                    if not include_raw_responses and "evaluation" in result_item:
                        eval_data = result_item["evaluation"]
                        if isinstance(eval_data, dict) and "llm" in eval_data:
                            llm_eval = eval_data["llm"]
                            if isinstance(llm_eval, dict):
                                for metric_data in llm_eval.values():
                                    if isinstance(metric_data, dict):
                                        metric_data.pop("_raw", None)
                                        metric_data.pop("all_responses", None)
                    all_items.append(result_item)
            except Exception as exc:
                logger.error("加载文件 %s 失败: %s", file_path, exc)
                continue
        all_items.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        results = all_items[:top_k]
        return {
            "message": "查询完成",
            "query": query_text,
            "filters": {
                "task_id": task_id,
                "only_filtered": only_filtered,
                "min_avg_score": min_avg_score,
            },
            "results": results,
            "total": len(all_items),
            "returned": len(results),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(exc)}")


@router.get("/task-qa/{task_id}", deprecated=True)
async def get_task_qa(
    task_id: str,
    only_filtered: bool = Query(True, description="是否只返回过滤后的问答对"),
    min_avg_score: float = Query(0.0, description="最低平均分"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量"),
    include_raw_responses: bool = Query(False, description="是否包含原始LLM响应"),
) -> Dict[str, Any]:
    """
    按任务ID从 Milvus 中分页获取问答对列表。
    """
    try:
        if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
            raise HTTPException(
                status_code=503,
                detail="Milvus服务不可用，请确认向量数据库已启动并连接",
            )
        filter_expressions: List[str] = [f'task_id == "{task_id}"']
        if only_filtered:
            filter_expressions.append("filtered == true")
        if min_avg_score > 0.0:
            filter_expressions.append(f"average_score >= {min_avg_score}")
        filter_expr = " and ".join(filter_expressions)
        milvus_service.milvus_client.load()
        source_field = _resolve_milvus_source_field()
        start_idx = (page - 1) * page_size
        rows = milvus_service.milvus_client.query(
            expr=filter_expr,
            output_fields=[
                "id",
                "task_id",
                "original_filename",
                source_field,
                "source_fact_text",
                "question",
                "answer",
                "question_type",
                "question_type_reason",
                "answer_explanation",
                "knowledge_category",
                "knowledge_category_reason",
                "knowledge_category_confidence",
                "difficulty_level",
                "difficulty_score",
                "llm_model",
                "embed_model",
                "embed_dim",
                "filtered",
                "average_score",
                "evaluation_method",
                "llm_scores",
                "llm_reasons",
                "local_scores",
                "unsupervised_method",
                "unsupervised_scores",
                "unsupervised_meta",
                "created_at",
                "filter_basis",
                "is_primary",
                "is_augmented",
                "variant_of",
            ],
            offset=start_idx,
            limit=page_size,
        )
        try:
            count_rows = milvus_service.milvus_client.query(
                expr=filter_expr, output_fields=["id"], limit=16384
            )
            total_items = len(count_rows)
        except Exception as exc:
            logger.warning("统计总数失败，将使用估算值: %s", exc)
            total_items = start_idx + len(rows)
        items: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}
        for r in rows:
            try:
                llm_scores = (
                    json.loads(r.get("llm_scores", "{}"))
                    if r.get("llm_scores")
                    else {}
                )
                llm_reasons = (
                    json.loads(r.get("llm_reasons", "{}"))
                    if r.get("llm_reasons")
                    else {}
                )
                local_scores = (
                    json.loads(r.get("local_scores", "{}"))
                    if r.get("local_scores")
                    else {}
                )
                unsup_method = str(r.get("unsupervised_method") or "").strip()
                unsup_scores = (
                    json.loads(r.get("unsupervised_scores", "{}"))
                    if r.get("unsupervised_scores")
                    else {}
                )
                unsup_meta = (
                    json.loads(r.get("unsupervised_meta", "{}"))
                    if r.get("unsupervised_meta")
                    else {}
                )
            except json.JSONDecodeError:
                llm_scores = {}
                llm_reasons = {}
                local_scores = {}
                unsup_method = ""
                unsup_scores = {}
                unsup_meta = {}
            evaluation: Dict[str, Any] = {}
            if llm_scores or llm_reasons:
                evaluation["llm"] = {"scores": llm_scores, "reasons": llm_reasons}
            if local_scores:
                evaluation["local"] = {"scores": local_scores}
            if evaluation.get("llm"):
                for metric_data in evaluation["llm"].values():
                    if isinstance(metric_data, dict):
                        if not include_raw_responses:
                            metric_data.pop("_raw", None)
                            metric_data.pop("all_responses", None)
                        else:
                            if "_raw" in metric_data and len(str(metric_data["_raw"])) > 1000:
                                raw_content = str(metric_data["_raw"])
                                metric_data["_raw"] = raw_content[:500] + "...[truncated]"
                                metric_data["_raw_length"] = len(raw_content)
            item = {
                "id": r.get("id"),
                "task_id": r.get("task_id"),
                "original_filename": r.get("original_filename"),
                "group_id": r.get(source_field),
                "source": r.get(source_field),
                "source_fact_text": r.get("source_fact_text"),
                "context": r.get("source_fact_text"),
                "question": r.get("question"),
                "answer": r.get("answer"),
                "question_type": r.get("question_type"),
                "question_type_reason": r.get("question_type_reason"),
                "answer_explanation": r.get("answer_explanation"),
                "knowledge_category": r.get("knowledge_category"),
                "knowledge_category_reason": r.get("knowledge_category_reason"),
                "knowledge_category_confidence": r.get("knowledge_category_confidence"),
                "difficulty_level": r.get("difficulty_level"),
                "difficulty_score": r.get("difficulty_score"),
                "filtered": r.get("filtered"),
                "average_score": r.get("average_score"),
                "evaluation_method": r.get("evaluation_method"),
                "evaluation": evaluation,
                "unsupervised_evaluation": {
                    "method": unsup_method,
                    "scores": unsup_scores,
                    "meta": unsup_meta,
                }
                if (unsup_method or unsup_scores or unsup_meta)
                else None,
                "llm_model": r.get("llm_model"),
                "embed_model": r.get("embed_model"),
                "embed_dim": r.get("embed_dim"),
                "created_at": r.get("created_at"),
                "filter_basis": r.get("filter_basis"),
                "is_primary": r.get("is_primary"),
                "is_augmented": r.get("is_augmented"),
                "variant_of": r.get("variant_of"),
            }
            items.append(item)
            kc = r.get("knowledge_category", "未分类")
            category_counts[kc] = category_counts.get(kc, 0) + 1
        task_info = {
            "task_id": task_id,
            "original_filename": items[0].get("original_filename") if items else None,
            "created_at": items[0].get("created_at") if items else None,
        }
        model_info = {
            "embed_model": items[0].get("embed_model")
            if items
            else CONFIG["milvus"]["embedding_model"],
            "embed_dim": items[0].get("embed_dim")
            if items
            else CONFIG["milvus"]["vector_dim"],
            "distance": CONFIG["milvus"].get("metric_type", "IP"),
        }
        return {
            "task_id": task_id,
            "task_info": task_info,
            "model_info": model_info,
            "counts": {
                "total_items": total_items,
                "current_page_items": len(items),
            },
            "category_distribution": category_counts,
            "filter_info": {
                "only_filtered": only_filtered,
                "min_avg_score": min_avg_score,
            },
            "filter_basis": items[0].get("filter_basis") if items else None,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": (total_items + page_size - 1) // page_size
                if total_items > 0
                else 0,
            },
            "items": items,
            "data_source": "milvus_database",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取任务数据失败: {str(exc)}")


@router.get("/file-qa", deprecated=True)
async def get_file_qa(
    original_filename: Optional[str] = Query(
        None, description="单个原始文件名，例如 1.1.txt"
    ),
    original_filenames: Optional[List[str]] = Query(
        None,
        description=(
            "批量文件名，支持重复传参："
            "original_filenames=a&original_filenames=b"
        ),
    ),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量"),
    include_details: bool = Query(
        False, description="是否包含评分理由等完整信息"
    ),
    task_id: Optional[str] = Query(
        None, description="可选：进一步按任务ID过滤"
    ),
) -> Dict[str, Any]:
    """
    按原始文件名从 Milvus 中分页获取问答对列表。
    """
    try:
        if not milvus_service.MILVUS_AVAILABLE or not milvus_service.milvus_client:
            raise HTTPException(
                status_code=503,
                detail="Milvus服务不可用，请确认向量数据库已启动并连接",
            )
        names: List[str] = []
        if original_filename:
            names.append(str(original_filename))
        if original_filenames:
            for n in original_filenames:
                if n:
                    names.append(str(n))
        names = list(dict.fromkeys(names))
        if not names:
            raise HTTPException(
                status_code=400,
                detail="请提供 original_filename 或 original_filenames 参数",
            )
        filter_expressions: List[str] = []
        if len(names) == 1:
            safe_name = names[0].replace('"', '\\"')
            filter_expressions.append(f'original_filename == "{safe_name}"')
        else:
            escaped_names = [n.replace('"', '\\"') for n in names]
            quoted = ",".join(f'"{safe}"' for safe in escaped_names)
            filter_expressions.append(f"original_filename in [{quoted}]")
        if task_id:
            safe_task = str(task_id).replace('"', '\\"')
            filter_expressions.append(f'task_id == "{safe_task}"')
        filter_expr = " and ".join(filter_expressions)
        source_field = _resolve_milvus_source_field()
        output_fields = [
            "id",
            "task_id",
            "original_filename",
            source_field,
            "source_fact_text",
            "question",
            "answer",
            "question_type",
            "question_type_reason",
            "answer_explanation",
            "knowledge_category",
            "knowledge_category_reason",
            "knowledge_category_confidence",
            "difficulty_level",
            "difficulty_score",
            "llm_model",
            "embed_model",
            "embed_dim",
            "filtered",
            "average_score",
            "evaluation_method",
            "llm_scores",
            "llm_reasons",
            "local_scores",
            "unsupervised_method",
            "unsupervised_scores",
            "unsupervised_meta",
            "created_at",
            "filter_basis",
            "is_primary",
            "is_augmented",
            "variant_of",
        ]
        milvus_service.milvus_client.load()
        start_idx = (page - 1) * page_size
        rows = milvus_service.milvus_client.query(
            expr=filter_expr,
            output_fields=output_fields,
            offset=start_idx,
            limit=page_size,
        )
        try:
            count_rows = milvus_service.milvus_client.query(
                expr=filter_expr, output_fields=["id"], limit=16384
            )
            total_items = len(count_rows)
        except Exception as exc:
            logger.warning("统计总数失败，将使用估算值: %s", exc)
            total_items = start_idx + len(rows)
        items: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}
        for r in rows:
            if include_details:
                try:
                    llm_scores = (
                        json.loads(r.get("llm_scores", "{}"))
                        if r.get("llm_scores")
                        else {}
                    )
                    llm_reasons = (
                        json.loads(r.get("llm_reasons", "{}"))
                        if r.get("llm_reasons")
                        else {}
                    )
                    local_scores = (
                        json.loads(r.get("local_scores", "{}"))
                        if r.get("local_scores")
                        else {}
                    )
                    unsup_method = str(r.get("unsupervised_method") or "").strip()
                    unsup_scores = (
                        json.loads(r.get("unsupervised_scores", "{}"))
                        if r.get("unsupervised_scores")
                        else {}
                    )
                    unsup_meta = (
                        json.loads(r.get("unsupervised_meta", "{}"))
                        if r.get("unsupervised_meta")
                        else {}
                    )
                except json.JSONDecodeError:
                    llm_scores = {}
                    llm_reasons = {}
                    local_scores = {}
                    unsup_method = ""
                    unsup_scores = {}
                    unsup_meta = {}
                evaluation: Dict[str, Any] = {}
                if llm_scores or llm_reasons:
                    evaluation["llm"] = {
                        "scores": llm_scores,
                        "reasons": llm_reasons,
                    }
                if local_scores:
                    evaluation["local"] = {"scores": local_scores}
                item = {
                    "id": r.get("id"),
                    "task_id": r.get("task_id"),
                    "original_filename": r.get("original_filename"),
                    "group_id": r.get(source_field),
                    "source": r.get(source_field),
                    "source_fact_text": r.get("source_fact_text"),
                    "context": r.get("source_fact_text"),
                    "question_type": r.get("question_type"),
                    "question_type_reason": r.get("question_type_reason"),
                    "answer_explanation": r.get("answer_explanation"),
                    "knowledge_category": r.get("knowledge_category"),
                    "knowledge_category_reason": r.get("knowledge_category_reason"),
                    "knowledge_category_confidence": r.get(
                        "knowledge_category_confidence"
                    ),
                    "difficulty_level": r.get("difficulty_level"),
                    "difficulty_score": r.get("difficulty_score"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "filtered": r.get("filtered"),
                    "average_score": r.get("average_score"),
                    "evaluation": evaluation,
                    "unsupervised_evaluation": {
                        "method": unsup_method,
                        "scores": unsup_scores,
                        "meta": unsup_meta,
                    }
                    if (unsup_method or unsup_scores or unsup_meta)
                    else None,
                    "llm_model": r.get("llm_model"),
                    "embed_model": r.get("embed_model"),
                    "embed_dim": r.get("embed_dim"),
                    "created_at": r.get("created_at"),
                    "filter_basis": r.get("filter_basis"),
                    "is_primary": r.get("is_primary"),
                    "is_augmented": r.get("is_augmented"),
                    "variant_of": r.get("variant_of"),
                }
            else:
                item = {
                    "id": r.get("id"),
                    "task_id": r.get("task_id"),
                    "original_filename": r.get("original_filename"),
                    "group_id": r.get(source_field),
                    "knowledge_category": r.get("knowledge_category"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "source": r.get(source_field),
                    "context": r.get("source_fact_text"),
                    "average_score": r.get("average_score"),
                    "created_at": r.get("created_at"),
                    "is_primary": r.get("is_primary"),
                    "is_augmented": r.get("is_augmented"),
                    "variant_of": r.get("variant_of"),
                }
            items.append(item)
            kc = r.get("knowledge_category", "未分类")
            category_counts[kc] = category_counts.get(kc, 0) + 1
        return {
            "files": names,
            "filters": {"task_id": task_id},
            "counts": {
                "total_items": total_items if total_items > 0 else 0,
                "current_page_items": len(items),
            },
            "category_distribution": category_counts,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_items if total_items > 0 else 0,
                "total_pages": (total_items + page_size - 1) // page_size
                if total_items > 0
                else 0,
            },
            "items": items,
            "data_source": "milvus_database",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"按文件名获取数据失败: {str(exc)}")
