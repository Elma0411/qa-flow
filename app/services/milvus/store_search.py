# 文件作用：封装问答向量写入、查询和结果排序逻辑。
# 关联说明：依赖 collection、embedding、meta 和 runtime，完成问答写入与搜索。

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from app.core.config import CONFIG
from app.core.logger import logger
from .runtime import MILVUS_RUNTIME as _rt
from .collection import (
    _resolve_category_field_names,
    _resolve_source_field_name,
)
from .embedding import generate_embeddings, load_embedding_model
from .meta import (
    _json_dumps_minified,
    _serialize_unsupervised_meta_for_milvus,
    _truncate_utf8_bytes,
    _utf8_size,
)
from app.services.debug import upsert_qa_debug_items


def store_qa_pairs_to_milvus(
    consolidated_json_path: str, enable_vector_storage: bool = True
) -> Dict[str, Any]:
    if not enable_vector_storage or not _rt.MILVUS_AVAILABLE or not _rt.milvus_client:
        return {"success": False, "message": "Milvus未启用或未连接"}
    try:
        with open(consolidated_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        if not items:
            return {"success": True, "message": "没有数据需要存储", "stored_count": 0}

        batch_data: List[Dict[str, Any]] = []
        text_for_embeddings: List[str] = []
        allowed_fields: Optional[set] = None
        filtered_missing_fields: set = set()
        try:
            allowed_fields = {f.name for f in _rt.milvus_client.schema.fields} if _rt.milvus_client else None
        except Exception:
            allowed_fields = None
        source_field = _resolve_source_field_name(allowed_fields)
        category_field, category_reason_field, category_conf_field = _resolve_category_field_names(allowed_fields)

        model_info = data.get("model", {}) if isinstance(data, dict) else {}
        embed_model_name = model_info.get("embed_model") or CONFIG["milvus"]["embedding_model"]
        try:
            embed_dim = int(model_info.get("embed_dim", CONFIG["milvus"]["vector_dim"]))
        except Exception:
            embed_dim = CONFIG["milvus"]["vector_dim"]

        # 统一截断，防止超过 Milvus schema 的 max_length。
        # 这里按 UTF-8 字节长度控制，避免中文场景下“字符数看起来没超、实际字节数已超”的问题。
        max_len_map = {
            "id": 128,
            "task_id": 128,
            "original_filename": 512,
            "source": 512,
            "source_id": 512,
            "source_fact_text": 4096,
            "question": 4096,
            "answer": 8192,
            "question_type": 64,
            "question_type_reason": 1024,
            "answer_explanation": 8192,
            "knowledge_category": 256,
            "knowledge_category_reason": 1024,
            "theme": 256,
            "theme_reason": 1024,
            "llm_model": 256,
            "embed_model": 256,
            "evaluation_method": 64,
            "llm_scores": 2048,
            "llm_reasons": 8192,
            "local_scores": 2048,
            "unsupervised_method": 64,
            "unsupervised_scores": 2048,
            "unsupervised_meta": 8192,
            "variant_of": 256,
            "filter_basis": 64,
        }
        json_string_fields = {
            "llm_scores",
            "llm_reasons",
            "local_scores",
            "unsupervised_scores",
            "unsupervised_meta",
        }

        def _prepare_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:

            # Backward compatibility: if the connected collection requires legacy theme fields,
            # populate them from the unified knowledge_category fields (or vice versa).
            if allowed_fields is not None:
                if "theme" in allowed_fields and "theme" not in row_dict:
                    row_dict["theme"] = row_dict.get("knowledge_category") or ""
                if "theme_reason" in allowed_fields and "theme_reason" not in row_dict:
                    row_dict["theme_reason"] = row_dict.get("knowledge_category_reason") or ""
                if "theme_confidence" in allowed_fields and "theme_confidence" not in row_dict:
                    row_dict["theme_confidence"] = row_dict.get("knowledge_category_confidence") or 0.0
                if "knowledge_category" in allowed_fields and "knowledge_category" not in row_dict:
                    row_dict["knowledge_category"] = row_dict.get("theme") or ""
                if "knowledge_category_reason" in allowed_fields and "knowledge_category_reason" not in row_dict:
                    row_dict["knowledge_category_reason"] = row_dict.get("theme_reason") or ""
                if "knowledge_category_confidence" in allowed_fields and "knowledge_category_confidence" not in row_dict:
                    row_dict["knowledge_category_confidence"] = row_dict.get("theme_confidence") or 0.0

            for key, max_len in max_len_map.items():
                value = row_dict.get(key)
                if not isinstance(value, str):
                    continue
                if key in json_string_fields:
                    continue
                if _utf8_size(value) > max_len:
                    row_dict[key] = _truncate_utf8_bytes(value, max_len)

            if allowed_fields is not None:
                extra_keys = set(row_dict.keys()) - allowed_fields
                if extra_keys:
                    filtered_missing_fields.update(extra_keys)
                row_dict = {k: v for k, v in row_dict.items() if k in allowed_fields}
            return row_dict

        for item in items:
            if not isinstance(item, dict):
                # 非法记录直接跳过
                print("? 跳过一条非字典 QA 记录:", repr(item))
                continue

            try:
                # 文本用于生成向量
                text_for_embedding = item.get("text_for_embedding", "") or ""
                if not text_for_embedding:
                    question = item.get("question", "") or ""
                    answer = item.get("answer", "") or ""
                    text_for_embedding = f"{question} [SEP] {answer}"

                # 归一化 evaluation 结构，避免 None/异常结构
                evaluation = item.get("evaluation") or {}
                if not isinstance(evaluation, dict):
                    evaluation = {}
                llm_eval = evaluation.get("llm") or {}
                if not isinstance(llm_eval, dict):
                    llm_eval = {}
                local_eval = evaluation.get("local") or {}
                if not isinstance(local_eval, dict):
                    local_eval = {}

                # 知识类别与置信度
                kc_value = item.get("knowledge_category") or item.get("theme") or ""
                kc_reason = item.get("knowledge_category_reason") or item.get("theme_reason") or ""
                kc_conf_val = item.get("knowledge_category_confidence") or item.get("theme_confidence")
                try:
                    kc_conf = float(kc_conf_val) if kc_conf_val is not None else 0.0
                except Exception:
                    kc_conf = 0.0

                # 难度分数：
                # 当前 Milvus schema 中 `difficulty_score` 是非 nullable FLOAT，
                # 因此不能传入 None。对缺失/非法值统一写入 -1.0，明确表示
                # “当前记录没有难度分数元数据”，避免和真实的 0.0（最简单）
                # 混淆。
                diff_score = -1.0
                diff_score_val = item.get("difficulty_score")
                if isinstance(diff_score_val, (int, float, str)):
                    try:
                        diff_score = float(diff_score_val)
                    except Exception:
                        diff_score = -1.0

                # 平均分：统一转为 float，无法解析时退回 0.0
                avg_raw = item.get("average_score", 0.0)
                try:
                    avg_score = float(avg_raw) if avg_raw is not None else 0.0
                except Exception:
                    avg_score = 0.0

                # 构造单条记录，先粗填充，再统一截断
                source_value = str(item.get("source") or item.get("source_id") or "")
                ue = item.get("unsupervised_evaluation") or {}
                if not isinstance(ue, dict):
                    ue = {}
                ue_method = str(ue.get("method") or "")
                ue_scores = ue.get("scores") or {}
                if not isinstance(ue_scores, dict):
                    ue_scores = {}
                ue_meta = ue.get("meta") or {}
                if not isinstance(ue_meta, dict):
                    ue_meta = {}
                faithfulness_raw = ue_scores.get("faithfulness")
                faithfulness = -1.0
                if isinstance(faithfulness_raw, (int, float, str)):
                    try:
                        faithfulness = float(faithfulness_raw)
                    except Exception:
                        faithfulness = -1.0
                primary_row: Dict[str, Any] = {
                    "id": item.get("id", "") or "",
                    "task_id": item.get("task_id", "") or "",
                    "original_filename": str(item.get("original_filename", "") or ""),
                    source_field: source_value,
                    "source_fact_text": (item.get("source_fact_text", "") or ""),
                    "question": (item.get("question", "") or ""),
                    "answer": (item.get("answer", "") or ""),
                    "question_type": (item.get("question_type", "简答题") or "简答题"),
                    "question_type_reason": (item.get("question_type_reason", "") or ""),
                    "answer_explanation": (item.get("answer_explanation", "") or ""),
                    category_field: kc_value,
                    category_reason_field: kc_reason,
                    category_conf_field: kc_conf,
                    "difficulty_level": (item.get("difficulty_level", "") or ""),
                    "difficulty_score": diff_score,
                    "llm_model": (model_info.get("llm_model", "") or ""),
                    "embed_model": str(embed_model_name or ""),
                    "embed_dim": embed_dim,
                    "filtered": bool(item.get("filtered", False)),
                    "average_score": avg_score,
                    "faithfulness": faithfulness,
                    "evaluation_method": (item.get("evaluation_method", "") or ""),
                    "llm_scores": _json_dumps_minified(
                        llm_eval.get("scores", {}) if isinstance(llm_eval, dict) else {},
                    ),
                    "llm_reasons": _json_dumps_minified(
                        llm_eval.get("reasons", {}) if isinstance(llm_eval, dict) else {},
                    ),
                    "local_scores": _json_dumps_minified(
                        local_eval.get("scores", {}) if isinstance(local_eval, dict) else {},
                    ),
                    "unsupervised_method": ue_method,
                    "unsupervised_scores": _json_dumps_minified(ue_scores),
                    "unsupervised_meta": _serialize_unsupervised_meta_for_milvus(
                        ue_meta,
                        ue_method=ue_method,
                        max_bytes=max_len_map["unsupervised_meta"],
                    ),
                    "is_primary": bool(item.get("is_primary", not item.get("is_augmented"))),
                    "is_augmented": bool(item.get("is_augmented", False)),
                    "variant_of": str(item.get("variant_of", "") or ""),
                    "created_at": int(item.get("created_at", time.time())),
                    "filter_basis": (item.get("filter_basis", "") or ""),
                }

                primary_row = _prepare_row(primary_row)
                batch_data.append(primary_row)
                text_for_embeddings.append(text_for_embedding)

                # Flatten similar_questions (augmentation variants) into separate Milvus rows.
                # This allows admin/search to directly find augmented queries.
                similar_questions = item.get("similar_questions")
                if isinstance(similar_questions, list) and similar_questions:
                    primary_id = str(item.get("id") or "")
                    base_task_id = str(item.get("task_id") or "")
                    base_filename = str(item.get("original_filename") or "")
                    base_answer = str(item.get("answer") or "")
                    for sq in similar_questions:
                        if not isinstance(sq, dict):
                            continue
                        variant_question = str(sq.get("question") or "").strip()
                        if not variant_question:
                            continue
                        variant_answer = str(sq.get("answer") or base_answer)
                        variant_id = hashlib.sha1(
                            f"{primary_id}|||{variant_question}|||{variant_answer}".encode("utf-8")
                        ).hexdigest()
                        variant_text_for_embedding = f"{variant_question} [SEP] {variant_answer}"
                        variant_avg = sq.get("score")
                        try:
                            variant_avg_score = (
                                float(variant_avg)
                                if isinstance(variant_avg, (int, float, str)) and str(variant_avg).strip() != ""
                                else avg_score
                            )
                        except Exception:
                            variant_avg_score = avg_score
                        variant_row = dict(primary_row)
                        variant_row.update(
                            {
                                "id": variant_id,
                                "task_id": base_task_id,
                                "original_filename": base_filename,
                                "question": variant_question,
                                "answer": variant_answer,
                                "question_type": str(sq.get("question_type") or item.get("question_type") or "简答题"),
                                "answer_explanation": str(
                                    sq.get("answer_explanation") or item.get("answer_explanation") or ""
                                ),
                                "average_score": float(variant_avg_score),
                                "is_primary": False,
                                "is_augmented": True,
                                "variant_of": primary_id,
                            }
                        )
                        variant_row = _prepare_row(variant_row)
                        batch_data.append(variant_row)
                        text_for_embeddings.append(variant_text_for_embedding)
            except Exception as row_exc:
                # 单条记录异常不影响整体插入，跳过并打印原因
                print(f"? 跳过一条 QA 记录，存储前校验失败: {row_exc}")
                continue

        if not batch_data:
            return {"success": False, "message": "无可用 QA 记录存储到 Milvus", "stored_count": 0}
        # Best-effort de-duplication: remove existing entities with the same primary key.
        # This helps when re-ingesting a consolidated JSON or rerunning a task.
        try:
            ids_to_delete = [str(r.get("id") or "") for r in batch_data if r.get("id")]
            if ids_to_delete:
                chunk_size = 200
                for i in range(0, len(ids_to_delete), chunk_size):
                    chunk = [x for x in ids_to_delete[i : i + chunk_size] if x]
                    if not chunk:
                        continue
                    expr = 'id in [' + ",".join(f'"{x}"' for x in chunk) + "]"
                    _rt.milvus_client.delete(expr=expr)
                _rt.milvus_client.flush()
        except Exception as dedup_exc:
            logger.warning("Milvus de-dup skipped: %s", dedup_exc)
        print(f"正在为{len(text_for_embeddings)}条记录生成嵌入向量...")
        embeddings = generate_embeddings(text_for_embeddings)
        for idx, embedding in enumerate(embeddings):
            batch_data[idx]["embedding_vector"] = embedding
        print(f"正在将{len(batch_data)}条记录插入Milvus...")
        _rt.milvus_client.insert(batch_data)
        _rt.milvus_client.flush()
        try:
            upsert_qa_debug_items(items)
        except Exception as debug_exc:
            logger.warning("Persist qa debug payloads skipped: %s", debug_exc)
        print(f" 成功存储{len(batch_data)}条问答对到Milvus")
        if filtered_missing_fields:
            print(f"⚠️ 部分字段不在现有集合 schema 中，已忽略: {sorted(filtered_missing_fields)}")
        return {
            "success": True,
            "message": f"成功存储{len(batch_data)}条记录",
            "stored_count": len(batch_data),
        }
    except Exception as exc:
        error_msg = f"存储到Milvus失败: {str(exc)}"
        print(f"? {error_msg}")
        return {"success": False, "message": error_msg, "stored_count": 0}


def search_qa_pairs_in_milvus(
    query_text: str,
    top_k: int = 10,
    task_id: Optional[str] = None,
    only_filtered: Optional[bool] = None,
    min_avg_score: Optional[float] = None,
    categories: Optional[List[str]] = None,
    question_types: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    themes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if not _rt.MILVUS_AVAILABLE or not _rt.milvus_client:
        return {"success": False, "message": "Milvus未启用或未连接", "results": []}
    try:
        model = load_embedding_model()
        query_embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
        allowed_fields: Optional[set] = None
        try:
            allowed_fields = {f.name for f in _rt.milvus_client.schema.fields} if _rt.milvus_client else None
        except Exception:
            allowed_fields = None
        source_field = _resolve_source_field_name(allowed_fields)
        category_field, category_reason_field, category_conf_field = _resolve_category_field_names(allowed_fields)

        filter_expressions = []
        if task_id:
            filter_expressions.append(f'task_id == "{task_id}"')
        if only_filtered is not None:
            filter_expressions.append(f"filtered == {str(only_filtered).lower()}")
        if min_avg_score is not None:
            filter_expressions.append(f"average_score >= {min_avg_score}")
        effective_categories = categories or themes
        if effective_categories:
            cate_expr = " or ".join([f'{category_field} == \"{cate}\"' for cate in effective_categories])
            filter_expressions.append(f"({cate_expr})")
        if question_types:
            qtype_expr = " or ".join([f'question_type == \"{qt}\"' for qt in question_types])
            filter_expressions.append(f"({qtype_expr})")
        if difficulty_levels:
            diff_expr = " or ".join([f'difficulty_level == \"{dl}\"' for dl in difficulty_levels])
            filter_expressions.append(f"({diff_expr})")
        filter_expr = " and ".join(filter_expressions) if filter_expressions else None
        search_params = CONFIG["milvus"]["search_params"]
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
            category_field,
            category_reason_field,
            category_conf_field,
            "difficulty_level",
            "difficulty_score",
            "llm_model",
            "embed_model",
            "embed_dim",
            "filtered",
            "average_score",
            "faithfulness",
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
        if allowed_fields is not None:
            output_fields = [f for f in output_fields if f in allowed_fields]
        results = _rt.milvus_client.search(
            data=[query_embedding],
            anns_field="embedding_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields,
        )
        search_results: List[Dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                if not entity:
                    continue
                llm_scores_raw = entity.get("llm_scores") if entity else None
                llm_reasons_raw = entity.get("llm_reasons") if entity else None
                local_scores_raw = entity.get("local_scores") if entity else None
                unsup_method_raw = entity.get("unsupervised_method") if entity else None
                unsup_scores_raw = entity.get("unsupervised_scores") if entity else None
                unsup_meta_raw = entity.get("unsupervised_meta") if entity else None
                search_results.append(
                    {
                        "id": entity.get("id"),
                        "similarity_score": float(hit.score),
                        "task_id": entity.get("task_id"),
                        "original_filename": entity.get("original_filename"),
                        "source": entity.get(source_field),
                        "source_fact_text": entity.get("source_fact_text"),
                        "question": entity.get("question"),
                        "answer": entity.get("answer"),
                        "question_type": entity.get("question_type"),
                        "question_type_reason": entity.get("question_type_reason"),
                        "answer_explanation": entity.get("answer_explanation"),
                        "knowledge_category": entity.get(category_field),
                        "knowledge_category_reason": entity.get(category_reason_field),
                        "knowledge_category_confidence": entity.get(category_conf_field),
                        "difficulty_level": entity.get("difficulty_level"),
                        "difficulty_score": entity.get("difficulty_score"),
                        "llm_model": entity.get("llm_model"),
                        "embed_model": entity.get("embed_model"),
                        "embed_dim": entity.get("embed_dim"),
                        "filtered": entity.get("filtered"),
                        "average_score": entity.get("average_score"),
                        "faithfulness": entity.get("faithfulness"),
                        "evaluation_method": entity.get("evaluation_method"),
                        "llm_scores": json.loads(llm_scores_raw or "{}"),
                        "llm_reasons": json.loads(llm_reasons_raw or "{}"),
                        "local_scores": json.loads(local_scores_raw or "{}"),
                        "unsupervised_evaluation": {
                            "method": unsup_method_raw,
                            "scores": json.loads(unsup_scores_raw or "{}"),
                            "meta": json.loads(unsup_meta_raw or "{}"),
                        }
                        if (
                            unsup_method_raw
                            or (unsup_scores_raw not in (None, "", "{}", "null"))
                            or (unsup_meta_raw not in (None, "", "{}", "null"))
                        )
                        else None,
                        "created_at": entity.get("created_at"),
                        "filter_basis": entity.get("filter_basis"),
                        "is_primary": entity.get("is_primary"),
                        "is_augmented": entity.get("is_augmented"),
                        "variant_of": entity.get("variant_of"),
                    }
                )
        return {
            "success": True,
            "message": f"找到{len(search_results)}条相关结果",
            "results": search_results,
            "total_count": len(search_results),
        }
    except Exception as exc:
        error_msg = f"Milvus搜索失败: {str(exc)}"
        print(f"? {error_msg}")
        return {"success": False, "message": error_msg, "results": []}

__all__ = ['search_qa_pairs_in_milvus', 'store_qa_pairs_to_milvus']
