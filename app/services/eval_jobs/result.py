# 文件作用：读取评测结果并支持结果入库。
# 关联说明：读取 run.py 写出的评分结果，并支撑 service.py 的分页和入库。

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import CONFIG
from .common import (
    _safe_float,
    _sanitize_task_token,
    _write_json,
    iter_scored_items,
)
from app.services.milvus import store_qa_pairs_to_milvus


def read_scored_items_page(
    scored_jsonl_path: str,
    *,
    offset: int,
    limit: int,
    threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(500, int(limit or 50)))
    threshold_value = float(threshold) if threshold is not None else None

    items: List[Dict[str, Any]] = []
    total = 0
    for row in iter_scored_items(scored_jsonl_path):
        if threshold_value is not None:
            ue = row.get("unsupervised_evaluation") or {}
            scores = ue.get("scores") if isinstance(ue, dict) else {}
            uf1 = scores.get("unsupervised_f1") if isinstance(scores, dict) else None
            filtered = _safe_float(uf1, 0.0) >= float(threshold_value)
            row = dict(row)
            row["filtered"] = bool(filtered)
        if total >= safe_offset and len(items) < safe_limit:
            items.append(row)
        total += 1
    return items, total


def ingest_scored_items_to_milvus(
    scored_jsonl_path: str,
    *,
    dataset_name: str,
    threshold: float,
    enable_vector_storage: bool,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = str(dataset_name or "").strip()
    if not dataset:
        raise ValueError("dataset_name 不能为空")
    job_token = _sanitize_task_token(str(job_id or "")[:12] or str(int(time.time())))
    dataset_token = _sanitize_task_token(dataset)[:48]
    task_id = f"eval_import_{dataset_token}_{job_token}_{int(time.time())}"
    threshold_value = float(threshold)

    items: List[Dict[str, Any]] = []
    created_at = int(time.time())
    for row in iter_scored_items(scored_jsonl_path):
        ue = row.get("unsupervised_evaluation") or {}
        scores = ue.get("scores") if isinstance(ue, dict) else {}
        uf1 = scores.get("unsupervised_f1") if isinstance(scores, dict) else None
        uf1_val = _safe_float(uf1, 0.0)
        if uf1_val < threshold_value:
            continue

        evaluation = (
            row.get("evaluation")
            if isinstance(row.get("evaluation"), dict)
            else {"llm": None, "local": None}
        )
        items.append(
            {
                "id": row.get("id") or "",
                "task_id": task_id,
                "original_filename": row.get("original_filename") or "",
                "source": row.get("group_id") or "",
                "source_fact_text": row.get("context") or row.get("source_fact_text") or "",
                "question": row.get("question") or "",
                "answer": row.get("answer") or "",
                "question_type": row.get("question_type") or "简答题",
                "answer_explanation": row.get("answer_explanation") or "",
                "filtered": True,
                "average_score": uf1_val,
                "evaluation_method": "unsupervised_f1",
                "evaluation": evaluation,
                "unsupervised_evaluation": ue,
                "text_for_embedding": row.get("text_for_embedding")
                or f"{row.get('question', '')} [SEP] {row.get('answer', '')}",
                "created_at": created_at,
                "filter_basis": "unsupervised_f1",
                "is_primary": True,
                "is_augmented": False,
            }
        )

    consolidated = {
        "task_id": task_id,
        "params": {"threshold": threshold_value, "dataset_name": dataset},
        "model": {
            "llm_model": CONFIG.get("model"),
            "embed_model": CONFIG["milvus"]["embedding_model"],
            "embed_dim": CONFIG["milvus"]["vector_dim"],
        },
        "items": items,
    }

    out_path = os.path.join(
        CONFIG["outputs_dir"],
        f"eval_import_{dataset}_consolidated_{int(time.time())}.json",
    )
    _write_json(out_path, consolidated)
    milvus_res = store_qa_pairs_to_milvus(out_path, bool(enable_vector_storage))
    return {
        "task_id": task_id,
        "milvus_task_id": task_id,
        "threshold": threshold_value,
        "selected": len(items),
        "consolidated_file": os.path.relpath(out_path, start=".").replace("\\", "/"),
        "milvus": milvus_res,
    }


__all__ = ["ingest_scored_items_to_milvus", "read_scored_items_page"]
