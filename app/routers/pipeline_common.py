# 文件作用：提供流水线路由共用的数据模型、文件解析和评分工具。
# 关联说明：为 batch、evaluation、history 等 pipeline 路由提供共享模型和工具。

import csv
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from app.core.clients import get_default_openai_client
from app.services.pipeline_common import (
    _ARTIFACT_TTL_SECONDS,
    _compute_average_scores_for_result,
    _normalize_artifact_path,
    _parent_key,
    parse_few_shot_examples,
)


def _collect_pipeline_output_artifact_paths(status_payload: Dict[str, Any]) -> List[str]:
    outputs = status_payload.get("outputs") if isinstance(status_payload.get("outputs"), list) else []
    collected: List[str] = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        for key in ("consolidated_json", "consolidated_csv", "evaluation_json", "debug_jsonl"):
            raw = str(output.get(key) or "").strip()
            if raw:
                collected.append(raw)
        for key in ("evaluation_json_files", "debug_json_files"):
            values = output.get(key)
            if not isinstance(values, list):
                continue
            for raw in values:
                text = str(raw or "").strip()
                if text:
                    collected.append(text)
    return collected


def get_llm_client():
    return get_default_openai_client()


class TextInput(BaseModel):
    text: str
    chunk_size: Optional[int] = 600


class PipelineInput(BaseModel):
    text: str
    chunk_size: Optional[int] = 600
    qa_per_fact: Optional[int] = 2


class EvaluationConfig(BaseModel):
    method: str = "llm"
    use_local_models: bool = True
    filter_by_threshold: bool = False
    score_threshold: float = 0.7


def _build_filtered_qa_row(
    qa_result: Dict[str, Any],
    criteria_list: List[str],
) -> Dict[str, Any]:
    evaluation = qa_result.get("evaluation", {})
    qa_item: Dict[str, Any] = {
        "question": qa_result.get("question", ""),
        "answer": qa_result.get("answer", ""),
        "knowledge_category": qa_result.get("knowledge_category", ""),
        "knowledge_category_confidence": qa_result.get("knowledge_category_confidence"),
        "question_type": qa_result.get("question_type"),
        "difficulty_level": qa_result.get("difficulty_level"),
        "difficulty_score": qa_result.get("difficulty_score"),
        "average_score": qa_result.get("average_score", 0.0),
        "evaluation_scores": {},
    }
    for metric in criteria_list:
        metric_entry = evaluation.get(metric)
        if isinstance(metric_entry, dict) and "score" in metric_entry:
            qa_item["evaluation_scores"][metric] = metric_entry["score"]
    return qa_item


def _write_filtered_qa_csv(
    csv_file: str,
    qa_items: List[Dict[str, Any]],
    criteria_list: List[str],
) -> None:
    fieldnames = [
        "question",
        "answer",
        "knowledge_category",
        "knowledge_category_confidence",
        "question_type",
        "difficulty_level",
        "difficulty_score",
        "average_score",
    ] + [f"{metric}_score" for metric in criteria_list]
    with open(csv_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for qa_item in qa_items:
            row = {
                "question": qa_item["question"],
                "answer": qa_item["answer"],
                "knowledge_category": qa_item["knowledge_category"],
                "knowledge_category_confidence": qa_item.get("knowledge_category_confidence"),
                "question_type": qa_item.get("question_type"),
                "difficulty_level": qa_item.get("difficulty_level"),
                "difficulty_score": qa_item.get("difficulty_score"),
                "average_score": qa_item["average_score"],
            }
            scores_map = qa_item.get("evaluation_scores", {})
            for metric in criteria_list:
                row[f"{metric}_score"] = scores_map.get(metric, 0)
            writer.writerow(row)


__all__ = [
    '_ARTIFACT_TTL_SECONDS',
    '_build_filtered_qa_row',
    '_collect_pipeline_output_artifact_paths',
    '_compute_average_scores_for_result',
    '_normalize_artifact_path',
    '_parent_key',
    '_write_filtered_qa_csv',
    'EvaluationConfig',
    'PipelineInput',
    'TextInput',
    'get_llm_client',
    'parse_few_shot_examples',
]
