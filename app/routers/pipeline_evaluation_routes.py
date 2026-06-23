# 文件作用：提供独立问答评估与本地评价接口。
# 关联说明：复用 pipeline_common 和 evaluation 服务，提供独立评价入口。

import asyncio
import csv
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from qa.qa_evaluation.llm_quality_evaluator import DEFAULT_CONFIG as EVAL_DEFAULT_CONFIG, evaluate_qa_pairs
from qa.common import extract_first_choice_content

from app.core.clients import build_openai_client
from app.core.config import LOCAL_EVALUATION_METRICS
from app.core.logger import logger
from app.routers.pipeline_common import (
    _build_filtered_qa_row,
    _compute_average_scores_for_result,
    _write_filtered_qa_csv,
)
from app.services.evaluation import execute_local_evaluation_blocking
from app.services.storage import (
    get_output_path,
    read_uploaded_file_content,
    read_uploaded_json_file,
)

router = APIRouter()

@router.post("/batch-upload-evaluate-qa")
async def batch_upload_evaluate_qa(
    qa_files: List[UploadFile] = File(...),
    text_files: Optional[List[UploadFile]] = File(None),
    filter_by_threshold: bool = Form(False, description="是否根据平均分数阈值过滤问答对"),
    score_threshold: float = Form(0.7, description="平均分数阈值，低于此分数的问答对将被过滤"),
    save_mode: str = Form("separate", description="保存模式: 'unified' 统一保存, 'separate' 分开保存"),
):
    """
    批量上传问答对 JSON/CSV（可选配对原始文本），执行质量评估并可按阈值过滤。
    """
    qa_temp_files: List[str] = []
    text_temp_files: List[str] = []
    try:
        if not qa_files:
            raise HTTPException(status_code=400, detail="没有上传任何问答对文件")
        start_time = time.time()
        processing_results: List[Dict[str, Any]] = []
        all_evaluation_results: List[Dict[str, Any]] = []
        all_filtered_qa_data: List[Dict[str, Any]] = []
        criteria_list = ["relevance", "completeness", "accuracy", "reasonableness", "agnosticism"]
        for idx, qa_file in enumerate(qa_files):
            try:
                import tempfile

                qa_temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
                qa_data = read_uploaded_json_file(qa_file)
                json.dump(qa_data, qa_temp, ensure_ascii=False, indent=2)
                qa_temp.close()
                qa_temp_files.append(qa_temp.name)

                text_temp_path: Optional[str] = None
                if text_files and idx < len(text_files) and text_files[idx] is not None:
                    text_temp = tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".txt",
                        delete=False,
                        encoding="utf-8",
                    )
                    text_content = read_uploaded_file_content(text_files[idx])
                    text_temp.write(text_content)
                    text_temp.close()
                    text_temp_files.append(text_temp.name)
                    text_temp_path = text_temp.name

                original_input_text = None
                if text_temp_path:
                    original_input_text = EVAL_DEFAULT_CONFIG.get("input_text")
                    EVAL_DEFAULT_CONFIG["input_text"] = text_temp_path

                try:
                    evaluation_results = evaluate_qa_pairs(qa_temp.name, criteria_list)
                    filtered_qa_pairs: Optional[List[Dict[str, Any]]] = None
                    filtered_qa_data: Optional[List[Dict[str, Any]]] = None
                    if filter_by_threshold and evaluation_results.get("results"):
                        filtered_qa_pairs = []
                        for qa_result in evaluation_results["results"]:
                            avg_score = _compute_average_scores_for_result(qa_result, criteria_list)
                            qa_result["average_score"] = avg_score
                            if avg_score >= score_threshold:
                                filtered_qa_pairs.append(qa_result)
                        evaluation_results["filtered_results"] = filtered_qa_pairs
                        evaluation_results["filter_info"] = {
                            "threshold": score_threshold,
                            "original_count": len(evaluation_results["results"]),
                            "filtered_count": len(filtered_qa_pairs),
                            "removed_count": len(evaluation_results["results"]) - len(filtered_qa_pairs),
                        }
                        if filtered_qa_pairs:
                            filtered_qa_data = [
                                _build_filtered_qa_row(qa_result, criteria_list)
                                for qa_result in filtered_qa_pairs
                            ]
                            all_filtered_qa_data.extend(filtered_qa_data)

                    processing_results.append(
                        {
                            "filename": qa_file.filename,
                            "evaluation_results": evaluation_results,
                            "filtered_qa_data": filtered_qa_data,
                            "status": "success",
                        }
                    )
                    all_evaluation_results.append(evaluation_results)
                finally:
                    if original_input_text is not None:
                        EVAL_DEFAULT_CONFIG["input_text"] = original_input_text
            except Exception as inner_exc:
                processing_results.append(
                    {
                        "filename": qa_file.filename,
                        "evaluation_results": None,
                        "filtered_qa_data": None,
                        "status": "error",
                        "error": str(inner_exc),
                    }
                )

        if save_mode == "unified":
            unified_evaluation_file = get_output_path("batch_evaluation", ".json")
            with open(unified_evaluation_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "batch_results": all_evaluation_results,
                        "summary": {
                            "total_files": len(qa_files),
                            "successful_files": len([r for r in processing_results if r["status"] == "success"]),
                            "failed_files": len([r for r in processing_results if r["status"] == "error"]),
                            "criteria": criteria_list,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            evaluation_save_result: Dict[str, Any] = {
                "mode": "unified",
                "file": unified_evaluation_file,
                "total_evaluations": len(all_evaluation_results),
            }
        else:
            evaluation_files: List[Dict[str, Any]] = []
            for result in processing_results:
                if result["status"] == "success" and result["evaluation_results"]:
                    original_name = os.path.splitext(result["filename"])[0]
                    evaluation_file = get_output_path(f"evaluation_{original_name}", ".json")
                    with open(evaluation_file, "w", encoding="utf-8") as f:
                        json.dump(result["evaluation_results"], f, ensure_ascii=False, indent=2)
                    evaluation_files.append(
                        {"source_file": result["filename"], "evaluation_file": evaluation_file}
                    )
            evaluation_save_result = {
                "mode": "separate",
                "files": evaluation_files,
                "total_files": len(evaluation_files),
            }

        filtered_save_result: Optional[Dict[str, Any]] = None
        if filter_by_threshold and all_filtered_qa_data:
            if save_mode == "unified":
                filtered_json_file = get_output_path("batch_filtered_qa", ".json")
                filtered_csv_file = get_output_path("batch_filtered_qa", ".csv")
                with open(filtered_json_file, "w", encoding="utf-8") as f:
                    json.dump(all_filtered_qa_data, f, ensure_ascii=False, indent=2)
                _write_filtered_qa_csv(filtered_csv_file, all_filtered_qa_data, criteria_list)
                filtered_save_result = {
                    "mode": "unified",
                    "json_file": filtered_json_file,
                    "csv_file": filtered_csv_file,
                    "total_qa_pairs": len(all_filtered_qa_data),
                }
            else:
                filtered_files: List[Dict[str, Any]] = []
                for result in processing_results:
                    if result["status"] == "success" and result.get("filtered_qa_data"):
                        original_name = os.path.splitext(result["filename"])[0]
                        filtered_json_file = get_output_path(f"filtered_qa_{original_name}", ".json")
                        filtered_csv_file = get_output_path(f"filtered_qa_{original_name}", ".csv")
                        with open(filtered_json_file, "w", encoding="utf-8") as f:
                            json.dump(result["filtered_qa_data"], f, ensure_ascii=False, indent=2)
                        _write_filtered_qa_csv(filtered_csv_file, result["filtered_qa_data"], criteria_list)
                        filtered_files.append(
                            {
                                "source_file": result["filename"],
                                "json_file": filtered_json_file,
                                "csv_file": filtered_csv_file,
                                "qa_count": len(result["filtered_qa_data"]),
                            }
                        )
                filtered_save_result = {
                    "mode": "separate",
                    "files": filtered_files,
                    "total_files": len(filtered_files),
                }

        duration = time.time() - start_time
        return {
            "status": "success",
            "batch_mode": True,
            "total_files": len(qa_files),
            "successful_files": len([r for r in processing_results if r["status"] == "success"]),
            "failed_files": len([r for r in processing_results if r["status"] == "error"]),
            "filter_applied": filter_by_threshold,
            "evaluation_save_result": evaluation_save_result,
            "filtered_save_result": filtered_save_result,
            "processing_details": processing_results,
            "duration": f"{duration:.2f}秒",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"批量评估失败: {str(exc)}")
    finally:
        for temp_file_path in qa_temp_files + text_temp_files:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass


@router.post("/upload-evaluate-qa")
async def upload_evaluate_qa(
    qa_file: UploadFile = File(...),
    text_file: Optional[UploadFile] = None,
    filter_by_threshold: bool = Form(False, description="是否根据平均分数阈值过滤问答对"),
    score_threshold: float = Form(0.7, description="平均分数阈值，低于此分数的问答对将被过滤"),
):
    """
    上传单个问答对文件（可选配对原始文本），执行评估并可按阈值过滤。
    """
    qa_temp_file_path: Optional[str] = None
    text_temp_file_path: Optional[str] = None
    try:
        import tempfile

        qa_temp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        qa_data = read_uploaded_json_file(qa_file)
        json.dump(qa_data, qa_temp, ensure_ascii=False, indent=2)
        qa_temp.close()
        qa_temp_file_path = qa_temp.name

        if text_file is not None:
            text_temp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            )
            text_content = read_uploaded_file_content(text_file)
            text_temp.write(text_content)
            text_temp.close()
            text_temp_file_path = text_temp.name

        criteria_list = ["relevance", "completeness", "accuracy", "reasonableness", "agnosticism"]
        start_time = time.time()

        original_input_text = None
        if text_temp_file_path:
            original_input_text = EVAL_DEFAULT_CONFIG.get("input_text")
            EVAL_DEFAULT_CONFIG["input_text"] = text_temp_file_path

        try:
            client = build_openai_client(EVAL_DEFAULT_CONFIG["api_key"], EVAL_DEFAULT_CONFIG["base_url"])
            try:
                resp = client.chat.completions.create(
                    model=EVAL_DEFAULT_CONFIG["model"],
                    messages=[{"role": "user", "content": "测试连接"}],
                    max_tokens=5,
                )
                logger.info("评估 API 连接正常: %s", extract_first_choice_content(resp).strip())
            except Exception as exc:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API 连接测试失败: {str(exc)}"},
                )

            evaluation_results = evaluate_qa_pairs(qa_temp_file_path, criteria_list)
            filtered_qa_data: Optional[List[Dict[str, Any]]] = None
            filtered_qa_file: Optional[str] = None
            filtered_csv_file: Optional[str] = None

            if filter_by_threshold and evaluation_results.get("results"):
                filtered_qa_pairs: List[Dict[str, Any]] = []
                for qa_result in evaluation_results["results"]:
                    avg_score = _compute_average_scores_for_result(qa_result, criteria_list)
                    qa_result["average_score"] = avg_score
                    if avg_score >= score_threshold:
                        filtered_qa_pairs.append(qa_result)
                evaluation_results["filtered_results"] = filtered_qa_pairs
                evaluation_results["filter_info"] = {
                    "threshold": score_threshold,
                    "original_count": len(evaluation_results["results"]),
                    "filtered_count": len(filtered_qa_pairs),
                    "removed_count": len(evaluation_results["results"]) - len(filtered_qa_pairs),
                }
                if filtered_qa_pairs:
                    filtered_qa_data = [
                        _build_filtered_qa_row(qa_result, criteria_list)
                        for qa_result in filtered_qa_pairs
                    ]
                    filtered_qa_file = get_output_path("filtered_qa_single", ".json")
                    filtered_csv_file = get_output_path("filtered_qa_single", ".csv")
                    with open(filtered_qa_file, "w", encoding="utf-8") as f:
                        json.dump(filtered_qa_data, f, ensure_ascii=False, indent=2)
                    _write_filtered_qa_csv(filtered_csv_file, filtered_qa_data, criteria_list)

            output_file = get_output_path("evaluation_results", ".json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

            duration = time.time() - start_time
            result: Dict[str, Any] = {
                "status": "success",
                "evaluation": evaluation_results,
                "evaluation_file": output_file,
                "qa_filename": qa_file.filename,
                "filter_applied": filter_by_threshold,
                "duration": f"{duration:.2f}秒",
            }
            if filter_by_threshold:
                result["filter_info"] = evaluation_results.get("filter_info", {})
                if filtered_qa_file:
                    result["filtered_qa_file"] = filtered_qa_file
                if filtered_csv_file:
                    result["filtered_csv_file"] = filtered_csv_file
            if text_file is not None:
                result["text_filename"] = text_file.filename
            return result
        finally:
            if original_input_text is not None:
                EVAL_DEFAULT_CONFIG["input_text"] = original_input_text
    except Exception as exc:
        import traceback

        error_details = traceback.format_exc()
        logger.error("评估问答对失败 %s\n%s", exc, error_details)
        raise HTTPException(status_code=500, detail=f"评估问答对失败: {str(exc)}")
    finally:
        for path in (qa_temp_file_path, text_temp_file_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass



@router.post("/evaluate-qa-local")
async def evaluate_qa_local(
    qa_file: UploadFile = File(...),
    use_local_models: bool = Form(True, description="兼容参数（已不再使用）"),
    filter_by_threshold: bool = Form(False, description="是否根据分数阈值过滤问答对"),
    score_threshold: float = Form(0.7, description="分数阈值，低于此分数的问答对将被过滤"),
):
    """
    使用“自动指标评估”对问答对进行质量评估（EM/Token_F1/ROUGE_L_F1/BLEU/BERTScore）。

    说明：
    - `use_local_models` 为兼容历史客户端保留，但不再影响评估行为。
    """
    try:
        start_time = time.time()

        qa_file_path = get_output_path("temp_qa_for_local_eval", ".json")
        with open(qa_file_path, "wb") as f:
            shutil.copyfileobj(qa_file.file, f)

        try:
            with open(qa_file_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)
            if not isinstance(qa_data, list) or not qa_data:
                raise HTTPException(status_code=400, detail="问答对文件格式错误：应为包含问答对的 JSON 数组")
            required_fields = ["question", "answer"]
            for i, qa in enumerate(qa_data):
                for field in required_fields:
                    if field not in qa:
                        raise HTTPException(
                            status_code=400,
                            detail=f"问答对 {i + 1} 缺少必要字段 '{field}'",
                        )
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="文件不是有效的 JSON 格式")

        local_eval = await asyncio.to_thread(
            execute_local_evaluation_blocking, qa_data, use_local_models
        )
        evaluation_results = (
            (local_eval or {}).get("results", []) if isinstance(local_eval, dict) else []
        )
        if not isinstance(evaluation_results, list):
            evaluation_results = []

        valid_results = [
            r
            for r in evaluation_results
            if isinstance(r, dict)
            and "error" not in r
            and isinstance(r.get("average_score"), (int, float))
        ]
        if valid_results:
            avg_scores: Dict[str, float] = {}
            for metric in LOCAL_EVALUATION_METRICS:
                vals = []
                for r in valid_results:
                    ev = r.get("evaluation", {}) or {}
                    metric_entry = ev.get(metric, {}) if isinstance(ev, dict) else {}
                    score = metric_entry.get("score") if isinstance(metric_entry, dict) else None
                    if isinstance(score, (int, float)):
                        vals.append(float(score))
                avg_scores[metric] = float(sum(vals) / len(vals)) if vals else 0.0
            statistics = {
                "total_qa_pairs": len(qa_data),
                "successful_evaluations": len(valid_results),
                "failed_evaluations": len(qa_data) - len(valid_results),
                "average_scores": avg_scores,
                "overall_average": float(
                    sum(r["average_score"] for r in valid_results) / len(valid_results)
                ),
            }
        else:
            statistics = {
                "total_qa_pairs": len(qa_data),
                "successful_evaluations": 0,
                "failed_evaluations": len(qa_data),
                "average_scores": {},
                "overall_average": 0.0,
            }

        filtered_results: Optional[List[Dict[str, Any]]] = None
        filter_info: Optional[Dict[str, Any]] = None
        if filter_by_threshold and valid_results:
            filtered_results = [r for r in valid_results if r["average_score"] >= score_threshold]
            filter_info = {
                "threshold": score_threshold,
                "original_count": len(valid_results),
                "filtered_count": len(filtered_results),
                "removed_count": len(valid_results) - len(filtered_results),
            }

        output_file = get_output_path("local_evaluation_results", ".json")
        result_data: Dict[str, Any] = {
            "evaluation_method": "local_auto_metrics",
            "local_eval": local_eval,
            "statistics": statistics,
            "results": evaluation_results,
            "timestamp": time.time(),
        }
        if filter_info:
            result_data["filter_info"] = filter_info
            result_data["filtered_results"] = filtered_results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        filtered_qa_file: Optional[str] = None
        filtered_csv_file: Optional[str] = None
        if filtered_results:
            filtered_qa_file = get_output_path("filtered_qa_local", ".json")
            filtered_csv_file = get_output_path("filtered_qa_local", ".csv")
            with open(filtered_qa_file, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, ensure_ascii=False, indent=2)
            with open(filtered_csv_file, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["question", "answer", "source_fact"]
                fieldnames.extend(LOCAL_EVALUATION_METRICS)
                fieldnames.append("average_score")
                for result in filtered_results:
                    for key in result.keys():
                        if key not in fieldnames and key not in ["evaluation", "source_fact"]:
                            fieldnames.append(key)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in filtered_results:
                    ev = result.get("evaluation", {}) or {}
                    row: Dict[str, Any] = {
                        "question": result.get("question", ""),
                        "answer": result.get("answer", ""),
                        "source_fact": result.get("source_fact", ""),
                        "average_score": result.get("average_score", 0.0),
                    }
                    for metric in LOCAL_EVALUATION_METRICS:
                        metric_entry = ev.get(metric, {}) if isinstance(ev, dict) else {}
                        score = metric_entry.get("score") if isinstance(metric_entry, dict) else None
                        row[metric] = score
                    for key, value in result.items():
                        if key not in [
                            "question",
                            "answer",
                            "evaluation",
                            "source_fact",
                            "average_score",
                        ]:
                            row[key] = value
                    writer.writerow(row)

        duration = time.time() - start_time
        response: Dict[str, Any] = {
            "status": "success",
            "evaluation_method": "local_auto_metrics",
            "bertscore": (local_eval or {}).get("bertscore") if isinstance(local_eval, dict) else None,
            "statistics": statistics,
            "results_file": output_file,
            "qa_filename": qa_file.filename,
            "filter_applied": filter_by_threshold,
            "duration": f"{duration:.2f}秒",
        }
        if filter_info:
            response["filter_info"] = filter_info
        if filtered_qa_file:
            response["filtered_qa_file"] = filtered_qa_file
        if filtered_csv_file:
            response["filtered_csv_file"] = filtered_csv_file
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"本地评估失败: {str(exc)}")
    finally:
        try:
            if "qa_file_path" in locals() and os.path.exists(qa_file_path):
                os.unlink(qa_file_path)
        except Exception:
            pass



__all__ = ["router"]
