# 文件作用：执行一组无监督评估指标并组织输出结构。
# 关联说明：编排 runners 输出并调用 aggregation，是无监督评估的批量执行层。

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import CONFIG
from .runtime import UNSUPERVISED_EVALUATION_RUNTIME as _rt
from app.services.gpu import clear_cuda_runtime_for_device, gpu_stage
from .aggregation import (
    _attach_suite_aggregates,
    _compute_suite_four_scores,
    _upgrade_faithfulness_to_suite,
)
from .runners import (
    execute_unsupervised_answerability_blocking,
    execute_unsupervised_coverage_recall_blocking,
    execute_unsupervised_faithfulness_blocking,
    execute_unsupervised_fluency_ppl_blocking,
)


def execute_unsupervised_suite_blocking(
    qa_items: List[Dict[str, Any]],
    *,
    only_primary: bool = True,
    enable_faithfulness: bool = True,
    enable_answerability: bool = True,
    enable_coverage_recall: bool = True,
    enable_fluency_ppl: Optional[bool] = None,
    precision_mode: str = "answerability",
    prune_item_details: bool = True,
    faith_model_path: Optional[str] = None,
    faith_device: Optional[str] = None,
    faith_max_length: Optional[int] = None,
    faith_batch_size: Optional[int] = None,
    hypothesis_mode: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_request_timeout: Optional[int] = None,
    llm_max_retries: Optional[int] = None,
    llm_max_concurrency: Optional[int] = None,
    qa_model_path: Optional[str] = None,
    qa_device: Optional[str] = None,
    qa_max_length: Optional[int] = None,
    qa_doc_stride: Optional[int] = None,
    qa_max_answer_length: Optional[int] = None,
    qa_n_best: Optional[int] = None,
    qa_batch_size: Optional[int] = None,
    qa_score_mode: Optional[str] = None,
    qa_temperature: Optional[float] = None,
    qa_use_fast_tokenizer: Optional[bool] = None,
    coverage_embed_model_path: Optional[str] = None,
    coverage_device: Optional[str] = None,
    coverage_embed_batch_size: Optional[int] = None,
    coverage_unit_type: Optional[str] = None,
    coverage_qa_text_mode: Optional[str] = None,
    coverage_similarity_mapping: Optional[str] = None,
    coverage_tau: Optional[float] = None,
    coverage_auto_tau: Optional[bool] = None,
    coverage_neg_quantile: Optional[float] = None,
    coverage_neg_samples_per_group: Optional[int] = None,
    coverage_random_seed: Optional[int] = None,
    coverage_min_unit_chars: Optional[int] = None,
    coverage_max_units: Optional[int] = None,
    fluency_model_path: Optional[str] = None,
    fluency_device: Optional[str] = None,
    fluency_sentence_length: Optional[int] = None,
    fluency_batch_size: Optional[int] = None,
    fluency_temperature: Optional[float] = None,
    fluency_norm_alpha: Optional[float] = None,
    fluency_norm_beta: Optional[float] = None,
    fluency_text_mode: Optional[str] = None,
    device_override: Optional[str] = None,
    gpu_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not qa_items:
        return {"computed": 0, "method": "unsupervised_suite_v1"}
    if not _rt.UNSUPERVISED_EVALUATION_AVAILABLE:
        raise RuntimeError(
            "无监督评价不可用：缺少必要依赖（transformers/torch 或 sentence-transformers）"
        )

    suite_summary: Dict[str, Any] = {"method": "unsupervised_suite_v1"}
    cfg = (CONFIG.get("unsupervised") or {}) if isinstance(CONFIG, dict) else {}
    resolved_enable_fluency_ppl = (
        bool(enable_fluency_ppl)
        if enable_fluency_ppl is not None
        else bool(cfg.get("enable_fluency_ppl", False))
    )
    resolved_faith_device = device_override or faith_device
    resolved_qa_device = device_override or qa_device
    resolved_coverage_device = device_override or coverage_device
    resolved_fluency_device = device_override or fluency_device

    def _run_metric_with_stage(
        *,
        stage_name: str,
        fallback_device: Optional[str],
        release_fn: Optional[Any],
        runner: Any,
    ) -> Any:
        if gpu_job_id:
            with gpu_stage(str(gpu_job_id), stage_name) as leased_device:
                effective_device = leased_device or fallback_device
                try:
                    return runner(effective_device)
                finally:
                    if callable(release_fn):
                        try:
                            release_fn(effective_device)
                        except Exception:
                            pass
                    clear_cuda_runtime_for_device(effective_device)
        result = runner(fallback_device)
        if callable(release_fn):
            try:
                release_fn(fallback_device)
            except Exception:
                pass
        clear_cuda_runtime_for_device(fallback_device)
        return result

    if enable_faithfulness:
        try:
            faith_summary = _run_metric_with_stage(
                stage_name="unsupervised_faithfulness",
                fallback_device=resolved_faith_device,
                release_fn=_rt.release_nli_device_cache,
                runner=lambda effective_device: execute_unsupervised_faithfulness_blocking(
                    qa_items,
                    model_path=faith_model_path,
                    device=effective_device,
                    max_length=faith_max_length,
                    batch_size=faith_batch_size,
                    only_primary=only_primary,
                    hypothesis_mode=hypothesis_mode,
                    llm_api_key=llm_api_key,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    llm_request_timeout=llm_request_timeout,
                    llm_max_retries=llm_max_retries,
                    llm_max_concurrency=llm_max_concurrency,
                ),
            )
            suite_summary["faithfulness"] = faith_summary
            suite_summary["faithfulness_upgraded"] = _upgrade_faithfulness_to_suite(
                qa_items,
                only_primary=only_primary,
            )
        except Exception as exc:
            suite_summary["faithfulness_error"] = str(exc)[:800]

    if enable_answerability:
        try:
            ans_summary = _run_metric_with_stage(
                stage_name="unsupervised_answerability",
                fallback_device=resolved_qa_device,
                release_fn=_rt.release_answerability_device_cache,
                runner=lambda effective_device: execute_unsupervised_answerability_blocking(
                    qa_items,
                    model_path=qa_model_path,
                    device=effective_device,
                    max_length=qa_max_length,
                    doc_stride=qa_doc_stride,
                    max_answer_length=qa_max_answer_length,
                    n_best=qa_n_best,
                    batch_size=qa_batch_size,
                    score_mode=qa_score_mode,
                    temperature=qa_temperature,
                    use_fast_tokenizer=qa_use_fast_tokenizer,
                    only_primary=only_primary,
                ),
            )
            suite_summary["answerability"] = ans_summary
        except Exception as exc:
            suite_summary["answerability_error"] = str(exc)[:800]

    if enable_coverage_recall:
        try:
            cov_summary = _run_metric_with_stage(
                stage_name="unsupervised_coverage",
                fallback_device=resolved_coverage_device,
                release_fn=_rt.release_coverage_device_cache,
                runner=lambda effective_device: execute_unsupervised_coverage_recall_blocking(
                    qa_items,
                    embed_model_path=coverage_embed_model_path,
                    device=effective_device,
                    embed_batch_size=coverage_embed_batch_size,
                    unit_type=coverage_unit_type,
                    qa_text_mode=coverage_qa_text_mode,
                    similarity_mapping=coverage_similarity_mapping,
                    tau=coverage_tau,
                    auto_tau=coverage_auto_tau,
                    neg_quantile=coverage_neg_quantile,
                    neg_samples_per_group=coverage_neg_samples_per_group,
                    random_seed=coverage_random_seed,
                    min_unit_chars=coverage_min_unit_chars,
                    max_units=coverage_max_units,
                    only_primary=only_primary,
                ),
            )
            suite_summary["coverage_recall"] = cov_summary
        except Exception as exc:
            suite_summary["coverage_recall_error"] = str(exc)[:800]

    if resolved_enable_fluency_ppl:
        try:
            flu_summary = _run_metric_with_stage(
                stage_name="unsupervised_fluency",
                fallback_device=resolved_fluency_device,
                release_fn=_rt.release_fluency_device_cache,
                runner=lambda effective_device: execute_unsupervised_fluency_ppl_blocking(
                    qa_items,
                    model_path=fluency_model_path,
                    device=effective_device,
                    sentence_length=fluency_sentence_length,
                    batch_size=fluency_batch_size,
                    temperature=fluency_temperature,
                    norm_alpha=fluency_norm_alpha,
                    norm_beta=fluency_norm_beta,
                    text_mode=fluency_text_mode,
                    only_primary=only_primary,
                ),
            )
            suite_summary["fluency_ppl"] = flu_summary
        except Exception as exc:
            suite_summary["fluency_ppl_error"] = str(exc)[:800]

    suite_summary["suite_aggregate"] = _attach_suite_aggregates(
        qa_items,
        only_primary=only_primary,
        precision_mode=precision_mode,
        prune_item_details=prune_item_details,
    )
    suite_summary["scores"] = _compute_suite_four_scores(qa_items, only_primary=only_primary)
    return suite_summary


__all__ = ["execute_unsupervised_suite_blocking"]
