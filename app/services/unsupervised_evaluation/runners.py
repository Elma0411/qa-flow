# 文件作用：调用各类无监督评估器并返回单项指标结果。
# 关联说明：依赖 runtime 加载评估器，供 suite.py 执行单项指标。

from __future__ import annotations

from typing import Dict, List, Optional

from app.core.config import CONFIG
from .runtime import UNSUPERVISED_EVALUATION_RUNTIME as _rt


def execute_unsupervised_faithfulness_blocking(
    qa_items: List[Dict[str, object]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    only_primary: bool = True,
    hypothesis_mode: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_request_timeout: Optional[int] = None,
    llm_max_retries: Optional[int] = None,
    llm_max_concurrency: Optional[int] = None,
) -> Dict[str, object]:
    if not _rt.UNSUPERVISED_FAITHFULNESS_AVAILABLE or _rt.attach_faithfulness is None:
        raise RuntimeError(
            "忠实度评价不可用（缺少 transformers/torch 或导入失败）："
            f"{_rt.UNSUPERVISED_FAITHFULNESS_IMPORT_ERROR or ''}"
        )

    cfg = (CONFIG.get("unsupervised") or {}) if isinstance(CONFIG, dict) else {}
    resolved_model_path = model_path or cfg.get("nli_model_path")
    resolved_device = device or cfg.get("nli_device")
    resolved_max_length = int(max_length or cfg.get("nli_max_length") or 512)
    resolved_batch_size = int(batch_size or cfg.get("nli_batch_size") or 16)
    resolved_hypothesis_mode = (
        str(hypothesis_mode or cfg.get("hypothesis_mode") or "llm").strip().lower()
    )
    if resolved_hypothesis_mode != "llm":
        resolved_hypothesis_mode = "llm"

    kwargs = {
        "model_path": resolved_model_path,
        "device": resolved_device,
        "max_length": resolved_max_length,
        "batch_size": resolved_batch_size,
        "only_primary": only_primary,
        "hypothesis_mode": resolved_hypothesis_mode,
        "llm_api_key": llm_api_key or cfg.get("hypothesis_api_key") or CONFIG.get("api_key"),
        "llm_base_url": llm_base_url or cfg.get("hypothesis_base_url") or CONFIG.get("base_url"),
        "llm_model": llm_model or cfg.get("hypothesis_model") or CONFIG.get("model"),
        "llm_request_timeout": int(
            llm_request_timeout or cfg.get("hypothesis_timeout") or CONFIG.get("request_timeout") or 60
        ),
        "llm_max_retries": int(
            llm_max_retries or cfg.get("hypothesis_max_retries") or CONFIG.get("max_retries") or 2
        ),
    }
    configured_concurrency = llm_max_concurrency or cfg.get("hypothesis_max_concurrency")
    if configured_concurrency is not None:
        kwargs["llm_max_concurrency"] = int(configured_concurrency)

    summary = _rt.attach_faithfulness(qa_items, **kwargs)
    if isinstance(summary, dict):
        summary = dict(summary)
        summary["hypothesis_llm"] = {
            "base_url": str(kwargs.get("llm_base_url") or ""),
            "model": str(kwargs.get("llm_model") or ""),
            "api_key_present": bool(str(kwargs.get("llm_api_key") or "").strip()),
            "timeout_seconds": int(kwargs.get("llm_request_timeout") or 0) or None,
            "max_retries": int(kwargs.get("llm_max_retries") or 0) or None,
            "max_concurrency": int(kwargs.get("llm_max_concurrency") or 0) or None,
        }
    return summary


def execute_unsupervised_answerability_blocking(
    qa_items: List[Dict[str, object]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
    doc_stride: Optional[int] = None,
    max_answer_length: Optional[int] = None,
    n_best: Optional[int] = None,
    batch_size: Optional[int] = None,
    score_mode: Optional[str] = None,
    temperature: Optional[float] = None,
    softmax_topk: Optional[int] = None,
    use_fast_tokenizer: Optional[bool] = None,
    only_primary: bool = True,
) -> Dict[str, object]:
    if not _rt.UNSUPERVISED_ANSWERABILITY_AVAILABLE or _rt.attach_answerability is None:
        raise RuntimeError(
            "可回答性评价不可用（缺少 transformers/torch 或导入失败）："
            f"{_rt.UNSUPERVISED_ANSWERABILITY_IMPORT_ERROR or ''}"
        )

    cfg = (CONFIG.get("unsupervised") or {}) if isinstance(CONFIG, dict) else {}
    resolved_model_path = model_path or cfg.get("qa_model_path")
    resolved_device = device or cfg.get("qa_device")
    resolved_max_length = int(max_length or cfg.get("qa_max_length") or 384)
    resolved_doc_stride = int(doc_stride or cfg.get("qa_doc_stride") or 128)
    resolved_max_answer_length = int(max_answer_length or cfg.get("qa_max_answer_length") or 64)
    resolved_n_best = int(n_best or cfg.get("qa_n_best") or 20)
    resolved_batch_size = int(batch_size or cfg.get("qa_batch_size") or 8)
    resolved_score_mode = str(score_mode or cfg.get("qa_score_mode") or "auto").strip().lower()
    resolved_temperature = float(
        temperature if temperature is not None else (cfg.get("qa_temperature") or 1.0)
    )
    resolved_softmax_topk = int(softmax_topk or cfg.get("qa_softmax_topk") or 8)
    resolved_use_fast = (
        bool(use_fast_tokenizer)
        if use_fast_tokenizer is not None
        else bool(cfg.get("qa_use_fast_tokenizer", True))
    )

    return _rt.attach_answerability(
        qa_items,
        model_path=resolved_model_path,
        device=resolved_device,
        max_length=resolved_max_length,
        doc_stride=resolved_doc_stride,
        max_answer_length=resolved_max_answer_length,
        n_best=resolved_n_best,
        batch_size=resolved_batch_size,
        score_mode=resolved_score_mode,
        temperature=resolved_temperature,
        softmax_topk=resolved_softmax_topk,
        use_fast_tokenizer=resolved_use_fast,
        only_primary=only_primary,
    )


def execute_unsupervised_coverage_recall_blocking(
    qa_items: List[Dict[str, object]],
    *,
    embed_model_path: Optional[str] = None,
    device: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    unit_type: Optional[str] = None,
    qa_text_mode: Optional[str] = None,
    similarity_mapping: Optional[str] = None,
    sigmoid_temperature: Optional[float] = None,
    tau: Optional[float] = None,
    auto_tau: Optional[bool] = None,
    neg_quantile: Optional[float] = None,
    neg_samples_per_group: Optional[int] = None,
    random_seed: Optional[int] = None,
    min_unit_chars: Optional[int] = None,
    max_units: Optional[int] = None,
    only_primary: bool = True,
) -> Dict[str, object]:
    if not _rt.UNSUPERVISED_COVERAGE_AVAILABLE or _rt.attach_coverage_recall is None:
        raise RuntimeError(
            "Coverage Recall 评价不可用（缺少 sentence-transformers/torch/numpy 或导入失败）："
            f"{_rt.UNSUPERVISED_COVERAGE_IMPORT_ERROR or ''}"
        )

    cfg = (CONFIG.get("unsupervised") or {}) if isinstance(CONFIG, dict) else {}
    resolved_embed_model_path = embed_model_path or cfg.get("coverage_embed_model_path")
    resolved_device = device or cfg.get("coverage_device")
    resolved_batch_size = int(embed_batch_size or cfg.get("coverage_embed_batch_size") or 32)
    resolved_unit_type = str(unit_type or cfg.get("coverage_unit_type") or "clause_sentence")
    resolved_qa_text_mode = str(qa_text_mode or "qa")
    resolved_sim_mapping = str(similarity_mapping or cfg.get("coverage_sim_mapping") or "clip0")
    resolved_sigmoid_temp = float(
        sigmoid_temperature
        if sigmoid_temperature is not None
        else (cfg.get("coverage_sigmoid_temperature") or 0.08)
    )
    resolved_tau = tau if tau is not None else cfg.get("coverage_tau")
    resolved_auto_tau = bool(auto_tau) if auto_tau is not None else bool(cfg.get("coverage_auto_tau", True))
    resolved_neg_q = float(
        neg_quantile if neg_quantile is not None else (cfg.get("coverage_neg_quantile") or 0.95)
    )
    resolved_neg_samples = int(
        neg_samples_per_group
        if neg_samples_per_group is not None
        else (cfg.get("coverage_neg_samples_per_group") or 24)
    )
    resolved_seed = int(random_seed if random_seed is not None else (cfg.get("coverage_random_seed") or 13))
    resolved_min_chars = int(min_unit_chars or cfg.get("coverage_min_unit_chars") or 10)
    resolved_max_units = int(max_units or cfg.get("coverage_max_units") or 256)

    return _rt.attach_coverage_recall(
        qa_items,
        embed_model_path=resolved_embed_model_path,
        device=resolved_device,
        embed_batch_size=resolved_batch_size,
        unit_type=resolved_unit_type,
        qa_text_mode=resolved_qa_text_mode,
        similarity_mapping=resolved_sim_mapping,
        sigmoid_temperature=resolved_sigmoid_temp,
        tau=float(resolved_tau) if isinstance(resolved_tau, (int, float)) else None,
        auto_tau=resolved_auto_tau,
        neg_quantile=resolved_neg_q,
        neg_samples_per_group=resolved_neg_samples,
        random_seed=resolved_seed,
        min_unit_chars=resolved_min_chars,
        max_units=resolved_max_units,
        only_primary=only_primary,
    )


def execute_unsupervised_fluency_ppl_blocking(
    qa_items: List[Dict[str, object]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    sentence_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    norm_alpha: Optional[float] = None,
    norm_beta: Optional[float] = None,
    text_mode: Optional[str] = None,
    only_primary: bool = True,
) -> Dict[str, object]:
    if not _rt.UNSUPERVISED_FLUENCY_PPL_AVAILABLE or _rt.attach_fluency_ppl is None:
        raise RuntimeError(
            "困惑度(PPL)流畅度评价不可用（缺少 transformers/torch 或导入失败）："
            f"{_rt.UNSUPERVISED_FLUENCY_PPL_IMPORT_ERROR or ''}"
        )

    cfg = (CONFIG.get("unsupervised") or {}) if isinstance(CONFIG, dict) else {}
    resolved_model_path = model_path or cfg.get("fluency_model_path")
    resolved_device = device or cfg.get("fluency_device")
    resolved_sentence_length = int(sentence_length or cfg.get("fluency_sentence_length") or 100)
    resolved_batch_size = int(batch_size or cfg.get("fluency_batch_size") or 100)
    resolved_temperature = float(
        temperature if temperature is not None else (cfg.get("fluency_temperature") or 1.0)
    )
    resolved_norm_alpha = float(
        norm_alpha if norm_alpha is not None else (cfg.get("fluency_norm_alpha") or 0.01)
    )
    resolved_norm_beta = float(
        norm_beta if norm_beta is not None else (cfg.get("fluency_norm_beta") or 0.8)
    )
    resolved_text_mode = str(text_mode or cfg.get("fluency_text_mode") or "qa").strip().lower()
    if resolved_text_mode not in {"qa", "answer", "question"}:
        resolved_text_mode = "qa"

    return _rt.attach_fluency_ppl(
        qa_items,
        model_path=resolved_model_path,
        device=resolved_device,
        sentence_length=resolved_sentence_length,
        batch_size=resolved_batch_size,
        temperature=resolved_temperature,
        norm_alpha=resolved_norm_alpha,
        norm_beta=resolved_norm_beta,
        text_mode=resolved_text_mode,
        only_primary=only_primary,
    )


__all__ = [
    "execute_unsupervised_answerability_blocking",
    "execute_unsupervised_coverage_recall_blocking",
    "execute_unsupervised_faithfulness_blocking",
    "execute_unsupervised_fluency_ppl_blocking",
]
