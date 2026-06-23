# 文件作用：计算无监督流畅度和困惑度指标。
# 关联说明：与其他 unsupervised_* 文件并列，依赖 language_models 提供流畅度指标。

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.core.runtime_paths import DEFAULT_FLUENCY_MODEL_NAME, resolve_model_reference
from qa.qa_evaluation.unsupervised_runtime import (
    get_or_create_infer_lock,
    release_cached_models_for_device,
    select_torch_device,
)

try:
    import torch

    from qa.qa_evaluation.language_models import MaskedBert

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    MaskedBert = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


FluencyTextMode = Literal["qa", "answer", "question"]


DEFAULT_FLUENCY_MODEL_PATH = resolve_model_reference(
    os.environ.get("UNSUPERVISED_FLUENCY_MODEL_PATH"),
    default_name=DEFAULT_FLUENCY_MODEL_NAME,
)
DEFAULT_FLUENCY_DEVICE = os.environ.get("UNSUPERVISED_FLUENCY_DEVICE", "auto").strip().lower()
DEFAULT_FLUENCY_SENTENCE_LENGTH = int(os.environ.get("UNSUPERVISED_FLUENCY_SENTENCE_LENGTH", "100") or 100)
DEFAULT_FLUENCY_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_FLUENCY_BATCH_SIZE", "100") or 100)
DEFAULT_FLUENCY_TEMPERATURE = float(os.environ.get("UNSUPERVISED_FLUENCY_TEMPERATURE", "1.0") or 1.0)
DEFAULT_FLUENCY_TEXT_MODE = str(os.environ.get("UNSUPERVISED_FLUENCY_TEXT_MODE", "qa") or "qa").strip().lower()
if DEFAULT_FLUENCY_TEXT_MODE not in {"qa", "answer", "question"}:
    DEFAULT_FLUENCY_TEXT_MODE = "qa"

# Normalize perplexity (lower is better) into a 0~1 fluency score (higher is better).
# This keeps the metric purely perplexity-driven (no grammar/spell checking).
DEFAULT_FLUENCY_NORM_ALPHA = float(os.environ.get("UNSUPERVISED_FLUENCY_NORM_ALPHA", "0.01") or 0.01)
DEFAULT_FLUENCY_NORM_BETA = float(os.environ.get("UNSUPERVISED_FLUENCY_NORM_BETA", "0.8") or 0.8)


def _select_device(device: Optional[str]) -> str:
    return select_torch_device(device, default_device=DEFAULT_FLUENCY_DEVICE, torch_module=torch)


def _to_char_seq(text: str) -> str:
    # Keep the old implementation behavior (character-wise split for sentence segmentation),
    # but feed the underlying BERT with the joined string.
    clean = str(text or "").replace(" ", "").strip()
    if not clean:
        return ""
    return " ".join(list(clean))


def _inverse_ppl_normalize(ppl: float, *, alpha: float, beta: float) -> float:
    # ppl >= 1.0 typically; convert to (0,1] where lower ppl => higher score.
    try:
        p = float(ppl)
    except Exception:
        return 0.0
    if p <= 0:
        return 1.0
    a = float(alpha)
    b = float(beta)
    if a < 0:
        a = 0.0
    if b <= 0:
        b = 1.0
    try:
        score = 1.0 / (1.0 + a * (p**b))
    except Exception:
        score = 0.0
    return max(0.0, min(1.0, float(score)))


@dataclass(frozen=True)
class FluencyPplResult:
    fluency: float
    ppl: float
    ppl_question: Optional[float]
    ppl_answer: Optional[float]
    score_question: Optional[float]
    score_answer: Optional[float]
    weight_question: Optional[float]
    weight_answer: Optional[float]
    text_mode: FluencyTextMode


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, str, int], Any] = {}
_INFER_LOCKS: Dict[Tuple[str, str, int], threading.Lock] = {}
_INFER_LOCKS_GUARD = threading.Lock()


def _get_infer_lock(cache_key: Tuple[str, str, int]) -> threading.Lock:
    return get_or_create_infer_lock(_INFER_LOCKS, _INFER_LOCKS_GUARD, cache_key)


def _get_ppl_model(
    model_path: Optional[str],
    *,
    device: Optional[str],
    sentence_length: int,
) -> Tuple[Any, str, str, int]:
    if not TRANSFORMERS_AVAILABLE or MaskedBert is None:
        raise RuntimeError("transformers/torch 未安装，无法计算困惑度（ppl）")

    resolved_path = str(model_path or DEFAULT_FLUENCY_MODEL_PATH).strip() or DEFAULT_FLUENCY_MODEL_PATH
    resolved_device = _select_device(device)
    resolved_sentence_length = max(10, int(sentence_length or DEFAULT_FLUENCY_SENTENCE_LENGTH))
    cache_key = (resolved_path, resolved_device, resolved_sentence_length)

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, resolved_path, resolved_device, resolved_sentence_length

        if not os.path.exists(resolved_path):
            raise RuntimeError(f"困惑度模型路径不存在: {resolved_path}")

        model = MaskedBert.from_pretrained(
            resolved_path,
            device=resolved_device,
            sentence_length=resolved_sentence_length,
        )
        _MODEL_CACHE[cache_key] = model
        return model, resolved_path, resolved_device, resolved_sentence_length


def release_fluency_device_cache(device: Optional[str]) -> None:
    resolved_device = _select_device(device)
    release_cached_models_for_device(
        _MODEL_CACHE,
        _INFER_LOCKS,
        _MODEL_LOCK,
        resolved_device,
        torch_module=torch,
    )


def compute_fluency_ppl(
    question: str,
    answer: str,
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    sentence_length: int = DEFAULT_FLUENCY_SENTENCE_LENGTH,
    batch_size: int = DEFAULT_FLUENCY_BATCH_SIZE,
    temperature: float = DEFAULT_FLUENCY_TEMPERATURE,
    norm_alpha: float = DEFAULT_FLUENCY_NORM_ALPHA,
    norm_beta: float = DEFAULT_FLUENCY_NORM_BETA,
    text_mode: Optional[str] = DEFAULT_FLUENCY_TEXT_MODE,
) -> FluencyPplResult:
    q = str(question or "").strip()
    a = str(answer or "").strip()
    mode = str(text_mode or DEFAULT_FLUENCY_TEXT_MODE).strip().lower()
    if mode not in {"qa", "answer", "question"}:
        mode = "qa"
    text_mode_typed: FluencyTextMode = "qa" if mode == "qa" else ("answer" if mode == "answer" else "question")

    ppl_model, resolved_path, resolved_device, resolved_sentence_length = _get_ppl_model(
        model_path,
        device=device,
        sentence_length=sentence_length,
    )
    cache_key = (resolved_path, resolved_device, resolved_sentence_length)
    infer_lock = _get_infer_lock(cache_key)

    safe_batch = max(1, int(batch_size or DEFAULT_FLUENCY_BATCH_SIZE))
    temp = float(temperature or DEFAULT_FLUENCY_TEMPERATURE or 1.0)
    if temp <= 0:
        temp = 1.0

    def _ppl_for_text(text: str) -> Optional[float]:
        seq = _to_char_seq(text)
        if not seq:
            return None
        with infer_lock:
            try:
                return float(
                    ppl_model.perplexity(seq, temperature=temp, batch_size=safe_batch, verbose=False)
                )
            except Exception:
                return None

    ppl_q: Optional[float] = None
    ppl_a: Optional[float] = None
    score_q: Optional[float] = None
    score_a: Optional[float] = None
    w_q: Optional[float] = None
    w_a: Optional[float] = None

    if text_mode_typed == "question":
        ppl_q = _ppl_for_text(q) if q else _ppl_for_text(a)
        if ppl_q is None:
            return FluencyPplResult(
                fluency=0.0,
                ppl=float("inf"),
                ppl_question=None,
                ppl_answer=None,
                score_question=None,
                score_answer=None,
                weight_question=None,
                weight_answer=None,
                text_mode=text_mode_typed,
            )
        score_q = _inverse_ppl_normalize(ppl_q, alpha=norm_alpha, beta=norm_beta)
        return FluencyPplResult(
            fluency=float(score_q),
            ppl=float(ppl_q),
            ppl_question=float(ppl_q),
            ppl_answer=None,
            score_question=float(score_q),
            score_answer=None,
            weight_question=None,
            weight_answer=None,
            text_mode=text_mode_typed,
        )

    if text_mode_typed == "answer":
        ppl_a = _ppl_for_text(a) if a else _ppl_for_text(q)
        if ppl_a is None:
            return FluencyPplResult(
                fluency=0.0,
                ppl=float("inf"),
                ppl_question=None,
                ppl_answer=None,
                score_question=None,
                score_answer=None,
                weight_question=None,
                weight_answer=None,
                text_mode=text_mode_typed,
            )
        score_a = _inverse_ppl_normalize(ppl_a, alpha=norm_alpha, beta=norm_beta)
        return FluencyPplResult(
            fluency=float(score_a),
            ppl=float(ppl_a),
            ppl_question=None,
            ppl_answer=float(ppl_a),
            score_question=None,
            score_answer=float(score_a),
            weight_question=None,
            weight_answer=None,
            text_mode=text_mode_typed,
        )

    # qa mode: compute both, then length-weighted blend (purely perplexity-based).
    ppl_q = _ppl_for_text(q) if q else None
    ppl_a = _ppl_for_text(a) if a else None
    if ppl_q is None and ppl_a is None:
        return FluencyPplResult(
            fluency=0.0,
            ppl=float("inf"),
            ppl_question=None,
            ppl_answer=None,
            score_question=None,
            score_answer=None,
            weight_question=None,
            weight_answer=None,
            text_mode=text_mode_typed,
        )

    if ppl_q is not None:
        score_q = _inverse_ppl_normalize(ppl_q, alpha=norm_alpha, beta=norm_beta)
    if ppl_a is not None:
        score_a = _inverse_ppl_normalize(ppl_a, alpha=norm_alpha, beta=norm_beta)

    # Weights: longer text has higher weight; sqrt smooth to avoid extreme dominance.
    q_len = max(0, len(q))
    a_len = max(0, len(a))
    total_len = q_len + a_len
    if total_len <= 0:
        if score_a is not None:
            return FluencyPplResult(
                fluency=float(score_a),
                ppl=float(ppl_a if ppl_a is not None else float("inf")),
                ppl_question=None,
                ppl_answer=float(ppl_a) if ppl_a is not None else None,
                score_question=None,
                score_answer=float(score_a),
                weight_question=None,
                weight_answer=None,
                text_mode=text_mode_typed,
            )
        return FluencyPplResult(
            fluency=float(score_q or 0.0),
            ppl=float(ppl_q if ppl_q is not None else float("inf")),
            ppl_question=float(ppl_q) if ppl_q is not None else None,
            ppl_answer=None,
            score_question=float(score_q) if score_q is not None else None,
            score_answer=None,
            weight_question=None,
            weight_answer=None,
            text_mode=text_mode_typed,
        )

    w_q = math.sqrt(q_len / total_len) if q_len > 0 else 0.0
    w_a = math.sqrt(a_len / total_len) if a_len > 0 else 0.0
    denom = w_q + w_a
    if denom <= 0:
        denom = 1.0

    sq = float(score_q) if score_q is not None else 0.0
    sa = float(score_a) if score_a is not None else 0.0
    fluency = (w_q * sq + w_a * sa) / denom

    # Keep a representative ppl for debugging: answer ppl preferred, else question ppl.
    ppl_repr = ppl_a if ppl_a is not None else ppl_q
    return FluencyPplResult(
        fluency=max(0.0, min(1.0, float(fluency))),
        ppl=float(ppl_repr if ppl_repr is not None else float("inf")),
        ppl_question=float(ppl_q) if ppl_q is not None else None,
        ppl_answer=float(ppl_a) if ppl_a is not None else None,
        score_question=float(score_q) if score_q is not None else None,
        score_answer=float(score_a) if score_a is not None else None,
        weight_question=float(w_q),
        weight_answer=float(w_a),
        text_mode=text_mode_typed,
    )


def attach_fluency_ppl(
    qa_items: List[Dict[str, Any]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    sentence_length: int = DEFAULT_FLUENCY_SENTENCE_LENGTH,
    batch_size: int = DEFAULT_FLUENCY_BATCH_SIZE,
    temperature: float = DEFAULT_FLUENCY_TEMPERATURE,
    norm_alpha: float = DEFAULT_FLUENCY_NORM_ALPHA,
    norm_beta: float = DEFAULT_FLUENCY_NORM_BETA,
    text_mode: str = DEFAULT_FLUENCY_TEXT_MODE,
    only_primary: bool = True,
) -> Dict[str, Any]:
    """
    Mutate qa_items in-place by adding/merging:
      item["unsupervised_evaluation"]["scores"]["fluency_ppl"]
      item["unsupervised_evaluation"]["meta"]["fluency_ppl"]

    Note: "fluency_ppl" is a 0~1 score derived solely from perplexity (no spell/grammar check).
    Raw perplexity values are stored in meta for debugging.
    """
    if not qa_items:
        return {"computed": 0, "eligible": 0, "skipped": 0, "method": "masked_bert_pseudo_ppl_v1"}

    computed = 0
    eligible = 0
    scores_vals: List[float] = []

    # Force model loading once (also validates path)
    ppl_model, resolved_path, resolved_device, resolved_sentence_length = _get_ppl_model(
        model_path, device=device, sentence_length=sentence_length
    )
    _ = ppl_model

    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question and not answer:
            continue
        eligible += 1

        result = compute_fluency_ppl(
            question,
            answer,
            model_path=resolved_path,
            device=resolved_device,
            sentence_length=resolved_sentence_length,
            batch_size=batch_size,
            temperature=temperature,
            norm_alpha=norm_alpha,
            norm_beta=norm_beta,
            text_mode=text_mode,
        )

        ue = item.get("unsupervised_evaluation")
        if not isinstance(ue, dict):
            ue = {"method": "unsupervised_suite_v1", "scores": {}, "meta": {}}
        scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
        meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}

        scores = dict(scores)
        meta = dict(meta)

        scores["fluency_ppl"] = float(result.fluency)
        meta["fluency_ppl"] = {
            "method": "masked_bert_pseudo_ppl_v1",
            "text_mode": result.text_mode,
            "ppl": float(result.ppl) if math.isfinite(result.ppl) else None,
            "ppl_question": result.ppl_question,
            "ppl_answer": result.ppl_answer,
            "score_question": result.score_question,
            "score_answer": result.score_answer,
            "weight_question": result.weight_question,
            "weight_answer": result.weight_answer,
            "temperature": float(temperature),
            "batch_size": int(batch_size),
            "sentence_length": int(sentence_length),
            "normalize": {
                "alpha": float(norm_alpha),
                "beta": float(norm_beta),
                "formula": "score = 1 / (1 + alpha * ppl**beta)",
            },
            "model_path": str(resolved_path),
            "device": str(resolved_device),
        }

        ue["method"] = "unsupervised_suite_v1"
        ue["scores"] = scores
        ue["meta"] = meta
        item["unsupervised_evaluation"] = ue

        computed += 1
        scores_vals.append(float(result.fluency))

    return {
        "computed": computed,
        "eligible": eligible,
        "skipped": max(0, eligible - computed),
        "method": "masked_bert_pseudo_ppl_v1",
        "model_path": str(resolved_path),
        "device": str(resolved_device),
        "sentence_length": int(resolved_sentence_length),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "text_mode": str(text_mode),
        "macro_fluency_ppl": float(sum(scores_vals) / len(scores_vals)) if scores_vals else 0.0,
    }


__all__ = [
    "TRANSFORMERS_AVAILABLE",
    "FluencyPplResult",
    "attach_fluency_ppl",
    "compute_fluency_ppl",
    "release_fluency_device_cache",
]
