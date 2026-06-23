# 文件作用：计算无监督可回答性指标。
# 关联说明：与其他 unsupervised_* 文件并列，提供单项可回答性指标。

from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.core.runtime_paths import DEFAULT_UNSUPERVISED_QA_MODEL_NAME, resolve_model_reference
from qa.qa_evaluation.unsupervised_runtime import (
    get_or_create_infer_lock,
    release_cached_models_for_device,
    resolve_first_existing_model_path,
    select_torch_device,
)

try:
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    AutoModelForQuestionAnswering = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False


ScoreMode = Literal["auto", "softmax2", "softmax_topk", "sigmoid"]


DEFAULT_QA_MODEL_PATHS = (
    resolve_model_reference(
        os.environ.get("UNSUPERVISED_QA_MODEL_PATH"),
        default_name=DEFAULT_UNSUPERVISED_QA_MODEL_NAME,
    ),
)
DEFAULT_QA_DEVICE = os.environ.get("UNSUPERVISED_QA_DEVICE", "auto").strip().lower()
DEFAULT_QA_MAX_LENGTH = int(os.environ.get("UNSUPERVISED_QA_MAX_LENGTH", "384") or 384)
DEFAULT_QA_DOC_STRIDE = int(os.environ.get("UNSUPERVISED_QA_DOC_STRIDE", "128") or 128)
DEFAULT_QA_MAX_ANSWER_LENGTH = int(os.environ.get("UNSUPERVISED_QA_MAX_ANSWER_LENGTH", "64") or 64)
DEFAULT_QA_N_BEST = int(os.environ.get("UNSUPERVISED_QA_N_BEST", "20") or 20)
DEFAULT_QA_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_QA_BATCH_SIZE", "8") or 8)
DEFAULT_QA_SCORE_MODE = str(os.environ.get("UNSUPERVISED_QA_SCORE_MODE", "auto") or "auto").strip().lower()
if DEFAULT_QA_SCORE_MODE not in {"auto", "softmax2", "softmax_topk", "sigmoid"}:
    DEFAULT_QA_SCORE_MODE = "auto"
DEFAULT_QA_TEMPERATURE = float(os.environ.get("UNSUPERVISED_QA_TEMPERATURE", "1.0") or 1.0)
DEFAULT_QA_SOFTMAX_TOPK = int(os.environ.get("UNSUPERVISED_QA_SOFTMAX_TOPK", "8") or 8)
DEFAULT_QA_USE_FAST_TOKENIZER = str(
    os.environ.get("UNSUPERVISED_QA_USE_FAST_TOKENIZER", "true") or "true"
).strip().lower() in {"1", "true", "yes", "y"}

_RE_OPEN_ENDED = re.compile(
    r"(简述|概述|说明|介绍|阐述|描述|总结|主要内容|内容是什么|有哪些|包括哪些|包括什么|如何|怎么|流程|步骤|原则|要求|措施|办法|规定|管理内容|管理办法|管理要求|意义|作用|目的)"
)


def _resolve_default_model_path() -> str:
    return resolve_first_existing_model_path(DEFAULT_QA_MODEL_PATHS)


def _select_device(device: Optional[str]) -> str:
    return select_torch_device(device, default_device=DEFAULT_QA_DEVICE, torch_module=torch)


def _softmax2(a: float, b: float) -> Tuple[float, float]:
    m = a if a >= b else b
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    denom = ea + eb
    if denom <= 0:
        return 0.5, 0.5
    return ea / denom, eb / denom


def _softmax_list(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    denom = float(sum(exps))
    if denom <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [float(x / denom) for x in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class AnswerabilityResult:
    answerability: float
    p_no_answer: float
    gap: float
    score_best: float
    score_null: float
    best_span_text: str
    best_span_start: Optional[int]
    best_span_end: Optional[int]
    window_count: int
    best_window_index: Optional[int]


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[Any, Any]] = {}
_INFER_LOCKS: Dict[Tuple[str, str, bool], threading.Lock] = {}
_INFER_LOCKS_GUARD = threading.Lock()


def _get_infer_lock(cache_key: Tuple[str, str, bool]) -> threading.Lock:
    return get_or_create_infer_lock(_INFER_LOCKS, _INFER_LOCKS_GUARD, cache_key)


def _get_qa_bundle(
    model_path: Optional[str] = None,
    *,
    device: Optional[str] = None,
    use_fast: Optional[bool] = None,
) -> Tuple[Any, Any, str, bool]:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers/torch 未安装，无法运行 Answerability（抽取式 QA）评估")

    resolved_path = (model_path or "").strip() or _resolve_default_model_path()
    resolved_device = _select_device(device)
    resolved_fast = DEFAULT_QA_USE_FAST_TOKENIZER if use_fast is None else bool(use_fast)

    cache_key = (resolved_path, resolved_device, resolved_fast)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            tokenizer, model = cached
            return tokenizer, model, resolved_device, resolved_fast

        if not os.path.exists(resolved_path):
            raise RuntimeError(f"Answerability QA 模型路径不存在: {resolved_path}")

        tokenizer_kwargs = {"local_files_only": True, "use_fast": resolved_fast}
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_path,
                fix_mistral_regex=True,
                **tokenizer_kwargs,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(resolved_path, **tokenizer_kwargs)

        torch_dtype = None
        if resolved_device == "cpu" and torch is not None:
            torch_dtype = torch.float32
        model_kwargs = {"local_files_only": True}
        if torch_dtype is None:
            model = AutoModelForQuestionAnswering.from_pretrained(resolved_path, **model_kwargs)
        else:
            try:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    resolved_path,
                    dtype=torch_dtype,
                    **model_kwargs,
                )
            except TypeError:
                model = AutoModelForQuestionAnswering.from_pretrained(
                    resolved_path,
                    torch_dtype=torch_dtype,
                    **model_kwargs,
                )
        model.eval()
        model.to(resolved_device)

        _MODEL_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model, resolved_device, resolved_fast


def release_answerability_device_cache(device: Optional[str]) -> None:
    resolved_device = _select_device(device)
    release_cached_models_for_device(
        _MODEL_CACHE,
        _INFER_LOCKS,
        _MODEL_LOCK,
        resolved_device,
        torch_module=torch,
    )


def _find_best_span_for_feature(
    *,
    start_logits: List[float],
    end_logits: List[float],
    sequence_ids: List[Optional[int]],
    offsets: List[Optional[Tuple[int, int]]],
    context_text: str,
    max_answer_length: int,
    n_best: int,
) -> Tuple[float, str, Optional[int], Optional[int]]:
    # Null score uses the first token (CLS/<s>), common across BERT/RoBERTa/XLM-R.
    score_null = float(start_logits[0]) + float(end_logits[0])

    context_token_indexes: List[int] = []
    for idx, seq_id in enumerate(sequence_ids):
        if seq_id != 1:
            continue
        off = offsets[idx] if idx < len(offsets) else None
        if not off or off[0] is None or off[1] is None:
            continue
        if off[0] == off[1]:
            continue
        context_token_indexes.append(idx)

    if not context_token_indexes:
        return float("-inf"), "", None, None

    # Top-N start/end candidates within context tokens.
    start_candidates = sorted(
        context_token_indexes, key=lambda i: float(start_logits[i]), reverse=True
    )[: max(1, n_best)]
    end_candidates = sorted(
        context_token_indexes, key=lambda i: float(end_logits[i]), reverse=True
    )[: max(1, n_best)]

    best_score = float("-inf")
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    for s_idx in start_candidates:
        for e_idx in end_candidates:
            if e_idx < s_idx:
                continue
            length = e_idx - s_idx + 1
            if length > max_answer_length:
                continue
            score = float(start_logits[s_idx]) + float(end_logits[e_idx])
            if score > best_score:
                best_score = score
                best_start = s_idx
                best_end = e_idx

    if best_start is None or best_end is None or best_score == float("-inf"):
        return float("-inf"), "", None, None

    start_off = offsets[best_start]
    end_off = offsets[best_end]
    if not start_off or not end_off:
        return best_score, "", None, None

    char_start = int(start_off[0])
    char_end = int(end_off[1])
    if char_start < 0 or char_end <= char_start or char_end > len(context_text):
        span_text = ""
    else:
        span_text = context_text[char_start:char_end]

    return best_score, span_text, char_start, char_end


def _top_span_scores_for_feature(
    *,
    start_logits: List[float],
    end_logits: List[float],
    sequence_ids: List[Optional[int]],
    offsets: List[Optional[Tuple[int, int]]],
    max_answer_length: int,
    n_best: int,
    topk: int,
) -> List[float]:
    context_token_indexes: List[int] = []
    for idx, seq_id in enumerate(sequence_ids):
        if seq_id != 1:
            continue
        off = offsets[idx] if idx < len(offsets) else None
        if not off or off[0] is None or off[1] is None:
            continue
        if off[0] == off[1]:
            continue
        context_token_indexes.append(idx)

    if not context_token_indexes:
        return []

    safe_n = max(1, int(n_best or 1))
    safe_topk = max(1, int(topk or 1))

    start_candidates = sorted(context_token_indexes, key=lambda i: float(start_logits[i]), reverse=True)[:safe_n]
    end_candidates = sorted(context_token_indexes, key=lambda i: float(end_logits[i]), reverse=True)[:safe_n]

    scores: List[float] = []
    for s_idx in start_candidates:
        for e_idx in end_candidates:
            if e_idx < s_idx:
                continue
            length = e_idx - s_idx + 1
            if length > max(1, int(max_answer_length or 1)):
                continue
            scores.append(float(start_logits[s_idx]) + float(end_logits[e_idx]))

    if not scores:
        return []
    scores.sort(reverse=True)
    return scores[:safe_topk]


def _is_open_ended_question(question: str) -> bool:
    q = str(question or "").strip()
    if not q:
        return False
    return bool(_RE_OPEN_ENDED.search(q))


def compute_answerability(
    question: str,
    context: str,
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = DEFAULT_QA_MAX_LENGTH,
    doc_stride: int = DEFAULT_QA_DOC_STRIDE,
    max_answer_length: int = DEFAULT_QA_MAX_ANSWER_LENGTH,
    n_best: int = DEFAULT_QA_N_BEST,
    batch_size: int = DEFAULT_QA_BATCH_SIZE,
    score_mode: Optional[str] = DEFAULT_QA_SCORE_MODE,
    temperature: float = DEFAULT_QA_TEMPERATURE,
    softmax_topk: int = DEFAULT_QA_SOFTMAX_TOPK,
    use_fast_tokenizer: Optional[bool] = None,
) -> AnswerabilityResult:
    question = str(question or "").strip()
    context = str(context or "").strip()
    if not question or not context:
        return AnswerabilityResult(
            answerability=0.0,
            p_no_answer=1.0,
            gap=float("-inf"),
            score_best=float("-inf"),
            score_null=float("inf"),
            best_span_text="",
            best_span_start=None,
            best_span_end=None,
            window_count=0,
            best_window_index=None,
        )

    tokenizer, model, resolved_device, resolved_fast = _get_qa_bundle(
        model_path, device=device, use_fast=use_fast_tokenizer
    )
    if not resolved_fast:
        raise RuntimeError("Answerability 需要 fast tokenizer 以获得 offsets/overflow 窗口信息")

    mode = str(score_mode or DEFAULT_QA_SCORE_MODE).strip().lower()
    if mode not in {"auto", "softmax2", "softmax_topk", "sigmoid"}:
        mode = "auto"
    if mode == "auto":
        mode = "softmax_topk" if _is_open_ended_question(question) else "softmax2"

    safe_max_len = max(64, int(max_length or DEFAULT_QA_MAX_LENGTH))
    safe_stride = max(0, int(doc_stride or DEFAULT_QA_DOC_STRIDE))
    safe_max_answer_len = max(1, int(max_answer_length or DEFAULT_QA_MAX_ANSWER_LENGTH))
    safe_n_best = max(1, int(n_best or DEFAULT_QA_N_BEST))
    safe_batch = max(1, int(batch_size or DEFAULT_QA_BATCH_SIZE))
    temp = float(temperature or DEFAULT_QA_TEMPERATURE or 1.0)
    if temp <= 0:
        temp = 1.0

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=safe_max_len,
        stride=safe_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    encodings = getattr(encoded, "encodings", None)
    if not isinstance(encodings, list) or not encodings:
        raise RuntimeError("fast tokenizer 未返回 encodings，无法计算 Answerability")

    total_features = len(encodings)
    all_start_logits: List[List[float]] = []
    all_end_logits: List[List[float]] = []

    cache_key = ((model_path or "").strip() or _resolve_default_model_path(), resolved_device, resolved_fast)
    infer_lock = _get_infer_lock(cache_key)
    with infer_lock:
        for start in range(0, total_features, safe_batch):
            end = min(total_features, start + safe_batch)
            batch = {
                k: torch.tensor(encoded[k][start:end]).to(resolved_device)
                for k in ("input_ids", "attention_mask", "token_type_ids")
                if k in encoded
            }
            with torch.no_grad():
                out = model(**batch)
            batch_start = out.start_logits.detach().cpu().tolist()
            batch_end = out.end_logits.detach().cpu().tolist()
            all_start_logits.extend(batch_start)
            all_end_logits.extend(batch_end)

    best_gap = float("-inf")
    best_score_best = float("-inf")
    best_score_null = float("inf")
    min_score_null = float("inf")
    best_span = ""
    best_span_start: Optional[int] = None
    best_span_end: Optional[int] = None
    global_span_scores: List[float] = []
    best_window_index: Optional[int] = None

    for feat_idx in range(total_features):
        encoding = encodings[feat_idx]
        seq_ids = encoding.sequence_ids
        offsets = encoded["offset_mapping"][feat_idx]
        start_logits = all_start_logits[feat_idx]
        end_logits = all_end_logits[feat_idx]

        score_null = float(start_logits[0]) + float(end_logits[0])
        if score_null < min_score_null:
            min_score_null = score_null
        score_best, span_text, span_start, span_end = _find_best_span_for_feature(
            start_logits=start_logits,
            end_logits=end_logits,
            sequence_ids=list(seq_ids),
            offsets=[tuple(x) if x is not None else None for x in offsets],
            context_text=context,
            max_answer_length=safe_max_answer_len,
            n_best=safe_n_best,
        )
        if mode == "softmax_topk":
            global_span_scores.extend(
                _top_span_scores_for_feature(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    sequence_ids=list(seq_ids),
                    offsets=[tuple(x) if x is not None else None for x in offsets],
                    max_answer_length=safe_max_answer_len,
                    n_best=safe_n_best,
                    topk=max(1, int(softmax_topk or DEFAULT_QA_SOFTMAX_TOPK)),
                )
            )
        gap = score_best - score_null
        if gap > best_gap:
            best_gap = gap
            best_score_best = score_best
            best_score_null = score_null
            best_span = span_text
            best_span_start = span_start
            best_span_end = span_end
            best_window_index = feat_idx

    if best_score_best == float("-inf") or best_gap == float("-inf"):
        return AnswerabilityResult(
            answerability=0.0,
            p_no_answer=1.0,
            gap=float("-inf"),
            score_best=float("-inf"),
            score_null=float(best_score_null) if best_score_null != float("inf") else float("inf"),
            best_span_text="",
            best_span_start=None,
            best_span_end=None,
            window_count=total_features,
            best_window_index=None,
        )

    if mode == "softmax2":
        p_ans, p_null = _softmax2(best_score_best, best_score_null)
        answerability = float(p_ans)
        p_no_answer = float(p_null)
    elif mode == "softmax_topk":
        topk = max(1, int(softmax_topk or DEFAULT_QA_SOFTMAX_TOPK))
        cand = [float(x) for x in global_span_scores if isinstance(x, (int, float))]
        cand.sort(reverse=True)
        cand = cand[:topk]
        null_score = float(min_score_null if min_score_null != float("inf") else best_score_null)
        probs = _softmax_list([null_score] + cand)
        p_no_answer = float(probs[0]) if probs else 1.0
        answerability = float(1.0 - p_no_answer)
    else:
        answerability = float(_sigmoid(best_gap / temp))
        p_no_answer = float(1.0 - answerability)

    answerability = max(0.0, min(1.0, answerability))
    p_no_answer = max(0.0, min(1.0, p_no_answer))

    return AnswerabilityResult(
        answerability=answerability,
        p_no_answer=p_no_answer,
        gap=float(best_gap),
        score_best=float(best_score_best),
        score_null=float(best_score_null),
        best_span_text=str(best_span or ""),
        best_span_start=best_span_start,
        best_span_end=best_span_end,
        window_count=total_features,
        best_window_index=best_window_index,
    )


def attach_answerability(
    qa_items: List[Dict[str, Any]],
    *,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    max_length: int = DEFAULT_QA_MAX_LENGTH,
    doc_stride: int = DEFAULT_QA_DOC_STRIDE,
    max_answer_length: int = DEFAULT_QA_MAX_ANSWER_LENGTH,
    n_best: int = DEFAULT_QA_N_BEST,
    batch_size: int = DEFAULT_QA_BATCH_SIZE,
    score_mode: Optional[str] = DEFAULT_QA_SCORE_MODE,
    temperature: float = DEFAULT_QA_TEMPERATURE,
    softmax_topk: int = DEFAULT_QA_SOFTMAX_TOPK,
    use_fast_tokenizer: Optional[bool] = None,
    only_primary: bool = True,
) -> Dict[str, Any]:
    """
    Mutate qa_items in-place by adding/merging:
      item["unsupervised_evaluation"]["scores"]["answerability"]
      item["unsupervised_evaluation"]["meta"]["answerability"]

    This function does NOT assume Faithfulness is present; it only attaches Answerability.
    """
    if not qa_items:
        return {"computed": 0, "skipped": 0, "method": "extractive_qa_answerability_v1"}

    computed = 0
    eligible = 0
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        context = (
            item.get("qa_generation_unit_text")
            or item.get("source_fact_text")
            or item.get("context")
            or item.get("source_text")
            or ""
        )
        question = item.get("question") or ""
        context = str(context or "").strip()
        question = str(question or "").strip()
        if not context or not question:
            continue

        eligible += 1
        raw_mode = str(score_mode or DEFAULT_QA_SCORE_MODE).strip().lower()
        if raw_mode not in {"auto", "softmax2", "softmax_topk", "sigmoid"}:
            raw_mode = "auto"
        if raw_mode == "auto":
            effective_mode = "softmax_topk" if _is_open_ended_question(question) else "softmax2"
        else:
            effective_mode = raw_mode
        result = compute_answerability(
            question,
            context,
            model_path=model_path,
            device=device,
            max_length=max_length,
            doc_stride=doc_stride,
            max_answer_length=max_answer_length,
            n_best=n_best,
            batch_size=batch_size,
            score_mode=score_mode,
            temperature=temperature,
            softmax_topk=softmax_topk,
            use_fast_tokenizer=use_fast_tokenizer,
        )

        ue = item.get("unsupervised_evaluation")
        if not isinstance(ue, dict):
            ue = {"method": "unsupervised_suite_v1", "scores": {}, "meta": {}}
        scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
        meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}

        scores = dict(scores)
        meta = dict(meta)
        best_span_char_length = (
            max(0, int(result.best_span_end) - int(result.best_span_start))
            if result.best_span_start is not None and result.best_span_end is not None
            else len(str(result.best_span_text or ""))
        )
        scores["answerability"] = float(result.answerability)
        meta["answerability"] = {
            "method": "extractive_qa_answerability_v1",
            "p_no_answer": float(result.p_no_answer),
            "gap": float(result.gap),
            "score_best": float(result.score_best),
            "score_null": float(result.score_null),
            "best_span": str(result.best_span_text or ""),
            "best_span_start": result.best_span_start,
            "best_span_end": result.best_span_end,
            "best_span_char_length": int(best_span_char_length),
            "window_count": int(result.window_count),
            "best_window_index": result.best_window_index,
            "max_length": int(max_length),
            "doc_stride": int(doc_stride),
            "max_answer_length": int(max_answer_length),
            "n_best": int(n_best),
            "score_mode": str(score_mode or DEFAULT_QA_SCORE_MODE),
            "score_mode_effective": str(effective_mode),
            "temperature": float(temperature),
            "softmax_topk": int(softmax_topk),
        }

        ue["method"] = "unsupervised_suite_v1"
        ue["scores"] = scores
        ue["meta"] = meta
        item["unsupervised_evaluation"] = ue
        computed += 1

    return {
        "computed": computed,
        "eligible": eligible,
        "skipped": max(0, eligible - computed),
        "method": "extractive_qa_answerability_v1",
        "model_path": (model_path or "").strip() or _resolve_default_model_path(),
    }


__all__ = [
    "TRANSFORMERS_AVAILABLE",
    "AnswerabilityResult",
    "attach_answerability",
    "compute_answerability",
    "release_answerability_device_cache",
]
