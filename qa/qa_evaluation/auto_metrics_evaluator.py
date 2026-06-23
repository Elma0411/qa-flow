# 文件作用：自动计算问答质量的多项传统和语义指标。
# 关联说明：与 qa_quality_evaluator/llm_quality_evaluator 并列，提供自动指标评价路径。

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json

try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_JIEBA = False

try:
    import sacrebleu  # type: ignore

    _HAS_SACREBLEU = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SACREBLEU = False

try:
    from bert_score import score as bertscore  # type: ignore

    _HAS_BERTSCORE = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_BERTSCORE = False


LOCAL_AUTO_METRIC_KEYS: List[str] = [
    "em",
    "token_f1",
    "rouge_l_f1",
    "bleu_100",
    "bleu",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "missing_reference",
]

LOCAL_AUTO_AVG_KEYS: List[str] = [
    "em",
    "token_f1",
    "rouge_l_f1",
    "bleu",
    "bertscore_f1",
]


@dataclass
class _AutoMetricsRow:
    em: float
    token_f1: float
    rouge_l_f1: float
    bleu_100: float
    bleu: float
    bertscore_p: float
    bertscore_r: float
    bertscore_f1: float
    missing_reference: float


def _normalize_text(text: Optional[str]) -> str:
    """Normalization for EM/F1: lowercase, strip, collapse spaces, remove some punctuation."""
    if text is None:
        return ""
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\u4e00-\u9fff ]+", "", normalized)
    return normalized.strip()


def _tokenize(text: Optional[str]) -> List[str]:
    """
    Tokenization for F1/ROUGE/BLEU.
    - If jieba exists: jieba.lcut
    - Else: fallback to whitespace split; if no whitespace, use char-level.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    if _HAS_JIEBA:
        return [t for t in jieba.lcut(raw) if t.strip()]
    if re.search(r"\s", raw):
        return [t for t in raw.split() if t]
    return [ch for ch in raw if ch.strip()]


def _exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(ref) else 0.0


def _token_f1(pred: str, ref: str) -> float:
    pred_toks = _tokenize(_normalize_text(pred))
    ref_toks = _tokenize(_normalize_text(ref))
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    from collections import Counter

    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(x: List[str], y: List[str]) -> int:
    n, m = len(x), len(y)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def _rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, ref_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _sentence_bleu_100(pred: str, ref: str) -> float:
    """
    Return BLEU score in [0,100].
    Uses sacrebleu if available; otherwise a simple fallback.
    """
    if _HAS_SACREBLEU:
        pred_tok = " ".join(_tokenize(pred))
        ref_tok = " ".join(_tokenize(ref))
        score = sacrebleu.sentence_bleu(pred_tok, [ref_tok]).score
        return float(score)

    pred_toks = _tokenize(pred)
    if not pred_toks:
        return 0.0
    ref_set = set(_tokenize(ref))
    hit = sum(1 for t in pred_toks if t in ref_set)
    return 100.0 * hit / len(pred_toks)


def _resolve_reference_text(qa: Dict[str, Any]) -> str:
    for key in ("qa_generation_unit_text", "source_fact_text", "source_fact", "source"):
        value = qa.get(key)
        if value:
            return str(value)
    return ""


def _default_bertscore_model_type() -> Optional[str]:
    # Prefer local model shipped with the repo/image to avoid downloads.
    local_dir = os.path.join(
        os.path.dirname(__file__), "models", "chinese_bert_wwm_ext_pytorch"
    )
    if os.path.isdir(local_dir):
        return local_dir
    return None


def _infer_num_layers_from_local_model(model_dir: str) -> Optional[int]:
    """
    bert-score expects `model_type` to be a known name, otherwise it needs `num_layers`.
    When we pass a local path as model_type, infer num_hidden_layers from config.json.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return None
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        val = cfg.get(key)
        if isinstance(val, int) and val > 0:
            return val
        if isinstance(val, str) and val.isdigit():
            iv = int(val)
            if iv > 0:
                return iv
    return None


def _compute_bertscore(
    preds: List[str],
    refs: List[str],
    *,
    lang: str,
    model_type: Optional[str],
    num_layers: Optional[int],
    batch_size: int,
    nthreads: int,
    rescale_with_baseline: bool,
    device: Optional[str],
) -> Tuple[List[float], List[float], List[float]]:
    if not _HAS_BERTSCORE:
        raise RuntimeError("bert-score not installed. Please `pip install bert-score`.")
    kwargs: Dict[str, Any] = {
        "lang": lang,
        "rescale_with_baseline": rescale_with_baseline,
        "batch_size": batch_size,
        "nthreads": nthreads,
    }
    if model_type:
        kwargs["model_type"] = model_type
    if num_layers:
        kwargs["num_layers"] = num_layers
    if device:
        kwargs["device"] = str(device)
    P, R, F1 = bertscore(preds, refs, **kwargs)
    return P.tolist(), R.tolist(), F1.tolist()


def evaluate_qa_pairs_auto_metrics(
    qa_data: List[Dict[str, Any]],
    *,
    bertscore_lang: Optional[str] = None,
    bertscore_model_type: Optional[str] = None,
    bertscore_device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate QA pairs with automatic metrics:
    EM, Token_F1, ROUGE_L_F1, BLEU(+normalized), BERTScore P/R/F1, missing_reference.

    Pred: qa["answer"]
    Ref priority: source_fact_text > source_fact > source > (missing)
    """
    started_at = time.time()
    if not qa_data:
        return {"method": "local_auto_metrics", "results": [], "total": 0}

    lang = bertscore_lang or os.environ.get("AUTO_EVAL_BERTSCORE_LANG") or "zh"
    model_type = (
        bertscore_model_type
        or os.environ.get("AUTO_EVAL_BERTSCORE_MODEL_TYPE")
        or _default_bertscore_model_type()
    )
    num_layers: Optional[int] = None
    if isinstance(model_type, str) and os.path.isdir(model_type):
        num_layers = _infer_num_layers_from_local_model(model_type)
    batch_size = max(1, int(os.environ.get("AUTO_EVAL_BERTSCORE_BATCH_SIZE", "16")))
    nthreads = max(1, int(os.environ.get("AUTO_EVAL_BERTSCORE_NTHREADS", "4")))
    rescale_env = os.environ.get("AUTO_EVAL_BERTSCORE_RESCALE")
    rescale_with_baseline = (
        str(rescale_env or "true").lower() not in ("0", "false", "no")
    )
    if rescale_env is None and isinstance(model_type, str) and os.path.isdir(model_type):
        # bert-score baseline files are keyed by model name; local paths usually don't have baselines.
        rescale_with_baseline = False

    preds: List[str] = []
    refs: List[str] = []
    idx_map: List[int] = []

    rows: List[Dict[str, Any]] = []
    for idx, qa in enumerate(qa_data):
        question = "" if qa.get("question") is None else str(qa.get("question"))
        answer = "" if qa.get("answer") is None else str(qa.get("answer"))
        ref = _resolve_reference_text(qa)
        has_ref = bool(ref.strip())

        if not has_ref:
            metrics = _AutoMetricsRow(
                em=0.0,
                token_f1=0.0,
                rouge_l_f1=0.0,
                bleu_100=0.0,
                bleu=0.0,
                bertscore_p=0.0,
                bertscore_r=0.0,
                bertscore_f1=0.0,
                missing_reference=1.0,
            )
        else:
            bleu_100 = _sentence_bleu_100(answer, ref)
            metrics = _AutoMetricsRow(
                em=_exact_match(answer, ref),
                token_f1=_token_f1(answer, ref),
                rouge_l_f1=_rouge_l_f1(answer, ref),
                bleu_100=float(bleu_100),
                bleu=float(bleu_100) / 100.0,
                bertscore_p=0.0,
                bertscore_r=0.0,
                bertscore_f1=0.0,
                missing_reference=0.0,
            )
            preds.append(answer)
            refs.append(ref)
            idx_map.append(idx)

        scores: Dict[str, float] = {
            "em": float(metrics.em),
            "token_f1": float(metrics.token_f1),
            "rouge_l_f1": float(metrics.rouge_l_f1),
            "bleu_100": float(metrics.bleu_100),
            "bleu": float(metrics.bleu),
            "bertscore_p": float(metrics.bertscore_p),
            "bertscore_r": float(metrics.bertscore_r),
            "bertscore_f1": float(metrics.bertscore_f1),
            "missing_reference": float(metrics.missing_reference),
        }

        row: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "source_fact": ref,
            "evaluation": {k: {"score": v} for k, v in scores.items()},
            "average_score": 0.0,
        }
        # keep other fields (id/task_id/filename/category/...) to preserve traceability
        for k, v in qa.items():
            if k not in row:
                row[k] = v
        rows.append(row)

    if preds:
        bert_p, bert_r, bert_f1 = _compute_bertscore(
            preds,
            refs,
            lang=lang,
            model_type=model_type,
            num_layers=num_layers,
            batch_size=batch_size,
            nthreads=nthreads,
            rescale_with_baseline=rescale_with_baseline,
            device=bertscore_device,
        )
        for j, qa_idx in enumerate(idx_map):
            ev = rows[qa_idx].get("evaluation", {})
            if isinstance(ev, dict):
                ev.setdefault("bertscore_p", {})["score"] = float(bert_p[j])
                ev.setdefault("bertscore_r", {})["score"] = float(bert_r[j])
                ev.setdefault("bertscore_f1", {})["score"] = float(bert_f1[j])

    for row in rows:
        ev = row.get("evaluation", {}) or {}
        values: List[float] = []
        for key in LOCAL_AUTO_AVG_KEYS:
            v = ev.get(key, {}).get("score") if isinstance(ev.get(key), dict) else None
            if isinstance(v, (int, float)):
                values.append(float(v))
        row["average_score"] = float(sum(values) / len(values)) if values else 0.0

    return {
        "method": "local_auto_metrics",
        "bertscore": {
            "lang": lang,
            "model_type": model_type,
            "device": bertscore_device,
            "batch_size": batch_size,
            "nthreads": nthreads,
            "rescale_with_baseline": rescale_with_baseline,
        },
        "results": rows,
        "total": len(rows),
        "duration_seconds": time.time() - started_at,
    }
