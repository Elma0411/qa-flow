"""
eval_jsonl_metrics.py

Usage (PowerShell / cmd):

1) Evaluate a benchmark output jsonl (default keys: gen_answer vs ref_answer),
   skip empty generations, and save per-item metrics (+ a final __AVERAGE__ line):

   python scripts/eval_jsonl_metrics.py `
     --input "qa/outputs/bench_sgcc_dev_triples_scored.jsonl" `
     --pred_key gen_answer --ref_key ref_answer --lang zh `
     --skip-empty `
     --save_per_item "qa/outputs/bench_sgcc_dev_triples_metrics.jsonl"

2) Override BERTScore model (HuggingFace name or local path):

   python scripts/eval_jsonl_metrics.py --input "..." --lang zh --skip-empty `
     --bertscore_model "runtime_assets/models/chinese_bert_wwm_ext_pytorch" `
     --save_per_item "qa/outputs/metrics.jsonl"

Notes:
- Metrics: EM / Token_F1 / ROUGE_L_F1 / BLEU / BERTScore(P/R/F1).
- If `bert-score` (and `torch`) is not installed, BERTScore will be 0 and a warning will be printed.
- When `--save_per_item` is set, the output file's last line is a batch summary:
  {"id":"__AVERAGE__", ...}.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.runtime_paths import DEFAULT_FLUENCY_MODEL_NAME, resolve_model_reference
from tqdm import tqdm

# ---------- Optional tokenizer (jieba) ----------
try:
    import jieba  # type: ignore
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

# ---------- BLEU ----------
try:
    import sacrebleu  # type: ignore
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

# ---------- BERTScore ----------
try:
    from bert_score import score as bertscore  # type: ignore
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False


def _default_local_bertscore_model() -> Optional[str]:
    """
    Prefer a repo-bundled Chinese BERT model when available.
    This avoids downloading models in offline environments.
    """
    local_path = resolve_model_reference(None, default_name=DEFAULT_FLUENCY_MODEL_NAME)
    return local_path if os.path.exists(local_path) else None


def _infer_bertscore_num_layers(model_path: str) -> Optional[int]:
    """
    For bert-score, unknown `model_type` requires `num_layers` to be provided.
    If model_path contains a HuggingFace-style `config.json`, infer num_hidden_layers.
    """
    try:
        cfg_path = os.path.join(model_path, "config.json")
        if not os.path.exists(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        val = cfg.get("num_hidden_layers")
        if isinstance(val, int) and val > 0:
            return val
    except Exception:
        return None
    return None


@dataclass
class Metrics:
    bert_p: float
    bert_r: float
    bert_f1: float
    rouge_l_f1: float
    f1: float
    bleu: float
    em: float


def normalize_text(s: str) -> str:
    """Normalization for EM/F1: lowercase, strip, collapse spaces, remove some punctuation."""
    if s is None:
        return ""
    s = s.strip().lower()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # remove common punctuation (keep chinese/english alnum and spaces)
    s = re.sub(r"[^\w\u4e00-\u9fff ]+", "", s)
    s = s.strip()
    return s


def tokenize(s: str) -> List[str]:
    """
    Tokenization for F1/ROUGE/BLEU.
    - If jieba exists: jieba.lcut
    - Else: fallback to character-level for CJK (robust to newlines/spaces), word-level for others.
    """
    s = (s or "").strip()
    if not s:
        return []
    if _HAS_JIEBA:
        return [t for t in jieba.lcut(s) if t.strip()]
    # Fallback:
    # - If the string contains CJK, use char-level tokens regardless of whitespace.
    #   This avoids pathological zero scores when one side contains newlines/spaces.
    if re.search(r"[\u4e00-\u9fff]", s):
        return [ch for ch in s if ch.strip()]
    if re.search(r"\s", s):
        return [t for t in s.split() if t]
    return [ch for ch in s if ch.strip()]


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_toks = tokenize(normalize_text(pred))
    gold_toks = tokenize(normalize_text(gold))
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    from collections import Counter
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def lcs_length(x: List[str], y: List[str]) -> int:
    """Classic DP LCS length; OK for typical QA answer lengths."""
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


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    lcs = lcs_length(pred_toks, gold_toks)
    prec = lcs / len(pred_toks)
    rec = lcs / len(gold_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def sentence_bleu(pred: str, gold: str) -> float:
    """
    Return BLEU score in [0,100].
    Uses sacrebleu if available; otherwise a simple fallback.
    """
    if _HAS_SACREBLEU:
        # tokenization: pass tokenized strings to reduce CJK penalty
        pred_tok = " ".join(tokenize(pred))
        gold_tok = " ".join(tokenize(gold))
        bleu = sacrebleu.sentence_bleu(pred_tok, [gold_tok]).score
        return float(bleu)

    # Fallback: very rough unigram precision BLEU-like (not standard BLEU).
    # Strongly recommended to install sacrebleu.
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)
    if not pred_toks:
        return 0.0
    gold_set = set(gold_toks)
    hit = sum(1 for t in pred_toks if t in gold_set)
    return 100.0 * hit / len(pred_toks)


def compute_bertscore(
    preds: List[str],
    refs: List[str],
    lang: str = "zh",
    model_type: Optional[str] = None,
    num_layers: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Returns per-sample (P, R, F1) lists.
    """
    if not _HAS_BERTSCORE:
        raise RuntimeError("bert-score not installed. Please `pip install bert-score torch`.")
    kwargs = {"lang": lang, "rescale_with_baseline": True}
    if model_type:
        kwargs["model_type"] = model_type
    if num_layers is not None:
        kwargs["num_layers"] = int(num_layers)

    try:
        P, R, F1 = bertscore(preds, refs, **kwargs)
    except KeyError as exc:
        # bert-score requires a known model_type unless num_layers is provided.
        # When using a local HF model path, infer num_layers from config.json and retry.
        if model_type and os.path.exists(model_type) and num_layers is None:
            inferred = _infer_bertscore_num_layers(model_type) or 12
            kwargs["num_layers"] = int(inferred)
            P, R, F1 = bertscore(preds, refs, **kwargs)
        else:
            raise exc
    return P.tolist(), R.tolist(), F1.tolist()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to jsonl file")
    ap.add_argument("--pred_key", type=str, default="gen_answer", help="Key for generated answer")
    ap.add_argument("--ref_key", type=str, default="ref_answer", help="Key for reference answer")
    ap.add_argument("--lang", type=str, default="zh", help="BERTScore lang, e.g., zh/en")
    default_bertscore_model = _default_local_bertscore_model()
    ap.add_argument(
        "--bertscore_model",
        type=str,
        default=default_bertscore_model,
        help=(
            "Optional BERTScore model_type (HF model name or local path), e.g. bert-base-chinese. "
            f"Default: {default_bertscore_model or 'None'}"
        ),
    )
    ap.add_argument(
        "--bertscore_num_layers",
        type=int,
        default=None,
        help="Optional: override BERTScore num_layers (needed for custom local model paths).",
    )
    ap.add_argument("--save_per_item", type=str, default=None, help="Optional output jsonl with per-item metrics")
    ap.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip items where pred/ref is empty (after stripping). Useful when some generations failed.",
    )
    args = ap.parse_args()

    data = load_jsonl(args.input)

    preds, refs = [], []
    ids = []
    skipped = 0
    for obj in data:
        pred = obj.get(args.pred_key, "")
        ref = obj.get(args.ref_key, "")
        pred_s = "" if pred is None else str(pred)
        ref_s = "" if ref is None else str(ref)
        if args.skip_empty and (not pred_s.strip() or not ref_s.strip()):
            skipped += 1
            continue
        preds.append(pred_s)
        refs.append(ref_s)
        ids.append(obj.get("id", None))

    # BERTScore (batch)
    if _HAS_BERTSCORE:
        if args.bertscore_model:
            print(f"[INFO] BERTScore model_type: {args.bertscore_model}")
        bert_p, bert_r, bert_f1 = compute_bertscore(
            preds,
            refs,
            lang=args.lang,
            model_type=args.bertscore_model,
            num_layers=args.bertscore_num_layers,
        )
    else:
        bert_p, bert_r, bert_f1 = [0.0]*len(preds), [0.0]*len(preds), [0.0]*len(preds)
        print("[WARN] bert-score not installed; BERTScore will be 0. Install: pip install bert-score torch")

    per_item = []
    sums = {"bert_p":0.0,"bert_r":0.0,"bert_f1":0.0,"rouge_l_f1":0.0,"f1":0.0,"bleu":0.0,"em":0.0}
    n = len(preds)

    for i in tqdm(range(n), desc="Scoring"):
        p, r, f1b = bert_p[i], bert_r[i], bert_f1[i]
        rl = rouge_l_f1(preds[i], refs[i])
        f1 = token_f1(preds[i], refs[i])
        bleu = sentence_bleu(preds[i], refs[i])
        em = exact_match(preds[i], refs[i])

        m = Metrics(
            bert_p=float(p),
            bert_r=float(r),
            bert_f1=float(f1b),
            rouge_l_f1=float(rl),
            f1=float(f1),
            bleu=float(bleu),
            em=float(em),
        )
        sums["bert_p"] += m.bert_p
        sums["bert_r"] += m.bert_r
        sums["bert_f1"] += m.bert_f1
        sums["rouge_l_f1"] += m.rouge_l_f1
        sums["f1"] += m.f1
        sums["bleu"] += m.bleu
        sums["em"] += m.em

        record = {
            "id": ids[i],
            "pred": preds[i],
            "ref": refs[i],
            "metrics": {
                "BERTScore_P": m.bert_p,
                "BERTScore_R": m.bert_r,
                "BERTScore_F1": m.bert_f1,
                "ROUGE_L_F1": m.rouge_l_f1,
                "Token_F1": m.f1,
                "BLEU": m.bleu,
                "EM": m.em,
            }
        }
        per_item.append(record)

    avg = {k: (v / n if n else 0.0) for k, v in sums.items()}

    print("\n==== Averages ====")
    if args.skip_empty:
        print(f"Evaluated items     : {n}")
        print(f"Skipped empty items : {skipped}")
    print(f"BERTScore P/R/F1 : {avg['bert_p']:.4f} / {avg['bert_r']:.4f} / {avg['bert_f1']:.4f}")
    print(f"ROUGE-L F1       : {avg['rouge_l_f1']:.4f}")
    print(f"Token F1         : {avg['f1']:.4f}")
    print(f"BLEU             : {avg['bleu']:.4f}")
    print(f"EM               : {avg['em']:.4f}")

    if args.save_per_item:
        with open(args.save_per_item, "w", encoding="utf-8") as f:
            for rec in per_item:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Append one-line batch summary as the last line
            summary = {
                "id": "__AVERAGE__",
                "count": n,
                "skipped_empty": skipped,
                "pred_key": args.pred_key,
                "ref_key": args.ref_key,
                "metrics": {
                    "BERTScore_P": avg["bert_p"],
                    "BERTScore_R": avg["bert_r"],
                    "BERTScore_F1": avg["bert_f1"],
                    "ROUGE_L_F1": avg["rouge_l_f1"],
                    "Token_F1": avg["f1"],
                    "BLEU": avg["bleu"],
                    "EM": avg["em"],
                },
            }
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"\nSaved per-item results to: {args.save_per_item}")


if __name__ == "__main__":
    main()
