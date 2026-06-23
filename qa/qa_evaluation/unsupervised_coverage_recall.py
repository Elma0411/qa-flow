# 文件作用：计算无监督覆盖召回指标。
# 关联说明：与其他 unsupervised_* 文件并列，提供单项覆盖召回指标。

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from app.core.runtime_paths import DEFAULT_COVERAGE_EMBED_MODEL_NAME, resolve_model_reference
from qa.qa_evaluation.unsupervised_runtime import (
    release_cached_models_for_device,
    resolve_first_existing_model_path,
    select_torch_device,
)

try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    torch = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False


SimilarityMapping = Literal["clip0", "linear01", "sigmoid_auto_tau", "neg_cdf"]


DEFAULT_EMBED_MODEL_PATHS = (
    resolve_model_reference(
        os.environ.get("UNSUPERVISED_COVERAGE_EMBED_MODEL_PATH"),
        default_name=DEFAULT_COVERAGE_EMBED_MODEL_NAME,
    ),
)
DEFAULT_COVERAGE_DEVICE = os.environ.get("UNSUPERVISED_COVERAGE_DEVICE", "auto").strip().lower()
DEFAULT_COVERAGE_BATCH_SIZE = int(os.environ.get("UNSUPERVISED_COVERAGE_BATCH_SIZE", "32") or 32)
DEFAULT_COVERAGE_UNIT_TYPE = str(os.environ.get("UNSUPERVISED_COVERAGE_UNIT_TYPE", "clause_sentence") or "clause_sentence")
DEFAULT_COVERAGE_MIN_UNIT_CHARS = int(os.environ.get("UNSUPERVISED_COVERAGE_MIN_UNIT_CHARS", "10") or 10)
DEFAULT_COVERAGE_MAX_UNITS = int(os.environ.get("UNSUPERVISED_COVERAGE_MAX_UNITS", "256") or 256)
DEFAULT_COVERAGE_SIM_MAPPING = str(os.environ.get("UNSUPERVISED_COVERAGE_SIM_MAPPING", "neg_cdf") or "neg_cdf").strip().lower()
if DEFAULT_COVERAGE_SIM_MAPPING not in {"clip0", "linear01", "sigmoid_auto_tau", "neg_cdf"}:
    DEFAULT_COVERAGE_SIM_MAPPING = "clip0"
DEFAULT_COVERAGE_TAU = float(os.environ.get("UNSUPERVISED_COVERAGE_TAU", "0.42") or 0.42)
DEFAULT_COVERAGE_AUTO_TAU = str(os.environ.get("UNSUPERVISED_COVERAGE_AUTO_TAU", "true") or "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
DEFAULT_COVERAGE_NEG_QUANTILE = float(os.environ.get("UNSUPERVISED_COVERAGE_NEG_QUANTILE", "0.95") or 0.95)
DEFAULT_COVERAGE_NEG_SAMPLES_PER_GROUP = int(os.environ.get("UNSUPERVISED_COVERAGE_NEG_SAMPLES_PER_GROUP", "24") or 24)
DEFAULT_COVERAGE_RANDOM_SEED = int(os.environ.get("UNSUPERVISED_COVERAGE_RANDOM_SEED", "13") or 13)
DEFAULT_COVERAGE_SIGMOID_TEMPERATURE = float(os.environ.get("UNSUPERVISED_COVERAGE_SIGMOID_TEMPERATURE", "0.08") or 0.08)
DEFAULT_COVERAGE_SIGMOID_FALLBACK_CENTER = float(os.environ.get("UNSUPERVISED_COVERAGE_SIGMOID_FALLBACK_CENTER", "0.42") or 0.42)


_RE_WS = re.compile(r"[ \t\u00A0]+")
_RE_LINE_SPLIT = re.compile(r"\n+")
_RE_ANCHOR_STRIP = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_RE_STRONG_HEAD = re.compile(
    r"^\s*(第[一二三四五六七八九十百千万0-9]+[章节条款]|（[一二三四五六七八九十]+）|[一二三四五六七八九十]+、|\d+[、\.\)）]|[A-Za-z][\.\)])"
)
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
_RE_FALLBACK_SPLIT = re.compile(r"(?<=[，,；;])")


def _resolve_default_embed_model_path() -> str:
    return resolve_first_existing_model_path(DEFAULT_EMBED_MODEL_PATHS)


def _select_device(device: Optional[str]) -> str:
    return select_torch_device(device, default_device=DEFAULT_COVERAGE_DEVICE, torch_module=torch)


def _normalize_context(text: str) -> str:
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = _RE_WS.sub(" ", s)
    s = _RE_LINE_SPLIT.sub("\n", s)
    return s.strip()


def _context_id(text: str) -> str:
    norm = _normalize_context(text)
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
    return f"sha1:{h}"


def split_units(
    context: str,
    *,
    unit_type: str = DEFAULT_COVERAGE_UNIT_TYPE,
    min_chars: int = DEFAULT_COVERAGE_MIN_UNIT_CHARS,
    max_units: int = DEFAULT_COVERAGE_MAX_UNITS,
) -> List[str]:
    """
    Split context paragraph into information units (default: clause/sentence).

    This is deterministic and does not rely on pipeline chunks.
    """
    raw = _normalize_context(context)
    if not raw:
        return []

    unit_type = str(unit_type or DEFAULT_COVERAGE_UNIT_TYPE).strip().lower()
    if unit_type != "clause_sentence":
        unit_type = "clause_sentence"

    blocks: List[str] = []
    cur: List[str] = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            continue
        if _RE_STRONG_HEAD.search(s) and cur:
            blocks.append(" ".join(cur).strip())
            cur = [s]
        else:
            cur.append(s)
    if cur:
        blocks.append(" ".join(cur).strip())

    units: List[str] = []
    for blk in blocks:
        if not blk:
            continue
        parts = [p.strip() for p in _RE_SENT_SPLIT.split(blk) if str(p or "").strip()]
        if len(parts) <= 1 and len(blk) >= 120:
            parts = [p.strip() for p in _RE_FALLBACK_SPLIT.split(blk) if str(p or "").strip()] or parts
        if not parts:
            parts = [blk]
        for p in parts:
            u = str(p or "").strip()
            if not u:
                continue
            if len(u) < int(min_chars or 1):
                continue
            units.append(u)
            if len(units) >= int(max_units or 1):
                break
        if len(units) >= int(max_units or 1):
            break

    dedup: List[str] = []
    seen: set = set()
    for u in units:
        key = _RE_WS.sub(" ", u).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(u)
    return dedup


def _apply_mapping(sim: "np.ndarray", mapping: SimilarityMapping) -> "np.ndarray":
    if mapping == "linear01":
        return (sim + 1.0) / 2.0
    if mapping in {"sigmoid_auto_tau", "neg_cdf"}:
        # This mapping is applied after taking max(sim) (see attach_coverage_recall).
        # Keep base similarity mapping consistent here.
        return np.maximum(0.0, sim)
    # clip0
    return np.maximum(0.0, sim)


def _sigmoid01(x: "np.ndarray") -> "np.ndarray":
    # Numerically stable sigmoid for float32 arrays.
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -30.0, 30.0, out=x)
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x: "np.ndarray") -> "np.ndarray":
    arr = np.asarray(x, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0, out=arr)


def _fuse_pair_score(sim_base: "np.ndarray", sim_calibrated: "np.ndarray") -> "np.ndarray":
    """
    Fuse absolute similarity and calibrated confidence.

    For default `neg_cdf`, this implements:
      p = sqrt(s_raw * F_neg(s_raw))

    This avoids the interpretability issue where a middling raw similarity is
    shown as a very high final score only because it ranks high against
    negatives.
    """
    base = _clip01(sim_base)
    calibrated = _clip01(sim_calibrated)
    return np.sqrt(base * calibrated).astype(np.float32, copy=False)


def _normalize_anchor_text(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    return _RE_ANCHOR_STRIP.sub("", raw)


def _anchor_ngrams(text: str, *, n: int = 2) -> List[str]:
    norm = _normalize_anchor_text(text)
    if not norm:
        return []
    if len(norm) <= n:
        return [norm]
    return [norm[i : i + n] for i in range(len(norm) - n + 1)]


def _lexical_anchor_score(unit_text: str, answer_text: str) -> float:
    unit_norm = _normalize_anchor_text(unit_text)
    answer_norm = _normalize_anchor_text(answer_text)
    if not unit_norm or not answer_norm:
        return 0.0
    if answer_norm in unit_norm:
        return 1.0
    grams = _anchor_ngrams(answer_norm)
    if not grams:
        return 0.0
    unit_grams = set(_anchor_ngrams(unit_norm))
    if not unit_grams:
        return 0.0
    hit = sum(1 for gram in grams if gram in unit_grams)
    return max(0.0, min(1.0, float(hit) / float(len(grams))))


def _build_anchor_matrix(units: List[str], answers: List[str]) -> "np.ndarray":
    rows = len(units)
    cols = len(answers)
    out = np.zeros((rows, cols), dtype=np.float32)
    if rows <= 0 or cols <= 0:
        return out
    for i, unit in enumerate(units):
        for j, answer in enumerate(answers):
            out[i, j] = float(_lexical_anchor_score(unit, answer))
    return out


@dataclass(frozen=True)
class CoverageGroupResult:
    context_id: str
    units_total: int
    coverage_recall_soft: float
    unit_type: str
    similarity_mapping: SimilarityMapping


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _get_embed_model(model_path: Optional[str], *, device: Optional[str]) -> Tuple[Any, str]:
    if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
        raise RuntimeError("sentence-transformers/torch/numpy 未安装，无法计算 Coverage Recall")

    resolved_path = (model_path or "").strip() or _resolve_default_embed_model_path()
    resolved_device = _select_device(device)
    cache_key = (resolved_path, resolved_device)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached, resolved_path
        if not os.path.exists(resolved_path):
            raise RuntimeError(f"Coverage Recall embedding 模型路径不存在: {resolved_path}")
        model = SentenceTransformer(resolved_path, device=resolved_device)
        _MODEL_CACHE[cache_key] = model
        return model, resolved_path


def release_coverage_device_cache(device: Optional[str]) -> None:
    resolved_device = _select_device(device)
    release_cached_models_for_device(
        _MODEL_CACHE,
        {},
        _MODEL_LOCK,
        resolved_device,
        torch_module=torch,
    )


def _encode_texts(model: Any, texts: List[str], *, batch_size: int) -> "np.ndarray":
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=max(1, int(batch_size or DEFAULT_COVERAGE_BATCH_SIZE)),
        show_progress_bar=False,
    )
    arr = np.asarray(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError("embedding encode 输出形状异常")
    return arr


def _estimate_tau_negative(
    group_embeddings: List[Tuple["np.ndarray", "np.ndarray"]],
    *,
    mapping: SimilarityMapping,
    quantile: float,
    samples_per_group: int,
    seed: int,
) -> Optional[float]:
    if np is None:
        return None
    neg = _collect_negative_unit_max_sims(
        group_embeddings,
        mapping=mapping,
        samples_per_group=samples_per_group,
        seed=seed,
    )
    if neg is None or getattr(neg, "size", 0) <= 0:
        return None
    q = float(quantile)
    q = max(0.5, min(0.999, q))
    try:
        return float(np.quantile(np.asarray(neg, dtype=np.float32), q))
    except Exception:
        neg_vals = [float(x) for x in np.asarray(neg, dtype=np.float32).tolist()]
        neg_vals.sort()
        k = int(round((len(neg_vals) - 1) * q))
        k = max(0, min(len(neg_vals) - 1, k))
        return float(neg_vals[k])


def _collect_negative_unit_max_sims(
    group_embeddings: List[Tuple["np.ndarray", "np.ndarray"]],
    *,
    mapping: SimilarityMapping,
    samples_per_group: int,
    seed: int,
) -> Optional["np.ndarray"]:
    """
    Collect negative samples of unit->QA "max similarity" under mismatched contexts.

    Returns a 1D float32 array of s_neg values where:
      s_neg = max_j f_base(cos(e(unit_i), e(qa_j)))
    """
    if np is None:
        return None
    if len(group_embeddings) < 2:
        return None

    # Hard-negative choice: for each group, pick the most similar *other* context group
    # (based on mean context/unit embeddings). This makes the negative distribution
    # less trivial than random mismatches, and improves interpretability/calibration.
    valid: List[bool] = []
    reprs: List[Optional["np.ndarray"]] = []
    dim: Optional[int] = None
    for units_emb, qa_emb in group_embeddings:
        if units_emb.size == 0 or qa_emb.size == 0:
            valid.append(False)
            reprs.append(None)
            continue
        v = units_emb.mean(axis=0).astype(np.float32, copy=False)
        if v.ndim != 1 or v.size <= 0:
            valid.append(False)
            reprs.append(None)
            continue
        nrm = float(np.linalg.norm(v))
        if not (nrm > 0.0):
            valid.append(False)
            reprs.append(None)
            continue
        v = (v / nrm).astype(np.float32, copy=False)
        dim = int(v.size) if dim is None else dim
        valid.append(True)
        reprs.append(v)

    if sum(1 for x in valid if x) < 2:
        return None

    if dim is None:
        return None
    ctx_mat = np.zeros((len(group_embeddings), int(dim)), dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool)
    for i, v in enumerate(reprs):
        if v is None:
            continue
        if int(v.size) != int(dim):
            continue
        ctx_mat[i, :] = v

    rng = random.Random(int(seed))
    neg_vals: List[float] = []
    for i, (units_emb, _qa_emb) in enumerate(group_embeddings):
        if units_emb.size == 0:
            continue
        if i < 0 or i >= int(valid_mask.size):
            continue
        if not bool(valid_mask[i]):
            continue

        sims = ctx_mat @ ctx_mat[i, :]
        sims = sims.astype(np.float32, copy=False)
        sims[i] = -1.0
        sims[~valid_mask] = -1.0
        j = int(np.argmax(sims))
        if j < 0 or j >= len(group_embeddings):
            continue

        other_qa = group_embeddings[j][1]
        if other_qa.size == 0:
            continue
        m = int(units_emb.shape[0])
        take = min(max(1, int(samples_per_group or 1)), m)
        picked = [rng.randrange(m) for _ in range(take)]
        picked_units = units_emb[picked, :]
        sim = picked_units @ other_qa.T
        sim_m = _apply_mapping(sim, mapping)
        pi = sim_m.max(axis=1)
        neg_vals.extend([float(x) for x in pi.tolist()])

    if not neg_vals:
        return None
    return np.asarray(neg_vals, dtype=np.float32)


def attach_coverage_recall(
    qa_items: List[Dict[str, Any]],
    *,
    embed_model_path: Optional[str] = None,
    device: Optional[str] = None,
    embed_batch_size: int = DEFAULT_COVERAGE_BATCH_SIZE,
    unit_type: str = DEFAULT_COVERAGE_UNIT_TYPE,
    qa_text_mode: str = "qa",
    similarity_mapping: Optional[str] = DEFAULT_COVERAGE_SIM_MAPPING,
    sigmoid_temperature: float = DEFAULT_COVERAGE_SIGMOID_TEMPERATURE,
    tau: Optional[float] = None,
    auto_tau: Optional[bool] = None,
    neg_quantile: float = DEFAULT_COVERAGE_NEG_QUANTILE,
    neg_samples_per_group: int = DEFAULT_COVERAGE_NEG_SAMPLES_PER_GROUP,
    random_seed: int = DEFAULT_COVERAGE_RANDOM_SEED,
    min_unit_chars: int = DEFAULT_COVERAGE_MIN_UNIT_CHARS,
    max_units: int = DEFAULT_COVERAGE_MAX_UNITS,
    only_primary: bool = True,
) -> Dict[str, Any]:
    """
    Coverage Recall is computed per-context (group) and copied to each QA in that group.

    Mutates qa_items in-place by adding/merging:
      scores.coverage_recall_soft
      scores.coverage_self
      scores.coverage_score

    Notes:
      - 本实现只落盘 soft 版 coverage，不落盘 hard@τ。
      - 默认 `neg_cdf` 下，pair 级分数不再直接使用 `F_neg(s)`，而是：
          - `s_raw = f_base(cos(...))`
          - `s_cal = F_neg(s_raw)`
          - `p = sqrt(s_raw * s_cal)`
        这样最终分数同时保留：
          - 绝对相似度是否足够高
          - 相对负样本是否足够异常
      - 单条 `coverage_self` 使用问题条件化权重：
          - 先判断每个 unit 与该 QA 的问题/答案有多相关
          - 再对这些相关 units 的 `p` 做加权平均
      - 最终单条 `coverage_score` 使用：
          - `coverage_score = sqrt(R_group * coverage_self)`
      - 为便于人工评审解释得分，本实现会在
        `unsupervised_evaluation.meta.coverage_recall` 写入诊断信息
        （units_total、question_relevant_units、coverage_support_units、worst_units 等）。
    """
    if not qa_items:
        return {"computed_groups": 0, "computed_items": 0, "method": "embedding_unit_coverage_v2"}

    model, resolved_path = _get_embed_model(embed_model_path, device=device)
    raw_mapping = str(similarity_mapping or DEFAULT_COVERAGE_SIM_MAPPING).strip().lower()
    mapping: SimilarityMapping = "clip0"
    if raw_mapping == "linear01":
        mapping = "linear01"
    elif raw_mapping in {"sigmoid_auto_tau", "sigmoid"}:
        mapping = "sigmoid_auto_tau"
    elif raw_mapping in {"neg_cdf", "negcdf", "cdf_neg", "neg_cdf_auto"}:
        mapping = "neg_cdf"
    else:
        mapping = "clip0"
    base_mapping: SimilarityMapping = "linear01" if mapping == "linear01" else "clip0"
    use_sigmoid = mapping == "sigmoid_auto_tau"
    use_neg_cdf = mapping == "neg_cdf"

    qa_text_mode = str(qa_text_mode or "qa").strip().lower()
    if qa_text_mode not in {"qa", "answer"}:
        qa_text_mode = "qa"

    groups: Dict[str, Dict[str, Any]] = {}
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        context = item.get("qa_generation_unit_text") or item.get("source_fact_text") or item.get("context") or ""
        context = str(context or "").strip()
        if not context:
            continue
        gid = _context_id(context)
        grp = groups.get(gid)
        if grp is None:
            grp = {"context": context, "items": []}
            groups[gid] = grp
        grp["items"].append(item)

    group_ids = list(groups.keys())
    if not group_ids:
        return {"computed_groups": 0, "computed_items": 0, "method": "embedding_unit_coverage_v2"}

    group_embs: List[Tuple["np.ndarray", "np.ndarray"]] = []
    group_results: Dict[str, CoverageGroupResult] = {}
    group_unit_pair_sim_raw: Dict[str, "np.ndarray"] = {}
    group_unit_pi_raw: Dict[str, "np.ndarray"] = {}
    group_unit_question_sim_raw: Dict[str, "np.ndarray"] = {}
    group_unit_answer_sim_raw: Dict[str, "np.ndarray"] = {}
    group_unit_answer_anchor: Dict[str, "np.ndarray"] = {}
    group_meta: Dict[str, Dict[str, Any]] = {}
    group_units: Dict[str, List[str]] = {}
    group_qa_item_idxs: Dict[str, List[int]] = {}

    for gid in group_ids:
        context = groups[gid]["context"]
        items = groups[gid]["items"]
        units = split_units(
            context,
            unit_type=unit_type,
            min_chars=min_unit_chars,
            max_units=max_units,
        )
        group_units[gid] = units
        qa_texts: List[str] = []
        question_texts: List[str] = []
        answer_texts: List[str] = []
        qa_item_idxs: List[int] = []
        for idx, it in enumerate(items):
            q = str(it.get("question") or "").strip()
            a = str(it.get("answer") or "").strip()
            if not q and not a:
                continue
            question_texts.append(q or a)
            answer_texts.append(a or q)
            if qa_text_mode == "answer":
                qa_texts.append(a or q)
            else:
                qa_texts.append(f"{q} [SEP] {a}".strip())
            qa_item_idxs.append(int(idx))
        group_qa_item_idxs[gid] = qa_item_idxs

        # Pre-fill meta for explanation (even if the group is skipped).
        group_meta[gid] = {
            "method": "embedding_unit_coverage_v2",
            "context_id": gid,
            "units_total": int(len(units)),
            "unit_type": str(unit_type or DEFAULT_COVERAGE_UNIT_TYPE),
            "qa_text_mode": str(qa_text_mode),
            "qa_total": int(len(items)),
            "qa_effective": int(len(qa_texts)),
            "similarity_mapping": str(mapping),
            "base_similarity_mapping": str(base_mapping),
            "coverage_item_mode": "question_conditioned_centered_sparse_v2",
            "coverage_weight_formula": "w(i,j)=max(relevance(i,j)-mean_i relevance(i,j),0)/sum_i max(relevance(i,j)-mean_i relevance(i,j),0)",
            "coverage_pair_score_fusion": "geometric_mean_raw_calibrated_v1",
            "coverage_score_formula": "sqrt(r_group * coverage_self)",
            "worst_units": [],
            "embed_model_path": resolved_path,
            "min_unit_chars": int(min_unit_chars),
            "max_units": int(max_units),
        }

        units_emb = _encode_texts(model, units, batch_size=embed_batch_size)
        qa_emb = _encode_texts(model, qa_texts, batch_size=embed_batch_size)
        question_emb = _encode_texts(model, question_texts, batch_size=embed_batch_size)
        answer_emb = _encode_texts(model, answer_texts, batch_size=embed_batch_size)
        group_embs.append((units_emb, qa_emb))

        if units_emb.size == 0:
            group_meta[gid]["error"] = "empty_units"
            group_unit_pi_raw[gid] = np.zeros((0,), dtype=np.float32)
            continue
        if qa_emb.size == 0:
            group_meta[gid]["error"] = "empty_qa_texts"
            group_unit_pi_raw[gid] = np.zeros((0,), dtype=np.float32)
            continue

        sim = units_emb @ qa_emb.T
        sim_m = _apply_mapping(sim, base_mapping)
        question_sim = _apply_mapping(units_emb @ question_emb.T, base_mapping)
        answer_sim = _apply_mapping(units_emb @ answer_emb.T, base_mapping)
        answer_anchor = _build_anchor_matrix(units, answer_texts)
        group_unit_pair_sim_raw[gid] = sim_m.astype(np.float32, copy=False)
        best_pi_raw = sim_m.max(axis=1).astype(np.float32, copy=False)
        group_unit_pi_raw[gid] = best_pi_raw
        group_unit_question_sim_raw[gid] = question_sim.astype(np.float32, copy=False)
        group_unit_answer_sim_raw[gid] = answer_sim.astype(np.float32, copy=False)
        group_unit_answer_anchor[gid] = answer_anchor.astype(np.float32, copy=False)

    sigmoid_center: Optional[float] = None
    sigmoid_center_source: Optional[str] = None
    resolved_auto_tau = bool(auto_tau) if auto_tau is not None else DEFAULT_COVERAGE_AUTO_TAU
    resolved_neg_q = float(neg_quantile if neg_quantile is not None else DEFAULT_COVERAGE_NEG_QUANTILE)
    resolved_neg_samples = int(neg_samples_per_group if neg_samples_per_group is not None else DEFAULT_COVERAGE_NEG_SAMPLES_PER_GROUP)
    resolved_seed = int(random_seed if random_seed is not None else DEFAULT_COVERAGE_RANDOM_SEED)

    neg_sorted: Optional["np.ndarray"] = None
    if use_neg_cdf:
        try:
            neg_vals = _collect_negative_unit_max_sims(
                group_embs,
                mapping=base_mapping,
                samples_per_group=resolved_neg_samples,
                seed=resolved_seed,
            )
            if neg_vals is not None and getattr(neg_vals, "size", 0) > 0:
                neg_sorted = np.sort(np.asarray(neg_vals, dtype=np.float32))
        except Exception:
            neg_sorted = None

        valid_neg_groups = 0
        neg_samples_total_expected = 0
        for units_emb, qa_emb in group_embs:
            if getattr(units_emb, "size", 0) <= 0 or getattr(qa_emb, "size", 0) <= 0:
                continue
            valid_neg_groups += 1
            try:
                neg_samples_total_expected += min(int(units_emb.shape[0]), int(resolved_neg_samples))
            except Exception:
                pass

        for gid in group_ids:
            gm = group_meta.get(gid)
            if isinstance(gm, dict):
                gm["neg_samples_total"] = int(getattr(neg_sorted, "size", 0) or 0)
                gm["neg_samples_per_group"] = int(resolved_neg_samples)
                gm["neg_valid_groups"] = int(valid_neg_groups)
                gm["neg_samples_total_expected"] = int(neg_samples_total_expected)
                gm["neg_sample_formula"] = "sum(min(neg_samples_per_group, units_total)) over valid context groups"
                gm["random_seed"] = int(resolved_seed)
                if neg_sorted is None or int(getattr(neg_sorted, "size", 0) or 0) <= 0:
                    gm["warning"] = "neg_cdf_no_negative_samples"

    if use_sigmoid:
        try:
            temp = float(sigmoid_temperature or DEFAULT_COVERAGE_SIGMOID_TEMPERATURE)
        except Exception:
            temp = DEFAULT_COVERAGE_SIGMOID_TEMPERATURE
        if not (temp > 0.0):
            temp = DEFAULT_COVERAGE_SIGMOID_TEMPERATURE
        sigmoid_temperature = temp

        if resolved_auto_tau:
            try:
                tau_est = _estimate_tau_negative(
                    group_embs,
                    mapping=base_mapping,
                    quantile=resolved_neg_q,
                    samples_per_group=resolved_neg_samples,
                    seed=resolved_seed,
                )
                if isinstance(tau_est, (int, float)) and 0.0 <= float(tau_est) <= 1.0:
                    sigmoid_center = float(tau_est)
                    sigmoid_center_source = "neg_sampling"
            except Exception:
                sigmoid_center = None
                sigmoid_center_source = None

        if sigmoid_center is None and isinstance(tau, (int, float)) and 0.0 <= float(tau) <= 1.0:
            sigmoid_center = float(tau)
            sigmoid_center_source = "tau"

        if sigmoid_center is None:
            sigmoid_center = float(DEFAULT_COVERAGE_SIGMOID_FALLBACK_CENTER)
            sigmoid_center_source = "fallback_default"

        for gid in group_ids:
            gm = group_meta.get(gid)
            if isinstance(gm, dict):
                gm["sigmoid_temperature"] = float(sigmoid_temperature)
                gm["sigmoid_center"] = float(sigmoid_center)
                gm["sigmoid_center_source"] = str(sigmoid_center_source or "")
                gm["auto_tau"] = bool(resolved_auto_tau)
                gm["neg_quantile"] = float(resolved_neg_q)
                gm["neg_samples_per_group"] = int(resolved_neg_samples)
                gm["random_seed"] = int(resolved_seed)

    computed_groups = 0
    computed_items = 0
    soft_vals: List[float] = []

    def map_sim_to_p(sim_vals: "np.ndarray") -> "np.ndarray":
        sim_vals = np.asarray(sim_vals, dtype=np.float32)
        if use_neg_cdf and neg_sorted is not None and getattr(neg_sorted, "size", 0) > 0:
            k = np.searchsorted(neg_sorted, sim_vals, side="right")
            return (k.astype(np.float32) / float(int(neg_sorted.size))).astype(np.float32)
        if use_sigmoid and sigmoid_center is not None:
            z = (sim_vals - float(sigmoid_center)) / float(sigmoid_temperature or DEFAULT_COVERAGE_SIGMOID_TEMPERATURE)
            return _sigmoid01(z)
        return sim_vals

    for gid in group_ids:
        items = groups[gid]["items"]
        pi_raw = group_unit_pi_raw.get(gid)
        if pi_raw is None or pi_raw.size == 0:
            # Still attach meta so UI can explain why R might be 0.0.
            for it in items:
                ue = it.get("unsupervised_evaluation")
                if not isinstance(ue, dict):
                    ue = {"method": "unsupervised_suite_v1", "scores": {}, "meta": {}}
                scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
                meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}
                scores = dict(scores)
                meta = dict(meta)
                scores["coverage_recall_soft"] = float(scores.get("coverage_recall_soft") or 0.0)
                scores["coverage_self"] = float(scores.get("coverage_self") or 0.0)
                scores["coverage_score"] = float(scores.get("coverage_score") or 0.0)
                cov_meta = dict(group_meta.get(gid) or {})
                cov_meta["coverage_recall_soft"] = float(scores["coverage_recall_soft"])
                cov_meta["coverage_self"] = float(scores["coverage_self"])
                cov_meta["coverage_score"] = float(scores["coverage_score"])
                cov_meta["coverage_support_units"] = []
                cov_meta["question_relevant_units"] = []
                meta["coverage_recall"] = cov_meta
                ue["method"] = "unsupervised_suite_v1"
                ue["scores"] = scores
                ue["meta"] = meta
                it["unsupervised_evaluation"] = ue
            continue
        units_total = int(pi_raw.shape[0])
        pi_cal = map_sim_to_p(pi_raw)
        pi = _fuse_pair_score(pi_raw, pi_cal)
        cov_soft = float(pi.mean()) if units_total else 0.0
        result = CoverageGroupResult(
            context_id=gid,
            units_total=units_total,
            coverage_recall_soft=max(0.0, min(1.0, cov_soft)),
            unit_type=str(unit_type or DEFAULT_COVERAGE_UNIT_TYPE),
            similarity_mapping=mapping,
        )
        group_results[gid] = result
        computed_groups += 1
        soft_vals.append(result.coverage_recall_soft)

        if use_sigmoid:
            try:
                group_meta[gid]["units_covered"] = int((pi >= 0.5).sum())
                group_meta[gid]["units_covered_threshold"] = 0.5
            except Exception:
                pass
        if use_neg_cdf:
            try:
                group_meta[gid]["units_covered_soft"] = float(pi.sum())
                group_meta[gid]["units_covered_soft_mode"] = "sum_p"
            except Exception:
                pass

        # Attach a few least-covered units for easy human inspection.
        worst_k = 5
        try:
            worst_k = max(1, min(12, int(worst_k)))
        except Exception:
            worst_k = 5
        units = group_units.get(gid) or []
        if pi.size > 0 and units:
            idxs = np.argsort(pi)[: min(worst_k, int(pi.size))]
            worst_units: List[Dict[str, Any]] = []
            for idx in idxs.tolist():
                if not isinstance(idx, (int, float)):
                    continue
                ii = int(idx)
                if ii < 0 or ii >= len(units):
                    continue
                entry: Dict[str, Any] = {
                    "text": str(units[ii] or "").strip()[:200],
                    "p": float(pi[ii]),
                }
                if use_sigmoid or use_neg_cdf:
                    entry["sim_max"] = float(pi_raw[ii])
                    entry["p_calibrated"] = float(pi_cal[ii])
                worst_units.append(entry)
            group_meta[gid]["worst_units"] = worst_units

        # --- Per-item question-conditioned self coverage ---
        coverage_by_item_idx: Dict[int, Dict[str, Any]] = {}
        qa_item_idxs = group_qa_item_idxs.get(gid) or []
        effective_qa_count = int(len(qa_item_idxs))
        topk = 6
        if units_total > 0 and effective_qa_count > 0:
            pair_sim_raw = group_unit_pair_sim_raw.get(gid)
            question_sim_raw = group_unit_question_sim_raw.get(gid)
            answer_sim_raw = group_unit_answer_sim_raw.get(gid)
            answer_anchor = group_unit_answer_anchor.get(gid)
            if (
                pair_sim_raw is not None
                and question_sim_raw is not None
                and answer_sim_raw is not None
                and answer_anchor is not None
                and int(pair_sim_raw.shape[0]) == units_total
                and int(pair_sim_raw.shape[1]) == effective_qa_count
            ):
                pair_calibrated = map_sim_to_p(pair_sim_raw)
                pair_p = _fuse_pair_score(pair_sim_raw, pair_calibrated)
                question_score = np.asarray(question_sim_raw, dtype=np.float32)
                answer_semantic = np.asarray(answer_sim_raw, dtype=np.float32)
                answer_anchor = np.asarray(answer_anchor, dtype=np.float32)
                answer_score = np.maximum(answer_semantic, answer_anchor)
                relevance = (question_score + answer_score + pair_p) / 3.0
                relevance = np.maximum(0.0, relevance, out=relevance)
                rel_sums = relevance.sum(axis=0).astype(np.float32, copy=False)

                for qa_j, item_idx in enumerate(qa_item_idxs):
                    item_idx_int = int(item_idx)
                    if item_idx_int < 0 or item_idx_int >= len(items):
                        continue

                    rel_col = relevance[:, qa_j].astype(np.float32, copy=False)
                    rel_sum = float(rel_sums[qa_j]) if qa_j < int(rel_sums.size) else 0.0
                    rel_mean = float(rel_col.mean()) if units_total > 0 else 0.0
                    centered_rel = np.maximum(rel_col - rel_mean, 0.0).astype(np.float32, copy=False)
                    centered_mass = float(centered_rel.sum())
                    weight_mode_effective = "centered_sparse_v1"
                    if centered_mass > 0.0:
                        weights = (centered_rel / centered_mass).astype(np.float32, copy=False)
                    elif rel_sum > 0.0:
                        weights = (rel_col / rel_sum).astype(np.float32, copy=False)
                        weight_mode_effective = "relevance_normalized_fallback_v1"
                    else:
                        weights = np.full((units_total,), 1.0 / float(units_total), dtype=np.float32)
                        weight_mode_effective = "uniform_fallback_v1"

                    p_self = pair_p[:, qa_j].astype(np.float32, copy=False)
                    pair_base = pair_sim_raw[:, qa_j].astype(np.float32, copy=False)
                    pair_conf = pair_calibrated[:, qa_j].astype(np.float32, copy=False)
                    support_units = (weights * p_self).astype(np.float32, copy=False)
                    coverage_self = float(support_units.sum())
                    coverage_score = float(
                        np.sqrt(
                            max(0.0, min(1.0, float(result.coverage_recall_soft)))
                            * max(0.0, min(1.0, coverage_self))
                        )
                    )

                    relevant_units: List[Dict[str, Any]] = []
                    rel_order = np.argsort(-rel_col)[: min(topk, units_total)]
                    for uidx in rel_order.tolist():
                        if uidx < 0 or uidx >= len(units):
                            continue
                        if float(rel_col[uidx]) <= 0.0:
                            continue
                        relevant_units.append(
                            {
                                "text": str(units[uidx] or "").strip()[:200],
                                "relevance": float(rel_col[uidx]),
                                "centered_relevance": float(centered_rel[uidx]),
                                "weight": float(weights[uidx]),
                                "question_score": float(question_score[uidx, qa_j]),
                                "answer_anchor": float(answer_score[uidx, qa_j]),
                                "qa_score_base": float(pair_base[uidx]),
                                "qa_score_calibrated": float(pair_conf[uidx]),
                                "qa_score": float(p_self[uidx]),
                            }
                        )

                    coverage_support_units: List[Dict[str, Any]] = []
                    support_order = np.argsort(-support_units)[: min(topk, units_total)]
                    for uidx in support_order.tolist():
                        if uidx < 0 or uidx >= len(units):
                            continue
                        if float(support_units[uidx]) <= 0.0:
                            continue
                        coverage_support_units.append(
                            {
                                "text": str(units[uidx] or "").strip()[:200],
                                "contribution": float(support_units[uidx]),
                                "relevance": float(rel_col[uidx]),
                                "centered_relevance": float(centered_rel[uidx]),
                                "weight": float(weights[uidx]),
                                "p": float(p_self[uidx]),
                                "qa_score": float(p_self[uidx]),
                                "qa_score_base": float(pair_base[uidx]),
                                "qa_score_calibrated": float(pair_conf[uidx]),
                                "question_score": float(question_score[uidx, qa_j]),
                                "answer_anchor": float(answer_score[uidx, qa_j]),
                            }
                        )

                    coverage_by_item_idx[item_idx_int] = {
                        "coverage_self": coverage_self,
                        "coverage_score": coverage_score,
                        "coverage_weight_mode_effective": weight_mode_effective,
                        "relevance_mean": rel_mean,
                        "relevance_sum": rel_sum,
                        "centered_relevance_mass": centered_mass,
                        "coverage_support_units": coverage_support_units,
                        "question_relevant_units": relevant_units,
                    }

        for item_idx, it in enumerate(items):
            ue = it.get("unsupervised_evaluation")
            if not isinstance(ue, dict):
                ue = {"method": "unsupervised_suite_v1", "scores": {}, "meta": {}}
            scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
            meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}
            scores = dict(scores)
            meta = dict(meta)
            scores["coverage_recall_soft"] = float(result.coverage_recall_soft)
            coverage_payload = coverage_by_item_idx.get(int(item_idx)) or {}
            scores["coverage_self"] = float(coverage_payload.get("coverage_self") or 0.0)
            scores["coverage_score"] = float(coverage_payload.get("coverage_score") or 0.0)
            cov_meta = dict(group_meta.get(gid) or {})
            cov_meta["coverage_recall_soft"] = float(result.coverage_recall_soft)
            cov_meta["coverage_self"] = float(scores["coverage_self"])
            cov_meta["coverage_score"] = float(scores["coverage_score"])
            cov_meta["coverage_self_label"] = "question_conditioned_self_coverage_centered_sparse"
            cov_meta["coverage_weight_mode_effective"] = str(coverage_payload.get("coverage_weight_mode_effective") or "centered_sparse_v1")
            cov_meta["relevance_mean"] = float(coverage_payload.get("relevance_mean") or 0.0)
            cov_meta["relevance_sum"] = float(coverage_payload.get("relevance_sum") or 0.0)
            cov_meta["centered_relevance_mass"] = float(coverage_payload.get("centered_relevance_mass") or 0.0)
            cov_meta["coverage_support_units"] = list(coverage_payload.get("coverage_support_units") or [])
            cov_meta["question_relevant_units"] = list(coverage_payload.get("question_relevant_units") or [])
            meta["coverage_recall"] = cov_meta
            ue["method"] = "unsupervised_suite_v1"
            ue["scores"] = scores
            ue["meta"] = meta
            it["unsupervised_evaluation"] = ue
            computed_items += 1

    macro_soft = float(sum(soft_vals) / len(soft_vals)) if soft_vals else 0.0
    coverage_self_vals: List[float] = []
    coverage_score_vals: List[float] = []
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        ue = item.get("unsupervised_evaluation")
        scores = ue.get("scores") if isinstance(ue, dict) and isinstance(ue.get("scores"), dict) else {}
        c_self = scores.get("coverage_self")
        c_score = scores.get("coverage_score")
        if isinstance(c_self, (int, float)):
            coverage_self_vals.append(float(c_self))
        if isinstance(c_score, (int, float)):
            coverage_score_vals.append(float(c_score))

    return {
        "computed_groups": computed_groups,
        "computed_items": computed_items,
        "macro_coverage_recall_soft": macro_soft,
        "macro_coverage_self": float(sum(coverage_self_vals) / len(coverage_self_vals)) if coverage_self_vals else 0.0,
        "macro_coverage_score": float(sum(coverage_score_vals) / len(coverage_score_vals)) if coverage_score_vals else 0.0,
        "method": "embedding_unit_coverage_v2",
        "embed_model_path": resolved_path,
        "similarity_mapping": mapping,
        "qa_text_mode": qa_text_mode,
        "coverage_item_mode": "question_conditioned_centered_sparse_v2",
        "coverage_weight_formula": "w(i,j)=max(relevance(i,j)-mean_i relevance(i,j),0)/sum_i max(relevance(i,j)-mean_i relevance(i,j),0)",
        "coverage_pair_score_fusion": "geometric_mean_raw_calibrated_v1",
        "coverage_score_formula": "sqrt(r_group * coverage_self)",
    }


__all__ = [
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "CoverageGroupResult",
    "attach_coverage_recall",
    "release_coverage_device_cache",
    "split_units",
]
