# 文件作用：转换 Milvus 记录中的问答元数据和评分字段。
# 关联说明：被 store_search/admin 查询逻辑复用，用于统一解释 Milvus 元数据字段。

import json
import math
from typing import Any, Dict, List, Optional

_JSON_DUMPS_SEPARATORS = (",", ":")


def _utf8_size(text: str) -> int:
    return len(str(text or "").encode("utf-8"))


def _truncate_utf8_bytes(text: Any, max_bytes: int, *, suffix: str = "…") -> str:
    raw = str(text or "")
    budget = max(0, int(max_bytes or 0))
    if budget <= 0 or not raw:
        return ""
    raw_bytes = raw.encode("utf-8")
    if len(raw_bytes) <= budget:
        return raw
    suffix_text = str(suffix or "")
    suffix_bytes = suffix_text.encode("utf-8") if suffix_text else b""
    if len(suffix_bytes) >= budget:
        suffix_text = ""
        suffix_bytes = b""
    clipped = raw_bytes[: max(0, budget - len(suffix_bytes))]
    while clipped:
        try:
            decoded = clipped.decode("utf-8")
            return decoded.rstrip() + suffix_text
        except UnicodeDecodeError as exc:
            clipped = clipped[: exc.start]
    return suffix_text[:budget]


def _finite_float(value: Any, digits: int = 4) -> Optional[float]:
    try:
        num = float(value)
    except Exception:
        return None
    if not math.isfinite(num):
        return None
    return round(num, digits)


def _json_dumps_minified(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=_JSON_DUMPS_SEPARATORS)


def _compact_prob_map(probs: Any) -> Dict[str, float]:
    if not isinstance(probs, dict):
        return {}
    compact: Dict[str, float] = {}
    for key in ("entailment", "contradiction", "neutral"):
        val = _finite_float(probs.get(key))
        if val is not None:
            compact[key] = val
    return compact


def _compact_clauses_list(
    clauses: Any,
    *,
    limit: int,
    text_bytes: int,
    excerpt_bytes: int,
    include_probs: bool,
    include_excerpt: bool,
) -> List[Dict[str, Any]]:
    if not isinstance(clauses, list):
        return []
    compacted: List[Dict[str, Any]] = []
    for clause in clauses[: max(0, int(limit or 0))]:
        if not isinstance(clause, dict):
            continue
        row: Dict[str, Any] = {}
        text = _truncate_utf8_bytes(clause.get("text"), text_bytes)
        if text:
            row["text"] = text
        p_expected = _finite_float(clause.get("p_expected"))
        if p_expected is not None:
            row["p_expected"] = p_expected
        pred_label = str(clause.get("pred_label") or "").strip()
        if pred_label:
            row["pred_label"] = pred_label
        if include_probs:
            probs = _compact_prob_map(clause.get("probs"))
            if probs:
                row["probs"] = probs
        premise_mode = str(clause.get("premise_mode") or "").strip()
        if premise_mode:
            row["premise_mode"] = premise_mode
        if include_excerpt:
            premise_excerpt = _truncate_utf8_bytes(clause.get("premise_excerpt"), excerpt_bytes)
            if premise_excerpt:
                row["premise_excerpt"] = premise_excerpt
        if row:
            compacted.append(row)
    return compacted


def _compact_coverage_units(
    units: Any,
    *,
    limit: int,
    text_bytes: int,
    keys: List[str],
) -> List[Dict[str, Any]]:
    if not isinstance(units, list):
        return []
    compacted: List[Dict[str, Any]] = []
    for unit in units[: max(0, int(limit or 0))]:
        if not isinstance(unit, dict):
            continue
        row: Dict[str, Any] = {}
        text = _truncate_utf8_bytes(unit.get("text"), text_bytes)
        if text:
            row["text"] = text
        for key in keys:
            value = _finite_float(unit.get(key))
            if value is not None:
                row[key] = value
        if row:
            compacted.append(row)
    return compacted


def _compact_faithfulness_meta(meta: Dict[str, Any], *, level: int) -> Dict[str, Any]:
    include_clauses = level <= 1
    include_probs = level <= 2
    include_excerpt = level <= 1
    clause_limit = 3 if level <= 1 else (2 if level == 2 else 1)
    text_bytes = 240 if level <= 1 else (180 if level == 2 else 120)
    excerpt_bytes = 320 if level <= 1 else (220 if level == 2 else 0)

    compact: Dict[str, Any] = {}
    for key in ("method", "expected_label", "pred_label", "strategy", "hypothesis_mode", "premise_mode"):
        value = str(meta.get(key) or "").strip()
        if value:
            compact[key] = value
    for key in ("hypothesis_error", "hypothesis", "premise_excerpt"):
        if key == "premise_excerpt" and not include_excerpt:
            continue
        trimmed = _truncate_utf8_bytes(meta.get(key), excerpt_bytes if key == "premise_excerpt" else text_bytes)
        if trimmed:
            compact[key] = trimmed
    clauses_used = meta.get("clauses_used")
    if clauses_used is not None:
        try:
            compact["clauses_used"] = int(clauses_used)
        except Exception:
            pass
    probs = _compact_prob_map(meta.get("probs"))
    if probs:
        compact["probs"] = probs
    worst = _compact_clauses_list(
        meta.get("worst_clauses"),
        limit=clause_limit,
        text_bytes=text_bytes,
        excerpt_bytes=excerpt_bytes,
        include_probs=include_probs,
        include_excerpt=include_excerpt,
    )
    if worst:
        compact["worst_clauses"] = worst
    if include_clauses:
        clauses = _compact_clauses_list(
            meta.get("clauses"),
            limit=clause_limit,
            text_bytes=text_bytes,
            excerpt_bytes=excerpt_bytes,
            include_probs=include_probs,
            include_excerpt=include_excerpt,
        )
        if clauses:
            compact["clauses"] = clauses
    return compact


def _compact_answerability_meta(meta: Dict[str, Any], *, level: int) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("method", "score_mode", "score_mode_effective"):
        value = str(meta.get(key) or "").strip()
        if value:
            compact[key] = value
    for key in (
        "p_no_answer",
        "gap",
        "score_best",
        "score_null",
        "temperature",
        "best_span_start",
        "best_span_end",
        "best_span_char_length",
        "window_count",
        "best_window_index",
        "max_length",
        "doc_stride",
        "max_answer_length",
        "n_best",
        "softmax_topk",
    ):
        value = meta.get(key)
        if value is None:
            continue
        if key.endswith("_start") or key.endswith("_end") or key in {"best_span_char_length", "window_count", "best_window_index", "max_length", "doc_stride", "max_answer_length", "n_best", "softmax_topk"}:
            try:
                compact[key] = int(value)
                continue
            except Exception:
                pass
        num = _finite_float(value)
        if num is not None:
            compact[key] = num
    best_span = _truncate_utf8_bytes(meta.get("best_span"), 300 if level <= 1 else 180)
    if best_span:
        compact["best_span"] = best_span
    return compact


def _compact_coverage_meta(meta: Dict[str, Any], *, level: int) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "unit_type",
        "similarity_mapping",
        "base_similarity_mapping",
        "coverage_self_label",
        "coverage_weight_mode_effective",
        "coverage_item_mode",
        "sigmoid_center_source",
        "warning",
    ):
        value = str(meta.get(key) or "").strip()
        if value:
            compact[key] = value
    for key in (
        "units_total",
        "units_covered",
        "qa_total",
        "qa_effective",
        "neg_samples_total",
        "neg_samples_per_group",
        "neg_valid_groups",
        "neg_samples_total_expected",
        "random_seed",
    ):
        value = meta.get(key)
        if value is None:
            continue
        try:
            compact[key] = int(value)
        except Exception:
            pass
    for key in (
        "coverage_recall_soft",
        "coverage_self",
        "coverage_score",
        "units_covered_soft",
        "relevance_mean",
        "relevance_sum",
        "centered_relevance_mass",
        "sigmoid_center",
        "sigmoid_temperature",
        "neg_quantile",
    ):
        value = _finite_float(meta.get(key))
        if value is not None:
            compact[key] = value
    unit_limit = 4 if level <= 1 else (2 if level == 2 else 1)
    text_bytes = 180 if level <= 1 else (130 if level == 2 else 100)
    compact["coverage_support_units"] = _compact_coverage_units(
        meta.get("coverage_support_units"),
        limit=unit_limit,
        text_bytes=text_bytes,
        keys=[
            "contribution",
            "relevance",
            "centered_relevance",
            "weight",
            "p",
            "qa_score",
            "qa_score_base",
            "qa_score_calibrated",
            "question_score",
            "answer_anchor",
        ],
    )
    if level <= 1:
        compact["question_relevant_units"] = _compact_coverage_units(
            meta.get("question_relevant_units"),
            limit=unit_limit,
            text_bytes=text_bytes,
            keys=[
                "relevance",
                "centered_relevance",
                "weight",
                "question_score",
                "answer_anchor",
                "qa_score_base",
                "qa_score_calibrated",
                "qa_score",
            ],
        )
    compact["worst_units"] = _compact_coverage_units(
        meta.get("worst_units"),
        limit=unit_limit,
        text_bytes=text_bytes,
        keys=["p", "sim_max", "p_calibrated"],
    )
    compact = {k: v for k, v in compact.items() if v not in (None, "", [], {})}
    return compact


def _compact_fluency_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("method", "text_mode", "model_path", "device"):
        value = str(meta.get(key) or "").strip()
        if value:
            compact[key] = _truncate_utf8_bytes(value, 180)
    for key in (
        "ppl",
        "ppl_question",
        "ppl_answer",
        "score_question",
        "score_answer",
        "weight_question",
        "weight_answer",
        "temperature",
        "batch_size",
        "sentence_length",
    ):
        value = meta.get(key)
        if value is None:
            continue
        if key in {"batch_size", "sentence_length"}:
            try:
                compact[key] = int(value)
                continue
            except Exception:
                pass
        num = _finite_float(value)
        if num is not None:
            compact[key] = num
    normalize = meta.get("normalize")
    if isinstance(normalize, dict):
        norm_obj: Dict[str, Any] = {}
        for key in ("alpha", "beta"):
            value = _finite_float(normalize.get(key))
            if value is not None:
                norm_obj[key] = value
        formula = str(normalize.get("formula") or "").strip()
        if formula:
            norm_obj["formula"] = _truncate_utf8_bytes(formula, 120)
        if norm_obj:
            compact["normalize"] = norm_obj
    return compact


def _compact_suite_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in ("method", "precision_mode", "precision_definition", "recall_definition", "f1_definition", "context_id"):
        value = str(meta.get(key) or "").strip()
        if value:
            compact[key] = _truncate_utf8_bytes(value, 120)
    for key in ("group_size",):
        value = meta.get(key)
        if value is None:
            continue
        try:
            compact[key] = int(value)
        except Exception:
            pass
    for key in ("p_group", "r_group", "f1_group"):
        value = _finite_float(meta.get(key))
        if value is not None:
            compact[key] = value
    return compact


def _build_history_unsupervised_meta(meta: Dict[str, Any], *, ue_method: str, level: int) -> Dict[str, Any]:
    source_meta = dict(meta or {})
    if "faithfulness" not in source_meta and ue_method == "nli_faithfulness_v1":
        source_meta = {"faithfulness": {"method": ue_method, **source_meta}}

    compact: Dict[str, Any] = {
        "_storage": {
            "mode": "history_compact_v1",
            "level": int(level),
        }
    }
    faith = source_meta.get("faithfulness")
    if isinstance(faith, dict):
        compact["faithfulness"] = _compact_faithfulness_meta(faith, level=level)
    answerability = source_meta.get("answerability")
    if isinstance(answerability, dict):
        compact["answerability"] = _compact_answerability_meta(answerability, level=level)
    coverage = source_meta.get("coverage_recall")
    if isinstance(coverage, dict):
        compact["coverage_recall"] = _compact_coverage_meta(coverage, level=level)
    fluency = source_meta.get("fluency_ppl")
    if isinstance(fluency, dict):
        compact["fluency_ppl"] = _compact_fluency_meta(fluency)
    suite = source_meta.get("suite")
    if isinstance(suite, dict):
        compact["suite"] = _compact_suite_meta(suite)
    return {k: v for k, v in compact.items() if v not in (None, "", [], {})}


def _serialize_unsupervised_meta_for_milvus(meta: Dict[str, Any], *, ue_method: str, max_bytes: int) -> str:
    source_meta = meta if isinstance(meta, dict) else {}
    levels = (1, 2, 3)
    selected_payload: Dict[str, Any] = {}
    selected_json = "{}"
    selected_size = 2
    for level in levels:
        payload = _build_history_unsupervised_meta(source_meta, ue_method=ue_method, level=level)
        serialized = _json_dumps_minified(payload)
        size = _utf8_size(serialized)
        if size <= max_bytes:
            payload["_storage"]["utf8_bytes"] = size
            payload["_storage"]["byte_budget"] = int(max_bytes)
            return _json_dumps_minified(payload)
        selected_payload = payload
        selected_json = serialized
        selected_size = size

    minimal: Dict[str, Any] = {
        "_storage": {
            "mode": "history_compact_v1",
            "level": 99,
            "truncated": True,
            "byte_budget": int(max_bytes),
            "original_utf8_bytes": int(selected_size),
        }
    }
    if isinstance(selected_payload.get("faithfulness"), dict):
        minimal["faithfulness"] = {
            k: v
            for k, v in selected_payload["faithfulness"].items()
            if k in {"method", "expected_label", "pred_label", "strategy", "hypothesis_mode", "clauses_used"}
        }
    if isinstance(selected_payload.get("answerability"), dict):
        minimal["answerability"] = {
            k: v
            for k, v in selected_payload["answerability"].items()
            if k in {"method", "p_no_answer", "gap", "best_span", "best_span_start", "best_span_end", "best_span_char_length"}
        }
    if isinstance(selected_payload.get("coverage_recall"), dict):
        minimal["coverage_recall"] = {
            k: v
            for k, v in selected_payload["coverage_recall"].items()
            if k in {
                "unit_type",
                "similarity_mapping",
                "coverage_recall_soft",
                "coverage_self",
                "coverage_score",
                "units_total",
                "units_covered_soft",
                "coverage_weight_mode_effective",
            }
        }
    if isinstance(selected_payload.get("suite"), dict):
        minimal["suite"] = {
            k: v
            for k, v in selected_payload["suite"].items()
            if k in {"method", "precision_mode", "group_size", "p_group", "r_group", "f1_group"}
        }
    serialized = _json_dumps_minified(minimal)
    if _utf8_size(serialized) <= max_bytes:
        return serialized
    minimal["_storage"]["original_utf8_bytes"] = int(selected_size)
    minimal["_storage"]["fallback"] = "storage_summary_only"
    return _json_dumps_minified(minimal)

__all__ = [
    '_build_history_unsupervised_meta',
    '_compact_answerability_meta',
    '_compact_clauses_list',
    '_compact_coverage_meta',
    '_compact_coverage_units',
    '_compact_faithfulness_meta',
    '_compact_fluency_meta',
    '_compact_prob_map',
    '_compact_suite_meta',
    '_finite_float',
    '_json_dumps_minified',
    '_serialize_unsupervised_meta_for_milvus',
    '_truncate_utf8_bytes',
    '_utf8_size',
]
