# 文件作用：汇总无监督评估指标并计算综合分。
# 关联说明：被 suite/service 调用，专注指标汇总而不执行单项模型。

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .common import _context_group_id, _safe_float


def _upgrade_faithfulness_to_suite(
    qa_items: List[Dict[str, Any]], *, only_primary: bool
) -> int:
    upgraded = 0
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        ue = item.get("unsupervised_evaluation")
        if not isinstance(ue, dict):
            continue
        method = str(ue.get("method") or "").strip()
        if method != "nli_faithfulness_v1":
            continue
        scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
        meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}
        suite_scores = dict(scores)
        suite_meta: Dict[str, Any] = {"faithfulness": {"method": method, **dict(meta)}}
        item["unsupervised_evaluation"] = {
            "method": "unsupervised_suite_v1",
            "scores": suite_scores,
            "meta": suite_meta,
        }
        upgraded += 1
    return upgraded


def _attach_suite_aggregates(
    qa_items: List[Dict[str, Any]],
    *,
    only_primary: bool,
    precision_mode: str,
    prune_item_details: bool,
) -> Dict[str, Any]:
    keep_keys = {
        "faithfulness",
        "answerability",
        "coverage_recall_soft",
        "coverage_self",
        "coverage_score",
        "unsupervised_f1",
    }
    mode = str(precision_mode or "answerability").strip().lower()
    if mode not in {"answerability", "product", "min"}:
        mode = "answerability"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        context = item.get("qa_generation_unit_text") or item.get("source_fact_text") or item.get("context") or ""
        gid = _context_group_id(str(context or ""))
        groups.setdefault(gid, []).append(item)

    computed_groups = 0
    computed_items = 0

    for items in groups.values():
        if not items:
            continue

        r_soft = 0.0
        for it in items:
            ue = it.get("unsupervised_evaluation") or {}
            scores = (
                ue.get("scores")
                if isinstance(ue, dict) and isinstance(ue.get("scores"), dict)
                else {}
            )
            rs = scores.get("coverage_recall_soft")
            if isinstance(rs, (int, float, str)):
                r_soft = _safe_float(rs, default=0.0)
                break
        r_soft = max(0.0, min(1.0, float(r_soft)))

        p_items: List[float] = []
        for it in items:
            ue = it.get("unsupervised_evaluation") or {}
            scores = (
                ue.get("scores")
                if isinstance(ue, dict) and isinstance(ue.get("scores"), dict)
                else {}
            )

            a_raw = scores.get("answerability")
            f_raw = scores.get("faithfulness")
            a_val = _safe_float(a_raw, default=0.0) if isinstance(a_raw, (int, float, str)) else 0.0
            f_val = _safe_float(f_raw, default=0.0) if isinstance(f_raw, (int, float, str)) else 0.0
            if mode == "product":
                p_item = a_val * f_val
            elif mode == "min":
                p_item = min(a_val, f_val)
            else:
                p_item = a_val
            p_item = max(0.0, min(1.0, float(p_item)))
            p_items.append(p_item)

            scores2: Dict[str, Any] = dict(scores)
            scores2["coverage_recall_soft"] = float(r_soft)
            ue2 = dict(ue) if isinstance(ue, dict) else {}
            ue2["method"] = "unsupervised_suite_v1"
            ue2["scores"] = scores2
            it["unsupervised_evaluation"] = ue2

        p_group = float(sum(p_items) / len(p_items)) if p_items else 0.0
        p_group = max(0.0, min(1.0, float(p_group)))
        denom = p_group + r_soft
        f1_group = (2.0 * p_group * r_soft / denom) if denom > 0 else 0.0
        f1_group = max(0.0, min(1.0, float(f1_group)))

        for it in items:
            ue = it.get("unsupervised_evaluation")
            if not isinstance(ue, dict):
                ue = {"method": "unsupervised_suite_v1", "scores": {}, "meta": {}}
            scores = ue.get("scores") if isinstance(ue.get("scores"), dict) else {}
            scores = dict(scores)
            scores["coverage_recall_soft"] = float(r_soft)
            scores["unsupervised_f1"] = float(f1_group)

            if prune_item_details:
                pruned_scores: Dict[str, Any] = {}
                for key in keep_keys:
                    if key in scores and isinstance(scores.get(key), (int, float)):
                        pruned_scores[key] = float(scores[key])
                for key in (
                    "faithfulness",
                    "answerability",
                    "coverage_recall_soft",
                    "coverage_self",
                    "coverage_score",
                    "unsupervised_f1",
                ):
                    if key not in pruned_scores:
                        pruned_scores[key] = 0.0
                it["unsupervised_evaluation"] = {
                    "method": "unsupervised_suite_v1",
                    "scores": pruned_scores,
                    "meta": {},
                }
                computed_items += 1
                continue

            meta = ue.get("meta") if isinstance(ue.get("meta"), dict) else {}
            meta = dict(meta)
            meta["suite"] = {
                "method": "unsupervised_suite_v1",
                "precision_mode": mode,
                "precision_definition": (
                    "answerability"
                    if mode == "answerability"
                    else (
                        "answerability*faithfulness"
                        if mode == "product"
                        else "min(answerability, faithfulness)"
                    )
                ),
                "recall_definition": "coverage_recall_soft",
                "f1_definition": "2PR/(P+R)",
                "group_size": len(items),
                "p_group": float(p_group),
                "r_group": float(r_soft),
                "f1_group": float(f1_group),
                "context_id": _context_group_id(
                    str(it.get("qa_generation_unit_text") or it.get("source_fact_text") or it.get("context") or "")
                ),
            }

            it["unsupervised_evaluation"] = {
                "method": "unsupervised_suite_v1",
                "scores": scores,
                "meta": meta,
            }
            computed_items += 1

        computed_groups += 1

    precision_definition = (
        "answerability"
        if mode == "answerability"
        else (
            "answerability*faithfulness"
            if mode == "product"
            else "min(answerability, faithfulness)"
        )
    )
    return {
        "computed_groups": computed_groups,
        "computed_items": computed_items,
        "method": "unsupervised_suite_v1",
        "precision_mode": mode,
        "precision_definition": precision_definition,
        "recall_definition": "coverage_recall_soft",
        "f1_definition": "2PR/(P+R)",
    }


def _compute_suite_four_scores(
    qa_items: List[Dict[str, Any]], *, only_primary: bool
) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in qa_items:
        if only_primary and bool(item.get("is_augmented", False)):
            continue
        context = item.get("qa_generation_unit_text") or item.get("source_fact_text") or item.get("context") or ""
        gid = _context_group_id(str(context))
        groups.setdefault(gid, []).append(item)

    faith_group_vals: List[float] = []
    ans_group_vals: List[float] = []
    cov_group_vals: List[float] = []
    cov_self_group_vals: List[float] = []
    cov_score_group_vals: List[float] = []
    f1_group_vals: List[float] = []

    for items in groups.values():
        if not items:
            continue
        faith_vals: List[float] = []
        ans_vals: List[float] = []
        cov_self_vals: List[float] = []
        cov_score_vals: List[float] = []
        r_soft: Optional[float] = None
        f1: Optional[float] = None

        for it in items:
            ue = it.get("unsupervised_evaluation") or {}
            scores = (
                ue.get("scores")
                if isinstance(ue, dict) and isinstance(ue.get("scores"), dict)
                else {}
            )

            faithfulness = scores.get("faithfulness")
            if isinstance(faithfulness, (int, float, str)):
                try:
                    faith_vals.append(float(faithfulness))
                except Exception:
                    pass

            answerability = scores.get("answerability")
            if isinstance(answerability, (int, float, str)):
                try:
                    ans_vals.append(float(answerability))
                except Exception:
                    pass

            rs = scores.get("coverage_recall_soft")
            if r_soft is None and isinstance(rs, (int, float, str)):
                r_soft = _safe_float(rs, default=0.0)

            coverage_self = scores.get("coverage_self")
            if isinstance(coverage_self, (int, float, str)):
                cov_self_vals.append(_safe_float(coverage_self, default=0.0))

            coverage_score = scores.get("coverage_score")
            if isinstance(coverage_score, (int, float, str)):
                cov_score_vals.append(_safe_float(coverage_score, default=0.0))

            uf1 = scores.get("unsupervised_f1")
            if f1 is None and isinstance(uf1, (int, float, str)):
                f1 = _safe_float(uf1, default=0.0)

        faith_group = float(sum(faith_vals) / len(faith_vals)) if faith_vals else 0.0
        ans_group = float(sum(ans_vals) / len(ans_vals)) if ans_vals else 0.0
        cov_self_group = float(sum(cov_self_vals) / len(cov_self_vals)) if cov_self_vals else 0.0
        cov_score_group = float(sum(cov_score_vals) / len(cov_score_vals)) if cov_score_vals else 0.0
        r_group = max(0.0, min(1.0, float(r_soft or 0.0)))
        if f1 is None:
            denom = ans_group + r_group
            f1 = (2.0 * ans_group * r_group / denom) if denom > 0 else 0.0
        f1 = max(0.0, min(1.0, float(f1)))

        faith_group_vals.append(max(0.0, min(1.0, float(faith_group))))
        ans_group_vals.append(max(0.0, min(1.0, float(ans_group))))
        cov_group_vals.append(r_group)
        cov_self_group_vals.append(max(0.0, min(1.0, float(cov_self_group))))
        cov_score_group_vals.append(max(0.0, min(1.0, float(cov_score_group))))
        f1_group_vals.append(f1)

    return {
        "faithfulness": float(sum(faith_group_vals) / len(faith_group_vals)) if faith_group_vals else 0.0,
        "p": float(sum(ans_group_vals) / len(ans_group_vals)) if ans_group_vals else 0.0,
        "r_soft": float(sum(cov_group_vals) / len(cov_group_vals)) if cov_group_vals else 0.0,
        "coverage_self": float(sum(cov_self_group_vals) / len(cov_self_group_vals))
        if cov_self_group_vals
        else 0.0,
        "coverage_score": float(sum(cov_score_group_vals) / len(cov_score_group_vals))
        if cov_score_group_vals
        else 0.0,
        "f1": float(sum(f1_group_vals) / len(f1_group_vals)) if f1_group_vals else 0.0,
        "p_definition": "answerability",
        "r_definition": "coverage_recall_soft",
        "coverage_self_definition": "macro mean of item-level coverage_self within each context group",
        "coverage_score_definition": "macro mean of item-level sqrt(coverage_recall_soft * coverage_self) within each context group",
        "f1_definition": "macro mean of group-wise F1(2PR/(P+R))",
    }


__all__ = [
    "_attach_suite_aggregates",
    "_compute_suite_four_scores",
    "_upgrade_faithfulness_to_suite",
]
