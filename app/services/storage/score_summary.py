# 文件作用：汇总 consolidated items 中的无监督评价分数。
# 关联说明：被 consolidation.py 和 merge.py 共用，避免单文件和多文件结果汇总逻辑重复。

import hashlib
import re
from typing import Any, Dict, List, Optional

_RE_WS = re.compile(r"\s+")

def _context_group_id(text: str) -> str:
    norm = _RE_WS.sub(" ", str(text or "").replace("\r\n", "\n").replace("\r", "\n")).strip()
    return "sha1:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()
def _compute_unsupervised_scores_from_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute unsupervised summary scores (macro over context groups):
    - faithfulness
    - p: answerability
    - r_soft: coverage_recall_soft
    - coverage_self
    - coverage_score
    - f1: group-wise F1 macro mean (2PR/(P+R))

    Grouping is based on normalized context (`source_fact_text` -> `context` -> `source` fallback).
    Missing metrics contribute 0.0.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        context = (
            it.get("qa_generation_unit_text")
            or it.get("source_fact_text")
            or it.get("context")
            or it.get("source")
            or ""
        )
        gid = _context_group_id(str(context))
        groups.setdefault(gid, []).append(it)

    faith_group_vals: List[float] = []
    ans_group_vals: List[float] = []
    cov_group_vals: List[float] = []
    cov_self_group_vals: List[float] = []
    cov_score_group_vals: List[float] = []
    f1_group_vals: List[float] = []

    for _, gitems in groups.items():
        if not gitems:
            continue
        faith_vals: List[float] = []
        ans_vals: List[float] = []
        cov_self_vals: List[float] = []
        cov_score_vals: List[float] = []
        r_soft: Optional[float] = None
        f1: Optional[float] = None

        for it in gitems:
            ue = it.get("unsupervised_evaluation") or {}
            scores = ue.get("scores") if isinstance(ue, dict) and isinstance(ue.get("scores"), dict) else {}

            f = scores.get("faithfulness")
            if isinstance(f, (int, float)):
                faith_vals.append(float(f))

            a = scores.get("answerability")
            if isinstance(a, (int, float)):
                ans_vals.append(float(a))

            rs = scores.get("coverage_recall_soft")
            if r_soft is None and isinstance(rs, (int, float)):
                r_soft = float(rs)

            cs = scores.get("coverage_self")
            if isinstance(cs, (int, float)):
                cov_self_vals.append(float(cs))

            cscore = scores.get("coverage_score")
            if isinstance(cscore, (int, float)):
                cov_score_vals.append(float(cscore))

            uf1 = scores.get("unsupervised_f1")
            if f1 is None and isinstance(uf1, (int, float)):
                f1 = float(uf1)

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
        "coverage_self": float(sum(cov_self_group_vals) / len(cov_self_group_vals)) if cov_self_group_vals else 0.0,
        "coverage_score": float(sum(cov_score_group_vals) / len(cov_score_group_vals)) if cov_score_group_vals else 0.0,
        "f1": float(sum(f1_group_vals) / len(f1_group_vals)) if f1_group_vals else 0.0,
        "p_definition": "answerability",
        "r_definition": "coverage_recall_soft",
        "coverage_self_definition": "macro mean of item-level coverage_self within each context group",
        "coverage_score_definition": "macro mean of item-level sqrt(coverage_recall_soft * coverage_self) within each context group",
        "f1_definition": "macro mean of group-wise F1(2PR/(P+R))",
    }

__all__ = ["_compute_unsupervised_scores_from_items"]
