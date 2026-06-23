# 文件作用：承载 baseline 评测脚本之间共享的样本读取、对齐常量和辅助函数。
# 关联说明：被 benchmark_synthetic_qa 以及各生成基线脚本复用，避免脚本间互相导入形成隐式耦合。

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

ALIGN_WEIGHTS = {"question": 0.65, "answer": 0.35}
ALIGN_THRESHOLD = 0.7


def _iter_triples(path: str, limit: int) -> Iterable[Dict[str, Any]]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict) and item.get("context") is not None:
                count += 1
                yield item


def _cosine_similarity(vec1: Any, vec2: Any) -> float:
    import numpy as np

    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    if v1.ndim != 1:
        v1 = v1.reshape(-1)
    if v2.ndim != 1:
        v2 = v2.reshape(-1)
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _pick_best_aligned_qa(
    aligner: Any,
    ref_question: str,
    ref_answer: str,
    synthetic_qas: List[Dict[str, Any]],
    align_weights: Dict[str, float],
    align_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not synthetic_qas:
        return {}, {
            "question_similarity": 0.0,
            "answer_similarity": 0.0,
            "alignment_score": 0.0,
        }

    questions = [ref_question] + [str(qa.get("question", "") or "") for qa in synthetic_qas]
    q_embs = aligner.st_model.encode(questions, convert_to_tensor=False)
    q_embs = list(q_embs)
    ref_q_emb = q_embs[0]
    cand_q_embs = q_embs[1:]

    has_ref_answer = bool(ref_answer and str(ref_answer).strip())
    if has_ref_answer:
        answers = [ref_answer] + [str(qa.get("answer", "") or "") for qa in synthetic_qas]
        a_embs = aligner.st_model.encode(answers, convert_to_tensor=False)
        a_embs = list(a_embs)
        ref_a_emb = a_embs[0]
        cand_a_embs = a_embs[1:]
    else:
        ref_a_emb = None
        cand_a_embs = []

    best_idx = 0
    best_score = -1.0
    best_q_sim = 0.0
    best_a_sim = 0.0

    for idx, _qa in enumerate(synthetic_qas):
        q_sim = _cosine_similarity(ref_q_emb, cand_q_embs[idx])
        if has_ref_answer and cand_a_embs:
            a_sim = _cosine_similarity(ref_a_emb, cand_a_embs[idx])
            align_score = (
                align_weights.get("question", 0.7) * q_sim
                + align_weights.get("answer", 0.3) * a_sim
            )
        else:
            a_sim = 0.0
            align_score = q_sim

        if align_score > best_score:
            best_score = align_score
            best_idx = idx
            best_q_sim = q_sim
            best_a_sim = a_sim

    if best_score < align_threshold:
        return {}, {
            "question_similarity": float(best_q_sim),
            "answer_similarity": float(best_a_sim),
            "alignment_score": float(best_score),
            "below_threshold": True,  # type: ignore[dict-item]
        }

    return synthetic_qas[best_idx], {
        "question_similarity": float(best_q_sim),
        "answer_similarity": float(best_a_sim),
        "alignment_score": float(best_score),
    }
