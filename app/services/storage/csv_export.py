# 文件作用：将 consolidated items 导出为考试导入格式 CSV。
# 关联说明：由 pipeline_execution 调用，和 merge/consolidation 的 JSON 产物并列输出。

import csv
from typing import Any, Dict, List

def write_consolidated_csv(items: List[Dict[str, Any]], csv_path: str) -> None:
    """
    按考试导入表头导出 CSV：
    D：知识类别
    E：题目类型
    F：题干
    G：相似问向
    H：答案
    I：答案解释
    J：题目难度
    K：题目来源-相关文件
    L：题目来源-相关段落
    M：备注（模型与平均得分）
    """
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "知识类别",
            "题目类型",
            "题干",
            "相似问向",
            "答案",
            "答案解释",
            "题目难度",
            "题目来源-相关文件",
            "题目来源-相关段落",
            "备注",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            knowledge_category = item.get("knowledge_category", "") or ""
            question_type = item.get("question_type", "") or ""
            question = item.get("question", "") or ""
            # 相似问向：收集 similar_questions 列表中的 question 文本，按 " || " 拼接
            sim_list = []
            for sq in item.get("similar_questions") or []:
                qtext = sq.get("question") if isinstance(sq, dict) else sq
                if qtext:
                    sim_list.append(str(qtext))
            similar_question = " || ".join(sim_list)
            answer = item.get("answer", "") or ""
            answer_explanation = item.get("answer_explanation", "") or ""
            difficulty_level = item.get("difficulty_level", "") or ""
            original_filename = item.get("original_filename", "") or ""
            # 段落/条款：使用 source（稳定的定位/摘要标签），不要用 source_fact_text 兜底成“原子事实全文”
            paragraph = item.get("source") or "文本内容"
            avg_score = item.get("average_score")
            llm_model = item.get("llm_model") or ""
            evaluation_method = item.get("evaluation_method") or ""
            if isinstance(avg_score, (int, float)):
                avg_str = f"{float(avg_score):.4f}"
            else:
                avg_str = ""
            remark_parts = []
            if llm_model:
                remark_parts.append(f"模型: {llm_model}")
            if evaluation_method:
                remark_parts.append(f"评估: {evaluation_method}")
            evidence_ids = item.get("evidence_chunk_ids")
            if isinstance(evidence_ids, list):
                remark_parts.append(f"补充证据块: {len(evidence_ids)}")
            if avg_str:
                remark_parts.append(f"平均得分: {avg_str}")

            faithfulness = None
            answerability = None
            cov_soft = None
            cov_self = None
            cov_score = None
            unsup_f1 = None
            ue = item.get("unsupervised_evaluation")
            if isinstance(ue, dict):
                scores = ue.get("scores") or {}
                if isinstance(scores, dict):
                    faithfulness = scores.get("faithfulness")
                    answerability = scores.get("answerability")
                    cov_soft = scores.get("coverage_recall_soft")
                    cov_self = scores.get("coverage_self")
                    cov_score = scores.get("coverage_score")
                    unsup_f1 = scores.get("unsupervised_f1")
            if isinstance(faithfulness, (int, float)):
                remark_parts.append(f"忠实度: {float(faithfulness):.4f}")
            if isinstance(answerability, (int, float)):
                remark_parts.append(f"可回答性: {float(answerability):.4f}")
            if isinstance(cov_soft, (int, float)):
                remark_parts.append(f"覆盖召回(soft): {float(cov_soft):.4f}")
            if isinstance(cov_self, (int, float)):
                remark_parts.append(f"CoverageSelf: {float(cov_self):.4f}")
            if isinstance(cov_score, (int, float)):
                remark_parts.append(f"Coverage: {float(cov_score):.4f}")
            if isinstance(unsup_f1, (int, float)):
                remark_parts.append(f"无监督F1: {float(unsup_f1):.4f}")
            remark = "；".join(remark_parts)
            row: Dict[str, Any] = {
                "知识类别": knowledge_category,
                "题目类型": question_type,
                "题干": question,
                "相似问向": similar_question,
                "答案": answer,
                "答案解释": answer_explanation,
                "题目难度": difficulty_level,
                "题目来源-相关文件": original_filename,
                "题目来源-相关段落": paragraph,
                "备注": remark,
            }
            writer.writerow(row)

__all__ = ["write_consolidated_csv"]
