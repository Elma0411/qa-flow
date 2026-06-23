# 文件作用：承载 QA 质量评估脚本的命令行入口与结果展示。
# 关联说明：与 qa_quality_evaluator 的类/评分逻辑分离，保持脚本能力但不污染核心评估器实现。

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .qa_quality_evaluator import QAEvaluator, load_data


def visualize_results(results: pd.DataFrame) -> None:
    metrics = ["Relevance", "Coverage", "Overlap", "Accuracy", "Fluency"]
    avg_scores = results[metrics].mean()

    plt.figure(figsize=(12, 6))
    plt.subplot(121, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values = avg_scores.tolist()
    values += values[:1]
    angles += angles[:1]
    plt.fill(angles, values, alpha=0.25)
    plt.plot(angles, values, marker="o")
    plt.title("Average Scores Radar")
    plt.thetagrids([a * 180 / np.pi for a in angles[:-1]], metrics)

    plt.subplot(122)
    sns.barplot(x=metrics, y=avg_scores)
    plt.title("Average Metrics Scores")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def main(filepath: str, use_local_models: bool = True) -> pd.DataFrame:
    df = load_data(filepath)
    evaluator = QAEvaluator(use_local_models=use_local_models)
    results = []
    for _, row in df.iterrows():
        scores = {
            "Question": row["question"],
            "Relevance": evaluator.relevance(row["question"], row["answer"]),
            "Coverage": evaluator.coverage(row["question"], row["answer"]),
            "Overlap": evaluator.overlap(row["answer"], row["source_fact"]),
            "Accuracy": evaluator.accuracy(row["answer"], row["source_fact"]),
            "Fluency": evaluator.qa_fluency(row["question"], row["answer"]),
        }
        results.append(scores)

    result_df = pd.DataFrame(results)
    print("评估结果样本：")
    print(result_df.head(3))
    print("\n各维度平均得分：")
    print(result_df[["Relevance", "Coverage", "Overlap", "Accuracy", "Fluency"]].mean())
    visualize_results(result_df)
    return result_df


if __name__ == "__main__":  # pragma: no cover
    evaluation_results = main("qa_pairs.json", use_local_models=True)

