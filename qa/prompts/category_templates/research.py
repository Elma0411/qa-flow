# 文件作用：定义研究报告类文档的问答生成提示词模板。
# 关联说明：注册到 registry.py，服务研究报告类文档。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="research",
    display_name="学术论文类文档",
    level1_values=("学术论文",),
    candidate_zh="""适用标签：学术论文。
出题重点：
1. 优先围绕研究对象、研究问题、方法模型、数据集/样本、实验设置、评价指标、结果结论、对比实验、消融实验、局限性出题。
2. 对方法段，优先问输入、输出、关键模块、公式变量含义、算法步骤或改进点。
3. 对实验段，优先问指标变化、对比对象、实验结论、适用场景。
4. 对结论段，优先问具体发现和限定条件，不要问“论文主要研究什么”这类泛化题。
5. 如果当前块只包含引言背景或相关工作堆叙，缺少本研究的具体信息，可以少出题或不出题。""",
    answer_zh="""答案要求：
1. 答案必须区分作者提出的方法、已有方法、实验现象和结论，不要混淆来源。
2. 涉及指标、数据集、模型名称、模块名称、变量含义时必须保留原文精确表达。
3. 对结论类问题，应回答具体发现及其条件，不要扩展成论文外的一般结论。
4. 如果答案需要跨段整合但主来源块没有核心证据，应输出空结果。""",
    candidate_en="""Applicable label: Academic papers.
Question focus:
1. Prefer research object, research problem, method/model, dataset/sample, experiment setup, metrics, results, baselines, ablations, and limitations.
2. For method sections, ask about input, output, key modules, formula variables, algorithm steps, or improvements.
3. For experiment sections, ask about metric changes, baselines, conclusions, and applicable scenarios.
4. For conclusion sections, ask about specific findings and conditions instead of broad paper-topic questions.
5. If the chunk is only introduction or related-work background, output fewer items or none.""",
    answer_en="""Answer requirements:
1. Distinguish the proposed method, prior methods, experimental observations, and conclusions.
2. Preserve metrics, datasets, model names, module names, and variable meanings.
3. For conclusion questions, answer the specific finding and condition without expanding beyond the paper.
4. If the answer requires cross-section integration but the source chunk lacks core evidence, return an empty result.""",
)
