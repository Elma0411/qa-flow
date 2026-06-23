# 文件作用：定义知识材料类文档的问答生成提示词模板。
# 关联说明：注册到 registry.py，服务知识材料类文档。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="knowledge_material",
    display_name="知识资料类文档",
    level1_values=("图书", "业务通识"),
    candidate_zh="""适用标签：图书、业务通识。
出题重点：
1. 优先围绕概念定义、原理机制、步骤流程、分类对比、常见问题、案例经验、公式计算、操作要点出题。
2. 对培训材料，优先问学习者需要掌握的操作步骤、注意事项、判断标准、常见错误。
3. 对案例材料，优先问事件背景、关键原因、处理措施、复盘教训、适用边界。
4. 对图书/教材内容，优先问概念边界、组成要素、因果关系、对比差异。
5. 避免过度考试化的记忆题；只有当原文确实给出明确枚举、定义、数值或步骤时才生成。""",
    answer_zh="""答案要求：
1. 答案应清楚解释概念、步骤、原因或判断标准，不能只摘一个孤立短语。
2. 对流程类问题，应按原文顺序回答关键步骤。
3. 对对比类问题，应明确比较对象和差异点。
4. 对案例类问题，应区分事实经过、原因分析和处理建议。""",
    candidate_en="""Applicable labels: Books and business knowledge materials.
Question focus:
1. Prefer concepts, mechanisms, steps, classifications, comparisons, common issues, case lessons, formulas, and operating points.
2. For training materials, ask about steps, precautions, criteria, and common mistakes.
3. For cases, ask about background, causes, handling measures, lessons, and boundaries.
4. For books or textbooks, ask about concept boundaries, components, causal relations, and differences.
5. Avoid exam-like memorization unless the source gives clear enumerations, definitions, values, or steps.""",
    answer_en="""Answer requirements:
1. Explain concepts, steps, causes, or criteria clearly rather than copying an isolated phrase.
2. For process questions, answer key steps in source order.
3. For comparison questions, state compared objects and differences.
4. For case questions, distinguish facts, cause analysis, and handling suggestions.""",
)
