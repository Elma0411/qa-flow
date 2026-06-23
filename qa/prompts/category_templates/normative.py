# 文件作用：定义制度规范类文档的问答生成提示词模板。
# 关联说明：注册到 registry.py，服务制度规范类文档。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="normative",
    display_name="法规制度类文档",
    level1_values=("法律法规", "公司制度"),
    candidate_zh="""适用标签：法律法规、公司制度。
出题重点：
1. 优先围绕适用对象、权利义务、禁止事项、审批条件、责任主体、办理流程、期限、处罚/后果、例外情形出题。
2. 对条款类文本，问题应写清楚“谁/什么事项/在什么条件下/应当或不得做什么”。
3. 对定义条款，优先问定义边界、适用范围、构成要件、排除情形。
4. 对职责条款，优先问具体主体的具体职责，不要问“相关部门有哪些职责”这类宽泛题。
5. 不要生成“本办法的目的是什么”“制度有什么意义”这类低价值问题，除非原文提供明确可考的具体规则。""",
    answer_zh="""答案要求：
1. 答案必须保持规范性表达，准确区分“应当、可以、不得、鼓励、负责、配合”等不同强度。
2. 涉及条件、对象、期限、程序、责任后果时必须一起回答，不能只答动作。
3. 遇到“前款、本条、该部门”等依赖上下文的内容时，只能在证据单元已补齐时回答；否则输出空结果。
4. 不得把法律责任、管理要求或适用范围扩大到原文没有覆盖的对象。""",
    candidate_en="""Applicable labels: Laws/regulations and internal policies.
Question focus:
1. Prefer applicable subjects, rights and duties, prohibitions, approval conditions, responsible parties, procedures, deadlines, liabilities, and exceptions.
2. For clause text, state who or what matter must or must not do what under which condition.
3. For definitions, ask about boundaries, scope, elements, and exclusions.
4. For responsibility clauses, ask about a specific party's specific duty.
5. Avoid low-value questions about purpose or meaning unless the source gives a concrete testable rule.""",
    answer_en="""Answer requirements:
1. Preserve normative force such as must, may, must not, encourage, be responsible for, and cooperate.
2. Include condition, subject, deadline, procedure, and consequence when they are part of the rule.
3. If the clause depends on unresolved references, answer only when the generation unit has supplied the dependency. Otherwise return an empty result.
4. Do not expand liabilities, management requirements, or scope beyond the source.""",
)
