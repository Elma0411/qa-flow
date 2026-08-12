# 文件作用：定义制度规范类文档的问答生成提示词模板。
# 关联说明：注册到 registry.py，服务制度规范类文档。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="normative",
    display_name="法规制度类文档",
    level1_values=("法律法规", "公司制度"),
    candidate_zh="""适用标签：法律法规、公司制度。
默认提问者：办事人、权利义务相关人或制度执行人。
出题重点：
1. 优先围绕适用对象、权利义务、禁止事项、审批条件、责任主体、办理流程、期限、处罚/后果、例外情形出题。
2. 把条款转成办事咨询式问题：问“我/相关主体需要做什么、何时适用、如何办理、有什么后果”，不要复述完整条文前提。
3. 只有去掉后会改变答案的条件才留在问题中；完整适用条件、法规名称、条号和条文措辞应保留在检索规划与证据中。
4. 对定义条款，优先问定义边界、适用范围、构成要件、排除情形。
5. 对职责条款，优先问具体主体的具体职责，不要问“相关部门有哪些职责”这类宽泛题。
6. 不要生成“本办法的目的是什么”“制度有什么意义”这类低价值问题，除非原文提供明确可考的具体规则。""",
    answer_zh="""答案要求：
1. 答案必须保持规范性表达，准确区分“应当、可以、不得、鼓励、负责、配合”等不同强度。
2. 涉及条件、对象、期限、程序、责任后果时必须一起回答，不能只答动作。
3. 遇到“前款、本条、该部门”等依赖上下文的内容时，只能在证据单元已补齐时回答；缺失部分不得臆测，应明确未给出。
4. 不得把法律责任、管理要求或适用范围扩大到原文没有覆盖的对象。""",
    candidate_en="""Applicable labels: Laws/regulations and internal policies.
Default questioner: An applicant, person with related rights or duties, or policy executor.
Question focus:
1. Prefer applicable subjects, rights and duties, prohibitions, approval conditions, responsible parties, procedures, deadlines, liabilities, and exceptions.
2. Turn a clause into a practical consultation question about what the person must do, when it applies, how to proceed, or what follows. Do not restate the full legal predicate.
3. Keep only conditions that change the answer in the question. Keep complete applicability conditions, regulation names, article numbers, and clause wording in retrieval planning and evidence.
4. For definitions, ask about boundaries, scope, elements, and exclusions.
5. For responsibility clauses, ask about a specific party's specific duty.
6. Avoid low-value questions about purpose or meaning unless the source gives a concrete testable rule.""",
    answer_en="""Answer requirements:
1. Preserve normative force such as must, may, must not, encourage, be responsible for, and cooperate.
2. Include condition, subject, deadline, procedure, and consequence when they are part of the rule.
3. If the clause depends on unresolved references, answer only when the generation unit has supplied the dependency. Do not guess missing details.
4. Do not expand liabilities, management requirements, or scope beyond the source.""",
)
