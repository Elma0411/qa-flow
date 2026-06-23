# 文件作用：定义公文通知类文档的问答生成提示词模板。
# 关联说明：注册到 registry.py，服务公文通知类文档。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="official_dispatch",
    display_name="发文通知类文档",
    level1_values=("公司发文", "政府发文"),
    candidate_zh="""适用标签：公司发文、政府发文。
出题重点：
1. 优先围绕工作任务、责任单位、时间节点、报送材料、实施步骤、工作要求、检查考核、反馈机制、成果物出题。
2. 对通知、方案、意见、报告类内容，优先问“谁在什么时间前完成什么事项/提交什么材料/落实什么要求”。
3. 对多项任务清单，按单项任务出题，不要把整段概括成“有哪些工作要求”。
4. 对会议报告或工作总结，优先问具体数据、已完成事项、下一步安排、问题整改要求。
5. 避免问“通知的主要目的/重要意义/总体要求是什么”这类宏观问题，除非原文给出明确可执行分解。""",
    answer_zh="""答案要求：
1. 答案应明确任务对象、责任主体、完成时间、提交材料或执行动作。
2. 如果问题涉及工作安排，答案不能只写“加强落实/统筹推进”，必须落到具体动作或成果。
3. 对报告类内容，答案要区分“已完成情况、存在问题、下一步计划”，不要混写。
4. 若主来源块只有宣传性、号召性或背景性语句，没有具体任务，应输出空结果。""",
    candidate_en="""Applicable labels: Company and government dispatches.
Question focus:
1. Prefer tasks, responsible units, deadlines, required submissions, implementation steps, work requirements, inspection, feedback, and deliverables.
2. For notices, plans, opinions, and reports, ask who must complete or submit what by when.
3. For task lists, ask one task at a time rather than summarizing the whole paragraph.
4. For meeting reports or summaries, prefer concrete data, completed work, next steps, and rectification requirements.
5. Avoid broad purpose or importance questions unless the source gives concrete executable breakdowns.""",
    answer_en="""Answer requirements:
1. State task object, responsible party, deadline, submitted material, or action.
2. For work arrangements, do not answer only with generic phrases such as strengthen implementation or coordinate promotion.
3. For reports, distinguish completed work, problems, and next plans.
4. If the source chunk is only promotional, motivational, or background text, return an empty result.""",
)
