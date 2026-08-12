# 文件作用：定义标准类文档的问答生成提示词模板。
# 关联说明：注册到 registry.py，服务标准类文档。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="standard",
    display_name="标准类文档",
    level1_values=("标准",),
    candidate_zh="""适用标签：标准。
默认提问者：工程人员、检测人员或验收人员。
出题重点：
1. 优先围绕适用范围、术语定义、技术要求、指标阈值、检测/试验方法、验收规则、记录要求、例外条件出题。
2. 先确定工程、检测或验收场景，再问实际对象应满足什么、如何检测或达到什么指标；不要使用“第 X 条规定”或逐句复述编号条款。
3. 对表格、参数、等级、限值、频次、单位、公式，优先生成可核对的精确题。
4. 避免问“标准的意义/作用/背景是什么”；除非原文给出具体可执行结论。
5. 不要把多个技术要求揉成一个宽泛问题，除非 qa_detail_mode 要求总结且原文确实是同一规则组。""",
    answer_zh="""答案要求：
1. 保留原文中的技术名词、指标、单位、等级、条件和例外，不要改写成泛泛表述。
2. 涉及数值、频次、比例、限值、时间、范围时必须完整回答，不能只答结论。
3. 对方法类问题，答案应包含关键步骤或判定依据；对要求类问题，答案应包含适用对象和条件。
4. 若主来源块未给出具体要求，应如实说明，不得补充标准外内容。""",
    candidate_en="""Applicable label: Standard.
Default questioner: An engineer, tester, or acceptance inspector.
Question focus:
1. Prefer scope, term definitions, technical requirements, thresholds, test methods, acceptance rules, records, and exceptions.
2. Identify the engineering, testing, or acceptance scenario first, then ask what the object must satisfy, how it is tested, or which threshold applies. Do not say "Article X specifies" or restate a numbered clause sentence by sentence.
3. For tables, parameters, grades, limits, frequencies, units, and formulas, generate precise verifiable questions.
4. Avoid broad questions about the standard's meaning, role, or background unless the source states a concrete actionable conclusion.
5. Do not merge unrelated technical requirements into one broad question unless qa_detail_mode requires summary and the source is one coherent rule group.""",
    answer_en="""Answer requirements:
1. Preserve technical terms, values, units, grades, conditions, and exceptions.
2. For values, frequencies, ratios, limits, times, or ranges, answer completely rather than giving only a conclusion.
3. For method questions, include key steps or decision criteria. For requirement questions, include object and condition.
4. If the source chunk does not state a concrete requirement, say so rather than adding content beyond the standard.""",
)
