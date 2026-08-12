# 文件作用：定义通用文档类别的问答生成提示词模板。
# 关联说明：注册到 registry.py，作为通用类别模板之一。

from __future__ import annotations

from .base import CategoryPromptTemplate


TEMPLATE = CategoryPromptTemplate(
    key="general",
    display_name="通用文档",
    level1_values=("其他文档", "其他", "未分类", "Uncategorized"),
    candidate_zh="""适用标签：其他或无法稳定归类的文档。
默认提问者：需要理解或执行当前主题的普通读者。
出题重点：
1. 优先选择当前块中最具体、最可验证、最不依赖外部背景的信息。
2. 能问事实就不问目的，能问条件就不问意义，能问步骤就不问宏观要求。
3. 用普通读者会使用的中性问法，不要出现“这份材料/本段/文中”。
4. 如果当前块信息密度低或缺少明确对象、动作、条件、结果，可以输出空列表。
5. 不要为了满足数量要求生成宽泛、空泛或元信息问题。""",
    answer_zh="""答案要求：
1. 答案必须直接由主来源块支撑。
2. 保持对象、条件、动作、结果完整。
3. 不确定文种时，不要套用某一类文档的外部常识或默认写法。
4. 缺少充分证据时如实说明未给出该细节，不得自行补充。""",
    candidate_en="""Applicable labels: Other or unstable categories.
Default questioner: A general reader who needs to understand or carry out the topic.
Question focus:
1. Prefer the most concrete, verifiable, and self-contained information in the chunk.
2. Ask facts over purposes, conditions over meanings, and steps over broad requirements.
3. Use neutral wording a general reader would use; do not refer to "this material", "this passage", or "the text".
4. Return an empty list if the chunk lacks clear subjects, actions, conditions, or results.
5. Do not generate broad, vague, or meta questions to satisfy quantity.""",
    answer_en="""Answer requirements:
1. The answer must be directly supported by the source chunk.
2. Preserve subject, condition, action, and result.
3. Do not apply external assumptions from a specific document type.
4. State that a detail is not given when evidence is insufficient; do not invent it.""",
)
