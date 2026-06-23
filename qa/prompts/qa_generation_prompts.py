# 文件作用：定义问答生成流程使用的候选问题和证据回答提示词。
# 关联说明：被 generation/qa_generation_flow 调用，并可结合 category_templates 调整风格。

import json
from typing import Any, Dict, List, Optional

from qa.prompts.category_templates import (
    build_category_answer_section,
    build_category_candidate_section,
)


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return "[]"


def _knowledge_category_fields(*, language_code: str, enabled: bool) -> str:
    if not enabled:
        if language_code == "en":
            return "- knowledge_category fields are auto-filled by the system; do not output them."
        return "- knowledge_category 字段由系统自动填充；请不要输出这些字段。"
    if language_code == "en":
        return (
            "- knowledge_category: string\n"
            "- knowledge_category_confidence: number from 0 to 1\n"
            "- knowledge_category_reason: string"
        )
    return (
        "- knowledge_category: string\n"
        "- knowledge_category_confidence: number（0~1）\n"
        "- knowledge_category_reason: string"
    )


def build_candidate_question_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    candidate_count: int,
    question_type_plan: Optional[List[str]],
    few_shot_examples: Optional[List[Dict[str, Any]]],
    knowledge_category: Optional[str] = None,
) -> str:
    plan_json = _safe_json_dumps(question_type_plan)
    examples_json = _safe_json_dumps(few_shot_examples)
    max_candidates = max(1, int(candidate_count))
    category_section = build_category_candidate_section(
        knowledge_category=knowledge_category,
        language_code=language_code,
    )

    if language_code == "en":
        return f"""# Role: Source-grounded question design expert
## Profile
- You design high-quality candidate questions for fine-tuning datasets.
- You receive exactly ONE source chunk plus its title path.
- Your job is to select only informative question angles, not to force a question from weak text.

## Language requirement
{language_instruction.strip()}

## Output goal
- Generate at most {max_candidates} candidate questions.
- It is valid to output fewer items, or an empty list, when the chunk only contains generic background, slogans, transition text, document purpose, or low-information wording.
- Do not answer the questions.
- Quality is more important than quantity. It is better to output 0 than to output weak questions.

## Workflow
1. Parse the source chunk and identify concrete entities, requirements, actions, conditions, scopes, responsibilities, deadlines, procedures, lists, exceptions, risks, thresholds, prohibitions, or measurable conclusions.
2. Use the title path only as disambiguation context. The real answer basis must still come from the source chunk itself.
3. Select only the most useful question angles for training data. Prefer points a real reader would need to understand, execute, remember, or verify.
4. Prioritize questions about:
   - who must do what
   - when or under what conditions something applies
   - what materials, records, approvals, or steps are required
   - what is prohibited, restricted, required, or exempted
   - what consequence, deadline, threshold, or handling rule is stated
5. De-prioritize or reject narrative-only background and management rhetoric.
6. Check every candidate:
   - The answer must be directly supported by the source chunk.
   - The question must have a clear answer direction.
   - The question must not duplicate another candidate.
   - The wording must be natural and specific.
   - The question must still make sense when read alone.

## Question quality constraints
1. Do not generate broad, low-value questions about "main purpose", "importance", "impact", "role", "meaning", or "what is unified" unless the chunk contains a concrete, distinctive, actionable answer.
2. Do not ask meta questions about the material itself, such as "what does the notice/document/text mention".
3. Do not use vague references such as this, that, the above, the document, the notice, these issues, it, they, or the department. Name the specific subject.
4. Do not create questions whose answer is only a generic management phrase, such as improving efficiency, reducing risk, strengthening management, unifying standards, or forming a closed loop.
5. Do not ask chapter-summary questions when the chunk actually contains multiple separate operational points. Ask one concrete point instead.
6. If the chunk mainly contains advocacy, principles, background interpretation, or high-level rationale without operational details, return fewer items or an empty list.
7. Only generate a multiple-choice candidate when the source chunk contains a stable, discriminative fact that can support one clearly correct option.
8. Prefer practical, concrete questions about who must do what, under what conditions, by which process, with which records, or with what consequences.

{category_section}

## Source anchor requirements
- source_anchor_text must be copied verbatim from the source chunk.
- source_anchor_text must contain enough concrete information to prove that the question belongs to this chunk.
- Do not use the title path alone as source_anchor_text.
- source_anchor_text should be the shortest sufficient span, not a random long excerpt.

## Question type plan
- Follow question_type_plan order when possible.
- If a planned type cannot be supported by a concrete point in the chunk, skip that item instead of inventing weak content.
- question_type_plan = {plan_json}
- Few-shot examples are style-only and must not be copied: {examples_json}

## Required item fields
- question: string
- source_anchor_text: string copied from the source chunk
- question_type: "简答题" | "单选题" | "判断题" | "计算题"
- question_type_reason: string
- difficulty_level: "简单" | "中等" | "困难"
- difficulty_score: number from 0 to 1

## Output format
Output ONLY raw JSON: {{"items":[...]}}.
"""

    return f"""# 角色：基于原文的问题设计专家
## Profile
- 你是一名用于微调数据集建设的文本分析与问题设计专家。
- 你只会收到一个主来源块及其标题路径。
- 你的任务是挑选真正有训练价值的问题角度，而不是从每段文字里硬凑问题。

## 语言要求
{language_instruction.strip()}

## 输出目标
- 最多生成 {max_candidates} 个候选问题。
- 如果主来源块只是泛化背景、口号、过渡说明、文件目的、意义阐述或低信息密度文字，可以少生成，甚至输出空列表。
- 不要生成答案。
- 质量优先于数量；宁可输出 0 条，也不要输出低质量问题。

## 工作流程
1. 通读主来源块，识别具体的主体、要求、动作、条件、范围、职责、时限、流程、清单、例外、风险、阈值、禁止项或可验证结论。
2. 标题路径只用于帮助你判断语境，不是问题答案的主要依据；真正的答案必须仍然直接来自主来源块正文。
3. 按信息密度和实用价值选择提问切入点，优先选择读者真正需要理解、执行、记忆或核对的内容。
4. 优先选择以下类型的信息出题：
   - 谁负责做什么
   - 在什么条件下适用或不适用
   - 需要哪些材料、记录、审批、步骤
   - 明确的禁止、限制、要求、例外
   - 明确的时限、后果、阈值、处理规则
5. 对只有背景铺垫、原则倡导、价值表述、管理话术的内容，优先少出题或不出题。
6. 逐条检查候选问题：
   - 答案必须能在主来源块中直接找到依据。
   - 问题必须有明确答案指向。
   - 问题之间不能重复主题或角度。
   - 表述必须自然、准确、具体。
   - 问题单独拿出来看时也必须意思完整。

## 问题质量约束
1. 不要生成“主要目的是什么”“有什么作用/影响/意义”“统一具体指什么”这类宽泛问题，除非原文给出了非常具体、独特、可执行的答案。
2. 不要生成关于材料元信息的问题，例如“通知中提到什么”“文件要求什么”“原文说明什么”。
3. 禁止指代不明，不要使用“该通知/本通知/上述/其中/这些问题/其/该部门/该资料”等模糊说法；必须写明具体对象。
4. 如果答案只是“提升效率、降低风险、加强管理、统一标准、形成闭环”等通用管理话术，不要为它单独设计问题。
5. 不要把一个章节的概括性标题直接改写成问题；如果正文里有多个具体动作，应优先问具体动作本身。
6. 如果当前块主要是背景说明、原则要求、倡议表态、长段论述，但缺少可执行细节，可以输出空列表。
7. 只有当当前块存在稳定、可区分、可验证的事实点时，才生成单选题候选；不要为了凑题型硬出单选题。
8. 优先设计关于“谁需要做什么、在什么条件下做、按什么流程做、留下什么记录、产生什么后果”的具体问题。

{category_section}

## 原文锚点要求
- source_anchor_text 必须逐字摘自主来源块。
- source_anchor_text 必须包含足够具体的信息，能够证明该问题确实归属于当前主来源块。
- 不要只把标题路径作为 source_anchor_text。
- source_anchor_text 应尽量短而足够，不要随意摘一大段无关内容。

## 题型计划
- 尽量按 question_type_plan 的顺序输出题型。
- 如果某个计划题型在当前块中找不到具体、可靠的问题点，就跳过该 item，不要硬凑低质量题。
- question_type_plan = {plan_json}
- few-shot 示例只学习风格，不得复用事实：{examples_json}

## 每条 item 必须包含
- question: string
- source_anchor_text: string（直接摘自主来源块）
- question_type: "简答题" | "单选题" | "判断题" | "计算题"
- question_type_reason: string
- difficulty_level: "简单" | "中等" | "困难"
- difficulty_score: number（0~1）

## 输出格式
只输出纯 JSON：{{"items":[...]}}。
"""


def build_evidence_answer_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    qa_detail_mode: str,
    include_knowledge_category_fields: bool = True,
    knowledge_category: Optional[str] = None,
) -> str:
    kc_fields = _knowledge_category_fields(
        language_code=language_code,
        enabled=include_knowledge_category_fields,
    )
    category_section = build_category_answer_section(
        knowledge_category=knowledge_category,
        language_code=language_code,
    )

    if language_code == "en":
        return f"""# Role: Fine-tuning QA answer generation expert
## Profile
- You generate one final QA item from a prepared QA generation unit.
- The candidate question has already been selected from the source chunk.
- Your answer must be accurate, relevant, complete enough for training data, and free of citation-style wording.

## Language requirement
{language_instruction.strip()}

## Input
- candidate_question
- source_anchor_text
- question_type
- qa_generation_unit_text with 【主来源块】, optional 【同章节上下文】, and optional 【相关补充】

## Workflow
1. Confirm whether candidate_question is specific, answerable, and valuable.
2. Apply evidence priority strictly:
   - First: 【主来源块】
   - Second: 【同章节上下文】 only when the main source has unresolved reference, omitted subject, definition, or direct local dependency
   - Third: 【相关补充】 only when a small missing fact is needed
3. If the core answer is not stated or clearly implied by 【主来源块】, reject the item.
4. Produce a direct, natural answer without saying "according to the text/reference/document".
5. Verify that source_fact_text contains all information needed to support the answer, with the main supporting snippet coming from 【主来源块】.

## Rejection rules
- If the candidate question is broad, generic, meta-level, duplicate-like, or only asks about document purpose/importance/impact/role/meaning, output {{"items":[]}}.
- If the answer mainly depends on supplemental evidence rather than 【主来源块】, output {{"items":[]}}.
- If source_fact_text cannot be copied from qa_generation_unit_text, output {{"items":[]}}.
- If the candidate uses vague references that cannot be made clear without changing the question, output {{"items":[]}}.
- If the candidate can be answered only as a generic management slogan, output {{"items":[]}}.

## Constraints
1. Keep question exactly the same as candidate_question.
2. The topic must remain centered on 【主来源块】.
3. source_fact_text must be copied from qa_generation_unit_text. It must contain a direct snippet from 【主来源块】. Add context snippets only when strictly necessary.
4. qa_detail_mode=point: source_fact_text must be one atomic, standalone fact.
5. qa_detail_mode=summary: source_fact_text may combine related snippets, but the first and most important supporting snippet must come from 【主来源块】, and every extra snippet must be necessary.
6. answer_explanation must explain why the answer is supported, not repeat vague rhetoric.
7. Do not add outside knowledge or assumptions.
8. Do not include citation-style phrases such as "according to the reference", "the document mentions", or "the text states".
9. Do not expand the question scope beyond what candidate_question asks.

{category_section}

## Required fields
- question, answer, answer_explanation, source_fact_text, source
{kc_fields}
- question_type, question_type_reason, difficulty_level, difficulty_score, options, correct_option

## Question type
- question_type must equal the provided question_type.
- For 单选题, provide exactly 4 options and one correct_option.
- For 单选题, all 4 options must be in the same category and similar wording style.
- For 单选题, exactly one option must be directly supported by the evidence. The other 3 must be plausible but contradicted or unsupported by the evidence.
- Do not use trick options such as "all of the above" or "none of the above".
- For non-choice questions, options and correct_option must be null.

## Output format
Output ONLY raw JSON: {{"items":[{{...}}]}} or {{"items":[]}}.
qa_detail_mode={qa_detail_mode}
"""

    return f"""# 角色：微调数据集问答生成专家
## Profile
- 你负责根据已经组织好的问答生成单元，生成 1 条最终问答。
- 候选问题已经来自主来源块；你的任务是判断它是否值得保留，并生成准确答案。
- 答案必须准确、相关、信息充分、适合训练数据使用，不能带“参考/依据/文中提到”等引用式表达。

## 语言要求
{language_instruction.strip()}

## 输入内容
- candidate_question
- source_anchor_text
- question_type
- qa_generation_unit_text，其中包含【主来源块】、可能存在的【同章节上下文】和【相关补充】

## 工作流程
1. 判断 candidate_question 是否具体、可回答、有训练价值。
2. 严格按以下证据优先级定位答案依据：
   - 第一优先：【主来源块】
   - 第二优先：【同章节上下文】；仅在主来源块存在定义缺失、主语省略、局部指代、前后条款直接依赖时使用
   - 第三优先：【相关补充】；仅在确实缺少一个小背景事实时使用
3. 如果答案核心不在【主来源块】中，而主要依赖补充内容，直接丢弃该题。
4. 生成直接、自然的答案，不要写“根据原文/根据通知/文中提到”。
5. 检查 source_fact_text 是否包含支撑答案所需的全部信息，且主证据必须来自【主来源块】。

## 丢弃规则
- 如果候选问题宽泛、泛化、元信息化、疑似重复，或只是询问文件目的、意义、作用、影响、重要性，输出 {{"items":[]}}。
- 如果答案主要依赖补充证据，而不是【主来源块】，输出 {{"items":[]}}。
- 如果 source_fact_text 无法从 qa_generation_unit_text 中摘取，输出 {{"items":[]}}。
- 如果候选问题存在无法在不改写问题的情况下消除的指代不明，输出 {{"items":[]}}。
- 如果该题最终只能回答成“加强管理、提高效率、降低风险、形成闭环”这类空泛管理话术，输出 {{"items":[]}}。

## 约束
1. question 必须与 candidate_question 完全一致。
2. 问题主题必须围绕【主来源块】。
3. source_fact_text 必须摘自 qa_generation_unit_text，并且必须包含来自【主来源块】的直接证据；只有严格必要时才补充上下文片段。
4. qa_detail_mode=point 时，source_fact_text 必须是单点、可独立成立的事实。
5. qa_detail_mode=summary 时，source_fact_text 可以合并相关片段，但第一条、最核心的证据必须来自【主来源块】，其余片段必须确实参与了答案成立。
6. answer_explanation 必须解释“为什么这个答案成立”，而不是重复空泛套话。
7. 禁止引入外部知识或常识补全。
8. 答案、解释和来源事实中不要出现“根据参考内容/根据通知/文中提到/原文说明”等引用式表达。
9. 不要把问题范围扩写到 candidate_question 之外。

{category_section}

## 必填字段
- question、answer、answer_explanation、source_fact_text、source
{kc_fields}
- question_type、question_type_reason、difficulty_level、difficulty_score、options、correct_option

## 题型要求
- question_type 必须等于用户消息中的 question_type。
- 单选题必须提供 4 个 options 和 1 个 correct_option。
- 单选题的 4 个 options 必须保持同一类别、同一粒度、相近表达风格。
- 单选题必须只有 1 个选项能被证据直接支持，其余 3 个要看起来合理，但能被证据排除或无法被证据支持。
- 不要使用“以上都是/以上都不是”这类取巧选项。
- 非单选题的 options 和 correct_option 必须为 null。

## 输出格式
只输出纯 JSON：{{"items":[{{...}}]}} 或 {{"items":[]}}。
qa_detail_mode={qa_detail_mode}
"""


__all__ = [
    "build_candidate_question_system_prompt",
    "build_evidence_answer_system_prompt",
]
