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


def _normalize_qa_detail_mode(value: str) -> str:
    mode = str(value or "point").strip().lower()
    return mode if mode in {"point", "summary"} else "point"


def _candidate_detail_mode_section(*, qa_detail_mode: str, language_code: str) -> str:
    mode = _normalize_qa_detail_mode(qa_detail_mode)
    if language_code == "en":
        if mode == "summary":
            return """## Detail mode: summary
- Write one standalone question around one coherent reader need. Never join independent questions.
- Related passages may support one answer only when they describe the same scenario or rule group.
- Summary means the answer may organize related facts; it does not mean asking several questions at once.
- Use "same_section" or "cross_chunk" only when the answer genuinely needs those passages.
"""
        return """## Detail mode: point
- Write one natural, standalone question about one reader need. It may ask about a rule, amount, step, condition, responsibility, or prohibition.
- Do not slice a source sentence into its first half as a question and its last half as an answer.
- Use "source_primary" unless nearby context is genuinely needed to resolve a reference.
"""

    if mode == "summary":
        return """## 粒度模式：总结型
- 每个 item 只写一个围绕同一读者需求的完整问句，禁止拼接多个独立问题。
- 相关段落只有描述同一场景或同一规则组时，才能共同支撑一个答案。
- “总结”是答案可以组织多个相关事实，不是一次问好几个问题。
- 只有答案确实需要时才使用 "same_section" 或 "cross_chunk"。
"""
    return """## 粒度模式：单点
- 每个 item 只写一个自然、独立的用户信息需求；可以问规则、金额、步骤、条件、责任或禁止事项。
- 不要把原文一句话的前半句拆成问题、后半句拆成答案。
- 除非确实需要补足局部指代，answer_scope_hint 使用 "source_primary"。
"""


def _answer_detail_mode_section(*, qa_detail_mode: str, language_code: str) -> str:
    mode = _normalize_qa_detail_mode(qa_detail_mode)
    if language_code == "en":
        if mode == "summary":
            return """## Detail mode contract: summary
- qa_detail_mode=summary.
- Keep the single candidate question unchanged and answer only that question.
- The final answer may use short bullets or clauses when that makes the grouped facts clearer.
- Summarize the related facts from the relevant paragraph or tightly connected passages; do not introduce a second question or an unrelated topic.
- source_fact_text should contain the direct evidence segments needed for the answer and may use semicolons or new lines.
- Every key fact in the answer must be represented in source_fact_text and evidence_usage.
- answer_explanation is a complete, reader-facing clarification of why the answer applies. It is not a citation trace or a continuation of a source sentence.
"""
        return """## Detail mode contract: point
- qa_detail_mode=point.
- Keep the question unchanged and answer exactly one core fact.
- The final answer must not combine multiple independent requirements, steps, conditions, or comparisons.
- source_fact_text must be one atomic, standalone fact copied from the readable evidence material.
- Do not use semicolons, line breaks, or multiple sentences in source_fact_text.
- Retrieved context may clarify a local reference, but it must not become the main answer basis.
"""

    if mode == "summary":
        return """## 粒度模式契约：总结型
- qa_detail_mode=summary。
- 保持这一个候选问题不变，答案只能回答该问题。
- 如果更清晰，答案可以使用简短分点、步骤或并列短句。
- 归纳相关段落或紧密衔接段落中的事实，不要在答案中引入第二个问题或无关主题。
- source_fact_text 应包含回答所需的直接证据片段，可以使用分号或换行分隔。
- 答案里的每个关键事实都必须在 source_fact_text 和 evidence_usage 中有对应证据。
- answer_explanation 是面向读者、独立完整的说明：解释答案为什么适用于该场景，不是证据追踪，也不是原文句子的后半截。
"""
    return """## 粒度模式契约：单点
- qa_detail_mode=point。
- question 必须保持不变，答案只能回答一个核心事实。
- 答案不得综合多个独立要求、步骤、条件或对比关系。
- source_fact_text 必须是从可读证据材料摘取的单点、可独立成立的事实。
- source_fact_text 不得包含分号、换行或多个句子。
- 检索上下文只能帮助消除局部指代或定义缺失，不能成为答案主体。
"""


def build_candidate_question_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    candidate_count: int,
    question_type_plan: Optional[List[str]],
    few_shot_examples: Optional[List[Dict[str, Any]]],
    knowledge_category: Optional[str] = None,
    qa_detail_mode: str = "point",
) -> str:
    plan_json = _safe_json_dumps(question_type_plan)
    examples_json = _safe_json_dumps(few_shot_examples)
    max_candidates = max(1, int(candidate_count))
    category_section = build_category_candidate_section(
        knowledge_category=knowledge_category,
        language_code=language_code,
    )
    detail_mode_section = _candidate_detail_mode_section(
        qa_detail_mode=qa_detail_mode,
        language_code=language_code,
    )

    if language_code == "en":
        return f"""# Role: Source-grounded question writer

Generate at most {max_candidates} training-data questions from the supplied source material. Output fewer items, including none, when no useful question is supported. Do not answer the questions.

## Language
{language_instruction.strip()}

## Internal method; do not expose it
1. Read the material and choose one concrete information focus that matters to a plausible reader.
2. Identify one reader scenario and one information need. Write the question that reader would actually ask.
3. Silently test the wording: it must be answerable from the material, standalone, and natural without the reader having seen the source. Rewrite or skip it when it sounds clause-shaped.

## Question rules
- Write one complete question about one coherent need. Keep only the context needed to identify the scenario.
- Do not convert the first half of a source sentence into a question, carry every legal predicate into the question, or imitate source syntax.
- Do not mention sources, documents, sections, article numbers, titles, "according to", or vague references.
- Prefer a practical rule, amount, step, condition, responsibility, prohibition, deadline, exception, mechanism, or comparison over background or slogans.
- Example: write "How much additional maternity leave is available after childbirth?", not "For employees who legally give birth, how many additional leave days apply beyond statutory maternity leave?"

{detail_mode_section}

{category_section}

## Planning fields
- Planning fields are for retrieval only. They must not make `question` more formal, longer, or source-shaped.
- `retrieval_query`: concise evidence query containing the precise subject, action, and full condition.
- `must_have_terms`: 1-6 precise entity/action/condition terms for evidence matching.
- `answer_scope_hint`: `source_primary` by default; use `same_section` or `cross_chunk` only when the answer truly needs that evidence.
- Follow `question_type_plan` when supported; otherwise skip rather than inventing a weak item.
- question_type_plan: {plan_json}
- Few-shot examples are style-only and must not copy facts: {examples_json}

## Required JSON fields
`question`, `retrieval_query`, `must_have_terms`, `answer_scope_hint`, `question_type`, `question_type_reason`, `difficulty_level`, `difficulty_score`.

Output ONLY raw JSON: {{"items":[...]}}.
"""

    return f"""# 角色：基于材料的自然问题撰写者

请从提供的材料中生成最多 {max_candidates} 个训练数据问题。没有值得问的信息时可以少出题或不出题。不要生成答案。

## 语言要求
{language_instruction.strip()}

## 内部工作法，不要写进问题
1. 通读材料，先选择一个对真实读者有用的具体信息焦点。
2. 在内部确定一个读者场景和一个信息需求，再写出这个读者实际会问的一句话。
3. 静默自检：问题能由材料回答、脱离材料也意思完整，而且读者没看过材料时仍会自然地这样问；不自然就改写或跳过。

## 问题规则
- 每条只问一个完整、连贯的信息需求；只保留识别场景所需的最少上下文。
- 不要把原文一句话的前半句改成问题，也不要把完整法规前提、原文句式或检索细节搬进问题。
- 不要出现“根据/依据”、条号、文件名、章节名、“文中指出”等来源视角，也不要使用指代不明的词。
- 优先问实际会关心的规则、金额、步骤、条件、责任、禁止、期限、例外、机制或对比；跳过背景、口号和空泛管理表述。
- 例如写“生育后还能增加多少天产假？”，不要写“职工合法生育子女的，在法定产假之外可以增加多少天产假？”。

{detail_mode_section}

{category_section}

## 检索规划字段
- 检索规划字段只服务检索，不能反过来让 `question` 变得正式、冗长或像原文。
- `retrieval_query`：用于寻找证据的短查询，保留精确对象、动作和完整条件。
- `must_have_terms`：1 到 6 个用于证据匹配的精确实体、动作或条件术语。
- `answer_scope_hint`：默认 `source_primary`；只有答案确实需要时才填 `same_section` 或 `cross_chunk`。
- 在有证据支持时尽量遵循 question_type_plan；不能支持就跳过，不要硬凑。
- question_type_plan：{plan_json}
- few-shot 示例只学习风格，不得复用事实：{examples_json}

## 每条 JSON 必填字段
`question`、`retrieval_query`、`must_have_terms`、`answer_scope_hint`、`question_type`、`question_type_reason`、`difficulty_level`、`difficulty_score`。

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
    detail_mode_section = _answer_detail_mode_section(
        qa_detail_mode=qa_detail_mode,
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
- answer_scope
- question_type
- readable evidence material with 【主来源材料】, optional 【同章节上下文】, and optional 【相关补充】
- allowed `evidence_ref` labels; these are the only evidence identifiers you may return

## Workflow
1. Generate the best evidence-grounded answer for candidate_question. The candidate has already been selected; do not re-filter or reject it.
2. Apply evidence priority strictly:
   - First: 【主来源材料】
   - Second: 【同章节上下文】 only when the primary material has unresolved reference, omitted subject, definition, or direct local dependency
   - Third: 【相关补充】 only when answer_scope is "cross_chunk" and the retrieved evidence directly supports the missing fact
3. If answer_scope is "source_primary", rely on 【主来源材料】. If a requested detail is not specified, state that limitation in the answer instead of returning an empty item.
4. If answer_scope is "same_section" or "cross_chunk", you may use selected evidence, but source_fact_text must still include a direct snippet from 【主来源材料】.
5. Produce a direct, natural answer without saying "according to the text/reference/document".
6. Fill evidence_usage with `evidence_ref`, a short snippet, and usage for every material section that supports the answer. Do not invent or output chunk IDs.
7. Treat labels such as `主材料-1` and `同章节补充-1` as bookkeeping only; never copy them into question, answer, answer_explanation, or source_fact_text.

{detail_mode_section}

## Retention contract
- Always return one QA item for candidate_question. Do not output an empty items list as a quality decision.
- Express uncertainty or missing detail explicitly in the answer when needed; downstream evaluation, not this generation call, decides whether the item is retained.

## Constraints
1. Keep question exactly the same as candidate_question.
2. The topic must remain centered on 【主来源材料】.
3. source_fact_text must be copied from the readable evidence material. It must contain a direct snippet from 【主来源材料】. Add supplemental snippets only when answer_scope permits them and they are necessary.
4. qa_detail_mode=point: source_fact_text must be one atomic, standalone fact.
5. qa_detail_mode=summary: source_fact_text may combine related snippets, but the first and most important supporting snippet must come from 【主来源材料】, and every extra snippet must be necessary.
6. answer_explanation must be 1-2 complete, reader-facing sentences explaining why the answer applies to the question's scenario.
   - Clarify the relevant rule, condition, causal link, or boundary instead of repeating source_fact_text.
   - source_fact_text and evidence_usage carry provenance. Do not turn answer_explanation into a source trace or a fragment of a source sentence.
   - Start with the concrete subject or rule, not a deictic phrase such as "this benefit", "this answer", "the above", or "it".
   - Do not mention the source container, such as "main source chunk", "source text", "document", "reference", "content", or "description".
   - Bad: "This benefit applies to eligible families." Better: "Eligible rural one-child or two-daughter families receive a reduction in their personally paid medical-insurance contribution."
   - Good style: "The benefit is limited to eligible families and is calculated from the amount they personally pay, so enrollment requires checking both eligibility and the contribution amount."
7. Do not add outside knowledge or assumptions.
8. Do not invent an amount, ratio, procedure, application step, authority, or deadline that is not in the evidence. State that the specific detail is not given only when it is needed to answer the question.
9. Do not include citation-style phrases such as "according to the reference", "the document mentions", or "the text states".
10. Do not expand the question scope beyond what candidate_question asks.

{category_section}

## Required fields
- question, answer, answer_explanation, source_fact_text, source
{kc_fields}
- evidence_usage: list of objects with evidence_ref, role, snippet, usage
- question_type, question_type_reason, difficulty_level, difficulty_score, options, correct_option

## Question type
- question_type must equal the provided question_type.
- For 单选题, provide exactly 4 options and one correct_option.
- For 单选题, all 4 options must be in the same category and similar wording style.
- For 单选题, exactly one option must be directly supported by the evidence. The other 3 must be plausible but contradicted or unsupported by the evidence.
- Do not use trick options such as "all of the above" or "none of the above".
- For non-choice questions, options and correct_option must be null.

## Output format
Output ONLY raw JSON with exactly one item: {{"items":[{{...}}]}}.
qa_detail_mode={qa_detail_mode}
"""

    return f"""# 角色：微调数据集问答生成专家
## Profile
- 你负责根据已经组织好的问答生成单元，生成 1 条最终问答。
- 候选问题已经由上一步选定；不要再次筛选或丢弃，只需生成准确答案。
- 答案必须准确、相关、信息充分、适合训练数据使用，不能带“参考/依据/文中提到”等引用式表达。

## 语言要求
{language_instruction.strip()}

## 输入内容
- candidate_question
- answer_scope
- question_type
- 可读证据材料，其中包含【主来源材料】、可能存在的【同章节上下文】和【相关补充】
- 可选的 `evidence_ref` 标签；这是 evidence_usage 中唯一允许使用的证据标识

## 工作流程
1. 为 candidate_question 生成当前证据能够支持的最佳答案，不要再次判断是否保留该题。
2. 严格按以下证据优先级定位答案依据：
   - 第一优先：【主来源材料】
   - 第二优先：【同章节上下文】；仅在主来源材料存在定义缺失、主语省略、局部指代、前后条款直接依赖时使用
   - 第三优先：【相关补充】；仅当 answer_scope 为 "cross_chunk" 且检索证据直接支撑缺失事实时使用
3. 如果 answer_scope 为 "source_primary"，以【主来源材料】为准；若问题要求的某个细节没有说明，应在答案中明确说明未给出，而不是返回空列表。
4. 如果 answer_scope 为 "same_section" 或 "cross_chunk"，可以使用选中的检索证据，但 source_fact_text 仍必须包含【主来源材料】直接片段。
5. 生成直接、自然的答案，不要写“根据原文/根据通知/文中提到”。
6. 填写 evidence_usage，列出每段真正支撑答案的 `evidence_ref`、短片段和用途；不得编造或输出 chunk_id。
7. `主材料-1`、`同章节补充-1` 等标签仅用于证据追踪，不得写进 question、answer、answer_explanation 或 source_fact_text。

{detail_mode_section}

## 保留契约
- 必须为 candidate_question 返回 1 条问答，不得基于质量判断输出空 items。
- 证据存在不确定或缺失细节时，在答案中如实说明；是否保留该问答由后续评价阶段决定，不由本次生成调用决定。

## 约束
1. question 必须与 candidate_question 完全一致。
2. 问题主题必须围绕【主来源材料】。
3. source_fact_text 必须摘自可读证据材料，并且必须包含来自【主来源材料】的直接证据；只有 answer_scope 允许且严格必要时才补充检索上下文片段。
4. qa_detail_mode=point 时，source_fact_text 必须是单点、可独立成立的事实。
5. qa_detail_mode=summary 时，source_fact_text 可以合并相关片段，但第一条、最核心的证据必须来自【主来源材料】，其余片段必须确实参与了答案成立。
6. answer_explanation 必须是 1 到 2 句完整、面向读者的说明，解释答案为什么适用于问题场景。
   - 说明应补足答案中的规则、条件、因果或适用边界，而不是复述 source_fact_text。
   - 证据追踪由 source_fact_text 和 evidence_usage 完成；不要把 explanation 写成“某句原文支持某结论”或原文的半句话。
   - 第一句直接说清具体主体或规则，不要以“该优惠、该答案、此项、上述、其中、它”等指代词开头。
   - 不要提到来源容器，例如“主来源材料、原文、文本、文档、参考内容、资料、内容、描述”。
   - 不好：“该优惠面向符合条件的家庭。”更好：“农村独生子女或双女户父母参加新型农村合作医疗时，减免的是个人缴费部分。”
   - 推荐写法：“补助面向符合条件的家庭，缴费减免按个人实际缴费部分计算，因此参保时只需核对家庭资格和缴费金额。”
7. 禁止引入外部知识或常识补全。
8. 证据未写明的金额、比例、办理流程、申请手续、主管机关或期限不得自行补全；只有回答问题确实需要时，才说明该具体细节未给出。
9. 答案、解释和来源事实中不要出现“根据参考内容/根据通知/文中提到/原文说明”等引用式表达。
10. 不要把问题范围扩写到 candidate_question 之外。

{category_section}

## 必填字段
- question、answer、answer_explanation、source_fact_text、source
{kc_fields}
- evidence_usage: 对象列表，每个对象包含 evidence_ref、role、snippet、usage
- question_type、question_type_reason、difficulty_level、difficulty_score、options、correct_option

## 题型要求
- question_type 必须等于用户消息中的 question_type。
- 单选题必须提供 4 个 options 和 1 个 correct_option。
- 单选题的 4 个 options 必须保持同一类别、同一粒度、相近表达风格。
- 单选题必须只有 1 个选项能被证据直接支持，其余 3 个要看起来合理，但能被证据排除或无法被证据支持。
- 不要使用“以上都是/以上都不是”这类取巧选项。
- 非单选题的 options 和 correct_option 必须为 null。

## 输出格式
只输出包含 1 个 item 的纯 JSON：{{"items":[{{...}}]}}。
qa_detail_mode={qa_detail_mode}
"""


__all__ = [
    "build_candidate_question_system_prompt",
    "build_evidence_answer_system_prompt",
]
