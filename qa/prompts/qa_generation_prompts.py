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
            return """## Detail mode contract: summary
- qa_detail_mode=summary.
- Each item must contain exactly one standalone question sentence about one coherent topic. Do not place two or more independent questions in the same `question` field.
- Use at most one question mark. Never concatenate questions in forms such as "Who is responsible? What is the ratio?" or "What is X, and how should an unrelated Y be handled?".
- Design the question from one relevant paragraph or one tightly connected passage group. If the source contains unrelated topics, generate separate items instead of combining them.
- "Summary" describes the expected answer: it may synthesize the related rules, steps, conditions, roles, or conclusions needed to answer that single question.
- Ask one umbrella question with one clear answer direction. Multiple related facts may support the answer, but they must all serve the same question intent.
- retrieval_query should connect the shared topic with the key facets that must be checked.
- must_have_terms should cover the shared topic plus the main facets of the expected answer.
- answer_scope_hint may be "same_section" or "cross_chunk" only when the summary cannot be completed from the source chunk alone; it is still only a hint.
"""
        return """## Detail mode contract: point
- qa_detail_mode=point.
- Generate only questions with one clear answer direction and one core fact.
- Prefer one entity/action/condition/deadline/material/threshold/prohibition/exception at a time.
- Do not generate procedure, checklist, comparison, responsibility-split, condition-set, or chapter-summary questions.
- retrieval_query should find evidence for the same single fact, not broaden the question.
- must_have_terms should focus on the single entity/action/condition that proves the answer.
- answer_scope_hint should normally be "source_primary"; use wider hints only for unresolved local reference or definition.
"""

    if mode == "summary":
        return """## 粒度模式契约：总结型
- qa_detail_mode=summary。
- 每个 item 的 question 只能包含一个完整问句，并且只围绕一个明确主题；禁止把两个或多个独立问题塞进同一个 question 字段。
- 每题最多使用一个问号。禁止写成“费用由谁承担？比例由谁规定？”或“X 如何处理，同时另一个无关的 Y 怎么办？”这类拼接问题。
- 应针对一个相关段落或一组紧密衔接的段落自行设计问题；原文包含多个无关主题时，应拆成不同 item，不能合并提问。
- “总结型”描述的是答案组织方式：答案可以归纳回答该单一问题所需的相关规则、步骤、条件、职责或结论。
- 问题必须只有一个清晰的回答方向；多个相关事实可以共同支撑答案，但必须服务于同一个提问意图。
- retrieval_query 应连接共同主题和需要核对的关键侧面。
- must_have_terms 应覆盖共同主题以及预期答案中的主要侧面。
- answer_scope_hint 只有在主来源块不足以完成总结时才可建议 "same_section" 或 "cross_chunk"；它仍只是系统裁决前的建议。
"""
    return """## 粒度模式契约：单点
- qa_detail_mode=point。
- 只生成答案方向单一、核心事实单一的问题。
- 每题只问一个主体、动作、条件、时限、材料、阈值、禁止项或例外。
- 不要生成流程题、清单题、对比题、职责分工题、条件集合题或章节总结题。
- retrieval_query 只用于寻找同一个单点事实的证据，不要扩大问题范围。
- must_have_terms 应聚焦证明答案所需的单个实体、动作或条件。
- answer_scope_hint 通常应为 "source_primary"；只有主来源块存在局部指代或定义缺失时才建议更宽范围。
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
- answer_explanation must map each answer point to concrete evidence directly; do not write meta explanations such as "this answer is based on the main source chunk".
"""
        return """## Detail mode contract: point
- qa_detail_mode=point.
- Keep the question unchanged and answer exactly one core fact.
- The final answer must not combine multiple independent requirements, steps, conditions, or comparisons.
- source_fact_text must be one atomic, standalone fact copied from qa_generation_unit_text.
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
- answer_explanation 必须直接说明“哪个事实支撑哪个结论”，不要写“这个答案基于主来源块/其中提到”这类元叙述。
"""
    return """## 粒度模式契约：单点
- qa_detail_mode=point。
- question 必须保持不变，答案只能回答一个核心事实。
- 答案不得综合多个独立要求、步骤、条件或对比关系。
- source_fact_text 必须是从 qa_generation_unit_text 摘取的单点、可独立成立的事实。
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
2. Internally determine the question intent before writing: the real user's subject, the information they need, and the minimum conditions that would change the answer.
3. Use the title path only as hidden disambiguation and retrieval context. The real answer basis must still come from the source chunk itself.
4. Select only the most useful question angles for training data. Prefer points a real reader would need to understand, execute, remember, or verify.
5. Prioritize questions about:
   - who must do what
   - when or under what conditions something applies
   - what materials, records, approvals, or steps are required
   - what is prohibited, restricted, required, or exempted
   - what consequence, deadline, threshold, or handling rule is stated
6. De-prioritize or reject narrative-only background and management rhetoric.
7. Check every candidate:
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
9. Write a natural user information need, not a paraphrase of the source sentence. Keep only the minimum context needed to make the answer unambiguous; do not mechanically copy a full legal predicate, procedural precondition, or qualifying clause into the question.
10. Do not start or frame a question with source metadata or citation language, including "according to", "under Article", "the document states", document titles, section titles, or clause numbers. Mention a named regulation, standard, paper, or policy only when omitting its name would make the question's subject ambiguous.
11. Keep source-specific wording needed for exact retrieval in retrieval_query and must_have_terms, rather than overloading question with it.
12. Before outputting, silently ask: "Would a real user naturally ask this without seeing the source text?" If not, rewrite it as a direct question about the user's information need.
13. Style examples:
   - Bad: "According to Article 2 of the regulation, what does it provide?"
   - Better: "What matters does this regulation cover?"
   - Bad: "For couples who have registered for marriage and completed a premarital examination before registration, how are the costs handled?"
   - Better: "How are premarital examination costs handled before marriage registration?"

{detail_mode_section}

{category_section}

## Retrieval planning fields
- retrieval_query: a concise search query for finding same-document evidence. Combine the concrete subject, action/condition, key terms, and title context. Do not use a generic restatement of the question.
- must_have_terms: 1 to 6 important entity/action/condition terms that should appear in useful evidence.
- answer_scope_hint: the model's non-authoritative evidence-range suggestion. Use "source_primary" when the main source chunk is enough; "same_section" when nearby same-section chunks may be needed; "cross_chunk" only when the question intentionally needs related chunks. The system will make the final scope decision from retrieval evidence and policy.

## Question type plan
- Follow question_type_plan order when possible.
- If a planned type cannot be supported by a concrete point in the chunk, skip that item instead of inventing weak content.
- question_type_plan = {plan_json}
- Few-shot examples are style-only and must not be copied: {examples_json}

## Required item fields
- question: string
- retrieval_query: string
- must_have_terms: string[]
- answer_scope_hint: "source_primary" | "same_section" | "cross_chunk"
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
2. 先在内部确定提问意图：真实用户关心的对象、想获得的信息，以及去掉后会改变答案的最小必要条件。
3. 标题路径只用于内部消歧和检索语境，不是问题答案的主要依据；真正的答案必须仍然直接来自主来源块正文。
4. 按信息密度和实用价值选择提问切入点，优先选择读者真正需要理解、执行、记忆或核对的内容。
5. 优先选择以下类型的信息出题：
   - 谁负责做什么
   - 在什么条件下适用或不适用
   - 需要哪些材料、记录、审批、步骤
   - 明确的禁止、限制、要求、例外
   - 明确的时限、后果、阈值、处理规则
6. 对只有背景铺垫、原则倡导、价值表述、管理话术的内容，优先少出题或不出题。
7. 逐条检查候选问题：
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
9. 问题必须表达自然的用户信息需求，不能只是原文句子的改写。只保留确保答案不歧义的最小上下文；不要把完整的法规前提、程序性修饰语或限定从句机械搬进问题。
10. 禁止用来源元信息或引用式语言开头或组织问题，包括“根据/依据”“第 X 条”“文件/通知/本文指出”、文件标题、章节标题、条号。只有不写名称会使提问对象混淆时，才可写法规、标准、论文或制度名称，但不得以“根据……规定”式引入。
11. 用于精确检索的来源术语、完整条件和专有名称，应优先放进 retrieval_query 与 must_have_terms，而不是堆进 question。
12. 输出前静默检查：“脱离原文后，真实用户会自然地这样提问吗？”如果不会，改写为直接询问信息需求的问句。
13. 风格示例：
   - 不好：“根据某条例第二条，该条例规定了什么？”
   - 更好：“该条例适用于哪些事项？”
   - 不好：“依法办理结婚登记且在登记前参加婚前医学检查的夫妻，费用如何处理？”
   - 更好：“办理结婚登记前参加婚前医学检查，费用如何承担？”

{detail_mode_section}

{category_section}

## 检索规划字段
- retrieval_query：用于检索同文档证据的短查询，必须包含具体对象、动作/条件、关键术语和必要标题语境；不要只是机械复述问题。
- must_have_terms：1 到 6 个关键实体、动作、条件或术语，用于帮助检索筛选证据。
- answer_scope_hint：大模型对证据范围的非最终建议。主来源块足够时填 "source_primary"；需要同章节上下文时填 "same_section"；确实需要跨 chunk 相关证据时才填 "cross_chunk"。系统会结合检索证据和前端策略做最终裁决。

## 题型计划
- 尽量按 question_type_plan 的顺序输出题型。
- 如果某个计划题型在当前块中找不到具体、可靠的问题点，就跳过该 item，不要硬凑低质量题。
- question_type_plan = {plan_json}
- few-shot 示例只学习风格，不得复用事实：{examples_json}

## 每条 item 必须包含
- question: string
- retrieval_query: string
- must_have_terms: string[]
- answer_scope_hint: "source_primary" | "same_section" | "cross_chunk"
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
- retrieval_query
- must_have_terms
- answer_scope
- question_type
- qa_generation_unit_text with 【主来源块】, optional 【同章节上下文】, and optional 【相关补充】

## Workflow
1. Generate the best evidence-grounded answer for candidate_question. The candidate has already been selected; do not re-filter or reject it.
2. Apply evidence priority strictly:
   - First: 【主来源块】
   - Second: 【同章节上下文】 only when the main source has unresolved reference, omitted subject, definition, or direct local dependency
   - Third: 【相关补充】 only when answer_scope is "cross_chunk" and the retrieved evidence directly supports the missing fact
3. If answer_scope is "source_primary", rely on 【主来源块】. If a requested detail is not specified, state that limitation in the answer instead of returning an empty item.
4. If answer_scope is "same_section" or "cross_chunk", you may use selected evidence, but source_fact_text must still include a direct snippet from 【主来源块】.
5. Produce a direct, natural answer without saying "according to the text/reference/document".
6. Fill evidence_usage with the chunk_id and short snippet for every evidence chunk that materially supports the answer.

{detail_mode_section}

## Retention contract
- Always return one QA item for candidate_question. Do not output an empty items list as a quality decision.
- Express uncertainty or missing detail explicitly in the answer when needed; downstream evaluation, not this generation call, decides whether the item is retained.

## Constraints
1. Keep question exactly the same as candidate_question.
2. The topic must remain centered on 【主来源块】.
3. source_fact_text must be copied from qa_generation_unit_text. It must contain a direct snippet from 【主来源块】. Add retrieved context snippets only when answer_scope permits it and they are necessary.
4. qa_detail_mode=point: source_fact_text must be one atomic, standalone fact.
5. qa_detail_mode=summary: source_fact_text may combine related snippets, but the first and most important supporting snippet must come from 【主来源块】, and every extra snippet must be necessary.
6. answer_explanation must explain why the answer is supported with concrete subjects and facts, not repeat vague rhetoric.
   - Do not mention the source container, such as "main source chunk", "source text", "document", "reference", "content", or "description".
   - Do not use deictic/meta phrasing such as "this answer", "the above", "it", or "these".
   - Good style: "External inspection and quantity check are assigned to the user department; supplier handling is assigned to the purchasing specialist; asset numbering and ledger updates are assigned to the asset administrator."
7. Do not add outside knowledge or assumptions.
8. Do not include citation-style phrases such as "according to the reference", "the document mentions", or "the text states".
9. Do not expand the question scope beyond what candidate_question asks.

{category_section}

## Required fields
- question, answer, answer_explanation, source_fact_text, source
{kc_fields}
- evidence_usage: list of objects with chunk_id, role, snippet, usage
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
- retrieval_query
- must_have_terms
- answer_scope
- question_type
- qa_generation_unit_text，其中包含【主来源块】、可能存在的【同章节上下文】和【相关补充】

## 工作流程
1. 为 candidate_question 生成当前证据能够支持的最佳答案，不要再次判断是否保留该题。
2. 严格按以下证据优先级定位答案依据：
   - 第一优先：【主来源块】
   - 第二优先：【同章节上下文】；仅在主来源块存在定义缺失、主语省略、局部指代、前后条款直接依赖时使用
   - 第三优先：【相关补充】；仅当 answer_scope 为 "cross_chunk" 且检索证据直接支撑缺失事实时使用
3. 如果 answer_scope 为 "source_primary"，以【主来源块】为准；若问题要求的某个细节没有说明，应在答案中明确说明未给出，而不是返回空列表。
4. 如果 answer_scope 为 "same_section" 或 "cross_chunk"，可以使用选中的检索证据，但 source_fact_text 仍必须包含【主来源块】直接片段。
5. 生成直接、自然的答案，不要写“根据原文/根据通知/文中提到”。
6. 填写 evidence_usage，列出每个真正支撑答案的 chunk_id、短片段和用途。

{detail_mode_section}

## 保留契约
- 必须为 candidate_question 返回 1 条问答，不得基于质量判断输出空 items。
- 证据存在不确定或缺失细节时，在答案中如实说明；是否保留该问答由后续评价阶段决定，不由本次生成调用决定。

## 约束
1. question 必须与 candidate_question 完全一致。
2. 问题主题必须围绕【主来源块】。
3. source_fact_text 必须摘自 qa_generation_unit_text，并且必须包含来自【主来源块】的直接证据；只有 answer_scope 允许且严格必要时才补充检索上下文片段。
4. qa_detail_mode=point 时，source_fact_text 必须是单点、可独立成立的事实。
5. qa_detail_mode=summary 时，source_fact_text 可以合并相关片段，但第一条、最核心的证据必须来自【主来源块】，其余片段必须确实参与了答案成立。
6. answer_explanation 必须用具体主体和事实解释“为什么答案成立”，而不是重复空泛套话。
   - 不要提到来源容器，例如“主来源块、原文、文本、文档、参考内容、资料、内容、描述”。
   - 不要使用元叙述或指代词，例如“这个答案、该答案、上述答案、其中、其、这些、该内容”。
   - 推荐写法：“外观检查和数量核对对应使用部门，异常处理对应采购专员，资产编号登记和台账更新对应资产管理员。”
7. 禁止引入外部知识或常识补全。
8. 答案、解释和来源事实中不要出现“根据参考内容/根据通知/文中提到/原文说明”等引用式表达。
9. 不要把问题范围扩写到 candidate_question 之外。

{category_section}

## 必填字段
- question、answer、answer_explanation、source_fact_text、source
{kc_fields}
- evidence_usage: 对象列表，每个对象包含 chunk_id、role、snippet、usage
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
