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
"""
        return """## Detail mode: point
- Write one natural, standalone question about one reader need. It may ask about a rule, amount, step, condition, responsibility, or prohibition.
- Do not slice a source sentence into its first half as a question and its last half as an answer.
"""

    if mode == "summary":
        return """## 粒度模式：总结型
- 每个 item 只写一个围绕同一读者需求的完整问句，禁止拼接多个独立问题。
- 相关段落只有描述同一场景或同一规则组时，才能共同支撑一个答案。
- “总结”是答案可以组织多个相关事实，不是一次问好几个问题。
"""
    return """## 粒度模式：单点
- 每个 item 只写一个自然、独立的用户信息需求；可以问规则、金额、步骤、条件、责任或禁止事项。
- 不要把原文一句话的前半句拆成问题、后半句拆成答案。
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


def build_scenario_planner_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    requested_count: int,
    qa_detail_mode: str,
) -> str:
    """Build typed, evidence-bound scenarios before any question is written."""
    mode = str(qa_detail_mode or "auto").strip().lower()
    allowed = (
        "point only" if mode == "point" else "summary only" if mode == "summary" else "point and summary"
    )
    if language_code == "en":
        return f"""# Role: QA scenario planner

Plan at most {max(0, int(requested_count))} useful QA scenarios from logical section materials. Do not write questions or answers.

## Language
{language_instruction.strip()}

## Material contract
- Each material is one logical section. Its body fragments, text, and accepted image descriptions have already been combined.
- The input uses an opaque `material_ref` such as `主材料-A`; the alias has no article number, chunk number, or ordering meaning.
- Use `node_path` and `parent_node_path` to understand the subject and scope. Never infer meaning from the alias.
- Never merge materials merely because they share a chapter or parent heading.
- Use only the supplied material_ref values and never invent one.

## Scenario contract
- point: exactly one material and one atomic fact need; one fact can fully answer the future question.
- In point-only planning, cover different useful materials before proposing another point scenario from the same material.
- summary: one coherent reader need requiring at least two distinct related facts. It may use one material containing a real list or multiple materials that genuinely serve the same reader need.
- Related location is not enough for summary. Skip a summary scenario when the facts do not belong in one answer.
- A material may appear in both a point and a summary scenario when the intents differ. The same material is not permanently assigned to either type.
- Allowed scenario types: {allowed}.
- Prefer distinct, practical reader needs and output fewer scenarios when evidence is insufficient.

## Required JSON fields
Each item must contain `scenario_type`, `intent`, `reader_need`, `required_material_refs`, and `optional_material_refs`.
- `required_material_refs` are the materials whose facts the answer must use.
- `optional_material_refs` are helpful context only; an answer may omit them without making the scenario invalid.
- A point scenario has exactly one required ref and no optional refs. A summary may use one material with a real list or multiple related materials.

Output ONLY raw JSON: {{"items":[...]}}.
"""
    return f"""# 角色：问答出题场景规划器

请根据逻辑 section 材料规划最多 {max(0, int(requested_count))} 个有价值的出题场景。不要生成问题或答案。

## 语言要求
{language_instruction.strip()}

## 材料契约
- 每份材料就是一个逻辑 section；其正文、物理 fragment 和已接受的图片描述已经合并。
- 输入中的 `material_ref` 是如 `主材料-A` 的临时别名，不包含条款号、chunk 编号或任何业务含义；不得从别名中的字符推断事实。
- 必须结合 `node_path`、`parent_node_path` 和正文理解材料的主体与范围；节点路径是给模型理解结构的，不是让问题照抄的来源标签。
- 不得仅因为材料同属一章或同一父标题就把它们合并。
- 只能引用输入给出的 material_ref，不得编造别名。

## 场景契约
- point：只绑定一份材料，围绕一个原子事实需求；未来问题用一个事实即可完整回答。
- 仅规划 point 时，应先覆盖不同的有价值材料，再考虑从同一材料提出第二个场景。
- summary：围绕一个连贯的读者需求，必须综合至少两个不同且相关的信息点。它既可以来自同一材料中的真实枚举，也可以绑定多份确实共同服务于该需求的材料。
- 位置相邻或同属一章不等于相关；若多个事实不适合放进同一个答案，就不得生成总结场景。
- 同一材料在意图不同的情况下，可以同时参与 point 和 summary 场景；材料本身不预先固定为某一种类型。
- 允许的场景类型：{allowed}。
- 优先选择真实读者会关心且互不重复的需求；证据不足时少生成，不要凑数。

## 每条 JSON 必填字段
`scenario_type`、`intent`、`reader_need`、`required_material_refs`、`optional_material_refs`。
- `required_material_refs` 是答案必须使用的材料；`optional_material_refs` 只是可选背景，未被答案使用时不应判为失败。
- point 必须恰好有一个 required ref 且没有 optional ref；summary 可以绑定同一材料中的真实列表，也可以绑定多个真正相关的材料。

只输出纯 JSON：{{"items":[...]}}。
"""


def build_question_editor_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    qa_detail_mode: str,
) -> str:
    """Edit one generated question without changing its evidence-bound intent."""
    mode = _normalize_qa_detail_mode(qa_detail_mode)
    if language_code == "en":
        return f"""# Role: final question editor

Review one generated {mode} question against its scenario and source material.

## Language
{language_instruction.strip()}

Return exactly one decision:
- keep: already natural, standalone, focused, and directly answerable.
- rewrite: preserve the exact intent and answer boundary while making it sound like a real reader question.
- drop: no faithful natural rewrite is possible from the supplied material.

Remove copied clause syntax, source-sentence prefixes, vague references, source/document viewpoints, and joined independent asks. Do not add a condition, subject, fact, or scope not present in the scenario. A summary question must still require all bound materials; a point question must still ask one fact. Keep question_type unchanged.
- Check that the question names the concrete subject and object; do not leave a pronoun without an explicit antecedent.
- Bad: "What should it do?" Good: "What should the applicant do after the application materials are accepted?"

## Strict rewrite examples
- Source-shaped: "For employees who lawfully give birth, how many additional days beyond statutory maternity leave may be taken?"
  Rewrite: "How many additional days of maternity leave can an employee take after giving birth?"
- Source-shaped: "Where application materials are complete and accepted, within how many working days shall review be completed?"
  Rewrite: "How long does the review take after a complete application is accepted?"
- A question is not natural merely because it is grammatical or directly answerable. If it copies a source precondition followed by a comma or conditional clause, choose rewrite unless that condition is indispensable to distinguish the rule.

Output ONLY raw JSON: {{"decision":"keep|rewrite|drop","question":"...","reason":"..."}}.
"""
    return f"""# 角色：最终问题编辑器

请结合场景和来源材料，审校一条已经生成的 {mode} 问题。

## 语言要求
{language_instruction.strip()}

只能作出一个决定：
- keep：问题已经自然、独立、聚焦且能由材料直接回答。
- rewrite：严格保持原意和答案边界，只把表达改成真实读者会问的自然问句。
- drop：无法在不改变事实或范围的前提下合理改写。

需要消除条文照搬、原句前半段式问法、模糊指代、文件/原文视角和多个独立事项拼问。不得新增场景中没有的条件、主体、事实或范围。总结题改写后仍必须需要全部绑定材料；单点题仍只能问一个事实。question_type 不得改变。
- 检查问题是否直接写出明确主体和对象；没有明确先行词时，不得保留“该、其、上述、其中、此类、他们”等指代。
- 反例：“该人员如何办理？”；正确：“申请材料齐全的申请人应当如何办理登记？”

## 严格改写示例
- 条文式：“女职工合法生育子女的，在法定产假之外可以增加多少天产假？”
  改写：“女职工生育后，可以额外休多少天产假？”
- 条文式：“申请材料齐全并受理后，应当在多少个工作日内完成审核？”
  改写：“材料齐全的申请获受理后，审核需要多长时间？”
- 语法通顺、能够作答，不等于自然。凡是把原文的条件从句直接搬到逗号前，再把后半句改成疑问的，原则上必须 rewrite；只有该条件用于区分不同规则时才保留最少必要部分。

只输出纯 JSON：{{"decision":"keep|rewrite|drop","question":"...","reason":"..."}}。
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
1. Follow the supplied scenario_intent and reader_need; do not choose a different information focus.
2. Write the one question that this reader would actually ask.
3. Silently test the wording: it must be answerable from the material, standalone, and natural without the reader having seen the source. Rewrite or skip it when it sounds clause-shaped.

## Question rules
- Write one complete question about the supplied coherent need. Keep only the context needed to identify the scenario.
- Do not convert the first half of a source sentence into a question, carry every legal predicate into the question, or imitate source syntax.
- Do not mention sources, documents, sections, article numbers, titles, "according to", or vague references.
- State the concrete subject, object, condition, and action whenever the source fragment could otherwise be ambiguous.
- Do not use "it", "this", "that", "the former", or another pronoun without an explicit antecedent in the question itself.
- Bad: "What obligations does it have?" Good: "What obligations does a physician have when treating a couple with a hereditary condition that makes pregnancy inadvisable?"
- Prefer a practical rule, amount, step, condition, responsibility, prohibition, deadline, exception, mechanism, or comparison over background or slogans.
- Example: write "How much additional maternity leave is available after childbirth?", not "For employees who legally give birth, how many additional leave days apply beyond statutory maternity leave?"

{detail_mode_section}

{category_section}

- Follow `question_type_plan` when supported; otherwise skip rather than inventing a weak item.
- question_type_plan: {plan_json}
- Few-shot examples are style-only and must not copy facts: {examples_json}

## Required JSON fields
`question`, `question_type`, `question_type_reason`, `difficulty_level`, `difficulty_score`.

Output ONLY raw JSON: {{"items":[...]}}.
"""

    return f"""# 角色：基于材料的自然问题撰写者

请从提供的材料中生成最多 {max_candidates} 个训练数据问题。没有值得问的信息时可以少出题或不出题。不要生成答案。

## 语言要求
{language_instruction.strip()}

## 内部工作法，不要写进问题
1. 严格遵循输入给出的 scenario_intent 和 reader_need，不得另选信息焦点。
2. 写出这个读者针对该需求实际会问的一句话。
3. 静默自检：问题能由材料回答、脱离材料也意思完整，而且读者没看过材料时仍会自然地这样问；不自然就改写或跳过。

## 问题规则
- 每条只问输入场景所规定的完整、连贯信息需求；只保留识别场景所需的最少上下文。
- 不要把原文一句话的前半句改成问题，也不要把完整法规前提、原文句式或检索细节搬进问题。
- 不要出现“根据/依据”、条号、文件名、章节名、“文中指出”等来源视角，也不要使用指代不明的词。
- 如果脱离材料后主体、对象、条件或动作可能不清楚，必须在问题中直接写出来。
- 禁止使用没有明确先行词的“该、其、上述、其中、此类、相关人员、他们”等指代。
- 反例：“其需要履行哪些义务？”；正确：“医师对患有不宜生育遗传性疾病的夫妻需要履行哪些义务？”
- 优先问实际会关心的规则、金额、步骤、条件、责任、禁止、期限、例外、机制或对比；跳过背景、口号和空泛管理表述。
- 例如写“生育后还能增加多少天产假？”，不要写“职工合法生育子女的，在法定产假之外可以增加多少天产假？”。

{detail_mode_section}

{category_section}

- 在有证据支持时尽量遵循 question_type_plan；不能支持就跳过，不要硬凑。
- question_type_plan：{plan_json}
- few-shot 示例只学习风格，不得复用事实：{examples_json}

## 每条 JSON 必填字段
`question`、`question_type`、`question_type_reason`、`difficulty_level`、`difficulty_score`。

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
- question_type
- readable evidence material with 【主来源材料】 and optional 【检索证据】
- allowed `evidence_ref` labels; these are the only evidence identifiers you may return

## Workflow
1. Generate the best evidence-grounded answer for candidate_question. The candidate has already been selected; do not re-filter or reject it.
2. Start with 【主来源材料】 and use 【检索证据】 only when it directly supports a fact needed by the question.
3. If a requested detail is not specified in the supplied evidence, state that limitation in the answer instead of returning an empty item.
5. Produce a direct, natural answer without saying "according to the text/reference/document".
5a. Make the subject, object, condition, and action explicit whenever a fragment could be read two ways; do not begin with an unexplained "it", "this", or "that". Bad: "It should be handled by them." Good: "The registration office should review the applicant's complete materials."
5b. The question, answer, answer_explanation, and source_fact_text must each be understandable without the source in view. Rewrite any sentence whose "it", "this", "that", "the above", or another pronoun has no explicit antecedent. Bad: "It must be completed within five days." Good: "The registration office must complete the review within five working days."
6. Fill evidence_usage with only `evidence_ref` and `role` for every material section that directly supports the answer. Do not invent or output chunk IDs, snippets, or usage descriptions.
7. Treat labels such as `主材料-1` and `检索证据-1` as bookkeeping only; never copy them into question, answer, answer_explanation, or source_fact_text.
8. For a summary question, cite every primary material required by the question. Do not answer only the first half of a multi-fact scenario.

{detail_mode_section}

## Retention contract
- Always return one QA item for candidate_question. Do not output an empty items list as a quality decision.
- Express uncertainty or missing detail explicitly in the answer when needed; downstream evaluation, not this generation call, decides whether the item is retained.

## Constraints
1. Keep question exactly the same as candidate_question.
2. The topic must remain centered on 【主来源材料】.
3. source_fact_text must be copied from the readable evidence material. It must contain a direct snippet from 【主来源材料】. Add retrieved snippets only when they are necessary.
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
- evidence_usage: list of objects with evidence_ref and role only
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
- question_type
- 可读证据材料，其中包含【主来源材料】和可选的【检索证据】
- 可选的 `evidence_ref` 标签；这是 evidence_usage 中唯一允许使用的证据标识

## 工作流程
1. 为 candidate_question 生成当前证据能够支持的最佳答案，不要再次判断是否保留该题。
2. 先使用【主来源材料】，只有【检索证据】直接支撑问题所需事实时才补充使用。
3. 如果提供的证据没有说明问题要求的某个细节，应在答案中明确说明未给出，而不是返回空列表。
5. 生成直接、自然的答案，不要写“根据原文/根据通知/文中提到”。
5a. 如果块内事实可能产生歧义，答案必须重复写出明确主体、对象、条件和动作；禁止没有先行词的“该、其、上述、其中、此类、他们、相关人员”。反例：“应当由其办理。”；正确：“登记机关应当审核申请人提交的完整材料。”
5b. 问题、答案、answer_explanation 和 source_fact_text 都必须脱离材料独立理解。若“该事项、其、上述、其中、此类、他们、相关人员”等没有明确先行词，必须改写为具体主体或对象。反例：“其中应在五日内完成。”；正确：“登记机关应在五个工作日内完成婚姻登记审核。”
6. 填写 evidence_usage，只列出真正支撑答案的 `evidence_ref` 和 `role`；不得编造或输出 chunk_id、snippet、usage。
7. `主材料-1`、`检索证据-1` 等标签仅用于证据追踪，不得写进 question、answer、answer_explanation 或 source_fact_text。
8. 总结题必须引用回答该问题所必需的每份主材料，不得只回答多事实场景的前半部分。

{detail_mode_section}

## 保留契约
- 必须为 candidate_question 返回 1 条问答，不得基于质量判断输出空 items。
- 证据存在不确定或缺失细节时，在答案中如实说明；是否保留该问答由后续评价阶段决定，不由本次生成调用决定。

## 约束
1. question 必须与 candidate_question 完全一致。
2. 问题主题必须围绕【主来源材料】。
3. source_fact_text 必须摘自可读证据材料，并且必须包含来自【主来源材料】的直接证据；只有严格必要时才补充检索证据片段。
4. qa_detail_mode=point 时，source_fact_text 必须是单点、可独立成立的事实。
5. qa_detail_mode=summary 时，source_fact_text 可以合并相关片段，但第一条、最核心的证据必须来自【主来源材料】，其余片段必须确实参与了答案成立。
6. answer_explanation 必须是 1 到 2 句完整、面向读者的说明，解释答案为什么适用于问题场景。
   - 说明应补足答案中的规则、条件、因果或适用边界，而不是复述 source_fact_text。
   - 证据追踪由 source_fact_text 和 evidence_usage 完成；不要把 explanation 写成“某句原文支持某结论”或原文的半句话。
   - 第一句直接说清具体主体或规则，不要以“该优惠、该答案、此项、上述、其中、它”等指代词开头；也不要以“其、此类、相关人员”等没有明确先行词的指代词开头。
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
- evidence_usage: 对象列表，每个对象只包含 evidence_ref、role
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
    "build_question_editor_system_prompt",
    "build_scenario_planner_system_prompt",
]
