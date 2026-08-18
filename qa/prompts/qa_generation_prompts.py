"""Small, closed-surface prompts for the QA generation pipeline."""

from __future__ import annotations

from typing import Optional

from qa.prompts.category_templates import resolve_category_prompt_template_key


def _normalize_qa_detail_mode(value: str) -> str:
    mode = str(value or "point").strip().lower()
    return mode if mode in {"point", "summary"} else "point"


def build_planner_category_profile(
    *,
    knowledge_category: Optional[str],
    language_code: str,
) -> str:
    """Return a short planning-only reader profile, never a writer template."""
    key = resolve_category_prompt_template_key(knowledge_category)
    if language_code == "en":
        profiles = {
            "normative": "Reader profile: an applicant or rule executor. Prefer applicable subjects, conditions, procedures, deadlines, consequences, and exceptions.",
            "standard": "Reader profile: an engineer, tester, or inspector. Prefer concrete requirements, thresholds, methods, acceptance rules, and exceptions.",
            "knowledge_material": "Reader profile: a learner or operator. Prefer useful concepts, steps, decision points, and operating precautions.",
            "research": "Reader profile: a researcher. Prefer methods, findings, assumptions, comparisons, and limitations.",
        }
    else:
        profiles = {
            "normative": "读者画像：办事人或制度执行人。优先规划适用对象、条件、流程、期限、后果和例外。",
            "standard": "读者画像：工程、检测或验收人员。优先规划具体要求、指标、方法、验收规则和例外。",
            "knowledge_material": "读者画像：学习者或经办人。优先规划实用概念、步骤、判断点和操作注意事项。",
            "research": "读者画像：研究人员。优先规划方法、发现、假设、比较和局限。",
        }
    return profiles.get(key, "")


def build_scenario_planner_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    requested_count: int,
    qa_detail_mode: str,
    category_profile: str = "",
) -> str:
    """Plan immutable, evidence-bound scenarios before wording any question."""
    mode = str(qa_detail_mode or "auto").strip().lower()
    if mode not in {"point", "summary", "auto"}:
        mode = "auto"
    allowed = "point" if mode == "point" else "summary" if mode == "summary" else "point or summary"
    schema_type = "point" if mode == "point" else "summary" if mode == "summary" else "point|summary"
    maximum = max(0, int(requested_count))
    profile = str(category_profile or "").strip()
    if mode == "point":
        mode_rule_en = "This is a Point-only request. Every `scenario_type` must be `point`; do not return a Summary item."
        mode_rule_zh = "本批次只规划单点场景。每一条 `scenario_type` 必须填写 `point`，不得返回总结场景。"
        shape_rule_en = "Point: bind exactly one required material and one atomic focus. Split independent amounts, conditions, stages, or procedures into separate Point scenarios."
        shape_rule_zh = "单点场景：恰好绑定一份必需材料，围绕一个原子提问焦点。金额、条件、阶段或流程彼此独立时应拆成不同 Point。"
    elif mode == "summary":
        mode_rule_en = "This is a Summary-only request. Every `scenario_type` must be `summary`; do not return a Point item."
        mode_rule_zh = "本批次只规划总结场景。每一条 `scenario_type` 必须填写 `summary`，不得返回单点场景。"
        shape_rule_en = "Summary: bind one real enumeration or a tightly connected group of at most three required materials. Its intent is one reader outcome, not a list of answer facets. Do not summarize a whole manual or join merely adjacent sections."
        shape_rule_zh = "总结场景：绑定同一真实枚举，或最多三份紧密相关的必需材料；`intent` 只表达一个读者结果，不能罗列答案要点。不得概括整份手册，也不得只因位置相邻就合并。"
    else:
        mode_rule_en = "Choose Point or Summary only when it matches the semantic scope of the reader need."
        mode_rule_zh = "只在确实符合读者需求的语义范围时选择 Point 或 Summary。"
        shape_rule_en = "Point: bind exactly one required material and one atomic focus. Split independent amounts, conditions, stages, or procedures into separate Point scenarios.\nSummary: bind one real enumeration or a tightly connected group of at most three required materials. Its intent is one reader outcome, not a list of answer facets. Do not summarize a whole manual or join merely adjacent sections."
        shape_rule_zh = "单点场景：恰好绑定一份必需材料，围绕一个原子提问焦点。金额、条件、阶段或流程彼此独立时应拆成不同 Point。\n总结场景：绑定同一真实枚举，或最多三份紧密相关的必需材料；`intent` 只表达一个读者结果，不能罗列答案要点。不得概括整份手册，也不得只因位置相邻就合并。"
    if language_code == "en":
        return f"""# QA scenario planner

Plan at most {maximum} useful scenarios. Do not write questions or answers.

{language_instruction.strip()}
{profile}

Each input material is one logical section. Use its path, ordinary text, and typed image descriptions to understand the subject and scope. Material and image labels are temporary aliases; use only labels supplied in the input.

{mode_rule_en}
For every item, write `intent` as one short information gap, not a list of answer facts. Include the actor, action, condition, or channel needed to distinguish the rule from a similar one. Never use document deictics such as "this guide", "the above", or "this item"; name the actual operation or object instead.
When closely related materials describe different stages, keep that stage explicit, such as providing materials to an agency for verification versus uploading a file during online declaration.
{shape_rule_en}
Visual: the future answer needs a fact directly observable in the image.
Mixed: the future answer needs both a text fact and a directly observable image fact.
Do not make a visual-only scenario from an outcome that occurs after scanning, clicking, or leaving the image.
For visual or mixed scenarios, prefer a visible action, state, branch, feedback, or required confirmation that the ordinary text does not already state completely. Do not choose a screenshot merely to ask a static value duplicated in text.
Text: the future answer needs text facts only.

Allowed scenario type: {allowed}.
Return only JSON: {{"items":[{{"scenario_type":"{schema_type}","intent":"...","reader_need":"...","required_material_refs":[],"optional_material_refs":[],"evidence_mode":"text|visual|mixed","required_image_refs":[]}}]}}.
"""
    return f"""# 问答场景规划器

规划最多 {maximum} 个有价值的场景，不生成问题和答案。

{language_instruction.strip()}
{profile}

每份输入材料就是一个逻辑 section。结合节点路径、正文和独立图片描述理解主体与范围。材料和图片标签只是本次调用的临时别名，只能使用输入中已有的标签。

{mode_rule_zh}
每条的 `intent` 必须是一个简短的信息缺口，而不是答案事实清单；读者可以自然地只问这一件事。必须保留区分相近规则所需的主体、动作、条件或渠道；不得使用“本说明”“该文件”“上述”等脱离文档便不清楚的指代，应写出实际业务或对象。
相近材料描述的是不同办理阶段时，必须把阶段写清，例如“向经办机构核定时提供资料”和“网上申报上传时提交文件”不能混为同一个场景。
{shape_rule_zh}
视觉场景：完整答案需要图片中可直接观察的事实。
混合场景：完整答案同时需要正文事实和图片中可直接观察的事实。
不能把扫码、点击或离开图片后才可能得到的外部结果规划成纯视觉场景。
视觉或混合场景优先选择正文没有完整写出的可观察操作、状态、分支、反馈或确认动作；不要只为询问正文已重复给出的静态数值而使用截图。
文本场景：完整答案只需要正文事实。

允许的场景类型：{allowed}。
只输出 JSON：{{"items":[{{"scenario_type":"{schema_type}","intent":"...","reader_need":"...","required_material_refs":[],"optional_material_refs":[],"evidence_mode":"text|visual|mixed","required_image_refs":[]}}]}}。
"""


def build_candidate_question_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    qa_detail_mode: str,
    style_example: str = "",
) -> str:
    """Ask for one natural question and nothing else."""
    mode = _normalize_qa_detail_mode(qa_detail_mode)
    example = str(style_example or "").strip()
    if language_code == "en":
        return f"""# Natural question writer

Write one question from the supplied writing brief.

{language_instruction.strip()}

Work silently in this order: choose the stated focus, keep only the identity context needed for that focus, then check that the question does not reveal its own answer.

Use one sentence a real reader would ask. The brief's answer evidence limits the answer scope; do not restate its numbers, lists, dates, steps, or screenshot details unless needed to identify the situation. Write one core ask for point mode. Write one umbrella ask for summary mode; leave answer facets for the answer. When the brief has a visual focus, ask naturally about that observable interface fact without saying "in the image" or copying its displayed values. A permission, prohibition, or eligibility question may validly use a yes/no form; keep that form when it is the stated focus, and never turn it into a how-to question or a broader policy question. If the brief supplies a question object, replace document deictics such as "this guide" with that object instead of repeating the deictic.
{example}

Return only JSON: {{"question":"..."}}.
"""
    return f"""# 自然问题撰写者

请根据给出的写作 brief 写出一句自然问题。

{language_instruction.strip()}

先在心里确定 brief 的提问焦点，再保留识别该焦点所需的最少场景条件，最后检查题干没有提前说出答案。

写成真实读者会提出的一句话。回答依据只用于限定答案范围，不要逐项复述其中的数值、名单、日期、步骤或截图细节，除非不写就无法识别场景。单点题只问一个核心事项；总结题只写一个总括问题，细节留给答案。若有视觉焦点，应自然地询问该可观察的界面事实，不写“图中/截图中”，也不要照搬图片里的数值。询问许可、禁止或资格限制时，“是否/能否/还能……吗”是合法问法，必须保留该肯否关系，不能改成“如何”或泛化成“有什么政策”。brief 给出问题对象时，题干中的“本/该/这份说明、通知或文件”必须换成该对象名称。
{example}

只输出 JSON：{{"question":"..."}}。
"""


def build_question_editor_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    qa_detail_mode: str,
    style_example: str = "",
) -> str:
    """Return a final natural wording; semantic ownership stays with planner."""
    mode = _normalize_qa_detail_mode(qa_detail_mode)
    example = str(style_example or "").strip()
    if language_code == "en":
        return f"""# Final question editor

Rewrite the supplied draft as one natural standalone reader question.

{language_instruction.strip()}

Keep the stated question focus and the minimum identity context. Remove source-shaped phrasing, answer leakage, and unnecessary legal predicates. Point mode keeps one core ask. Summary mode keeps one umbrella ask. Do not turn answer evidence into a checklist in the question. If the brief includes a required visual fact, retain that observable visual target rather than broadening the question into a text-only how-to question. A yes/no permission, prohibition, or eligibility question is already natural when it matches the focus: preserve its affirmative/negative scope, and never rewrite it as a how-to question or a broader policy question. Replace document deictics such as "this guide" with the supplied question object when one is available.
{example}

Return only JSON: {{"question":"..."}}.
"""
    return f"""# 最终问题编辑器

请把给出的草稿改写为一句自然、独立的读者问题。

{language_instruction.strip()}

保持既定提问焦点和最少身份条件，去掉条文式前半句、答案泄露和不必要的法律谓词。单点题只保留一个核心问项；总结题只保留一个总括问项。不要把回答依据改写成题干里的清单。brief 若给出必须涉及的图片事实，必须保留该可观察的视觉提问目标，不能泛化成只靠正文即可回答的“如何办理”。“是否/能否/还能……吗”在询问许可、禁止或资格限制时本来就是自然问题，必须保留原有肯否关系，不能改成“如何”或扩大成“有什么政策”。brief 给出问题对象时，必须用该对象替换题干中的“本/该/这份说明、通知或文件”。
{example}

只输出 JSON：{{"question":"..."}}。
"""


def build_evidence_answer_system_prompt(
    *,
    language_code: str,
    language_instruction: str,
    qa_detail_mode: str,
    question_type: str,
) -> str:
    """Generate an answer from readable evidence blocks and cite their labels."""
    mode = _normalize_qa_detail_mode(qa_detail_mode)
    qtype = str(question_type or "简答题").strip() or "简答题"
    choice_section_en = ""
    choice_section_zh = ""
    output_choice_fields_en = ""
    output_choice_fields_zh = ""
    if qtype == "单选题":
        choice_section_en = "Include exactly four options and one correct_option."
        choice_section_zh = "提供恰好四个 options 和一个 correct_option。"
        output_choice_fields_en = ',"options":["..."],"correct_option":"A"'
        output_choice_fields_zh = ',"options":["..."],"correct_option":"A"'
    elif qtype != "简答题":
        choice_section_en = "Use the requested question form."
        choice_section_zh = "按给定题目形式作答。"
    output_schema_en = (
        '{"items":[{"answer":"...","answer_explanation":"...",'
        '"source_fact_text":"...","evidence_usage":[{"evidence_ref":"...",'
        f'"role":"primary_source|primary_visual"}}]{output_choice_fields_en}}}]}}'
    )
    output_schema_zh = (
        '{"items":[{"answer":"...","answer_explanation":"...",'
        '"source_fact_text":"...","evidence_usage":[{"evidence_ref":"...",'
        f'"role":"primary_source|primary_visual"}}]{output_choice_fields_zh}}}]}}'
    )
    if language_code == "en":
        return f"""# Evidence-grounded answer writer

Answer the supplied question using the readable evidence blocks.

{language_instruction.strip()}

Keep the answer focused on the question. For point mode, answer one core fact. For summary mode, organize only the related facts needed by the question. Make subjects and conditions clear. Do not answer only "yes", "no", "can", or "cannot" when the question contains a rule, condition, amount, timing, or next step: state the conclusion and the key supporting rule. `source_fact_text` is the direct supporting fact or facts from the evidence blocks. In `evidence_usage`, cite every required evidence block actually used. When a required image block is present, use and cite it.
{choice_section_en}

Return only JSON: {output_schema_en}.
"""
    return f"""# 证据问答生成器

请只依据可读证据块回答给出的题目。

{language_instruction.strip()}

答案只回答当前问题。单点题只回答一个核心事实；总结题只组织回答该问题所需的相关事实。主体和条件要清楚。题目涉及规则、条件、金额、期限或后续动作时，不能只答“是/否/可以/不可以”，要给出结论和关键依据。`source_fact_text` 写出证据块中的直接支撑事实；`evidence_usage` 列出实际使用的必需证据块。出现必需图片证据块时，必须使用并引用它。
{choice_section_zh}

只输出 JSON：{output_schema_zh}。
"""


__all__ = [
    "build_candidate_question_system_prompt",
    "build_evidence_answer_system_prompt",
    "build_planner_category_profile",
    "build_question_editor_system_prompt",
    "build_scenario_planner_system_prompt",
]
