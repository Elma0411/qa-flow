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
    allowed = "point" if mode == "point" else "summary" if mode == "summary" else "point or summary"
    maximum = max(0, int(requested_count))
    profile = str(category_profile or "").strip()
    if language_code == "en":
        return f"""# QA scenario planner

Plan at most {maximum} useful scenarios. Do not write questions or answers.

{language_instruction.strip()}
{profile}

Each input material is one logical section. Use its path, ordinary text, and typed image descriptions to understand the subject and scope. Material and image labels are temporary aliases; use only labels supplied in the input.

Point: bind exactly one required material and one atomic reader need.
Summary: bind one real enumeration or a tightly connected group of at most three required materials. Do not summarize a whole manual or join merely adjacent sections.
Visual: the future answer needs an image fact.
Mixed: the future answer needs both a text fact and an image fact.
Text: the future answer needs text facts only.

Allowed scenario type: {allowed}.
Return only JSON: {{"items":[{{"scenario_type":"point|summary","intent":"...","reader_need":"...","required_material_refs":[],"optional_material_refs":[],"evidence_mode":"text|visual|mixed","required_image_refs":[]}}]}}.
"""
    return f"""# 问答场景规划器

规划最多 {maximum} 个有价值的场景，不生成问题和答案。

{language_instruction.strip()}
{profile}

每份输入材料就是一个逻辑 section。结合节点路径、正文和独立图片描述理解主体与范围。材料和图片标签只是本次调用的临时别名，只能使用输入中已有的标签。

单点场景：恰好绑定一份必需材料，围绕一个原子读者需求。
总结场景：绑定同一真实枚举，或最多三份紧密相关的必需材料；不得概括整份手册，也不得只因位置相邻就合并。
视觉场景：完整答案需要图片事实。
混合场景：完整答案同时需要正文事实和图片事实。
文本场景：完整答案只需要正文事实。

允许的场景类型：{allowed}。
只输出 JSON：{{"items":[{{"scenario_type":"point|summary","intent":"...","reader_need":"...","required_material_refs":[],"optional_material_refs":[],"evidence_mode":"text|visual|mixed","required_image_refs":[]}}]}}。
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

Use one sentence a real reader would ask. Keep only the context needed to distinguish the rule or situation. Write one core ask for point mode. Write one umbrella ask for summary mode; leave details for the answer. Preserve every mandatory fact named in the brief, including a required visual fact when present.
{example}

Return only JSON: {{"question":"..."}}.
"""
    return f"""# 自然问题撰写者

请根据给出的写作 brief 写出一句自然问题。

{language_instruction.strip()}

写成真实读者会提出的一句话，只保留区分规则或场景所需的最少上下文。单点题只问一个核心事项；总结题只写一个总括问题，细节留给答案。brief 标明必须保留的事实必须体现在问题中；若有必须涉及的图片事实，也必须保留该提问目标。
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

Keep the stated question goal and mandatory facts. Remove source-shaped phrasing and unnecessary legal predicates. Point mode keeps one core ask. Summary mode keeps one umbrella ask. If the brief includes a required visual fact, retain that visual target rather than broadening the question into a text-only how-to question.
{example}

Return only JSON: {{"question":"..."}}.
"""
    return f"""# 最终问题编辑器

请把给出的草稿改写为一句自然、独立的读者问题。

{language_instruction.strip()}

保持既定提问目标和必须保留的事实，去掉条文式前半句和不必要的法律谓词。单点题只保留一个核心问项；总结题只保留一个总括问项。brief 若给出必须涉及的图片事实，必须保留该视觉提问目标，不能把问题泛化成只靠正文即可回答的“如何办理”。
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

Keep the answer focused on the question. For point mode, answer one core fact. For summary mode, organize only the related facts needed by the question. Make subjects and conditions clear. `source_fact_text` is the direct supporting fact or facts from the evidence blocks. In `evidence_usage`, cite every required evidence block actually used. When a required image block is present, use and cite it.
{choice_section_en}

Return only JSON: {output_schema_en}.
"""
    return f"""# 证据问答生成器

请只依据可读证据块回答给出的题目。

{language_instruction.strip()}

答案只回答当前问题。单点题只回答一个核心事实；总结题只组织回答该问题所需的相关事实。主体和条件要清楚。`source_fact_text` 写出证据块中的直接支撑事实；`evidence_usage` 列出实际使用的必需证据块。出现必需图片证据块时，必须使用并引用它。
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
