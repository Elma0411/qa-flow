# 文件作用：注册并解析知识类别到提示词模板的映射。
# 关联说明：汇总同目录各类别模板，并给 qa_generation_prompts 提供解析入口。

from __future__ import annotations

from .base import CategoryPromptTemplate
from .general import TEMPLATE as GENERAL_TEMPLATE
from .knowledge_material import TEMPLATE as KNOWLEDGE_MATERIAL_TEMPLATE
from .normative import TEMPLATE as NORMATIVE_TEMPLATE
from .official_dispatch import TEMPLATE as OFFICIAL_DISPATCH_TEMPLATE
from .research import TEMPLATE as RESEARCH_TEMPLATE
from .standard import TEMPLATE as STANDARD_TEMPLATE


TEMPLATES: tuple[CategoryPromptTemplate, ...] = (
    STANDARD_TEMPLATE,
    NORMATIVE_TEMPLATE,
    OFFICIAL_DISPATCH_TEMPLATE,
    RESEARCH_TEMPLATE,
    KNOWLEDGE_MATERIAL_TEMPLATE,
    GENERAL_TEMPLATE,
)


def resolve_category_prompt_template(
    knowledge_category: str | None,
) -> CategoryPromptTemplate:
    level1 = _extract_level1(knowledge_category)
    for template in TEMPLATES:
        if level1 in template.level1_values:
            return template
    return GENERAL_TEMPLATE


def build_category_candidate_section(
    *,
    knowledge_category: str | None,
    language_code: str,
) -> str:
    return resolve_category_prompt_template(knowledge_category).candidate_section(language_code)


def build_category_answer_section(
    *,
    knowledge_category: str | None,
    language_code: str,
) -> str:
    return resolve_category_prompt_template(knowledge_category).answer_section(language_code)


def resolve_category_prompt_template_key(knowledge_category: str | None) -> str:
    return resolve_category_prompt_template(knowledge_category).key


def _extract_level1(knowledge_category: str | None) -> str:
    raw = str(knowledge_category or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("＞", ">").replace("／", "/")
    for separator in ("/", ">", "\\"):
        if separator in normalized:
            return normalized.split(separator, 1)[0].strip()
    return normalized.strip()
