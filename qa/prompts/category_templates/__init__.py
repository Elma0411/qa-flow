# 文件作用：作为知识类别提示词模板注册表的公共入口。
# 关联说明：导出 registry 中的模板解析能力，供 qa_generation_prompts 使用。

from .registry import (
    build_category_answer_section,
    build_category_candidate_section,
    resolve_category_prompt_template,
    resolve_category_prompt_template_key,
)

__all__ = [
    "build_category_answer_section",
    "build_category_candidate_section",
    "resolve_category_prompt_template",
    "resolve_category_prompt_template_key",
]
