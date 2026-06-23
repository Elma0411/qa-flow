# 文件作用：定义知识类别提示词模板的数据结构。
# 关联说明：被各类别模板文件复用，定义统一模板结构。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryPromptTemplate:
    key: str
    display_name: str
    level1_values: tuple[str, ...]
    candidate_zh: str
    answer_zh: str
    candidate_en: str
    answer_en: str

    def candidate_section(self, language_code: str) -> str:
        content = self.candidate_en if language_code == "en" else self.candidate_zh
        return _format_section(self.display_name, content, language_code=language_code)

    def answer_section(self, language_code: str) -> str:
        content = self.answer_en if language_code == "en" else self.answer_zh
        return _format_section(self.display_name, content, language_code=language_code)


def _format_section(display_name: str, content: str, *, language_code: str) -> str:
    clean_content = str(content or "").strip()
    if not clean_content:
        return ""
    if language_code == "en":
        return f"## Category-specific template: {display_name}\n{clean_content}"
    return f"## 分类专用模板：{display_name}\n{clean_content}"
