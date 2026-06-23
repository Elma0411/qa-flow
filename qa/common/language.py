# 文件作用：检测文本语言并生成对应的大模型语言约束提示。
# 关联说明：被 generation、evaluation、augmentation 调用，提供统一语言判断。

import re


def detect_language(text: str) -> str:
    """粗略检测文本语言（中文、英文/未知）。"""
    if not text:
        return "unknown"

    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    if chinese_chars and (len(chinese_chars) / max(len(text), 1)) > 0.05:
        return "zh"

    latin_chars = re.findall(r"[A-Za-z]", text)
    if latin_chars and (len(latin_chars) / max(len(text), 1)) > 0.05:
        return "en"

    return "unknown"


def build_language_instruction(language: str) -> str:
    """根据检测结果生成对大模型的语言约束提示。"""
    if language == "zh":
        return (
            "请确保题干、选项（如有）、答案、答案解释、各类理由等自然语言内容全部使用简体中文，"
            "并与原文语气保持一致。枚举字段请严格按模板/Schema 规定的取值输出。"
        )

    if language == "en":
        return (
            "Please write the natural-language fields in English (question, answer, "
            "answer_explanation, options text if any, knowledge_category, and all reason fields). "
            "Keep enum fields exactly as specified by the schema/template (e.g., question_type and "
            "difficulty_level use the provided Chinese enum values; correct_option is one of A/B/C/D)."
        )

    # 兜底：跟随事实文本的主语言
    return (
        "Please keep the output language consistent with the source text; "
        "if the source mixes languages, follow the main language of the "
        "factual content."
    )
