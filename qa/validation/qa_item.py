# 文件作用：校验并归一化问答条目的题型、难度和必要字段。
# 关联说明：位于 generation/grounding 之后，统一归一化最终问答条目字段。

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from qa.generation import contains_ambiguous_reference

ALLOWED_QUESTION_TYPES = {"简答题", "单选题", "判断题", "计算题"}
ALLOWED_DIFFICULTY_LEVELS = {"简单", "中等", "困难"}
ALLOWED_CORRECT_OPTIONS = {"A", "B", "C", "D"}


def normalize_question_type(raw: Any, *, expected: Optional[str] = None) -> str:
    s = str(raw or "").strip()
    if s in ALLOWED_QUESTION_TYPES:
        return s
    if not s and expected in ALLOWED_QUESTION_TYPES:
        return str(expected)

    slim = re.sub(r"[\s（）()\[\]{}<>《》“”\"'`]+", "", s)
    lower = slim.lower()

    if slim in {"选择题", "单选", "单项选择题", "单项选择"}:
        return "单选题"
    if slim in {"判断", "是非题", "是非"}:
        return "判断题"
    if slim in {"简答", "问答题", "问答"}:
        return "简答题"
    if slim in {"计算", "计算"}:
        return "计算题"

    if any(key in slim for key in ("单选", "选择")):
        return "单选题"
    if any(key in slim for key in ("判断", "是非")):
        return "判断题"
    if any(key in slim for key in ("简答", "问答")):
        return "简答题"
    if "计算" in slim:
        return "计算题"

    if any(key in lower for key in ("mcq", "multiplechoice", "singlechoice", "choice")):
        return "单选题"
    if any(key in lower for key in ("truefalse", "true/false", "t/f", "boolean")):
        return "判断题"
    if any(key in lower for key in ("shortanswer", "short", "openended", "freeform")):
        return "简答题"
    if any(key in lower for key in ("calculation", "math", "compute")):
        return "计算题"

    if expected in ALLOWED_QUESTION_TYPES:
        return str(expected)
    return "简答题"


def normalize_difficulty_level(raw: Any) -> str:
    s = str(raw or "").strip()
    if s in ALLOWED_DIFFICULTY_LEVELS:
        return s
    lower = s.lower()
    if "简单" in s or lower in {"easy", "easier", "basic"}:
        return "简单"
    if "困难" in s or lower in {"hard", "difficult"}:
        return "困难"
    if "中等" in s or lower in {"medium", "normal", "moderate"}:
        return "中等"
    return "中等"


def normalize_judge_answer(answer: str, language_code: str) -> Optional[str]:
    s = (answer or "").strip()
    if not s:
        return None
    s = re.sub(r"[。．\.\!\?！？,，;；:：\s]+$", "", s).strip()
    lower = s.lower()
    if language_code == "en":
        if s in {"True", "False"}:
            return s
        if lower in {"true", "t", "yes"}:
            return "True"
        if lower in {"false", "f", "no"}:
            return "False"
        return None

    if s in {"正确", "错误"}:
        return s
    if s in {"对", "是"} or lower == "true":
        return "正确"
    if s in {"错", "否", "没有", "无", "不是"} or lower == "false":
        return "错误"
    return None


def strip_option_prefix(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^[A-Da-d]\s*[\.\)\:：、．]\s*", "", t)
    return t.strip()


def parse_options_from_text(text: str) -> Optional[List[str]]:
    s = (text or "").strip()
    if not s:
        return None

    patterns = [
        r"(?s)(?:^|[\n\r;；])\s*([ABCD])\s*[\.\)\:：、．]\s*(.*?)\s*(?=(?:[\n\r;；]\s*[ABCD]\s*[\.\)\:：、．]|$))",
        r"(?s)\b([ABCD])\s*[\.\)\:：、．]\s*(.*?)\s*(?=\b[ABCD]\s*[\.\)\:：、．]|$)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, s, flags=re.IGNORECASE)
        if not matches:
            continue
        mapping: Dict[str, str] = {}
        for letter, content in matches:
            key = str(letter).strip().upper()
            if key not in {"A", "B", "C", "D"}:
                continue
            value = str(content).strip()
            if not value or key in mapping:
                continue
            mapping[key] = value
        if all(key in mapping for key in ("A", "B", "C", "D")):
            return [
                strip_option_prefix(mapping["A"]),
                strip_option_prefix(mapping["B"]),
                strip_option_prefix(mapping["C"]),
                strip_option_prefix(mapping["D"]),
            ]
    return None


def normalize_mcq_options(raw_options: Any) -> Optional[List[str]]:
    if raw_options is None:
        return None

    if isinstance(raw_options, str):
        return parse_options_from_text(raw_options)

    if isinstance(raw_options, dict):
        mapping: Dict[str, str] = {}
        for key, value in raw_options.items():
            normalized_key = str(key).strip().upper()
            if normalized_key in {"A", "B", "C", "D"}:
                mapping[normalized_key] = str(value).strip()
        if all(key in mapping for key in ("A", "B", "C", "D")):
            return [
                strip_option_prefix(mapping["A"]),
                strip_option_prefix(mapping["B"]),
                strip_option_prefix(mapping["C"]),
                strip_option_prefix(mapping["D"]),
            ]
        return None

    if isinstance(raw_options, list):
        if len(raw_options) == 1 and isinstance(raw_options[0], str):
            parsed = parse_options_from_text(raw_options[0])
            if parsed:
                return parsed

        if raw_options and all(isinstance(item, dict) for item in raw_options):
            mapping: Dict[str, str] = {}
            for obj in raw_options:
                label = obj.get("label") or obj.get("option") or obj.get("key")
                key = str(label or "").strip().upper()
                if key not in {"A", "B", "C", "D"}:
                    continue
                value = (
                    obj.get("text")
                    or obj.get("content")
                    or obj.get("value")
                    or obj.get(key)
                )
                if value is None:
                    continue
                mapping[key] = str(value).strip()
            if all(key in mapping for key in ("A", "B", "C", "D")):
                return [
                    strip_option_prefix(mapping["A"]),
                    strip_option_prefix(mapping["B"]),
                    strip_option_prefix(mapping["C"]),
                    strip_option_prefix(mapping["D"]),
                ]

        if len(raw_options) == 4:
            options = [strip_option_prefix(str(option)) for option in raw_options]
            if all(option for option in options):
                return options

    return None


def normalize_correct_option(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        index = int(raw)
        if 0 <= index <= 3:
            return ["A", "B", "C", "D"][index]
        return None

    s = str(raw).strip()
    if not s:
        return None
    s_up = s.upper()
    if s_up in ALLOWED_CORRECT_OPTIONS:
        return s_up

    match = re.match(r"^\s*([ABCD])\s*[\.\)\:：、．]\s*", s_up)
    if match:
        return match.group(1)
    match = re.search(r"(?:选项|OPTION|CHOICE)\s*([ABCD])\b", s_up, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if len(s_up) <= 3:
        match = re.search(r"\b([ABCD])\b", s_up)
        if match:
            return match.group(1)
    return None


def infer_correct_option_from_answer(answer: str, options: List[str]) -> Optional[str]:
    ans = (answer or "").strip()
    if not ans:
        return None
    ans_up = ans.upper()
    if ans_up in ALLOWED_CORRECT_OPTIONS:
        return ans_up
    match = re.match(r"^\s*([ABCD])\s*[\.\)\:：、．]?\s*$", ans_up)
    if match:
        return match.group(1)
    match = re.match(r"^\s*([ABCD])\s*[\.\)\:：、．,，\-—]\s*.+$", ans_up)
    if match:
        return match.group(1)

    for index, option in enumerate(options):
        if ans == option:
            return ["A", "B", "C", "D"][index]
        if ans and option and ans.lower() == option.lower():
            return ["A", "B", "C", "D"][index]
    return None


def normalize_mcq_answer_and_correct_option(
    answer: str,
    raw_correct_option: Any,
    options: List[str],
) -> Tuple[Optional[str], str]:
    correct = normalize_correct_option(raw_correct_option)
    if correct not in ALLOWED_CORRECT_OPTIONS:
        correct = infer_correct_option_from_answer(answer, options)

    normalized_answer = answer
    if correct in ALLOWED_CORRECT_OPTIONS:
        index = ord(correct) - ord("A")
        if 0 <= index < len(options):
            ans = (answer or "").strip()
            ans_up = ans.upper()
            has_letter_prefix = bool(
                re.match(r"^\s*[ABCD]\s*[\.\)\:：、．,，\-—]\s*.+$", ans_up)
            )
            if not ans_up or ans_up in ALLOWED_CORRECT_OPTIONS or has_letter_prefix:
                normalized_answer = options[index]
    return correct, normalized_answer


def validate_and_normalize_item_with_reason(
    item: Dict[str, Any],
    language_code: str,
    expected_question_type: Optional[str] = None,
    *,
    fixed_knowledge_category: Optional[str] = None,
    fixed_knowledge_category_confidence: Optional[float] = None,
    fixed_knowledge_category_reason: str = "",
) -> Tuple[Optional[Dict[str, Any]], str]:
    def get_first_str(keys: List[str]) -> str:
        for key in keys:
            if key not in item:
                continue
            s = str(item.get(key) or "").strip()
            if s:
                return s
        return ""

    question = get_first_str(["question", "q"])
    answer = get_first_str(["answer", "a"])
    answer_explanation = get_first_str(["answer_explanation", "explanation"])
    source_fact_text = get_first_str(
        ["source_fact_text", "atomic_fact", "fact", "source_fact"]
    )
    source = get_first_str(["source", "source_id"])
    if not question:
        return None, "missing_question"
    if not answer_explanation:
        return None, "missing_answer_explanation"
    if not source_fact_text:
        return None, "missing_source_fact_text"
    if not source:
        source = "text content" if language_code == "en" else "文本内容"

    question_type = normalize_question_type(
        get_first_str(["question_type", "type"]),
        expected=expected_question_type,
    )
    if expected_question_type == "单选题" and question_type != "单选题":
        if any(key in item for key in ("options", "choices", "correct_option", "correct_answer")):
            question_type = "单选题"
    if expected_question_type == "判断题" and question_type != "判断题":
        if normalize_judge_answer(answer, language_code=language_code) is not None:
            question_type = "判断题"

    difficulty_level = normalize_difficulty_level(get_first_str(["difficulty_level"]))

    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    if question_type == "单选题":
        options = normalize_mcq_options(item.get("options") or item.get("choices"))
        if not options:
            options = parse_options_from_text(question)
        if not options:
            return None, "mcq_invalid_options"
        if len(options) != 4:
            return None, "mcq_options_not_4"
        correct_option, answer = normalize_mcq_answer_and_correct_option(
            answer,
            item.get("correct_option") or item.get("correct_answer"),
            options,
        )
        if correct_option not in ALLOWED_CORRECT_OPTIONS:
            return None, "mcq_invalid_correct_option"
    else:
        options = None
        correct_option = None

    if question_type == "判断题":
        normalized = normalize_judge_answer(answer, language_code=language_code)
        if not normalized:
            return None, "judge_invalid_answer"
        answer = normalized
    if not answer:
        return None, "missing_answer"

    if contains_ambiguous_reference(question, language_code=language_code):
        return None, "ambiguous_reference_question"
    if contains_ambiguous_reference(answer, language_code=language_code):
        return None, "ambiguous_reference_answer"
    if contains_ambiguous_reference(answer_explanation, language_code=language_code):
        return None, "ambiguous_reference_answer_explanation"
    if contains_ambiguous_reference(source_fact_text, language_code=language_code):
        return None, "ambiguous_reference_source_fact_text"

    fixed_kc = str(fixed_knowledge_category or "").strip()
    if fixed_kc:
        knowledge_category = fixed_kc
        knowledge_category_reason = (
            str(fixed_knowledge_category_reason or "").strip()
            or get_first_str(["knowledge_category_reason"])
        )
        try:
            kc_conf = (
                float(fixed_knowledge_category_confidence)
                if fixed_knowledge_category_confidence is not None
                else 0.0
            )
        except Exception:
            kc_conf = 0.0
    else:
        knowledge_category = get_first_str(["knowledge_category"])
        if not knowledge_category:
            knowledge_category = "Uncategorized" if language_code == "en" else "未分类"
        knowledge_category_reason = get_first_str(["knowledge_category_reason"])
        kc_conf_raw = item.get("knowledge_category_confidence")
        try:
            kc_conf = float(kc_conf_raw) if kc_conf_raw is not None else 0.0
        except Exception:
            kc_conf = 0.0
    kc_conf = max(0.0, min(1.0, kc_conf))

    difficulty_score_raw = item.get("difficulty_score")
    try:
        difficulty_score = (
            float(difficulty_score_raw) if difficulty_score_raw is not None else None
        )
    except Exception:
        difficulty_score = None
    if difficulty_score is not None:
        difficulty_score = max(0.0, min(1.0, difficulty_score))

    normalized = {
        "question": question,
        "answer": answer,
        "answer_explanation": answer_explanation,
        "source_fact_text": source_fact_text,
        "source": source,
        "knowledge_category": knowledge_category,
        "knowledge_category_confidence": kc_conf,
        "knowledge_category_reason": knowledge_category_reason,
        "question_type": question_type,
        "question_type_reason": get_first_str(["question_type_reason"]),
        "difficulty_level": difficulty_level,
        "difficulty_score": difficulty_score,
        "options": options,
        "correct_option": correct_option,
    }
    return normalized, "ok"


def validate_and_normalize_item(
    item: Dict[str, Any],
    language_code: str,
) -> Optional[Dict[str, Any]]:
    normalized, _reason = validate_and_normalize_item_with_reason(
        item,
        language_code=language_code,
        expected_question_type=None,
    )
    return normalized


__all__ = [
    "validate_and_normalize_item",
    "validate_and_normalize_item_with_reason",
]
