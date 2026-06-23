# 文件作用：修正 OCR/Markdown 文档中的标题层级。
# 关联说明：作为 easy_dataset 的标题修正辅助，也可通过 __init__ 单独调用。

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from qa.common import extract_first_choice_content, safe_response_dump


_RE_MARKDOWN_HEADING = re.compile(
    r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)(?:\s*\{#[\w-]+\})?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MarkdownHeadingLine:
    index: int
    level: int
    title: str
    line_start: int
    line_end: int
    prev_title: str
    next_title: str


def extract_markdown_heading_lines(markdown_text: str) -> List[MarkdownHeadingLine]:
    matches = list(_RE_MARKDOWN_HEADING.finditer(str(markdown_text or "")))
    result: List[MarkdownHeadingLine] = []
    for idx, match in enumerate(matches, start=1):
        title = str(match.group("title") or "").strip()
        prev_title = str(matches[idx - 2].group("title") or "").strip() if idx > 1 else ""
        next_title = str(matches[idx].group("title") or "").strip() if idx < len(matches) else ""
        result.append(
            MarkdownHeadingLine(
                index=idx,
                level=len(match.group("hashes") or ""),
                title=title,
                line_start=match.start(),
                line_end=match.end(),
                prev_title=prev_title,
                next_title=next_title,
            )
        )
    return result


def looks_like_ocr_markdown(
    markdown_text: str,
    *,
    original_filename: str = "",
    force_enable: bool = False,
) -> bool:
    if force_enable:
        return True
    text = str(markdown_text or "")
    headings = extract_markdown_heading_lines(text)
    if not headings:
        return False
    if len(headings) >= 3:
        return True
    if original_filename.lower().endswith(".md") and not any(token in text for token in ("# ", "## ", "### ")):
        return False
    return len(headings) >= 1 and len(text) >= 200


def build_heading_correction_payload(headings: Sequence[MarkdownHeadingLine]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in headings:
        payload.append(
            {
                "index": item.index,
                "current_level": item.level,
                "title": item.title,
                "prev_title": item.prev_title,
                "next_title": item.next_title,
            }
        )
    return payload


def _parse_level_items(raw: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads((raw or "").strip()) if raw else None
    except Exception:
        return []
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def build_heading_correction_prompt(*, headings_payload: Sequence[Dict[str, Any]]) -> str:
    return (
        "你是一个严谨的中文文档 Markdown 标题层级校正器，擅长处理 OCR 转出的公文、制度、通知、报告、纪要等文档。\n"
        "任务：只根据标题列表，判断每个标题更合理的 Markdown 层级。\n"
        "硬约束：\n"
        "1. 不允许改变标题顺序。\n"
        "2. 不允许新增、删除或改写标题文本。\n"
        "3. 只能为每个标题返回 new_level，范围 1 到 6。\n"
        "4. 输出 items 数量必须与输入完全一致，index 必须一一对应。\n"
        "5. 你的目标是纠正 OCR 导致的层级误判，不是重写目录。\n\n"
        "层级判断规则：\n"
        "1. 优先遵循最小改动原则：如果当前层级看起来合理，就保持不变；只有在明显不合理时才调整。\n"
        "2. 优先遵循保守修正原则：如果无法确定，应保留原层级，不要凭空重构目录。\n"
        "3. 相同编号样式的标题，默认应保持同一层级；例如多个“一、二、三、”通常并列，多个“（一）（二）（三）”通常并列，多个“1. 2. 3.”通常并列。\n"
        "4. 常见中文层级可参考：文档标题 -> “一、二、三、” -> “（一）（二）（三）” -> “1. 2. 3.” -> “（1）（2）（3）”。如果上下文支持，可按这个顺序逐级下降。\n"
        "5. 不要把明显并列的标题错误地改成父子关系；也不要把明显父子关系全部压成同一级。\n"
        "6. 除非上下文非常明确，不要让相邻标题层级发生超过 1 级的突变。\n"
        "7. 文档主标题通常只能有一个最高层标题；除非确实存在多个并列主章节，否则不要把大量标题都提升到第 1 级。\n"
        "8. 如果前后相邻标题的编号样式相近、语义上也像并列项，应尽量保持同层级。\n"
        "9. 如果某个标题文本明显是前一个标题的子项，例如从“一、总则”进入“（一）适用范围”，则子项层级应比父项低一级。\n"
        "10. 你的任务只是校正层级深度，不是优化标题文本，也不是补全遗漏标题。\n\n"
        "输出 JSON：\n"
        '{ "items": [ { "index": 1, "new_level": 1 }, ... ] }\n\n'
        f"标题列表：\n{json.dumps(list(headings_payload), ensure_ascii=False)}"
    )


def request_heading_level_corrections(
    client: OpenAI,
    *,
    headings: Sequence[MarkdownHeadingLine],
    model: str,
    request_timeout: int,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[List[int]]:
    if not headings:
        return None
    payload = build_heading_correction_payload(headings)
    prompt = build_heading_correction_prompt(headings_payload=payload)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        timeout=request_timeout,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "你只输出合法 JSON，不输出任何额外解释。"
                    "你必须严格执行最小改动、保守修正、同编号样式保持一致层级这三条原则。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    raw_content = extract_first_choice_content(response)
    items = _parse_level_items(raw_content)
    if debug_writer:
        debug_writer(
            {
                "event": "markdown_heading_correction_llm_call",
                "input_headings": payload,
                "raw_response": safe_response_dump(response),
                "parsed_items": items,
            }
        )
    if len(items) != len(headings):
        return None

    result_levels: List[int] = []
    for expected, item in zip(headings, items):
        try:
            idx = int(item.get("index"))
            new_level = int(item.get("new_level"))
        except Exception:
            return None
        if idx != expected.index:
            return None
        if new_level < 1 or new_level > 6:
            return None
        result_levels.append(new_level)
    return result_levels


def apply_heading_level_corrections(
    markdown_text: str,
    headings: Sequence[MarkdownHeadingLine],
    corrected_levels: Sequence[int],
) -> str:
    if not headings or not corrected_levels or len(headings) != len(corrected_levels):
        return str(markdown_text or "")
    text = str(markdown_text or "")
    parts: List[str] = []
    cursor = 0
    for heading, new_level in zip(headings, corrected_levels):
        parts.append(text[cursor : heading.line_start])
        replacement = f'{"#" * int(new_level)} {heading.title}'
        parts.append(replacement)
        cursor = heading.line_end
    parts.append(text[cursor:])
    return "".join(parts)


def correct_markdown_heading_levels(
    markdown_text: str,
    *,
    client: OpenAI,
    model: str,
    request_timeout: int,
    original_filename: str = "",
    force_enable: bool = False,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, Dict[str, Any]]:
    raw_text = str(markdown_text or "")
    headings = extract_markdown_heading_lines(raw_text)
    report: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "reason": "",
        "heading_count": len(headings),
        "changed_count": 0,
        "before_levels": [item.level for item in headings],
        "after_levels": [item.level for item in headings],
    }
    if not looks_like_ocr_markdown(
        raw_text,
        original_filename=original_filename,
        force_enable=force_enable,
    ):
        report["reason"] = "not_ocr_markdown"
        return raw_text, report
    if len(headings) < 2:
        report["reason"] = "insufficient_headings"
        return raw_text, report

    report["enabled"] = True
    corrected_levels = request_heading_level_corrections(
        client,
        headings=headings,
        model=model,
        request_timeout=request_timeout,
        debug_writer=debug_writer,
    )
    if not corrected_levels:
        report["reason"] = "invalid_llm_output"
        return raw_text, report

    changed_count = sum(1 for old, new in zip((item.level for item in headings), corrected_levels) if int(old) != int(new))
    report["after_levels"] = list(corrected_levels)
    report["changed_count"] = changed_count
    if changed_count <= 0:
        report["reason"] = "no_change"
        return raw_text, report

    corrected_text = apply_heading_level_corrections(raw_text, headings, corrected_levels)
    report["applied"] = True
    report["reason"] = "corrected"
    return corrected_text, report


__all__ = [
    "MarkdownHeadingLine",
    "extract_markdown_heading_lines",
    "looks_like_ocr_markdown",
    "correct_markdown_heading_levels",
]
