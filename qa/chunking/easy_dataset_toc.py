# 文件作用：处理 Markdown 标题锚点、目录提取和目录 Markdown 输出。
# 关联说明：被 easy_dataset.py 的 markdown 切分流程调用，和 outline/section 处理逻辑配套。

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

_RE_MARKDOWN_HEADING = re.compile(
    r"^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$",
    re.MULTILINE,
)

def generate_anchor_id(title: str) -> str:
    anchor = re.sub(r"\s+", "-", str(title or "").strip().lower())
    anchor = re.sub(r"[^\w\-]", "", anchor)
    anchor = re.sub(r"\-+", "-", anchor)
    return anchor.strip("-")
def _build_nested_toc(items: Sequence[Dict[str, Any]], include_links: bool) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = [{"level": 0, "children": result}]
    for item in items:
        toc_item = {
            "title": item["title"],
            "level": item["level"],
            "position": item["position"],
            "children": [],
        }
        if include_links:
            toc_item["link"] = f"#{item['anchorId']}"
        while stack and int(stack[-1]["level"]) >= int(item["level"]):
            stack.pop()
        stack[-1]["children"].append(toc_item)
        stack.append(toc_item)
    return result
def extract_table_of_contents(
    text: str,
    options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    options = dict(options or {})
    max_level = int(options.get("maxLevel") or 6)
    include_links = bool(options.get("includeLinks", True))
    flat_list = bool(options.get("flatList", False))
    toc_items: List[Dict[str, Any]] = []
    for match in _RE_MARKDOWN_HEADING.finditer(str(text or "")):
        level = len(match.group(1))
        if level > max_level:
            continue
        title = str(match.group(2) or "").strip()
        toc_items.append(
            {
                "level": level,
                "title": title,
                "position": match.start(),
                "anchorId": generate_anchor_id(title),
                "children": [],
            }
        )
    if flat_list:
        result = []
        for item in toc_items:
            entry = {
                "level": item["level"],
                "title": item["title"],
                "position": item["position"],
            }
            if include_links:
                entry["link"] = f"#{item['anchorId']}"
            result.append(entry)
        return result
    return _build_nested_toc(toc_items, include_links)
def _nested_toc_to_markdown(
    items: Sequence[Dict[str, Any]],
    *,
    indent: int = 0,
    include_links: bool = True,
) -> str:
    result = ""
    indent_str = "  " * indent
    for item in items:
        title_text = (
            f"[{item['title']}]({item['link']})"
            if include_links and item.get("link")
            else str(item.get("title") or "")
        )
        result += f"{indent_str}- {title_text}\n"
        children = item.get("children") or []
        if children:
            result += _nested_toc_to_markdown(
                children,
                indent=indent + 1,
                include_links=include_links,
            )
    return result
def _flat_toc_to_markdown(
    items: Sequence[Dict[str, Any]],
    *,
    include_links: bool = True,
) -> str:
    result = ""
    for item in items:
        indent = "  " * max(0, int(item.get("level") or 1) - 1)
        title_text = (
            f"[{item['title']}]({item['link']})"
            if include_links and item.get("link")
            else str(item.get("title") or "")
        )
        result += f"{indent}- {title_text}\n"
    return result
def toc_to_markdown(
    toc: Sequence[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
) -> str:
    options = dict(options or {})
    is_nested = bool(options.get("isNested", True))
    include_links = bool(options.get("includeLinks", True))
    return (
        _nested_toc_to_markdown(toc, include_links=include_links)
        if is_nested
        else _flat_toc_to_markdown(toc, include_links=include_links)
    )

__all__ = ["extract_table_of_contents", "generate_anchor_id", "toc_to_markdown"]
