# 文件作用：实现 Easy Dataset 风格的文档预处理、标题识别和切块。
# 关联说明：切块主编排，调用 easy_dataset_* 内部模块和 markdown_heading_correction，并被 tree_chunks 包装。

from __future__ import annotations

import re
import time
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from app.core.config import CONFIG
from .easy_dataset_common import _path_stem
from .easy_dataset_errors import EasyDatasetChunkingError
from .easy_dataset_manual import manual_split, normalize_split_points, preview_split_points
from .easy_dataset_preprocessing import preprocess_file
from .easy_dataset_split_modes import (
    _split_code_mode,
    _split_recursive_mode,
    _split_text_mode,
    _split_token_mode,
)
from .easy_dataset_toc import extract_table_of_contents, generate_anchor_id, toc_to_markdown
from .markdown_heading_correction import correct_markdown_heading_levels


ENGINE_VERSION = "easy_dataset_chunking_py_20260416"

DEFAULT_CONFIG: Dict[str, Any] = {
    "splitType": None,
    "textSplitMinLength": 1500,
    "textSplitMaxLength": 2000,
    "chunkSize": 1500,
    "chunkOverlap": 200,
    "separator": "\n\n",
    "separators": ["|", "##", ">", "-"],
    "splitLanguage": "js",
    "customSeparator": "---",
    "manualSplitPoints": None,
    "markdownHeadingCorrectionEnabled": True,
}

ORIGINAL_PROJECT_INPUT_EXTENSIONS = [".md", ".txt", ".docx", ".pdf", ".epub"]
STANDALONE_INPUT_EXTENSIONS = [".md", ".txt", ".docx", ".epub"]
SUPPORTED_SPLIT_TYPES = ("markdown", "text", "token", "recursive", "code", "custom")

CAPABILITIES: Dict[str, Any] = {
    "splitTypes": list(SUPPORTED_SPLIT_TYPES),
    "markdownFeatures": [
        "extractOutline",
        "splitByHeadings",
        "processSections",
        "splitLongSection",
        "generateEnhancedSummary",
        "extractTableOfContents",
        "tocToMarkdown",
        "combineMarkdown",
        "saveToSeparateFiles",
    ],
    "manualSplitFeatures": [
        "previewSplitPoints",
        "manualSplit",
    ],
    "preprocessors": [
        ".md",
        ".txt",
        ".docx",
        ".epub",
    ],
    "originalButNotExtracted": [
        "project-scoped PDF preprocess strategies (default/mineru/vision/mineru-local)",
        "database save/query operations",
        "Next.js API routes and UI dialogs",
    ],
}

_RE_MARKDOWN_HEADING = re.compile(
    r"^(#{1,6})\s+(.+?)(?:\s*\{#[\w-]+\})?\s*$",
    re.MULTILINE,
)
_RE_SENTENCE = re.compile(r"[^.!?。！？]+[.!?。！？]*", re.MULTILINE)
_RE_PART_SUFFIX = re.compile(r"^(?P<head>.+?)\s+-\s+Part\s+(?P<part>\d+/\d+)$", re.IGNORECASE)









def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def _same_text(left: str, right: str) -> bool:
    return _compact_text(left) == _compact_text(right)


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _dedupe_path_lists(paths: Iterable[Sequence[str]]) -> List[List[str]]:
    result: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()
    for path in paths:
        normalized = tuple(part.strip() for part in path if str(part or "").strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(list(normalized))
    return result


def _common_prefix(paths: Sequence[Sequence[str]]) -> List[str]:
    if not paths:
        return []
    prefix = list(paths[0])
    for path in paths[1:]:
        max_len = min(len(prefix), len(path))
        idx = 0
        while idx < max_len and prefix[idx] == path[idx]:
            idx += 1
        prefix = prefix[:idx]
        if not prefix:
            break
    return prefix


def _collect_heading_paths(section: Dict[str, Any]) -> List[List[str]]:
    headings = section.get("headings") or []
    collected: List[List[str]] = []
    for heading in headings:
        if not isinstance(heading, dict):
            continue
        path_parts = heading.get("pathParts") or heading.get("path_parts") or []
        if isinstance(path_parts, list):
            collected.append([str(part).strip() for part in path_parts if str(part).strip()])
    if not collected:
        path_parts = section.get("pathParts") or section.get("path_parts") or []
        if isinstance(path_parts, list):
            collected.append([str(part).strip() for part in path_parts if str(part).strip()])
    return _dedupe_path_lists(collected)


def _effective_doc_title(outline: Sequence[Dict[str, Any]], document_title_hint: Optional[str]) -> str:
    if outline:
        first = outline[0]
        if int(first.get("level") or 0) == 1:
            title = str(first.get("title") or "").strip()
            if title:
                return title
    hint = str(document_title_hint or "").strip()
    return hint or "Document"


def _build_title_path_parts(
    section: Dict[str, Any],
    outline: Sequence[Dict[str, Any]],
    *,
    document_title_hint: Optional[str],
    part_index: Optional[int] = None,
    total_parts: Optional[int] = None,
) -> List[str]:
    heading_paths = _collect_heading_paths(section)
    doc_title = _effective_doc_title(outline, document_title_hint)
    if not heading_paths:
        parts = [doc_title, "Preface"]
    elif len(heading_paths) == 1:
        parts = list(heading_paths[0])
    else:
        prefix = _common_prefix(heading_paths)
        if prefix:
            suffixes = _dedupe_strings(
                " > ".join(path[len(prefix) :]) if path[len(prefix) :] else path[-1]
                for path in heading_paths
            )
            parts = list(prefix) + [f"[{', '.join(suffixes)}]"]
        else:
            parts = [", ".join(" > ".join(path) for path in heading_paths)]
    if part_index is not None and total_parts and total_parts > 1:
        parts = list(parts) + [f"Part {part_index}/{total_parts}"]
    return [str(part).strip() for part in parts if str(part).strip()]


def _heading_paths_to_summary(
    section: Dict[str, Any],
    outline: Sequence[Dict[str, Any]],
    *,
    document_title_hint: Optional[str],
    part_index: Optional[int] = None,
    total_parts: Optional[int] = None,
) -> str:
    heading_paths = _collect_heading_paths(section)
    doc_title = _effective_doc_title(outline, document_title_hint)
    if not heading_paths:
        summary = f"{doc_title} Preface"
    elif len(heading_paths) == 1:
        summary = " > ".join(heading_paths[0])
    else:
        prefix = _common_prefix(heading_paths)
        if prefix:
            suffixes = _dedupe_strings(
                " > ".join(path[len(prefix) :]) if path[len(prefix) :] else path[-1]
                for path in heading_paths
            )
            summary = " > ".join(list(prefix) + [f"[{', '.join(suffixes)}]"])
        else:
            summary = ", ".join(" > ".join(path) for path in heading_paths)
    if part_index is not None and total_parts and total_parts > 1:
        summary += f" - Part {part_index}/{total_parts}"
    return summary


def _section_heading_entry(section: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "heading": section.get("heading"),
        "level": int(section.get("level") or 0),
        "position": int(section.get("position") or 0),
        "pathParts": list(section.get("pathParts") or []),
    }


def _section_heading_line(section: Dict[str, Any]) -> str:
    heading = str(section.get("heading") or "").strip()
    level = max(1, int(section.get("level") or 1))
    if not heading:
        return ""
    return f'{"#" * level} {heading}\n'


def _merge_section_content(base_content: str, section: Dict[str, Any]) -> str:
    heading_line = _section_heading_line(section)
    section_content = str(section.get("content") or "")
    if not base_content:
        return f"{heading_line}{section_content}".strip()
    return f"{base_content}\n\n{heading_line}{section_content}".strip()


def _resolve_markdown_banner(summary: str, content: str) -> str:
    return f"> **📑 Summarization：** *{summary}*\n\n---\n\n{content}"


def get_default_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    if not isinstance(config.get("separators"), list):
        config["separators"] = list(DEFAULT_CONFIG["separators"])
    return config


def get_capabilities() -> Dict[str, Any]:
    return {
        "defaults": deepcopy(DEFAULT_CONFIG),
        "originalProjectInputExtensions": list(ORIGINAL_PROJECT_INPUT_EXTENSIONS),
        "standaloneInputExtensions": list(STANDALONE_INPUT_EXTENSIONS),
        **deepcopy(CAPABILITIES),
    }


def extract_outline(text: str) -> List[Dict[str, Any]]:
    outline: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    for match in _RE_MARKDOWN_HEADING.finditer(str(text or "")):
        level = len(match.group(1))
        title = str(match.group(2) or "").strip()
        while stack and int(stack[-1]["level"]) >= level:
            stack.pop()
        path_parts = [str(item["title"]) for item in stack] + [title]
        current = {
            "level": level,
            "title": title,
            "position": match.start(),
            "pathParts": path_parts,
            "anchorId": generate_anchor_id(title),
        }
        outline.append(current)
        stack.append({"level": level, "title": title})
    return outline


def split_by_headings(text: str, outline: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_text = str(text or "")
    if not outline:
        return [
            {
                "heading": None,
                "level": 0,
                "content": raw_text,
                "position": 0,
                "pathParts": [],
            }
        ]

    sections: List[Dict[str, Any]] = []
    first_position = int(outline[0].get("position") or 0)
    if first_position > 0:
        front_matter = raw_text[:first_position].strip()
        if front_matter:
            sections.append(
                {
                    "heading": None,
                    "level": 0,
                    "content": front_matter,
                    "position": 0,
                    "pathParts": [],
                }
            )

    for index, current in enumerate(outline):
        current_position = int(current.get("position") or 0)
        next_position = (
            int(outline[index + 1].get("position") or 0)
            if index < len(outline) - 1
            else len(raw_text)
        )
        line_end = raw_text.find("\n", current_position)
        if line_end < 0:
            line_end = len(raw_text)
        start_position = min(len(raw_text), line_end + 1)
        sections.append(
            {
                "heading": current.get("title"),
                "level": int(current.get("level") or 0),
                "content": raw_text[start_position:next_position].strip(),
                "position": current_position,
                "pathParts": list(current.get("pathParts") or []),
            }
        )
    return sections


def split_long_section(section: Dict[str, Any], max_split_length: int) -> List[str]:
    paragraphs = re.split(r"\n\n+", str(section.get("content") or ""))
    result: List[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_split_length:
            if current_chunk:
                result.append(current_chunk)
                current_chunk = ""
            sentences = [item for item in _RE_SENTENCE.findall(paragraph) if item] or [paragraph]
            sentence_chunk = ""
            for sentence in sentences:
                if len(sentence_chunk + sentence) <= max_split_length:
                    sentence_chunk += sentence
                    continue
                if sentence_chunk:
                    result.append(sentence_chunk)
                if len(sentence) > max_split_length:
                    for start in range(0, len(sentence), max_split_length):
                        result.append(sentence[start : start + max_split_length])
                    sentence_chunk = ""
                else:
                    sentence_chunk = sentence
            if sentence_chunk:
                current_chunk = sentence_chunk
        elif len(f"{current_chunk}\n\n{paragraph}".strip()) <= max_split_length:
            current_chunk = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        else:
            if current_chunk:
                result.append(current_chunk)
            current_chunk = paragraph
    if current_chunk:
        result.append(current_chunk)
    return result


def _split_section_payload(section: Dict[str, Any], max_split_length: int) -> List[str]:
    content = str(section.get("content") or "").strip()
    if not content:
        return []
    heading_line = _section_heading_line(section).strip()
    if not heading_line:
        return split_long_section(section, max_split_length)
    whole = f"{heading_line}\n{content}".strip()
    if len(whole) <= max_split_length:
        return [whole]
    payload_budget = max(40, max_split_length - len(heading_line) - 1)
    payload_parts = split_long_section({"content": content}, payload_budget)
    return [f"{heading_line}\n{part}".strip() for part in payload_parts if str(part or "").strip()]


def generate_enhanced_summary(
    section: Dict[str, Any],
    outline: Sequence[Dict[str, Any]],
    part_index: Optional[int] = None,
    total_parts: Optional[int] = None,
) -> str:
    return _heading_paths_to_summary(
        section,
        outline,
        document_title_hint=None,
        part_index=part_index,
        total_parts=total_parts,
    )


def process_sections(
    sections: Sequence[Dict[str, Any]],
    outline: Sequence[Dict[str, Any]],
    min_split_length: int,
    max_split_length: int,
    *,
    document_title_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for raw_section in sections:
        section = dict(raw_section)
        content = str(section.get("content") or "").strip()
        if not content:
            continue
        if section.get("heading"):
            section["headings"] = [_section_heading_entry(section)]
        else:
            section["headings"] = []
        content_length = len(str(section.get("content") or "").strip())
        if content_length > max_split_length:
            sub_sections = _split_section_payload(section, max_split_length)
            heading_paths = _collect_heading_paths(section)
            for idx, sub_section in enumerate(sub_sections, start=1):
                result.append(
                    {
                        "summary": _heading_paths_to_summary(
                            section,
                            outline,
                            document_title_hint=document_title_hint,
                            part_index=idx,
                            total_parts=len(sub_sections),
                        ),
                        "content": sub_section,
                        "titlePathParts": _build_title_path_parts(
                            section,
                            outline,
                            document_title_hint=document_title_hint,
                            part_index=idx,
                            total_parts=len(sub_sections),
                        ),
                        "_headingPaths": heading_paths,
                    }
                )
        else:
            heading_paths = _collect_heading_paths(section)
            result.append(
                {
                    "summary": _heading_paths_to_summary(
                        section,
                        outline,
                        document_title_hint=document_title_hint,
                    ),
                    "content": (
                        f"{_section_heading_line(section)}{str(section.get('content') or '')}".strip()
                        if section.get("heading")
                        else str(section.get("content") or "").strip()
                    ),
                    "titlePathParts": _build_title_path_parts(
                        section,
                        outline,
                        document_title_hint=document_title_hint,
                    ),
                    "_headingPaths": heading_paths,
                }
            )

    return result














def split_markdown(
    markdown_text: str,
    min_split_length: int,
    max_split_length: int,
    *,
    document_title_hint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    outline = extract_outline(markdown_text)
    sections = split_by_headings(markdown_text, outline)
    processed = process_sections(
        sections,
        outline,
        min_split_length,
        max_split_length,
        document_title_hint=document_title_hint,
    )
    result: List[Dict[str, Any]] = []
    for item in processed:
        summary = str(item.get("summary") or "")
        content = str(item.get("content") or "")
        result.append(
            {
                "result": _resolve_markdown_banner(summary, content),
                "summary": summary,
                "content": content,
                "titlePathParts": list(item.get("titlePathParts") or []),
            }
        )
    return result


def combine_markdown(split_result: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in split_result:
        summary = str(item.get("summary") or "")
        content = str(item.get("content") or "")
        parts.append(_resolve_markdown_banner(summary, content))
    return "\n\n---\n\n".join(parts)


def _ensure_directory_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_to_separate_files(
    split_result: Sequence[Dict[str, Any]],
    base_filename: str,
) -> Dict[str, Any]:
    base_path = Path(base_filename)
    filename_without_ext = base_path.stem
    output_dir = base_path.parent / f"{filename_without_ext}_parts"
    _ensure_directory_exists(output_dir)

    count = 0
    for index, item in enumerate(split_result, start=1):
        summary = str(item.get("summary") or "")
        content = str(item.get("content") or "")
        output_file = output_dir / f"{filename_without_ext}_part{index:03d}.md"
        output_file.write_text(
            _resolve_markdown_banner(summary, content),
            encoding="utf-8",
        )
        count += 1
    return {"outputDir": str(output_dir), "count": count}




























def _map_split_result(parts: Sequence[Dict[str, Any]], file_name: str, split_type: str) -> List[Dict[str, Any]]:
    base_name = _path_stem(file_name)
    mapped: List[Dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        content = str(part.get("content") or "").strip()
        mapped.append(
            {
                "name": f"{base_name}-part-{index}",
                "fileName": file_name,
                "content": content,
                "summary": str(part.get("summary") or ""),
                "size": len(content),
                "titlePathParts": list(part.get("titlePathParts") or []),
                "splitType": split_type,
            }
        )
    return mapped


def split_content(
    *,
    content: str,
    file_name: str = "document.md",
    config: Optional[Dict[str, Any]] = None,
    force_heading_correction: bool = False,
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    normalized_config = get_default_config(config)
    requested_split_type = str(normalized_config.get("splitType") or "markdown").strip().lower()
    document_title_hint = _path_stem(file_name)
    normalized_content = str(content or "")
    correction_enabled = bool(normalized_config.get("markdownHeadingCorrectionEnabled", True))
    heading_correction_report: Dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "reason": "not_markdown_mode",
        "heading_count": 0,
        "changed_count": 0,
        "before_levels": [],
        "after_levels": [],
    }
    if requested_split_type == "markdown" and normalized_content.strip() and correction_enabled:
        try:
            client = OpenAI(
                api_key=CONFIG["api_key"],
                base_url=CONFIG["base_url"],
            )
            normalized_content, heading_correction_report = correct_markdown_heading_levels(
                normalized_content,
                client=client,
                model=CONFIG["model"],
                request_timeout=int(CONFIG.get("request_timeout", 120) or 120),
                original_filename=file_name,
                force_enable=force_heading_correction,
                debug_writer=debug_writer,
            )
        except Exception as exc:
            heading_correction_report = {
                "enabled": True,
                "applied": False,
                "reason": f"correction_failed: {str(exc)}",
                "heading_count": 0,
                "changed_count": 0,
                "before_levels": [],
                "after_levels": [],
            }
            if debug_writer:
                debug_writer(
                    {
                        "event": "markdown_heading_correction_failed",
                        "error": str(exc),
                        "file_name": file_name,
                    }
                )
    elif requested_split_type == "markdown" and not correction_enabled:
        heading_correction_report["reason"] = "disabled_by_config"

    toc_nested = extract_table_of_contents(normalized_content)
    toc_flat = extract_table_of_contents(normalized_content, {"flatList": True})
    toc_markdown = toc_to_markdown(toc_nested, {"isNested": True})
    toc_markdown_flat = toc_to_markdown(toc_flat, {"isNested": False})
    manual_split_points = (
        normalize_split_points(list(normalized_config.get("manualSplitPoints") or []))
        if normalized_config.get("manualSplitPoints")
        else []
    )
    manual_split_preview = (
        preview_split_points(content, manual_split_points) if manual_split_points else []
    )
    effective_split_type = "manual" if manual_split_points else requested_split_type

    if not manual_split_points and requested_split_type not in SUPPORTED_SPLIT_TYPES:
        raise EasyDatasetChunkingError(
            "unsupported_split_type",
            f"Unsupported splitType: {requested_split_type}",
        )

    chunks: List[Dict[str, Any]]
    if manual_split_points:
        split_result = manual_split(
            content=normalized_content,
            file_name=file_name,
            split_points=manual_split_points,
        )
        chunks = _map_split_result(split_result, file_name, effective_split_type)
    elif requested_split_type == "text":
        split_result = _split_text_mode(
            normalized_content,
            str(normalized_config.get("separator") or "\n\n"),
            max(1, int(normalized_config.get("chunkSize") or 1)),
            max(0, int(normalized_config.get("chunkOverlap") or 0)),
        )
        chunks = [
            {
                "name": f"{document_title_hint}-part-{index}",
                "fileName": file_name,
                "content": part,
                "summary": "",
                "size": len(part),
                "titlePathParts": [document_title_hint, f"Part {index}"],
                "splitType": requested_split_type,
            }
            for index, part in enumerate(split_result, start=1)
            if str(part or "").strip()
        ]
    elif requested_split_type == "token":
        split_result = _split_token_mode(
            normalized_content,
            max(1, int(normalized_config.get("chunkSize") or 1)),
            max(0, int(normalized_config.get("chunkOverlap") or 0)),
        )
        chunks = [
            {
                "name": f"{document_title_hint}-part-{index}",
                "fileName": file_name,
                "content": part,
                "summary": "",
                "size": len(part),
                "titlePathParts": [document_title_hint, f"Part {index}"],
                "splitType": requested_split_type,
            }
            for index, part in enumerate(split_result, start=1)
            if str(part or "").strip()
        ]
    elif requested_split_type == "code":
        split_result = _split_code_mode(
            normalized_content,
            str(normalized_config.get("splitLanguage") or "js"),
            max(1, int(normalized_config.get("chunkSize") or 1)),
            max(0, int(normalized_config.get("chunkOverlap") or 0)),
        )
        chunks = [
            {
                "name": f"{document_title_hint}-part-{index}",
                "fileName": file_name,
                "content": part,
                "summary": "",
                "size": len(part),
                "titlePathParts": [document_title_hint, f"Part {index}"],
                "splitType": requested_split_type,
            }
            for index, part in enumerate(split_result, start=1)
            if str(part or "").strip()
        ]
    elif requested_split_type == "recursive":
        split_result = _split_recursive_mode(
            normalized_content,
            list(normalized_config.get("separators") or []),
            max(1, int(normalized_config.get("chunkSize") or 1)),
            max(0, int(normalized_config.get("chunkOverlap") or 0)),
        )
        chunks = [
            {
                "name": f"{document_title_hint}-part-{index}",
                "fileName": file_name,
                "content": part,
                "summary": "",
                "size": len(part),
                "titlePathParts": [document_title_hint, f"Part {index}"],
                "splitType": requested_split_type,
            }
            for index, part in enumerate(split_result, start=1)
            if str(part or "").strip()
        ]
    elif requested_split_type == "custom":
        split_result = [
            str(part).strip()
            for part in str(normalized_content or "").split(str(normalized_config.get("customSeparator") or "---"))
            if str(part).strip()
        ]
        chunks = [
            {
                "name": f"{document_title_hint}-part-{index}",
                "fileName": file_name,
                "content": part,
                "summary": "",
                "size": len(part),
                "titlePathParts": [document_title_hint, f"Part {index}"],
                "splitType": requested_split_type,
            }
            for index, part in enumerate(split_result, start=1)
        ]
    else:
        split_result = split_markdown(
            normalized_content,
            max(1, int(normalized_config.get("textSplitMinLength") or 1)),
            max(1, int(normalized_config.get("textSplitMaxLength") or 1)),
            document_title_hint=document_title_hint,
        )
        chunks = _map_split_result(split_result, file_name, requested_split_type)

    return {
        "fileName": file_name,
        "config": normalized_config,
        "requestedSplitType": requested_split_type,
        "effectiveSplitType": effective_split_type,
        "manualSplitPoints": manual_split_points,
        "manualSplitPreview": manual_split_preview,
        "totalChunks": len(chunks),
        "chunks": chunks,
        "tocJson": toc_nested,
        "tocFlat": toc_flat,
        "tocNested": toc_nested,
        "tocMarkdown": toc_markdown,
        "tocMarkdownFlat": toc_markdown_flat,
        "normalizedContent": normalized_content,
        "headingCorrection": heading_correction_report,
    }


def split_file(*, file_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    preprocessed = preprocess_file(file_path)
    split_result = split_content(
        content=str(preprocessed.get("content") or ""),
        file_name=str(preprocessed.get("normalizedFileName") or preprocessed.get("fileName") or "document.md"),
        config=config,
    )
    return {
        "preprocessed": preprocessed,
        **split_result,
    }


def _resolve_doc_title(split_result: Dict[str, Any], original_filename: str) -> str:
    toc_json = split_result.get("tocJson") or []
    if isinstance(toc_json, list):
        for item in toc_json:
            if not isinstance(item, dict):
                continue
            if int(item.get("level") or 0) == 1:
                title = str(item.get("title") or "").strip()
                if title:
                    return title
    return _path_stem(original_filename)


def _normalize_title_parts(
    raw_parts: Sequence[str],
    *,
    doc_title: str,
) -> List[str]:
    parts = [str(part).strip() for part in raw_parts if str(part).strip()]
    if parts:
        match = _RE_PART_SUFFIX.match(parts[-1])
        if match:
            head = str(match.group("head") or "").strip()
            part = f"Part {str(match.group('part') or '').strip()}"
            parts[-1] = head
            if part:
                parts.append(part)
    if doc_title:
        if not parts:
            parts = [doc_title, "Part 1"]
        elif not _same_text(parts[0], doc_title):
            parts = [doc_title] + parts
    return [part for part in parts if part]


def _build_index_path(title_parts: Sequence[str], state: Dict[Tuple[str, str], int]) -> str:
    numeric_parts: List[str] = []
    current_path = ""
    for title in title_parts:
        key = (current_path, title)
        if key not in state:
            siblings = [value for (parent, _child), value in state.items() if parent == current_path]
            state[key] = max(siblings, default=0) + 1
        numeric_parts.append(str(state[key]))
        current_path = ".".join(numeric_parts)
    return current_path


def _metrics_summary(chunks_meta: Sequence[Dict[str, Any]], chunk_size: int) -> Dict[str, Any]:
    texts = [str(chunk.get("text") or "") for chunk in chunks_meta]
    hard_max = max(int(chunk_size) * 2, int(chunk_size))
    short_chunks = sum(1 for text in texts if len(text) < 40)
    long_chunks = sum(1 for text in texts if len(text) > hard_max)
    average = round(sum(len(text) for text in texts) / len(texts), 2) if texts else 0.0
    return {
        "chunks": len(chunks_meta),
        "short_chunks_lt_40": short_chunks,
        "long_chunks_gt_hard_max": long_chunks,
        "hard_max_chars": hard_max,
        "avg_chunk_chars": average,
    }


def build_tree_chunks_easy_dataset(
    text: str,
    *,
    chunk_size: int,
    original_filename: str,
    task_id: str,
    doc_id: str,
    prefix_max_depth: int = 4,
    title_sep: str = ">",
    debug_writer: Optional[Callable[[Dict[str, Any]], None]] = None,
    split_type: Optional[str] = None,
    text_split_min_length: Optional[int] = None,
    text_split_max_length: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separator: Optional[str] = None,
    separators: Optional[Sequence[str]] = None,
    split_language: Optional[str] = None,
    custom_separator: Optional[str] = None,
    manual_split_points: Optional[Sequence[Dict[str, Any]]] = None,
    force_heading_correction: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]:
    raw_text = str(text or "")
    if not raw_text.strip():
        raise EasyDatasetChunkingError("empty_text", "Input text is empty")

    started_at = time.time()
    resolved_chunk_size = max(1, int(chunk_size))
    markdown_min = (
        max(120, int(text_split_min_length))
        if text_split_min_length is not None
        else max(120, int(resolved_chunk_size * 0.75))
    )
    markdown_max = (
        max(markdown_min, int(text_split_max_length))
        if text_split_max_length is not None
        else max(markdown_min, resolved_chunk_size)
    )
    effective_overlap = (
        max(0, int(chunk_overlap))
        if chunk_overlap is not None
        else min(max(0, resolved_chunk_size // 6), max(0, resolved_chunk_size - 1))
    )
    adapter_config = get_default_config(
        {
            "splitType": split_type,
            "textSplitMinLength": markdown_min,
            "textSplitMaxLength": markdown_max,
            "chunkSize": resolved_chunk_size,
            "chunkOverlap": effective_overlap,
            "separator": separator,
            "separators": list(separators) if separators is not None else None,
            "splitLanguage": split_language,
            "customSeparator": custom_separator,
            "manualSplitPoints": list(manual_split_points) if manual_split_points is not None else None,
        }
    )

    if debug_writer:
        debug_writer(
            {
                "event": "easy_dataset_chunking_start",
                "engine_version": ENGINE_VERSION,
                "original_filename": original_filename,
                "config": adapter_config,
            }
        )

    split_result = split_content(
        content=raw_text,
        file_name=_normalize_filename_to_markdown(original_filename or "document.txt", ".txt"),
        config=adapter_config,
        force_heading_correction=force_heading_correction,
        debug_writer=debug_writer,
    )
    raw_chunks = list(split_result.get("chunks") or [])
    if not raw_chunks:
        raise EasyDatasetChunkingError("no_chunks", "Chunk splitter produced no chunks")

    doc_title = _resolve_doc_title(split_result, original_filename)
    effective_split_type = str(
        split_result.get("effectiveSplitType")
        or adapter_config.get("splitType")
        or "markdown"
    )
    requested_split_type = str(
        split_result.get("requestedSplitType")
        or adapter_config.get("splitType")
        or "markdown"
    )
    title_index_state: Dict[Tuple[str, str], int] = {}
    chunks_for_llm: List[str] = []
    chunks_meta: List[Dict[str, Any]] = []
    now_ts = int(time.time())
    emitted_chunk_index = 0

    for chunk in raw_chunks:
        chunk_text = str(chunk.get("content") or "").strip()
        if not chunk_text:
            continue
        emitted_chunk_index += 1
        title_parts = _normalize_title_parts(
            list(chunk.get("titlePathParts") or []),
            doc_title=doc_title,
        )
        index_path = _build_index_path(title_parts, title_index_state)
        parent_index_path = ".".join(index_path.split(".")[:-1]) if "." in index_path else ""
        root_index_path = index_path.split(".")[0] if index_path else ""
        title_path = title_sep.join(title_parts)
        ancestor_parts = title_parts[:-1] if len(title_parts) >= 2 else []
        prefix = title_sep.join(ancestor_parts[-max(0, int(prefix_max_depth)) :]).strip()
        text_for_embedding = f"{prefix}\n{chunk_text}".strip() if prefix else chunk_text
        chunk_id = sha1(
            f"{task_id}|||{doc_id}|||{index_path}|||{title_path}|||{chunk_text}".encode("utf-8")
        ).hexdigest()
        chunks_for_llm.append(text_for_embedding)
        chunks_meta.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "task_id": task_id,
                "original_filename": str(original_filename or "").strip() or "input.txt",
                "chunk_index": emitted_chunk_index,
                "index_path": index_path,
                "title_path": title_path,
                "parent_index_path": parent_index_path,
                "root_index_path": root_index_path,
                "level": len([part for part in index_path.split(".") if part.strip()]),
                "is_leaf": True,
                "text": chunk_text,
                "text_for_embedding": text_for_embedding,
                "created_at": now_ts,
                "engine_version": ENGINE_VERSION,
                "chunk_kind": f"{effective_split_type}_chunk",
                "path_summary": str(chunk.get("summary") or ""),
                "split_type": effective_split_type,
            }
        )

    if not chunks_meta:
        raise EasyDatasetChunkingError("no_chunks", "Chunk splitter produced only empty chunks")

    metrics_summary = _metrics_summary(chunks_meta, resolved_chunk_size)
    manual_split_points_used = list(split_result.get("manualSplitPoints") or [])
    report = {
        "engine_version": ENGINE_VERSION,
        "mode": "easy_dataset_python",
        "doc_title": doc_title,
        "doc_profile": (
            "easy_dataset_manual"
            if effective_split_type == "manual"
            else "easy_dataset_markdown"
            if effective_split_type == "markdown"
            else "easy_dataset_flat"
        ),
        "validation_passed": True,
        "hard_fail_reason": None,
        "metrics_summary": metrics_summary,
        "chunks": len(chunks_meta),
        "text_chars_total": len(raw_text),
        "duration_seconds": round(time.time() - started_at, 4),
        "config": adapter_config,
        "requested_split_type": requested_split_type,
        "effective_split_type": effective_split_type,
        "toc_items": len(list(split_result.get("tocFlat") or [])),
        "toc_markdown": str(split_result.get("tocMarkdown") or ""),
        "toc_markdown_flat": str(split_result.get("tocMarkdownFlat") or ""),
        "toc_flat": list(split_result.get("tocFlat") or []),
        "toc_nested": list(split_result.get("tocNested") or []),
        "heading_correction": dict(split_result.get("headingCorrection") or {}),
        "manual_split_enabled": bool(manual_split_points_used),
        "manual_split_points": manual_split_points_used,
        "manual_split_preview": list(split_result.get("manualSplitPreview") or []),
    }

    if debug_writer:
        debug_writer(
            {
                "event": "easy_dataset_chunking_result",
                "engine_version": ENGINE_VERSION,
                "chunk_count": len(chunks_meta),
                "report": {
                    "mode": report["mode"],
                    "doc_title": report["doc_title"],
                    "effective_split_type": report["effective_split_type"],
                    "metrics_summary": report["metrics_summary"],
                },
                "sample_titles": [chunk.get("title_path") for chunk in chunks_meta[:5]],
            }
        )

    return chunks_for_llm, chunks_meta, report


getDefaultConfig = get_default_config
getCapabilities = get_capabilities
preprocessFile = preprocess_file
splitFile = split_file
splitContent = split_content
splitMarkdown = split_markdown
extractOutline = extract_outline
splitByHeadings = split_by_headings
processSections = process_sections
splitLongSection = split_long_section
generateEnhancedSummary = generate_enhanced_summary
extractTableOfContents = extract_table_of_contents
tocToMarkdown = toc_to_markdown
combineMarkdown = combine_markdown
saveToSeparateFiles = save_to_separate_files
previewSplitPoints = preview_split_points
normalizeSplitPoints = normalize_split_points
manualSplit = manual_split


__all__ = [
    "ENGINE_VERSION",
    "DEFAULT_CONFIG",
    "ORIGINAL_PROJECT_INPUT_EXTENSIONS",
    "STANDALONE_INPUT_EXTENSIONS",
    "EasyDatasetChunkingError",
    "build_tree_chunks_easy_dataset",
    "get_default_config",
    "get_capabilities",
    "preprocess_file",
    "split_file",
    "split_content",
    "split_markdown",
    "extract_outline",
    "split_by_headings",
    "process_sections",
    "split_long_section",
    "generate_enhanced_summary",
    "extract_table_of_contents",
    "toc_to_markdown",
    "combine_markdown",
    "save_to_separate_files",
    "preview_split_points",
    "normalize_split_points",
    "manual_split",
    "getDefaultConfig",
    "getCapabilities",
    "preprocessFile",
    "splitFile",
    "splitContent",
    "splitMarkdown",
    "extractOutline",
    "splitByHeadings",
    "processSections",
    "splitLongSection",
    "generateEnhancedSummary",
    "extractTableOfContents",
    "tocToMarkdown",
    "combineMarkdown",
    "saveToSeparateFiles",
    "previewSplitPoints",
    "normalizeSplitPoints",
    "manualSplit",
]
