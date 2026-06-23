# 文件作用：提供结构化文本切分、标题提取和 chunk 报告能力。
# 关联说明：提供轻量文本切分和标题抽取，供 __init__ facade 和 tree_chunks 体系互补使用。

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


_FULLWIDTH_SPACE = "\u3000"
_IDEOGRAPHIC_COMMA = "\u3001"  # 、
_FULLWIDTH_COLON = "\uff1a"  # ：
_FULLWIDTH_LPAREN = "\uff08"  # （
_FULLWIDTH_RPAREN = "\uff09"  # ）

_END_PUNCT = set("。！？；.!?;")

_CN_NUMS = "一二三四五六七八九十"
_CN_NUMS_EXT = _CN_NUMS + "百千万"

_RE_PAGE_MARKER = re.compile(r"^[\-\u2014\u2013]{0,3}\s*\d{1,4}\s*[\-\u2014\u2013]{1,3}$")
_RE_DATE_YYYY_MM_DD = re.compile(r"\b20\d{2}-\d{1,2}-\d{1,2}\b")
_RE_DATE_YYYY_MM = re.compile(r"\b20\d{2}-\d{1,2}\b")
_RE_DATE_MM_DD = re.compile(r"^\d{1,2}-\d{1,2}$")
_RE_JUNK_NUM = re.compile(r"^\d{1,3}$")

# Top-level structure boundaries
_RE_MD_HEADER = re.compile(r"^#{1,6}\s+")
_RE_RULE_EN = re.compile(r"^RULE\b", flags=re.IGNORECASE)
_RE_ARTICLE_EN = re.compile(r"^(Article|Section|Chapter)\s+\d+\b", flags=re.IGNORECASE)
_RE_EN_SUBSECTION = re.compile(r"^\d+\.\d+\b")

_RE_CN_CLAUSE = re.compile(rf"^第[{_CN_NUMS_EXT}0-9]+\s*[章节条款目]")
_RE_CN_MAIN_SECTION = re.compile(rf"^(?P<idx>[{_CN_NUMS_EXT}]+)\s*{_IDEOGRAPHIC_COMMA}")
_RE_SPECIAL_TITLES = re.compile(r"^(目录|目\s*录|附则|附录|释义|定义|术语)$")
_RE_APPENDIX_NUM = re.compile(r"^附件\s*(?P<num>\d+)\b")
_RE_APPENDIX_COLON = re.compile(rf"^附件\s*[{_FULLWIDTH_COLON}:]")

# Sub-level boundaries (within a top-level segment)
_RE_CN_PAREN_ITEM = re.compile(rf"^{_FULLWIDTH_LPAREN}(?P<idx>[{_CN_NUMS_EXT}0-9]+){_FULLWIDTH_RPAREN}")
_RE_ARABIC_DOT_ITEM = re.compile(r"^(?P<idx>\d+)\s*[\.．]\s*")
_RE_ARABIC_COMMA_ITEM = re.compile(r"^(?P<idx>\d+)\s*[、]\s*")
_RE_ARABIC_PAREN_ITEM = re.compile(r"^[（(]?(?P<idx>\d+)[)）]\s*")

_RE_SENTENCES = re.compile(r".+?[。！？；.!?;]+|.+?$", flags=re.DOTALL)


@dataclass(frozen=True)
class _Segment:
    heading: Optional[str]
    body_lines: List[str]


def _normalize_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace(_FULLWIDTH_SPACE, " ")


def _is_top_boundary(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    return bool(
        _RE_MD_HEADER.match(s)
        or _RE_RULE_EN.match(s)
        or _RE_ARTICLE_EN.match(s)
        or _RE_CN_CLAUSE.match(s)
        or _RE_CN_MAIN_SECTION.match(s)
        or _RE_APPENDIX_NUM.match(s)
        or _RE_APPENDIX_COLON.match(s)
        or _RE_SPECIAL_TITLES.match(s)
    )


def _should_promote_remainder_to_body(kind: str, remainder: str) -> bool:
    rem = str(remainder or "").strip()
    if not rem:
        return False
    if any(ch in rem for ch in _END_PUNCT):
        return True
    if "\n" in rem:
        return True
    if kind in {"cn_clause", "en_article"} and len(rem) >= 40:
        return True
    return False


def _extract_top_heading(line: str) -> tuple[Optional[str], Optional[str]]:
    s = (line or "").strip()
    if not s:
        return None, None

    md = _RE_MD_HEADER.match(s)
    if md:
        return s, None

    en_article = _RE_ARTICLE_EN.match(s)
    if en_article:
        prefix = s[: en_article.end()].strip()
        remainder = s[en_article.end() :].strip()
        if prefix and _should_promote_remainder_to_body("en_article", remainder):
            return prefix, remainder or None
        return s, None

    if _RE_RULE_EN.match(s):
        return s, None

    cn_clause = _RE_CN_CLAUSE.match(s)
    if cn_clause:
        prefix = s[: cn_clause.end()].strip()
        remainder = s[cn_clause.end() :].strip()
        if prefix and _should_promote_remainder_to_body("cn_clause", remainder):
            return prefix, remainder or None
        return s, None

    cn_main = _RE_CN_MAIN_SECTION.match(s)
    if cn_main:
        return s, None

    appendix_num = _RE_APPENDIX_NUM.match(s)
    if appendix_num:
        return s, None

    appendix_colon = _RE_APPENDIX_COLON.match(s)
    if appendix_colon:
        return s, None

    if _RE_SPECIAL_TITLES.match(s):
        return s, None

    return None, None


def _is_sub_boundary(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    return bool(
        _RE_EN_SUBSECTION.match(s)
        or _RE_CN_PAREN_ITEM.match(s)
        or _RE_ARABIC_DOT_ITEM.match(s)
        or _RE_ARABIC_COMMA_ITEM.match(s)
        or _RE_ARABIC_PAREN_ITEM.match(s)
    )


def _is_any_boundary(line: str) -> bool:
    return _is_top_boundary(line) or _is_sub_boundary(line)


def _looks_like_table_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if "\t" in s or "|" in s:
        return True
    # Multiple wide gaps often indicate tabular layout from OCR/PDF.
    if re.search(r"\s{3,}", s) and len(s) <= 120:
        return True
    return False


def _compress_blank_lines(lines: Iterable[str]) -> List[str]:
    out: List[str] = []
    last_blank = False
    for line in lines:
        s = (line or "").strip()
        if not s:
            if not last_blank:
                out.append("")
            last_blank = True
            continue
        out.append(s)
        last_blank = False
    return out


def _drop_noise_lines(lines: List[str]) -> List[str]:
    """
    Best-effort OCR cleanup:
    - Drop page markers (— 1 —, -1—, 15—, ...)
    - Drop extremely repetitive short fragments (often headers/footers)
    - Drop repeated date-like watermark fragments (e.g. 2025-12-09)
    """

    def key(s: str) -> str:
        return re.sub(r"\s+", "", s)

    candidates = [ln for ln in lines if ln.strip() and not _is_any_boundary(ln)]
    keyed = [key(ln) for ln in candidates]
    counts = Counter(keyed)

    noise_keys: set[str] = set()
    for k, c in counts.items():
        if c >= 6 and len(k) <= 80:
            noise_keys.add(k)
            continue
        if c >= 4 and len(k) <= 20:
            noise_keys.add(k)

    cleaned: List[str] = []
    for raw in lines:
        s = (raw or "").strip()
        if not s:
            cleaned.append("")
            continue
        if _RE_PAGE_MARKER.match(s):
            cleaned.append("")
            continue

        if not _is_any_boundary(s):
            k = key(s)
            if k in noise_keys:
                cleaned.append("")
                continue
            if _RE_DATE_MM_DD.match(s):
                cleaned.append("")
                continue
            if _RE_JUNK_NUM.match(s) and len(s) <= 2:
                cleaned.append("")
                continue
            # Watermark/footer lines often contain repeated date stamps.
            if len(_RE_DATE_YYYY_MM_DD.findall(s)) >= 2:
                cleaned.append("")
                continue
            if _RE_DATE_YYYY_MM_DD.search(s) and len(s) <= 120:
                cleaned.append("")
                continue
            if _RE_DATE_YYYY_MM.search(s) and len(s) <= 60:
                cleaned.append("")
                continue
        cleaned.append(s)
    return _compress_blank_lines(cleaned)


def _should_merge_hard_wrapped_lines(lines: List[str]) -> bool:
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) < 30:
        return False

    blank_ratio = (len(lines) - len(non_empty)) / max(1, len(lines))
    if blank_ratio >= 0.08:
        # Already has stable paragraph spacing.
        return False

    avg_len = sum(len(ln) for ln in non_empty) / max(1, len(non_empty))
    if avg_len > 80:
        return False

    punct_ratio = sum(1 for ln in non_empty if ln[-1] in _END_PUNCT) / max(1, len(non_empty))
    if punct_ratio > 0.6:
        return False

    return True


def _merge_hard_wrapped_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        s = (raw or "").strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue

        cur_is_boundary = _is_any_boundary(s)
        prev = out[-1] if out else ""
        prev_is_boundary = _is_any_boundary(prev)

        if not out or prev == "" or cur_is_boundary or prev_is_boundary:
            out.append(s)
            continue

        if _looks_like_table_line(prev) or _looks_like_table_line(s):
            out.append(s)
            continue

        if prev and prev[-1] in _END_PUNCT:
            out.append(s)
            continue

        # Merge into previous line without adding extra punctuation.
        out[-1] = prev + s

    return _compress_blank_lines(out)


def _split_into_segments(lines: List[str]) -> List[_Segment]:
    segments: List[_Segment] = []
    current_heading: Optional[str] = None
    current_body: List[str] = []

    for raw in lines:
        s = (raw or "").strip()
        if not s:
            if current_body and current_body[-1] != "":
                current_body.append("")
            continue

        heading, inline_body = _extract_top_heading(s)
        if heading:
            if current_heading is not None or any(x.strip() for x in current_body):
                segments.append(_Segment(heading=current_heading, body_lines=_compress_blank_lines(current_body)))
            current_heading = _normalize_heading(heading)
            current_body = [inline_body] if inline_body else []
            continue

        current_body.append(s)

    if current_heading is not None or any(x.strip() for x in current_body):
        segments.append(_Segment(heading=current_heading, body_lines=_compress_blank_lines(current_body)))
    return segments


def _normalize_heading(heading: str) -> str:
    s = (heading or "").strip()
    if not s:
        return s
    # Normalize "一、 关于" -> "一、关于"
    s = re.sub(rf"\s*{_IDEOGRAPHIC_COMMA}\s*", _IDEOGRAPHIC_COMMA, s)
    # Normalize "附件 1" -> "附件 1" (keep a single space for readability)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _split_segment_into_blocks(seg: _Segment) -> List[str]:
    blocks: List[str] = []
    buf: List[str] = []

    for raw in seg.body_lines:
        s = (raw or "").strip()
        if not s:
            if buf and buf[-1] != "":
                buf.append("")
            continue

        if _is_sub_boundary(s) and any(x.strip() for x in buf):
            blocks.append(_join_block(buf))
            buf = [s]
            continue
        buf.append(s)

    if any(x.strip() for x in buf):
        blocks.append(_join_block(buf))
    return [b for b in (blk.strip() for blk in blocks) if b]


def _join_block(lines: List[str]) -> str:
    # Treat empty strings as paragraph separators, but keep output compact.
    out: List[str] = []
    last_blank = False
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            if out and not last_blank:
                out.append("")
            last_blank = True
            continue
        out.append(s)
        last_blank = False
    return "\n".join(out).strip()


def _split_block_to_fit(block: str, max_len: int) -> List[str]:
    """
    Split a single oversized block into smaller pieces.
    Prefer sentence boundaries; fall back to fixed windows.
    """
    text = (block or "").strip()
    if not text:
        return []
    if max_len <= 0:
        return [text]
    if len(text) <= max_len:
        return [text]

    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    sentences = [s.strip() for s in _RE_SENTENCES.findall(flat) if s.strip()]
    if not sentences or len(sentences) == 1:
        return [flat[i : i + max_len] for i in range(0, len(flat), max_len)]

    pieces: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for sent in sentences:
        if len(sent) > max_len:
            if buf:
                pieces.append(" ".join(buf).strip())
                buf, buf_len = [], 0
            pieces.extend([sent[i : i + max_len] for i in range(0, len(sent), max_len)])
            continue

        if buf and buf_len + len(sent) + 1 > max_len:
            pieces.append(" ".join(buf).strip())
            buf, buf_len = [], 0

        buf.append(sent)
        buf_len += len(sent) + (1 if buf_len else 0)

    if buf:
        pieces.append(" ".join(buf).strip())
    return [p for p in pieces if p]


def _pack_segment(
    seg: _Segment,
    *,
    chunk_size: int,
    max_chunk_size: int,
) -> List[str]:
    heading = (seg.heading or "").strip() or None
    blocks = _split_segment_into_blocks(seg)
    if not blocks:
        return []

    prefix = f"{heading}\n" if heading else ""
    prefix_len = len(prefix)
    effective_target = max(1, chunk_size - prefix_len)
    effective_max = max(1, max_chunk_size - prefix_len)

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for block in blocks:
        if not block:
            continue

        pieces = (
            _split_block_to_fit(block, effective_max)
            if len(block) > effective_max
            else [block]
        )
        for piece in pieces:
            if not piece:
                continue
            piece_len = len(piece)
            if buf and buf_len + piece_len > effective_target:
                chunks.append(prefix + "\n".join(buf).strip())
                buf, buf_len = [], 0
            buf.append(piece)
            buf_len += piece_len

    if buf:
        chunks.append(prefix + "\n".join(buf).strip())
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, chunk_size: int) -> List[str]:
    """
    Split long text into chunks while preserving structure as much as possible:
    - Top-level: chapter/section/article/appendix markers
    - Within a segment: paragraph/list items
    - Oversized blocks: sentence fallback

    The budget is character-based (len(text)) to stay dependency-free.
    """
    if chunk_size <= 0:
        return []

    normalized = _normalize_text(text)
    if not normalized.strip():
        return []

    raw_lines = normalized.split("\n")
    lines = _compress_blank_lines(raw_lines)
    lines = _drop_noise_lines(lines)
    if _should_merge_hard_wrapped_lines(lines):
        lines = _merge_hard_wrapped_lines(lines)

    segments = _split_into_segments(lines)
    if not segments:
        # Fallback: treat everything as one segment without heading.
        segments = [_Segment(heading=None, body_lines=lines)]

    max_chunk_size = max(chunk_size, int(chunk_size * 1.25))
    all_chunks: List[str] = []
    for seg in segments:
        all_chunks.extend(
            _pack_segment(seg, chunk_size=chunk_size, max_chunk_size=max_chunk_size)
        )

    # Last-resort fallback: ensure no empty output for non-empty input.
    if not all_chunks:
        compact = re.sub(r"\s+", " ", normalized).strip()
        return [compact[i : i + chunk_size] for i in range(0, len(compact), chunk_size)]
    return all_chunks


def extract_chunk_heading(chunk_text: str) -> Optional[str]:
    """
    Best-effort heading extraction from a chunk.
    Returns the first non-empty line if it looks like a structure boundary.
    """
    for raw in (chunk_text or "").splitlines():
        s = (raw or "").strip()
        if not s:
            continue
        if _is_any_boundary(s):
            return s
        return None
    return None


def build_chunk_report(
    chunks: List[str],
    *,
    original_filename: Optional[str] = None,
    qa_items: Optional[Iterable[Dict[str, Any]]] = None,
    chunk_meta_list: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a per-chunk report for debugging/analysis.

    If qa_items are provided and each item includes `chunk_index`,
    the report will include `qa_count` for each chunk.
    """
    counts: Dict[int, int] = {}
    if qa_items is not None:
        for item in qa_items:
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("chunk_index")
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            if idx <= 0:
                continue
            counts[idx] = counts.get(idx, 0) + 1

    report: List[Dict[str, Any]] = []
    meta_by_index: Dict[int, Dict[str, Any]] = {}
    if chunk_meta_list is not None:
        for meta in chunk_meta_list:
            if not isinstance(meta, dict):
                continue
            try:
                meta_index = int(meta.get("chunk_index") or 0)
            except Exception:
                continue
            if meta_index <= 0:
                continue
            meta_by_index[meta_index] = meta
    for idx, chunk_text in enumerate(chunks or [], start=1):
        entry: Dict[str, Any] = {
            "chunk_index": idx,
            "char_len": len(chunk_text or ""),
            "heading": extract_chunk_heading(chunk_text),
            "text": chunk_text or "",
        }
        meta = meta_by_index.get(idx) or {}
        if original_filename:
            entry["original_filename"] = original_filename
        if meta:
            for key in (
                "chunk_id",
                "index_path",
                "title_path",
                "parent_index_path",
                "root_index_path",
                "split_type",
            ):
                value = meta.get(key)
                if value not in (None, ""):
                    entry[key] = value
            path_summary = str(meta.get("path_summary") or "").strip()
            if path_summary:
                entry["path_summary"] = path_summary
        if qa_items is not None:
            entry["qa_count"] = counts.get(idx, 0)
        report.append(entry)
    return report
