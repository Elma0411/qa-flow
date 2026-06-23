# 文件作用：从文档文本构建分类所需的标题和正文画像。
# 关联说明：被 classifier_core 调用，把原始文档变成可匹配的画像。

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千0-9]+条")
CHAPTER_PATTERN = re.compile(r"^第[一二三四五六七八九十百千0-9]+章")
ARTIFACT_LINE_PATTERN = re.compile(r"^[A-Za-z0-9_.:/\\-]{1,40}$")
PAGE_MARKER_PATTERN = re.compile(r"^[-—‐\s]*\d+[-—‐\s]*$")
DOC_CODE_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9〔〕\[\]（）()]+[〔\[](19|20)\d{2}[〕\]][A-Za-z0-9一二三四五六七八九十]+号$")
STANDARD_CODE_PATTERN = re.compile(r"^(ICS|GB/?T?|DL/?T|Q/?GDW|DB\d{2,3}(?:/T)?|AQ(?:/T)?)", re.IGNORECASE)
STANDARD_PREFIX_PATTERN = re.compile(
    r"^(备案号[:：]?[A-Za-z0-9\- ]*|ICS[0-9.\- ]*|国家电网(?:公司|有限公司)?企业标准|企业标准|Q/GDW[0-9A-Za-z—\-× ]*)+"
)
DATE_PATTERN = re.compile(r"(19|20)\d{2}年\d{1,2}月\d{1,2}日")
DOC_TITLE_SPLIT_PATTERN = re.compile(r"[（(].{0,20}(修订|修正|修订版|修正版).{0,10}[)）]")
ORDER_LINE_PATTERN = re.compile(
    r"^(中华人民共和国国务院令|国务院令|.*公告(?:（第\d+号）)?|备案号[:：].*|规章制度编号[:：].*|制度编号[:：].*)$"
)
WEAK_TITLE_PREFIXES = (
    "规章制度编号",
    "制度编号",
    "下载时间",
    "发布日期",
    "实施日期",
    "备案号",
    "公司内部文件",
    "内部文件",
)
TITLE_KEYWORDS = (
    "条例",
    "办法",
    "规定",
    "规则",
    "规范",
    "导则",
    "标准",
    "通知",
    "通报",
    "公告",
    "公报",
    "意见",
    "方案",
    "报告",
    "批复",
    "决定",
    "细则",
)
STANDARD_PATTERNS = {
    "standard_gb_t": re.compile(r"\bGB/T\s*\d", re.IGNORECASE),
    "standard_gb": re.compile(r"\bGB\s*\d", re.IGNORECASE),
    "standard_dl_t": re.compile(r"\bDL/T\s*\d", re.IGNORECASE),
    "standard_aq": re.compile(r"\bAQ(?:/T)?\s*\d", re.IGNORECASE),
    "standard_db": re.compile(r"\bDB\d{2,3}(?:/T)?\s*\d", re.IGNORECASE),
    "standard_qgdw": re.compile(r"\bQ/GDW\s*\d", re.IGNORECASE),
}
LOG_FIELDS = ("日期", "天气", "值班记录", "事件监测", "保电工作", "资源核查", "日报填报")
IGNORED_ARTIFACT_LINES = {
    "Root Entry",
    "SummaryInformation",
    "DocumentSummaryInformation",
    "ObjectPool",
    "WordDocument",
    "1Table",
    "0Table",
    "Data",
    "CompObj",
    "Normal.dot",
    "Microsoft Office Word",
    "WPS Office",
    "Kingsoft Office",
    "KSOProductBuildVer",
}


@dataclass(frozen=True)
class DocumentProfile:
    source_id: str
    source_name: str
    source_path: str
    text: str
    title: str
    head: str
    signals: tuple[str, ...]

    def scope_text(self, scope: str) -> str:
        if scope == "filename":
            return self.source_name
        if scope == "title":
            return self.title
        if scope == "head":
            return self.head
        if scope == "title_head":
            return f"{self.title}\n{self.head}".strip()
        if scope == "fulltext":
            return self.text
        if scope == "signals":
            return " ".join(self.signals)
        raise KeyError(f"unsupported scope: {scope}")


def _clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    lines = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line in IGNORED_ARTIFACT_LINES:
            continue
        if ARTIFACT_LINE_PATTERN.fullmatch(line) and not any("\u4e00" <= ch <= "\u9fff" for ch in line):
            # Drop short metadata-like English identifiers from degraded OLE extraction.
            continue
        lines.append(line)
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _line_has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _is_weak_title_line(line: str) -> bool:
    if not line:
        return True
    if len(line) <= 2:
        return True
    if PAGE_MARKER_PATTERN.fullmatch(line):
        return True
    if CHAPTER_PATTERN.match(line):
        return True
    if DOC_CODE_PATTERN.fullmatch(line):
        return True
    if STANDARD_CODE_PATTERN.match(line):
        return True
    if any(line.startswith(prefix) for prefix in WEAK_TITLE_PREFIXES):
        return True
    if line in ("国家电网公司企业标准", "国家电网有限公司企业标准", "企业标准"):
        return True
    if line.endswith("企业标准"):
        return True
    if line.startswith(("ICS", "Q/GDW", "GB", "DL/T", "AQ", "DB")):
        return True
    if line.startswith("—") or line.startswith("-"):
        return True
    return False


def _normalize_title_candidate(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.strip())
    if not normalized:
        return normalized

    # Collapse repeated consecutive duplicate title fragments.
    half = len(normalized) // 2
    if len(normalized) % 2 == 0 and normalized[:half] == normalized[half:]:
        normalized = normalized[:half]

    # Collapse "标题(修订)标题" style duplication.
    match = DOC_TITLE_SPLIT_PATTERN.search(normalized)
    if match:
        prefix = normalized[: match.end()]
        suffix = normalized[match.end() :]
        if suffix and (suffix == prefix or suffix == re.sub(DOC_TITLE_SPLIT_PATTERN, "", prefix)):
            normalized = prefix

    chapter_match = re.search(r"第[一二三四五六七八九十百千0-9]+章.*$", normalized)
    if chapter_match:
        normalized = normalized[: chapter_match.start()]

    standard_trimmed = STANDARD_PREFIX_PATTERN.sub("", normalized).strip()
    if standard_trimmed and _line_has_cjk(standard_trimmed) and len(standard_trimmed) >= 6:
        normalized = standard_trimmed

    normalized = re.sub(r"^(备案号[:：][A-Za-z0-9\- ]+)", "", normalized).strip()
    normalized = re.sub(r"(中华人民共和国国务院令|国务院令|.*公告（第\d+号）)$", "", normalized).strip()

    return normalized


def _looks_complete_title(line: str) -> bool:
    if not line:
        return False
    if any(line.endswith(keyword) for keyword in TITLE_KEYWORDS):
        return True
    if line.endswith((")", "）")) and any(keyword in line for keyword in TITLE_KEYWORDS):
        return True
    return False


def _is_agency_line(line: str) -> bool:
    if not line:
        return False
    if any(keyword in line for keyword in TITLE_KEYWORDS):
        return False
    agency_tokens = ("委员会", "人民政府", "人民代表大会", "人大常委会", "公司", "供电公司", "管理局", "司法部", "财政部", "国务院")
    if any(token in line for token in agency_tokens):
        return True
    if line.count("、") >= 1 and not any(ch in line for ch in "《》"):
        return True
    return False


def _is_order_line(line: str) -> bool:
    if not line:
        return False
    if ORDER_LINE_PATTERN.fullmatch(line):
        return True
    if STANDARD_CODE_PATTERN.match(line):
        return True
    if line in ("国家电网公司企业标准", "国家电网有限公司企业标准", "企业标准"):
        return True
    return False


def _should_combine_title_lines(current: str, next_line: str) -> bool:
    if not current or not next_line:
        return False
    if _is_weak_title_line(current):
        return False
    if ARTICLE_PATTERN.match(next_line) or _is_weak_title_line(next_line):
        return False
    if _looks_complete_title(current) and (_is_agency_line(next_line) or _is_order_line(next_line)):
        return False
    if _looks_complete_title(current) and next_line.startswith("关于印发《"):
        return False
    if current.startswith("关于") and _is_agency_line(next_line):
        return False
    return True


def _score_title_line(line: str) -> float:
    if _is_weak_title_line(line):
        return -5.0

    score = 0.0
    line = _normalize_title_candidate(line)
    length = len(line)
    if 4 <= length <= 40:
        score += 2.5
    elif 41 <= length <= 70:
        score += 1.5
    elif length > 90:
        score -= 2.0

    if _line_has_cjk(line):
        score += 2.0
    if not any(ch.isdigit() for ch in line):
        score += 0.5

    for keyword in TITLE_KEYWORDS:
        if keyword in line:
            score += 2.5

    if "关于" in line:
        score += 2.0
    if "文件" in line:
        score += 1.0
    if "第" in line and "条" in line:
        score -= 3.0
    if "附件" in line or "目录" in line or "前言" in line:
        score -= 2.5
    if "发布" in line or "实施" in line:
        score -= 1.5
    if DATE_PATTERN.search(line):
        score -= 1.8
    if "人民代表大会常务委员会" in line and DATE_PATTERN.search(line):
        score -= 3.0
    if line.startswith(("中华人民共和国国务院令", "国务院令", "甘肃省人民代表大会常务委员会公告")):
        score -= 3.0
    if "中华人民共和国国务院令" in line or "国务院令" in line:
        score -= 4.0
    if "关于印发《" in line and "通知" in line and any(keyword in line for keyword in ("条例", "办法", "规定", "规则")):
        score -= 3.5
    if line.endswith("企业标准"):
        score -= 4.0
    if _is_agency_line(line):
        score -= 2.5
    if _is_order_line(line):
        score -= 3.0
    if "公司内部文件" in line or "请注意保密" in line:
        score -= 4.0
    if line.endswith(("。", "；", ";")):
        score -= 3.0
    if line.count("，") + line.count(",") >= 2:
        score -= 3.0
    if any(token in line for token in ("根据", "负责", "要求", "适用于", "工作", "情况")):
        score -= 2.0
    if "《" in line or "》" in line:
        score -= 1.5
    return score


def _pick_title(text: str, source_name: str) -> str:
    stem = Path(source_name).stem
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return stem

    candidates: list[tuple[float, str]] = []
    limit = min(len(lines), 20)
    for index in range(limit):
        line = lines[index]
        if ARTICLE_PATTERN.match(line):
            continue
        position_bonus = max(0.0, 2.5 - index * 0.25)
        normalized_line = _normalize_title_candidate(line)
        score = _score_title_line(normalized_line) + position_bonus
        if score > -5:
            candidates.append((score, normalized_line))

        if index + 1 < limit:
            next_line = lines[index + 1]
            combined = _normalize_title_candidate(f"{normalized_line}{next_line}")
            if len(combined) <= 60 and _should_combine_title_lines(normalized_line, next_line):
                combined_score = _score_title_line(combined) + position_bonus + 0.8
                if combined_score > score:
                    candidates.append((combined_score, combined))

        if index + 2 < limit:
            next_line = lines[index + 1]
            third_line = lines[index + 2]
            combined3 = _normalize_title_candidate(f"{normalized_line}{next_line}{third_line}")
            if (
                len(combined3) <= 70
                and _should_combine_title_lines(normalized_line, next_line)
                and _should_combine_title_lines(_normalize_title_candidate(f"{normalized_line}{next_line}"), third_line)
            ):
                combined3_score = _score_title_line(combined3) + position_bonus + 1.0
                if combined3_score > score:
                    candidates.append((combined3_score, combined3))

    if candidates:
        best_score, best_title = max(candidates, key=lambda item: (item[0], len(item[1])))
        if best_score > 0:
            return best_title
    return stem


def _build_head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _extract_signals(title: str, head: str, text: str, source_name: str) -> list[str]:
    combined = f"{source_name}\n{title}\n{head}"
    signals: list[str] = []

    if len(ARTICLE_PATTERN.findall(text[:5000])) >= 2:
        signals.append("law_style")
    if any(keyword in combined for keyword in ("人民代表大会常务委员会", "人大常委会", "自治区人民代表大会")):
        signals.append("npc_name")
    if any(keyword in combined for keyword in ("人民政府", "国务院", "省政府", "市政府", "县政府")):
        signals.append("government_name")
    if any(keyword in combined for keyword in ("国家电网", "国网", "供电公司", "电力公司")):
        signals.append("state_grid_name")
    if any(keyword in title for keyword in ("通知", "通报", "函", "批复", "印发")):
        signals.append("notice_style")
    if any(keyword in title for keyword in ("方案", "规划", "计划", "行动方案", "工作方案", "实施方案")):
        signals.append("plan_style")
    if any(keyword in combined for keyword in ("会议纪要", "工作会议", "讲话", "会议报告", "会议材料", "座谈会")):
        signals.append("meeting_style")
    if any(keyword in combined for keyword in ("培训", "课件", "讲义", "试题", "教程")):
        signals.append("training_style")
    if sum(1 for field in LOG_FIELDS if field in combined) >= 3 or any(term in title for term in ("日志", "台账", "日报", "记录")):
        signals.append("log_template")
    if any(keyword in combined for keyword in ("出版社", "ISBN", "作者", "主编")):
        signals.append("book_style")
    for signal_name, pattern in STANDARD_PATTERNS.items():
        if pattern.search(combined):
            signals.append(signal_name)
            signals.append("standard_style")
    if any(keyword in title for keyword in ("条例", "办法", "规定", "规则", "法")):
        signals.append("normative_title")
    return sorted(set(signals))


def build_document_profile(
    *,
    doc_id: str,
    text: str,
    source_name: str,
    source_path: str = "",
    head_chars: int = 1600,
    supplied_title: str = "",
) -> DocumentProfile:
    cleaned = _clean_text(text)
    title = _clean_text(supplied_title) or _pick_title(cleaned, source_name)
    head = _build_head(cleaned, head_chars)
    signals = tuple(_extract_signals(title, head, cleaned, source_name))
    return DocumentProfile(
        source_id=doc_id,
        source_name=source_name,
        source_path=source_path,
        text=cleaned,
        title=title,
        head=head,
        signals=signals,
    )
