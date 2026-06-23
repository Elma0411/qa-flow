# 文件作用：提供分类规则中的关键词、正则和匹配算法。
# 关联说明：被 classifier_core 调用，封装具体匹配算法。

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


TERM_PATTERN = re.compile(r'"([^"]+)"|(\S+)')


def normalize_text(text: str, *, case_sensitive: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized if case_sensitive else normalized.lower()


def split_terms(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [item.strip() for item in raw if item and item.strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    terms = []
    for quoted, plain in TERM_PATTERN.findall(text):
        term = quoted or plain
        term = term.strip()
        if term:
            terms.append(term)
    return terms or [text]


def _partial_ratio(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    short, long_text = (needle, haystack) if len(needle) <= len(haystack) else (haystack, needle)
    window = len(short)
    step = max(1, window // 6)
    best = 0.0
    for start in range(0, max(1, len(long_text) - window + 1), step):
        segment = long_text[start : start + window + step]
        best = max(best, SequenceMatcher(None, short, segment).ratio())
        if best >= 0.999:
            return best
    return max(best, SequenceMatcher(None, short, long_text).ratio())


def match_algorithm(
    text: str,
    algorithm: str,
    pattern: str | list[str],
    *,
    case_sensitive: bool = False,
    threshold: float = 0.85,
) -> tuple[bool, dict[str, object]]:
    normalized_text = normalize_text(text, case_sensitive=case_sensitive)
    terms = split_terms(pattern)
    normalized_terms = [normalize_text(term, case_sensitive=case_sensitive) for term in terms]

    if algorithm == "none":
        return False, {"matched_terms": [], "reason": "disabled"}
    if not normalized_text or not normalized_terms:
        return False, {"matched_terms": [], "reason": "empty"}

    if algorithm == "any":
        matched = [term for term, norm in zip(terms, normalized_terms) if norm and norm in normalized_text]
        return bool(matched), {"matched_terms": matched}

    if algorithm == "all":
        matched = [term for term, norm in zip(terms, normalized_terms) if norm and norm in normalized_text]
        return len(matched) == len(normalized_terms), {"matched_terms": matched}

    if algorithm == "exact":
        needle = normalize_text(" ".join(terms), case_sensitive=case_sensitive)
        ok = bool(needle) and needle in normalized_text
        return ok, {"matched_terms": terms if ok else []}

    if algorithm == "regex":
        regex = pattern if isinstance(pattern, str) else "|".join(pattern)
        flags = 0 if case_sensitive else re.IGNORECASE
        matched = re.search(str(regex), text or "", flags)
        return bool(matched), {"matched_terms": [matched.group(0)] if matched else []}

    if algorithm == "fuzzy":
        best_term = ""
        best_score = 0.0
        for term, norm in zip(terms, normalized_terms):
            if not norm:
                continue
            score = _partial_ratio(norm, normalized_text)
            if score > best_score:
                best_term = term
                best_score = score
        return best_score >= threshold, {"matched_terms": [best_term] if best_term else [], "score": round(best_score, 4)}

    raise ValueError(f"unsupported matching algorithm: {algorithm}")
