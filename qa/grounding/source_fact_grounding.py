# 文件作用：校验问答来源事实是否能在原文中定位。
# 关联说明：位于 generation 之后、validation 之前，校验答案事实能否回到来源文本。

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Tuple


def validate_source_fact_text_detail_mode(
    source_fact_text: str,
    *,
    qa_detail_mode: str,
    language_code: str,
) -> Tuple[bool, str]:
    """
    Enforce a visible distinction between qa_detail_mode=point vs summary.
    """
    fact = (source_fact_text or "").strip()
    if not fact:
        return False, "missing_source_fact_text"
    detail = (qa_detail_mode or "point").strip().lower()
    if detail not in {"point", "summary"}:
        detail = "point"

    if detail == "point":
        if "\n" in fact:
            return False, "point_source_fact_multiline"
        if language_code == "zh":
            if "；" in fact:
                return False, "point_source_fact_has_semicolon"
            if fact.count("。") > 1:
                return False, "point_source_fact_multi_sentence"
        else:
            if ";" in fact:
                return False, "point_source_fact_has_semicolon"
            if fact.count(".") > 1:
                return False, "point_source_fact_multi_sentence"
        return True, "ok"

    segments = split_summary_grounding_segments(fact)
    if len(segments) >= 2:
        return True, "ok"
    return False, "summary_source_fact_segments_insufficient"


_GROUNDING_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "\u3000": " ",
    }
)


def normalize_grounding_text(text: str) -> str:
    normalized = str(text or "").translate(_GROUNDING_TRANSLATION).lower().strip()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def char_ngram_recall(needle: str, haystack: str, n: int = 3) -> float:
    if not needle or not haystack:
        return 0.0
    if len(needle) < n:
        return 1.0 if needle in haystack else 0.0
    grams = [needle[index : index + n] for index in range(len(needle) - n + 1)]
    if not grams:
        return 0.0
    matched = sum(1 for gram in grams if gram in haystack)
    return matched / len(grams)


def build_chunk_grounding_candidates(chunk_text: str) -> List[str]:
    raw = str(chunk_text or "").strip()
    if not raw:
        return []
    lines = [part.strip() for part in re.split(r"[\r\n]+", raw) if part.strip()]
    if len(lines) >= 2:
        return lines
    sentences = [
        part.strip() for part in re.split(r"(?<=[。！？；.!?;])", raw) if part.strip()
    ]
    if len(sentences) >= 2:
        return sentences
    return [raw]


def merge_grounding_candidates(chunk_text: str) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for candidate in build_chunk_grounding_candidates(chunk_text):
        normalized = normalize_grounding_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(candidate)
    return candidates


def best_chunk_grounding_score(text: str, candidates: List[str]) -> Tuple[float, bool]:
    target = normalize_grounding_text(text)
    if not target:
        return 0.0, False
    best_score = 0.0
    exact_or_substring = False
    for candidate in candidates:
        normalized_candidate = normalize_grounding_text(candidate)
        if not normalized_candidate:
            continue
        if target in normalized_candidate:
            return 1.0, True
        ratio = SequenceMatcher(None, target, normalized_candidate).ratio()
        recall = char_ngram_recall(target, normalized_candidate)
        score = max(ratio, recall)
        if score > best_score:
            best_score = score
        if normalized_candidate in target and len(normalized_candidate) >= 8:
            exact_or_substring = True
    return best_score, exact_or_substring


def split_summary_grounding_segments(source_fact_text: str) -> List[str]:
    fact = str(source_fact_text or "").strip()
    if not fact:
        return []
    primary_parts = [part.strip() for part in re.split(r"[；;\n]+", fact) if part.strip()]
    long_parts = [part for part in primary_parts if len(normalize_grounding_text(part)) >= 8]
    if len(long_parts) >= 2:
        return long_parts
    comma_parts = [part.strip() for part in re.split(r"[，,、]+", fact) if part.strip()]
    long_parts = [part for part in comma_parts if len(normalize_grounding_text(part)) >= 8]
    if len(long_parts) >= 2:
        return long_parts
    sentence_parts = [
        part.strip() for part in re.split(r"(?<=[。.!?！？])", fact) if part.strip()
    ]
    long_sentences = [
        part for part in sentence_parts if len(normalize_grounding_text(part)) >= 8
    ]
    if len(long_sentences) >= 2:
        return long_sentences
    return [fact]


def validate_source_fact_grounding(
    source_fact_text: str,
    *,
    chunk_text: str,
    qa_detail_mode: str,
    language_code: str,
) -> Tuple[bool, str]:
    candidates = merge_grounding_candidates(chunk_text)
    if not candidates:
        return False, "grounding_chunk_empty"

    detail = (qa_detail_mode or "point").strip().lower()
    if detail not in {"point", "summary"}:
        detail = "point"

    if detail == "point":
        score, matched = best_chunk_grounding_score(source_fact_text, candidates)
        if matched or score >= 0.86:
            return True, "ok"
        return False, "source_fact_not_grounded_in_chunk"

    segments = split_summary_grounding_segments(source_fact_text)
    if len(segments) < 2:
        return False, "summary_source_fact_segments_insufficient"
    for segment in segments:
        score, matched = best_chunk_grounding_score(segment, candidates)
        if matched or score >= 0.84:
            continue
        return False, "summary_source_fact_segment_not_grounded_in_chunk"
    return True, "ok"


__all__ = [
    "split_summary_grounding_segments",
    "validate_source_fact_text_detail_mode",
    "validate_source_fact_grounding",
]
