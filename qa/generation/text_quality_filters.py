# 文件作用：识别含糊指代和低质量问答文本。
# 关联说明：被 qa_generation_flow 和 validation 复用，用于过滤含糊或低质量文本。

from __future__ import annotations

import re

_RE_AMBIGUOUS_ZH = re.compile(
    r"(这份|上述|其中|它们|他们|她们|"
    r"该表|本表|此表|该通知|本通知|此通知|该文件|本文件|此文件|"
    r"该办法|本办法|此办法|该制度|本制度|此制度|该规定|本规定|此规定|"
    r"该附件|本附件|此附件|"
    r"(?:该|本|此)[\u4e00-\u9fffA-Za-z0-9]{0,10}(?:表|通知|文件|办法|制度|规定|附件|报告|方案|意见|决定|计划|总结|函|公告|说明|指南|规范))"
)
_RE_AMBIGUOUS_ZH_QI = re.compile(r"其(?!他|它|余)")
_RE_AMBIGUOUS_EN = re.compile(
    r"\b(this|that|the above|above|aforementioned|herein|thereof)\b",
    flags=re.IGNORECASE,
)
_RE_EXPLICIT_EN_PHRASE = re.compile(
    r"\b(this|that)\s+[a-z][a-z0-9_-]{2,}(?:\s+[a-z][a-z0-9_-]{2,}){0,5}\b",
    flags=re.IGNORECASE,
)
_EN_REFERENCE_VERBS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "has",
    "have",
    "is",
    "may",
    "might",
    "must",
    "need",
    "needs",
    "require",
    "required",
    "requires",
    "should",
    "was",
    "were",
    "will",
    "would",
}


def _zh_reference_has_local_head(text: str, marker_start: int) -> bool:
    prefix = text[max(0, marker_start - 24) : marker_start]
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]{2,}(包括|分为|包含|设有|设置|由|按)", prefix))


def contains_ambiguous_reference(text: str, *, language_code: str) -> bool:
    """
    Heuristic filter for "指代不明" QA items.

    It blocks deictic/ana-phoric references such as:
    - zh: “这份/上述/其中/其…/该XXX表/本XXX文件/此XXX通知 …”
    - en: “this/that/the above/aforementioned/herein/thereof …”
    """
    raw = str(text or "").strip()
    if not raw:
        return False

    if language_code == "zh":
        compact = re.sub(r"\s+", "", raw)
        match = _RE_AMBIGUOUS_ZH.search(compact)
        if match and not _zh_reference_has_local_head(compact, match.start()):
            return True
        qi_match = _RE_AMBIGUOUS_ZH_QI.search(compact)
        if qi_match and not _zh_reference_has_local_head(compact, qi_match.start()):
            return True
        return False

    match = _RE_AMBIGUOUS_EN.search(raw)
    if not match:
        return False
    if match.group(1).lower() in {"this", "that"} and _RE_EXPLICIT_EN_PHRASE.search(raw):
        tail = raw[match.end() :].strip().lower()
        next_word = re.match(r"[a-z][a-z0-9_-]*", tail)
        if next_word and next_word.group(0) in _EN_REFERENCE_VERBS:
            return True
        return False
    return True


__all__ = ["contains_ambiguous_reference"]
