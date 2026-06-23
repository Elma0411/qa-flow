# 文件作用：承载本地 QA 评估器运行时共用的设备与文本处理工具。
# 关联说明：被 qa_quality_evaluator、model_loading 和无监督流畅度指标复用，避免评分逻辑重复持有基础设施代码。

from __future__ import annotations

from typing import List

import jieba
import jieba.posseg
import torch

DEFAULT_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一个", "这个",
    "也", "要", "说", "去", "会", "为", "可以", "与", "等", "好", "这", "那", "而",
    "年", "月", "日", "个", "对", "中", "到", "没有", "还", "又", "并", "或", "能",
    "如何", "什么", "吗", "了", "与", "及", "其", "已", "通过", "自己", "之", "需要",
}


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_key_terms(text: str) -> List[str]:
    pos_words = jieba.posseg.lcut(text)
    return [
        word.word
        for word in pos_words
        if word.flag.startswith(("n", "v"))
        and word.word not in DEFAULT_STOPWORDS
        and len(word.word) > 1
    ]


def extract_text_elements(text: str) -> List[str]:
    words_with_pos = jieba.posseg.lcut(text)
    elements: List[str] = []
    for word, pos in words_with_pos:
        if pos.startswith("n") and len(word) > 1 and word not in DEFAULT_STOPWORDS:
            elements.append(word)
        elif pos.startswith("v") and len(word) > 1 and word not in DEFAULT_STOPWORDS:
            elements.append(word)
        elif pos.startswith("a") and len(word) > 1:
            elements.append(word)
        elif pos.startswith(("m", "q")) and len(word) > 0:
            elements.append(word)
    seen = set()
    unique_elements: List[str] = []
    for element in elements:
        if element not in seen:
            seen.add(element)
            unique_elements.append(element)
    return unique_elements


def extract_information_units(text: str) -> List[str]:
    import re

    units = re.split(r"[。；！？;!?]", text)
    units = [unit.strip() for unit in units if unit.strip()]
    if len(units) <= 1:
        units = re.split(r"[，,]", text)
        units = [unit.strip() for unit in units if unit.strip()]
    units = [unit for unit in units if len(unit) > 3]
    return units if units else [text]


def extract_factual_information(text: str) -> List[str]:
    words_with_pos = jieba.posseg.lcut(text)
    factual_phrases: List[str] = []
    current_phrase: List[str] = []
    for word, pos in words_with_pos:
        if pos.startswith(("n", "v", "a")) and len(word) > 1:
            current_phrase.append(word)
        else:
            if len(current_phrase) >= 2:
                factual_phrases.append("".join(current_phrase))
            current_phrase = []
    if len(current_phrase) >= 2:
        factual_phrases.append("".join(current_phrase))
    return factual_phrases if factual_phrases else extract_information_units(text)


def extract_entities_and_numbers(text: str) -> List[str]:
    import re

    entities: List[str] = []
    numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
    entities.extend(numbers)
    dates = re.findall(r"\d{4}年|\d{1,2}月|\d{1,2}日", text)
    entities.extend(dates)
    words_with_pos = jieba.posseg.lcut(text)
    for word, pos in words_with_pos:
        if pos in ["nr", "ns", "nt", "nz"] and len(word) > 1:
            entities.append(word)
        elif pos.startswith("n") and len(word) > 2:
            entities.append(word)
    return list(set(entities))


def extract_key_information(text: str) -> List[str]:
    import re

    key_info: List[str] = []
    numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
    key_info.extend(numbers)
    dates = re.findall(r"\d{4}年|\d{1,2}月|\d{1,2}日", text)
    key_info.extend(dates)
    words_with_pos = jieba.posseg.lcut(text)
    noun_phrase: List[str] = []
    for word, pos in words_with_pos:
        if pos.startswith("n") and len(word) > 1:
            noun_phrase.append(word)
        else:
            if len(noun_phrase) >= 1:
                key_info.append("".join(noun_phrase))
            noun_phrase = []
    if len(noun_phrase) >= 1:
        key_info.append("".join(noun_phrase))
    verb_phrases: List[str] = []
    for word, pos in words_with_pos:
        if pos.startswith("v") and len(word) > 1 and word not in DEFAULT_STOPWORDS:
            verb_phrases.append(word)
    key_info.extend(verb_phrases)
    key_info = list(dict.fromkeys([info for info in key_info if len(info) > 1]))
    return key_info


def is_number_or_date(text: str) -> bool:
    import re

    number_pattern = r"^\d+(?:\.\d+)?%?$"
    date_pattern = r"^\d{4}年$|^\d{1,2}月$|^\d{1,2}日$"
    return bool(re.match(number_pattern, text) or re.match(date_pattern, text))


def extract_question_segments(question: str) -> List[str]:
    import re

    segments = re.split(r"[，。？！；,;?!]", question)
    segments = [seg.strip() for seg in segments if seg.strip()]
    if len(segments) <= 1:
        conjunctions = ["和", "与", "以及", "还有", "或者", "或", "及"]
        for conj in conjunctions:
            if conj in question:
                segments = question.split(conj)
                segments = [seg.strip() for seg in segments if seg.strip()]
                break
    if len(segments) <= 1 and len(question) > 20:
        mid = len(question) // 2
        for i in range(mid - 5, mid + 5):
            if i < len(question) and question[i] in " ，。？！；,;?!":
                segments = [question[:i].strip(), question[i + 1 :].strip()]
                break
    return [seg for seg in segments if len(seg) > 2]


def normalize_text(text: str) -> str:
    return text.replace(" ", "")


def inverse_ppl_normalize(ppl: float) -> float:
    return 1 / (1 + 0.01 * ppl ** 0.8)


def tokenize(text: str) -> List[str]:
    return [token for token in jieba.lcut(text) if token]
