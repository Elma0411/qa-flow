# 文件作用：封装 QA 质量评估的具体评分算法。
# 关联说明：由 qa_quality_evaluator 组合调用，便于把 scoring 逻辑与模型/CLI 分离。

from __future__ import annotations

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from .runtime import (
    extract_entities_and_numbers,
    extract_factual_information,
    extract_key_information,
    extract_information_units,
    extract_key_terms,
    extract_question_segments,
    extract_text_elements,
    inverse_ppl_normalize,
    is_number_or_date,
    normalize_text,
    tokenize,
)


def score_relevance(st_model: Any, question: str, answer: str) -> float:
    embeddings = st_model.encode([question, answer], convert_to_tensor=True)
    return cosine_similarity(
        embeddings[0].cpu().numpy().reshape(1, -1),
        embeddings[1].cpu().numpy().reshape(1, -1),
    )[0][0]


def score_coverage(st_model: Any, question: str, answer: str) -> float:
    lexical_coverage = lexical_coverage_score(question, answer)
    semantic_coverage = semantic_coverage_score(st_model, question, answer)
    element_coverage = question_element_coverage_score(st_model, question, answer)
    final_coverage = 0.3 * lexical_coverage + 0.5 * semantic_coverage + 0.2 * element_coverage
    return round(min(final_coverage, 1.0), 4)


def lexical_coverage_score(question: str, answer: str) -> float:
    question_terms = extract_key_terms(question)
    answer_terms = extract_key_terms(answer)
    if not question_terms:
        return 0.0
    from collections import Counter

    q_counter = Counter(question_terms)
    a_counter = Counter(answer_terms)
    matched = 0
    for term, count in q_counter.items():
        if a_counter.get(term, 0) >= count:
            matched += 1
    return matched / len(q_counter)


def semantic_coverage_score(st_model: Any, question: str, answer: str, similarity_threshold: float = 0.7) -> float:
    question_segments = extract_question_segments(question)
    if not question_segments:
        return 0.0
    covered_segments = 0
    for segment in question_segments:
        embeddings = st_model.encode([segment, answer], convert_to_tensor=True)
        similarity = cosine_similarity(
            embeddings[0].cpu().numpy().reshape(1, -1),
            embeddings[1].cpu().numpy().reshape(1, -1),
        )[0][0]
        if similarity >= similarity_threshold:
            covered_segments += 1
    return covered_segments / len(question_segments)


def question_element_coverage_score(st_model: Any, question: str, answer: str) -> float:
    question_elements = extract_text_elements(question)
    answer_elements = extract_text_elements(answer)
    if not question_elements:
        return 0.0
    covered_elements = 0
    for q_element in question_elements:
        max_similarity = 0.0
        for a_element in answer_elements:
            embeddings = st_model.encode([q_element, a_element], convert_to_tensor=True)
            similarity = cosine_similarity(
                embeddings[0].cpu().numpy().reshape(1, -1),
                embeddings[1].cpu().numpy().reshape(1, -1),
            )[0][0]
            max_similarity = max(max_similarity, similarity)
        if max_similarity >= 0.6:
            covered_elements += 1
    return covered_elements / len(question_elements)


def score_overlap(st_model: Any, answer: str, source: str) -> float:
    tfidf_overlap = tfidf_overlap_score(answer, source)
    semantic_overlap = semantic_overlap_score(st_model, answer, source)
    unit_overlap = information_unit_overlap_score(st_model, answer, source)
    key_info_overlap = key_information_overlap_score(st_model, answer, source)
    final_overlap = 0.2 * tfidf_overlap + 0.4 * semantic_overlap + 0.3 * unit_overlap + 0.1 * key_info_overlap
    return round(min(final_overlap, 1.0), 4)


def tfidf_overlap_score(answer: str, source: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vectors = TfidfVectorizer(tokenizer=lambda x: tokenize(x), token_pattern=None).fit_transform([answer, source])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except Exception:
        return 0.0


def semantic_overlap_score(st_model: Any, answer: str, source: str) -> float:
    embeddings = st_model.encode([answer, source], convert_to_tensor=True)
    similarity = cosine_similarity(
        embeddings[0].cpu().numpy().reshape(1, -1),
        embeddings[1].cpu().numpy().reshape(1, -1),
    )[0][0]
    return float(similarity)


def information_unit_overlap_score(st_model: Any, answer: str, source: str) -> float:
    answer_units = extract_information_units(answer)
    source_units = extract_information_units(source)
    if not source_units or not answer_units:
        return 0.0
    total_overlap = 0.0
    matched_units = 0
    for source_unit in source_units:
        max_similarity = 0.0
        for answer_unit in answer_units:
            embeddings = st_model.encode([source_unit, answer_unit], convert_to_tensor=True)
            similarity = cosine_similarity(
                embeddings[0].cpu().numpy().reshape(1, -1),
                embeddings[1].cpu().numpy().reshape(1, -1),
            )[0][0]
            max_similarity = max(max_similarity, similarity)
        total_overlap += max_similarity
        if max_similarity >= 0.6:
            matched_units += 1
    avg_similarity = total_overlap / len(source_units)
    match_ratio = matched_units / len(source_units)
    return 0.7 * avg_similarity + 0.3 * match_ratio


def key_information_overlap_score(st_model: Any, answer: str, source: str) -> float:
    answer_key_info = extract_key_information(answer)
    source_key_info = extract_key_information(source)
    if not source_key_info:
        return 1.0
    if not answer_key_info:
        return 0.0
    overlapped_info = 0
    for source_info in source_key_info:
        max_match = 0.0
        for answer_info in answer_key_info:
            if is_number_or_date(source_info) and is_number_or_date(answer_info):
                if source_info == answer_info:
                    max_match = 1.0
                    break
            else:
                embeddings = st_model.encode([source_info, answer_info], convert_to_tensor=True)
                similarity = cosine_similarity(
                    embeddings[0].cpu().numpy().reshape(1, -1),
                    embeddings[1].cpu().numpy().reshape(1, -1),
                )[0][0]
                max_match = max(max_match, similarity)
        if max_match >= 0.75:
            overlapped_info += 1
    return overlapped_info / len(source_key_info)


def score_accuracy(st_model: Any, answer: str, source: str) -> float:
    lexical_accuracy = lexical_accuracy_score(answer, source)
    semantic_accuracy = semantic_accuracy_score(st_model, answer, source)
    factual_accuracy = factual_accuracy_score(st_model, answer, source)
    entity_accuracy = entity_accuracy_score(st_model, answer, source)
    final_accuracy = 0.2 * lexical_accuracy + 0.4 * semantic_accuracy + 0.3 * factual_accuracy + 0.1 * entity_accuracy
    return round(min(final_accuracy, 1.0), 4)


def lexical_accuracy_score(answer: str, source: str) -> float:
    source_keywords = set(tokenize(source))
    answer_keywords = set(tokenize(answer))
    if not source_keywords:
        return 0.0
    return len(source_keywords & answer_keywords) / len(source_keywords)


def semantic_accuracy_score(st_model: Any, answer: str, source: str, similarity_threshold: float = 0.75) -> float:
    source_units = extract_information_units(source)
    answer_units = extract_information_units(answer)
    if not source_units:
        return 0.0
    accurate_units = 0
    for source_unit in source_units:
        max_similarity = 0.0
        for answer_unit in answer_units:
            embeddings = st_model.encode([source_unit, answer_unit], convert_to_tensor=True)
            similarity = cosine_similarity(
                embeddings[0].cpu().numpy().reshape(1, -1),
                embeddings[1].cpu().numpy().reshape(1, -1),
            )[0][0]
            max_similarity = max(max_similarity, similarity)
        if max_similarity >= similarity_threshold:
            accurate_units += 1
    return accurate_units / len(source_units)


def factual_accuracy_score(st_model: Any, answer: str, source: str) -> float:
    source_facts = extract_factual_information(source)
    answer_facts = extract_factual_information(answer)
    if not source_facts:
        return 0.0
    accurate_facts = 0
    for source_fact in source_facts:
        for answer_fact in answer_facts:
            embeddings = st_model.encode([source_fact, answer_fact], convert_to_tensor=True)
            similarity = cosine_similarity(
                embeddings[0].cpu().numpy().reshape(1, -1),
                embeddings[1].cpu().numpy().reshape(1, -1),
            )[0][0]
            if similarity >= 0.8:
                accurate_facts += 1
                break
    return accurate_facts / len(source_facts)


def entity_accuracy_score(st_model: Any, answer: str, source: str) -> float:
    source_entities = extract_entities_and_numbers(source)
    answer_entities = extract_entities_and_numbers(answer)
    if not source_entities:
        return 1.0
    matched_entities = 0
    for source_entity in source_entities:
        for answer_entity in answer_entities:
            if is_number_or_date(source_entity) and is_number_or_date(answer_entity):
                if source_entity == answer_entity:
                    matched_entities += 1
                    break
            else:
                embeddings = st_model.encode([source_entity, answer_entity], convert_to_tensor=True)
                similarity = cosine_similarity(
                    embeddings[0].cpu().numpy().reshape(1, -1),
                    embeddings[1].cpu().numpy().reshape(1, -1),
                )[0][0]
                if similarity >= 0.85:
                    matched_entities += 1
                    break
    return matched_entities / len(source_entities)


def score_fluency(bert_model: Any, grammar_tool: Any | None, grammar_available: bool, question: str, answer: str) -> float:
    q_score = single_fluency(bert_model, grammar_tool, grammar_available, question)
    a_score = single_fluency(bert_model, grammar_tool, grammar_available, answer)
    q_len = len(question)
    a_len = len(answer)
    total_len = q_len + a_len
    if total_len <= 0:
        return 0.0
    q_weight = (q_len / total_len) ** 0.5
    a_weight = (a_len / total_len) ** 0.5
    combined_score = (q_weight * q_score + a_weight * a_score) / (q_weight + a_weight)
    return round(max(0.0, min(combined_score, 1.0)), 4)


def single_fluency(bert_model: Any, grammar_tool: Any | None, grammar_available: bool, text: str) -> float:
    clean_text = normalize_text(text)
    raw_score = grammar_score(grammar_tool, grammar_available, clean_text)
    char_seq = " ".join(list(clean_text))
    ppl = bert_model.perplexity(char_seq, verbose=False)
    ppl_norm = inverse_ppl_normalize(ppl)
    return 0.5 * ppl_norm + 0.5 * raw_score


def grammar_score(grammar_tool: Any | None, grammar_available: bool, text: str) -> float:
    if not grammar_available or grammar_tool is None:
        return 0.8
    try:
        matches = grammar_tool.check(text)
        error_count = len(matches)
        raw_score = max(0, 1 - error_count * 0.1)
        return raw_score ** 0.5
    except Exception:
        return 0.8
