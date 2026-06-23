# 文件作用：作为知识标签服务的公共 facade。
# 关联说明：聚合同目录 classifier.py，供路由和 pipeline 固定知识类别逻辑使用。

"""Public facade for knowledge tagging services."""

from .classifier import (
    DEFAULT_KNOWLEDGE_CLASSIFIER,
    NEW_RULE_CLASSIFIER,
    OLD_MODEL_CLASSIFIER,
    SUPPORTED_KNOWLEDGE_CLASSIFIERS,
    KnowledgeTaggingResult,
    classify_document_text,
    get_knowledge_tagger,
    get_rule_classifier,
    normalize_knowledge_classifier,
    release_knowledge_tagger_device_cache,
)

__all__ = [
    "DEFAULT_KNOWLEDGE_CLASSIFIER",
    "NEW_RULE_CLASSIFIER",
    "OLD_MODEL_CLASSIFIER",
    "SUPPORTED_KNOWLEDGE_CLASSIFIERS",
    "KnowledgeTaggingResult",
    "classify_document_text",
    "get_knowledge_tagger",
    "get_rule_classifier",
    "normalize_knowledge_classifier",
    "release_knowledge_tagger_device_cache",
]

