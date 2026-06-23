# 文件作用：加载文档分类器并执行知识类别预测。
# 关联说明：包装 qa.doc_level3_classifier，向 app 层提供稳定服务接口。

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import torch

from app.core.logger import logger
from app.core.runtime_paths import DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR
from qa.doc_level3_classifier import TextDocumentLevel3Classifier


NEW_RULE_CLASSIFIER = "doc_level3_rule"
OLD_MODEL_CLASSIFIER = "legacy_model"
DEFAULT_KNOWLEDGE_CLASSIFIER = NEW_RULE_CLASSIFIER
SUPPORTED_KNOWLEDGE_CLASSIFIERS = {NEW_RULE_CLASSIFIER, OLD_MODEL_CLASSIFIER}


@dataclass(frozen=True)
class KnowledgeTaggingResult:
    knowledge_category: str
    knowledge_category_confidence: float
    knowledge_category_reason: str
    knowledge_category_source: str
    knowledge_category_detail: Dict[str, Any]


_LEGACY_TAGGERS: Dict[str, Any] = {}
_LEGACY_TAGGER_LOCK = Lock()
_RULE_CLASSIFIER: Optional[TextDocumentLevel3Classifier] = None
_RULE_CLASSIFIER_LOCK = Lock()


def normalize_knowledge_classifier(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        normalized = str(os.environ.get("KNOWLEDGE_CLASSIFIER_MODE") or DEFAULT_KNOWLEDGE_CLASSIFIER).strip().lower()
    aliases = {
        "new": NEW_RULE_CLASSIFIER,
        "rule": NEW_RULE_CLASSIFIER,
        "rules": NEW_RULE_CLASSIFIER,
        "doc_level3": NEW_RULE_CLASSIFIER,
        "doc_level3_classifier": NEW_RULE_CLASSIFIER,
        "local_rule": NEW_RULE_CLASSIFIER,
        "old": OLD_MODEL_CLASSIFIER,
        "legacy": OLD_MODEL_CLASSIFIER,
        "model": OLD_MODEL_CLASSIFIER,
        "knowledge_tagging_3lvl": OLD_MODEL_CLASSIFIER,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_KNOWLEDGE_CLASSIFIERS:
        raise ValueError(
            "knowledge_classifier 仅支持 "
            f"{NEW_RULE_CLASSIFIER} / {OLD_MODEL_CLASSIFIER}"
        )
    return normalized


def _resolve_device(device: str) -> str:
    device = (device or "cpu").strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in {"cuda", "gpu"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            idx = int(device.split(":", 1)[1])
        except Exception:
            return "cpu"
        return f"cuda:{idx}" if 0 <= idx < torch.cuda.device_count() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _load_legacy_tagger(*, device_override: Optional[str] = None) -> Any:
    from qa.knowledge_tagging_3lvl.predictor import KnowledgeTagger

    model_dir = os.environ.get(
        "KNOWLEDGE_TAGGER_MODEL_DIR",
        DEFAULT_KNOWLEDGE_TAGGER_MODEL_DIR,
    )
    device = _resolve_device(device_override or os.environ.get("KNOWLEDGE_TAGGER_DEVICE", "auto"))
    rule_threshold = float(os.environ.get("KNOWLEDGE_TAGGER_RULE_THRESHOLD", "0.95") or 0.95)
    model_fallback_threshold = float(os.environ.get("KNOWLEDGE_TAGGER_MODEL_FALLBACK_THRESHOLD", "0.0") or 0.0)

    model_path = Path(model_dir)
    labels_json = model_path / "labels.json"
    if not model_path.exists() or not labels_json.exists():
        raise FileNotFoundError(
            f"Knowledge tagger model not found. Expecting {model_path} with {labels_json.name}."
        )

    logger.info(
        "Loading legacy knowledge tagger model_dir=%s device=%s rule_threshold=%.2f model_fallback_threshold=%.2f",
        str(model_path),
        device,
        rule_threshold,
        model_fallback_threshold,
    )
    return KnowledgeTagger(
        labels_json=str(labels_json),
        model_dir=str(model_path),
        device=device,
        rule_threshold=rule_threshold,
        model_fallback_threshold=model_fallback_threshold,
    )


def get_knowledge_tagger(*, device_override: Optional[str] = None) -> Any:
    device = _resolve_device(device_override or os.environ.get("KNOWLEDGE_TAGGER_DEVICE", "auto"))
    cached = _LEGACY_TAGGERS.get(device)
    if cached is not None:
        return cached
    with _LEGACY_TAGGER_LOCK:
        cached = _LEGACY_TAGGERS.get(device)
        if cached is None:
            cached = _load_legacy_tagger(device_override=device)
            _LEGACY_TAGGERS[device] = cached
    return cached


def get_rule_classifier() -> TextDocumentLevel3Classifier:
    global _RULE_CLASSIFIER
    if _RULE_CLASSIFIER is not None:
        return _RULE_CLASSIFIER
    with _RULE_CLASSIFIER_LOCK:
        if _RULE_CLASSIFIER is None:
            config_path = str(os.environ.get("DOC_LEVEL3_CLASSIFIER_CONFIG") or "").strip()
            _RULE_CLASSIFIER = (
                TextDocumentLevel3Classifier(config_path)
                if config_path
                else TextDocumentLevel3Classifier.from_default_config()
            )
    return _RULE_CLASSIFIER


def release_knowledge_tagger_device_cache(device: Optional[str]) -> None:
    resolved = _resolve_device(device or "auto")
    with _LEGACY_TAGGER_LOCK:
        _LEGACY_TAGGERS.pop(resolved, None)
    gc.collect()
    if resolved.startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def classify_document_text(
    text: str,
    *,
    filename: str = "",
    device_override: Optional[str] = None,
    classifier_mode: Optional[str] = None,
) -> KnowledgeTaggingResult:
    """
    Classify a whole document into a single 3-level label path.
    The result is intended to be applied to all QA pairs generated from the file.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty document text; cannot classify")

    mode = normalize_knowledge_classifier(classifier_mode)
    if mode == OLD_MODEL_CLASSIFIER:
        return _classify_with_legacy_model(text, filename=filename, device_override=device_override)
    return _classify_with_rule_classifier(text, filename=filename)


def _classify_with_rule_classifier(text: str, *, filename: str = "") -> KnowledgeTaggingResult:
    normalized_text = str(text or "").strip()
    source_name = str(filename or "").strip() or "untitled"
    classifier = get_rule_classifier()
    pred = classifier.classify_text(text=normalized_text, source_name=source_name)

    level1 = str(pred.get("level1") or "").strip()
    level2 = str(pred.get("level2") or "").strip()
    level3 = str(pred.get("level3") or "").strip()
    label_path = "/".join(part for part in (level1, level2, level3) if part)
    confidence = pred.get("confidence") if isinstance(pred.get("confidence"), dict) else {}
    evidence = pred.get("evidence") if isinstance(pred.get("evidence"), dict) else {}
    selected = evidence.get("selected") if isinstance(evidence.get("selected"), dict) else {}
    reason = str(selected.get("id") or "rule").strip()[:512]

    detail = dict(pred)
    detail["classifier_mode"] = NEW_RULE_CLASSIFIER

    return KnowledgeTaggingResult(
        knowledge_category=label_path or "其他文档/其他/其他",
        knowledge_category_confidence=float(confidence.get("label") or 0.0),
        knowledge_category_reason=reason,
        knowledge_category_source=NEW_RULE_CLASSIFIER,
        knowledge_category_detail=detail,
    )


def _classify_with_legacy_model(
    text: str,
    *,
    filename: str = "",
    device_override: Optional[str] = None,
) -> KnowledgeTaggingResult:
    header = str(filename or "").strip()
    combined = f"{header}\n{text}" if header else text
    combined = combined.strip()
    if not combined:
        raise ValueError("Document text is empty after stripping; cannot classify")
    if len(combined) > 200_000:
        combined = combined[:200_000]

    tagger = get_knowledge_tagger(device_override=device_override)
    pred = tagger.predict_one(combined)

    reason = ""
    if pred.source == "rule":
        reason = str((pred.detail or {}).get("reason") or "").strip()
    elif pred.source == "model":
        reason = "model"
    reason = reason[:512]

    detail = dict(pred.detail or {})
    detail["classifier_mode"] = OLD_MODEL_CLASSIFIER

    return KnowledgeTaggingResult(
        knowledge_category=pred.label_path,
        knowledge_category_confidence=float(pred.confidence),
        knowledge_category_reason=reason,
        knowledge_category_source=str(pred.source),
        knowledge_category_detail=detail,
    )
