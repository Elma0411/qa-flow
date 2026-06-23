# 文件作用：封装文档三级标签分类器的服务化入口。
# 关联说明：包装 classifier_core/config/profile，对外提供分类器服务入口。

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .classifier_core import DocumentRuleClassifier
from .config import load_doc_classifier_config
from .profile import build_document_profile


def build_default_config_path() -> Path:
    return Path(__file__).resolve().parent / "doc_labels.yaml"


def _build_doc_id(source_name: str, source_path: str, text: str) -> str:
    seed = source_path or source_name or text[:120]
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]


class TextDocumentLevel3Classifier:
    """Text-only adapter for the document level1/level2/level3 classifier.

    This module is intended for integration into other projects that already
    provide extracted file text and optional title metadata. It deliberately
    excludes local file traversal and document parsing.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_doc_classifier_config(self.config_path)
        self.classifier = DocumentRuleClassifier(self.config)

    @classmethod
    def from_default_config(cls) -> "TextDocumentLevel3Classifier":
        return cls(build_default_config_path())

    @classmethod
    def from_repo_default(cls) -> "TextDocumentLevel3Classifier":
        return cls.from_default_config()

    def classify_text(
        self,
        *,
        text: str,
        source_name: str,
        source_path: str = "",
        title: str = "",
        doc_id: str = "",
    ) -> dict[str, object]:
        normalized_text = str(text or "").strip()
        normalized_source_name = str(source_name or "").strip()
        normalized_source_path = str(source_path or "").strip()
        normalized_title = str(title or "").strip()
        normalized_doc_id = str(doc_id or "").strip()

        if not normalized_text:
            raise ValueError("text must not be empty")
        if not normalized_source_name:
            raise ValueError("source_name must not be empty")

        profile = build_document_profile(
            doc_id=normalized_doc_id or _build_doc_id(normalized_source_name, normalized_source_path, normalized_text),
            text=normalized_text,
            source_name=normalized_source_name,
            source_path=normalized_source_path,
            supplied_title=normalized_title,
            head_chars=self.config.head_chars,
        )
        return self.classifier.classify(profile)

    def classify_record(self, record: Mapping[str, object]) -> dict[str, object]:
        return self.classify_text(
            text=str(record.get("text") or record.get("content") or record.get("body") or ""),
            source_name=str(record.get("source_name") or record.get("filename") or record.get("name") or ""),
            source_path=str(record.get("source_path") or record.get("path") or ""),
            title=str(record.get("title") or ""),
            doc_id=str(record.get("doc_id") or record.get("id") or ""),
        )

    def classify_records(self, records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
        return [self.classify_record(record) for record in records]

    def classify_records_to_jsonl(
        self,
        records: Iterable[Mapping[str, object]],
        output_path: str | Path,
    ) -> Path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for result in self.classify_records(records):
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        return target
