# 文件作用：加载并校验文档三级标签分类配置。
# 关联说明：读取 doc_labels.yaml，向 classifier_core 提供标签与规则配置。

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_SCOPES = {
    "filename",
    "title",
    "head",
    "title_head",
    "fulltext",
    "signals",
}
VALID_ALGORITHMS = {"none", "any", "all", "exact", "regex", "fuzzy"}


@dataclass(frozen=True)
class RuleSpec:
    scope: str
    algorithm: str
    match: str | list[str]
    weight: float = 1.0
    threshold: float = 0.85
    case_sensitive: bool = False
    note: str = ""


@dataclass(frozen=True)
class LabelSpec:
    id: str
    level1: str
    level2: str
    level3: str
    priority: int = 0
    rules: tuple[RuleSpec, ...] = field(default_factory=tuple)
    excludes: tuple[RuleSpec, ...] = field(default_factory=tuple)
    note: str = ""


@dataclass(frozen=True)
class FallbackLabel:
    level1: str
    level2: str
    level3: str


@dataclass(frozen=True)
class DocClassifierConfig:
    source: str
    head_chars: int
    min_score: float
    fallback_label: FallbackLabel
    labels: tuple[LabelSpec, ...]


def _as_list(value: Any, *, field_name: str) -> list[Any]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{field_name} must be a list")


def _load_rule(raw: dict[str, Any], *, field_name: str) -> RuleSpec:
    scope = str(raw.get("scope", "")).strip()
    algorithm = str(raw.get("algorithm", "")).strip().lower()
    match_value = raw.get("match", "")
    weight = float(raw.get("weight", 1.0))
    threshold = float(raw.get("threshold", 0.85))
    case_sensitive = bool(raw.get("case_sensitive", False))
    note = str(raw.get("note", "")).strip()

    if scope not in VALID_SCOPES:
        raise ValueError(f"{field_name}.scope must be one of {sorted(VALID_SCOPES)}")
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(f"{field_name}.algorithm must be one of {sorted(VALID_ALGORITHMS)}")
    if not isinstance(match_value, (str, list)):
        raise ValueError(f"{field_name}.match must be a string or list of strings")
    if isinstance(match_value, list) and not all(isinstance(item, str) for item in match_value):
        raise ValueError(f"{field_name}.match list must contain only strings")
    return RuleSpec(
        scope=scope,
        algorithm=algorithm,
        match=match_value,
        weight=weight,
        threshold=threshold,
        case_sensitive=case_sensitive,
        note=note,
    )


def load_doc_classifier_config(config_path: Path) -> DocClassifierConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults", {}) or {}
    fallback = raw.get("fallback_label", {}) or {}

    labels: list[LabelSpec] = []
    for index, label_raw in enumerate(_as_list(raw.get("labels", []), field_name="labels"), start=1):
        field_prefix = f"labels[{index}]"
        rules = tuple(
            _load_rule(item, field_name=f"{field_prefix}.rules[{idx}]")
            for idx, item in enumerate(_as_list(label_raw.get("rules", []), field_name=f"{field_prefix}.rules"), start=1)
        )
        excludes = tuple(
            _load_rule(item, field_name=f"{field_prefix}.excludes[{idx}]")
            for idx, item in enumerate(
                _as_list(label_raw.get("excludes", []), field_name=f"{field_prefix}.excludes"),
                start=1,
            )
        )
        labels.append(
            LabelSpec(
                id=str(label_raw.get("id") or f"label_{index}"),
                level1=str(label_raw.get("level1", "")).strip(),
                level2=str(label_raw.get("level2", "")).strip(),
                level3=str(label_raw.get("level3", "")).strip(),
                priority=int(label_raw.get("priority", 0)),
                rules=rules,
                excludes=excludes,
                note=str(label_raw.get("note", "")).strip(),
            )
        )

    if not labels:
        raise ValueError("doc classifier config must define at least one label")

    fallback_label = FallbackLabel(
        level1=str(fallback.get("level1", "其他文档")).strip(),
        level2=str(fallback.get("level2", "其他")).strip(),
        level3=str(fallback.get("level3", "其他")).strip(),
    )
    return DocClassifierConfig(
        source=str(raw.get("source", config_path)),
        head_chars=int(defaults.get("head_chars", 1600)),
        min_score=float(defaults.get("min_score", 4.0)),
        fallback_label=fallback_label,
        labels=tuple(labels),
    )
