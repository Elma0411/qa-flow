from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

try:
    from .model_paths import classifier_model_dir
except ImportError:  # pragma: no cover - script mode
    from model_paths import classifier_model_dir


@dataclass(frozen=True)
class ClassConfig:
    class_id: int
    model_label: str
    category_key: str
    display_name: str


_REQUIRED_CLASS_FIELDS = {"class_id", "model_label", "category_key", "display_name"}


BUILTIN_CLASS_CONFIGS = [
    ClassConfig(0, "其他", "others", "其他"),
    ClassConfig(1, "印章签名", "seal", "印章签名"),
    ClassConfig(2, "宏观线路总图", "macro_line_overview", "宏观线路总图"),
    ClassConfig(3, "局部电路原理图", "local_circuit_schematic", "局部电路原理图"),
    ClassConfig(4, "数据表格", "data_table", "数据表格"),
    ClassConfig(5, "文本描述性记录表", "text_record_table", "文本描述性记录表"),
    ClassConfig(6, "甘特图", "gantt_chart", "甘特图"),
    ClassConfig(7, "示意图（流程图／架构图）", "flow_architecture_diagram", "示意图（流程图／架构图）"),
    ClassConfig(8, "空白通用表格", "blank_generic_table", "空白通用表格"),
    ClassConfig(9, "设备布局图", "equipment_layout", "设备布局图"),
]

# Interim 10-class routing used before dedicated prompts were added:
#
# BUILTIN_CLASS_CONFIGS = [
#     ClassConfig(0, "其他", "others", "其他"),
#     ClassConfig(1, "印章签名", "seal", "印章签名"),
#     ClassConfig(2, "宏观线路总图", "circuit", "宏观线路总图"),
#     ClassConfig(3, "局部电路原理图", "circuit", "局部电路原理图"),
#     ClassConfig(4, "数据表格", "sheet", "数据表格"),
#     ClassConfig(5, "文本描述性记录表", "sheet", "文本描述性记录表"),
#     ClassConfig(6, "甘特图", "sheet", "甘特图"),
#     ClassConfig(7, "示意图（流程图／架构图）", "schematic", "示意图（流程图／架构图）"),
#     ClassConfig(8, "空白通用表格", "sheet", "空白通用表格"),
#     ClassConfig(9, "设备布局图", "schematic", "设备布局图"),
# ]

# Legacy 6-class configuration used by the previous classifier weights:
#
# BUILTIN_CLASS_CONFIGS = [
#     ClassConfig(0, "0", "sheet", "表格"),
#     ClassConfig(1, "1", "circuit", "电路图"),
#     ClassConfig(2, "2", "seal", "印章签名"),
#     ClassConfig(3, "3", "architecture", "架构图"),
#     ClassConfig(4, "4", "schematic", "示意图"),
#     ClassConfig(5, "5", "others", "其他"),
# ]
# DEFAULT_CLASS_CONFIG = BUILTIN_CLASS_CONFIGS[-1]


def _configured_class_file() -> Optional[Path]:
    configured = str(os.getenv("CLASSIFIER_CLASS_CONFIG_FILE") or "").strip()
    if not configured:
        return None
    return Path(configured).expanduser()


def _class_file_candidates() -> Iterable[Path]:
    configured = _configured_class_file()
    if configured is not None:
        yield configured
    yield classifier_model_dir() / "classes.json"


def _parse_class_config(item: Any, index: int) -> ClassConfig:
    if not isinstance(item, dict):
        raise ValueError(f"classes[{index}] must be an object")
    keys = set(item)
    if keys != _REQUIRED_CLASS_FIELDS:
        expected = ", ".join(sorted(_REQUIRED_CLASS_FIELDS))
        got = ", ".join(sorted(keys))
        raise ValueError(f"classes[{index}] fields must be exactly [{expected}], got [{got}]")

    class_id = item.get("class_id")
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise ValueError(f"classes[{index}].class_id must be a non-negative integer")

    def _required_text(field: str) -> str:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"classes[{index}].{field} must be a non-empty string")
        return value.strip()

    return ClassConfig(
        class_id=class_id,
        model_label=_required_text("model_label"),
        category_key=_required_text("category_key"),
        display_name=_required_text("display_name"),
    )


def _validate_class_configs(configs: List[ClassConfig]) -> List[ClassConfig]:
    if not configs:
        raise ValueError("classes must not be empty")

    configs = sorted(configs, key=lambda config: config.class_id)
    class_ids = [config.class_id for config in configs]
    expected_ids = list(range(len(configs)))
    if class_ids != expected_ids:
        raise ValueError(
            f"class_id values must be sequential 0..{len(configs) - 1}; got {class_ids}"
        )

    labels = [config.model_label for config in configs]
    if len(labels) != len(set(labels)):
        raise ValueError("model_label values must be unique")

    return configs


def load_class_configs_from_file(path: Path) -> List[ClassConfig]:
    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("classes root must be a JSON array")
        return _validate_class_configs(
            [_parse_class_config(item, index) for index, item in enumerate(payload)]
        )
    except Exception as exc:
        raise RuntimeError(f"Invalid classifier class config file {source}: {exc}") from exc


def load_class_configs() -> List[ClassConfig]:
    for candidate in _class_file_candidates():
        if candidate.exists():
            return load_class_configs_from_file(candidate)
    return list(BUILTIN_CLASS_CONFIGS)


CLASS_CONFIGS = load_class_configs()

CLASS_NAMES = [config.model_label for config in CLASS_CONFIGS]
DEFAULT_CLASS_CONFIG = CLASS_CONFIGS[0]
CLASS_BY_MODEL_LABEL = {config.model_label: config for config in CLASS_CONFIGS}


def get_class_config_by_model_label(model_label: str) -> ClassConfig:
    return CLASS_BY_MODEL_LABEL.get(str(model_label), DEFAULT_CLASS_CONFIG)
