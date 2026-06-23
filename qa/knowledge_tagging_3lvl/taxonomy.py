# 文件作用：解析知识三级标签体系并构建标签映射。
# 关联说明：被 synth 和 train 脚本使用，定义标签层级映射。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from typing import Iterable


@dataclass(frozen=True)
class LeafLabel:
    level1: str
    level2: str
    level3: str
    description: str

    @property
    def path(self) -> str:
        return f"{self.level1}/{self.level2}/{self.level3}"


def _split_name_desc(text: str) -> tuple[str, str]:
    # Prefer Chinese colon; fall back to ASCII colon.
    for sep in ("：", ":"):
        if sep in text:
            name, desc = text.split(sep, 1)
            return name.strip(), desc.strip()
    return text.strip(), ""


def parse_taxonomy(label_file: Union[str, Path]) -> list[LeafLabel]:
    """
    Parse `qa/dataset/三级知识标签.txt` into leaf labels (三级).

    File format is mostly:
      一级(无缩进)
       -二级(缩进1)
        -三级：描述(缩进2)

    But the file may contain inconsistent indentation (e.g. some leaf lines use
    3 spaces; some 二级 lines may also be indented). This parser infers the
    "二级缩进" per 一级 block by the first bullet indentation.
    """

    path = Path(label_file)
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip("\n") for line in text.splitlines() if line.strip()]

    leaves: list[LeafLabel] = []
    current_l1: Optional[str] = None
    current_l2: Optional[str] = None
    l2_indent: Optional[int] = None

    for raw_line in lines:
        stripped = raw_line.lstrip(" ")
        indent = len(raw_line) - len(stripped)

        is_bullet = stripped.startswith("-")
        if not is_bullet and indent == 0:
            current_l1 = stripped.strip()
            current_l2 = None
            l2_indent = None
            continue

        if not is_bullet:
            continue

        if current_l1 is None:
            continue

        bullet_text = stripped[1:].strip()

        if l2_indent is None:
            l2_indent = indent

        if indent == l2_indent:
            current_l2 = bullet_text
            continue

        if current_l2 is None:
            continue

        level3_name, desc = _split_name_desc(bullet_text)
        leaves.append(
            LeafLabel(
                level1=current_l1,
                level2=current_l2,
                level3=level3_name,
                description=desc,
            )
        )

    # Deduplicate by full path (keep first occurrence).
    seen: set[str] = set()
    unique: list[LeafLabel] = []
    for leaf in leaves:
        if leaf.path in seen:
            continue
        seen.add(leaf.path)
        unique.append(leaf)

    return unique


def build_label_mappings(leaves: Iterable[LeafLabel]) -> dict[str, object]:
    """
    Build ids and hierarchy masks used by training/inference.
    Returns a JSON-serializable dict.
    """

    leaf_list = list(leaves)
    level1 = sorted({l.level1 for l in leaf_list})
    level1_to_id = {name: i for i, name in enumerate(level1)}

    level2_pairs = sorted({(l.level1, l.level2) for l in leaf_list})
    level2_to_id = {f"{a}/{b}": i for i, (a, b) in enumerate(level2_pairs)}

    leaf_paths = [l.path for l in leaf_list]
    leaf_to_id = {p: i for i, p in enumerate(leaf_paths)}

    level1_to_level2_ids: dict[int, list[int]] = {level1_to_id[a]: [] for a in level1}
    level2_to_leaf_ids: dict[int, list[int]] = {level2_to_id[f"{a}/{b}"]: [] for a, b in level2_pairs}
    for leaf in leaf_list:
        l1_id = level1_to_id[leaf.level1]
        l2_id = level2_to_id[f"{leaf.level1}/{leaf.level2}"]
        l3_id = leaf_to_id[leaf.path]
        level1_to_level2_ids[l1_id].append(l2_id)
        level2_to_leaf_ids[l2_id].append(l3_id)

    for k in list(level1_to_level2_ids.keys()):
        level1_to_level2_ids[k] = sorted(set(level1_to_level2_ids[k]))
    for k in list(level2_to_leaf_ids.keys()):
        level2_to_leaf_ids[k] = sorted(set(level2_to_leaf_ids[k]))

    return {
        "level1": level1,
        "level1_to_id": level1_to_id,
        "level2_pairs": [{"level1": a, "level2": b} for a, b in level2_pairs],
        "level2_to_id": level2_to_id,
        "leaf_labels": [
            {
                "path": l.path,
                "level1": l.level1,
                "level2": l.level2,
                "level3": l.level3,
                "description": l.description,
            }
            for l in leaf_list
        ],
        "leaf_to_id": leaf_to_id,
        "level1_to_level2_ids": {str(k): v for k, v in level1_to_level2_ids.items()},
        "level2_to_leaf_ids": {str(k): v for k, v in level2_to_leaf_ids.items()},
    }
