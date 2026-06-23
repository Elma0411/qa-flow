# 文件作用：提供知识标签模型训练过程中的指标和工具函数。
# 关联说明：被 scripts/train 使用，提供训练指标、采样和批处理辅助。

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    y1: torch.Tensor
    y2: torch.Tensor
    y3: torch.Tensor
    weight: torch.Tensor


class JsonlClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        label_maps: dict[str, Any],
        max_length: int,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.level1_to_id: dict[str, int] = label_maps["level1_to_id"]
        self.level2_to_id: dict[str, int] = label_maps["level2_to_id"]
        self.leaf_to_id: dict[str, int] = label_maps["leaf_to_id"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        text = r["text"]
        label_path = r["label_path"]
        l1, l2, _l3 = label_path.split("/", 2)
        return {
            "text": text,
            "y1": self.level1_to_id[l1],
            "y2": self.level2_to_id[f"{l1}/{l2}"],
            "y3": self.leaf_to_id[label_path],
            "weight": float(r.get("label_confidence", 1.0)),
        }


def collate_fn(tokenizer: Any, max_length: int):
    def _collate(items: list[dict[str, Any]]) -> Batch:
        texts = [it["text"] for it in items]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return Batch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            y1=torch.tensor([it["y1"] for it in items], dtype=torch.long),
            y2=torch.tensor([it["y2"] for it in items], dtype=torch.long),
            y3=torch.tensor([it["y3"] for it in items], dtype=torch.long),
            weight=torch.tensor([it["weight"] for it in items], dtype=torch.float),
        )

    return _collate


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def weighted_ce_loss(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    weight = weight.to(loss.device)
    return (loss * weight).mean()


def format_steps(current: int, total: int) -> str:
    if total <= 0:
        return str(current)
    width = int(math.log10(total)) + 1
    return f"{current:{width}d}/{total}"

