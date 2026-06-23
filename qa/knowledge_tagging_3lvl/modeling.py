# 文件作用：定义、保存和加载知识三级标签模型。
# 关联说明：被 predictor 和 scripts/train/evaluate 调用，负责模型定义和持久化。

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass(frozen=True)
class ModelSpec:
    base_model_name: str
    num_level1: int
    num_level2: int
    num_leaf: int
    max_length: int


class HierarchicalTagger(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        num_level1: int,
        num_level2: int,
        num_leaf: int,
        *,
        base_model: Any | None = None,
        base_model_config: Any | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        if base_model is not None:
            self.base = base_model
        elif base_model_config is not None:
            self.base = AutoModel.from_config(base_model_config)
        else:
            self.base = AutoModel.from_pretrained(base_model_name, local_files_only=local_files_only)
        hidden = getattr(self.base.config, "hidden_size", 768)
        self.dropout = nn.Dropout(0.1)
        self.head1 = nn.Linear(hidden, num_level1)
        self.head2 = nn.Linear(hidden, num_level2)
        self.head3 = nn.Linear(hidden, num_leaf)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        return {
            "logits1": self.head1(cls),
            "logits2": self.head2(cls),
            "logits3": self.head3(cls),
        }


def default_hf_endpoint() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def save_model(output_dir: Union[str, Path], model: HierarchicalTagger, tokenizer: Any, spec: ModelSpec) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model_state.pt")
    tokenizer.save_pretrained(out)
    # Save base encoder config for fully offline loading (state_dict already contains encoder weights).
    try:
        model.base.config.to_json_file(str(out / "base_config.json"))
    except Exception:
        pass
    (out / "model_spec.json").write_text(json.dumps(spec.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(model_dir: Union[str, Path], device: torch.device) -> tuple[HierarchicalTagger, Any, ModelSpec]:
    model_dir = Path(model_dir)
    spec = ModelSpec(**json.loads((model_dir / "model_spec.json").read_text(encoding="utf-8")))
    default_hf_endpoint()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    base_config_path = model_dir / "base_config.json"
    base_config = None
    if base_config_path.exists():
        base_config = AutoConfig.from_pretrained(str(base_config_path))

    model = HierarchicalTagger(
        base_model_name=spec.base_model_name,
        num_level1=spec.num_level1,
        num_level2=spec.num_level2,
        num_leaf=spec.num_leaf,
        base_model_config=base_config,
        local_files_only=True,
    )
    state_path = model_dir / "model_state.pt"
    try:
        state = torch.load(state_path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:  # older torch
        state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer, spec
