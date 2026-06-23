# 文件作用：加载模型和规则并执行知识标签预测。
# 关联说明：依赖 modeling、rules、text_cleaning，为 app 服务和脚本提供预测入口。

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Union

import torch

from .modeling import load_model
from .rules import classify_with_rules
from .text_cleaning import clean_for_model


@dataclass(frozen=True)
class Prediction:
    label_path: str
    confidence: float
    source: str  # "rule" or "model"
    detail: dict[str, Any]


def _softmax_masked(logits: torch.Tensor, allowed_ids: list[int]) -> tuple[int, float]:
    mask = torch.full_like(logits, fill_value=-1e9)
    mask[allowed_ids] = 0
    masked = logits + mask
    probs = torch.softmax(masked, dim=-1)
    pred = int(probs.argmax(dim=-1).item())
    conf = float(probs.max(dim=-1).values.item())
    return pred, conf


def _load_label_maps(labels_json: Union[str, Path]) -> dict[str, Any]:
    return json.loads(Path(labels_json).read_text(encoding="utf-8"))


class KnowledgeTagger:
    def __init__(
        self,
        labels_json: Union[str, Path],
        model_dir: Union[str, Path],
        device: str = "cpu",
        rule_threshold: float = 0.95,
        model_fallback_threshold: float = 0.35,
    ):
        self.label_maps = _load_label_maps(labels_json)
        self.device = torch.device(device)
        self.model, self.tokenizer, self.spec = load_model(model_dir, device=self.device)

        self.rule_threshold = rule_threshold
        self.model_fallback_threshold = model_fallback_threshold

        self.level1 = self.label_maps["level1"]
        self.level2_pairs = self.label_maps["level2_pairs"]
        self.leaf_labels = self.label_maps["leaf_labels"]
        self.level1_to_level2_ids = {int(k): v for k, v in self.label_maps["level1_to_level2_ids"].items()}
        self.level2_to_leaf_ids = {int(k): v for k, v in self.label_maps["level2_to_leaf_ids"].items()}

    def _select_snippets(self, raw_text: str, *, max_snippets: int = 5, max_chars: int = 2200) -> list[str]:
        """
        Select a few representative snippets from long OCR text for robust inference.

        Rationale:
        - The encoder max_length is small (e.g. 192 tokens).
        - OCR text can be very long; key information may appear in different parts.
        - Using multiple snippets + confidence-weighted voting is more universal than
          hard-coded label-specific rules.
        """
        if not raw_text:
            return []
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        if len(blocks) < 3:
            # Fallback to fixed windows for line-break-heavy OCR.
            t = " ".join(text.split())
            if not t:
                return []
            n = len(t)
            windows: list[str] = []
            head = t[:max_chars]
            windows.append(head)
            if n > max_chars * 2:
                mid_start = max(0, n // 2 - max_chars // 2)
                windows.append(t[mid_start : mid_start + max_chars])
                windows.append(t[-max_chars:])
            return [w for w in dict.fromkeys(windows) if w]

        # Always include a few early blocks (often title/doc number).
        selected: list[str] = []
        selected.extend(blocks[:2])

        # Add spaced blocks across the document.
        idxs = sorted({0, len(blocks) // 3, (2 * len(blocks)) // 3, len(blocks) - 1})
        for i in idxs:
            if 0 <= i < len(blocks):
                selected.append(blocks[i])

        # Add a couple of longer blocks (more content).
        blocks_sorted = sorted(blocks, key=len, reverse=True)
        selected.extend(blocks_sorted[:2])

        # Normalize + cap and dedup while preserving order.
        uniq: list[str] = []
        seen: set[str] = set()
        for s in selected:
            s = s.strip()
            if not s:
                continue
            s = s[: max_chars * 3]  # keep raw a bit longer before clean_for_model
            key = " ".join(s.split())
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(s)
            if len(uniq) >= max_snippets:
                break
        return uniq

    def predict_one(self, text: str) -> Prediction:
        raw_text = text or ""
        rule_text = clean_for_model(raw_text, max_chars=4000)

        rule = classify_with_rules(rule_text)
        if rule and rule.confidence >= self.rule_threshold:
            return Prediction(
                label_path=rule.label_path,
                confidence=rule.confidence,
                source="rule",
                detail={"reason": rule.reason},
            )

        def _predict_model_one(clean_text: str) -> tuple[str, float, dict[str, Any]]:
            enc = self.tokenizer(
                [clean_text],
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            logits1 = out["logits1"][0]
            logits2 = out["logits2"][0]
            logits3 = out["logits3"][0]

            probs1 = torch.softmax(logits1, dim=-1)
            l1_id = int(probs1.argmax().item())
            l1_conf = float(probs1.max().item())

            l2_allowed = self.level1_to_level2_ids.get(l1_id, list(range(logits2.shape[-1])))
            l2_id, l2_conf = _softmax_masked(logits2, l2_allowed)

            leaf_allowed = self.level2_to_leaf_ids.get(l2_id, list(range(logits3.shape[-1])))
            leaf_id, leaf_conf = _softmax_masked(logits3, leaf_allowed)

            label_path = self.leaf_labels[leaf_id]["path"]
            conf = (l1_conf * l2_conf * leaf_conf) ** (1 / 3)

            if conf < self.model_fallback_threshold:
                fallback_id = None
                for cand_id in leaf_allowed:
                    name = self.leaf_labels[cand_id]["level3"]
                    if name in ("其他", "其他主题"):
                        fallback_id = cand_id
                        break
                if fallback_id is not None:
                    label_path = self.leaf_labels[fallback_id]["path"]

            return (
                label_path,
                conf,
                {
                    "level1": self.level1[l1_id],
                    "level1_conf": l1_conf,
                    "level2": self.level2_pairs[l2_id],
                    "level2_conf": l2_conf,
                    "leaf_conf": leaf_conf,
                },
            )

        snippets = self._select_snippets(raw_text, max_snippets=5)
        if not snippets:
            snippets = [rule_text] if rule_text else []
        if not snippets:
            return Prediction(label_path="未分类", confidence=0.0, source="model", detail={"error": "empty_text"})

        votes: dict[str, float] = {}
        per_snippet: list[dict[str, Any]] = []
        best_single = ("", -1.0)
        for idx, snippet in enumerate(snippets):
            clean = clean_for_model(snippet, max_chars=4000)
            if not clean:
                continue
            path, conf, detail = _predict_model_one(clean)
            votes[path] = votes.get(path, 0.0) + float(conf)
            if float(conf) > best_single[1]:
                best_single = (path, float(conf))
            per_snippet.append(
                {
                    "index": idx,
                    "chars": len(snippet),
                    "pred": path,
                    "conf": float(conf),
                    "detail": detail,
                }
            )

        if not votes:
            return Prediction(label_path="未分类", confidence=0.0, source="model", detail={"error": "no_valid_snippet"})

        # Pick by total confidence (sum). Tie-break by best single confidence.
        best_path = max(votes.items(), key=lambda kv: kv[1])[0]
        best_sum = float(votes[best_path])
        conf = min(1.0, best_sum / max(1.0, float(len(per_snippet))))
        label_path = best_path

        return Prediction(
            label_path=label_path,
            confidence=conf,
            source="model",
            detail={
                "snippets": per_snippet,
                "vote_sum": votes,
                "best_single": {"path": best_single[0], "conf": best_single[1]},
            },
        )
