# 文件作用：清洗知识标签训练和预测文本。
# 关联说明：被 govcn、synth、predictor 等模块复用，统一文本清洗。

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\ufeff", " ").replace("\ufffd", " ")
    return " ".join(text.strip().split())


def clean_for_model(text: str, max_chars: int = 2000) -> str:
    """
    Keep only a short, information-dense slice for CPU training/inference.
    """

    text = normalize_whitespace(text)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
