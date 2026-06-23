# 文件作用：初始化知识三级标签训练与预测包的环境。
# 关联说明：与 scripts、modeling、predictor 同级，负责包级环境初始化。

"""Self-contained 3-level taxonomy tagger (rules + small Transformer).

Note: set `HF_ENDPOINT` early so `transformers`/`huggingface_hub` can use a mirror
in CN environments. Users can override it before running scripts.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
