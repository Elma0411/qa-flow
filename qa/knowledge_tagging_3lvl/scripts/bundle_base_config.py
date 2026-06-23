# 文件作用：命令行打包基础模型配置到训练产物目录。
# 关联说明：辅助 train.py 的离线部署产物，补齐基础模型配置。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig

from qa.knowledge_tagging_3lvl.modeling import default_hf_endpoint


def main() -> int:
    ap = argparse.ArgumentParser(description="Bundle base encoder config into a trained model dir for offline loading.")
    ap.add_argument("--model-dir", required=True, help="A directory that contains model_spec.json/model_state.pt")
    ap.add_argument(
        "--base-model",
        default="",
        help="Override base model name (default: read from model_spec.json:base_model_name)",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not try to download from HuggingFace; only use local cache",
    )
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    spec_path = model_dir / "model_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"Missing {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base_model_name = str(args.base_model or spec.get("base_model_name") or "").strip()
    if not base_model_name:
        raise ValueError("base_model_name is empty; pass --base-model or fix model_spec.json")

    out_path = model_dir / "base_config.json"
    if out_path.exists():
        print(f"base_config.json already exists: {out_path}")
        return 0

    default_hf_endpoint()
    cfg = AutoConfig.from_pretrained(base_model_name, local_files_only=bool(args.local_files_only))
    cfg.to_json_file(str(out_path))
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

