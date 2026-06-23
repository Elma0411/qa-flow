# 文件作用：命令行执行知识三级标签预测。
# 关联说明：调用 predictor.py，对训练产物执行命令行预测。

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa.knowledge_tagging_3lvl.predictor import KnowledgeTagger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to 三级知识标签.txt (currently unused at predict time)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    labels_json = Path(args.model_dir) / "labels.json"
    if not labels_json.exists():
        raise FileNotFoundError(f"Missing {labels_json} (train.py writes it).")

    tagger = KnowledgeTagger(labels_json=labels_json, model_dir=args.model_dir, device=args.device)
    pred = tagger.predict_one(args.text)
    print(json.dumps(pred.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

