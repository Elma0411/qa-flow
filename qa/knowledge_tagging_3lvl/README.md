# 三级知识标签：规则优先 + 小Transformer 兜底（自包含实验模块）

本目录是一个**独立**的“三级单标签分类”实验模块：从 `qa/dataset/三级知识标签.txt` 解析出叶子标签，先用高置信规则命中（标准号/ISBN/DOI 等），其余交给小型中文 Transformer（默认 `hfl/rbt3`）预测。

> 说明：当前仓库环境直连 `huggingface.co` 可能不稳定，建议在运行前设置镜像：
> - PowerShell：`$env:HF_ENDPOINT='https://hf-mirror.com'`

## 推荐：在 `conda` 的 `pytorch` 环境用 GPU 训练

如果你的 GPU 环境是 `conda env`（例如 `pytorch`），建议用：
- `conda run -n pytorch python -m qa.knowledge_tagging_3lvl.scripts.build_dataset ...`
- `conda run -n pytorch python -m qa.knowledge_tagging_3lvl.scripts.train ... --device cuda --amp`

注意：如需抓取 openstd（国标元数据），环境里需要 `beautifulsoup4`：
`conda run -n pytorch pip install beautifulsoup4`

## 1) 构造数据集（抓取 + 合成）

默认会为每个三级叶子标签生成少量“合成样本”，并可选抓取 `openstd.samr.gov.cn` 的国标元数据（只保存标准号+名称）。

```powershell
python -m qa.knowledge_tagging_3lvl.scripts.build_dataset `
  --labels qa/dataset/三级知识标签.txt `
  --out runtime_assets/knowledge_tagging_3lvl/outputs/dataset `
  --synth-per-label 200 `
  --crawl-openstd `
  --openstd-max-per-type 2000
```

输出：
- `runtime_assets/knowledge_tagging_3lvl/outputs/dataset/train.jsonl`
- `runtime_assets/knowledge_tagging_3lvl/outputs/dataset/val.jsonl`
- `runtime_assets/knowledge_tagging_3lvl/outputs/dataset/test.jsonl`
- `runtime_assets/knowledge_tagging_3lvl/outputs/dataset/labels.json`

## 2) 训练小Transformer（CPU可跑）

```powershell
python -m qa.knowledge_tagging_3lvl.scripts.train `
  --labels qa/dataset/三级知识标签.txt `
  --train runtime_assets/knowledge_tagging_3lvl/outputs/dataset/train.jsonl `
  --val runtime_assets/knowledge_tagging_3lvl/outputs/dataset/val.jsonl `
  --out runtime_assets/knowledge_tagging_3lvl/outputs/model_rbt3 `
  --model-name hfl/rbt3 `
  --epochs 6 `
  --device cuda `
  --amp `
  --batch-size 16 `
  --grad-accum-steps 2 `
  --max-length 192
```

继续训练（在已有微调权重基础上继续）：
`--resume-from runtime_assets/knowledge_tagging_3lvl/outputs/model_rbt3`

## 3) 评估

```powershell
python -m qa.knowledge_tagging_3lvl.scripts.evaluate `
  --labels qa/dataset/三级知识标签.txt `
  --model-dir runtime_assets/knowledge_tagging_3lvl/outputs/model_rbt3 `
  --test runtime_assets/knowledge_tagging_3lvl/outputs/dataset/test.jsonl `
  --device cuda `
  --batch-size 64
```

## 4) 推理（规则优先 + 模型兜底）

```powershell
python -m qa.knowledge_tagging_3lvl.scripts.predict `
  --labels qa/dataset/三级知识标签.txt `
  --model-dir runtime_assets/knowledge_tagging_3lvl/outputs/model_rbt3 `
  --text "GB/T 32843-2016 科技资源标识"
```
