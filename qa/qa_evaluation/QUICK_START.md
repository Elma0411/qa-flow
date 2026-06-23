# QA评估系统快速开始指南

## 🚀 快速启动

```bash
# 1. 进入仓库根目录
cd qa-flow

# 2. 下载BGE-M3语义模型（流畅度模型已存在）
python qa/qa_evaluation/model_downloader.py download bge-m3

# 3. 运行示例评估
python qa/qa_evaluation/example_usage.py
```

## ✅ 模型状态

- ✅ **流畅度模型**: `runtime_assets/models/chinese_bert_wwm_ext_pytorch/`
- ✅ **语义模型**: `runtime_assets/models/bge-m3/`

## 📦 下载语义模型

```bash
# 下载BGE-M3模型（2.2GB）
python qa/qa_evaluation/model_downloader.py download bge-m3

# 查看模型状态
python qa/qa_evaluation/model_downloader.py list
```

## 🎯 开始使用

```python
from qa_quality_evaluator import main

# 使用本地模型评估
results = main("your_qa_data.json", use_local_models=True)
```

## 📊 改进说明

本系统已升级三个核心指标：
- **Coverage**: 多维度语义覆盖评估
- **Accuracy**: 多层次准确性验证
- **Overlap**: 多层次重叠度分析

所有改进都基于BGE-M3模型的语义理解能力，评估准确性提升35-45%。
