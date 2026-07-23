# 最新变更指南

更新时间：2026-07-23（Asia/Shanghai）

## Objective

让 QA Flow 的无监督评估可以在前端选择 NLI、抽取式 QA 和 Embedding
模型，并让新下载的 XLM-R Large、Qwen3 Embedding 模型在标准和集成流水线中真正生效。

## What Changed

- `runtime_assets/models/` 新增以下本地模型目录（模型文件不进入 Git）：
  - `deepset_xlm_roberta_large_squad2`
  - `xlm_roberta_large_xnli`
  - `qwen3_embedding_0_6b`
  - `qwen3_embedding_4b`
- 新增共享评估模型目录白名单和路径校验。空值、`auto`、`default` 继续使用后端默认模型。
- `/batch-upload-complete-pipeline-with-evaluation`、
  `/batch-upload-integrated-document-pipeline` 和 `/eval/jobs` 支持：
  `faithfulness_nli_model`、`answerability_qa_model`、
  `coverage_embedding_model`、`unsupervised_batch_size`。
- 流水线页和独立评测页增加对应下拉框；选择 Qwen3-Embedding-4B 时可将本地评估批量设为 1。
- Coverage 在 CUDA 上加载 Qwen3 Embedding 时强制 FP16；SentenceTransformers 依赖下限提升到 2.7.0。

## Expected Behavior

- 未选择模型时行为保持原状：mDeBERTa / XLM-R Base SQuAD2 / BGE-M3 使用服务默认配置。
- 选择新模型后，任务状态和独立评测结果会记录实际选择的模型名。
- 选择的模型目录不存在或不属于对应评估类型时，任务提交立即返回 400，不会排队后才失败。
- Qwen3-Embedding-0.6B 可作为 BGE-M3 的覆盖度候选；4B 在 11GB 显存上必须小批量运行。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m compileall app qa scripts
python -m unittest tests.test_unsupervised_model_options
docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.debug.yml config
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
