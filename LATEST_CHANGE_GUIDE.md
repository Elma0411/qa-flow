# Latest Change Guide

更新时间：2026-07-05（Asia/Shanghai）

## Objective

优化“候选问题 -> 同文档检索 -> evidence 答案生成”的检索质量和可调试性，并把第三方参考项目下载到本地-only 参考区，供后续继续学习 RAG、prompt 和 rerank 方案。

## What Changed

- 问答生成检索默认从 `semantic` 升级为 `hybrid`。
  - `semantic` 仍是旧逻辑：归一化 dense embedding + dot product。
  - `hybrid` 会融合 dense 分数、词项匹配分数、同章节/相邻 chunk/title_path 结构加权。
  - 分数接近时会先取 dense 与词项候选池，再按综合分轻量重排。
- 候选问题 prompt 新增检索规划字段。
  - `retrieval_query`：用于检索 evidence 的短查询。
  - `must_have_terms`：检索时必须重点命中的实体/动作/条件词。
  - `answer_scope`：`source_primary`、`same_section`、`cross_chunk`。
- 答案生成 prompt 会接收检索规划字段和证据范围策略。
  - 默认仍以主来源块为第一依据。
  - 只有配置允许时，才使用同章节或跨 chunk evidence。
  - 最终 QA 会记录 `evidence_usage`、`retrieval_trace` 和选中 evidence 的评分诊断。
- 前端流水线参数新增“检索证据”模块。
  - 可配置：检索排序模式、evidence 数量、轻量重排候选数、dense/lexical/structure 权重、答案证据范围。
  - 任务状态面板显示 resolved `retrieval_config`。
  - QA 详情里的检索诊断改为中文字段，并展示综合/向量/词项/结构分、排序、top1-top2 差距、同章节/相邻块标记。
- consolidated JSON、debug QA store、admin 查询响应会保留顶层检索字段。
  - `retrieval_query`
  - `must_have_terms`
  - `answer_scope`
  - `evidence_usage`
  - `retrieval_trace`
- 本地参考目录仍使用 `external_repos/`。
  - 该目录已被 `.git/info/exclude` 忽略，不会提交或 push。
  - 下载完成后执行 `codegraph sync external_repos`，让 CodeGraph 可以检索参考代码。

## Practical Defaults

- `retrieval_mode=hybrid`
- `semantic_top_k=3`
- `rerank_top_n=12`
- `hybrid_weight_dense=0.68`
- `hybrid_weight_lexical=0.24`
- `retrieval_structure_weight=0.08`
- `answer_scope_policy=source_primary`

如果追求更强跨块补证据，可以把 `answer_scope_policy` 调成 `same_section` 或 `cross_chunk`，但建议抽查答案是否发生证据漂移。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile \
  qa/generation/evidence_units.py \
  qa/generation/qa_generation_flow.py \
  qa/prompts/qa_generation_prompts.py \
  qa/pipeline_runtime.py \
  qa/text_to_qa_pipeline.py \
  app/services/pipeline_execution/service.py \
  app/services/storage/consolidation.py \
  app/services/debug/qa_store.py \
  app/services/admin/qa_query.py \
  app/routers/pipeline_batch_routes.py \
  app/routers/pipeline_integrated_routes.py

node --check static/app.js
node --check static/app_query.js
node --check static/app_config.js
node --check static/app_render.js
node --check static/app_runtime.js
node --check static/ui.js

git diff --check
git status --short
git status --short --ignored external_repos
```

Docker runtime 可用时再做：

```bash
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
