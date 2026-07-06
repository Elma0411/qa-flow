# Latest Change Guide

更新时间：2026-07-06（Asia/Shanghai）

## Objective

把问答生成里的证据范围控制从“大模型决定”改成“大模型只建议，系统按证据质量裁决”。目标是继续保留候选问题阶段的检索规划能力，但避免模型单方面扩大答案可用证据范围。

## What Changed

- 候选问题 prompt 的证据范围字段改为 `answer_scope_hint`。
  - 这是模型建议范围，只用于诊断和申请放宽上下文。
  - 解析层兼容旧响应里的 `answer_scope`，但新 prompt 会要求输出 `answer_scope_hint`。
- 新增系统最终范围裁决。
  - `answer_scope_policy` 仍来自前端，是当前任务允许的最大范围。
  - 系统会结合同章节/相邻 chunk、top1/top2 分数差距、`must_have_terms` 覆盖率、dense/lexical/structure 综合分，以及未来 `rerank_score`，生成最终 `effective_answer_scope`。
  - `answer_scope` 保留为兼容字段，现在与 `effective_answer_scope` 同义，表示系统最终生效范围。
- 生成上下文按最终范围硬过滤。
  - `source_primary`：只给答案模型主来源块，不再把补充 evidence 塞进上下文后只靠 prompt 约束。
  - `same_section`：只允许同章节或相邻 chunk 的补充证据。
  - `cross_chunk`：只有前端允许、模型建议且检索质量通过阈值时才允许跨 chunk 补证据。
- 调试输出新增裁决解释。
  - 顶层 QA、consolidated JSON、debug QA store、admin 查询都会保留：
    - `answer_scope_hint`
    - `answer_scope`
    - `effective_answer_scope`
    - `answer_scope_decision`
  - `retrieval_trace` 中同步保留这些字段，并在 `raw_semantic_hits` 中增加 `must_term_hits`、`must_term_total`、`must_term_coverage`。
- 前端 QA 详情拆开展示：
  - 模型建议范围
  - 系统最终范围
  - 前端范围策略
  - 范围裁决原因
  - 关键词覆盖率和未来 rerank 分

## Practical Behavior

- 默认 `answer_scope_policy=source_primary` 时，系统永远不会使用补充 evidence。
- 如果希望同章节补证据，把前端最大证据范围调到 `same_section`。
- 如果希望跨 chunk 补证据，把前端最大证据范围调到 `cross_chunk`，但系统仍会因分数差距小、关键词覆盖弱、章节关系差等原因自动收窄。
- 大模型的 `answer_scope_hint` 只能申请放宽，不能直接授权放宽。

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
docker exec qa-flow-runtime python -m py_compile \
  qa/generation/evidence_units.py \
  qa/generation/qa_generation_flow.py \
  qa/prompts/qa_generation_prompts.py \
  qa/pipeline_runtime.py

curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
