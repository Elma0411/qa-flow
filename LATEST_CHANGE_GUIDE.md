# 最新变更指南

更新时间：2026-08-12（Asia/Shanghai）

## Objective

收敛同文档证据问答生成的来源字段：删除候选问题阶段的
`source_anchor_text`，保留最终 QA 阶段生成的 `source_fact_text` 作为直接
事实证据，避免两套相近来源摘录并存。

## What Changed

- `qa/prompts/qa_generation_prompts.py`
  - 候选问题 prompt 只要求输出问题和检索规划字段：`retrieval_query`、
    `must_have_terms`、`answer_scope_hint`。
  - 答案 prompt 不再接收或引用 `source_anchor_text`。
  - 最终 QA 的 `source_fact_text` 必须摘自 `qa_generation_unit_text`，并包含
    主来源块的直接证据；使用补充证据时由 `evidence_usage` 说明具体块、片段和用途。
- `qa/generation/qa_generation_flow.py`
  - 候选题归一化不读取或输出 `source_anchor_text`。
  - 答案生成调用不再传递或回写该字段。
- `qa/pipeline_runtime.py` 与 `qa/generation/evidence_units.py`
  - 检索查询回退由问题、标题路径和 `must_have_terms` 组成。
  - 问答生成单元不再保存候选题原文锚点。
- 存储、调试、管理接口和审阅界面删除该字段；Milvus 本身没有此列，因此不需要数据库迁移。
- `INTEGRATION_CONTRACT.md` 与契约测试同步为新的字段职责。

## Expected Behavior

生成链路为：

```text
主来源块 -> LLM 生成 question + 检索规划
        -> 检索并组装 evidence
        -> LLM 生成 answer + source_fact_text + evidence_usage
```

- 候选题模型无需复制主来源块文本，减少无效输出负担。
- `source_fact_text` 只在最终答案阶段产生，供人工审阅、存储兼容和后续评价使用。
- `source_chunk_id` 仍定位主来源块；`qa_generation_unit_text` 是实际证据上下文；
  `evidence_chunk_ids` 与 `evidence_usage` 记录补充证据及其用途。
- 该变更不改变 Milvus 表结构，也不会对旧已存储记录做兼容读取或改写。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m unittest tests.test_qa_generation_contract
python -m py_compile \
  qa/generation/qa_generation_flow.py \
  qa/generation/evidence_units.py \
  qa/pipeline_runtime.py \
  qa/prompts/qa_generation_prompts.py \
  app/services/storage/consolidation.py \
  app/services/debug/qa_store.py \
  app/services/admin/qa_query.py \
  app/routers/doc_chunks.py
node --check static/app_query.js
git diff --check

docker exec qa-flow-runtime bash -lc \
  'cd /app && python -m unittest tests.test_qa_generation_contract'
curl http://localhost:12000/test-connection
```
