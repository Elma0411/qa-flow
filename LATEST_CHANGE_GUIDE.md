# Latest Change Guide

更新时间：2026-08-13（Asia/Shanghai）

## Objective

把问答生成阶段的可读材料与内部检索追踪信息分离，并让工作台可以打开和提交证据范围设置；同时按 Easy Dataset 的读者场景/信息焦点流程和 Ragas 的单跳证据约束收紧自然问题生成。

## Effective Changes

- 候选问题调用只接收正文，不再把 `chunk_id`、`chunk_index`、`title_path` 或内部分类元数据拼进用户消息。
- generation unit 和答案证据上下文使用 `主材料-N`、`同章节补充-N`、`相关补充-N` 标签，正文不含真实块 ID、位置和标题路径。
- 答案模型只接收候选问题、题型、允许范围、可读证据和可用 `evidence_ref` 标签；模型返回的标签由程序映射为持久化 `evidence_usage[].chunk_id`。
- 真实块 ID、标题路径、检索分数、排名和范围裁决仍保存在 `evidence_hits` / `retrieval_trace` / debug JSONL。
- `openPipelineSettingsModal()` 现在包含 `pipeline.retrieval`，前端摘要同时显示 `answer_scope_policy`，提交逻辑继续发送该值。
- 问题提示词改为“通读材料 -> 选择一个读者场景和信息焦点 -> 写自然问句 -> 静默自检”，并明确禁止把条款前半句改成问题或把完整来源句式搬入问题。总结模式仍是一个问题，只有答案可以组织相关事实。

## Expected Behavior

- LLM 不会因看到内部块 ID、标题路径和检索字段而生成元数据式问题。
- 答案证据仍可追溯到真实 chunk；旧的 QA 持久化字段和评价输入保持不变。
- 工作台的“任务设置 -> 检索证据”可以调整 `source_primary`、`same_section`、`cross_chunk` 并在提交时生效。
- `source_primary` 表示只把当前 generation unit 的主材料交给答案模型；`semantic_top_k` 只控制允许加入的单元外补充块数量。

## Validation

```bash
python -m py_compile \
  qa/generation/structure_units.py \
  qa/generation/evidence_units.py \
  qa/generation/qa_generation_flow.py \
  qa/prompts/qa_generation_prompts.py
node --check static/app.js
python -m unittest tests.test_qa_generation_contract -v
git diff --check
```

Docker runtime verification:

```bash
docker exec qa-flow-runtime sh -lc 'cd /app && python -m unittest tests.test_qa_generation_contract -v'
docker exec qa-flow-runtime sh -lc 'cd /app && python -m compileall app qa scripts'
curl http://localhost:12000/test-connection
```
