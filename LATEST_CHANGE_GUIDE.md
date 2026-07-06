# Latest Change Guide

更新时间：2026-07-06（Asia/Shanghai）

## Objective

让问答生成的 `qa_detail_mode=point` 和 `qa_detail_mode=summary` 真正成为两套不同生成策略。候选问题阶段现在会知道当前粒度模式，答案阶段也会按同一模式约束 `source_fact_text` 和证据使用，避免单点题被生成成多事实总结，或总结题只有一个孤立事实。

## What Changed

- 候选问题 prompt 新增粒度模式契约。
  - `point` 只允许单事实、单答案方向的问题。
  - `summary` 只允许需要 2 个以上相关事实共同回答的问题。
  - `answer_scope_hint` 仍只是模型建议，最终证据范围仍由系统裁决。
- 答案生成 prompt 新增粒度模式契约。
  - `point` 要求 `source_fact_text` 是单个 atomic fact，不能多句、分号或跨行。
  - `summary` 要求 `source_fact_text` 至少包含 2 个证据片段，使用分号或换行分隔，并在 `evidence_usage` 中覆盖关键证据。
- 生成调用链会把当前 `qa_detail_mode` 传入候选问题 LLM。
  - debug JSONL 的 `candidate_question_llm_call` 会记录 `qa_detail_mode`。
  - 不改变前端字段、后端 API、最终 QA 主字段结构。
- summary 校验收紧。
  - `summary_source_fact_segments_insufficient` 表示总结型来源事实没有拆出至少 2 个有效片段。
  - summary 的每个有效片段都必须能在 `qa_generation_unit_text` 中定位。
  - 主来源锚定会按拆分后的片段检查，至少一个片段需要锚回主来源块或候选问题原文锚点。
- 前端丢弃原因补充中文解释。

## Practical Behavior

- 使用 `point` 时，模型会更倾向于生成“谁/何时/什么条件/什么材料/什么阈值”这类单点直答问题。
- 使用 `summary` 时，模型会更倾向于生成流程、清单、条件集合、职责分工、处理规则、对比归纳类问题。
- 如果 summary 只生成了一个事实片段，会被丢弃并显示“总结模式下来源事实片段不足”。
- 如果 point 生成了多句或分号拼接的来源事实，会继续按单点规则丢弃。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile \
  qa/prompts/qa_generation_prompts.py \
  qa/generation/qa_generation_flow.py \
  qa/grounding/source_fact_grounding.py \
  qa/text_to_qa_pipeline.py \
  qa/pipeline_runtime.py

node --check static/app.js

git diff --check
git status --short
git status --short --ignored external_repos
```

Docker runtime 可用时再做：

```bash
docker exec qa-flow-runtime python -m py_compile \
  qa/prompts/qa_generation_prompts.py \
  qa/generation/qa_generation_flow.py \
  qa/grounding/source_fact_grounding.py \
  qa/text_to_qa_pipeline.py \
  qa/pipeline_runtime.py

curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
