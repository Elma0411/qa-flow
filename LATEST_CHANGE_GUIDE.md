# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

修复流水线调试面板的计时口径和 chunk 明细展示，让主界面只展示墙钟耗时，避免把并发 worker 的累计耗时误认为真实等待时间。

## What Changed

- QA 生成 worker 现在记录候选题生成、检索、答案生成、校验/丢弃的绝对时间区间。
- 文档级生成完成时新增墙钟归因结果 `generation_wall_detail`：
  - `candidate_question_seconds`
  - `retrieval_seconds`
  - `answer_generation_seconds`
  - `validation_and_bookkeeping_seconds`
  - `scheduler_gap_seconds`
  - `document_total_seconds`
- 保留 `generation_cumulative_detail` 作为并发累计诊断；前端主耗时不再使用它。
- 任务终态 message 改为中文，例如 `任务完成：1 成功，0 失败`，旧英文终态在前端也会映射为中文。
- 前端调试面板改为：
  - 大流程：文档解析、问答生成、评估、存储输出、总耗时。
  - QA 生成细分：候选题、检索、答案、校验/丢弃、调度/等待，细分合计应接近 QA 生成合计。
  - chunk 明细：固定高度表格，超出后内部滚动，丢弃原因/错误默认折叠。

## Expected Behavior

- 新任务完成后，`generation_wall_detail` 中的小阶段耗时相加应与 `document_total_seconds` 基本一致。
- 无监督评估归入 `评估`，不再作为大流程里的独立重复项。
- 旧任务如果没有墙钟细分，前端会提示“旧任务未记录墙钟细分”，不再展示误导性的累计小阶段。
- 20 个以上 chunk 的明细不会撑高整个页面。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile app/core/time_utils.py app/routers/pipeline_batch_routes.py app/routers/pipeline_integrated_routes.py app/services/pipeline_execution/service.py qa/text_to_qa_pipeline.py qa/pipeline_runtime.py
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
docker compose -f docker/docker-compose.yml restart qa-flow-runtime
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
