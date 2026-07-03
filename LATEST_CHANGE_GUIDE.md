# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

修复流水线调试视图在任务轮询刷新时反复跳回顶部的问题，并确认运行过程有可展示的实时计时。目标是任务状态继续自动刷新，但不打断用户查看原始 JSON 或页面中部内容。

## What Changed

- 前端状态刷新会保留滚动位置：
  - 刷新前记录页面 `window` 滚动位置。
  - 记录 `原始 JSON` 内部 `pre` 的横向和纵向滚动位置。
  - 重绘后立即恢复，并在下一帧再恢复一次，避免布局更新造成回跳。
- `原始 JSON` 仍保持展开/关闭状态，不会因 2 秒轮询自动合上。
- 调试面板新增 `实时运行` 耗时：
  - 优先从任务 `started_at` 计算。
  - 如果 integrated 预处理阶段还没进入完整流水线，则退回使用任务 `updated_at`，保证排队/dispatch 阶段也有可见耗时。
- 后端为标准和 integrated 的 `file_progress[filename].stages[stage]` 写入通用 stage 计时：
  - `started_at`
  - `updated_at`
  - `elapsed_seconds`
  - `completed_at`（终态时）
- 标准和 integrated 新任务状态都会写入任务级 `created_at`，前端实时计时优先使用 `started_at/created_at`；老任务没有这些字段时使用页面首次看到该 task 的时间兜底。
- 前端阶段耗时读取增强：
  - OCR、生成、无监督评估、评估优先使用原有专用 timing。
  - 专用 timing 暂未产出时，使用 stage `elapsed_seconds` 作为 fallback。
- `INTEGRATION_CONTRACT.md` 已补充 stage 计时字段的生产者、语义和 fallback 用途。

## Expected Behavior

- 任务运行时，停在调试面板中部或原始 JSON 内部滚动查看，不会被下一次轮询拉回顶部。
- 原始 JSON 的横向滚动条和纵向滚动条位置会保持。
- 运行早期即使 OCR/生成等阶段还没完成，也会看到 `实时运行` 时间增长。
- 进入具体阶段后，对应阶段会逐步有 `elapsed_seconds` 或专用 timing，不再只能显示 0 或未记录。
- 后端 API、提交参数和任务查询接口不变；只是任务状态里的 stage 条目增加可选计时字段。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile app/core/time_utils.py app/routers/pipeline_batch_routes.py app/routers/pipeline_integrated_routes.py app/services/pipeline_execution/service.py
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
