# Latest Change Guide

更新时间：2026-07-04（Asia/Shanghai）

## Objective

修复流水线工作台里存储设置不可见、终止任务入口含糊、Chunk QA 详情撑高页面的问题，并明确临时流水线产物的 24 小时清理范围。

## What Changed

- 工作台顶部摘要新增 `存储` chip。
  - 显示 `自动入库/人工审阅`、`保存溯源/不保存溯源`、`等待完成/异步返回`。
  - 点击会打开任务设置里的 `存储输出` 面板。
- 终止任务入口拆分为两个清晰动作。
  - 主操作区按钮改为 `终止当前任务`。
  - task_id 输入区新增 `终止此 task_id`。
  - 修复点击事件对象可能被误当成 task_id 的问题。
- Chunk 溯源的右侧 `QA 详情` 固定在调试面板高度内。
  - 长内容在右侧详情区域内部滚动，不再把页面撑得很长。
- 静态资源版本更新到 `2026-07-04-1`，避免浏览器使用旧缓存。

## Artifact Cleanup Rule

- 流水线临时产物默认 TTL 是 24 小时：`consolidated JSON/CSV`、`evaluation JSON`、`one_step_debug JSONL` 等会登记到 artifact lifecycle，过期后由后台清理。
- `pipeline_jobs_store.json`、`llm_configs.json`、`ocr_configs.json`、`artifact_lifecycle_registry.json`、SQLite 元数据等长期状态/配置文件不会按 24 小时删除。
- 已自动入库到 Milvus 的任务可清理 consolidated JSON/CSV，但 debug JSONL 会保留到 TTL 后再清理。
- 任务历史记录本身不会因为 24 小时 TTL 自动消失；只是临时产物过期后，历史记录里可下载/预览的文件会不可用。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
```
