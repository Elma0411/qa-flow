# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

补齐流水线顶部摘要栏中的评估配置入口，让主界面展示内容与 `任务设置` modal 的模块保持一致。

## What Changed

- 顶部摘要栏新增 `评估` chip。
- `评估` chip 显示当前评估配置摘要：
  - 评估关闭时显示 `关闭`。
  - 评估开启时显示 `评估方法 / 是否过滤`，例如 `llm / 过滤 0.7`。
- 点击顶部 `评估` chip 会打开 `任务设置` 中的 `评估过滤` 模块。
- 任务设置卡片和顶部 chip 复用同一个 `pipelineEvaluationSummary()` 函数，避免两处摘要不一致。
- 前端脚本版本号更新，方便浏览器刷新到最新静态资源。

## Expected Behavior

- 主界面顶部摘要现在包含 `LLM / OCR / 流程 / 切分 / 生成 / 评估 / 并发`。
- 修改任务设置里的评估方式、过滤开关或阈值后，顶部 `评估` 摘要会同步更新。
- 后端 API、表单字段和提交链路不变。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
```
