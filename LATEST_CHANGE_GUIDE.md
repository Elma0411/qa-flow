# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

把 QA Flow 流水线页从“长参数表单”调整为更轻的任务控制台。主页面保留上传、
流程模式、核心数量和执行按钮；详细参数按模块放入右侧配置抽屉，并在浏览器本地
记住最近一次非敏感配置。

## What Changed

- 新增模块化配置入口，覆盖 LLM、OCR、独立文档解析和完整流水线参数。
- 完整流水线主表单现在只保留关键启动项，切分、生成、评估、并发、存储等参数通过
  模块卡片打开右侧抽屉配置。
- 抽屉复用现有 DOM 字段和字段 id，不改变后端表单字段、FastAPI 参数或提交逻辑。
- 新增本地模块缓存 `qa_flow_module_settings_v1`：
  - 每个模块自动记住最近配置。
  - `cfgKey`、`dwVlmApiKey`、`integratedVlmApiKey` 不写入模块缓存。
  - 每个模块提供“恢复默认”和“应用并关闭”。
- 模块卡片展示当前配置摘要，例如切分方式、生成题型、评估策略、LLM/VLM API 请求并发。
- 更新静态资源版本号，避免浏览器继续使用旧版 `app.js` 和 `styles.css`。

## Expected Behavior

- 进入流水线页后，LLM/OCR/文档解析/完整流水线配置区会显示为紧凑模块卡片。
- 点击模块卡片会从右侧打开配置抽屉，字段标签和说明保持中文可读。
- 修改非敏感参数后刷新页面，模块会恢复最近一次配置。
- API Key 类字段不会通过模块缓存恢复。
- 提交完整流水线或独立文档解析任务时，仍使用原来的字段 id 和后端接口。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/app_config.js static/app_render.js static/app_runtime.js
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
