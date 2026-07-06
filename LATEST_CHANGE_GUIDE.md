# Latest Change Guide

更新时间：2026-07-06（Asia/Shanghai）

## Objective

修复工作台重构后“开始执行流水线”按钮点击后看起来没有响应的问题。顶部工作台会把原表单按钮移动到表单外，如果继续依赖浏览器对外置 submit 按钮的兼容行为，部分环境可能无法稳定触发表单提交；同时缺少上传文件时，错误提示写在下方状态区，用户在顶部看不到反馈。

## What Changed

- `static/index.html`
  - 给流水线启动按钮增加稳定 id：`pipelineSubmitBtn`。
  - 更新 `app.js` 查询版本号，避免浏览器继续使用旧缓存。
- `static/app.js`
  - 工作台模式下把顶部启动按钮改为显式 `button` 点击处理。
  - 点击时直接调用 `handlePipelineSubmit()`，不再依赖 `form="pipelineForm"` 的外置 submit 行为。
  - `handlePipelineSubmit()` 优先使用 `pipelineSubmitBtn` 作为 loading 按钮。
  - 未选择文件时：
    - 下方任务状态区显示“请先选择要上传的文件”。
    - 自动切换到“进度耗时”tab。
    - 弹出 toast 提示，避免顶部看起来没有响应。

## Expected Behavior

- 点击顶部“开始执行流水线”应稳定触发提交逻辑。
- 如果未选择文件，会立即看到中文提示。
- 如果已选择文件，按钮会进入 loading 状态，并向后端提交原有表单字段。
- 不改变后端 API、请求字段、任务状态结构或生成流程。

## Validation

```bash
cd /data2/hjk/qa-flow

node --check static/app.js
node --check static/app_config.js
node --check static/app_render.js
node --check static/app_runtime.js
node --check static/ui.js

python -m py_compile \
  qa/prompts/qa_generation_prompts.py \
  qa/generation/qa_generation_flow.py \
  qa/grounding/source_fact_grounding.py \
  qa/text_to_qa_pipeline.py \
  qa/pipeline_runtime.py

git diff --check
curl http://localhost:12000/test-connection
```

浏览器验证：

- 刷新 `/ui/`。
- 如果浏览器仍无响应，先强制刷新一次页面，确保加载 `app.js?v=2026-07-06-1`。
- 不选文件点击“开始执行流水线”，应看到 toast 和状态区提示。
- 选择一个小文件点击“开始执行流水线”，按钮应进入 loading，并出现任务进度。
