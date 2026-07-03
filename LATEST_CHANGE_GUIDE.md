# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

把流水线页从“旧 section 重新套皮”继续收敛成任务优先的信息架构：主屏按用户操作顺序展示运行任务、当前任务、结果检查和工具箱，避免 LLM/OCR/调试/QA 预览等内容重复抢占主流程。

本次补充修复顶部摘要和工具面板体验：LLM/OCR 激活状态会自动同步到顶部 chip，工具箱里打开的独立工作面板改为大尺寸自适应 modal，并支持手动拖拽缩放。

## What Changed

- 流水线页主顺序调整为：
  - `运行任务`：上传文件、流程模式、每块 QA 数、开始/终止任务、任务设置和环境检测。
  - `当前任务`：合并原“最近一次流水线结果”和“流水线调试视图”，用 `进度耗时 / 输出文件 / 任务历史` tabs 展示。
  - `结果检查`：承载 chunk 溯源和 QA 预览；没有任务或结果时显示空态，不再常驻占满页面。
  - `工具箱`：收起环境检测、LLM 配置、OCR 配置、单独文档解析、知识分类测试、本地 JSON 预览和 QA 管理入口。
- `任务设置` 改为一个总 modal：
  - 仍复用原字段和 DOM id，不改变后端表单字段。
  - modal 内按 `文档解析 / 切分 / 问答生成 / 评估过滤 / 性能并发 / 存储输出` 分 tab。
  - 保存后继续使用本地模块缓存；API Key 类字段仍不缓存。
- 当前任务绑定优化：
  - 选中或恢复 task 后，`chunkTaskId` 会跟随当前 task_id，减少手工重复填写。
  - 任务输出出现时自动切到“输出文件”，提交/轮询时显示“进度耗时”。
- 工具类 section 不再直接铺在主流程里，点击工具箱卡片时才以 modal 打开。
- 顶部 `LLM / OCR` 摘要 chip 会监听激活状态文本变化；在配置弹窗中激活 profile 后不需要刷新页面。
- 工具箱打开的整块工作面板（如文档解析、连接与模型、本地 JSON 预览）使用更宽、更高的 workspace modal，并按可用宽度自适应表单列数。
- workspace modal 右下角新增缩放手柄：
  - 拖动手柄可调整弹窗宽高。
  - 尺寸会写入浏览器 localStorage，下次打开沿用。
  - 双击手柄或聚焦后按 `Home` 可恢复默认尺寸。

## Expected Behavior

- `/ui/index.html` 第一屏只看到运行任务，不再先铺开 LLM/OCR/文档解析/知识分类等工具面板。
- 点击 `任务设置` 会打开一个总配置弹窗，左侧 tab 按流程分组。
- 当前任务的结果、耗时、输出、历史在同一个区域查看，不再出现多个相似调试面板。
- chunk 溯源和 QA 预览只作为结果检查区域出现；工具类能力从底部工具箱进入。
- 激活 LLM 或 OCR 配置后，顶部摘要立即显示 `当前激活: xxx`。
- `文档解析` 等工具弹窗在桌面端接近工作台宽度，内部上传、参数和任务列表不再挤在窄列里；需要时可手动拖动右下角放大或缩小。
- `/ui/eval.html` 的格式、编码、字段映射和任务参数默认不铺在主页面。
- `/ui/admin.html` 的筛选条件仍通过弹窗配置，并显示中文按钮。
- 后端 API、字段名、提交链路不变。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```

本次还用 Playwright 检查了 `/ui/index.html`：

- 桌面视口无控制台错误、无横向溢出。
- 移动视口无横向溢出。
- `任务设置` 能打开 6 个流程 tab。
- `环境检测` 会先打开“连接与模型”modal。
- 模拟更新 `cfgActive` 后，顶部 LLM 摘要 chip 会同步变化。
- `文档解析` 工具 modal 桌面宽度约 1228px，无横向溢出。
- Playwright 拖动缩放手柄后，workspace modal 从约 `1228x891` 调整到约 `1045x743`，并保存到 localStorage；普通任务设置 modal 不显示缩放手柄。
