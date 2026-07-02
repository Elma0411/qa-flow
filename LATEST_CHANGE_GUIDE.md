# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

把静态前端从“参数长表单”改成更轻的产品工作台：主屏只保留日常启动和查询动作，复杂参数进入独立配置弹窗。视觉上按 Open Design / product UI guidelines 收敛为安静、紧凑、可扫描的控制台风格。

## What Changed

- 安装本地 Codex skill：`/home/lich/.codex/skills/web-design-guidelines/SKILL.md`。
- 流水线页改为生成任务台：
  - 主屏保留上传、文档解析模式、每块数量、开始/终止任务和运行状态摘要。
  - Pipeline 参数拆成 6 个独立入口：文档解析、切分、问答生成、评估过滤、性能并发、存储输出。
  - 每个入口打开居中 modal；字段仍复用原 DOM id，提交参数不变。
- 评测页降噪：
  - 主屏只保留上传、预览、启动/终止任务和任务状态。
  - 解析方式、字段映射、任务参数移动到“评测设置”modal。
- 通用 UI 改造：
  - 三页统一使用新的工作台 token、左侧导航、顶部工作栏和无横向溢出的布局。
  - 文件选择改成中文自定义控件，原生 file input 仍保留用于提交。
  - 管理页 `Filters` 改为中文“筛选条件”，modal 按中文按钮操作。
  - 高级字段 helper 文本在 modal 中压缩显示，避免说明文字占满屏幕。

## Expected Behavior

- `/ui/index.html` 第一屏是生成任务台，不再先铺开 LLM/OCR/流水线高级参数。
- 流水线高级参数分别从模块卡打开，不再用一个超长表单承载所有字段。
- `/ui/eval.html` 的格式、编码、字段映射和任务参数默认不铺在主页面。
- `/ui/admin.html` 的筛选条件仍通过弹窗配置，并显示中文按钮。
- API Key 类字段不进入模块缓存；后端 API、字段名、提交链路不变。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```

Playwright 已在本机安装 Chromium 缓存，用于检查三页桌面视口截图和横向溢出。
