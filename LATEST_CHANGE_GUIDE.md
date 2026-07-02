# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

把 QA Flow 静态前端调整为接近 Docker Desktop 的桌面应用风格。三页统一使用左侧导航、
顶部工作区栏、干净的浅色/深色 token、薄边框输入和居中配置 modal；后端接口和字段名
保持不变。

## What Changed

- `ui.js` 新增共享应用壳增强：
  - 自动把三页变成左侧导航 + 顶部工作区栏布局。
  - 顶部栏显示当前页面标题、运行环境提示和主题切换按钮。
- `styles.css` 新增 Docker-like 覆盖层：
  - 移除蓝绿背景 glow、玻璃感和过重阴影。
  - 统一使用 6-8px 圆角、1px 边框、白色 surface、蓝色主按钮。
  - 输入框、select、textarea、按钮和 modal 都改为更接近桌面应用的密度。
- 流水线高级参数从多个模块卡片合并为一个 `Pipeline settings` 入口。
  - 点击后打开居中 modal。
  - modal 内按文档解析、切分、问答生成、评估与过滤、性能、输出分组。
  - 继续复用原有字段 id 和本地缓存 `qa_flow_module_settings_v1`。
- 管理页筛选条件收进 `Filters` 居中 modal。
  - 原位置保留轻量 filter bar。
  - `Apply` 会触发当前查询模式下的列表查询或语义检索。
- 三页静态资源版本号已更新，避免浏览器继续加载旧 CSS/JS。

## Expected Behavior

- `/ui/index.html`、`/ui/admin.html`、`/ui/eval.html` 都显示统一左侧导航和顶部工作区栏。
- 流水线主屏不再铺开大量高级参数，只保留关键启动项和 `Pipeline settings`。
- 参数配置和管理筛选都通过居中 modal 完成。
- API Key 类字段仍不写入模块缓存。
- 现有流水线提交、文档解析、管理查询和评测任务请求参数保持不变。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
