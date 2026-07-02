# Latest Change Guide

更新时间：2026-07-02（Asia/Shanghai）

## Objective

优化 QA Flow 静态前端的整体可读性和表单可用性。重点解决长表单压成窄列、
checkbox 与文字错位、参数分组层级弱、按钮区不稳定等问题，让流水线控制台更适合
日常配置和调试。

## What Changed

- 普通前端表单统一改为响应式 grid 布局，字段自动分栏，小屏自动单列。
- 新增 `setupFormPresentation()`，自动给 checkbox、文件上传和普通控制台表单加表现类。
  - 不依赖 CSS `:has()`，避免旧浏览器解析问题。
  - 不改变原有表单字段名、接口参数或提交逻辑。
- checkbox 字段改成横向控制行，说明文字自动换到下一行。
- 文件上传字段单独占一行并限制最大宽度，避免入口控件被挤在角落。
- `details` 高级参数区统一成带边框的子面板，展开后内容有稳定间距。
- 主内容宽度从 1180px 调整到 1360px，减少参数区无意义换行。
- 降低整块卡片 hover 位移，保留边框反馈，减少工具型界面的视觉干扰。
- 为 `0.6 文档解析与图片理解` 关键参数补充中文 helper text。

## Expected Behavior

- `0.6 文档解析与图片理解` 不再出现字段集中在页面中间窄列的问题。
- checkbox、文件上传、VLM 覆盖参数、流水线高级参数会按统一规则对齐。
- 移动端和窄屏下按钮、输入框、参数组会自然堆叠，不挤压文字。
- 这次改动只影响静态前端布局和说明文字，不改变后端接口和运行参数语义。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/app_config.js static/app_render.js
curl http://localhost:12000/environment-check
```
