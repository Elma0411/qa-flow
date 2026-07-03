# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

修复流水线运行时“当前任务”区域被轮询反复抢回、原始 JSON 无法保持展开、状态不够醒目的问题。目标是让任务状态持续刷新，但不打断用户正在查看的 tab 或展开内容。

## What Changed

- `当前任务` 的 tab 状态现在由前端记住：
  - 提交新任务或主动查询任务时，会切到 `进度耗时`。
  - 后续 2 秒轮询只刷新内容，不再强制切回 `进度耗时`。
  - 后台轮询发现任务完成且首次出现输出时，只有用户仍停留在进度页、且未展开原始 JSON，才自动切到 `输出文件`。
- `原始 JSON` 的展开状态会在状态重绘前后保留：
  - 用户展开后，轮询刷新不会自动合上。
  - 用户手动合上后，后续刷新保持合上。
- 流水线调试视图顶部新增更突出的状态头：
  - 中文状态 pill：`排队中 / 运行中 / 已完成 / 失败 / 已终止`。
  - 当前阶段文案优先显示后端 `message`，例如 `图片所在 chunk 摘要生成中`。
  - 运行中显示轻量进度条；没有百分比时使用活动条。
- 状态区视觉层级调整：
  - 状态头部使用更明确的强调边框。
  - 调试分区边框降低权重，减少重复边框造成的视觉噪声。

## Expected Behavior

- 任务运行时，用户点击 `输出文件` 或 `任务历史` 后，不会被下一次轮询自动拉回 `进度耗时`。
- 展开 `原始 JSON` 后，即使任务仍在刷新，也不会马上自动关闭。
- 当前状态和阶段比以前更醒目，不再只藏在灰色小字里。
- 后端 API、任务状态结构、表单字段和提交链路不变。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
```

运行 Docker 服务后建议补充：

```bash
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
