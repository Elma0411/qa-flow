# Latest Change Guide

更新时间：2026-07-04（Asia/Shanghai）

## Objective

修复流水线工作台中 Chunk 溯源右侧 QA 预览过长、撑高整个页面的问题，让左右审阅面板保持固定高度，内容多时在面板内部滚动。

## What Changed

- 工作台审阅区 `Chunk 溯源` 和 `QA 预览` 面板改为固定视口高度。
  - 桌面端高度为 `min(74vh, 820px)`。
  - 中小屏使用更适合窗口的高度限制。
- 右侧 `QA 预览` 的 `#qaResults` 改为内部滚动。
  - QA 数量很多时不再把页面撑得很长。
  - 顶部标题保持在面板内，列表内容独立滚动。
- 左侧 Chunk 溯源面板也补齐固定高度上下文。
  - Tree、Chunk 调试面板、按 QA 看详情都在各自容器内滚动。
  - `按 Chunk 看` 里的“该块关联 QA”列表增加最大高度和内部滚动。
- 静态样式版本更新到 `2026-07-04-2`，避免浏览器沿用旧 CSS 缓存。

## Expected Behavior

- 打开流水线工作台后，Chunk 溯源和右侧 QA 预览保持同一高度，不再把整页拉长。
- QA 预览很多时，只滚动右侧 QA 列表区域。
- Chunk 树、Chunk 调试、QA 详情内容很多时，只滚动对应内部区域。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js static/app_query.js
git diff --check
```
