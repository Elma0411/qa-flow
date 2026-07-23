# 最新变更指南

更新时间：2026-07-23（Asia/Shanghai）

## Objective

改善 Chunk 溯源“按 QA 看”的分栏布局，让长问题在 QA 列表中正常换行，
并把主要可用高度留给 QA 详情。

## What Changed

- QA 列表禁止横向滚动，只保留纵向滚动。
- QA 问题标题最多显示两行；长文本可在任意必要位置换行，不再被全局按钮的
  单行样式撑宽列表。
- QA 元信息允许换行，列表项宽度始终限制在列表容器内。
- 窄容器下 QA 列表高度改为 `160px` 至 `220px` 的受限区域，QA 详情使用
  剩余高度并独立滚动。
- 更新三个前端页面的 CSS 资源版本号，避免浏览器沿用旧缓存。

## Expected Behavior

- 长 QA 问题在列表中最多展示两行，不出现横向滚动条。
- QA 数量较多时只滚动左侧或上方列表，不拉高整个工作台。
- 工作台较窄并改为上下布局时，QA 详情仍占据主要空间。
- 点击列表项和双击打开详情弹窗的交互保持不变。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app_query.js
git diff --check
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
