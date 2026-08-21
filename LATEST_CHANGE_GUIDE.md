# Latest Change Guide

更新时间：2026-08-21（Asia/Shanghai）

## Objective

移除 QA Flow 前端中的国家电网品牌展示和相关字样，保留 QA Flow 自身导航、
功能布局与视觉主题。

## Effective Changes

- 流水线、管理和评测三个页面不再展示原国家电网图片 Logo。
- 原图片不再作为页面 favicon，改用独立的 QA Flow `QA` 标记。
- 评测页面的数据集名称示例由 `sgcc_human_v1` 改为 `qa_human_v1`。
- 清理原 Logo 的桌面端和移动端占位样式，品牌区只保留 QA Flow 文案。
- 前端构建标识和静态资源版本更新为 `2026-08-21-1`，避免代理继续使用旧页面资源。

## Validation

```bash
rg -ni '国家电网|电网|国网|state\s*grid|sgcc|59e73d111ceb75d7cefaa8ec07f9fb07|app-brand-logo' static
bash -ic 'node --check static/app.js && node --check static/admin.js && node --check static/eval.js'
curl -fsS http://localhost:12000/ | grep 'qa-ui-build'
curl -fsS http://localhost:12000/admin.html | grep 'qa-ui-build'
curl -fsS http://localhost:12000/eval.html | grep 'qa-ui-build'
```

修改文件必须为 UTF-8 无 BOM。`AGENTS.md` 的既有本地修改不纳入本次提交。
