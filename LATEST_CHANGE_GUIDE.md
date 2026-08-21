# Latest Change Guide

更新时间：2026-08-21（Asia/Shanghai）

## Objective

将 QA Flow 桌面端左侧栏的尺寸和排版节奏对齐参考页面
`http://115.154.137.34:12005/`，保留现有导航内容、路由、主题切换和移动端行为。

## Effective Changes

- 桌面侧栏宽度由 `224px` 调整为 `248px`，所有主内容和工作栏偏移统一读取
  `--sidebar-width`，不再存在多处硬编码宽度。
- 品牌区对齐参考尺寸：最小高度 `112px`、内边距 `24px 20px`、标题 `18px`、
  副标题上间距 `6px`。
- 导航区对齐参考节奏：内边距 `18px 12px`、项目高度 `44px`、项目间距
  `5px`、活动项左侧强调线和背景比例一致。
- 侧栏底部主题按钮补充 `14px` 安全边距；移动端恢复紧凑品牌区和横向导航，
  不继承桌面放大尺寸。
- 前端构建标识更新为 `2026-08-21-2`，确保 VPN/WebVPN 不沿用旧侧栏缓存。

## Validation

```bash
docker exec qa-flow-runtime bash -lc 'cd /app && python -m unittest tests.test_ui_cache_contract -v'
bash -ic 'node --check static/app.js && node --check static/admin.js && node --check static/eval.js && node --check static/ui.js'
curl -fsS http://localhost:12000/ui/ | grep '2026-08-21-2'
```

使用 Playwright 在相同 `1440x900` 视口分别渲染 QA Flow 和参考页，已确认侧栏宽度、
品牌区高度、导航起始位置和活动项行高一致。修改文件保持 UTF-8 无 BOM；用户现有的
`AGENTS.md` 修改不纳入本次提交。
