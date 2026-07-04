# Latest Change Guide

更新时间：2026-07-04（Asia/Shanghai）

## Objective

新增本地第三方开源代码参考区，方便下载成熟项目供 QA Flow 后续改造时对照、模仿和学习，同时避免把参考仓库推送到远程。

## What Changed

- 新建本地目录：`external_repos/`。
  - 用于放第三方开源仓库或参考代码。
  - 每个外部项目应放在独立子目录中。
  - 建议在对应子目录记录来源 URL、分支/commit 和参考目的。
- `external_repos/` 已加入本机 `.git/info/exclude`。
  - 不会出现在普通 `git status`。
  - 不会被提交或 push 到 `origin/master`。
  - 该 exclude 是本机 Git 配置，不会随仓库同步；其他开发用户需要在自己的 `.git/info/exclude` 中添加同样规则。
- `external_repos/` 已单独初始化为 CodeGraph 项目。
  - 后续添加参考代码后，从 QA Flow 根目录执行：

```bash
codegraph sync external_repos
```

  - 大量新增或替换参考仓库后可执行：

```bash
codegraph index --force external_repos
```

- `AGENTS.md` 已记录使用规则。
  - 分析本项目代码仍使用 QA Flow 根目录的 CodeGraph。
  - 分析参考代码时，CodeGraph MCP 传入 `projectPath: "/data2/hjk/qa-flow/external_repos"`。

## Validation

```bash
cd /data2/hjk/qa-flow
git status --short
git status --short --ignored external_repos
codegraph status external_repos
```
