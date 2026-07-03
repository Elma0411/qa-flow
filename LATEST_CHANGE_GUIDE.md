# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

补全流水线调试面板里 chunk 丢弃原因的中文展示，避免排查时看到难以理解的英文 reason key。

## What Changed

- 前端 `chunk 明细` 的丢弃/错误原因新增中文映射。
- 覆盖本次看到的原因：
  - `summary_source_fact_not_compound`：总结模式下来源事实过于单一，不像多信息点汇总。
  - `ambiguous_reference_answer`：答案里有指代不明的词。
- 同时补全常见生成校验原因，如缺少问题/答案/来源事实、来源事实未定位、单选题选项无效、判断题答案无效等。
- 不改变后端接口、任务状态结构、生成算法或过滤规则。

## Expected Behavior

- 新旧任务在前端调试面板中展示 chunk 丢弃原因时，常见英文 key 会显示为中文。
- 如果出现尚未映射的新 reason，前端仍会回退显示原始 key，便于后续继续补充。

## Validation

```bash
cd /data2/hjk/qa-flow
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
```
