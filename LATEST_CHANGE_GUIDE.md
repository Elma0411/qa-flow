# Latest Change Guide

更新时间：2026-07-10（Asia/Shanghai）

## Objective

把一步式 QA 生成从“按 chunk 出题”升级为“按 generation unit 出题”，并用总题数上限替代前端 `qa_per_chunk` 控制。

## What Changed

- `qa/generation/structure_units.py`
  - 新增轻量结构图、chunk 质量门控、generation unit planner。
  - 支持 `leaf`、`section`、`virtual_parent` 三类 unit。
- `qa/text_to_qa_pipeline.py`、`qa/pipeline_runtime.py`
  - 一步式生成先规划 unit，再按 unit 并发生成。
  - `qa_detail_mode=auto` 时，leaf 走 point，section/virtual_parent 走 summary。
  - 结果和调试信息增加 `qa_generation_unit_*`、`generation_unit_details`、`chunk_quality_details`。
- `qa/generation/evidence_units.py`
  - `build_generation_unit` 支持多 chunk source unit。
  - source chunk 与 evidence 去重；同父级命中覆盖率达到阈值时记录 `auto_merge_trace`。
- `app/routers/*`、`app/services/pipeline_execution/service.py`
  - 新增 `qa_total_limit` 和 `qa_total_limit_scope=per_file|batch`。
  - `qa_per_chunk` 仅保留为兼容旧调用方的 fallback。
  - batch 范围题数上限会预分配到成功文件，避免并发生成超出批次总量。
- `static/index.html`、`static/app.js`
  - 前端移除 `qa_per_chunk` 控件，新增总题数上限和上限范围。
  - 调试面板展示 generation unit 明细。

## Expected Behavior

- 默认前端按 `qa_total_limit=20`、`qa_total_limit_scope=per_file` 提交。
- 标题、目录、图片占位、表格残片、重复页眉页脚等低质量 chunk 不会单独出题。
- 同一父章节下多个可用 chunk 会优先合并为 section unit。
- 长且有内部段落/条款结构的 chunk 会走 virtual_parent summary。
- 最终主问答数量不会超过配置的总题数上限。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m compileall qa app
bash -ic 'node --check static/app.js'
curl http://localhost:12000/test-connection
```
