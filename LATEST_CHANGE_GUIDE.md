# Latest Change Guide

更新时间：2026-07-04（Asia/Shanghai）

## Objective

减少完整流水线的重复临时产物，明确 `consolidated JSON/CSV`、`evaluation JSON`、`one_step_debug JSONL` 的职责边界。

## What Changed

- 完整流水线新任务不再单独生成 `*_evaluation_*.json`。
  - LLM/local/无监督评估结果仍会进入最终 `consolidated JSON`。
  - 前端预览、人工审阅入库、Milvus 入库继续以 `consolidated JSON` 为权威结果。
- `consolidated CSV` 继续保留。
  - 它是从 `consolidated JSON` 派生出来的表格导出文件，数据上有重叠，但面向人工下载和表格查看。
- `one_step_debug JSONL` 继续保留。
  - 它包含 chunk 级模型调用、原始响应、检索证据和丢弃原因，是“查看模型原始响应”的排查文件，不和 consolidated 结果合并。
- 旧任务兼容逻辑保留。
  - 已存在的 `evaluation_json/evaluation_json_files` 仍会被状态读取和 artifact lifecycle 识别，避免历史临时文件漏清。

## Artifact Cleanup Rule

- 新生成的完整流水线临时产物主要是 `consolidated JSON/CSV` 和 `one_step_debug JSONL`。
- 默认 TTL 仍是 24 小时：未自动入库的 consolidated 文件和 debug JSONL 会登记到 artifact lifecycle，过期后由后台清理。
- 已自动入库到 Milvus 的任务会清理 consolidated JSON/CSV；debug JSONL 仍保留到 TTL 后再清理，方便短期排查。
- 历史任务里已经存在的 `evaluation JSON` 仍按旧兼容规则参与清理；新任务不再新增它。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile app/services/pipeline_execution/service.py app/routers/pipeline_common.py app/services/pipeline_state/status.py app/services/artifacts/lifecycle.py
git diff --check
```
