# Latest Change Guide

更新时间：2026-07-03（Asia/Shanghai）

## Objective

补全流水线排查和人工入库闭环：在 chunk 明细里查看模型原始响应；自动入库后 debug JSONL 继续保留到 TTL；关闭自动入库时，可先人工审阅 QA，再勾选或一键全选入库。

## What Changed

- 新增 `GET /pipeline-tasks/{task_id}/debug-jsonl`。
  - 只读取任务状态中登记过的 debug JSONL basename。
  - 支持按 `chunk_index` 和 `event` 过滤。
  - 返回候选题生成、答案生成、检索 trace、丢弃原因、prompt 和 raw response。
- 新增 `POST /pipeline-tasks/{task_id}/ingest-selected-qa`。
  - 从当前任务未过期的 `consolidated_json` 中读取 QA。
  - 支持 `selected_ids` 勾选入库，也支持 `select_all_task` 一键全选当前任务。
  - 入库成功后更新任务输出记录的 `history_source`、`milvus_task_id`、`vector_storage_result` 和 manual ingest 元数据。
- 自动 Milvus 入库成功后不再删除 `debug_jsonl` / `debug_json_files`。
  - consolidated JSON/CSV/evaluation 仍可按原逻辑清理。
  - debug JSONL 会继续注册到 artifact lifecycle，保留到 `artifacts_expire_at` 后自动清理。
- 前端流水线调试面板：
  - chunk 明细新增“查看”按钮，打开“模型原始响应”弹窗。
  - 长字段默认折叠，不写入 localStorage。
- 前端 QA 预览：
  - 未自动入库且临时 JSON 未过期时显示“人工审阅入库”工具栏。
  - 支持逐条勾选、清空选择、一键全选当前任务、入库所选 QA。
- 存储输出文案统一：
  - `写入 Milvus 向量库` -> `自动入库 QA 到向量库`
  - `chunk 入库` -> `保存 chunk 溯源索引`
  - `入库失败即失败` -> `溯源索引失败时终止任务`
  - `同步等待完成` -> `提交后等待任务完成再返回`

## Expected Behavior

- 自动入库任务完成后，仍可在 TTL 内从 chunk 明细查看模型原始响应。
- 关闭“自动入库 QA 到向量库”后，任务完成会保留 consolidated JSON；QA 预览区可人工勾选入库。
- 一键全选当前任务不会让前端传所有 QA ID，而是由后端按任务 consolidated JSON 读取。
- debug JSONL 过期或缺失时，前端显示中文空态/错误提示。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile app/routers/pipeline_history_routes.py app/services/milvus/store_search.py app/services/milvus/service.py app/services/pipeline_execution/service.py app/services/pipeline_state/status.py app/routers/pipeline_batch_routes.py app/routers/pipeline_integrated_routes.py app/services/doc_chunks/service.py
node --check static/app.js static/admin.js static/eval.js static/app_config.js static/app_render.js static/app_runtime.js static/ui.js
git diff --check
```

Docker runtime smoke:

```bash
docker compose -f docker/docker-compose.yml restart qa-flow-runtime
curl http://localhost:12000/test-connection
curl http://localhost:12000/environment-check
```
