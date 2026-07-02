# Latest Change Guide

更新时间：2026-07-02（Asia/Shanghai）

## Objective

让完整流水线的生成耗时更可解释、可调、可视化。本次改造把
`LLM/VLM API 请求并发` 和 `chunk 生成最大尝试次数` 放到前端任务表单，
并把最近一次流水线结果从原始 JSON 改为中文结构化调试视图。

## What Changed

- 新增任务级 `llm_max_concurrent_requests` 表单参数。
  - 不填写时继续使用 Docker 环境变量 `VLM_API_MAX_CONCURRENT_REQUESTS`。
  - 填写时只影响本次任务，不修改容器全局环境。
  - QA 生成、图片理解、图片契合度判断都会走同一个任务级 LLM/VLM client 并发配置。
- 前端新增 `chunk_max_attempts` 配置项，后端继续映射到生成运行时的
  `strict_max_attempts`。
- `LLMClientConfig`、`VLMClientConfig.from_values()` 和
  `build_llm_client_config()` 支持 `max_concurrent_requests`。
- 生成阶段新增 timing 数据：
  - 文档级：索引构建、chunk 总耗时、候选题生成、检索、答案生成、校验/丢弃。
  - chunk 级：尝试次数、候选题数量、进入答案生成数量、有效 QA 数量、丢弃原因、细分耗时。
  - 检索细分：query embedding、排序/命中、证据组装。
- 前端调试面板改为中文结构化视图：
  - 顶部展示 OCR、生成、无监督评估、评估、总耗时。
  - 中部展示生成阶段小阶段耗时。
  - 底部按 chunk 顺序展示明细。
  - 原始 JSON 保留在默认收起的折叠项里。
- “查看入库记录”按钮改为只要结果里有 `task_id` 或 `milvus_task_id` 就显示。

## Expected Behavior

- `chunk_max_concurrency` 控制同一文件内 chunk worker 数量。
- `llm_max_concurrent_requests` 控制同一 LLM/VLM client 同时外发 API 请求数。
- 当 `chunk_max_concurrency` 大于 `llm_max_concurrent_requests` 时，chunk worker
  仍会并行做本地工作，但外发 LLM/VLM 请求会按 client 并发排队。
- 调试面板不再默认堆英文 JSON，优先按 OCR、生成、评估顺序展示中文字段。

## Validation

```bash
cd /data2/hjk/qa-flow
python -m py_compile app/core/clients.py app/services/llm/client_pool.py app/services/llm/vlm_client.py app/routers/pipeline_batch_routes.py app/routers/pipeline_integrated_routes.py app/services/pipeline_execution/service.py app/services/integrated_pipeline/service.py qa/text_to_qa_pipeline.py qa/pipeline_runtime.py qa/generation/evidence_units.py
node --check static/app.js static/app_config.js static/app_render.js
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:12000/environment-check
```
