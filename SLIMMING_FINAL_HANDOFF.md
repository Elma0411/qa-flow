# 代码瘦身最终交付文档

更新时间：2026-04-16（Asia/Shanghai）

## 文档目的

这份文档汇总当前这一轮代码瘦身到目前为止的全部有效改动。

你可以直接用它回答下面几个问题：

1. 这轮瘦身到底改了什么。
2. 哪些文件是新增的。
3. 哪些文件需要覆盖上传。
4. 服务器这次最小要传哪些文件。
5. 哪些模型、数据、目录完全不用重传。

## 运行时目录重构补充

在原有瘦身基础上，当前又额外完成了一次运行时资源目录重构，目的不是继续
拆代码，而是把原来分散的大目录统一收到一个根目录里，方便你以后打包和上
传时只排除一处目录。

当前统一目录为：

- `runtime_assets/`

它下面承载的内容包括：

- `runtime_assets/models/`
- `runtime_assets/cache/torch/`
- `runtime_assets/cache/transformers/`
- `runtime_assets/cache/huggingface/`
- `runtime_assets/outputs/`
- `runtime_assets/uploads/`
- `runtime_assets/volumes/`
- `runtime_assets/knowledge_tagging_3lvl/outputs/`

这样以后你打压缩包时，主要只需要记住排除 `runtime_assets/` 这一处。

## 本轮瘦身的总结果

这轮瘦身的核心目标始终一致：
在不改现有接口契约和业务行为的前提下，把体量过大、职责混杂的文件按层拆开，
让后续开发、排障、继续迭代更可控。

截至当前，已经完成的关键收敛有 9 组：

- 一步式 QA 主编排收敛
- 流水线路由主文件收敛
- storage service 收敛
- 首页主脚本 `app.js` 收敛
- 管理端 `admin_v1` 收敛
- Milvus 总服务 `milvus_service` 收敛
- 管理端服务 `admin_qa_service` 收敛
- 无监督评测服务 `unsupervised_evaluation_service` 收敛
- 评测作业服务 `eval_job_service` 收敛

当前与瘦身直接相关的核心结构如下。

### 一步式 QA 侧

- `qa/prompts/qa_generation_prompts.py`
- `qa/generation/qa_generation_flow.py`
- `qa/pipeline_runtime.py`
- `qa/validation/qa_item.py`
- `qa/grounding/source_fact_grounding.py`
- `qa/text_to_qa_pipeline.py`

### 流水线路由侧

- `app/routers/pipeline_common.py`
- `app/routers/pipeline_generation_routes.py`
- `app/routers/pipeline_evaluation_routes.py`
- `app/routers/pipeline_batch_routes.py`
- `app/routers/pipeline_history_routes.py`
- `app/routers/pipeline.py`

### storage service 侧

- `app/services/storage_uploads.py`
- `app/services/storage_paths.py`
- `app/services/storage_consolidation.py`
- `app/services/storage_service.py`

### 首页脚本侧

- `static/app_runtime.js`
- `static/app_config.js`
- `static/app_render.js`
- `static/app_query.js`
- `static/app.js`
- `static/index.html`

### 管理端路由侧

- `app/routers/admin_v1_common.py`
- `app/routers/admin_v1_item_routes.py`
- `app/routers/admin_v1_job_routes.py`
- `app/routers/admin_v1.py`

### Milvus 服务侧

- `app/services/milvus_runtime.py`
- `app/services/milvus_meta_utils.py`
- `app/services/milvus_collection_service.py`
- `app/services/milvus_embedding_service.py`
- `app/services/milvus_store_search.py`
- `app/services/milvus_service.py`

### 管理端服务侧

- `app/services/admin_qa_common.py`
- `app/services/admin_qa_query_service.py`
- `app/services/admin_qa_write_service.py`
- `app/services/admin_qa_service.py`

### 无监督评测服务侧

- `app/services/unsupervised_eval_runtime.py`
- `app/services/unsupervised_eval_common.py`
- `app/services/unsupervised_eval_aggregation.py`
- `app/services/unsupervised_eval_runners.py`
- `app/services/unsupervised_eval_suite.py`
- `app/services/unsupervised_evaluation_service.py`

### 评测作业服务侧

- `app/services/eval_job_common.py`
- `app/services/eval_job_run_service.py`
- `app/services/eval_job_result_service.py`
- `app/services/eval_job_service.py`

## 各阶段改动摘要

下面按阶段说明目前已经完成的瘦身内容。

### Phase 1：前端共享工具收敛

这一步处理的是前端重复、容易互相覆盖的基础代码。

主要结果：

- 收敛共享前端工具到 `static/ui.js`
- 清理 `static/app.js` 中重复定义的问题
- 抽出最小共享评分组渲染能力

涉及文件：

- `static/ui.js`
- `static/app.js`
- `static/admin.js`
- `static/index.html`
- `static/admin.html`

### Phase 2：抽离 prompt / template 层

主要结果：

- 新增 `qa/prompts/qa_generation_prompts.py`

### Phase 3：抽离 chunk 级生成流程层

主要结果：

- 新增 `qa/generation/qa_generation_flow.py`

### Phase 4：抽离 runtime 配置与 worker 层

主要结果：

- 新增 `qa/pipeline_runtime.py`

### Phase 5：抽离 item 校验与归一化层

主要结果：

- 新增 `qa/validation/qa_item.py`

### Phase 6：抽离 grounding 层

主要结果：

- 新增 `qa/grounding/source_fact_grounding.py`

### Phase 7：拆分流水线路由主文件

主要结果：

- 新增 `app/routers/pipeline_common.py`
- 新增 `app/routers/pipeline_generation_routes.py`
- 新增 `app/routers/pipeline_evaluation_routes.py`
- 新增 `app/routers/pipeline_batch_routes.py`
- 新增 `app/routers/pipeline_history_routes.py`
- 将 `app/routers/pipeline.py` 改成聚合入口

### Phase 8：拆分 storage service

原来的 `app/services/storage_service.py` 同时承载了上传读取、批量落盘、
输出路径、consolidated 组装、CSV 导出、outputs 清理等职责。

主要结果：

- 新增 `app/services/storage_uploads.py`
- 新增 `app/services/storage_paths.py`
- 新增 `app/services/storage_consolidation.py`
- 将 `app/services/storage_service.py` 改成聚合导出入口

### Phase 9：拆分首页主脚本 `app.js`

原来的 `static/app.js` 同时承载了运行时状态、配置管理、结果渲染、
流水线提交、chunk 查询、数据库/本地查询等多类职责。

主要结果：

- 新增 `static/app_runtime.js`
- 新增 `static/app_config.js`
- 新增 `static/app_render.js`
- 新增 `static/app_query.js`
- 将 `static/app.js` 收敛为页面主编排脚本
- 更新 `static/index.html` 的脚本加载链路

### Phase 10：拆分管理端总路由 `admin_v1.py`

原来的 `app/routers/admin_v1.py` 同时承载了共享模型、QA 条目治理、
语义搜索、评测任务、导出任务和 job 路由。

主要结果：

- 新增 `app/routers/admin_v1_common.py`
- 新增 `app/routers/admin_v1_item_routes.py`
- 新增 `app/routers/admin_v1_job_routes.py`
- 将 `app/routers/admin_v1.py` 改成聚合入口

### Phase 11：拆分 Milvus 总服务 `milvus_service.py`

原来的 `app/services/milvus_service.py` 同时承载了运行时对象、建表、
embedding、入库、检索、meta 压缩等多类职责。

主要结果：

- 新增 `app/services/milvus_runtime.py`
- 新增 `app/services/milvus_meta_utils.py`
- 新增 `app/services/milvus_collection_service.py`
- 新增 `app/services/milvus_embedding_service.py`
- 新增 `app/services/milvus_store_search.py`
- 将 `app/services/milvus_service.py` 改成聚合导出入口

### Phase 12：拆分管理端服务 `admin_qa_service.py`

原来的 `app/services/admin_qa_service.py` 同时承载了 Milvus 公共校验、
表达式拼装、查询、详情、replace、delete 和导出。

主要结果：

- 新增 `app/services/admin_qa_common.py`
- 新增 `app/services/admin_qa_query_service.py`
- 新增 `app/services/admin_qa_write_service.py`
- 将 `app/services/admin_qa_service.py` 改成聚合导出入口

### Phase 13：拆分无监督评测服务 `unsupervised_evaluation_service.py`

原来的 `app/services/unsupervised_evaluation_service.py` 同时承载了可选依赖、
运行时标志、单指标执行器、suite 聚合、GPU stage 编排。

主要结果：

- 新增 `app/services/unsupervised_eval_runtime.py`
- 新增 `app/services/unsupervised_eval_common.py`
- 新增 `app/services/unsupervised_eval_aggregation.py`
- 新增 `app/services/unsupervised_eval_runners.py`
- 新增 `app/services/unsupervised_eval_suite.py`
- 将 `app/services/unsupervised_evaluation_service.py` 改成聚合导出入口

### Phase 14：拆分评测作业服务 `eval_job_service.py`

原来的 `app/services/eval_job_service.py` 同时承载了任务执行、结果写出、
分页读取、按阈值入库和通用工具。

主要结果：

- 新增 `app/services/eval_job_common.py`
- 新增 `app/services/eval_job_run_service.py`
- 新增 `app/services/eval_job_result_service.py`
- 将 `app/services/eval_job_service.py` 改成聚合导出入口

## 新增文件清单

下面这些文件是这轮瘦身新增出来的。服务器上没有它们时，运行会直接缺模块。

### 后端运行必需新增文件

- `app/core/runtime_paths.py`
- `qa/prompts/qa_generation_prompts.py`
- `qa/generation/qa_generation_flow.py`
- `qa/pipeline_runtime.py`
- `qa/validation/qa_item.py`
- `qa/grounding/source_fact_grounding.py`
- `app/routers/pipeline_common.py`
- `app/routers/pipeline_generation_routes.py`
- `app/routers/pipeline_evaluation_routes.py`
- `app/routers/pipeline_batch_routes.py`
- `app/routers/pipeline_history_routes.py`
- `app/routers/admin_v1_common.py`
- `app/routers/admin_v1_item_routes.py`
- `app/routers/admin_v1_job_routes.py`
- `app/services/storage_uploads.py`
- `app/services/storage_paths.py`
- `app/services/storage_consolidation.py`
- `app/services/milvus_runtime.py`
- `app/services/milvus_meta_utils.py`
- `app/services/milvus_collection_service.py`
- `app/services/milvus_embedding_service.py`
- `app/services/milvus_store_search.py`
- `app/services/admin_qa_common.py`
- `app/services/admin_qa_query_service.py`
- `app/services/admin_qa_write_service.py`
- `app/services/unsupervised_eval_runtime.py`
- `app/services/unsupervised_eval_common.py`
- `app/services/unsupervised_eval_aggregation.py`
- `app/services/unsupervised_eval_runners.py`
- `app/services/unsupervised_eval_suite.py`
- `app/services/eval_job_common.py`
- `app/services/eval_job_run_service.py`
- `app/services/eval_job_result_service.py`

### 前端运行新增文件

- `static/app_runtime.js`
- `static/app_config.js`
- `static/app_render.js`
- `static/app_query.js`

### 文档新增文件

- `SLIMMING_FINAL_HANDOFF.md`

## 需要覆盖上传的文件

下面这些文件不是新增，而是已有文件被修改过。服务器已有旧版本时，你要覆盖上传。

### 后端运行必需覆盖文件

- `app/core/config.py`
- `app/services/knowledge_tagging_service.py`
- `qa/text_to_qa_pipeline.py`
- `app/routers/pipeline.py`
- `app/routers/admin_v1.py`
- `app/services/storage_service.py`
- `app/services/milvus_service.py`
- `app/services/admin_qa_service.py`
- `app/services/unsupervised_evaluation_service.py`
- `app/services/eval_job_service.py`
- `app/services/admin_meta_service.py`
- `app/services/admin_job_store.py`
- `app/services/admin_qa_write_service.py`
- `app/services/artifact_lifecycle_service.py`
- `app/services/pipeline_status_store.py`
- `app/services/eval_job_run_service.py`
- `qa/qa_evaluation/unsupervised_faithfulness.py`
- `qa/qa_evaluation/unsupervised_answerability.py`
- `qa/qa_evaluation/unsupervised_coverage_recall.py`
- `qa/qa_evaluation/unsupervised_fluency_ppl.py`
- `qa/qa_evaluation/qa_quality_evaluator.py`
- `scripts/eval_jsonl_metrics.py`
- `qa/baseline_eval/summarize_llm_scores.py`
- `scripts/start_api.py`
- `api_server.py`
- `docker/Dockerfile`
- `docker/Dockerfile.dockerignore`
- `docker/docker-compose.yml`
- `docker/docker-compose.debug.yml`
- `docker/runtime-common.sh`
- `docker/start-qa-flow-services.sh`
- `docker/start-debug-shell.sh`

### 使用项目前端页面时需要覆盖的文件

如果你的服务器还会直接提供这个项目自己的前端页面，这些也要覆盖：

- `static/ui.js`
- `static/app.js`
- `static/admin.js`
- `static/index.html`
- `static/admin.html`

### 文档类覆盖文件

如果你希望服务器代码目录也保留最新版说明，可以一起覆盖：

- `LATEST_CHANGE_GUIDE.md`
- `SLIMMING_FINAL_HANDOFF.md`
- `qa/knowledge_tagging_3lvl/README.md`

### 训练脚本类覆盖文件

如果你后续还会在服务器或本地继续训练知识分类模型，这两个脚本也要覆盖：

- `qa/knowledge_tagging_3lvl/scripts/run_pytorch_webhq.ps1`
- `qa/knowledge_tagging_3lvl/scripts/run_pytorch_large.ps1`

## 服务器最小上传清单

下面这部分是最关键的。按你当前服务器的同步状态选最小集合即可。

### 场景 0：服务器代码已跟上之前的瘦身，只补这次运行时目录重构

#### 必需上传的代码与配置文件

1. `app/core/runtime_paths.py`
2. `app/core/config.py`
3. `app/services/admin_meta_service.py`
4. `app/services/admin_job_store.py`
5. `app/services/admin_qa_write_service.py`
6. `app/services/artifact_lifecycle_service.py`
7. `app/services/pipeline_status_store.py`
8. `app/services/eval_job_run_service.py`
9. `qa/qa_evaluation/unsupervised_faithfulness.py`
10. `qa/qa_evaluation/unsupervised_answerability.py`
11. `qa/qa_evaluation/unsupervised_coverage_recall.py`
12. `qa/qa_evaluation/unsupervised_fluency_ppl.py`
13. `qa/qa_evaluation/qa_quality_evaluator.py`
14. `scripts/eval_jsonl_metrics.py`
15. `scripts/start_api.py`
16. `api_server.py`
17. `docker/Dockerfile`
18. `docker/Dockerfile.dockerignore`
19. `docker/docker-compose.yml`
20. `docker/docker-compose.debug.yml`
21. `docker/runtime-common.sh`
22. `docker/start-qa-flow-services.sh`
23. `docker/start-debug-shell.sh`

#### 这次不用跟代码一起重新上传的大目录

如果服务器上已经有对应数据，这次不要把下面这些目录重新打包上传：

- `runtime_assets/models/`
- `runtime_assets/cache/`
- `runtime_assets/outputs/`
- `runtime_assets/uploads/`
- `runtime_assets/volumes/`

#### 服务器侧要做的目录迁移

把服务器上原来分散的目录迁到 `runtime_assets/` 下即可：

1. 旧 QA 评估模型目录 → `runtime_assets/models`
2. `.torch` → `runtime_assets/cache/torch`
3. `.transformers` → `runtime_assets/cache/transformers`
4. `.huggingface` → `runtime_assets/cache/huggingface`
5. `qa/outputs` → `runtime_assets/outputs`
6. `qa/uploads` → `runtime_assets/uploads`
7. `volumes` → `runtime_assets/volumes`

#### 这次是否需要重建镜像

不需要重新 `build` 镜像；只需要在服务器上替换上面的代码文件和 compose 文
件后，执行容器重建：

- `docker compose -f docker/docker-compose.yml up -d --force-recreate`

### 场景 0B：服务器代码已跟上 `runtime_assets` 重构，只补知识分类 outputs 迁移

#### 只关心在线运行

你至少要上传：

1. `app/core/runtime_paths.py`
2. `app/services/knowledge_tagging_service.py`

#### 如果你还要继续在仓库里训练知识分类模型

再额外上传：

3. `qa/knowledge_tagging_3lvl/scripts/run_pytorch_webhq.ps1`
4. `qa/knowledge_tagging_3lvl/scripts/run_pytorch_large.ps1`
5. `qa/knowledge_tagging_3lvl/README.md`

#### 服务器侧要做的目录迁移

把服务器上的：

- `qa/knowledge_tagging_3lvl/outputs`

迁到：

- `runtime_assets/knowledge_tagging_3lvl/outputs`

#### 这次不用重新上传的大目录

如果服务器上这个目录已经有完整内容，这次不要重新打包上传：

- `runtime_assets/knowledge_tagging_3lvl/outputs/`

#### 这次是否需要重建镜像

不需要重新 `build` 镜像；代码替换完成后，直接重启或重建容器即可：

- `docker compose -f docker/docker-compose.yml up -d --force-recreate`

### 场景 A：服务器已经同步过前 13 个阶段，只补这次 `eval_job_service` 瘦身

这是当前最新一轮的最小增量上传清单。

#### 只关心后端运行

你至少要上传：

1. `app/services/eval_job_common.py`
2. `app/services/eval_job_run_service.py`
3. `app/services/eval_job_result_service.py`
4. `app/services/eval_job_service.py`

### 场景 B：服务器已经同步过前 12 个阶段，还没同步 `eval_job_service` 与 `unsupervised_evaluation_service`

#### 只关心后端运行

你至少要上传：

1. `app/services/unsupervised_eval_runtime.py`
2. `app/services/unsupervised_eval_common.py`
3. `app/services/unsupervised_eval_aggregation.py`
4. `app/services/unsupervised_eval_runners.py`
5. `app/services/unsupervised_eval_suite.py`
6. `app/services/unsupervised_evaluation_service.py`
7. `app/services/eval_job_common.py`
8. `app/services/eval_job_run_service.py`
9. `app/services/eval_job_result_service.py`
10. `app/services/eval_job_service.py`

### 场景 C：服务器已经同步过前 11 个阶段，还没同步 admin service + unsupervised service + eval job service

#### 只关心后端运行

你至少要上传：

1. `app/services/admin_qa_common.py`
2. `app/services/admin_qa_query_service.py`
3. `app/services/admin_qa_write_service.py`
4. `app/services/admin_qa_service.py`
5. `app/services/unsupervised_eval_runtime.py`
6. `app/services/unsupervised_eval_common.py`
7. `app/services/unsupervised_eval_aggregation.py`
8. `app/services/unsupervised_eval_runners.py`
9. `app/services/unsupervised_eval_suite.py`
10. `app/services/unsupervised_evaluation_service.py`
11. `app/services/eval_job_common.py`
12. `app/services/eval_job_run_service.py`
13. `app/services/eval_job_result_service.py`
14. `app/services/eval_job_service.py`

### 场景 D：服务器已经同步过前 9 个阶段，还没同步 admin 路由 + Milvus + admin service + unsupervised service + eval job service

#### 只关心后端运行

你至少要上传：

1. `app/routers/admin_v1_common.py`
2. `app/routers/admin_v1_item_routes.py`
3. `app/routers/admin_v1_job_routes.py`
4. `app/routers/admin_v1.py`
5. `app/services/milvus_runtime.py`
6. `app/services/milvus_meta_utils.py`
7. `app/services/milvus_collection_service.py`
8. `app/services/milvus_embedding_service.py`
9. `app/services/milvus_store_search.py`
10. `app/services/milvus_service.py`
11. `app/services/admin_qa_common.py`
12. `app/services/admin_qa_query_service.py`
13. `app/services/admin_qa_write_service.py`
14. `app/services/admin_qa_service.py`
15. `app/services/unsupervised_eval_runtime.py`
16. `app/services/unsupervised_eval_common.py`
17. `app/services/unsupervised_eval_aggregation.py`
18. `app/services/unsupervised_eval_runners.py`
19. `app/services/unsupervised_eval_suite.py`
20. `app/services/unsupervised_evaluation_service.py`
21. `app/services/eval_job_common.py`
22. `app/services/eval_job_run_service.py`
23. `app/services/eval_job_result_service.py`
24. `app/services/eval_job_service.py`

### 场景 E：服务器已经同步过前 7 个阶段，还没同步 storage + app.js + admin + Milvus + admin service + unsupervised service + eval job service

#### 只关心后端运行

你至少要上传：

1. `app/services/storage_uploads.py`
2. `app/services/storage_paths.py`
3. `app/services/storage_consolidation.py`
4. `app/services/storage_service.py`
5. `app/routers/admin_v1_common.py`
6. `app/routers/admin_v1_item_routes.py`
7. `app/routers/admin_v1_job_routes.py`
8. `app/routers/admin_v1.py`
9. `app/services/milvus_runtime.py`
10. `app/services/milvus_meta_utils.py`
11. `app/services/milvus_collection_service.py`
12. `app/services/milvus_embedding_service.py`
13. `app/services/milvus_store_search.py`
14. `app/services/milvus_service.py`
15. `app/services/admin_qa_common.py`
16. `app/services/admin_qa_query_service.py`
17. `app/services/admin_qa_write_service.py`
18. `app/services/admin_qa_service.py`
19. `app/services/unsupervised_eval_runtime.py`
20. `app/services/unsupervised_eval_common.py`
21. `app/services/unsupervised_eval_aggregation.py`
22. `app/services/unsupervised_eval_runners.py`
23. `app/services/unsupervised_eval_suite.py`
24. `app/services/unsupervised_evaluation_service.py`
25. `app/services/eval_job_common.py`
26. `app/services/eval_job_run_service.py`
27. `app/services/eval_job_result_service.py`
28. `app/services/eval_job_service.py`

#### 后端运行 + 继续使用项目自带首页

在上面 24 个后端文件之外，再上传：

29. `static/app_runtime.js`
30. `static/app_config.js`
31. `static/app_render.js`
32. `static/app_query.js`
33. `static/app.js`
34. `static/index.html`

### 场景 F：服务器已经同步过前 6 个阶段，还没同步流水线路由拆分及其后续全部改动

#### 后端最小集合

你至少要上传：

1. `app/routers/pipeline_common.py`
2. `app/routers/pipeline_generation_routes.py`
3. `app/routers/pipeline_evaluation_routes.py`
4. `app/routers/pipeline_batch_routes.py`
5. `app/routers/pipeline_history_routes.py`
6. `app/routers/pipeline.py`
7. `app/services/storage_uploads.py`
8. `app/services/storage_paths.py`
9. `app/services/storage_consolidation.py`
10. `app/services/storage_service.py`
11. `app/routers/admin_v1_common.py`
12. `app/routers/admin_v1_item_routes.py`
13. `app/routers/admin_v1_job_routes.py`
14. `app/routers/admin_v1.py`
15. `app/services/milvus_runtime.py`
16. `app/services/milvus_meta_utils.py`
17. `app/services/milvus_collection_service.py`
18. `app/services/milvus_embedding_service.py`
19. `app/services/milvus_store_search.py`
20. `app/services/milvus_service.py`
21. `app/services/admin_qa_common.py`
22. `app/services/admin_qa_query_service.py`
23. `app/services/admin_qa_write_service.py`
24. `app/services/admin_qa_service.py`
25. `app/services/unsupervised_eval_runtime.py`
26. `app/services/unsupervised_eval_common.py`
27. `app/services/unsupervised_eval_aggregation.py`
28. `app/services/unsupervised_eval_runners.py`
29. `app/services/unsupervised_eval_suite.py`
30. `app/services/unsupervised_evaluation_service.py`
31. `app/services/eval_job_common.py`
32. `app/services/eval_job_run_service.py`
33. `app/services/eval_job_result_service.py`
34. `app/services/eval_job_service.py`

#### 如果服务器还要继续使用项目自带首页

再额外上传：

35. `static/app_runtime.js`
36. `static/app_config.js`
37. `static/app_render.js`
38. `static/app_query.js`
39. `static/app.js`
40. `static/index.html`

### 场景 G：服务器还没有同步过这整轮瘦身，要一次补齐当前全部成果

#### 后端最小完整集合

1. `qa/prompts/qa_generation_prompts.py`
2. `qa/generation/qa_generation_flow.py`
3. `qa/pipeline_runtime.py`
4. `qa/validation/qa_item.py`
5. `qa/grounding/source_fact_grounding.py`
6. `qa/text_to_qa_pipeline.py`
7. `app/routers/pipeline_common.py`
8. `app/routers/pipeline_generation_routes.py`
9. `app/routers/pipeline_evaluation_routes.py`
10. `app/routers/pipeline_batch_routes.py`
11. `app/routers/pipeline_history_routes.py`
12. `app/routers/pipeline.py`
13. `app/routers/admin_v1_common.py`
14. `app/routers/admin_v1_item_routes.py`
15. `app/routers/admin_v1_job_routes.py`
16. `app/routers/admin_v1.py`
17. `app/services/storage_uploads.py`
18. `app/services/storage_paths.py`
19. `app/services/storage_consolidation.py`
20. `app/services/storage_service.py`
21. `app/services/milvus_runtime.py`
22. `app/services/milvus_meta_utils.py`
23. `app/services/milvus_collection_service.py`
24. `app/services/milvus_embedding_service.py`
25. `app/services/milvus_store_search.py`
26. `app/services/milvus_service.py`
27. `app/services/admin_qa_common.py`
28. `app/services/admin_qa_query_service.py`
29. `app/services/admin_qa_write_service.py`
30. `app/services/admin_qa_service.py`
31. `app/services/unsupervised_eval_runtime.py`
32. `app/services/unsupervised_eval_common.py`
33. `app/services/unsupervised_eval_aggregation.py`
34. `app/services/unsupervised_eval_runners.py`
35. `app/services/unsupervised_eval_suite.py`
36. `app/services/unsupervised_evaluation_service.py`
37. `app/services/eval_job_common.py`
38. `app/services/eval_job_run_service.py`
39. `app/services/eval_job_result_service.py`
40. `app/services/eval_job_service.py`

#### 如果服务器还要继续使用项目自带首页

再额外上传：

41. `static/app_runtime.js`
42. `static/app_config.js`
43. `static/app_render.js`
44. `static/app_query.js`
45. `static/app.js`
46. `static/index.html`
47. `static/ui.js`
48. `static/admin.js`
49. `static/admin.html`

## 明确不用重新上传的内容

下面这些内容不属于这轮瘦身必须重传的范围。

### 模型与大文件

你不用重新上传：

- `runtime_assets/models/`
- `runtime_assets/cache/torch/`
- `runtime_assets/cache/transformers/`
- `runtime_assets/cache/huggingface/`
- `runtime_assets/knowledge_tagging_3lvl/outputs/`

### 数据目录

你不用因为这轮瘦身去重传：

- `milvus_data/`
- `runtime_assets/volumes/`
- `runtime_assets/outputs/`
- `runtime_assets/uploads/`

### 部署文件

这次运行时目录重构已经要求同步 compose 文件：

- `docker/Dockerfile`
- `docker/Dockerfile.dockerignore`
- `docker/docker-compose.yml`
- `docker/docker-compose.debug.yml`
- `docker/runtime-common.sh`
- `docker/start-qa-flow-services.sh`
- `docker/start-debug-shell.sh`

新的 `docker/Dockerfile` 来自 `/data2/hjk/Dockerfile`，需要和 Compose 及
启动脚本一起同步。

## 最后给你的直接建议

如果你当前服务器代码已经同步过前面的瘦身阶段，那么你现在最应该按“场景
0”执行：上传这次目录重构涉及的代码和 compose 文件，但不要重新上传
`runtime_assets/`。

也就是说，这次你以后最该记住的不是一串零散目录，而是这一条原则：

- 打包代码时排除 `runtime_assets/`

这样就能同时避开模型、缓存、输出、上传和 Milvus 持久化数据。
