# QA Flow 使用指南

本项目提供基于文档切块的一步式问答生成、评估、入库、查询和管理能力。
当前正式流程已经移除旧的“原子事实抽取、事实分类、事实转问答”三段式
入口，问答生成统一走批量完整流水线。

## 快速启动

本地开发环境使用以下命令启动服务：

```bash
pip install -r requirements.txt
python scripts/start_api.py
```

Docker 正式部署使用以下命令启动。该 Compose 直接使用本机已有的
`qa-flow-runtime:latest` 镜像，并在同一容器内同时启动 QA Flow API、
QA Flow OCR API、etcd、MinIO 和 Milvus：

```bash
docker compose -f docker/docker-compose.yml up -d
```

离线镜像部署同样不要使用 `--build`，需要先确认本机已导入
`qa-flow-runtime:latest`：

```bash
docker compose -f docker/docker-compose.yml up -d
```

调试容器不启动两个主 API，默认只启动基础服务并进入 bash，便于 attach 后
手动运行进程：

```bash
docker compose -f docker/docker-compose.debug.yml up
```

## 基础地址

默认 API 地址为：

```text
http://localhost:12000
```

QA Flow OCR API 默认地址为：

```text
http://localhost:11169
```

## 可选运行时配置

VLM 图片分析不再内置 endpoint、model 或 key 默认值。启用图片分析时，通过
API 表单参数或以下环境变量提供配置：

- `VLM_API_BASE`
- `VLM_MODEL_NAME`
- `VLM_API_KEY`
- `VLM_API_TYPE`，默认 `openai`
- `VLM_MODEL_VERSION`

本地 OCR 图片替换默认保持开启，可用 `OCR_REPLACE_IMAGES=false` 修改默认值。
`POST /process` 和 `POST /batch-upload-integrated-document-pipeline` 都支持
`replace_images` 表单参数，且请求参数优先于环境变量。

图片分类类别可用 `CLASSIFIER_CLASS_CONFIG_FILE` 指向 JSON 文件；未设置时会尝试
`${CLASSIFIER_MODEL_DIR}/classes.json`，仍不存在则使用内置 10 类。文件格式为
JSON 数组，每项必须包含 `class_id`、`model_label`、`category_key`、
`display_name`。

常用健康检查接口：

- `GET /`
- `GET /health`
- `GET /test-connection`

## 正式问答生成入口

当前正式生成入口只有批量完整流水线：

```text
POST /batch-upload-complete-pipeline-with-evaluation
```

这个接口支持上传一个或多个文件，并完成以下流程：

- 文档抽取或 OCR。
- 结构化切块。
- 问答生成。
- 问答增广。
- 有监督或无监督评价。
- 问答和文档块入库。
- 任务状态记录和结果文件管理。

常用表单参数：

- `files`：上传文件，支持多个。
- `chunk_size`：目标块大小，默认 `600`。
- `qa_per_chunk`：每个块期望生成的主问答数量，默认 `1`。
- `qa_detail_mode`：问答粒度，支持 `point` 和 `summary`。
- `prompt_language`：提示词语言，支持 `auto`、`zh`、`en`。
- `question_type_mode`：题型模式，支持 `fixed` 和 `mixed`。
- `question_types`：题型列表，例如 `简答题,判断题`。
- `augment_per_qa`：每条主问答的增广数量，默认 `0`。
- `include_evaluation`：是否执行有监督评价。
- `include_unsupervised_evaluation`：是否执行无监督评价。
- `evaluation_method`：评价方式，支持 `llm`、`local`、`faithfulness`、
  `answerability`、`unsupervised_f1`。
- `filter_by_threshold`：是否按分数阈值过滤。
- `score_threshold`：过滤阈值，默认 `0.7`。
- `enable_vector_storage`：是否写入问答向量库。
- `enable_chunk_storage`：是否写入文档块树。
- `chunking_split_type`：切分方式，支持 `markdown`、`text`、`token`、
  `recursive`、`code`、`custom`。
- `chunking_prefix_max_depth`：标题路径最多向上保留的层数。
- `ocr_enabled`：是否启用 OCR。

示例请求：

```bash
curl -X POST "http://localhost:12000/batch-upload-complete-pipeline-with-evaluation" \
  -F "files=@qa/chunking/testdata/input/01_关于加强考勤与请休假管理的通知.md" \
  -F "chunk_size=600" \
  -F "qa_per_chunk=1" \
  -F "include_unsupervised_evaluation=true" \
  -F "evaluation_method=unsupervised_f1" \
  -F "enable_vector_storage=true" \
  -F "enable_chunk_storage=true"
```

## 任务状态和产物

后台任务返回 `task_id` 后，使用以下接口查看状态、下载产物或删除历史任务：

- `GET /task-status/{task_id}`
- `POST /cancel-task/{task_id}`
- `GET /pipeline/jobs`
- `DELETE /pipeline/jobs/{task_id}`
- `GET /task-file-csv/{task_id}`
- `GET /task-csv/{task_id}`
- `GET /download/{file_path}`
- `GET /list-files`

如果任务已经成功入库且本地产物被清理，前端和调用方应优先到管理接口或
Milvus 查询结果，而不是继续依赖已过期的 JSON 或 CSV 文件。

## 问答评价入口

独立评价接口仍然保留：

- `POST /batch-upload-evaluate-qa`
- `POST /upload-evaluate-qa`
- `POST /evaluate-qa-local`

当前评价逻辑会优先使用问答生成阶段写入的 `qa_generation_unit_text` 作为
来源文本；如果结果文件中没有该字段，才会退回使用旧字段。

## 文档块查询

文档块树和块级溯源使用以下接口：

- `GET /doc-chunks/by-task/{task_id}`
- `GET /doc-chunks/tree`
- `GET /doc-chunks/{chunk_id}`
- `GET /doc-chunks/{chunk_id}/qa`

这些接口用于查看切块结果、树状结构、单块详情，以及某个块关联的问答。

## 管理接口

管理端接口统一位于 `/admin/v1` 前缀下：

- `POST /admin/v1/ingest-consolidated`
- `GET /admin/v1/qa-items`
- `GET /admin/v1/qa-items/{qa_id}`
- `PATCH /admin/v1/qa-items/{qa_id}`
- `PATCH /admin/v1/qa-items/{qa_id}/admin-meta`
- `POST /admin/v1/qa-items/batch-update`
- `POST /admin/v1/qa-items/batch-admin-update`
- `POST /admin/v1/qa-items/batch-delete`
- `POST /admin/v1/qa-search`
- `POST /admin/v1/evaluation-jobs`
- `POST /admin/v1/unsupervised-evaluation-jobs`
- `GET /admin/v1/jobs/{job_id}`
- `POST /admin/v1/jobs/{job_id}/cancel`
- `POST /admin/v1/exports`

## 配置和调试接口

运行时配置接口：

- `GET /llm-configs`
- `POST /llm-configs`
- `POST /llm-configs/{name}/activate`
- `DELETE /llm-configs/{name}`
- `GET /ocr-configs`
- `POST /ocr-configs`
- `POST /ocr-configs/{name}/activate`
- `DELETE /ocr-configs/{name}`
- `POST /ocr-configs/{name}/test`

LLM 配置项包含 `name`、`api_key`、`base_url`、`model`，以及可选的
`api_type` 和 `model_version`。`api_type` 支持 `openai`、`lmp_cloud`；
其他高级请求参数继续由统一 LLM client 的默认值或环境变量控制。

调试和辅助接口：

- `POST /llm-debug/chat`
- `POST /knowledge-tagging/predict`
- `GET /milvus-status`
- `POST /init-milvus`

## 当前有效提示词文件

正式流程只使用以下提示词文件：

- `qa/prompts/qa_generation_prompts.py`
- `qa/prompts/llm_quality_evaluation_prompts.py`
- `qa/prompts/qa_augmentation_prompts.py`

旧的事实抽取、事实分类和事实转问答提示词已经删除，不再作为正式流程的一
部分。

## 旧流程说明

旧的三段式生成入口不再维护。前端或调用脚本需要统一改为调用正式批量完整
流水线入口。
