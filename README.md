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

通过校园 SSLVPN/WebVPN 的带前缀地址访问页面时，直接打开代理后的
`.../ui/` 或 `.../ui/index.html` 即可。前端资源使用相对 `/ui/` 路径，页面会根据
当前地址自动推断代理 API 基址，因此不需要把 API 地址改成代理根域；直连
`http://服务器地址:12000/ui/` 仍按原方式工作。`/ui`、`/ui/`、HTML、脚本、样式
和图片响应统一禁止浏览器及 VPN 网关缓存，并通过统一构建号刷新页面导航，避免
旧 HTML 与新版脚本混合加载。
从服务器根地址进入时也使用相对重定向，WebVPN 前缀不会被丢弃。

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
- `qa_total_limit`：主问答总数上限，前端默认 `20`。
- `qa_total_limit_scope`：题数上限范围，支持 `per_file` 和 `batch`。
- `qa_detail_mode`：问答粒度，支持 `auto`、`point` 和 `summary`。
  系统先把同一逻辑 section 的 fragment、正文和关联图片整理成一份材料，再由
  LLM 建立 PointScenario 与 SummaryScenario 候选池。`auto` 目标为 35% 总结题，
  总结场景不足时额度自动让给单点题。每个场景只生成一道题，并统一经过一次
  LLM `keep/rewrite/drop` 编辑；不会靠硬编码语言规则判断自然度。长文档由后端
  按内部字符预算分批规划，最终题型配额和问题去重仍在文档级统一完成。
- `qa_per_chunk`：兼容旧调用方；未传 `qa_total_limit` 时才用于估算题量。
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
- `final_evidence_k`：最终保留的检索证据窗口数，默认 `5`；设为 `0`
  时答案只使用当前 generation unit 的主材料。它表示上限而不是必须填满；BGE
  相关性准入后可能保留 0 组补充证据。
- `evidence_token_budget`：检索证据窗口的近似 token 总预算，默认 `4000`。
- `chunking_split_type`：切分方式，支持 `markdown`、`text`、`token`、
  `recursive`、`code`、`custom`。
- `chunking_prefix_max_depth`：标题路径最多向上保留的层数。
- `ocr_enabled`：是否启用 OCR。

示例请求：

```bash
curl -X POST "http://localhost:12000/batch-upload-complete-pipeline-with-evaluation" \
  -F "files=@qa/chunking/testdata/input/01_关于加强考勤与请休假管理的通知.md" \
  -F "chunk_size=600" \
  -F "qa_total_limit=20" \
  -F "qa_total_limit_scope=per_file" \
  -F "qa_detail_mode=auto" \
  -F "include_unsupervised_evaluation=true" \
  -F "evaluation_method=unsupervised_f1" \
  -F "final_evidence_k=5" \
  -F "evidence_token_budget=4000" \
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

当前评价逻辑会优先使用答案实际引用的 `qa_evaluation_evidence_text` 作为
来源文本；它只包含答案引用的正文/图片证据块。旧结果没有该字段时，才会退回
使用完整 `qa_generation_unit_text` 和更早的来源字段。

## 文档块查询

切块元数据把“章节节点”“内容 chunk”“物理 fragment”分开：内部章节可以有
自己的正文，空父章节不会制造重复正文，长章节的 `Part N/M` 只记录为 fragment，
不会成为假章节。检索固定使用 BM25 + BGE-M3 dense、RRF、本地
`bge-reranker-v2-m3` 原子块重排、相关性准入、结构窗口补全和窗口二次重排；
窗口还要通过第二次相关性准入，模型缺失会明确报错，不会退回旧手工排序。
准入同时比较问题与真实主材料的 BGE 得分，补充块明显弱于主材料时直接丢弃；
`scripts/calibrate_bge_relevance.py` 可复现实测阈值。

默认模型目录为 `${APP_MODELS_DIR}/bge-reranker-v2-m3`；Docker 正式与调试
入口都可通过 `QA_RERANKER_MODEL_PATH`、`QA_RERANKER_DEVICE`、
`QA_RERANKER_BATCH_SIZE`、`QA_RERANKER_MAX_LENGTH` 覆盖。

文档块只使用 `doc_content_chunks_v2`（schema v2）集合；旧
`doc_tree_chunks` 集合及其兼容路径已经移除。块级溯源使用以下接口：

- `GET /doc-chunks/by-task/{task_id}`
- `GET /doc-chunks/by-doc/assets`（按 `doc_id`，或按 `task_id` 加可选文件名，
  一次返回完整纯文本、全部 QA 和可选 chunk 列表）
- `GET /doc-chunks/tree`
- `GET /doc-chunks/{chunk_id}`
- `GET /doc-chunks/{chunk_id}/qa`
- `POST /doc-chunks/rebuild`（提交原始正文、`task_id`、文件名和切块参数，
  服务重新切块并安全替换该任务下对应文件的 v2 索引）

这些接口用于查看切块结果、树状结构、单块详情，以及某个块关联的问答。
`/doc-chunks/by-doc/assets` 默认返回全文和活跃 QA；可用
`include_full_text`、`include_qas`、`include_chunks` 和 `qa_only_active`
控制返回内容。QA 详情会保留 Point/Summary 场景、场景意图、材料 ID 以及
总结题的完整多主来源字段。按 chunk 查询 QA 时会按完整主来源集合匹配，
因此多材料总结题会在每个实际主来源块下显示，而不是只出现在第一条来源块。

重建请求示例：

```bash
curl -X POST "http://localhost:12000/doc-chunks/rebuild" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "历史任务ID",
    "original_filename": "manual.md",
    "text": "# 使用说明\n\n## 材料\n\n提交身份证明。",
    "chunk_size": 600,
    "split_type": "markdown"
  }'
```

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
