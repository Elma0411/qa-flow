# Docker Compose Parameters

更新时间：2026-07-07（Asia/Shanghai）

本文档说明 `docker/docker-compose.yml` 和
`docker/docker-compose.debug.yml` 中与运行、接口端口、OCR、图片理解和并发
相关的配置项。

## 启动方式

正式运行：

```bash
cd /data2/hjk/qa-flow
docker compose -f docker/docker-compose.yml up -d
```

调试容器：

```bash
cd /data2/hjk/qa-flow
docker compose -f docker/docker-compose.debug.yml up -d
```

可以在命令前通过环境变量覆盖默认配置：

```bash
DOC_MAX_CONCURRENCY=2 \
OCR_MAX_CONCURRENCY=1 \
IMAGE_ANALYSIS_MAX_CONCURRENCY=4 \
IMAGE_FIT_MAX_CONCURRENCY=4 \
VLM_API_MAX_CONCURRENT_REQUESTS=4 \
docker compose -f docker/docker-compose.yml up -d
```

## 端口参数

| 变量 | 默认值 | 容器端口 | 说明 |
| --- | --- | --- | --- |
| `QA_FLOW_API_HOST_PORT` | `12000` | `12000` | QA Flow 主 API 端口。 |
| `OCR_API_HOST_PORT` | `11169` | `11169` | OCR-compatible API 端口。 |
| `CLASSIFIER_HOST_PORT` | `10488` | `10488` | 图片分类器 API 端口。 |
| `MILVUS_HOST_PORT` | `12530` | `19530` | Milvus 端口。 |
| `MILVUS_METRICS_HOST_PORT` | `12091` | `9091` | Milvus metrics/health 端口。 |
| `ETCD_HOST_PORT` | `12379` | `2379` | etcd 端口。 |
| `MINIO_API_HOST_PORT` | `12900` | `9000` | MinIO API 端口。 |
| `MINIO_CONSOLE_HOST_PORT` | `12901` | `9001` | MinIO Console 端口。 |

## 路径参数

容器内默认路径：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_RUNTIME_ROOT` | `/app/runtime_assets` | 运行资产根目录。 |
| `APP_MODELS_DIR` | `/app/runtime_assets/models` | 模型目录。 |
| `APP_OUTPUTS_DIR` | `/app/runtime_assets/outputs` | 输出目录。 |
| `APP_UPLOADS_DIR` | `/app/runtime_assets/uploads` | 上传目录。 |
| `MODEL_BASE_DIR` | `/app/runtime_assets/models/ocr` | OCR 模型目录。 |
| `CLASSIFIER_MODEL_DIR` | `/app/runtime_assets/models/image_classifier` | 图片分类器模型目录。 |

compose 会挂载：

- `../:/app`
- `../runtime_assets:/app/runtime_assets`
- `../runtime_assets/volumes/etcd:/data/etcd`
- `../runtime_assets/volumes/minio:/data/minio`
- `../runtime_assets/volumes/milvus:/data/milvus`

## OCR 与图片配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OCR_USE_GPU` | `true` | OCR 是否使用 GPU。 |
| `OCR_REPLACE_IMAGES` | `true` | 默认是否用原文档高质量裁图替换 OCR 导出图片。 |
| `SOFFICE_BINARY` | `/usr/bin/soffice` | LibreOffice 转 PDF 命令路径。 |
| `CLASSIFIER_CLASS_CONFIG_FILE` | 空 | 自定义图片分类类别配置文件。 |
| `CLASSIFIER_API_BASE` | `http://127.0.0.1:10488` | 图片分类服务地址。 |
| `START_CLASSIFIER_SERVICE` | `true` | 容器启动时是否启动图片分类服务。 |

DOC 和 DOCX 在 API 路径中固定先转 PDF 再 OCR；不需要配置其他处理策略。

## VLM 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VLM_API_BASE` | 空 | 图片理解 VLM API Base。 |
| `VLM_MODEL_NAME` | 空 | 图片理解 VLM 模型名。 |
| `VLM_API_KEY` | 空 | 图片理解 VLM API Key。 |
| `VLM_API_TYPE` | `openai` | VLM API 类型，支持 `openai`、`lmp_cloud`。 |
| `VLM_MODEL_VERSION` | 空 | VLM 模型版本，可选。 |
| `VLM_API_MAX_CONCURRENT_REQUESTS` | `1` | 单个共享 VLM client 的最大并发请求数。 |

`VLM_API_MAX_CONCURRENT_REQUESTS` 是 client 内部请求闸门。如果
`IMAGE_ANALYSIS_MAX_CONCURRENCY` 设置为 `4`，但该值仍为 `1`，同一个 VLM
配置上的请求仍可能被串行化。

## 集成文档预处理并发

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOC_MAX_CONCURRENCY` | `1` | 一体流程中文档预处理的最大文件并发。 |
| `OCR_MAX_CONCURRENCY` | `1` | OCR 提取最大并发。 |
| `IMAGE_ANALYSIS_MAX_CONCURRENCY` | `1` | 图片理解 API 模式最大并发。 |
| `IMAGE_FIT_MAX_CONCURRENCY` | `1` | 图片契合度判断最大并发。 |

推荐调优顺序：

1. 先保持 `OCR_MAX_CONCURRENCY=1`，避免单 GPU 上多个 OCR 任务抢显存。
2. 多文件批处理时先提高 `DOC_MAX_CONCURRENCY`，例如 `2`。
3. VLM API 能承受更多请求时，同时提高
   `IMAGE_ANALYSIS_MAX_CONCURRENCY` 和 `VLM_API_MAX_CONCURRENT_REQUESTS`。
4. `IMAGE_FIT_MAX_CONCURRENCY` 主要消耗 LLM/API 请求或 CPU，按实际模型延迟调整。

## 前端与请求级覆盖

compose/env 是推荐的部署默认值。前端“一体流程”参数区提供以下可选字段：

- `doc_max_concurrency`
- `ocr_max_concurrency`
- `image_analysis_max_concurrency`
- `image_fit_max_concurrency`

这些字段为空时不随请求提交，接口读取 compose/env 默认值。填写后只影响本次
`POST /batch-upload-integrated-document-pipeline` 请求，不会修改容器环境变量。

## 运行环境参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NVIDIA_VISIBLE_DEVICES` | `all` | 容器可见 GPU。 |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | NVIDIA runtime capabilities。 |
| `TOKENIZERS_PARALLELISM` | `false` | 禁用 tokenizer 并行警告和额外线程。 |
| `PYTHONUNBUFFERED` | `1` | Python 日志不缓冲。 |
| `PYTHONDONTWRITEBYTECODE` | `1` | 不写 `.pyc`。 |
| `PYTHONIOENCODING` | `utf-8` | Python IO 编码。 |
| `QA_FLOW_API_RELOAD` | `true` | QA Flow 主 API 是否启用 Uvicorn 热重载。代码通过 `../:/app` 挂载时，修改 `app/`、`qa/`、`scripts/` 下 Python 文件会自动重启 API 进程。 |
| `UVICORN_LOG_LEVEL` | `info` | 正式 compose 下 Uvicorn 日志级别。 |
| `UVICORN_ACCESS_LOG` | `true` | 正式 compose 下是否开启 access log。 |
