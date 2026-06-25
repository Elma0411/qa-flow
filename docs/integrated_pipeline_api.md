# Integrated Document Pipeline API

更新时间：2026-06-25（Asia/Shanghai）

本文档说明当前文档解析、图片理解、切块和问答一体流程的调用接口。

## Base URL

- QA Flow API: `http://localhost:12000`
- OCR-compatible API: `http://localhost:11169`

具体端口可通过 docker-compose 环境变量调整，见
`docs/docker_compose_parameters.md`。

## 一体流程提交

`POST /batch-upload-integrated-document-pipeline`

`multipart/form-data` 请求。该接口接收一个或多个文件，先执行文档解析、
OCR、图片理解、图片回填和预切块，再进入后续问答流水线。

### 文件字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `files` | file[] | 是 | 上传文件。支持 PDF、图片、OFD、DOCX、DOC、文本和 Markdown 等现有支持格式。 |

### 文档解析字段

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `ocr_enabled` | bool | `true` | 是否启用需要 OCR 的文件解析。 |
| `ocr_fail_fast` | bool | `false` | 任一文件解析失败时是否立即让整体任务失败。 |
| `remove_watermark` | bool | `false` | OCR 前是否执行水印预处理。 |
| `watermark_dpi` | int | `200` | 水印预处理渲染 DPI。 |
| `replace_images` | bool | `OCR_REPLACE_IMAGES` | 是否使用原文档高质量裁图替换 OCR 导出图片。 |
| `docx_strategy` | string | `pdf` | 兼容字段。DOC 和 DOCX 固定转 PDF 后 OCR。 |

### 图片理解字段

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `enable_image_analysis` | bool | `true` | 是否执行图片理解。 |
| `image_analysis_use_api` | bool | `true` | 是否使用 VLM API。 |
| `enable_image_classification` | bool | `false` | 图片理解前是否先分类选择 prompt。 |
| `classification_confidence_threshold` | float | `0.0` | 图片分类置信阈值，范围 `0~1`。 |
| `image_context_summary_mode` | string | `lightweight` | 图片上下文摘要模式：`lightweight` 或 `llm`。 |
| `image_fit_check_enabled` | bool | `true` | 是否判断图片解析结果与 chunk 上下文的契合度。 |
| `image_fit_min_score` | float | `0.65` | 图片回填最小契合度分数，范围 `0~1`。 |

### VLM 覆盖字段

这些字段只影响本次一体流程图片理解请求；为空时读取后端当前配置或
compose 环境变量。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `vlm_api_base` | string | VLM API Base。 |
| `vlm_model_name` | string | VLM 模型名。 |
| `vlm_api_key` | string | VLM API Key。 |
| `vlm_api_type` | string | `openai` 或 `lmp_cloud`。 |
| `vlm_model_version` | string | VLM 模型版本，可选。 |

### 并发字段

这些字段是单次请求覆盖项；不传则读取 compose/env 默认值。

| 字段 | 类型 | 默认来源 | 说明 |
| --- | --- | --- | --- |
| `doc_max_concurrency` | int | `DOC_MAX_CONCURRENCY` | 文档预处理最大文件并发。 |
| `ocr_max_concurrency` | int | `OCR_MAX_CONCURRENCY` | OCR 提取最大并发。 |
| `image_analysis_max_concurrency` | int | `IMAGE_ANALYSIS_MAX_CONCURRENCY` | 图片理解最大并发。 |
| `image_fit_max_concurrency` | int | `IMAGE_FIT_MAX_CONCURRENCY` | 图片契合度判断最大并发。 |

建议先在 compose 中设置稳定默认值；前端或调用方只在单次任务需要压测或临时
放大吞吐时传这些字段。

### 常用 QA 字段

一体流程继续复用完整问答流水线字段，例如：

- `qa_per_chunk`
- `augment_per_qa`
- `chunk_size`
- `chunking_split_type`
- `include_evaluation`
- `evaluation_method`
- `enable_vector_storage`
- `enable_chunk_storage`
- `max_concurrency`
- `chunk_max_concurrency`
- `augment_max_concurrency`
- `eval_max_concurrency`
- `sync_mode`
- `save_mode`

这些字段的后续问答语义没有在本次并发改造中改变。

### 请求示例

```bash
curl -X POST "http://localhost:12000/batch-upload-integrated-document-pipeline" \
  -F "files=@demo.pdf" \
  -F "ocr_enabled=true" \
  -F "enable_image_analysis=true" \
  -F "doc_max_concurrency=2" \
  -F "ocr_max_concurrency=1" \
  -F "image_analysis_max_concurrency=4" \
  -F "image_fit_max_concurrency=4" \
  -F "qa_per_chunk=1" \
  -F "sync_mode=false"
```

### 异步响应

`sync_mode=false` 时立即返回任务信息：

```json
{
  "status": "processing",
  "batch_mode": true,
  "integrated_pipeline": true,
  "task_id": "integrated_document_task_1782380000",
  "doc_max_concurrency": 2,
  "ocr_max_concurrency": 1,
  "image_analysis_max_concurrency": 4,
  "image_fit_max_concurrency": 4
}
```

## 任务状态查询

`GET /task-status/{task_id}`

返回完整流水线任务状态。文档预处理进度位于：

```json
{
  "file_progress": {
    "demo.pdf": {
      "status": "processing",
      "stages": {
        "doc_input": {"state": "completed"},
        "doc_ocr": {"state": "completed"},
        "doc_pre_chunking": {"state": "completed"},
        "doc_image_analysis": {"state": "completed"},
        "doc_placement": {"state": "completed"},
        "doc_handoff": {"state": "completed"}
      }
    }
  }
}
```

`doc_handoff` 只表示文档预处理已经把 `file_contents` 和
`pre_split_chunks` 交给问答流水线，不表示整个任务已完成。

## 独立文档解析任务

### 提交

`POST /document-processing/jobs`

该接口只执行文档解析、OCR、图片理解和文本整合，不进入问答流水线。

核心字段：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `file` | file | 必填 | 上传一个文档。 |
| `output_format` | string | `text` | `text`、`markdown` 或 `ocr_markdown`。 |
| `docx_strategy` | string | `pdf` | 兼容字段。DOC 和 DOCX 固定转 PDF 后 OCR。 |
| `enable_image_analysis` | bool | `true` | 是否执行图片理解。 |
| `enable_classification` | bool | `false` | 是否先分类再选择 prompt。 |
| `replace_images` | bool | `OCR_REPLACE_IMAGES` | 是否替换为高质量裁图。 |

提交成功返回 `document_job_*` 格式的 `job_id`。

### 查询

`GET /document-processing/jobs/{job_id}`

### 取消

`POST /document-processing/jobs/{job_id}/cancel`

### 下载产物

`GET /document-processing/jobs/{job_id}/download?file_key=text`

常见 `file_key`：

- `text`
- `markdown`
- `ocr_markdown`
- `summary`
- `image_analysis_summary`
