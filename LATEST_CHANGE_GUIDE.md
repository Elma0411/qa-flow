# Latest Change Guide

更新时间：2026-07-06（Asia/Shanghai）

## Objective

修复一体流程和单独文档解析中 VLM 覆盖参数“留空未使用当前 OCR 配置默认值”的问题。现在如果前端不填写 `vlm_api_base`、`vlm_model_name`、`vlm_api_key`、`vlm_api_type`、`vlm_model_version`，后端会先读取当前激活 OCR 配置的 `request.extra_form_fields` 中同名字段。

## What Changed

- `app/services/ocr/store.py`
  - 新增 `get_active_vlm_defaults()`，从当前激活 OCR profile 的 `request.extra_form_fields` 读取 VLM 默认参数。
  - 只返回非空字段，不输出或记录密钥。
- `app/services/ocr/__init__.py`、`app/services/ocr/config.py`
  - 导出 `get_active_vlm_defaults()`。
- `app/routers/pipeline_integrated_routes.py`
  - 一体流程图片理解 VLM 参数改为：前端表单值优先；表单留空时使用当前激活 OCR 配置里的 VLM 默认。
- `app/routers/ocr_compat_routes.py`
  - 单独文档解析异步任务 `/document-processing/jobs` 和同步兼容接口 `/process` 使用同样的默认解析逻辑。
- `static/index.html`
  - VLM 覆盖参数的 placeholder 改为说明“留空使用当前激活 OCR 配置里的 VLM 参数”。

## Expected Behavior

- `runtime_assets/outputs/ocr_configs.json` 中当前 active profile 的 `request.extra_form_fields.vlm_*` 会作为默认 VLM 参数使用。
- 前端手动填写的 VLM 参数仍然优先，不会被 OCR 配置覆盖。
- 标准 OCR 流程原本会直接把 `extra_form_fields` 转发给 OCR 服务；本次补齐的是一体流程和单独文档解析的本地 VLM 链路。
- 任务状态中仍不展示 `vlm_api_key`。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m py_compile \
  app/services/ocr/store.py \
  app/services/ocr/__init__.py \
  app/services/ocr/config.py \
  app/routers/pipeline_integrated_routes.py \
  app/routers/ocr_compat_routes.py

node --check static/app.js
git diff --check
curl http://localhost:12000/test-connection
```

轻量验证：

```bash
python - <<'PY'
from app.services.ocr import get_active_vlm_defaults
defaults = get_active_vlm_defaults()
assert defaults.get("vlm_api_base")
assert defaults.get("vlm_model_name")
assert defaults.get("vlm_api_key")
print({k: ("***" if k == "vlm_api_key" else v) for k, v in defaults.items()})
PY
```
