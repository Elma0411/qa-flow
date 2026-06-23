# Latest Change Guide

更新时间：2026-06-23（Asia/Shanghai）

## Objective

Configure the three approved hardcoded surfaces in `qa-flow`: VLM
connection defaults, OCR image replacement, and image classifier classes.

## What Changed

- VLM endpoint, model, and key no longer have code-level business defaults.
  They resolve from API form parameters first and `VLM_API_BASE`,
  `VLM_MODEL_NAME`, `VLM_API_KEY`, `VLM_API_TYPE`, `VLM_MODEL_VERSION` second.
- `VLM_API_TYPE` still defaults to `openai`. Missing endpoint/model/key now
  produces a clear configuration error when image analysis needs VLM.
- Local OCR image replacement is controlled by `OCR_REPLACE_IMAGES`, defaulting
  to `true`.
- `POST /process` and `POST /batch-upload-integrated-document-pipeline` accept
  optional `replace_images`; request values override the environment default.
- Integrated task status and `ocr_summary` record the resolved
  `replace_images` value.
- The classifier class catalog now loads from `CLASSIFIER_CLASS_CONFIG_FILE`,
  then `${CLASSIFIER_MODEL_DIR}/classes.json`, then the built-in 10-class
  fallback.
- Docker Compose and `docker/runtime-common.sh` now pass through the selected
  VLM, OCR, and classifier class config environment variables.

## Expected Behavior

- Pure OCR calls can run without VLM env when image analysis is disabled.
- Image analysis with `use_api=true` requires explicit VLM config through API
  parameters or environment variables.
- Unset `OCR_REPLACE_IMAGES` preserves the previous `true` behavior.
- A present but malformed classifier `classes.json` fails startup; a missing
  file falls back to the built-in 10 classes.

## Validation

```bash
python -m compileall app qa scripts
python -m unittest discover -s tests
docker compose -f docker/docker-compose.yml config
docker compose -f docker/docker-compose.debug.yml config
```

Useful runtime checks:

- Call `/process` with `replace_images=true` and `replace_images=false`, then
  confirm OCR processor cache keys differ by the value.
- Call `/batch-upload-integrated-document-pipeline` with
  `replace_images=false` and confirm task status records `replace_images=false`.
- Set `CLASSIFIER_CLASS_CONFIG_FILE` to a valid JSON catalog and confirm
  `GET /classes` returns that catalog.
- Set VLM env values and run image analysis; the VLM client signature should
  show the configured `base_url` and `model_name`.
