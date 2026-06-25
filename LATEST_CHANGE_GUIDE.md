# Latest Change Guide

更新时间：2026-06-25（Asia/Shanghai）

## Objective

Allow the active frontend LLM profile to select the unified LLM client protocol
without exposing advanced client-tuning parameters.

## What Changed

- `llm_configs.json` profiles now support two additional fields:
  `api_type` and `model_version`.
- The LLM config API and frontend LLM settings panel can save, display, and
  activate those fields.
- Activating a profile writes `api_key`, `base_url`, `model`, `api_type`, and
  `model_version` into the backend runtime `CONFIG`.
- Hao-side chunking and QA generation continue to use the unified
  `app.services.llm` client path, now with the active profile's protocol
  fields included.
- Advanced options such as timeout, stream mode, request concurrency, interval,
  `top_p`, and penalties remain controlled by existing defaults/environment
  variables, not by `llm_configs`.

## Configuration

For OpenAI-compatible providers, use:

```json
{
  "name": "default",
  "api_key": "...",
  "base_url": "https://open.bigmodel.cn/api/paas/v4/",
  "model": "glm-4-flash",
  "api_type": "openai",
  "model_version": ""
}
```

For LMP Cloud, set `api_type` to `lmp_cloud`. `model_version` is optional and
is only sent when non-empty.

## Expected Behavior

- Existing profiles without `api_type` are auto-filled as `openai` when read.
- New frontend profile saves include `api_type` and `model_version`.
- Activating an `lmp_cloud` profile makes hao-side markdown heading correction
  and QA generation use the LMP Cloud client implementation.

## Validation

```bash
cd qa-flow
python -m py_compile app/core/config.py app/core/clients.py app/services/llm_config/store.py app/routers/llm_config.py app/services/pipeline_execution/service.py qa/chunking/easy_dataset.py
```
