# Latest Change Guide

更新时间：2026-07-10（Asia/Shanghai）

## Objective

避免 OCR 生成的印章图片占位 `<div><img ...></div>` 进入后续图片理解、切块和 QA 生成流程。

## What Changed

- `app/services/integrated_pipeline/markers.py`
  - 新增 `remove_seal_image_divs()`，只移除图片路径包含 `img_in_seal_box_` 的 OCR 印章图片 div。
  - 普通图片 div、图片 marker、VLM 图片理解逻辑保持不变。
- `app/services/integrated_pipeline/service.py`
  - 在 OCR 完成后、图片 marker 替换和切块之前清理印章图片 div。
  - `doc_marker` 进度和 `ocr_raw_entry.integrated_pipeline` 增加 `removed_seal_image_divs`，方便确认本次清理数量。
- `INTEGRATION_CONTRACT.md`
  - 记录集成流程中 seal-cleaned markdown 的边界行为。

## Expected Behavior

- OCR 原始 markdown 中的印章图片占位不会进入 `marked_markdown`、`pre_split_chunks`、图片理解 prompt 或最终 QA evidence。
- 非印章图片仍按原流程转换为 `[[IMAGE_REF:...]]` marker，并按原规则做图片理解和位置回填。
- 如需取消本改动，删除 `service.py` 中对 `remove_seal_image_divs()` 的调用及 `markers.py` 中对应 helper 即可。

## Validation

```bash
cd /data2/hjk/qa-flow

python -m compileall app qa
docker exec qa-flow-runtime python -m compileall app qa
curl http://localhost:12000/test-connection
```
