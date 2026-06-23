# AI Programming Guide

This repository is the `qa-flow` engineering tree.

## Document Roles

- `AI_PROGRAMMING_GUIDE.md` is the high-level common development protocol.
- `INTEGRATION_CONTRACT.md` is the field-level handoff contract for the
  document-processing and QA-generation subsystems.
- `CONTEXT.md` defines shared project language.
- `AGENTS.md` contains standing repository rules for future agents.

## Public Import Rules

- Import package capabilities from package facades (`__init__.py`) unless a module is explicitly private to one package.
- Keep `app/routers/*` declarative: parse requests, validate parameters, call services, and return responses.
- Stateful capabilities must live behind a class or manager. Do not add new module-level mutable state as the source of truth.
- Promote any cross-package implementation dependency through the owning package facade before treating it as a stable API.

## Main Data Flow

The integrated pipeline has one canonical path:

`OCRResult -> marked markdown -> ChunkContext -> ImageAnchorContext -> accepted image descriptions -> final chunks -> QA`

Important details:

- OCR keeps image `<div>` tags long enough to map each image to a stable `[[IMAGE_REF:image_id]]` marker.
- Chunking happens before image analysis in the integrated flow.
- Image analysis receives chunk-level context, not only the original immediate OCR before/after text.
- Image descriptions are inserted only when `ImagePlacementJudge` accepts them or when fit checking is disabled.
- Final chunks are passed to the QA pipeline through `pre_split_chunks` and `pre_split_chunk_meta`.
- Field-level handoff details are governed by `INTEGRATION_CONTRACT.md`.

## Module Ownership

- `app/services/document_processing/`: document extraction, OCR models, input adapters, watermark removal, image replacement, and OCR-compatible text integration.
- `app/services/image_understanding/`: production image analyzer, prompts, image classification client, and image model contracts.
- `app/services/llm/`: shared LLM/VLM client config, client pool, and compatibility `create_chat_completion_text` contract.
- `app/services/integrated_pipeline/`: shared OCR-image-QA handoff orchestration and marker/summary/placement logic.
- `qa/`: chunking, generation, grounding, validation, evaluation, and QA pipeline facade.

## Parallel Development Rules

- Document-processing changes should stay inside `document_processing`, `image_understanding`, and OCR-compatible API behavior unless the handoff contract changes.
- QA changes should stay inside `qa`, evaluation, storage, Milvus/search/admin, and batch execution unless the handoff contract changes.
- Changes to `integrated_pipeline`, `file_contents`, `pre_split_chunks`, `pre_split_chunk_meta`, `job_context`, endpoint parameters, runtime model paths, or deployment dependencies are shared boundary changes.
- Boundary changes must update `INTEGRATION_CONTRACT.md` and add or update focused tests in the same change.
- Do not rely on a markdown-only update for a changed runtime contract; code and tests must enforce the new behavior.

## Compatibility Contracts

- Keep `POST /batch-upload-complete-pipeline-with-evaluation` behavior-compatible with the QA pipeline.
- Keep the OCR-compatible API at standalone `POST /process`, with `output_format=text|markdown|ocr_markdown`.
- The new integrated API is `POST /batch-upload-integrated-document-pipeline`.
- Do not migrate historical `simulate_image_processor*.py` variants. Only the production implementation is carried as `app/services/image_understanding/analyzer.py`.
- Any input/output change to a compatibility contract must be reflected in `INTEGRATION_CONTRACT.md`.

## Runtime Rules

- Heavy OCR dependencies are imported lazily. Do not import PaddleOCR at app startup.
- OCR can run in-process through `OCRWorkerManager` or as a standalone service via `scripts/start_ocr_api.py`.
- `LLMClientPool` owns reusable VLM/text LLM clients and must be closed during app shutdown.
- OCR model instances are controlled by config/device keys. Do not add v1 multi-OCR model instance parallelism without a GPU memory and Paddle thread-safety review.
- OCR device selection is controlled by `OCR_USE_GPU`; Docker deployment defaults to `true` because the dependency image follows the CUDA/Paddle GPU stack.
- OCR model directories should be rooted at `MODEL_BASE_DIR` or `APP_MODELS_DIR/ocr`. Do not hard-code local absolute model paths.
- Image classifier weights live under `runtime_assets/models/image_classifier/` and are located through `CLASSIFIER_MODEL_DIR`.
- Default local QA models live directly under `runtime_assets/models/`, keyed by the names in `app/core/runtime_paths.py`.
- Knowledge-tagging model artifacts live under `runtime_assets/knowledge_tagging_3lvl/outputs/`.

## Deployment Rules

- Docker deployment assets live under `docker/`: `Dockerfile` is the pure dependency image, `docker-compose.yml` is the formal QA Flow runtime, and `docker-compose.debug.yml` is the attachable bash/debug runtime.
- Any dependency, environment variable, port, mount, startup command, or model-loading change must be checked against both Compose files and the shared startup scripts under `docker/`.
- Large runtime artifacts belong under `runtime_assets/`, not source packages.
- OCR model paths should be provided through `MODEL_BASE_DIR` or `APP_MODELS_DIR`.
- The dependency image must stay source-free; bind mounts and process orchestration belong to Compose/startup scripts.

## Validation Checklist

- `python -m compileall app qa scripts`
- `python -m unittest discover -s tests`
- `python -c "import app.main; import app.ocr_compat_app"`
- `curl http://localhost:12000/test-connection`
- `curl http://localhost:11169/health` when the standalone OCR app is running.
