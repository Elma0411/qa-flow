# Integration Contract

This document defines the shared development contract for the `qa-flow`
repository. The filename intentionally follows the current project reference
`INTEGRATION_CONTRACT.md`.

Use this file when a change crosses the boundary between document extraction,
image understanding, integrated preprocessing, QA generation, evaluation, and
storage. Local implementation details should stay in the owning package docs or
code comments.

## Purpose

- Make parallel document-processing and QA-generation development possible
  without guessing hidden assumptions.
- Record the stable handoff shapes that are still represented as dictionaries
  in code.
- Define when both maintainers need to review a change.
- Keep `AI_PROGRAMMING_GUIDE.md` focused on high-level rules while this file
  carries field-level contracts.

This file is not a replacement for tests. When a boundary changes, update this
file and add or update focused contract tests in the same change.

## Ownership

- Document processing owns document extraction, OCR models, input adapters, watermark removal,
  image replacement, OCR-compatible text integration, image understanding, and
  VLM-specific parsing behavior.
- QA generation owns QA chunking, question generation, grounding, validation,
  evaluation, storage, Milvus search, admin workflows, and normal batch
  pipeline execution.
- Shared ownership applies to `app/services/integrated_pipeline/`, integrated
  route parameters, cross-pipeline file records, chunk metadata handoff,
  deployment dependencies, model paths, and runtime configuration that affects
  both OCR and QA.

## Change Classes

**Local Change**

A change is local when it stays inside one owner area and does not alter public
imports, endpoint behavior, runtime configuration, persisted outputs, or fields
consumed by another package. Local changes do not need this file updated.

**Boundary Change**

A change is a boundary change when it modifies any producer or consumer field
listed below, changes endpoint compatibility, changes required environment
variables, moves public facades, or changes error/status semantics. Boundary
changes must update this file and relevant tests.

**Shared Runtime Change**

A change is shared runtime work when it touches dependencies, Docker image or
Compose behavior, GPU/CPU behavior, model directories, OCR service startup,
LLM/VLM client configuration, Milvus connectivity, or artifact retention. Check
both formal and debug Docker Compose entries.

## Canonical Flows

**Standard QA Flow**

`upload -> app.services.ocr.resolve_uploaded_files_with_auto_ocr -> file_contents -> app.services.pipeline_execution.run_batch_complete_pipeline_async -> qa`

This flow keeps the batch endpoint behavior-compatible:
`POST /batch-upload-complete-pipeline-with-evaluation`.

**Integrated OCR-Image-QA Flow**

`upload -> OCRWorkerManager -> OCRResult -> marked markdown -> tree chunks -> image analysis -> placement judging -> final pre_split chunks -> run_batch_complete_pipeline_async -> qa`

This flow is exposed as:
`POST /batch-upload-integrated-document-pipeline`.

Integrated image analysis context is built from the image's own chunk. The VLM
prompt receives the chunk summary plus chunk text split at
`[[IMAGE_REF:image_id]]`; immediate OCR `context_before` and `context_after`
remain available for diagnostics but are not mixed into the VLM prompt context.
Async integrated requests must create the task status before document
preprocessing starts. Document preprocessing progress is stored in the normal
task status `file_progress` map using `doc_*` stage names, then QA stages
continue in the same per-file stage map.

**OCR-Compatible Flow**

`POST /process -> DocumentPipeline -> PDF conversion when needed -> OCR pipeline -> selected text/markdown output`

This flow must keep `output_format=text|markdown|ocr_markdown` compatible.
Standalone async document processing is exposed separately through
`/document-processing/jobs`; it must not change the compatibility behavior of
`POST /process`.

## Runtime Configuration Surface

VLM API configuration:

- Frontend LLM profiles persist `api_key`, `base_url`, `model`, `api_type`,
  and `model_version`. Activating a profile updates the backend runtime config
  used by the unified `app.services.llm` client path.
- LLM profile `api_type` supports `openai` and `lmp_cloud`; `model_version` is
  optional. Advanced client controls remain environment/default driven.
- Request-level VLM fields, when present, take precedence over environment
  variables.
- Integrated complete-pipeline requests may pass `vlm_api_base`,
  `vlm_model_name`, `vlm_api_key`, `vlm_api_type`, and `vlm_model_version` for
  image understanding. When omitted, integrated preprocessing may fall back to
  the backend active LLM/VLM configuration.
- Supported environment variables are `VLM_API_BASE`, `VLM_MODEL_NAME`,
  `VLM_API_KEY`, `VLM_API_TYPE`, and `VLM_MODEL_VERSION`.
- `VLM_API_TYPE` defaults to `openai`. Endpoint, model, and key have no code
  business defaults; enabling image analysis without them must fail with a clear
  configuration error instead of attempting a hardcoded local endpoint.
- API keys must not be persisted into task status payloads or frontend
  localStorage caches.

OCR image replacement:

- `OCR_REPLACE_IMAGES` controls the default local OCR image replacement
  behavior and defaults to `true` when unset.
- `POST /process` and `POST /batch-upload-integrated-document-pipeline` accept
  optional `replace_images` form parameters. Request parameters override the
  environment default.
- Integrated preprocessing records the resolved value in task status
  (`replace_images`) and each `ocr_summary` item.

Docker API ports:

- `QA_FLOW_API_HOST_PORT` controls the host port mapped to container port
  `12000` and defaults to `12000`.
- `OCR_API_HOST_PORT` controls the host port mapped to container port `11169`
  and defaults to `11169`.

Document preprocessing concurrency:

- `DOC_MAX_CONCURRENCY` controls the default number of uploaded files processed
  concurrently during integrated document preprocessing. It defaults to `1`.
- `OCR_MAX_CONCURRENCY` controls the number of concurrent OCR extraction calls
  admitted by integrated document preprocessing. It defaults to `1`.
- `IMAGE_ANALYSIS_MAX_CONCURRENCY` controls concurrent VLM image-analysis
  calls in API mode. It defaults to `1`.
- `IMAGE_FIT_MAX_CONCURRENCY` controls concurrent image placement-fit checks.
  It defaults to `1`.
- `VLM_API_MAX_CONCURRENT_REQUESTS` remains the per shared VLM client request
  gate. Raising image-analysis concurrency without raising this gate may still
  serialize calls for the same VLM profile.
- `POST /batch-upload-integrated-document-pipeline` accepts optional
  `doc_max_concurrency`, `ocr_max_concurrency`,
  `image_analysis_max_concurrency`, and `image_fit_max_concurrency` form
  fields. Request fields override environment defaults for that request.
- `docx_strategy` is accepted for compatibility, but document processing
  normalizes DOC and DOCX handling to PDF conversion before OCR.

Image classifier classes:

- The classifier class catalog is loaded from `CLASSIFIER_CLASS_CONFIG_FILE`,
  then `${CLASSIFIER_MODEL_DIR}/classes.json`, then the built-in 10-class
  fallback.
- `classes.json` must be a JSON array of objects with exactly `class_id`,
  `model_label`, `category_key`, and `display_name`.
- Existing but invalid class config files must fail service startup. Missing
  files fall back to the next candidate.

## Contract A: OCRResult And ImageInfo

Producer:

- `app/services/document_processing/`

Consumers:

- `app/services/image_understanding/`
- `app/services/integrated_pipeline/`
- `app/services/document_processing/text_integrator/`

Stable `OCRResult` fields:

- `pdf_name`: source document name.
- `total_pages`: page count when known.
- `markdown_content`: OCR markdown content containing image `<div>` tags
  before integrated marker replacement.
- `images_info`: ordered list of `ImageInfo`.
- `figure_titles`: optional figure title metadata.
- `processing_time`: OCR extraction seconds.
- `output_dir`: directory where relative image paths can be resolved.
- `to_dict()`: serializable summary for status/debug output.

Stable `ImageInfo` fields:

- `image_id`: stable image identifier; must match marker and description IDs.
- `file_path`: absolute path or path relative to `OCRResult.output_dir`.
- `page_number`: source page number when available.
- `div_tag`: original markdown image block used for marker replacement.
- `context_before` and `context_after`: immediate OCR context.

Rules:

- Do not change `image_id` semantics without updating image analysis,
  integrated marker logic, and this contract.
- Keep image paths resolvable until downstream image analysis finishes.
- If markdown no longer contains image `<div>` tags, provide an equivalent
  stable marker source before integrated chunking.
- Integrated image understanding should use the chunk summary and marker-split
  chunk text as its prompt context. Immediate OCR context is retained as
  metadata and may be used for diagnostics, but should not be concatenated into
  the image VLM prompt by default.
- Adding optional fields is allowed when consumers tolerate absence.

## Contract B: File Content Record

Producer:

- Standard QA flow: `resolve_uploaded_files_with_auto_ocr`
- Integrated flow: `resolve_uploaded_files_with_integrated_processing`

Consumer:

- `run_batch_complete_pipeline_async`

Each uploaded source is represented as one dictionary with these stable keys:

- `filename`: original safe display filename.
- `content`: preferred content string for classification and QA.
- `size`: character count of `content`.
- `status`: `success`, `error`, or internal pending states before final
  handoff.
- `error`: human-readable error when `status=error`.
- `ocr_seconds`: OCR/extraction elapsed seconds, or `0.0` for local text.
- `content_format`: `markdown` or `text`.
- `markdown_content`: markdown version when available.
- `plain_text`: plain text version when available.
- `ocr_pages`: OCR page records when provided by an external OCR service.
- `ocr_raw_entry`: raw OCR/debug payload when available.
- `pre_split_chunks`: optional list of final chunk texts already prepared for
  QA.
- `pre_split_chunk_meta`: optional list of metadata aligned with
  `pre_split_chunks`.
- `chunking_report`: optional report from the chunking stage.

Success rules:

- `status=success` requires a non-empty `content` string unless a future
  documented binary handoff is introduced.
- If `pre_split_chunks` is present, `pre_split_chunk_meta` must also be present,
  non-empty, and aligned by `chunk_index`.
- The QA execution layer may skip re-chunking only when both `pre_split_chunks`
  and `pre_split_chunk_meta` are valid lists.

Error rules:

- `status=error` must include `error`.
- Error records should still preserve `filename`, `content_format`, and
  `ocr_seconds` when available.

## Contract C: Pre-Split Chunk Metadata

Producer:

- `qa.chunking.build_tree_chunks`
- Integrated preprocessing when it enriches chunk metadata with image results.

Consumers:

- `run_batch_complete_pipeline_async`
- `qa.process_text_to_qa_one_step`
- document chunk storage
- QA source attribution and search/admin views

Stable metadata keys:

- `chunk_index`: 1-based integer position.
- `chunk_id`: stable chunk identifier.
- `text`: chunk text used for generation display.
- `text_for_embedding`: text used for retrieval/embedding; may include accepted
  image descriptions.
- `index_path`: tree path within the document.
- `title_path`: human-readable heading path.
- `parent_index_path`: parent tree path.
- `root_index_path`: root tree path.
- `level`: heading/tree level.
- `path_summary`: optional concise path summary.
- `split_type`: chunking mode.
- `doc_id`: document identifier; execution may set or normalize this.
- `task_id`: pipeline task ID; execution may set or normalize this.
- `original_filename`: source filename; execution may set or normalize this.
- `image_context_summary`: optional integrated image context summary.
- `image_replacements`: optional integrated image placement details.

Rules:

- `pre_split_chunks[index - 1]` must correspond to metadata with
  `chunk_index=index`.
- `chunk_id` must remain stable enough to be used as QA `source`.
- `text_for_embedding` should preserve all facts needed for retrieval.
- Integrated image descriptions should be inserted before QA generation and
  before embedding text is built.

## Contract D: QA Job Context

Producer:

- Batch and integrated FastAPI routes.

Consumer:

- `app.services.pipeline_execution.run_batch_complete_pipeline_async`

Required groups:

- Identity and input: `task_id`, `file_contents`, `status_data`.
- Generation: `chunk_size`, `qa_per_chunk`, `qa_detail_mode`,
  `prompt_language`, `question_type_mode`, `question_types`,
  `question_type_weights`, `few_shot_examples`.
- Chunking: `chunking_prefix_max_depth`, `chunking_split_type`,
  `chunking_markdown_heading_correction_enabled`,
  `chunking_text_split_min_length`, `chunking_text_split_max_length`,
  `chunking_chunk_overlap`, `chunking_separator`, `chunking_separators`,
  `chunking_split_language`, `chunking_custom_separator`,
  `chunking_manual_split_points`.
- Evaluation: `include_evaluation`, `include_unsupervised_evaluation`,
  `evaluation_method`, `faithfulness_hypothesis_mode`,
  `faithfulness_hypothesis_max_concurrency`, `filter_by_threshold`,
  `score_threshold`, `criteria_list`, `eval_max_concurrency`.
- Storage: `save_mode`, `enable_vector_storage`, `enable_chunk_storage`,
  `chunk_storage_fail_fast`.
- Runtime: `llm_config`, `max_concurrency`, `chunk_max_concurrency`,
  `chunk_max_attempts`, `augment_per_qa`, `augment_max_concurrency`.
- Classification: `knowledge_classifier`, `use_category_prompt_templates`.
- Integrated image understanding: `enable_image_analysis`,
  `enable_image_classification`, `classification_confidence_threshold`,
  `image_context_summary_mode`, `image_fit_check_enabled`,
  `image_fit_min_score`, and request-level VLM override fields when present.

Rules:

- New route parameters that affect QA generation, chunking, evaluation,
  storage, or runtime behavior must be added to both the route status payload
  and `job_context` when they need to be visible after scheduling.
- Standard and integrated pipeline task status payloads include task-level
  `created_at` and `updated_at`. `created_at` is set once when the task is
  accepted; `updated_at` changes on status updates.
- Do not add route-only defaults that differ between standard and integrated
  flows unless the difference is documented here.
- Integrated document progress stages use `file_progress[filename].stages` with
  stage names prefixed by `doc_` (`doc_input`, `doc_ocr`,
  `doc_pre_chunking`, `doc_image_analysis`, `doc_placement`, `doc_handoff`,
  and error variants). They are additive and must not remove later QA stage
  entries.
- Every standard and integrated `file_progress[filename].stages[stage]` entry
  records generic timing metadata:
  - `started_at`: first time that stage entry was written.
  - `updated_at`: latest status update for that stage.
  - `elapsed_seconds`: seconds between `started_at` and the latest update.
  - `completed_at`: present when the stage reaches a terminal state such as
    `completed`, `failed`, or `canceled`.
  Stage-specific timing in `extra` remains authoritative for domain metrics
  such as QA candidate generation, retrieval, and answer generation; generic
  `elapsed_seconds` is the fallback for live progress display.
- QA generation timing may include both wall-clock and cumulative diagnostic
  views:
  - `generation_wall_detail`: wall-clock attribution for the QA generation
    document run. Its `candidate_question_seconds`, `retrieval_seconds`,
    `answer_generation_seconds`, `validation_and_bookkeeping_seconds`, and
    `scheduler_gap_seconds` sum to `document_total_seconds` within normal
    floating-point tolerance. Frontend main timing views must use this object
    when present.
  - `generation_cumulative_detail`: per-worker cumulative diagnostics across
    concurrent chunks. These values can be much larger than wall-clock elapsed
    time and must not be added to task or stage totals.
  - `generation_detail`: retained for compatibility. New tasks write the
    wall-clock view here; consumers that need an explicit contract should read
    `generation_wall_detail`.
  `generation_chunk_details` remains the compact per-chunk diagnostic list and
  does not carry raw timing intervals.
- `doc_handoff` means document preprocessing has produced `file_contents` /
  `pre_split_chunks` for QA; it is not the terminal state of the full pipeline.

## Contract D1: Standalone Document Job Status

Producer:

- `/document-processing/jobs`
- `app.services.document_processing.jobs.DocumentProcessingJobManager`

Consumers:

- static frontend document-processing panel
- operators inspecting persisted job store

Stable job fields:

- `job_id`: document job identifier.
- `status`: `queued`, `running`, `completed`, `failed`, `canceled`, or
  temporary cancellation states.
- `message`: human-readable current status or error reason.
- `input_filename`: original display filename.
- `params`: normalized document-processing parameters.
- `file_progress`: same stage map shape as pipeline task status.
- `result`: final `DocumentPipeline` result when completed.
- `files`: existing output file paths keyed by `text`, `markdown`,
  `ocr_markdown`, `summary`, and `image_analysis_summary` when available.

Rules:

- Job progress callbacks are optional and must never fail the underlying
  document processing.
- `files` should only include outputs that exist on disk.
- `/process` remains the synchronous compatibility endpoint; async job APIs are
  additive.

## Contract E: QA Item Output

Producer:

- `qa.process_text_to_qa_one_step`
- optional QA augmentation.

Consumers:

- evaluation services
- consolidated JSON/CSV writers
- Milvus storage
- admin/search views

Stable fields for primary QA items:

- `question`
- `answer`
- `source_fact_text`
- `source`
- `chunk_index`
- `knowledge_category`
- `knowledge_category_confidence`
- `knowledge_category_reason`
- `question_type`
- `difficulty_level`
- `difficulty_score`
- `qa_generation_unit_id`
- `qa_generation_unit_text`
- `evidence_hits`
- `evidence_chunk_ids`

Rules:

- `source` should be normalized to the stable `chunk_id` when chunk metadata is
  available.
- Evaluation should prefer `qa_generation_unit_text` as source context when it
  exists.
- Optional enrichment fields are allowed, but removal or semantic change of the
  stable fields is a boundary change.

## Public Import Rules

- Repository code should import package capabilities through `__init__.py`
  facades unless a module is private to the importing package.
- A direct import from another package's implementation file is a boundary
  dependency. Avoid adding new ones.
- If an internal implementation must be shared, promote it through the owning
  package facade first.

## Boundary Review Checklist

Before merging a boundary change:

- Update this file.
- Update `AI_PROGRAMMING_GUIDE.md` if ownership, canonical flow, endpoint
  compatibility, or runtime rules changed.
- Update `CONTEXT.md` if new shared terminology is introduced.
- Update `AGENTS.md` if future agents need a new standing rule.
- Add or update contract tests for changed handoff fields.
- Run at least `python -m compileall app qa scripts` and
  `python -m unittest discover -s tests` in the appropriate runtime.
- For runtime/deployment changes, check `docker/docker-compose.yml`,
  `docker/docker-compose.debug.yml`, and shared scripts under `docker/`.

## Future Code-Level Contracts

The current handoffs still use dictionaries in several places. When boundary
fields start changing frequently, prefer adding dataclass or Pydantic models for
the relevant handoff first, then keep this document as the human-readable
summary of the same contract.
