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
- QA generation owns QA chunking, question/answer generation, structural
  normalization, evaluation, storage, Milvus search, admin workflows, and
  normal batch pipeline execution.
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

`upload -> OCRWorkerManager -> OCRResult -> seal-cleaned markdown -> marked markdown -> tree chunks -> image analysis -> placement judging -> final pre_split chunks -> run_batch_complete_pipeline_async -> qa`

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
Before marker replacement, integrated preprocessing removes OCR seal image
`<div>` tags whose image path contains `img_in_seal_box_`. These seal images do
not become image markers, VLM inputs, chunks, or QA evidence text.

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
- The Docker deployment does not publish classifier, Milvus, etcd, MinIO, or
  Milvus metrics ports to the host by default. They remain container-local and
  are probed through the runtime healthcheck and `GET /environment-check`.
- `QA_FLOW_API_RELOAD` controls whether the mounted QA Flow API runs with
  Uvicorn reload enabled. It defaults to `true` in the formal and debug Docker
  environment so changes under `app/`, `qa/`, and `scripts/` are picked up
  without recreating the container.

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

QA retrieval configuration:

- Standard and integrated complete-pipeline requests expose exactly two
  request-level retrieval controls:
  - `final_evidence_k`: final evidence-window count; defaults to `5`. `0`
    keeps only the generation unit's primary material.
  - `evidence_token_budget`: approximate total token budget for retrieved
    evidence windows; defaults to `4000` and has a minimum of `256`.
- Routes write the resolved values to both `job_context` and task status
  `retrieval_config`. The two complete-pipeline entry points must keep the same
  defaults and validation ranges.
- The internal path is fixed: deterministic question normalization -> standard
  BM25 plus BGE-M3 dense recall -> RRF -> chunk-ID de-duplication -> local
  BGE reranking of atomic chunks -> structural window completion -> BGE
  reranking of windows -> de-duplication and token budgeting.
- `QA_RERANKER_MODEL_PATH` defaults to
  `${APP_MODELS_DIR}/bge-reranker-v2-m3`; optional runtime controls are
  `QA_RERANKER_DEVICE`, `QA_RERANKER_BATCH_SIZE`, and
  `QA_RERANKER_MAX_LENGTH`. A missing/incomplete model or load failure is a
  task error; the service must not silently use the former hand-written ranker.
- BGE relevance admission is an internal, calibrated contract rather than a
  request parameter. Atomic candidates below raw logit `-1.0` or more than
  `8.0` below the best atomic candidate are rejected before structural window
  construction. They must also score within `1.0` of the best primary source
  chunk for the same question. Windows are reranked again and rejected below
  `-1.0`, more than `4.0` below the best window, or more than `2.0` below that
  primary-source baseline. These constants were calibrated with
  Chinese and English relevant pairs, same-topic hard negatives, and unrelated
  negatives for the bundled `bge-reranker-v2-m3`; the fixture and executable
  check live in `tests/testdata/bge_reranker_relevance_cases.json` and
  `scripts/calibrate_bge_relevance.py`. Changes require updating and running
  that calibration plus the focused tests. `final_evidence_k` is only an upper
  bound after admission, so the selected supplemental-window count may be zero.

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
- Integrated preprocessing may strip OCR-generated seal image divs matching
  `img_in_seal_box_*` before marker replacement. This does not alter
  `OCRResult.images_info`; it only prevents seal placeholders from entering
  downstream understanding and QA text.
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
- `section_chunk_index`: 1-based content-chunk position inside one section.
- `section_path`: structural section path; it never includes physical `Part N/M`
  labels or content-chunk positions.
- `section_parent_path`: parent section path, empty at the root.
- `section_level`: section depth.
- `section_is_leaf`: whether the section has child sections. This describes the
  section, not whether the content chunk is an image or a terminal record.
- `text`: chunk text used for generation display.
- `text_for_embedding`: text used for retrieval/embedding; may include accepted
  image descriptions.
- `title_path`: human-readable heading path.
- `fragment_group_id`, `fragment_index`, `fragment_count`: physical-split
  identity and order. All pieces in a group are restored together at evidence
  time.
- `content_kind`: `text`, `image_description`, or `mixed`.
- `source_asset_ids`: image/asset IDs that actually occurred in this chunk.
- `path_summary`: optional concise path summary.
- `split_type`: chunking mode.
- `doc_id`: document identifier; execution may set or normalize this.
- `task_id`: pipeline task ID; execution may set or normalize this.
- `original_filename`: source filename; execution may set or normalize this.
- `image_context_summary`: optional integrated image context summary.
- `image_replacements`: optional integrated image placement details local to
  this chunk only; unrelated accepted image IDs/details must not be copied in.

Rules:

- `pre_split_chunks[index - 1]` must correspond to metadata with
  `chunk_index=index`.
- `chunk_id` must remain stable enough to be used as QA `source`.
- `text_for_embedding` should preserve all facts needed for retrieval.
- Integrated image descriptions should be inserted before QA generation and
  before embedding text is built.
- A section may own body content and child sections simultaneously. An empty
  parent remains a structural node and does not produce duplicate aggregate
  body text.
- `chunk_id` includes `section_chunk_index` as part of its stable identity, so
  equal text repeated inside one section remains independently addressable.
- Structure completion first restores every physical fragment in a matched
  `fragment_group_id`. Multiple hits in one section are merged only when their
  `section_chunk_index` values are consecutive. A single atomic hit adds one
  preceding/following section chunk only when deterministic wording signals an
  explicit context dependency; no LLM hint or range policy participates.
- A Markdown heading plus accepted image marker, with no other visible body,
  is `image_description`; the heading alone does not make it `mixed`.
- Structural chunks have one canonical Milvus collection:
  `doc_content_chunks_v2` (`schema_version=2`). Its name is fixed in code and
  cannot be redirected to the removed `doc_tree_chunks` schema through config.
  There is no legacy read, write, or fallback collection. `POST
  /doc-chunks/rebuild` accepts source text plus chunking options, regenerates
  complete v2 metadata, validates and embeds all rows before replacing the
  matching `task_id + original_filename` scope, and restores the current v2
  rows if the replacement write fails.

## Contract D: QA Job Context

Producer:

- Batch and integrated FastAPI routes.

Consumer:

- `app.services.pipeline_execution.run_batch_complete_pipeline_async`

Required groups:

- Identity and input: `task_id`, `file_contents`, `status_data`.
- Generation: `chunk_size`, `qa_total_limit`, `qa_total_limit_scope`,
  `qa_detail_mode`, `prompt_language`, `question_type_mode`,
  `question_types`, `question_type_weights`, `few_shot_examples`.
  `qa_per_chunk` is retained only as a compatibility input when
  `qa_total_limit` is not supplied.
- Chunking: `chunking_prefix_max_depth`, `chunking_split_type`,
  `chunking_markdown_heading_correction_enabled`,
  `chunking_text_split_min_length`, `chunking_text_split_max_length`,
  `chunking_chunk_overlap`, `chunking_separator`, `chunking_separators`,
  `chunking_split_language`, `chunking_custom_separator`,
  `chunking_manual_split_points`.
- Evaluation: `include_evaluation`, `include_unsupervised_evaluation`,
  `evaluation_method`, `faithfulness_hypothesis_mode`,
  `faithfulness_hypothesis_max_concurrency`, `unsupervised_batch_size`,
  `faithfulness_nli_model`, `answerability_qa_model`,
  `coverage_embedding_model`, `filter_by_threshold`, `score_threshold`,
  `criteria_list`, `eval_max_concurrency`.
- Storage: `save_mode`, `enable_vector_storage`, `enable_chunk_storage`,
  `chunk_storage_fail_fast`.
- Runtime: `llm_config`, `max_concurrency`, `chunk_max_concurrency`,
  `chunk_max_attempts`, `augment_per_qa`, `augment_max_concurrency`.
- Retrieval: `final_evidence_k`, `evidence_token_budget`.
- Classification: `knowledge_classifier`, `use_category_prompt_templates`.
- Integrated image understanding: `enable_image_analysis`,
  `enable_image_classification`, `classification_confidence_threshold`,
  `image_context_summary_mode`, `image_fit_check_enabled`,
  `image_fit_min_score`, and request-level VLM override fields when present.

Rules:

- New route parameters that affect QA generation, chunking, evaluation,
  storage, or runtime behavior must be added to both the route status payload
  and `job_context` when they need to be visible after scheduling.
- The three model selection fields are optional model directory names. Empty,
  `auto`, and `default` mean the service-configured default. Non-empty values
  are validated against the shared catalog before a task is scheduled:
  `faithfulness_nli_model` selects the NLI classifier,
  `answerability_qa_model` selects the extractive SQuAD2 model, and
  `coverage_embedding_model` selects the SentenceTransformers embedding
  model. The selected names must resolve under `APP_MODELS_DIR`.
- `unsupervised_batch_size`, when present, is clamped to 1..512 and is passed
  to all local unsupervised metric runners. Qwen3-Embedding-4B deployments
  on 11GB GPUs should use 1.
- Supported local model directories are:
  `mdeberta_v3_base_xnli_nli_2mil7`, `erlangshen_roberta_110m_nli`,
  `xlm_roberta_large_xnli`, `deepset_xlm_roberta_base_squad2`,
  `deepset_xlm_roberta_large_squad2`, `bge-m3`,
  `qwen3_embedding_0_6b`, and `qwen3_embedding_4b`.
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
  `generation_unit_details` is the compact per-generation-unit diagnostic list.
  Each entry includes the unit index, unit type, selected QA mode, anchor chunk,
  source chunk indexes, target budget, generated item count, drop reasons, and
  per-worker timing. `generation_chunk_details` is retained as a compatibility
  alias for older frontend/status consumers and does not carry raw timing
  intervals.
- QA generation first reorganizes content into `section materials`: every
  logical `section_path` is one atomic material containing that section's body
  fragments, text, and accepted image descriptions. Different sections are
  never merged merely because they share a chapter or parent heading.
- A scenario-planning LLM then returns evidence-bound `PointScenario` and
  `SummaryScenario` candidates. Point scenarios bind exactly one section
  material and one fact need. Summary scenarios bind one material with a real
  multi-fact enumeration or multiple materials that jointly serve one reader
  need. Every material ID is validated against the supplied material catalog.
  In `qa_detail_mode=auto`, the planner builds both pools and the allocator
  targets 35% summary scenarios; missing summary capacity flows to point
  scenarios rather than being fabricated. Explicit `point` or `summary` mode
  selects only that pool. Each selected scenario generates exactly one main
  question. Large documents are planned in internal character-bounded batches:
  point batches may contain independent sections, while summary batches retain
  structural parent neighborhoods so related sibling sections remain visible.
  Pool selection and punctuation-insensitive final-question de-duplication are
  document-wide. If a planner call underfills the point pool, deterministic
  one-material point scenarios fill only the missing capacity; summary
  scenarios are never synthesized as fallback. No frontend/request batch-size
  control is exposed. A planner response is accepted only into the matching
  Point/Summary pool; a mismatched `scenario_type` is discarded. One section
  material may still contribute several Point scenarios when their intents
  cover distinct facts.
- Point and summary scenarios use distinct question-generation instructions.
  Every generated candidate then passes through one question-editor LLM call
  that returns `keep`, `rewrite`, or `drop`. The editor may naturalize wording,
  remove copied clause syntax and vague references, but may not change the
  scenario intent, evidence boundary, or question type. Only JSON shape,
  non-empty values, valid IDs/types, and exact duplicates are checked in code;
  linguistic naturalness is not decided by hard-coded rules.
- After the candidate-question and answer LLM calls, generation performs only
  structural normalization: required JSON fields, supported question types,
  valid multiple-choice options/correct option, valid judgment answers, and
  exact-question deduplication. It does not discard an otherwise structured QA
  item through ambiguous-reference, question-shape, source-fact segment,
  grounding, or source-anchor heuristics. Quality acceptance belongs to the
  downstream evaluation stage.
- In `qa_detail_mode=summary`, each generated item contains one standalone
  question with one central intent. The answer may summarize multiple related
  facts from one paragraph or a tightly connected passage group; unrelated
  questions must be emitted as separate items rather than concatenated.
- Answer `evidence_usage` references are resolved back through the exact
  `主材料-N` mapping. `source_chunk_id`, `source_chunk_index`, and
  `source_chunk_title_path` identify the first directly cited primary chunk;
  `source_chunk_ids`, `source_chunk_indexes`, and
  `source_chunk_title_paths` retain the complete ordered primary-evidence set
  for multi-material answers. A summary bound to multiple Section Materials is
  retained only when `evidence_usage` cites at least one primary chunk from
  every bound material; otherwise it enters the existing generation retry path.
  Retrieved-only evidence never becomes the scalar primary source.
- `qa_total_limit_scope=per_file` applies the total main-QA cap to each file.
  `qa_total_limit_scope=batch` pre-allocates the cap across successful files
  before concurrent generation so the final batch output does not exceed the
  requested main-QA total.
- `doc_handoff` means document preprocessing has produced `file_contents` /
  `pre_split_chunks` for QA; it is not the terminal state of the full pipeline.

## Contract D2: Dataset Evaluation Model Selection

Producer:

- `POST /eval/jobs`
- `static/eval.html` and `static/eval.js`

Consumer:

- `app.services.eval_jobs.evaluate_dataset_job`
- `app.services.unsupervised_evaluation.execute_unsupervised_suite_blocking`

Optional request fields:

- `faithfulness_nli_model`
- `answerability_qa_model`
- `coverage_embedding_model`
- `unsupervised_batch_size`

The job result stores the normalized model names under
`unsupervised.models`. This records the requested override (`auto` when no
override was supplied); it does not copy model weights into job artifacts.

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
- `evidence_usage`
- `retrieval_trace`

Rules:

- Primary QA item production does not imply a quality-pass decision. Consumers
  that need acceptance/filtering must use the evaluation fields and policies;
  generation-time structural validity alone is not a quality score.
- `source` should be normalized to the stable `chunk_id` when chunk metadata is
  available.
- Evaluation should prefer `qa_generation_unit_text` as source context when it
  exists.
- Candidate-question generation emits the natural question and question-quality
  fields only. Retrieval uses that question deterministically; it does not ask
  the LLM to emit a query, mandatory terms, or an answer-scope hint.
- LLM-facing material is separate from retrieval trace. Candidate-question
  generation receives readable source prose only. Answer generation receives
  readable sections labelled `主材料-N` or `检索证据-N`; it returns
  `evidence_ref` labels rather than real chunk IDs. The generation layer maps
  those labels back to `evidence_usage[].chunk_id` before persistence. Real IDs,
  title paths, ranks, and scores remain in `retrieval_trace` and debug artifacts.
- `llm_evidence_ref_map` is an ephemeral generation-unit handoff used for that
  mapping; it is not a persisted QA item field and must not be sent to the LLM.
- `retrieval_trace` is optional on old items. New primary QA items should carry
  it when generated by the same-document evidence flow. Stable diagnostic keys
  include `pipeline`, `query`, `dense_hits`, `bm25_hits`, `rrf_hits`,
  `atomic_rerank`, `window_candidates`, `selected_windows`,
  `selected_evidence_chunk_ids`, `final_evidence_k`,
  `evidence_token_budget`, and `evidence_tokens_estimated`.
- Optional enrichment fields are allowed, but removal or semantic change of the
  stable fields is a boundary change.

## Contract F: Pipeline Debug Artifacts And Manual Ingest

Producer:

- `qa.process_text_to_qa_one_step` writes per-task debug JSONL records.
- `app.services.pipeline_execution` registers pipeline artifacts and output
  records.

Consumers:

- pipeline status/debug UI
- manual review ingest workflow
- Milvus storage service

Stable task output fields:

- `debug_jsonl`
- `debug_json_files`
- `consolidated_json`
- `consolidated_csv`
- `history_source`
- `milvus_task_id`
- `vector_storage_result`
- `manual_ingest`
- `manual_ingest_selected_count`
- `manual_ingest_select_all`
- `artifacts_expire_at`

Integrated pipeline evaluation scores, reasons, filtering metadata, and timing
belong in `consolidated_json`. New integrated pipeline tasks do not create a
standalone `evaluation_json` / `evaluation_json_files` artifact; those fields
are legacy-compatible cleanup/read fields only when present on older task
records.

Public endpoints:

- `GET /pipeline-tasks/{task_id}/debug-jsonl`
  - Query: optional `chunk_index`, optional `event`.
  - Reads only debug JSONL basenames already registered in the task status.
  - Returns matching debug records, debug file basenames, filters, and
    `artifacts_expire_at`.
- `POST /pipeline-tasks/{task_id}/ingest-selected-qa`
  - Body: optional `source_file`, optional `selected_ids`, optional
    `select_all_task`.
  - Reads only the task's registered `consolidated_json` basename.
  - Writes selected valid QA items to Milvus and updates the task output record
    with manual ingest metadata.

Rules:

- Debug JSONL may contain prompts, source text, retrieval traces, and raw model
  output. It must be read through the task-scoped endpoint, not from arbitrary
  paths or browser cache.
- Automatic Milvus ingest may delete consolidated JSON/CSV/evaluation artifacts
  after success, but must keep debug JSONL registered until the artifact TTL.
- Manual ingest depends on non-expired `consolidated_json`; it does not
  regenerate, re-evaluate, or change QA filtering semantics.

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
