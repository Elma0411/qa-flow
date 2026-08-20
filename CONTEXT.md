# QA Pipeline Context

This context defines the project language used when discussing the QA
generation and service pipeline.

## Language

**Processing Stage**:
A cohesive step in the QA pipeline with a clear responsibility, such as
chunking, question/answer generation, structural normalization, evaluation,
storage, or search.
_Avoid_: Misc module, utility bucket, file group

**Full Pipeline**:
The end-to-end orchestration that connects processing stages into one runnable
service workflow.
_Avoid_: Feature folder, helper collection

**Service Capability**:
A cohesive backend service responsibility that supports one processing stage or
one operational concern, such as evaluation, storage, search, OCR, or runtime
health.
_Avoid_: Service bucket, miscellaneous backend helper

**Assembly Layer**:
The thin FastAPI wiring layer that creates the app, mounts middleware and
static assets, registers routers, and manages lifecycle hooks.
_Avoid_: Business service layer, pipeline logic, module bucket

**Public Facade**:
The only supported import surface for a package, usually its `__init__.py`.
_Avoid_: Importing internal implementation files directly

**Stateful Capability**:
A service capability that owns mutable runtime state or external resources,
such as clients, connections, caches, locks, schedulers, or loaded models.
_Avoid_: Stateless helper, pure function bundle

**Runtime QA Code**:
QA code that is called by the API service, pipeline execution path, evaluation
runtime, or operational scripts.
_Avoid_: Research asset, benchmark-only code, dataset conversion utility

**Integration Contract**:
The documented handoff agreement between `dw`-owned extraction/image stages,
shared integrated preprocessing, and `hao`-owned QA/evaluation/storage stages.
The active field-level contract is `INTEGRATION_CONTRACT.md`.
_Avoid_: Informal note, best-effort reminder, stale interface comment

**Boundary Change**:
A change that modifies data consumed by another owner area, public endpoint
behavior, runtime configuration, deployment dependencies, or persisted output
shape.
_Avoid_: Local refactor, implementation cleanup, private helper change

**File Content Record**:
The per-upload dictionary passed into batch execution after local text reading,
external OCR, or integrated OCR-image preprocessing.
_Avoid_: Upload object, OCR response, raw file metadata

**Pre-Split Chunk Metadata**:
The structured metadata aligned with `pre_split_chunks`, used for generation,
retrieval, storage, source attribution, and admin/search views.
_Avoid_: Debug chunk info, optional display-only data

**Section Node**:
A structural heading location identified by `section_path`. It may own body
content and child sections at the same time; an empty section is still a real
structural node.
_Avoid_: Treating every content chunk as a leaf section

**Content Chunk**:
An independently addressable body record with a stable `chunk_id` and
`section_chunk_index`, attached to a section but not itself a tree node.
_Avoid_: Encoding the chunk position into `section_path`

**Physical Fragment**:
One storage-sized piece of a content block. Members share
`fragment_group_id` and are restored together before evidence use.
_Avoid_: `Part N/M` as a heading level

**Evidence Window**:
A transient, query-specific grouping of real content chunks produced after
atomic reranking. It has no persisted fake chunk ID; attribution remains on
the member `chunk_id` values.
_Avoid_: Persisted aggregate parent text

**Section Material**:
The atomic source material used for question planning. It contains the body,
physical fragments, and typed accepted-image blocks belonging to one exact
`section_path`; ordinary `text_content` and `image_materials` remain distinct
for planning, and it never means all children under a chapter.
_Avoid_: Parent-chapter aggregate, arbitrary chunk group

**Question Scenario**:
An evidence-bound reader need selected before wording a question.
`PointScenario` requires one atomic fact from one Section Material;
`SummaryScenario` requires exactly two or three related atomic sub-questions
that jointly answer one coherent need.
_Avoid_: A generated question, a fixed section type, forced question quota

**Summary Hop**:
One auditable atomic sub-question inside a `SummaryScenario`. It binds one
Section Material and declares whether that contribution needs text, visual, or
mixed evidence plus any required image IDs. Several hops may reuse one material
only when they represent genuinely different parts of a real enumeration.
_Avoid_: Treating every selected material as a hop, a free-form reasoning trace

**Scenario Contract**:
The immutable backend form of a Question Scenario after temporary aliases have
been mapped. It owns scenario type, Summary Hops when applicable, optional
materials, and question type. Point evidence fields are frozen directly;
Summary required materials, images, and overall mode are derived from its hops.
Wording models may only rewrite the reader-facing question.
_Avoid_: Letting a wording editor downgrade mixed/visual evidence to text

**Shared Boundary**:
A module, field set, endpoint, runtime variable, or deployment asset that both
`dw` and `hao` work depends on.
_Avoid_: Single-owner implementation detail

**Contract Test**:
A focused test that verifies a documented boundary shape or compatibility
behavior, especially where dictionaries still carry cross-module data.
_Avoid_: Broad smoke test, incidental coverage
