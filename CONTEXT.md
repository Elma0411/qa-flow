# QA Pipeline Context

This context defines the project language used when discussing the QA
generation and service pipeline.

## Language

**Processing Stage**:
A cohesive step in the QA pipeline with a clear responsibility, such as
chunking, generation, grounding, validation, evaluation, storage, or search.
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

**Shared Boundary**:
A module, field set, endpoint, runtime variable, or deployment asset that both
`dw` and `hao` work depends on.
_Avoid_: Single-owner implementation detail

**Contract Test**:
A focused test that verifies a documented boundary shape or compatibility
behavior, especially where dictionaries still carry cross-module data.
_Avoid_: Broad smoke test, incidental coverage
