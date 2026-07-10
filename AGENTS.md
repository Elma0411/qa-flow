# Repository Guidelines

## Project Structure & Module Organization
The FastAPI application factory lives in `app/main.py`; `api_server.py` and `scripts/start_api.py` are thin launchers only. Domain-specific processors and batch utilities live under `qa/` (use the `qa` package facade for the full text-to-QA pipeline and `qa.qa_evaluation` for scoring). Persistent artifacts and vector database volumes are isolated under `runtime_assets/`, `milvus_data/`, and `volumes/`, while `static/` hosts demo assets. Docker deployment assets live under `docker/` and should remain aligned with the QA Flow API at `http://localhost:12000` and the QA Flow OCR API at `http://localhost:11169`.

## Build, Test, and Development Commands
- `python -m venv .venv && .venv\Scripts\activate` (or `source .venv/bin/activate`) keeps Milvus and FastAPI deps isolated.
- `pip install -r requirements.txt` installs the API server plus QA evaluation models.
- `python scripts/start_api.py` launches the dev server with reload.
- `uvicorn api_server:app --host 0.0.0.0 --port 12000 --reload` is preferred for iterative debugging.
- `docker compose -f docker/docker-compose.yml up -d` creates the Milvus + QA Flow API + QA Flow OCR API stack for parity tests.

## Docker Deployment Layout
This repository keeps one Docker dependency image and two Compose entry points.
Treat all files under `docker/` as one deployment surface during design,
implementation, verification, and documentation updates.

- `docker/Dockerfile` is a pure dependency image derived from
  `/data2/hjk/Dockerfile`; it should not copy application source or own process
  orchestration.
- `docker/docker-compose.yml` is the formal deployment entry. It starts etcd,
  MinIO, Milvus, the QA Flow API, and the QA Flow OCR API.
- `docker/docker-compose.debug.yml` is the attachable debug entry. It keeps the
  main APIs stopped and opens bash for manual process debugging.
- When a change affects dependencies, startup commands, environment variables,
  mounted paths, resource limits, device selection, model loading, or runtime
  behavior, check the Dockerfile, both Compose files, and shared startup scripts
  together.

## Coding Style & Naming Conventions
Write Python 3.10+ code with 4-space indentation, `snake_case` functions, `PascalCase` classes, and UPPER_SNAKE_CASE env constants. Keep FastAPI routers declarative, place one Pydantic request/response model near its endpoint, and move heavy logic into helper modules under `qa/` or new packages rather than bloating `api_server.py`. Prefer descriptive filenames such as `milvus_ingest_service.py` and keep async endpoints `async def`. Reuse type hints and docstrings so auto-generated API docs stay accurate.

## Testing Guidelines
Run meaningful dependency, import, API, and pipeline verification inside the
Docker runtime by default. `localhost:12000` is the host-facing QA Flow UI/API
debug address, but the project dependencies and model/runtime environment are
owned by Docker. Smoke-test every change with `curl
http://localhost:12000/test-connection` or an equivalent Postman call after the
Docker service is running. For content generation pipelines, use the batch
complete endpoint or the one-step pipeline module that backs it. Quality gates
rely on the evaluator: `python qa/qa_evaluation/llm_quality_evaluator.py
--input-text qa/1.1.txt --qa-file runtime_assets/outputs/doc.qa.json --output
runtime_assets/outputs/doc.eval.json`. Capture sample requests/responses under
`runtime_assets/outputs/` for regression review and delete API keys from
configs before committing.

## Docker runtime testing context
The project may be mounted into a running Docker runtime, and meaningful API /
dependency tests may need to be executed inside that container rather than on
the host Python environment. Before treating missing host dependencies as a
project failure, check the active containers with `docker ps`.

Known runtime containers from the current environment:

- `qa-flow-runtime`: formal runtime container, with host port
  `12000` mapped to the QA Flow API and host port `11169` mapped to the
  QA Flow OCR API. Milvus-related host ports include `12530` -> `19530`,
  `12379` -> `2379`, `12900` -> `9000`, `12901` -> `9001`, and `12091` ->
  `9091`.
- `qa-flow-debug`: debug shell container from
  `docker/docker-compose.debug.yml`; the main APIs are intentionally stopped
  until started manually.

For runtime smoke tests, prefer `docker exec` into the relevant running
container when the host environment lacks project dependencies such as
`fastapi`, `openai`, `pymilvus`, or model libraries. Re-check container names
and health before relying on them, because `docker ps` state can change.

### Verification failure patterns seen during refactors

When a refactor appears to "fail verification", check these first:

- The host Python environment may be missing project dependencies. Do not
  treat host import errors as code failures until the same smoke test has been
  run inside the runtime container.
- The repo may be mounted at `/app` inside the runtime container, not at the
  path you used on the host. Always confirm the actual mount point before
  running import smoke tests.
- A container can stay `unhealthy` even when `/test-connection` is already
  working. In this repo that usually means the running container image is
  stale or its baked-in healthcheck script does not match the current Dockerfile.
- Import-time errors in package facades or singleton initialization order can
  stop Uvicorn before the API starts. After structural changes, smoke-test
  `import app.main` and the `/test-connection` endpoint inside the container.
- If a change touches Dockerfile content, bundled helper scripts, or
  healthcheck commands, rebuild the image and recreate the container before
  judging the result.

## Commit & Pull Request Guidelines
Follow an imperative, scoped format such as `feat(api): add batch upload guard`. Each commit should bundle related code, docs, and config changes; rerun the commands above before pushing. Pull requests must describe the scenario, reproduction steps, and any schema or env changes; attach screenshots or curl transcripts for new endpoints and link the corresponding issue ID.

## Security & Configuration Tips
Do not hard-code real OpenAI/DeepSeek keys--export them or use `.env` files ignored by git. Keep Milvus credentials in Docker secrets, and sanitize anything dropped into `milvus_data/` or `volumes/` before sharing.

## Codex Shell Environment
When running commands from Codex that depend on the user's interactive shell
setup, load `~/.bashrc` through an interactive bash invocation. This is required
for tools and helpers defined there, including `nvm`-provided `node`/`npm` and
the proxy helper functions.

- For Node-related commands (`node`, `npm`, `npx`, `codegraph`, and similar),
  use `bash -ic '<command>'` if the command is not found or may depend on
  `~/.bashrc`. In non-TTY Codex shells, `bash -ic` may print harmless job
  control warnings while still loading the environment correctly.
- For this repository's Codex setup, prefer the `nvm`/npm global CodeGraph
  executable. Verify the path with `bash -ic 'command -v codegraph'`, verify
  the version with `bash -ic 'codegraph version'`, keep the MCP `command`
  pointed at that same resolved executable, and remove or demote standalone
  installer paths such as `~/.local/bin/codegraph` if they shadow the `nvm`
  path.
- To refresh the `nvm` CodeGraph installation, run `bash -ic 'npm install -g @colbymchenry/codegraph@latest'`, then re-check both `command -v codegraph` and the MCP config path. Do not hard-code another user's home directory.
- For network commands such as `git clone`, `git pull`, `curl`, `npm install`,
  or package downloads, first try the normal command. If it times out or appears
  blocked by network access, retry through the proxy helpers from `~/.bashrc`,
  for example `bash -ic 'proxyon; <command>'`.
- For git downloads that need the git proxy helpers, use
  `bash -ic 'proxyon; gitproxyon; trap gitproxyoff EXIT; <git command>'` so the
  global git proxy is cleared after the command finishes.
- Do not create a separate `~/.shell_env` unless the user explicitly asks for
  that setup.

## Codex CodeGraph Workflow
When CodeGraph MCP tools are available, use them actively for repository
understanding, architecture tracing, impact analysis, and code modification
planning before opening many files manually. Treat the MCP tool surface as
version-dependent: use the tool or tools actually exposed by the current MCP
server, and prefer the broad exploration tool for area surveys, flow questions,
and edit planning. Current CodeGraph versions may expose only
`codegraph_explore`; do not assume narrower tools exist unless they are listed
in the active session.

If CodeGraph is unavailable in a conversation, tell the user briefly and then
try to fix it before falling back permanently. "Unavailable" includes the MCP
tool missing, `Transport closed`, hung tool calls, stale MCP sessions, missing
indexes, PATH mismatches, or version/configuration conflicts. Check local state
first with commands such as `command -v codegraph`, `codegraph version`,
`codegraph status <path>`, the agent MCP config, and the project's `.codegraph/`
directory. Resolve common local problems by aligning the MCP command with the
shell `PATH` resolved through `bash -ic`, restarting or reinstalling CodeGraph
through the same `nvm`/npm environment, applying upstream-recommended MCP
environment variables when relevant, and running `codegraph init`,
`codegraph sync`, or `codegraph index` from the repository root as appropriate.
If local checks do not explain the failure, consult upstream sources such as the
official README, release notes, and GitHub issues or PRs for the exact error and
apply the documented workaround. Use the CLI equivalent, such as
`codegraph explore`, as a temporary fallback while MCP is being restored.

If `codegraph` is not on `PATH`, locate the current user's installation first
rather than using another user's home directory. Each Linux user who works on
this repository needs their own CodeGraph executable/MCP setup and filesystem
permission to read the repository.

## Local Third-Party Reference Repositories
Use `external_repos/` as the local-only workspace for downloading third-party
open-source repositories that can be studied, compared, or adapted while
working on QA Flow.

- Do not commit or push `external_repos/` to the QA Flow GitHub remote.
- On each developer machine, add `external_repos/` to the parent repository's
  local `.git/info/exclude` so large reference repositories do not appear in
  normal `git status`.
- Keep every downloaded project in its own subdirectory and record its source
  URL and checkout/commit in that subdirectory when practical.
- Treat third-party code as reference material unless the user explicitly asks
  to import or adapt it. Preserve licenses and attribution when copying code.
- `external_repos/` is initialized as its own CodeGraph project. After adding
  or updating reference repositories, run `codegraph sync external_repos` from
  the QA Flow repository root, or `codegraph index --force external_repos` after
  large changes.
- When using CodeGraph MCP for reference code, pass
  `projectPath: "/data2/hjk/qa-flow/external_repos"`. Use the normal QA Flow
  repository CodeGraph project for first-party code.

## Collaboration rules
This section defines repository-level collaboration expectations for future
work. Treat these rules as default constraints when discussing requirements,
planning changes, and implementing code.

### First-principles thinking
Start from the original problem, the target outcome, and the actual business
constraint. Do not assume the user already has a complete or correct solution
path.

If the motivation, goal, scope, or acceptance standard is unclear, stop and
discuss the ambiguity before implementation. Do not fill missing business logic
with guesses.

When a new user task is vague, overloaded, or missing key scope/acceptance
details, use the local `grill-with-docs` skill before implementation. Ask
focused questions, one at a time, and continue only after the requirement is
clear enough to execute safely.

### Plan constraints
When proposing a modification or refactor plan, follow these constraints:

- Do not provide compatibility-oriented or patch-style plans.
- Do not overdesign. Choose the shortest valid path.
- Do not introduce fallback, downgrade, or extra out-of-scope schemes unless
  the user explicitly asks for them.
- Ensure the plan is logically correct and can pass end-to-end reasoning
  validation before implementation.

### Architecture and layering
- Keep `app/` as a thin FastAPI assembly layer. `app/main.py` owns app
  creation, middleware, lifespan/startup/shutdown, static mounts, and router
  registration; `api_server.py` and `scripts/start_api.py` only launch the
  app.
- Any capability that owns state or external resources must be class-based:
  `milvus`, `ocr`, `llm_config`, `artifacts`, admin job/meta stores, and
  `unsupervised_evaluation` runtime state should live behind a class, manager,
  or service object.
- Do not initialize heavy dependencies or long-lived connections at import
  time.
- Keep business workflows function-based: pipeline orchestration, stage
  handoff, data transformation, batching, validation, and similar flow logic
  should stay as functions unless they truly need shared state.
- For stateful capabilities, centralize mutable runtime objects in a class or
  manager and expose them through the package facade. Keep the callable API
  stable for existing routers and services, but do not reintroduce module-level
  state as the source of truth.
- Keep `app/routers/*` declarative. Routers should only adapt requests,
  validate input, and call facades; they should not host core business logic.
- Treat each package's `__init__.py` as the only public facade. Repository
  code should import from the facade, not from internal implementation files.
- Organize `qa/` by processing stage. Root-level files should be reserved for
  complete service entry points and necessary launch scripts; filenames should
  describe the stage or capability, not history, ambiguity, or opaque
  numbering.
- Put samples, fixtures, and test data in explicit `testdata/` locations
  instead of mixing them with runtime modules.
- Any structural refactor must preserve behavior, and after it lands the repo
  should update `LATEST_CHANGE_GUIDE.md` and pass Docker-based compile/import
  smoke verification before it is considered done.

### Parallel QA Flow development
- Treat `AI_PROGRAMMING_GUIDE.md` as the high-level common development
  protocol for the qa-flow repository.
- Treat `INTEGRATION_CONTRACT.md` as the field-level handoff contract for
  cross-owner data exchange.
- Changes contained within one owner area do not need contract updates unless
  they change public imports, endpoint behavior, runtime configuration,
  deployment dependencies, persisted output shape, or fields consumed by another
  owner area.
- Changes to `app/services/integrated_pipeline/`, `file_contents`,
  `pre_split_chunks`, `pre_split_chunk_meta`, `job_context`, OCR result/image
  contracts, model paths, LLM/VLM client behavior, or Docker deployment
  behavior are shared boundary changes. Update `INTEGRATION_CONTRACT.md` and
  focused tests with the code change.
- If a new shared field is introduced, document its producer, consumer,
  required/optional status, default behavior, and failure behavior. Do not add
  undocumented route-only defaults that differ between standard and integrated
  flows.

### Latest change guide maintenance
`LATEST_CHANGE_GUIDE.md` is the repository's current-change handoff document.
Keep it aligned with the newest effective change set whenever you implement a
real development change.

- After each substantive development task, update
  `X:\qanew\apiuse\LATEST_CHANGE_GUIDE.md`.
- Treat it as a "latest change" guide, not a cumulative changelog. Replace old
  guidance when it no longer describes the newest implementation.
- Include the current change's objective, the logic that changed, the expected
  behavior, and the most practical validation steps.
- If a task does not change behavior, logic, interface, deployment, metrics,
  or operational handling, you may leave `LATEST_CHANGE_GUIDE.md` unchanged.
- Do not finish a substantive implementation while leaving
  `LATEST_CHANGE_GUIDE.md` stale.

### Slimming initiative final handoff
For the long-running code-slimming and structure-optimization initiative, do
not require a separate deployment-summary document after each phase.

- When the slimming initiative is fully completed, produce one consolidated
  final handoff document.
- That final document must summarize the full set of changes across the whole
  initiative.
- That final document must include:
  - newly added files
  - files that must be uploaded to the server
  - files that must be overwritten on the server
  - files, directories, models, or large artifacts that do not need to be
    uploaded again
  - the minimal deployment/update steps needed on the server
- During the initiative, keep enough internal change tracking to ensure the
  final handoff document is complete and accurate.
- This consolidated final handoff document is separate from
  `LATEST_CHANGE_GUIDE.md`. Unless the user explicitly changes the rule,
  `LATEST_CHANGE_GUIDE.md` still follows the repository-level maintenance rule
  above for substantive implementation changes.

## Encoding
After making modifications, check the encoding of the changed files to ensure they are all in UTF-8 without BOM format. If any issues are found, please restore the correct encoding.
