(() => {
  const $ = (id) => document.getElementById(id);

  let currentJobId = null;
  let currentJobSnapshot = null;
  let pollTimer = null;
  let lastPreviewData = null;
  let previewFileSignature = "";

  const RUNTIME_STATE_KEY = "apiuse_runtime_state_v1";
  const RUNTIME_PAGE = "eval";

  function loadRuntimeState() {
    try {
      const raw = localStorage.getItem(RUNTIME_STATE_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return parsed && typeof parsed === "object" ? parsed : { pages: {} };
    } catch (e) {
      return { pages: {} };
    }
  }

  function saveRuntimeState(state) {
    try {
      localStorage.setItem(RUNTIME_STATE_KEY, JSON.stringify(state || { pages: {} }));
    } catch (e) {
      // ignore
    }
  }

  function mutateRuntimePage(mutator) {
    const state = loadRuntimeState();
    if (!state.pages || typeof state.pages !== "object") state.pages = {};
    const page = state.pages[RUNTIME_PAGE] && typeof state.pages[RUNTIME_PAGE] === "object" ? state.pages[RUNTIME_PAGE] : {};
    mutator(page);
    state.pages[RUNTIME_PAGE] = page;
    saveRuntimeState(state);
    return page;
  }

  function getRuntimePage() {
    const state = loadRuntimeState();
    const pages = state.pages && typeof state.pages === "object" ? state.pages : {};
    return pages[RUNTIME_PAGE] && typeof pages[RUNTIME_PAGE] === "object" ? pages[RUNTIME_PAGE] : {};
  }

  function fmtWhen(ts) {
    if (ts === null || ts === undefined || ts === "") return "";
    try {
      const raw = String(ts).trim();
      const isNum = /^\d+$/.test(raw);
      const num = isNum ? Number(raw) : NaN;
      const date = Number.isFinite(num)
        ? new Date(raw.length > 11 ? num : num * 1000)
        : new Date(raw);
      if (!date || Number.isNaN(date.getTime())) return raw;
      return date.toLocaleString();
    } catch {
      return String(ts || "");
    }
  }

  function isTerminalJobStatus(status) {
    const st = String(status || "").toLowerCase();
    return st === "succeeded" || st === "failed" || st === "canceled";
  }

  function getJobStatusLabel(status) {
    const st = String(status || "").toLowerCase();
    if (st === "succeeded") return "已完成";
    if (st === "failed") return "失败";
    if (st === "canceled") return "已取消";
    if (st === "running") return "运行中";
    if (st === "queued") return "排队中";
    return st || "未知状态";
  }

  function resultHistorySource(result) {
    return String(result && result.history_source || "").trim().toLowerCase();
  }

  function isMilvusResult(result) {
    return resultHistorySource(result) === "milvus" && String(result && result.milvus_task_id || "").trim() !== "";
  }

  function hasLocalEvalArtifacts(result) {
    const files = result && typeof result.files === "object" ? result.files : {};
    return Boolean(files.scored_jsonl || files.summary_json);
  }

  function hasExpiredEvalArtifacts(result) {
    return Boolean(result && result.artifacts_deleted) && !isMilvusResult(result) && !hasLocalEvalArtifacts(result);
  }

  function normalizeJobRecord(job) {
    if (!job || typeof job !== "object") return null;
    const jobId = String(job.job_id || "").trim();
    if (!jobId) return null;
    const params = job.params && typeof job.params === "object" ? job.params : {};
    const result = job.result && typeof job.result === "object" ? job.result : {};
    return {
      job_id: jobId,
      status: String(job.status || ""),
      job_type: String(job.job_type || "eval"),
      dataset_name: String(params.dataset_name || job.dataset_name || ""),
      message: String(job.message || ""),
      created_at: job.created_at ?? null,
      started_at: job.started_at ?? null,
      finished_at: job.finished_at ?? null,
      history_source: resultHistorySource(result),
      artifacts_deleted: Boolean(result.artifacts_deleted),
      milvus_task_id: String(result.milvus_task_id || ""),
      has_local_artifacts: hasLocalEvalArtifacts(result),
    };
  }

  function getSelectedInputFiles() {
    const input = $("inputFile");
    return input && input.files ? Array.from(input.files) : [];
  }

  function buildFileSignature(files) {
    return (Array.isArray(files) ? files : [])
      .map((file) => `${file.name}:${file.size}:${file.lastModified}`)
      .join("|");
  }

  function getRecentJobs() {
    const page = getRuntimePage();
    return Array.isArray(page.recentJobs) ? page.recentJobs : [];
  }

  function replaceRecentJobs(jobs) {
    const normalized = Array.isArray(jobs)
      ? jobs
          .map((job) => normalizeJobRecord(job))
          .filter((job) => !!job)
          .slice(0, 20)
      : [];
    mutateRuntimePage((page) => {
      const ids = new Set(normalized.map((job) => String(job.job_id || "")));
      page.recentJobs = normalized;
      if (page.selectedJobId && !ids.has(String(page.selectedJobId || ""))) {
        page.selectedJobId = normalized[0] ? normalized[0].job_id : "";
      }
      if (page.activeJobId && !ids.has(String(page.activeJobId || ""))) {
        page.activeJobId = "";
      }
    });
    return normalized;
  }

  function removeRecentJob(jobId) {
    const normalized = String(jobId || "").trim();
    if (!normalized) return getRecentJobs();
    return mutateRuntimePage((page) => {
      const prev = Array.isArray(page.recentJobs) ? page.recentJobs : [];
      page.recentJobs = prev.filter((it) => String(it.job_id || "") !== normalized);
      if (String(page.selectedJobId || "") === normalized) {
        page.selectedJobId = page.recentJobs[0] ? String(page.recentJobs[0].job_id || "") : "";
      }
      if (String(page.activeJobId || "") === normalized) {
        page.activeJobId = "";
      }
    }).recentJobs || [];
  }

  function rememberJob(job, { select = true } = {}) {
    const record = normalizeJobRecord(job);
    if (!record) return getRecentJobs();
    return mutateRuntimePage((page) => {
      const prev = Array.isArray(page.recentJobs) ? page.recentJobs : [];
      const merged = [record, ...prev.filter((it) => String(it.job_id || "") !== record.job_id)];
      page.recentJobs = merged.slice(0, 12);
      if (select) {
        page.selectedJobId = record.job_id;
        page.activeJobId = isTerminalJobStatus(record.status) ? (page.activeJobId === record.job_id ? "" : page.activeJobId || "") : record.job_id;
      }
    }).recentJobs || [];
  }

  function setJobSelection(jobId, { active = null } = {}) {
    const normalized = String(jobId || "").trim();
    currentJobId = normalized || null;
    const input = $("jobIdInput");
    if (input && input.value !== normalized) {
      input.value = normalized;
      persistUiField(input);
    }
    mutateRuntimePage((page) => {
      page.selectedJobId = normalized;
      if (active === true) page.activeJobId = normalized;
      else if (active === false && page.activeJobId === normalized) page.activeJobId = "";
    });
  }

  function clearSelectedEvalJob(jobId) {
    const normalized = String(jobId || "").trim();
    if (!normalized) return;
    if (String(currentJobId || "").trim() === normalized) {
      currentJobId = null;
    }
    if (currentJobSnapshot && String(currentJobSnapshot.job_id || "").trim() === normalized) {
      currentJobSnapshot = null;
    }
    const input = $("jobIdInput");
    if (input && String(input.value || "").trim() === normalized) {
      input.value = "";
      persistUiField(input);
    }
    mutateRuntimePage((page) => {
      if (String(page.selectedJobId || "").trim() === normalized) page.selectedJobId = "";
      if (String(page.activeJobId || "").trim() === normalized) page.activeJobId = "";
    });
    clearResultPanels();
    setPre("jobStatus", null);
    $("jobHint").textContent = "历史任务已删除";
  }

  function renderRecentJobs() {
    const wrap = $("evalRecentJobs");
    if (!wrap) return;
    const jobs = getRecentJobs();
    wrap.innerHTML = "";
    if (!jobs.length) {
      const empty = document.createElement("div");
      empty.className = "task-history-empty";
      empty.textContent = "暂无最近评测任务。启动一次任务或输入 job_id 查询后，这里会保留最近记录。";
      wrap.appendChild(empty);
      return;
    }

    jobs.forEach((job) => {
      const row = document.createElement("div");
      row.className = "task-history-item";

      const main = document.createElement("div");
      main.className = "task-history-main";

      const title = document.createElement("div");
      title.className = "task-history-title";
      const pill = document.createElement("span");
      pill.className = "status-pill";
      pill.dataset.status = String(job.status || "").toLowerCase();
      pill.textContent = String(job.status || "unknown");
      title.appendChild(pill);
      const name = document.createElement("span");
      name.className = "task-history-name";
      if (job.history_source === "milvus") {
        name.textContent = job.dataset_name ? `${job.dataset_name}（已入库）` : "已入库评测任务";
      } else if (job.artifacts_deleted) {
        name.textContent = job.dataset_name ? `${job.dataset_name}（结果文件已清理）` : "结果文件已清理";
      } else {
        name.textContent = job.dataset_name || "未命名评测任务";
      }
      title.appendChild(name);
      const id = document.createElement("span");
      id.className = "task-history-id";
      id.textContent = job.job_id;
      title.appendChild(id);
      main.appendChild(title);

      const meta = document.createElement("div");
      meta.className = "task-history-meta";
      const timeText = fmtWhen(job.finished_at || job.started_at || job.created_at);
      if (timeText) {
        const span = document.createElement("span");
        span.textContent = `时间：${timeText}`;
        meta.appendChild(span);
      }
      if (job.message) {
        const span = document.createElement("span");
        span.textContent = `状态说明：${job.message}`;
        meta.appendChild(span);
      }
      if (job.history_source === "milvus") {
        const span = document.createElement("span");
        span.textContent = "结果已入库；明细与后续筛选请到管理页查询。";
        meta.appendChild(span);
      } else if (job.artifacts_deleted) {
        const span = document.createElement("span");
        span.textContent = "临时 scored / summary 文件已被清理，不能再直接下载。";
        meta.appendChild(span);
      }
      main.appendChild(meta);
      row.appendChild(main);

      const actions = document.createElement("div");
      actions.className = "task-history-actions";

      const viewBtn = document.createElement("button");
      viewBtn.type = "button";
      viewBtn.className = "secondary";
      viewBtn.textContent = "查看";
      viewBtn.addEventListener("click", () => {
        loadJobById(job.job_id, { resumePolling: true, autoLoadPage: true }).catch(() => {});
      });
      actions.appendChild(viewBtn);

      if (!isTerminalJobStatus(job.status)) {
        const stopBtn = document.createElement("button");
        stopBtn.type = "button";
        stopBtn.className = "danger";
        stopBtn.textContent = "终止";
        stopBtn.addEventListener("click", () => {
          const input = $("jobIdInput");
          if (input) {
            input.value = job.job_id;
            persistUiField(input);
          }
          cancelJob();
        });
        actions.appendChild(stopBtn);
      }

      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "danger";
      deleteBtn.textContent = "删除记录";
      deleteBtn.addEventListener("click", () => {
        deleteEvalJobHistory(job.job_id).catch(() => {});
      });
      actions.appendChild(deleteBtn);

      row.appendChild(actions);
      wrap.appendChild(row);
    });
  }

  // ---------- UI 缓存（跨页面保留输入状态） ----------

  const UI_CACHE_KEY = "apiuse_ui_cache_v1";
  const UI_CACHE_PAGE = "eval";

  function loadUiCache() {
    try {
      const raw = localStorage.getItem(UI_CACHE_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return parsed && typeof parsed === "object" ? parsed : { shared: {}, pages: {} };
    } catch (e) {
      return { shared: {}, pages: {} };
    }
  }

  function saveUiCache(cache) {
    try {
      localStorage.setItem(UI_CACHE_KEY, JSON.stringify(cache || { shared: {}, pages: {} }));
    } catch (e) {
      // ignore
    }
  }

  function getUiCachePage() {
    const cache = loadUiCache();
    const pages = cache.pages && typeof cache.pages === "object" ? cache.pages : {};
    const page = pages[UI_CACHE_PAGE] && typeof pages[UI_CACHE_PAGE] === "object" ? pages[UI_CACHE_PAGE] : {};
    return { cache, page };
  }

  function readCacheValue(el) {
    if (!el) return null;
    const tag = String(el.tagName || "").toLowerCase();
    const type = String(el.type || "").toLowerCase();
    if (type === "checkbox") return Boolean(el.checked);
    if (tag === "select" || tag === "textarea" || tag === "input") return String(el.value ?? "");
    return null;
  }

  function applyCacheValue(el, value) {
    if (!el) return;
    const type = String(el.type || "").toLowerCase();
    if (type === "checkbox") {
      el.checked = Boolean(value);
      return;
    }
    if (value === null || value === undefined) return;
    el.value = String(value);
  }

  function restoreUiCache() {
    const cache = loadUiCache();
    const shared = cache.shared && typeof cache.shared === "object" ? cache.shared : {};
    const pages = cache.pages && typeof cache.pages === "object" ? cache.pages : {};
    const page = pages[UI_CACHE_PAGE] && typeof pages[UI_CACHE_PAGE] === "object" ? pages[UI_CACHE_PAGE] : {};

    const apiInput = $("apiBaseUrl");
    if (apiInput && shared.apiBaseUrl) applyCacheValue(apiInput, shared.apiBaseUrl);

    Object.keys(page).forEach((id) => {
      const el = $(id);
      if (!el) return;
      if (String(el.type || "").toLowerCase() === "file") return;
      if (el.dataset && String(el.dataset.noCache || "").toLowerCase() === "true") return;
      applyCacheValue(el, page[id]);
    });
  }

  function persistUiField(el) {
    if (!el || !el.id) return;
    const id = String(el.id);
    if (!id) return;
    if (String(el.type || "").toLowerCase() === "file") return;
    if (el.dataset && String(el.dataset.noCache || "").toLowerCase() === "true") return;

    const cache = loadUiCache();
    if (!cache.shared || typeof cache.shared !== "object") cache.shared = {};
    if (!cache.pages || typeof cache.pages !== "object") cache.pages = {};
    if (!cache.pages[UI_CACHE_PAGE] || typeof cache.pages[UI_CACHE_PAGE] !== "object") {
      cache.pages[UI_CACHE_PAGE] = {};
    }

    const value = readCacheValue(el);
    if (id === "apiBaseUrl") {
      cache.shared.apiBaseUrl = value;
    } else {
      cache.pages[UI_CACHE_PAGE][id] = value;
    }
    saveUiCache(cache);
  }

  function bindUiCache() {
    const elements = document.querySelectorAll("input[id], textarea[id], select[id]");
    elements.forEach((el) => {
      if (String(el.type || "").toLowerCase() === "file") return;
      if (el.dataset && String(el.dataset.noCache || "").toLowerCase() === "true") return;

      const type = String(el.type || "").toLowerCase();
      const onChange = () => persistUiField(el);
      el.addEventListener("change", onChange);
      if (type === "text" || type === "number" || type === "search" || String(el.tagName || "").toLowerCase() === "textarea") {
        el.addEventListener("input", onChange);
      }
    });
  }

  function apiBase() {
    const input = $("apiBaseUrl");
    const raw = ((input && input.value) || "").trim();
    const origin = String(window.location.origin || "").replace(/\/+$/, "");
    const val = raw || origin;
    const normalized = normalizeApiBaseUrl(val, origin);
    return normalized.endsWith("/") ? normalized.slice(0, -1) : normalized;
  }

  function normalizeApiBaseUrl(value, fallbackOrigin) {
    const v = String(value || "").trim();
    const origin = String(fallbackOrigin || "").trim();
    if (!v) return origin;
    if (v.startsWith("http://") || v.startsWith("https://")) return v;
    if (v.startsWith("//")) return `${window.location.protocol}${v}`;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(v)) return v;
    if (/^[^\/\s]+(:\d+)?$/.test(v)) {
      const proto = (origin && origin.split("://")[0]) || window.location.protocol.replace(":", "");
      return `${proto}://${v}`;
    }
    return origin;
  }

  function initApiBaseUrl() {
    const input = $("apiBaseUrl");
    if (!input) return;
    const raw = (input.value || "").trim();
    const origin = String(window.location.origin || "").replace(/\/+$/, "");
    if (!origin) return;
    if (!raw || raw === "http://localhost:12000") input.value = origin;
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    const text = await res.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (e) {
      payload = { raw: text };
    }
    if (!res.ok) {
      const msg = (payload && (payload.detail || payload.error)) || text || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return payload;
  }

  function setPre(id, obj) {
    $(id).textContent = obj ? JSON.stringify(obj, null, 2) : "";
  }

  function populateSelect(selectEl, columns, { allowEmpty }) {
    selectEl.innerHTML = "";
    if (allowEmpty) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(空)";
      selectEl.appendChild(opt);
    }
    for (const col of columns) {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      selectEl.appendChild(opt);
    }
  }

  function guessSelect(selectEl, candidates) {
    const opts = Array.from(selectEl.options).map((o) => o.value);
    for (const c of candidates) {
      if (opts.includes(c)) {
        selectEl.value = c;
        return;
      }
    }
    // try case-insensitive
    const lowerMap = new Map(opts.map((o) => [String(o).toLowerCase(), o]));
    for (const c of candidates) {
      const hit = lowerMap.get(String(c).toLowerCase());
      if (hit) {
        selectEl.value = hit;
        return;
      }
    }
  }

  function renderScores(scores) {
    const grid = $("scoreGrid");
    grid.innerHTML = "";
    if (!scores) return;
    const cards = [
      { k: "faithfulness", name: "忠实度 faithfulness" },
      { k: "answerability", name: "可回答性 P=answerability" },
      { k: "coverage_score", name: "Coverage=coverage_score" },
      { k: "unsupervised_f1", name: "无监督 F1" },
    ];
    for (const c of cards) {
      const div = document.createElement("div");
      div.className = "score-card";
      const v = typeof scores[c.k] === "number" ? scores[c.k] : 0;
      div.innerHTML = `<div class="score-name">${c.name}</div><div class="score-val">${v.toFixed(4)}</div>`;
      grid.appendChild(div);
    }
  }

  function clearPreviewState({ clearOutput = false } = {}) {
    lastPreviewData = null;
    previewFileSignature = "";
    const overview = $("previewOverview");
    if (overview) overview.innerHTML = "";
    const tbody = $("fileRangeTbody");
    if (tbody) tbody.innerHTML = "";
    const panel = $("fileRangePanel");
    if (panel) panel.hidden = true;
    if (clearOutput) {
      const output = $("previewOutput");
      if (output) output.textContent = "";
    }
  }

  function parseOptionalPositiveInt(raw) {
    const text = String(raw ?? "").trim();
    if (!text) return null;
    if (!/^\d+$/.test(text)) {
      throw new Error("必须填写正整数");
    }
    const value = parseInt(text, 10);
    if (value < 1) {
      throw new Error("必须填写大于等于 1 的整数");
    }
    return value;
  }

  function resolveLocalRowRange(totalRows, rawStart, rawEnd) {
    const total = Math.max(0, parseInt(totalRows || 0, 10) || 0);
    const start = parseOptionalPositiveInt(rawStart);
    const end = parseOptionalPositiveInt(rawEnd);
    if (start !== null && end !== null && start > end) {
      throw new Error("起始行不能大于结束行");
    }
    if (total <= 0) {
      return { requested: { start, end }, resolved: { start: null, end: null }, selected_rows: 0 };
    }
    const resolvedStart = start === null ? 1 : Math.min(Math.max(1, start), total);
    const resolvedEnd = end === null ? total : Math.min(Math.max(1, end), total);
    if (resolvedStart > resolvedEnd) {
      throw new Error("起始行不能大于结束行");
    }
    return {
      requested: { start, end },
      resolved: { start: resolvedStart, end: resolvedEnd },
      selected_rows: Math.max(0, resolvedEnd - resolvedStart + 1),
    };
  }

  function updatePreviewOverview() {
    const wrap = $("previewOverview");
    if (!wrap) return;
    const files = Array.isArray(lastPreviewData && lastPreviewData.files) ? lastPreviewData.files : [];
    if (!files.length) {
      wrap.innerHTML = "";
      return;
    }
    const rows = Array.from(document.querySelectorAll("#fileRangeTbody tr[data-file-index]"));
    let selectedRows = 0;
    let invalidRows = 0;
    rows.forEach((row) => {
      const selected = parseInt(String(row.dataset.selectedRows || "0"), 10) || 0;
      selectedRows += selected;
      if (row.dataset.invalid === "true") invalidRows += 1;
    });
    const sharedColumns = Array.isArray(lastPreviewData.shared_columns) ? lastPreviewData.shared_columns : [];
    const schemaText = lastPreviewData.schema_consistent ? "一致" : "不一致";
    wrap.innerHTML = [
      { label: "文件数", value: String(files.length) },
      { label: "结构校验", value: invalidRows > 0 ? `范围待修正（${invalidRows}）` : schemaText },
      { label: "共享列数", value: String(sharedColumns.length) },
      { label: "当前选中行数", value: String(selectedRows) },
    ]
      .map(
        (item) =>
          `<div class="mini-stat"><div class="mini-stat-label">${item.label}</div><div class="mini-stat-value">${item.value}</div></div>`
      )
      .join("");
  }

  function updateDatasetNameFromFileRanges() {
    const dsEl = $("datasetName");
    if (!dsEl) return;
    const rows = Array.from(document.querySelectorAll("#fileRangeTbody tr[data-file-index]"));
    if (!rows.length) return;
    const resolved = [];
    for (const row of rows) {
      if (row.dataset.invalid === "true") return;
      const startInput = row.querySelector("input[data-role='row-start']");
      const endInput = row.querySelector("input[data-role='row-end']");
      const start = String(startInput ? startInput.value : "").trim();
      const end = String(endInput ? endInput.value : "").trim();
      if (!start || !end) return;
      resolved.push(`${start}-${end}`);
    }
    const uniqueRanges = Array.from(new Set(resolved));
    if (uniqueRanges.length !== 1) return;

    const cur = String(dsEl.value || "").trim();
    if (!cur) return;
    const parsed = parseRangeSuffix(cur);
    let base = parsed ? parsed.base : cur;
    if (!parsed && base && /\d$/.test(base) && !/[_-]$/.test(base)) base = `${base}_`;
    const next = `${base}${uniqueRanges[0]}`;
    if (next === cur) return;
    dsEl.value = next;
    persistUiField(dsEl);
  }

  function updateFileRangeRow(row) {
    if (!row) return;
    const totalRows = parseInt(String(row.dataset.totalRows || "0"), 10) || 0;
    const startInput = row.querySelector("input[data-role='row-start']");
    const endInput = row.querySelector("input[data-role='row-end']");
    const selectedEl = row.querySelector("[data-role='selected-rows']");
    const errorEl = row.querySelector("[data-role='schema-detail']");
    try {
      const range = resolveLocalRowRange(totalRows, startInput ? startInput.value : "", endInput ? endInput.value : "");
      row.dataset.invalid = "false";
      row.dataset.selectedRows = String(range.selected_rows || 0);
      row.dataset.rangeStart = range.resolved.start == null ? "" : String(range.resolved.start);
      row.dataset.rangeEnd = range.resolved.end == null ? "" : String(range.resolved.end);
      if (selectedEl) selectedEl.textContent = String(range.selected_rows || 0);
      if (errorEl) {
        const schemaNotes = [];
        const schemaStatus = String(row.dataset.schemaStatus || "ok");
        if (schemaStatus !== "ok") {
          if (row.dataset.missingColumns) schemaNotes.push(`缺列：${row.dataset.missingColumns}`);
          if (row.dataset.extraColumns) schemaNotes.push(`多列：${row.dataset.extraColumns}`);
        }
        errorEl.textContent = schemaNotes.join("；");
      }
    } catch (e) {
      row.dataset.invalid = "true";
      row.dataset.selectedRows = "0";
      row.dataset.rangeStart = "";
      row.dataset.rangeEnd = "";
      if (selectedEl) selectedEl.textContent = "无效";
      if (errorEl) errorEl.textContent = String(e.message || e);
    }
    updatePreviewOverview();
    updateDatasetNameFromFileRanges();
  }

  function renderFileRangeTable(files) {
    const panel = $("fileRangePanel");
    const tbody = $("fileRangeTbody");
    if (!panel || !tbody) return;
    tbody.innerHTML = "";
    const items = Array.isArray(files) ? files : [];
    panel.hidden = !items.length;
    if (!items.length) return;

    items.forEach((file) => {
      const tr = document.createElement("tr");
      tr.dataset.fileIndex = String(file.file_index ?? "");
      tr.dataset.totalRows = String(file.total_rows ?? 0);
      tr.dataset.schemaStatus = String(file.schema_status || "ok");
      tr.dataset.missingColumns = Array.isArray(file.missing_columns) ? file.missing_columns.join(", ") : "";
      tr.dataset.extraColumns = Array.isArray(file.extra_columns) ? file.extra_columns.join(", ") : "";
      const rowRange = file.row_range && typeof file.row_range === "object" ? file.row_range : {};
      const requested = rowRange.requested && typeof rowRange.requested === "object" ? rowRange.requested : {};

      tr.innerHTML = `
        <td class="mono">${file.file_index ?? ""}</td>
        <td>${file.filename || ""}</td>
        <td class="mono">${file.detected_format || ""}</td>
        <td class="mono">${file.total_rows ?? 0}</td>
        <td><input type="number" class="range-input" data-role="row-start" min="1" value="${requested.start ?? ""}" placeholder="起始行" /></td>
        <td><input type="number" class="range-input" data-role="row-end" min="1" value="${requested.end ?? ""}" placeholder="结束行" /></td>
        <td class="mono" data-role="selected-rows">${rowRange.selected_rows ?? 0}</td>
        <td>
          <span class="schema-badge" data-status="${file.schema_status || "ok"}">${file.schema_status === "ok" ? "一致" : "不一致"}</span>
          <div class="hint" data-role="schema-detail" style="margin-top:6px"></div>
        </td>
      `;
      tbody.appendChild(tr);
      tr.querySelectorAll("input[data-role='row-start'], input[data-role='row-end']").forEach((input) => {
        input.addEventListener("input", () => updateFileRangeRow(tr));
        input.addEventListener("change", () => updateFileRangeRow(tr));
      });
      updateFileRangeRow(tr);
    });
  }

  function collectFileRanges() {
    const rows = Array.from(document.querySelectorAll("#fileRangeTbody tr[data-file-index]"));
    const payload = [];
    for (const row of rows) {
      if (row.dataset.invalid === "true") {
        throw new Error("文件级评测范围存在无效配置，请先修正");
      }
      const startInput = row.querySelector("input[data-role='row-start']");
      const endInput = row.querySelector("input[data-role='row-end']");
      const start = parseOptionalPositiveInt(startInput ? startInput.value : "");
      const end = parseOptionalPositiveInt(endInput ? endInput.value : "");
      if (start !== null && end !== null && start > end) {
        throw new Error("文件级评测范围存在起止顺序错误");
      }
      if (start === null && end === null) continue;
      payload.push({
        file_index: parseInt(String(row.dataset.fileIndex || "0"), 10) || 0,
        row_start: start,
        row_end: end,
      });
    }
    return payload;
  }

  function renderResultSummaryMeta(summary) {
    const wrap = $("resultSummaryMeta");
    if (!wrap) return;
    wrap.innerHTML = "";
    const input = summary && typeof summary.input === "object" ? summary.input : {};
    const files = Array.isArray(input.files) ? input.files : [];
    if (!files.length) return;
    const counts = summary && typeof summary.counts === "object" ? summary.counts : {};
    const sharedColumns = Array.isArray(input.shared_columns) ? input.shared_columns : [];
    const cards = [
      { label: "输入文件数", value: String(input.input_files_count || files.length) },
      { label: "共享列数", value: String(sharedColumns.length) },
      { label: "评测样本数", value: String(counts.total || 0) },
      { label: "含参考答案样本数", value: String(counts.with_ref_answer || 0) },
    ];
    cards.forEach((item) => {
      const div = document.createElement("div");
      div.className = "mini-stat";
      div.innerHTML = `<div class="mini-stat-label">${item.label}</div><div class="mini-stat-value">${item.value}</div>`;
      wrap.appendChild(div);
    });
  }

  function appendResultNotice(text) {
    const wrap = $("resultSummaryMeta");
    if (!wrap || !String(text || "").trim()) return;
    const div = document.createElement("div");
    div.className = "mini-stat";
    div.innerHTML = `<div class="mini-stat-label">结果状态</div><div class="mini-stat-value" style="font-size:14px;font-weight:600;line-height:1.6">${String(text)}</div>`;
    wrap.appendChild(div);
  }

  function renderDetailUnavailable(message) {
    const tbody = $("itemsTbody");
    if (!tbody) return;
    tbody.innerHTML = `<tr><td colspan="7" class="muted">${String(message || "当前没有可展示的明细")}</td></tr>`;
  }

  async function doPreview(ev) {
    ev.preventDefault();
    const files = getSelectedInputFiles();
    if (!files.length) {
      clearPreviewState({ clearOutput: true });
      setPre("previewOutput", { error: "请先选择文件" });
      return;
    }
    const fd = new FormData();
    files.forEach((file) => fd.append("files", file));
    fd.append("input_format", $("inputFormat").value);
    fd.append("encoding", ($("encoding").value || "").trim());
    fd.append("delimiter", ($("delimiter").value || ",").trim());
    fd.append("sheet_name", ($("sheetName").value || "").trim());
    fd.append("sample_size", String(parseInt($("sampleSize").value || "5", 10)));
    const url = `${apiBase()}/eval/preview`;
    $("previewOutput").textContent = "加载中…";
    try {
      const previewRanges =
        previewFileSignature && previewFileSignature === buildFileSignature(files) ? collectFileRanges() : [];
      if (previewRanges.length) {
        fd.append("file_ranges_json", JSON.stringify(previewRanges));
      }
      const data = await fetchJson(url, { method: "POST", body: fd });
      lastPreviewData = data && typeof data === "object" ? data : null;
      previewFileSignature = buildFileSignature(files);
      setPre("previewOutput", data);
      renderFileRangeTable(Array.isArray(data.files) ? data.files : []);
      updatePreviewOverview();

      const cols = Array.isArray(data.shared_columns) ? data.shared_columns : Array.isArray(data.columns) ? data.columns : [];
      populateSelect($("questionField"), cols, { allowEmpty: false });
      populateSelect($("answerField"), cols, { allowEmpty: false });
      populateSelect($("contextField"), cols, { allowEmpty: false });
      populateSelect($("refAnswerField"), cols, { allowEmpty: true });
      populateSelect($("idField"), cols, { allowEmpty: true });
      populateSelect($("originalFilenameField"), cols, { allowEmpty: true });

      const { page } = getUiCachePage();
      if (page.questionField) $("questionField").value = String(page.questionField);
      else guessSelect($("questionField"), ["question", "问题", "题干", "q"]);
      if (page.answerField) $("answerField").value = String(page.answerField);
      else guessSelect($("answerField"), ["answer", "答案", "a"]);
      if (page.contextField) $("contextField").value = String(page.contextField);
      else guessSelect($("contextField"), ["context", "来源", "证据", "文本", "source_fact_text"]);
      if (page.refAnswerField) $("refAnswerField").value = String(page.refAnswerField);
      else guessSelect($("refAnswerField"), ["ref_answer", "reference", "参考答案", "标准答案"]);
      if (page.idField) $("idField").value = String(page.idField);
      else guessSelect($("idField"), ["id", "ID"]);
      if (page.originalFilenameField) $("originalFilenameField").value = String(page.originalFilenameField);
      else guessSelect($("originalFilenameField"), ["original_filename", "文件名", "filename"]);

      ["questionField", "answerField", "contextField", "refAnswerField", "idField", "originalFilenameField"].forEach((id) => {
        const el = $(id);
        if (el) persistUiField(el);
      });
    } catch (e) {
      clearPreviewState();
      setPre("previewOutput", { error: String(e.message || e) });
    }
  }

  async function pollJob() {
    if (!currentJobId) return;
    const url = `${apiBase()}/eval/jobs/${currentJobId}`;
    try {
      const job = await fetchJson(url, { method: "GET" });
      await applyJobSnapshot(job, { autoLoadPage: true });
    } catch (e) {
      $("jobStatus").textContent = `poll error: ${String(e.message || e)}`;
    }
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function beginPolling(jobId) {
    const normalized = String(jobId || "").trim();
    if (!normalized) return;
    stopPolling();
    setJobSelection(normalized, { active: true });
    pollTimer = setInterval(pollJob, 1200);
  }

  function clearResultPanels() {
    const scored = $("dlScored");
    const summary = $("dlSummary");
    if (scored) {
      scored.href = "#";
      scored.style.display = "none";
    }
    if (summary) {
      summary.href = "#";
      summary.style.display = "none";
    }
    const grid = $("scoreGrid");
    if (grid) grid.innerHTML = "";
    const resultMeta = $("resultSummaryMeta");
    if (resultMeta) resultMeta.innerHTML = "";
    const tbody = $("itemsTbody");
    if (tbody) tbody.innerHTML = "";
  }

  async function applyJobSnapshot(job, { autoLoadPage = false } = {}) {
    if (!job || typeof job !== "object") return;
    currentJobSnapshot = job;
    setPre("jobStatus", job);
    setJobSelection(job.job_id || "", { active: !isTerminalJobStatus(job.status) });
    rememberJob(job);
    renderRecentJobs();

    const st = String(job.status || "");
    if (isTerminalJobStatus(st)) {
      stopPolling();
      if (st === "succeeded" && job.result) {
        $("jobHint").textContent = `当前任务：job_id=${String(job.job_id || "")}（已完成，正在加载结果）`;
        await onJobDone(job.result);
        if (autoLoadPage && (isMilvusResult(job.result) || hasLocalEvalArtifacts(job.result))) {
          await loadPage();
        } else if (autoLoadPage && hasExpiredEvalArtifacts(job.result)) {
          renderDetailUnavailable("该评测任务的临时 scored / summary 文件已被清理；如果之前已入库，请到管理页查询。");
        }
        $("jobHint").textContent = `当前任务：job_id=${String(job.job_id || "")}（已完成，结果已加载）`;
      } else {
        const msg = String(job.message || "").trim();
        $("jobHint").textContent = `当前任务：job_id=${String(job.job_id || "")}（${getJobStatusLabel(st)}${msg ? `，${msg}` : ""}）`;
      }
      return;
    }

    $("jobHint").textContent = `当前任务：job_id=${String(job.job_id || "")}（${getJobStatusLabel(st)}）`;
  }

  async function onJobDone(result) {
    const files = result.files || {};
    const scored = files.scored_jsonl;
    const summary = files.summary_json;
    if (scored) {
      const parts = String(scored).split("/");
      const fileName = parts[parts.length - 1];
      $("dlScored").href = `${apiBase()}/download/outputs/${fileName}`;
      $("dlScored").style.display = "inline-block";
    } else {
      $("dlScored").style.display = "none";
    }
    if (summary) {
      const parts = String(summary).split("/");
      const fileName = parts[parts.length - 1];
      $("dlSummary").href = `${apiBase()}/download/outputs/${fileName}`;
      $("dlSummary").style.display = "inline-block";
    } else {
      $("dlSummary").style.display = "none";
    }

    const scores = result.unsupervised && result.unsupervised.scores;
    renderScores(scores);
    renderResultSummaryMeta(result);
    if (isMilvusResult(result)) {
      appendResultNotice(`结果已入库，详情请到管理页按 task_id 查询；当前 task_id=${result.milvus_task_id}`);
      renderDetailUnavailable("该评测结果已入库；明细浏览建议去管理页查询，当前页只保留概览信息。");
      $("ingestHint").textContent = `结果已入库，当前历史视图改走 Milvus。task_id=${result.milvus_task_id}`;
    } else if (hasExpiredEvalArtifacts(result)) {
      appendResultNotice("临时 scored / summary 文件已过期清理，当前不能再下载或加载明细。");
      renderDetailUnavailable("临时 scored / summary 文件已被清理；如果没有提前入库，则这批结果已无法恢复。");
      $("ingestHint").textContent = "当前评测结果的临时文件已被清理；未入库结果不能再从这里恢复。";
    } else {
      $("ingestHint").textContent = "当前结果仍通过临时 scored/summary 文件支撑看板；成功入库后会切换到 Milvus 历史视图。";
    }
  }

  async function loadMilvusPage(taskId, pageNum, pageSize) {
    const params = new URLSearchParams();
    params.set("only_filtered", "false");
    params.set("min_avg_score", "0");
    params.set("page", String(pageNum));
    params.set("page_size", String(pageSize));
    params.set("include_raw_responses", "false");
    return fetchJson(`${apiBase()}/task-qa/${encodeURIComponent(taskId)}?${params.toString()}`, { method: "GET" });
  }

  async function loadJobById(jobId, { resumePolling = true, autoLoadPage = true, silent = false } = {}) {
    const normalized = String(jobId || "").trim();
    if (!normalized) {
      if (!silent) $("jobHint").textContent = "请先输入 job_id";
      return;
    }
    if (!resumePolling) stopPolling();
    try {
      if (!silent) $("jobHint").textContent = `正在加载 job_id=${normalized}…`;
      setJobSelection(normalized, { active: null });
      const job = await fetchJson(`${apiBase()}/eval/jobs/${normalized}`, { method: "GET" });
      await applyJobSnapshot(job, { autoLoadPage });
      if (!isTerminalJobStatus(job.status) && resumePolling) {
        beginPolling(normalized);
        await pollJob();
      }
    } catch (e) {
      if (!silent) $("jobHint").textContent = `load failed: ${String(e.message || e)}`;
      throw e;
    }
  }

  async function refreshRecentJobs({ silent = false } = {}) {
    try {
      const data = await fetchJson(`${apiBase()}/eval/jobs?limit=20`, { method: "GET" });
      const jobs = replaceRecentJobs(Array.isArray(data.jobs) ? data.jobs : []);
      renderRecentJobs();
      if (!silent && jobs.length) {
        $("jobHint").textContent = `已刷新最近任务，共 ${jobs.length} 条`;
      }
      return jobs;
    } catch (e) {
      renderRecentJobs();
      if (!silent) {
        $("jobHint").textContent = `refresh failed: ${String(e.message || e)}`;
      }
      return getRecentJobs();
    }
  }

  async function deleteEvalJobHistory(jobId) {
    const normalized = String(jobId || "").trim();
    if (!normalized) {
      $("jobHint").textContent = "缺少 job_id";
      return;
    }
    $("jobHint").textContent = `正在删除 job_id=${normalized}…`;
    try {
      await fetchJson(`${apiBase()}/eval/jobs/${normalized}/history`, { method: "DELETE" });
      removeRecentJob(normalized);
      clearSelectedEvalJob(normalized);
      await refreshRecentJobs({ silent: true });
      $("jobHint").textContent = `已删除 job_id=${normalized}`;
    } catch (e) {
      $("jobHint").textContent = `删除失败：${String(e.message || e)}`;
      throw e;
    }
  }

  async function startJob() {
    const files = getSelectedInputFiles();
    if (!files.length) {
      $("jobHint").textContent = "请先选择文件";
      return;
    }
    const currentSignature = buildFileSignature(files);
    if (!lastPreviewData || previewFileSignature !== currentSignature) {
      $("jobHint").textContent = "请先点击“预览”，确认当前文件和字段映射";
      return;
    }
    if (!lastPreviewData.schema_consistent) {
      $("jobHint").textContent = "当前批量文件字段结构不一致，不能启动评测任务";
      return;
    }
    const ds = ($("datasetName").value || "").trim();
    if (!ds) {
      $("jobHint").textContent = "dataset_name 不能为空";
      return;
    }
    const qf = $("questionField").value;
    const af = $("answerField").value;
    const cf = $("contextField").value;
    if (!qf || !af || !cf) {
      $("jobHint").textContent = "question/answer/context 映射不能为空";
      return;
    }

    const fd = new FormData();
    files.forEach((file) => fd.append("files", file));
    fd.append("dataset_name", ds);
    fd.append("question_field", qf);
    fd.append("answer_field", af);
    fd.append("context_field", cf);
    const raf = $("refAnswerField").value;
    const idf = $("idField").value;
    const fnf = $("originalFilenameField").value;
    if (raf) fd.append("ref_answer_field", raf);
    if (idf) fd.append("id_field", idf);
    if (fnf) fd.append("original_filename_field", fnf);

    fd.append("input_format", $("inputFormat").value);
    const enc = ($("encoding").value || "").trim();
    if (enc) fd.append("encoding", enc);
    fd.append("delimiter", ($("delimiter").value || ",").trim());
    const sheet = ($("sheetName").value || "").trim();
    if (sheet) fd.append("sheet_name", sheet);
    const ub = ($("unsupBatchSize") && $("unsupBatchSize").value ? String($("unsupBatchSize").value).trim() : "");
    if (ub) fd.append("unsupervised_batch_size", ub);

    const fm = ($("faithNliModel") && $("faithNliModel").value ? String($("faithNliModel").value).trim() : "");
    if (fm) fd.append("faithfulness_nli_model", fm);

    const qm = ($("answerabilityQaModel") && $("answerabilityQaModel").value
      ? String($("answerabilityQaModel").value).trim()
      : "");
    if (qm) fd.append("answerability_qa_model", qm);

    const cm = ($("coverageEmbeddingModel") && $("coverageEmbeddingModel").value
      ? String($("coverageEmbeddingModel").value).trim()
      : "");
    if (cm) fd.append("coverage_embedding_model", cm);

    const hm = ($("hypMode").value || "").trim();
    const ht = ($("hypTimeout").value || "").trim();
    const hr = ($("hypRetries").value || "").trim();
    const hc = ($("hypConc").value || "").trim();
    if (hm) fd.append("faithfulness_hypothesis_mode", hm);
    if (ht) fd.append("faithfulness_hypothesis_timeout", ht);
    if (hr) fd.append("faithfulness_hypothesis_max_retries", hr);
    if (hc) fd.append("faithfulness_hypothesis_max_concurrency", hc);

    $("jobHint").textContent = "启动中…";
    $("jobStatus").textContent = "";
    clearResultPanels();
    try {
      const fileRanges = collectFileRanges();
      if (fileRanges.length) fd.append("file_ranges_json", JSON.stringify(fileRanges));
      const data = await fetchJson(`${apiBase()}/eval/jobs`, { method: "POST", body: fd });
      const placeholder = {
        job_id: data.job_id,
        status: "queued",
        message: "queued",
        created_at: Math.floor(Date.now() / 1000),
        params: { dataset_name: ds, input_files_count: files.length },
      };
      rememberJob(placeholder);
      renderRecentJobs();
      setJobSelection(data.job_id, { active: true });
      $("jobHint").textContent = `job_id=${data.job_id}`;
      beginPolling(data.job_id);
      await pollJob();
    } catch (e) {
      $("jobHint").textContent = `start failed: ${String(e.message || e)}`;
    }
  }

  async function cancelJob() {
    const requestedId = String($("jobIdInput")?.value || "").trim() || currentJobId;
    if (!requestedId) {
      $("jobHint").textContent = "当前没有 job_id";
      return;
    }
    try {
      await fetchJson(`${apiBase()}/eval/jobs/${requestedId}`, { method: "DELETE" });
      rememberJob({
        job_id: requestedId,
        status: "canceled",
        message: "canceled by user",
        finished_at: Math.floor(Date.now() / 1000),
      });
      renderRecentJobs();
      $("jobHint").textContent = `canceled job_id=${requestedId}`;
      if (currentJobId === requestedId) {
        stopPolling();
      }
      await loadJobById(requestedId, { resumePolling: false, autoLoadPage: false, silent: true }).catch(() => {});
    } catch (e) {
      const message = String(e.message || e);
      const extra =
        message.includes("不存在") || message.includes("已结束") || message.includes("已完成")
          ? "；如果只是清理历史记录，可直接点任务列表里的“删除记录”"
          : "";
      $("jobHint").textContent = `cancel failed: ${message}${extra}`;
    }
  }

  function fmtScore(v) {
    return typeof v === "number" ? v.toFixed(4) : "0.0000";
  }

  function parseRangeSuffix(name) {
    const raw = String(name || "").trim();
    if (!raw) return null;
    const m = raw.match(/^(.*?)(\d+)\s*-\s*(\d+)$/);
    if (!m) return null;
    return { base: String(m[1] || ""), start: String(m[2] || ""), end: String(m[3] || "") };
  }

  async function loadPage() {
    const requestedId = String($("jobIdInput")?.value || "").trim() || currentJobId;
    if (!requestedId) {
      $("ingestHint").textContent = "请先启动并完成评测任务";
      return;
    }
    if (requestedId !== currentJobId) {
      setJobSelection(requestedId, { active: null });
    }
    const pageSize = Math.max(1, Math.min(200, parseInt($("pageSize").value || "50", 10)));
    const pageNum = Math.max(1, parseInt($("pageNum").value || "1", 10));
    const offset = (pageNum - 1) * pageSize;
    const thr = parseFloat($("threshold").value || "0.7");
    const currentResult =
      currentJobSnapshot &&
      String(currentJobSnapshot.job_id || "").trim() === requestedId &&
      currentJobSnapshot.result &&
      typeof currentJobSnapshot.result === "object"
        ? currentJobSnapshot.result
        : null;
    if (currentResult && hasExpiredEvalArtifacts(currentResult)) {
      $("ingestHint").textContent = "当前评测结果的临时文件已被清理；不能再加载明细。";
      renderDetailUnavailable("临时 scored / summary 文件已被清理；如果没有提前入库，则这批结果已无法恢复。");
      return;
    }
    const url = `${apiBase()}/eval/jobs/${requestedId}/result?offset=${offset}&limit=${pageSize}&threshold=${thr}`;
    $("itemsTbody").innerHTML = `<tr><td colspan="7" class="muted">加载中…</td></tr>`;
    try {
      const data = await fetchJson(url, { method: "GET" });
      if (data && data.history_redirect && data.history_redirect.source === "milvus" && data.history_redirect.task_id) {
        $("ingestHint").textContent = `当前结果已切换到 Milvus 历史视图。task_id=${data.history_redirect.task_id}；如需完整筛选与治理，请到管理页查询。`;
        renderScores(data.summary && data.summary.unsupervised && data.summary.unsupervised.scores);
        renderResultSummaryMeta(data.summary);
        appendResultNotice(`该结果已入库，建议去管理页按 task_id=${data.history_redirect.task_id} 查询。`);
        renderDetailUnavailable("该评测结果已入库；当前页不再直接展开历史明细，请到管理页查询。");
        return;
      }
      const items = Array.isArray(data.items) ? data.items : [];
      const total = typeof data.total === "number" ? data.total : 0;
      $("ingestHint").textContent = `total=${total}，当前页=${pageNum}，page_size=${pageSize}（threshold=${thr} 仅用于展示 filtered，不会写回文件）`;
      renderScores(data.summary && data.summary.unsupervised && data.summary.unsupervised.scores);
      renderResultSummaryMeta(data.summary);
      renderItems(items);
    } catch (e) {
      $("itemsTbody").innerHTML = `<tr><td colspan="7" class="muted">error: ${String(e.message || e)}</td></tr>`;
    }
  }

  function renderItems(items) {
    const tbody = $("itemsTbody");
    tbody.innerHTML = "";
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="muted">empty</td></tr>`;
      return;
    }
    for (const it of items) {
      const ue = it.unsupervised_evaluation || {};
      const scores = (ue && ue.scores) || {};
      const faith = fmtScore(scores.faithfulness);
      const ans = fmtScore(scores.answerability);
      const cov = fmtScore(scores.coverage_score);
      const covSelf = fmtScore(scores.coverage_self);
      const rg = fmtScore(scores.coverage_recall_soft);
      const f1 = fmtScore(scores.unsupervised_f1);
      const filtered = it.filtered === true ? "true" : "false";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono">${it.id || ""}</td>
        <td class="mono">${it.group_id || ""}</td>
        <td>${it.original_filename || ""}</td>
        <td>${(it.question || "").slice(0, 180)}</td>
        <td class="mono">F=${faith} P=${ans}<br/>Cov=${cov} Self=${covSelf}<br/>Rg=${rg} F1=${f1}</td>
        <td class="mono">${filtered}</td>
        <td></td>
      `;

      const actions = document.createElement("div");
      actions.className = "actions-row";

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "secondary";
      btn.textContent = "解释";
      btn.addEventListener("click", () => openExplainDrawer(it));
      actions.appendChild(btn);

      const details = document.createElement("details");
      const sum = document.createElement("summary");
      sum.textContent = "JSON";
      details.appendChild(sum);
      const pre = document.createElement("pre");
      pre.style.maxHeight = "220px";
      pre.textContent = JSON.stringify(
        {
          id: it.id,
          group_id: it.group_id,
          original_filename: it.original_filename,
          question: it.question,
          answer: it.answer,
          context: it.context,
          unsupervised_evaluation: it.unsupervised_evaluation,
        },
        null,
        2
      );
      details.appendChild(pre);
      actions.appendChild(details);

      tr.children[6].appendChild(actions);
      tbody.appendChild(tr);
    }
  }

  // ---------- 评分解释抽屉 ----------

  let explainLastFocus = null;

  function getUi() {
    return window.apiuseUi && typeof window.apiuseUi === "object" ? window.apiuseUi : {};
  }

  function setExplainOpen(isOpen) {
    const overlay = $("explainOverlay");
    const drawer = $("explainDrawer");
    if (!overlay || !drawer) return;

    if (isOpen) {
      explainLastFocus = document.activeElement;
      overlay.hidden = false;
      overlay.classList.add("is-open");
      drawer.classList.add("is-open");
      drawer.setAttribute("aria-hidden", "false");
      document.body.classList.add("drawer-open");
      const closeBtn = $("explainClose");
      if (closeBtn) closeBtn.focus();
      return;
    }

    overlay.classList.remove("is-open");
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    document.body.classList.remove("drawer-open");
    window.setTimeout(() => {
      overlay.hidden = true;
    }, 220);
    try {
      if (explainLastFocus && typeof explainLastFocus.focus === "function") explainLastFocus.focus();
    } catch {
      // ignore
    }
    explainLastFocus = null;
  }

  function trapDrawerTab(e) {
    if (e.key !== "Tab") return;
    const drawer = $("explainDrawer");
    if (!drawer) return;
    const focusables = Array.from(
      drawer.querySelectorAll(
        'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )
    ).filter((x) => x && x.offsetParent !== null);
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (e.shiftKey) {
      if (active === first || !drawer.contains(active)) {
        e.preventDefault();
        last.focus();
      }
      return;
    }
    if (active === last) {
      e.preventDefault();
      first.focus();
    }
  }

  function openExplainDrawer(item) {
    const title = $("explainTitle");
    const body = $("explainBody");
    if (!body) return;
    if (title) title.textContent = `评分解释 · ${String(item?.id || "").slice(0, 24) || "item"}`;
    body.innerHTML = "";

    const apiuseUi = getUi();
    if (apiuseUi && typeof apiuseUi.renderUnsupervisedExplain === "function") {
      body.appendChild(apiuseUi.renderUnsupervisedExplain(item, { includeQa: true, includeRaw: true }));
    } else {
      const pre = document.createElement("pre");
      pre.textContent = JSON.stringify(item, null, 2);
      body.appendChild(pre);
    }

    setExplainOpen(true);
  }

  async function hydrateEvalRuntime() {
    renderRecentJobs();
    const recentJobs = await refreshRecentJobs({ silent: true });

    const page = getRuntimePage();
    const rememberedId =
      String(page.activeJobId || "").trim() ||
      String(page.selectedJobId || "").trim() ||
      String($("jobIdInput")?.value || "").trim();
    const recentList = Array.isArray(recentJobs) ? recentJobs : [];
    const existingIds = new Set(recentList.map((job) => String(job.job_id || "").trim()).filter(Boolean));
    const targetId = rememberedId && existingIds.has(rememberedId)
      ? rememberedId
      : (recentList[0] && String(recentList[0].job_id || "").trim()) || "";
    if (!targetId) {
      setJobSelection("", { active: false });
      return;
    }
    try {
      await loadJobById(targetId, { resumePolling: true, autoLoadPage: true, silent: true });
      $("jobHint").textContent = `已恢复 job_id=${targetId}`;
    } catch (e) {
      $("jobHint").textContent = `恢复最近任务失败：${String(e.message || e)}`;
    }
  }

  async function ingest() {
    const requestedId = String($("jobIdInput")?.value || "").trim() || currentJobId;
    if (!requestedId) {
      $("ingestHint").textContent = "请先完成评测任务";
      return;
    }
    const ds = ($("datasetName").value || "").trim();
    if (!ds) {
      $("ingestHint").textContent = "dataset_name 不能为空";
      return;
    }
    const thr = parseFloat($("threshold").value || "0.7");
    const enableVector = $("enableVector").value === "true";
    $("ingestHint").textContent = "入库中…";
    try {
      const res = await fetchJson(`${apiBase()}/eval/jobs/${requestedId}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_name: ds,
          threshold: thr,
          enable_vector_storage: enableVector,
        }),
      });
      $("ingestHint").textContent = `入库完成：task_id=${res.milvus_task_id || res.task_id}，selected=${res.selected}，milvus=${JSON.stringify(res.milvus)}`;
      await loadJobById(requestedId, { resumePolling: false, autoLoadPage: true, silent: true });
    } catch (e) {
      $("ingestHint").textContent = `ingest failed: ${String(e.message || e)}`;
    }
  }

  function closestNodeFor(id, selector) {
    const node = $(id);
    return node && selector ? node.closest(selector) : node;
  }

  function createEvalSettingsModal() {
    if ($("evalSettingsModal")) return $("evalSettingsModal");
    const overlay = document.createElement("div");
    overlay.id = "evalSettingsOverlay";
    overlay.className = "drawer-overlay";
    overlay.hidden = true;

    const modal = document.createElement("aside");
    modal.id = "evalSettingsModal";
    modal.className = "settings-modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", "evalSettingsTitle");
    modal.setAttribute("aria-hidden", "true");
    modal.hidden = true;
    modal.innerHTML = [
      '<div class="settings-modal-header">',
      '<div>',
      '<h2 class="settings-modal-title" id="evalSettingsTitle">评测设置</h2>',
      '<p class="settings-modal-desc">配置解析方式、字段映射和任务参数。保存后仍按原接口字段提交。</p>',
      '</div>',
      '<button type="button" class="icon-btn settings-modal-close" id="evalSettingsClose" aria-label="关闭设置" title="关闭"><span aria-hidden="true">×</span></button>',
      '</div>',
      '<div class="settings-modal-body">',
      '<div class="settings-tab-layout">',
      '<div class="settings-tab-nav" role="tablist" aria-label="评测设置分组">',
      '<button type="button" class="settings-tab-btn is-active" data-eval-tab="parse" role="tab" aria-selected="true">解析</button>',
      '<button type="button" class="settings-tab-btn" data-eval-tab="mapping" role="tab" aria-selected="false">字段映射</button>',
      '<button type="button" class="settings-tab-btn" data-eval-tab="runtime" role="tab" aria-selected="false">任务参数</button>',
      '</div>',
      '<div class="settings-tab-panels">',
      '<div class="settings-tab-panel is-active" data-eval-panel="parse" role="tabpanel"></div>',
      '<div class="settings-tab-panel" data-eval-panel="mapping" role="tabpanel"></div>',
      '<div class="settings-tab-panel" data-eval-panel="runtime" role="tabpanel"></div>',
      '</div>',
      '</div>',
      '</div>',
      '<div class="settings-modal-footer">',
      '<button type="button" class="secondary" id="evalSettingsCancel">取消</button>',
      '<button type="button" id="evalSettingsSave">保存</button>',
      '</div>',
    ].join("");
    document.body.append(overlay, modal);

    const close = () => {
      overlay.classList.remove("is-open");
      modal.classList.remove("is-open");
      modal.setAttribute("aria-hidden", "true");
      document.body.classList.remove("drawer-open");
      window.setTimeout(() => {
        overlay.hidden = true;
        modal.hidden = true;
      }, 180);
    };
    const open = () => {
      overlay.hidden = false;
      modal.hidden = false;
      window.requestAnimationFrame(() => {
        overlay.classList.add("is-open");
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");
        document.body.classList.add("drawer-open");
      });
    };
    modal.openEvalSettings = open;
    modal.closeEvalSettings = close;
    overlay.addEventListener("click", close);
    $("evalSettingsClose")?.addEventListener("click", close);
    $("evalSettingsCancel")?.addEventListener("click", close);
    $("evalSettingsSave")?.addEventListener("click", close);
    modal.querySelectorAll("[data-eval-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-eval-tab") || "";
        modal.querySelectorAll("[data-eval-tab]").forEach((item) => {
          item.classList.toggle("is-active", item === btn);
          item.setAttribute("aria-selected", item === btn ? "true" : "false");
        });
        modal.querySelectorAll("[data-eval-panel]").forEach((panel) => {
          panel.classList.toggle("is-active", panel.getAttribute("data-eval-panel") === key);
        });
      });
    });
    return modal;
  }

  function setupEvalWorkbench() {
    const modal = createEvalSettingsModal();
    const previewForm = $("previewForm");
    const uploadSection = previewForm?.closest("section.card");
    const mappingSection = $("datasetName")?.closest("section.card");
    if (!modal || !previewForm || !uploadSection || !mappingSection) return;

    const parsePanel = modal.querySelector('[data-eval-panel="parse"]');
    const mappingPanel = modal.querySelector('[data-eval-panel="mapping"]');
    const runtimePanel = modal.querySelector('[data-eval-panel="runtime"]');

    ["inputFormat", "encoding", "delimiter", "sheetName", "sampleSize"].forEach((id) => {
      const node = closestNodeFor(id, "label");
      if (node) parsePanel.appendChild(node);
    });

    const mappingPanelSource = mappingSection.querySelector(".eval-panel");
    if (mappingPanelSource) mappingPanel.appendChild(mappingPanelSource);
    const runtimePanelSource = Array.from(mappingSection.querySelectorAll(".eval-panel")).find((section) => section.querySelector("#unsupBatchSize"));
    if (runtimePanelSource) runtimePanel.appendChild(runtimePanelSource);

    const startActions = $("btnStartJob")?.closest(".actions-row");
    if (startActions) {
      startActions.classList.add("eval-run-actions");
      uploadSection.appendChild(startActions);
    }

    const settingsBar = document.createElement("div");
    settingsBar.className = "eval-settings-bar";
    settingsBar.innerHTML = [
      '<div>',
      '<strong>评测配置</strong>',
      '<span>解析方式、字段映射和任务参数已收进设置弹窗。</span>',
      '</div>',
      '<button type="button" class="secondary" id="btnOpenEvalSettings">评测设置</button>',
    ].join("");
    previewForm.insertAdjacentElement("afterend", settingsBar);
    $("btnOpenEvalSettings")?.addEventListener("click", () => modal.openEvalSettings?.());

    const heading = uploadSection.querySelector("h2");
    if (heading) heading.textContent = "上传与预览";
    uploadSection.classList.add("eval-upload-section");
    previewForm.classList.add("eval-upload-form");
    const hint = uploadSection.querySelector(":scope > .hint, :scope > p");
    if (hint) hint.textContent = "选择数据集文件，先预览列名，再在评测设置里完成字段映射。";
  }

  $("previewForm").addEventListener("submit", doPreview);
  $("btnStartJob").addEventListener("click", startJob);
  $("btnCancelJob").addEventListener("click", cancelJob);
  $("btnLoadJob")?.addEventListener("click", () => loadJobById($("jobIdInput")?.value || "", { resumePolling: true, autoLoadPage: true }));
  $("btnRefreshJobs")?.addEventListener("click", () => refreshRecentJobs());
  $("inputFile")?.addEventListener("change", () => clearPreviewState({ clearOutput: true }));
  $("jobIdInput")?.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    loadJobById($("jobIdInput")?.value || "", { resumePolling: true, autoLoadPage: true }).catch(() => {});
  });
  $("btnLoadPage").addEventListener("click", loadPage);
  $("btnIngest").addEventListener("click", ingest);

  // Drawer events
  $("explainOverlay")?.addEventListener("click", () => setExplainOpen(false));
  $("explainClose")?.addEventListener("click", () => setExplainOpen(false));
  $("explainDrawer")?.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setExplainOpen(false);
    trapDrawerTab(e);
  });

  restoreUiCache();
  initApiBaseUrl();
  bindUiCache();
  setupEvalWorkbench();
  hydrateEvalRuntime().catch(() => {});
})();
