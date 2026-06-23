const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const ui = () =>
  window.apiuseUi && typeof window.apiuseUi === 'object' ? window.apiuseUi : {};

const notify = (message, type = 'info') => {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.notify === 'function') {
    apiuseUi.notify(message, type);
    return;
  }
  const text = String(message ?? '').trim();
  if (text) alert(text);
};

function setBtnLoading(btn, isLoading) {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.setButtonLoading === 'function') {
    apiuseUi.setButtonLoading(btn, !!isLoading);
    return;
  }
  if (btn) btn.disabled = !!isLoading;
}

let lastTaskId = null;
let lastCsvPath = null;
let lastPipelineOutputs = [];
let lastTaskQaItems = [];
let lastTaskQaFiles = [];
let currentTaskPoller = null;
let llmConfigStore = { active: '', profiles: {} };
let ocrConfigStore = { active: '', profiles: {} };

const RUNTIME_STATE_KEY = 'apiuse_runtime_state_v1';
const RUNTIME_PAGE = 'index';

function loadRuntimeState() {
  try {
    const raw = localStorage.getItem(RUNTIME_STATE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === 'object' ? parsed : { pages: {} };
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
  if (!state.pages || typeof state.pages !== 'object') state.pages = {};
  const page = state.pages[RUNTIME_PAGE] && typeof state.pages[RUNTIME_PAGE] === 'object' ? state.pages[RUNTIME_PAGE] : {};
  mutator(page);
  state.pages[RUNTIME_PAGE] = page;
  saveRuntimeState(state);
  return page;
}

function getRuntimePageState() {
  const state = loadRuntimeState();
  const pages = state.pages && typeof state.pages === 'object' ? state.pages : {};
  return pages[RUNTIME_PAGE] && typeof pages[RUNTIME_PAGE] === 'object' ? pages[RUNTIME_PAGE] : {};
}

function fmtRuntimeTime(ts) {
  if (ts === null || ts === undefined || ts === '') return '';
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
    return String(ts || '');
  }
}

function isTaskTerminal(status) {
  const st = String(status || '').toLowerCase();
  return ['completed', 'failed', 'canceled'].includes(st);
}

function normalizeTaskRecord(task) {
  if (!task || typeof task !== 'object') return null;
  const taskId = String(task.task_id || '').trim();
  if (!taskId) return null;
  const outputs = Array.isArray(task.outputs) ? task.outputs : [];
  const availableOutputs = outputs.filter((output) => {
    if (!output || typeof output !== 'object') return false;
    if (String(output.history_source || '').trim().toLowerCase() === 'milvus') return true;
    if (output.consolidated_json || output.consolidated_csv) return true;
    return false;
  });
  return {
    task_id: taskId,
    status: String(task.status || ''),
    message: String(task.message || ''),
    updated_at: task.updated_at ?? task.finished_at ?? task.created_at ?? '',
    outputs_count: outputs.length,
    available_outputs_count: availableOutputs.length,
    history_source: String(task.history_source || '').trim().toLowerCase(),
    artifacts_deleted: Boolean(task.artifacts_deleted),
  };
}

function getRecentTasks() {
  const page = getRuntimePageState();
  return Array.isArray(page.recentTasks) ? page.recentTasks : [];
}

function replaceRecentTasks(tasks) {
  const normalized = Array.isArray(tasks)
    ? tasks
        .map((task) => normalizeTaskRecord(task))
        .filter((task) => !!task)
        .slice(0, 12)
    : [];
  mutateRuntimePage((page) => {
    const ids = new Set(normalized.map((task) => String(task.task_id || '')));
    page.recentTasks = normalized;
    if (page.selectedTaskId && !ids.has(String(page.selectedTaskId || ''))) {
      page.selectedTaskId = normalized[0] ? normalized[0].task_id : '';
    }
    if (page.activeTaskId && !ids.has(String(page.activeTaskId || ''))) {
      page.activeTaskId = '';
    }
  });
  return normalized;
}

function removeRecentTask(taskId) {
  const normalized = String(taskId || '').trim();
  if (!normalized) return getRecentTasks();
  return mutateRuntimePage((page) => {
    const prev = Array.isArray(page.recentTasks) ? page.recentTasks : [];
    page.recentTasks = prev.filter((it) => String(it.task_id || '') !== normalized);
    if (String(page.selectedTaskId || '') === normalized) {
      page.selectedTaskId = page.recentTasks[0] ? String(page.recentTasks[0].task_id || '') : '';
    }
    if (String(page.activeTaskId || '') === normalized) {
      page.activeTaskId = '';
    }
  }).recentTasks || [];
}

function clearSelectedPipelineTask(taskId) {
  const normalized = String(taskId || '').trim();
  if (!normalized) return;
  if (String(lastTaskId || '').trim() === normalized) {
    lastTaskId = null;
  }
  const input = $('#taskIdInput');
  if (input && String(input.value || '').trim() === normalized) {
    input.value = '';
    persistUiField(input);
  }
  mutateRuntimePage((page) => {
    if (String(page.selectedTaskId || '').trim() === normalized) page.selectedTaskId = '';
    if (String(page.activeTaskId || '').trim() === normalized) page.activeTaskId = '';
  });
  clearPipelineOutputsView();
  const statusEl = $('#pipelineStatus');
  if (statusEl) statusEl.textContent = '';
  updatePipelineTaskHint();
}

function rememberTask(task) {
  const record = normalizeTaskRecord(task);
  if (!record) return getRecentTasks();
  return mutateRuntimePage((page) => {
    const prev = Array.isArray(page.recentTasks) ? page.recentTasks : [];
    const merged = [record, ...prev.filter((it) => String(it.task_id || '') !== record.task_id)];
    page.recentTasks = merged.slice(0, 12);
    page.selectedTaskId = record.task_id;
    page.activeTaskId = isTaskTerminal(record.status) ? (page.activeTaskId === record.task_id ? '' : page.activeTaskId || '') : record.task_id;
  }).recentTasks || [];
}

function setTaskSelection(taskId, { active = null } = {}) {
  const normalized = String(taskId || '').trim();
  if (normalized) lastTaskId = normalized;

  const taskInput = $('#taskIdInput');
  if (taskInput && taskInput.value !== normalized) {
    taskInput.value = normalized;
    persistUiField(taskInput);
  }
  const chunkTaskInput = $('#chunkTaskId');
  if (chunkTaskInput && !chunkTaskInput.value && normalized) {
    chunkTaskInput.value = normalized;
    persistUiField(chunkTaskInput);
  }

  mutateRuntimePage((page) => {
    page.selectedTaskId = normalized;
    if (active === true) page.activeTaskId = normalized;
    else if (active === false && page.activeTaskId === normalized) page.activeTaskId = '';
  });
}

function renderPipelineTaskHistory() {
  const wrap = $('#pipelineTaskHistory');
  if (!wrap) return;
  const tasks = getRecentTasks();
  wrap.innerHTML = '';
  if (!tasks.length) {
    const empty = document.createElement('div');
    empty.className = 'task-history-empty';
    empty.textContent = '暂无最近流水线任务。执行一次流水线或输入 task_id 查询后，这里会保留最近记录。';
    wrap.appendChild(empty);
    return;
  }

  tasks.forEach((task) => {
    const row = document.createElement('div');
    row.className = 'task-history-item';

    const main = document.createElement('div');
    main.className = 'task-history-main';

    const title = document.createElement('div');
    title.className = 'task-history-title';
    const pill = document.createElement('span');
    pill.className = 'status-pill';
    pill.dataset.status = String(task.status || '').toLowerCase();
    pill.textContent = String(task.status || 'unknown');
    title.appendChild(pill);

    const name = document.createElement('span');
    name.className = 'task-history-name';
    if (task.history_source === 'milvus') {
      name.textContent = task.outputs_count > 0 ? `已入库（${task.outputs_count} 个结果）` : '已入库任务';
    } else if (task.artifacts_deleted && task.outputs_count > 0) {
      name.textContent = '输出文件已清理';
    } else if (task.outputs_count > 0) {
      name.textContent = `已有 ${task.outputs_count} 个输出文件`;
    } else {
      name.textContent = '流水线任务';
    }
    title.appendChild(name);

    const id = document.createElement('span');
    id.className = 'task-history-id';
    id.textContent = task.task_id;
    title.appendChild(id);
    main.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'task-history-meta';
    const timeText = fmtRuntimeTime(task.updated_at);
    if (timeText) {
      const span = document.createElement('span');
      span.textContent = `更新时间：${timeText}`;
      meta.appendChild(span);
    }
    if (task.message) {
      const span = document.createElement('span');
      span.textContent = `状态说明：${task.message}`;
      meta.appendChild(span);
    }
    if (task.history_source === 'milvus') {
      const span = document.createElement('span');
      span.textContent = '结果已入库，详情请到管理页查询或在当前任务详情中查看入库记录。';
      meta.appendChild(span);
    } else if (task.artifacts_deleted && task.outputs_count > 0) {
      const span = document.createElement('span');
      span.textContent = '临时 JSON/CSV 已被清理，不能再直接下载。';
      meta.appendChild(span);
    }
    main.appendChild(meta);
    row.appendChild(main);

    const actions = document.createElement('div');
    actions.className = 'task-history-actions';

    const viewBtn = document.createElement('button');
    viewBtn.type = 'button';
    viewBtn.className = 'secondary';
    viewBtn.textContent = '查看';
    viewBtn.addEventListener('click', () => {
      loadTaskStatusById(task.task_id, { resumePolling: true, silent: false }).catch(() => {});
    });
    actions.appendChild(viewBtn);

    if (!isTaskTerminal(task.status)) {
      const stopBtn = document.createElement('button');
      stopBtn.type = 'button';
      stopBtn.className = 'danger';
      stopBtn.textContent = '终止';
      stopBtn.addEventListener('click', () => {
        const input = $('#taskIdInput');
        if (input) {
          input.value = task.task_id;
          persistUiField(input);
        }
        handleCancelTask(task.task_id);
      });
      actions.appendChild(stopBtn);
    }

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'danger';
    deleteBtn.textContent = '删除记录';
    deleteBtn.addEventListener('click', () => {
      deletePipelineHistory(task.task_id).catch(() => {});
    });
    actions.appendChild(deleteBtn);

    row.appendChild(actions);
    wrap.appendChild(row);
  });
}

// ---------- UI 缓存（跨页面保留输入状态） ----------

const UI_CACHE_KEY = 'apiuse_ui_cache_v1';
const UI_CACHE_PAGE = 'index';
const UI_CACHE_EXCLUDE_IDS = new Set([
  'cfgKey', // LLM API Key 不写入浏览器缓存（避免泄露）
]);

function restoreUiCache() {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.restoreUiCache === 'function') {
    apiuseUi.restoreUiCache({
      cacheKey: UI_CACHE_KEY,
      pageKey: UI_CACHE_PAGE,
      excludeIds: UI_CACHE_EXCLUDE_IDS,
      apiInputId: 'apiBaseUrl',
    });
  }
}

function persistUiField(el) {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.persistUiField === 'function') {
    apiuseUi.persistUiField(el, {
      cacheKey: UI_CACHE_KEY,
      pageKey: UI_CACHE_PAGE,
      excludeIds: UI_CACHE_EXCLUDE_IDS,
      apiInputId: 'apiBaseUrl',
    });
  }
}

function bindUiCache() {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.bindUiCache === 'function') {
    apiuseUi.bindUiCache({
      cacheKey: UI_CACHE_KEY,
      pageKey: UI_CACHE_PAGE,
      excludeIds: UI_CACHE_EXCLUDE_IDS,
      apiInputId: 'apiBaseUrl',
    });
  }
}

function getApiBaseUrl() {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.getApiBaseUrl === 'function') {
    return apiuseUi.getApiBaseUrl({ inputSelector: '#apiBaseUrl' });
  }
  return String(window.location.origin || '').replace(/\/+$/, '');
}

async function fetchJson(url, options) {
  const resp = await fetch(url, options);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = { raw: text };
  }
  if (!resp.ok) {
    const detail = data && (data.detail || data.message || data.error);
    throw new Error(detail || text || resp.statusText || `HTTP ${resp.status}`);
  }
  return data;
}

async function fetchTaskStatusMaybe(base, taskId) {
  const resp = await fetch(`${base}/task-status/${encodeURIComponent(taskId)}`);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = { raw: text };
  }
  if (resp.status === 404) {
    return { found: false, data };
  }
  if (!resp.ok) {
    const detail = data && (data.detail || data.message || data.error);
    throw new Error(detail || text || resp.statusText || `HTTP ${resp.status}`);
  }
  return { found: true, data };
}

async function refreshPipelineHistory({ silent = false } = {}) {
  const base = getApiBaseUrl();
  try {
    const data = await fetchJson(`${base}/pipeline/jobs?limit=20`);
    const tasks = replaceRecentTasks(Array.isArray(data?.jobs) ? data.jobs : []);
    renderPipelineTaskHistory();
    if (!silent) {
      updatePipelineTaskHint(`已刷新最近流水线任务，共 ${tasks.length} 条。`);
    }
    return tasks;
  } catch (err) {
    renderPipelineTaskHistory();
    if (!silent) {
      updatePipelineTaskHint(`刷新最近流水线任务失败：${String(err)}`);
    }
    return getRecentTasks();
  }
}

async function deletePipelineHistory(taskId) {
  const normalized = String(taskId || '').trim();
  if (!normalized) {
    notify('缺少 task_id', 'warning');
    return;
  }
  const base = getApiBaseUrl();
  const statusEl = $('#pipelineStatus');
  try {
    if (statusEl) statusEl.textContent = `正在删除历史记录：${normalized}…`;
    const res = await fetchJson(`${base}/pipeline/jobs/${encodeURIComponent(normalized)}`, {
      method: 'DELETE',
    });
    removeRecentTask(normalized);
    clearSelectedPipelineTask(normalized);
    await refreshPipelineHistory({ silent: true });
    if (statusEl) {
      statusEl.textContent = JSON.stringify(res, null, 2);
    }
    notify(`已删除流水线历史记录：${normalized}`, 'success');
  } catch (err) {
    if (statusEl) statusEl.textContent = '删除历史记录失败：' + String(err);
    notify(`删除流水线历史记录失败：${String(err)}`, 'error');
  }
}

function normalizeApiBaseUrl(value, fallbackOrigin) {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.normalizeApiBaseUrl === 'function') {
    return apiuseUi.normalizeApiBaseUrl(value, fallbackOrigin);
  }
  return String(fallbackOrigin || '').trim();
}

function initApiBaseUrl() {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.initApiBaseUrl === 'function') {
    apiuseUi.initApiBaseUrl({ inputSelector: '#apiBaseUrl' });
  }
}
