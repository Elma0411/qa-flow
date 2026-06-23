const $ = (sel) => document.querySelector(sel);

let currentItems = [];
let selectedIds = new Set();
let jobIds = [];
let jobCache = {};
let currentDetailId = null;

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

// ---------- UI 缓存（跨页面保留输入状态） ----------

const UI_CACHE_KEY = 'apiuse_ui_cache_v1';
const UI_CACHE_PAGE = 'admin';
const QUERY_MODE_CACHE_KEY = 'apiuse_admin_query_mode_v1';
const QUERY_MODE_LIST = 'list';
const QUERY_MODE_SEMANTIC = 'semantic';

let currentQueryMode = QUERY_MODE_LIST;

function restoreUiCache() {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.restoreUiCache === 'function') {
    apiuseUi.restoreUiCache({
      cacheKey: UI_CACHE_KEY,
      pageKey: UI_CACHE_PAGE,
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
      apiInputId: 'apiBaseUrl',
    });
  }
}

function normalizeQueryMode(value) {
  return value === QUERY_MODE_SEMANTIC ? QUERY_MODE_SEMANTIC : QUERY_MODE_LIST;
}

function getStoredQueryMode() {
  return normalizeQueryMode(window.localStorage.getItem(QUERY_MODE_CACHE_KEY));
}

function setStoredQueryMode(mode) {
  window.localStorage.setItem(QUERY_MODE_CACHE_KEY, normalizeQueryMode(mode));
}

function getApiBaseUrl() {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.getApiBaseUrl === 'function') {
    return apiuseUi.getApiBaseUrl({ inputSelector: '#apiBaseUrl' });
  }
  return String(window.location.origin || '').replace(/\/+$/, '');
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

restoreUiCache();
initApiBaseUrl();
bindUiCache();

function setStatus(text, isError = false) {
  const el = $('#statusText');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = isError ? '#b91c1c' : '';
}

function applyQueryModeUI(mode) {
  currentQueryMode = normalizeQueryMode(mode);
  const isListMode = currentQueryMode === QUERY_MODE_LIST;
  const listBtn = $('#btnLoadList');
  const semanticBtn = $('#btnSemanticSearch');
  if (listBtn) {
    listBtn.classList.toggle('secondary', !isListMode);
    listBtn.setAttribute('aria-pressed', String(isListMode));
  }
  if (semanticBtn) {
    semanticBtn.classList.toggle('secondary', isListMode);
    semanticBtn.setAttribute('aria-pressed', String(!isListMode));
  }
  document.querySelectorAll('[data-query-mode-panel]').forEach((el) => {
    const targetMode = el.getAttribute('data-query-mode-panel');
    el.hidden = targetMode !== currentQueryMode;
  });
}

function setQueryMode(mode, { persist = true } = {}) {
  applyQueryModeUI(mode);
  if (persist) setStoredQueryMode(currentQueryMode);
}

function parseCommaList(text) {
  const raw = (text || '').trim();
  if (!raw) return null;
  const parts = raw
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s);
  return parts.length ? parts : null;
}

function formatTriBool(value) {
  if (value === true || value === 'true' || value === 1 || value === '1') return 'true';
  if (value === false || value === 'false' || value === 0 || value === '0') return 'false';
  return '';
}

function formatFiltered(value) {
  const v = formatTriBool(value);
  if (v === 'true') return '保留（true）';
  if (v === 'false') return '剔除（false）';
  return '';
}

function truncate(text, maxLen = 80) {
  const raw = (text || '').toString();
  if (raw.length <= maxLen) return raw;
  return raw.slice(0, maxLen) + '…';
}

function splitTitlePath(titlePath) {
  const raw = String(titlePath || '').trim();
  if (!raw) return [];
  if (raw.includes('>')) return raw.split('>').map((s) => s.trim()).filter(Boolean);
  if (raw.includes('/')) return raw.split('/').map((s) => s.trim()).filter(Boolean);
  return [raw];
}

function isProbablyChunkId(value) {
  const s = String(value || '').trim();
  return /^[a-f0-9]{40}$/i.test(s);
}

function clearRowSelected() {
  currentDetailId = null;
  const tbody = $('#itemsTbody');
  if (!tbody) return;
  tbody.querySelectorAll('tr.row-selected').forEach((tr) => tr.classList.remove('row-selected'));
}

function setRowSelected(id) {
  currentDetailId = id ? String(id) : null;
  const tbody = $('#itemsTbody');
  if (!tbody) return;
  tbody.querySelectorAll('tr').forEach((tr) => {
    if (currentDetailId && tr.dataset && tr.dataset.id === currentDetailId) tr.classList.add('row-selected');
    else tr.classList.remove('row-selected');
  });
}

function setQaDetailPlaceholder(text) {
  const qaPanel = $('#qaDetailPanel');
  if (!qaPanel) return;
  qaPanel.classList.add('muted');
  qaPanel.textContent = text || '';
}

function setChunkDetailPlaceholder(text) {
  const chunkPanel = $('#chunkDetailPanel');
  if (!chunkPanel) return;
  chunkPanel.classList.add('muted');
  chunkPanel.textContent = text || '';
}

function setRawDetail(jsonObj) {
  const rawPanel = $('#detailRawPanel');
  if (!rawPanel) return;
  if (jsonObj == null) {
    rawPanel.textContent = '';
    return;
  }
  try {
    rawPanel.textContent = JSON.stringify(jsonObj, null, 2);
  } catch {
    rawPanel.textContent = String(jsonObj);
  }
}

function renderBadges(panel, badges) {
  const wrap = document.createElement('div');
  wrap.className = 'badges';
  (badges || []).forEach((b) => {
    if (!b || !b.text) return;
    const el = document.createElement('span');
    el.className = `badge${b.className ? ' ' + b.className : ''}`;
    el.textContent = String(b.text);
    if (b.title) el.title = String(b.title);
    wrap.appendChild(el);
  });
  panel.appendChild(wrap);
}

function renderKv(panel, rows) {
  const kv = document.createElement('div');
  kv.className = 'kv';
  (rows || []).forEach((r) => {
    const key = r?.k != null ? String(r.k) : '';
    const val = r?.v != null ? String(r.v) : '';
    if (!key && !val) return;
    const row = document.createElement('div');
    row.className = 'kv-row';
    const k = document.createElement('div');
    k.className = 'kv-k';
    k.textContent = key;
    const v = document.createElement('div');
    v.className = 'kv-v';
    v.textContent = val;
    row.appendChild(k);
    row.appendChild(v);
    kv.appendChild(row);
  });
  panel.appendChild(kv);
}

function renderScoreGroup(panel, title, scores, reasons) {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.createScoreGroup === 'function') {
    const group = apiuseUi.createScoreGroup(title, scores, reasons, {
      sortKeys: true,
      digits: 3,
      reasonPlacement: 'group',
    });
    if (group && panel) panel.appendChild(group);
  }
}

function renderQaDetail(data) {
  const qaPanel = $('#qaDetailPanel');
  if (!qaPanel) return;
  qaPanel.classList.remove('muted');
  qaPanel.innerHTML = '';

  const qaId = String(data?.id || '');
  const taskId = String(data?.task_id || '');
  const filename = String(data?.original_filename || '');
  const filtered = formatTriBool(data?.filtered);
  const avg = typeof data?.average_score === 'number' ? data.average_score.toFixed(4) : '';
  const faith = typeof data?.faithfulness === 'number' ? data.faithfulness.toFixed(4) : '';
  const isAug = data?.is_augmented === true;
  const source = data?.source != null ? String(data.source) : '';

  renderBadges(qaPanel, [
    { text: `id: ${truncate(qaId, 16)}`, title: qaId, className: 'theme' },
    taskId ? { text: `task: ${truncate(taskId, 16)}`, title: taskId, className: 'theme' } : null,
    filename ? { text: `file: ${truncate(filename, 24)}`, title: filename, className: 'file' } : null,
    avg ? { text: `avg: ${avg}`, className: 'score' } : null,
    faith ? { text: `faith: ${faith}`, className: 'faith' } : null,
    filtered ? { text: `filtered: ${filtered}`, className: 'theme' } : null,
    isAug ? { text: '增广 (augmented)', className: 'theme' } : null,
  ]);

  const admin = data?.admin && typeof data.admin === 'object' ? data.admin : null;
  const adminIsActive =
    admin && admin.is_active !== undefined && admin.is_active !== null ? String(!!admin.is_active) : '';
  const adminReviewStatus = admin && admin.review_status != null ? String(admin.review_status) : '';
  const adminReviewNote = admin && admin.review_note != null ? String(admin.review_note) : '';

  renderKv(qaPanel, [
    { k: 'question', v: data?.question || '' },
    { k: 'answer', v: data?.answer || '' },
    source ? { k: 'source', v: source } : null,
    data?.source_fact_text ? { k: 'source_fact_text', v: data.source_fact_text } : null,
    data?.knowledge_category ? { k: 'category', v: data.knowledge_category } : null,
    data?.question_type ? { k: 'q_type', v: data.question_type } : null,
    data?.difficulty_level ? { k: 'difficulty', v: data.difficulty_level } : null,
    data?.evaluation_method ? { k: 'eval_method', v: data.evaluation_method } : null,
    data?.created_at != null ? { k: 'created_at', v: String(data.created_at) } : null,
    adminIsActive ? { k: 'is_active', v: adminIsActive } : null,
    adminReviewStatus ? { k: 'review_status', v: adminReviewStatus } : null,
    adminReviewNote ? { k: 'review_note', v: adminReviewNote } : null,
  ].filter(Boolean));

  const evaluation = data?.evaluation;
  if (evaluation && typeof evaluation === 'object') {
    const evalWrap = document.createElement('div');
    evalWrap.className = 'evaluation';
    const h4 = document.createElement('h4');
    h4.textContent = 'Evaluation';
    evalWrap.appendChild(h4);

    const llm = evaluation?.llm;
    if (llm && typeof llm === 'object') {
      renderScoreGroup(evalWrap, 'LLM', llm.scores, llm.reasons);
    }
    const local = evaluation?.local;
    if (local && typeof local === 'object') {
      renderScoreGroup(evalWrap, 'Local', local.scores, null);
    }

    if (evalWrap.childElementCount > 1) qaPanel.appendChild(evalWrap);
  }

  const ue = data?.unsupervised_evaluation;
  if (ue && typeof ue === 'object' && ue.scores && typeof ue.scores === 'object') {
    const explainWrap = document.createElement('div');
    explainWrap.className = 'evaluation';
    const h4 = document.createElement('h4');
    h4.textContent = '无监督四维评分解释';
    explainWrap.appendChild(h4);

    const apiuseUi = ui();
    if (apiuseUi && typeof apiuseUi.renderUnsupervisedExplain === 'function') {
      explainWrap.appendChild(apiuseUi.renderUnsupervisedExplain(data, { includeQa: false, includeRaw: false }));
    } else {
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(ue, null, 2);
      explainWrap.appendChild(pre);
    }

    qaPanel.appendChild(explainWrap);
  }
}

function renderChunkDetail(chunk) {
  const chunkPanel = $('#chunkDetailPanel');
  if (!chunkPanel) return;
  chunkPanel.classList.remove('muted');
  chunkPanel.innerHTML = '';

  const chunkId = String(chunk?.chunk_id || chunk?.id || '');
  const titlePath = String(chunk?.title_path || '');
  const titleParts = splitTitlePath(titlePath);

  renderBadges(chunkPanel, [
    chunkId ? { text: `chunk: ${truncate(chunkId, 16)}`, title: chunkId, className: 'theme' } : null,
    chunk?.doc_id ? { text: `doc: ${truncate(String(chunk.doc_id), 16)}`, title: String(chunk.doc_id), className: 'theme' } : null,
    chunk?.task_id ? { text: `task: ${truncate(String(chunk.task_id), 16)}`, title: String(chunk.task_id), className: 'theme' } : null,
    chunk?.original_filename
      ? { text: `file: ${truncate(String(chunk.original_filename), 24)}`, title: String(chunk.original_filename), className: 'file' }
      : null,
    chunk?.chunk_index != null ? { text: `idx: ${chunk.chunk_index}`, className: 'theme' } : null,
    chunk?.level != null ? { text: `level: ${chunk.level}`, className: 'theme' } : null,
    chunk?.is_leaf != null ? { text: `leaf: ${chunk.is_leaf ? 'true' : 'false'}`, className: 'theme' } : null,
  ]);

  if (titleParts.length) {
    const path = document.createElement('div');
    path.className = 'path-graph';
    titleParts.forEach((p, idx) => {
      const node = document.createElement('span');
      node.className = 'path-node';
      const inner = document.createElement('span');
      inner.textContent = p;
      node.appendChild(inner);
      path.appendChild(node);
      if (idx < titleParts.length - 1) {
        const arrow = document.createElement('span');
        arrow.className = 'path-arrow';
        arrow.textContent = '→';
        path.appendChild(arrow);
      }
    });
    chunkPanel.appendChild(path);
  }

  renderKv(chunkPanel, [
    chunk?.index_path ? { k: 'index_path', v: String(chunk.index_path) } : null,
    chunk?.parent_index_path ? { k: 'parent_index_path', v: String(chunk.parent_index_path) } : null,
    chunk?.root_index_path ? { k: 'root_index_path', v: String(chunk.root_index_path) } : null,
    titlePath ? { k: 'title_path', v: titlePath } : null,
    chunk?.created_at != null ? { k: 'created_at', v: String(chunk.created_at) } : null,
  ].filter(Boolean));

  const text = chunk?.text != null ? String(chunk.text) : '';
  if (text.trim()) {
    const details = document.createElement('details');
    details.open = true;
    const summary = document.createElement('summary');
    summary.textContent = 'chunk 正文';
    const body = document.createElement('div');
    body.className = 'chunk-text';
    body.textContent = text;
    details.appendChild(summary);
    details.appendChild(body);
    chunkPanel.appendChild(details);
  }
}

function syncSelectedCount() {
  const el = $('#selectedCount');
  if (el) el.textContent = String(selectedIds.size);
}

function getCheckedIds() {
  return Array.from(selectedIds);
}

function clearSelection() {
  selectedIds = new Set();
  syncSelectedCount();
  const chkAll = $('#chkAll');
  if (chkAll) chkAll.checked = false;
}

function buildListQueryParams() {
  const params = new URLSearchParams();
  const taskId = $('#fTaskId')?.value.trim();
  const filename = $('#fFilename')?.value.trim();
  const categories = parseCommaList($('#fCategory')?.value);
  const qtypes = parseCommaList($('#fQType')?.value);
  const diffs = parseCommaList($('#fDiff')?.value);
  const filtered = $('#fFiltered')?.value || 'all';
  const evaluated = $('#fEvaluated')?.value || 'all';
  const isActive = $('#fActive')?.value || 'true';
  const reviewStatus = $('#fReviewStatus')?.value.trim();
  const minScore = $('#fMinScore')?.value;
  const q = $('#fQ')?.value.trim();
  const page = $('#fPage')?.value || '1';
  const pageSize = $('#fPageSize')?.value || '20';

  if (taskId) params.set('task_id', taskId);
  if (filename) params.set('original_filename', filename);
  if (categories) categories.forEach((v) => params.append('knowledge_category', v));
  if (qtypes) qtypes.forEach((v) => params.append('question_type', v));
  if (diffs) diffs.forEach((v) => params.append('difficulty_level', v));
  if (filtered) params.set('filtered', filtered);
  if (evaluated) params.set('evaluated', evaluated);
  if (isActive) params.set('is_active', isActive);
  if (reviewStatus) params.set('review_status', reviewStatus);
  if (minScore !== undefined && minScore !== null && String(minScore).trim() !== '') {
    params.set('min_avg_score', String(minScore));
  }
  if (q) params.set('q', q);
  params.set('page', String(page));
  params.set('page_size', String(pageSize));
  return params;
}

async function loadList() {
  clearSelection();
  setStatus('加载中…');
  const base = getApiBaseUrl();
  const params = buildListQueryParams();
  const url = `${base}/admin/v1/qa-items?${params.toString()}`;
  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    const items = Array.isArray(data.items) ? data.items : [];
    currentItems = items;
    renderMeta(data);
    renderTable(items);
    setStatus(`列表：返回 ${items.length} 条`);
  } catch (err) {
    setStatus('列表查询失败：' + String(err), true);
    currentItems = [];
    renderTable([]);
  }
}

async function rerunCurrentQuery() {
  if (currentQueryMode === QUERY_MODE_SEMANTIC) {
    const queryText = $('#fSemantic')?.value.trim();
    if (queryText) {
      await semanticSearch();
      return;
    }
  }
  await loadList();
}

async function ingestConsolidated() {
  const base = getApiBaseUrl();
  const name = $('#ingestOutputFile')?.value.trim();
  if (!name) {
    setStatus('请先填写 consolidated JSON 文件名', true);
    return;
  }
  if (!confirm(`重新入库 ${name}？\n将覆盖同 id 的记录，并补全增广问句入库。`)) return;
  setStatus('回填入库中…');
  try {
    const resp = await fetch(`${base}/admin/v1/ingest-consolidated`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ output_file: name }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    const msg = data?.milvus?.message || 'ok';
    setStatus(`回填完成：${msg}`);
    await rerunCurrentQuery();
  } catch (err) {
    setStatus('回填失败：' + String(err), true);
  }
}

async function semanticSearch() {
  clearSelection();
  setStatus('语义检索中…');
  const base = getApiBaseUrl();
  const queryText = $('#fSemantic')?.value.trim();
  if (!queryText) {
    setStatus('请先输入 semantic_query', true);
    return;
  }
  const taskId = $('#fTaskId')?.value.trim() || null;
  const categories = parseCommaList($('#fCategory')?.value);
  const qtypes = parseCommaList($('#fQType')?.value);
  const diffs = parseCommaList($('#fDiff')?.value);
  const filtered = $('#fFiltered')?.value || 'all';
  const isActive = $('#fActive')?.value || 'true';
  const minScore = $('#fMinScore')?.value;
  const topK = Number($('#fTopK')?.value || 30);

  const body = {
    query_text: queryText,
    top_k: Math.max(1, Math.min(200, topK)),
    task_id: taskId,
    filtered,
    is_active: isActive,
    min_avg_score:
      minScore !== undefined && minScore !== null && String(minScore).trim() !== ''
        ? Number(minScore)
        : null,
    knowledge_category: categories,
    question_type: qtypes,
    difficulty_level: diffs,
  };

  try {
    const resp = await fetch(`${base}/admin/v1/qa-search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    const items = Array.isArray(data.results) ? data.results : [];
    currentItems = items;
    renderMeta({
      filters: body,
      pagination: { page: 1, page_size: items.length, total_items: items.length, total_pages: 1 },
    });
    renderTable(items, true);
    setStatus(`语义检索：返回 ${items.length} 条`);
  } catch (err) {
    setStatus('语义检索失败：' + String(err), true);
    currentItems = [];
    renderTable([]);
  }
}

function renderMeta(data) {
  const el = $('#metaText');
  if (!el) return;
  const pagination = data?.pagination || {};
  const filters = data?.filters || {};
  el.textContent = `total=${pagination.total_items ?? '-'} page=${pagination.page ?? '-'} page_size=${
    pagination.page_size ?? '-'
  } filters=${JSON.stringify(filters)}`;
}

function renderTable(items, isSearch = false) {
  const tbody = $('#itemsTbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const list = Array.isArray(items) ? items : [];
  list.forEach((it) => {
    const id = it.id;
    const avg =
      typeof it.average_score === 'number' && it.average_score >= 0
        ? it.average_score.toFixed(4)
        : '';
    const faith =
      typeof it.faithfulness === 'number' && it.faithfulness >= 0
        ? it.faithfulness.toFixed(4)
        : '';
    const unsupScores =
      it.unsupervised_scores ||
      (it.unsupervised_evaluation && it.unsupervised_evaluation.scores) ||
      null;
    const ans =
      unsupScores && typeof unsupScores.answerability === 'number'
        ? unsupScores.answerability.toFixed(4)
        : '';
    const f1 =
      unsupScores && typeof unsupScores.unsupervised_f1 === 'number'
        ? unsupScores.unsupervised_f1.toFixed(4)
        : '';
    const filtered = formatFiltered(it.filtered);
    const active = it.admin?.is_active === false ? 'false' : 'true';
    const q = truncate(it.question, 70);
    const a = truncate(it.answer, 70);
    const cat = truncate(it.knowledge_category || '', 24);
    const file = truncate(it.original_filename || '', 18);
    const sim = typeof it.similarity_score === 'number' ? it.similarity_score.toFixed(4) : null;
    const augPrefix = it.is_augmented === true ? '[增广] ' : '';

    const tr = document.createElement('tr');
    tr.dataset.id = id;
    tr.tabIndex = 0;
    tr.setAttribute('role', 'button');
    tr.setAttribute('aria-label', `查看详情 ${truncate(id, 16)}`);
    if (currentDetailId && String(id) === String(currentDetailId)) tr.classList.add('row-selected');
    tr.addEventListener('click', (e) => {
      const target = e?.target;
      if (target && target.closest) {
        if (target.closest('button')) return;
        if (target.closest('input')) return;
        if (target.closest('a')) return;
      }
      loadDetail(id);
    });
    tr.addEventListener('keydown', (e) => {
      const key = e?.key;
      if (key !== 'Enter' && key !== ' ') return;
      const target = e?.target;
      if (target && target.closest) {
        if (target.closest('button')) return;
        if (target.closest('input')) return;
        if (target.closest('a')) return;
      }
      e.preventDefault();
      loadDetail(id);
    });

    const tdChk = document.createElement('td');
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.setAttribute('aria-label', `选择记录 ${truncate(id, 16)}`);
    chk.setAttribute('title', String(id));
    chk.checked = selectedIds.has(id);
    chk.addEventListener('change', () => {
      if (chk.checked) selectedIds.add(id);
      else selectedIds.delete(id);
      syncSelectedCount();
    });
    tdChk.appendChild(chk);

    const tdAvg = document.createElement('td');
    tdAvg.textContent = sim !== null ? `${avg} (sim=${sim})` : avg;

    const tdFaith = document.createElement('td');
    tdFaith.textContent = faith;

    const tdAns = document.createElement('td');
    tdAns.textContent = ans;

    const tdF1 = document.createElement('td');
    tdF1.textContent = f1;

    const tdFiltered = document.createElement('td');
    tdFiltered.textContent = filtered;

    const tdActive = document.createElement('td');
    tdActive.textContent = active;

    const tdQ = document.createElement('td');
    tdQ.textContent = augPrefix + q;

    const tdA = document.createElement('td');
    tdA.textContent = a;

    const tdCat = document.createElement('td');
    tdCat.textContent = cat;

    const tdFile = document.createElement('td');
    tdFile.textContent = file;

    const tdOp = document.createElement('td');
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = '详情';
    btn.className = 'secondary';
    btn.addEventListener('click', () => loadDetail(id));
    tdOp.appendChild(btn);

    tr.appendChild(tdChk);
    tr.appendChild(tdAvg);
    tr.appendChild(tdFaith);
    tr.appendChild(tdAns);
    tr.appendChild(tdF1);
    tr.appendChild(tdFiltered);
    tr.appendChild(tdActive);
    tr.appendChild(tdQ);
    tr.appendChild(tdA);
    tr.appendChild(tdCat);
    tr.appendChild(tdFile);
    tr.appendChild(tdOp);

    tbody.appendChild(tr);
  });

  const btnPrev = $('#btnPrev');
  const btnNext = $('#btnNext');
  if (btnPrev) btnPrev.disabled = isSearch;
  if (btnNext) btnNext.disabled = isSearch;
}

async function loadDetail(id) {
  const base = getApiBaseUrl();
  setRowSelected(id);
  setQaDetailPlaceholder('加载中…');
  setChunkDetailPlaceholder('等待加载…');
  setRawDetail(null);
  let data = null;
  try {
    const resp = await fetch(`${base}/admin/v1/qa-items/${encodeURIComponent(id)}`);
    data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    renderQaDetail(data);
    setRawDetail(data);
  } catch (err) {
    setQaDetailPlaceholder('详情获取失败：' + String(err));
    setChunkDetailPlaceholder('');
    return;
  }

  const source = data?.source != null ? String(data.source).trim() : '';
  if (!source) {
    setChunkDetailPlaceholder('该 QA 没有 source，无法定位对应 chunk。');
    return;
  }
  if (!isProbablyChunkId(source)) {
    setChunkDetailPlaceholder(
      `source 不是 chunk_id（可能是旧数据/未开启 chunk 入库），无法溯源到 doc_tree_chunks。\nsource=${source}`,
    );
    return;
  }

  setChunkDetailPlaceholder('加载 chunk 详情中…');
  try {
    const chunkResp = await fetch(`${base}/doc-chunks/${encodeURIComponent(source)}`);
    const chunkData = await chunkResp.json();
    if (!chunkResp.ok) throw new Error(chunkData?.detail || chunkResp.statusText);
    renderChunkDetail(chunkData?.chunk || chunkData);
  } catch (err) {
    setChunkDetailPlaceholder('chunk 详情获取失败：' + String(err));
  }
}

async function postJson(path, body) {
  const base = getApiBaseUrl();
  const resp = await fetch(`${base}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.detail || resp.statusText);
  }
  return data;
}

async function doBatchUpdate(patch) {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  setStatus('提交中…');
  try {
    await postJson('/admin/v1/qa-items/batch-update', { ids, patch });
    setStatus('批量更新成功');
    await rerunCurrentQuery();
  } catch (err) {
    setStatus('批量更新失败：' + String(err), true);
  }
}

async function doBatchAdminUpdate(patch) {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  setStatus('提交中…');
  try {
    await postJson('/admin/v1/qa-items/batch-admin-update', { ids, patch });
    setStatus('批量治理更新成功');
    await rerunCurrentQuery();
  } catch (err) {
    setStatus('批量治理更新失败：' + String(err), true);
  }
}

async function doSoftDelete() {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  const note = prompt('（可选）review_note：', '') || '';
  if (!confirm(`确认下架（软删） ${ids.length} 条吗？`)) return;
  setStatus('下架中…');
  try {
    await postJson('/admin/v1/qa-items/batch-delete', { ids, mode: 'soft', review_note: note });
    setStatus('已下架');
    await rerunCurrentQuery();
  } catch (err) {
    setStatus('下架失败：' + String(err), true);
  }
}

async function doRestore() {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  if (!confirm(`确认恢复上架 ${ids.length} 条吗？`)) return;
  setStatus('恢复中…');
  try {
    await postJson('/admin/v1/qa-items/batch-admin-update', {
      ids,
      patch: { is_active: true, review_status: 'approved' },
    });
    setStatus('已恢复');
    await rerunCurrentQuery();
  } catch (err) {
    setStatus('恢复失败：' + String(err), true);
  }
}

async function doHardDelete() {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  const ok = confirm(
    `确认硬删 ${ids.length} 条吗？\n建议先用“导出”备份，硬删会从 Milvus 中真正删除。`,
  );
  if (!ok) return;
  setStatus('硬删中…');
  try {
    await postJson('/admin/v1/qa-items/batch-delete', { ids, mode: 'hard', backup_enabled: true });
    setStatus('硬删完成');
    await rerunCurrentQuery();
  } catch (err) {
    setStatus('硬删失败：' + String(err), true);
  }
}

async function startEval() {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  const method = $('#evalMethod')?.value || 'llm';
  const force = ($('#evalForce')?.value || 'false') === 'true';
  const autoFilter = ($('#evalAutoFilter')?.value || 'false') === 'true';
  const threshold = Number($('#evalThreshold')?.value || 0.7);
  const criteriaRaw = $('#evalCriteria')?.value || '';
  const criteria = parseCommaList(criteriaRaw) || [];

  const body = {
    selection: { ids },
    evaluation_method: method,
    force,
    write_back: true,
    auto_filter: { enabled: autoFilter, score_threshold: threshold },
  };
  if (method === 'llm') body.criteria_list = criteria;

  setStatus('提交评估任务…');
  try {
    const data = await postJson('/admin/v1/evaluation-jobs', body);
    const jobId = data.job_id;
    addJob(jobId);
    setStatus(`已创建评估任务 job_id=${jobId}`);
  } catch (err) {
    setStatus('创建评估任务失败：' + String(err), true);
  }
}

function syncEvalMethodUI() {
  const method = $('#evalMethod')?.value || 'llm';
  const criteriaWrap = $('#evalCriteriaWrap');
  const localHint = $('#evalLocalMetricsHint');
  if (criteriaWrap) criteriaWrap.style.display = method === 'llm' ? '' : 'none';
  if (localHint) localHint.style.display = method === 'local' ? '' : 'none';
}

async function startUnsupervisedEval() {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  const force = ($('#unsupForce')?.value || 'false') === 'true';

  const body = {
    selection: { ids },
    force,
    write_back: true,
    include_inactive: false,
  };

  setStatus('提交无监督评估任务…');
  try {
    const data = await postJson('/admin/v1/unsupervised-evaluation-jobs', body);
    const jobId = data.job_id;
    addJob(jobId);
    setStatus(`已创建无监督评估任务 job_id=${jobId}`);
  } catch (err) {
    setStatus('创建无监督评估任务失败：' + String(err), true);
  }
}

async function startExport() {
  const ids = getCheckedIds();
  if (!ids.length) {
    notify('请先勾选记录', 'warning');
    return;
  }
  setStatus('提交导出任务…');
  try {
    const data = await postJson('/admin/v1/exports', { ids });
    const jobId = data.job_id;
    addJob(jobId);
    setStatus(`已创建导出任务 job_id=${jobId}`);
  } catch (err) {
    setStatus('创建导出任务失败：' + String(err), true);
  }
}

function addJob(jobId) {
  if (!jobId) return;
  if (!jobIds.includes(jobId)) {
    jobIds.unshift(jobId);
    if (jobIds.length > 20) jobIds = jobIds.slice(0, 20);
  }
  refreshJobs().catch(() => {});
}

async function refreshJobs() {
  const base = getApiBaseUrl();
  const panel = $('#jobsPanel');
  if (!panel) return;
  if (!jobIds.length) {
    panel.textContent = '暂无 job';
    return;
  }
  const updates = [];
  for (const id of jobIds) {
    updates.push(
      fetch(`${base}/admin/v1/jobs/${encodeURIComponent(id)}`)
        .then((r) => r.json().then((d) => ({ ok: r.ok, data: d })))
        .catch((e) => ({ ok: false, data: { detail: String(e) } })),
    );
  }
  const results = await Promise.all(updates);
  results.forEach((res, idx) => {
    jobCache[jobIds[idx]] = res.data;
  });
  renderJobs();
}

function renderJobs() {
  const panel = $('#jobsPanel');
  if (!panel) return;
  if (!jobIds.length) {
    panel.textContent = '暂无 job';
    return;
  }
  panel.innerHTML = '';
  jobIds.forEach((id) => {
    const j = jobCache[id];
    const div = document.createElement('div');
    const status = j?.status || 'unknown';
    const processed = j?.processed ?? '-';
    const total = j?.total ?? '-';
    const msg = j?.message || '';
    const lastLog = Array.isArray(j?.logs) && j.logs.length ? j.logs[j.logs.length - 1] : '';
    div.style.padding = '6px 0';
    div.style.borderBottom = '1px solid var(--border)';
    const paramText = formatJobParams(j);
    div.textContent = `${id} [${j?.job_type || '-'}] ${status} (${processed}/${total}) ${msg}${
      paramText ? ' | ' + paramText : ''
    }${lastLog ? ' | ' + lastLog : ''}`;
    if (status === 'running' || status === 'queued') {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = '取消';
      btn.className = 'secondary';
      btn.style.marginLeft = '8px';
      btn.addEventListener('click', () => cancelOneJob(id));
      div.appendChild(btn);
    }
    panel.appendChild(div);
  });
}

function formatJobParams(job) {
  const p = job?.params;
  if (!p || typeof p !== 'object') return '';

  if (job?.job_type === 'evaluation') {
    const method = String(p.evaluation_method || '');
    const methodLabel = method === 'llm' ? '大模型（llm）' : method === 'local' ? '本地（local）' : method;
    const parts = [];
    if (methodLabel) parts.push(`方式:${methodLabel}`);
    if (method === 'llm' && Array.isArray(p.criteria_list) && p.criteria_list.length) {
      parts.push(`维度(criteria_list):${p.criteria_list.join(',')}`);
    }
    if (method === 'local' && Array.isArray(p.local_metrics) && p.local_metrics.length) {
      parts.push(`指标(local_metrics):${p.local_metrics.join(',')}`);
    }
    if (p.auto_filter && typeof p.auto_filter === 'object') {
      const enabled = !!p.auto_filter.enabled;
      const th = p.auto_filter.score_threshold;
      parts.push(`自动筛选(auto_filter):${enabled ? 'true' : 'false'}`);
      if (th !== undefined && th !== null && String(th).trim() !== '') {
        parts.push(`阈值(threshold):${th}`);
      }
    }
    return parts.join('；');
  }

  if (job?.job_type === 'unsupervised_evaluation') {
    const parts = [];
    parts.push('类型:无监督（unsupervised）');
    if (p.force !== undefined) parts.push(`force:${p.force ? 'true' : 'false'}`);
    if (p.write_back !== undefined) parts.push(`write_back:${p.write_back ? 'true' : 'false'}`);
    return parts.join('；');
  }

  if (job?.job_type === 'export') {
    const parts = [];
    parts.push('类型:导出（export）');
    if (p.include_inactive !== undefined) {
      parts.push(`include_inactive:${p.include_inactive ? 'true' : 'false'}`);
    }
    return parts.join('；');
  }

  try {
    return `参数:${JSON.stringify(p)}`;
  } catch {
    return '';
  }
}

async function cancelOneJob(jobId) {
  if (!confirm(`取消 job ${jobId}？`)) return;
  const base = getApiBaseUrl();
  try {
    const resp = await fetch(`${base}/admin/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    await refreshJobs();
  } catch (err) {
    notify('取消失败：' + String(err), 'error');
  }
}

function bindEvents() {
  $('#btnLoadList')?.addEventListener('click', () => setQueryMode(QUERY_MODE_LIST));
  $('#btnSemanticSearch')?.addEventListener('click', () => setQueryMode(QUERY_MODE_SEMANTIC));
  $('#btnRunListQuery')?.addEventListener('click', loadList);
  $('#btnRunSemanticSearch')?.addEventListener('click', semanticSearch);
  $('#btnIngestConsolidated')?.addEventListener('click', ingestConsolidated);
  $('#evalMethod')?.addEventListener('change', syncEvalMethodUI);
  $('#btnClear')?.addEventListener('click', () => {
    $('#filterForm')?.reset();
    $('#fFiltered').value = 'all';
    $('#fEvaluated').value = 'all';
    $('#fActive').value = 'true';
    $('#fTopK').value = '30';
    $('#fPage').value = '1';
    $('#fPageSize').value = '20';
    clearSelection();
    clearRowSelected();
    setQaDetailPlaceholder('点击列表中的「详情」查看');
    setChunkDetailPlaceholder('等待选择 QA…');
    setRawDetail(null);
    setStatus('');
  });
  $('#btnFilteredTrue')?.addEventListener('click', () => doBatchUpdate({ filtered: true }));
  $('#btnFilteredFalse')?.addEventListener('click', () => doBatchUpdate({ filtered: false }));
  $('#btnSetCategory')?.addEventListener('click', async () => {
    const v = prompt('设置 knowledge_category：', '') || '';
    if (!v.trim()) return;
    await doBatchUpdate({ knowledge_category: v.trim() });
  });
  $('#btnSetQType')?.addEventListener('click', async () => {
    const v = prompt('设置 question_type：', '') || '';
    if (!v.trim()) return;
    await doBatchUpdate({ question_type: v.trim() });
  });
  $('#btnSetDiff')?.addEventListener('click', async () => {
    const v = prompt('设置 difficulty_level：', '') || '';
    if (!v.trim()) return;
    await doBatchUpdate({ difficulty_level: v.trim() });
  });
  $('#btnSetReview')?.addEventListener('click', async () => {
    const status = (prompt('设置 review_status（如 pending/approved/rejected/deleted）：', '') || '').trim();
    const note = (prompt('（可选）review_note：', '') || '').trim();
    if (!status && !note) return;
    await doBatchAdminUpdate({ review_status: status || undefined, review_note: note || undefined });
  });
  $('#btnSoftDelete')?.addEventListener('click', doSoftDelete);
  $('#btnRestore')?.addEventListener('click', doRestore);
  $('#btnHardDelete')?.addEventListener('click', doHardDelete);
  $('#btnExport')?.addEventListener('click', startExport);
  $('#btnEvalStart')?.addEventListener('click', startEval);
  $('#btnUnsupStart')?.addEventListener('click', startUnsupervisedEval);

  $('#btnJobsRefresh')?.addEventListener('click', refreshJobs);

  $('#chkAll')?.addEventListener('change', (e) => {
    const checked = !!e.target.checked;
    const tbody = $('#itemsTbody');
    if (!tbody) return;
    tbody.querySelectorAll('input[type="checkbox"]').forEach((chk) => {
      chk.checked = checked;
      const tr = chk.closest('tr');
      const id = tr?.dataset?.id;
      if (!id) return;
      if (checked) selectedIds.add(id);
      else selectedIds.delete(id);
    });
    syncSelectedCount();
  });

  $('#btnPrev')?.addEventListener('click', async () => {
    if (currentQueryMode !== QUERY_MODE_LIST) return;
    const page = Number($('#fPage')?.value || 1);
    if (page <= 1) return;
    $('#fPage').value = String(page - 1);
    await loadList();
  });
  $('#btnNext')?.addEventListener('click', async () => {
    if (currentQueryMode !== QUERY_MODE_LIST) return;
    const page = Number($('#fPage')?.value || 1);
    $('#fPage').value = String(page + 1);
    await loadList();
  });
}

function init() {
  const apiInput = $('#apiBaseUrl');
  if (apiInput) apiInput.value = window.location.origin;
  setQueryMode(getStoredQueryMode(), { persist: false });
  bindEvents();
  syncEvalMethodUI();
  syncSelectedCount();
  refreshJobs().catch(() => {});
  setInterval(() => {
    const running = jobIds.some((id) => {
      const st = jobCache[id]?.status;
      return st === 'running' || st === 'queued';
    });
    if (running) refreshJobs().catch(() => {});
  }, 2000);
}

init();
