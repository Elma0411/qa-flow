window.__QA_UI_APPJS_READY__ = true;
window.__QA_UI_APPJS_VERSION__ = '2026-06-23-1';

initApiBaseUrl();
restoreUiCache();
bindUiCache();

// ---------- 完整流水线调用 ----------

const pipelineForm = $('#pipelineForm');
if (pipelineForm) {
  pipelineForm.addEventListener('submit', handlePipelineSubmit);
}
setupFewShotUI();
setupEvaluationUI();
setupQuestionTypeUI();
setupChunkingModeUI();
setupDocumentProcessingModeUI();
const cancelTaskBtn = $('#cancelTaskBtn');
if (cancelTaskBtn) {
  cancelTaskBtn.addEventListener('click', handleCancelTask);
}
const loadTaskStatusBtn = $('#btnLoadTaskStatus');
if (loadTaskStatusBtn) {
  loadTaskStatusBtn.addEventListener('click', () => {
    loadTaskStatusById($('#taskIdInput')?.value || '', { resumePolling: true, silent: false }).catch(() => {});
  });
}
const taskIdInputEl = $('#taskIdInput');
if (taskIdInputEl) {
  taskIdInputEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    loadTaskStatusById($('#taskIdInput')?.value || '', { resumePolling: true, silent: false }).catch(() => {});
  });
}
const restoreLastTaskBtn = $('#btnRestoreLastTask');
if (restoreLastTaskBtn) {
  restoreLastTaskBtn.addEventListener('click', () => {
    const page = getRuntimePageState();
    const rememberedId =
      String(page.activeTaskId || '').trim() ||
      String(page.selectedTaskId || '').trim() ||
      String($('#taskIdInput')?.value || '').trim();
    if (!rememberedId) {
      notify('当前没有可恢复的 task_id', 'warning');
      return;
    }
    loadTaskStatusById(rememberedId, { resumePolling: true, silent: false }).catch(() => {});
  });
}
const environmentCheckBtn = $('#btnEnvironmentCheck');
if (environmentCheckBtn) {
  environmentCheckBtn.addEventListener('click', handleEnvironmentCheck);
}
hydratePipelineRuntime().catch(() => {});

async function handleEnvironmentCheck() {
  const btn = $('#btnEnvironmentCheck');
  const statusEl = $('#environmentCheckStatus');
  const summaryEl = $('#environmentCheckSummary');
  const resultsEl = $('#environmentCheckResults');
  const base = getApiBaseUrl();

  setBtnLoading(btn, true);
  if (statusEl) {
    statusEl.className = 'env-check-status is-running';
    statusEl.textContent = '检测中…';
  }
  if (summaryEl) {
    summaryEl.hidden = false;
    summaryEl.textContent = '正在检查接口、数据库、CUDA、模型文件和运行目录';
  }
  if (resultsEl) {
    resultsEl.hidden = true;
    resultsEl.replaceChildren();
  }

  try {
    const data = await fetchJson(`${base}/environment-check`);
    renderEnvironmentCheck(data);
    const status = String(data.status || '').toLowerCase();
    if (status === 'ok') notify('环境检测通过', 'success');
    else if (status === 'warning') notify('环境检测完成：存在警告项', 'warning');
    else notify('环境检测完成：存在失败项', 'error');
  } catch (err) {
    if (statusEl) {
      statusEl.className = 'env-check-status is-error';
      statusEl.textContent = '检测失败';
    }
    if (summaryEl) {
      summaryEl.hidden = false;
      summaryEl.textContent = `无法完成环境检测：${String(err)}`;
    }
    notify(`环境检测失败：${String(err)}`, 'error');
  } finally {
    setBtnLoading(btn, false);
  }
}

function renderEnvironmentCheck(data) {
  const statusEl = $('#environmentCheckStatus');
  const summaryEl = $('#environmentCheckSummary');
  const resultsEl = $('#environmentCheckResults');
  const status = String(data?.status || 'error').toLowerCase();
  const summary = data?.summary && typeof data.summary === 'object' ? data.summary : {};
  const checks = Array.isArray(data?.checks) ? data.checks : [];
  const elapsedMs = Number(data?.elapsed_ms || 0);

  const statusLabel =
    status === 'ok' ? '全部通过' : status === 'warning' ? '存在警告' : '存在失败';
  if (statusEl) {
    statusEl.className = `env-check-status is-${status}`;
    statusEl.textContent = statusLabel;
  }

  if (summaryEl) {
    summaryEl.hidden = false;
    summaryEl.replaceChildren(
      buildEnvSummaryChip('通过', summary.ok || 0, 'ok'),
      buildEnvSummaryChip('警告', summary.warning || 0, 'warning'),
      buildEnvSummaryChip('失败', summary.error || 0, 'error'),
      buildEnvSummaryChip('耗时', `${elapsedMs}ms`, 'time'),
    );
  }

  if (!resultsEl) return;
  resultsEl.hidden = false;
  resultsEl.replaceChildren();

  const grouped = new Map();
  checks.forEach((item) => {
    const category = String(item.category || 'other');
    if (!grouped.has(category)) grouped.set(category, []);
    grouped.get(category).push(item);
  });

  const categoryNames = {
    api: 'API',
    endpoint: '外部端点',
    database: '数据库',
    runtime: 'CUDA / 运行时',
    model: '模型文件',
    storage: '运行目录',
    dependency: '关键依赖',
    other: '其他',
  };

  const statusRank = { error: 0, warning: 1, ok: 2 };
  grouped.forEach((items, category) => {
    const groupEl = document.createElement('section');
    groupEl.className = 'env-check-group';
    const title = document.createElement('h4');
    title.textContent = `${categoryNames[category] || category} · ${items.length}`;
    groupEl.appendChild(title);

    [...items]
      .sort((left, right) => {
        const leftRank = statusRank[String(left?.status || 'error').toLowerCase()] ?? 9;
        const rightRank = statusRank[String(right?.status || 'error').toLowerCase()] ?? 9;
        return leftRank - rightRank;
      })
      .forEach((item) => {
        groupEl.appendChild(buildEnvCheckItem(item));
      });
    resultsEl.appendChild(groupEl);
  });
}

function buildEnvSummaryChip(label, value, type) {
  const chip = document.createElement('span');
  chip.className = `env-summary-chip env-summary-chip--${type}`;
  chip.textContent = `${label}: ${value}`;
  return chip;
}

function buildEnvCheckItem(item) {
  const status = String(item?.status || 'error').toLowerCase();
  const card = document.createElement('article');
  card.className = `env-check-item is-${status}`;

  const head = document.createElement('div');
  head.className = 'env-check-item__head';

  const title = document.createElement('strong');
  title.textContent = String(item?.name || item?.id || '未命名检测项');

  const badge = document.createElement('span');
  badge.className = `env-check-badge is-${status}`;
  badge.textContent = status === 'ok' ? '通过' : status === 'warning' ? '警告' : '失败';

  head.append(title, badge);

  const message = document.createElement('p');
  message.textContent = String(item?.message || '');

  card.append(head, message);

  const detailBtn = document.createElement('button');
  detailBtn.type = 'button';
  detailBtn.className = 'env-detail-button';
  detailBtn.textContent = '查看详情';
  detailBtn.addEventListener('click', () => openEnvironmentCheckDetail(item));
  card.appendChild(detailBtn);

  return card;
}

function openEnvironmentCheckDetail(item) {
  const modal = ensureEnvironmentCheckDetailModal();
  const status = String(item?.status || 'error').toLowerCase();
  const titleEl = modal.querySelector('[data-env-detail-title]');
  const badgeEl = modal.querySelector('[data-env-detail-badge]');
  const messageEl = modal.querySelector('[data-env-detail-message]');
  const metaEl = modal.querySelector('[data-env-detail-meta]');
  const codeEl = modal.querySelector('[data-env-detail-code]');
  const closeBtn = modal.querySelector('[data-env-detail-close]');

  if (titleEl) titleEl.textContent = String(item?.name || item?.id || '未命名检测项');
  if (badgeEl) {
    badgeEl.className = `env-check-badge is-${status}`;
    badgeEl.textContent = status === 'ok' ? '通过' : status === 'warning' ? '警告' : '失败';
  }
  if (messageEl) messageEl.textContent = String(item?.message || '暂无说明');
  if (metaEl) {
    metaEl.replaceChildren(
      buildEnvDetailMeta('检测 ID', item?.id || '-'),
      buildEnvDetailMeta('分组', item?.category || '-'),
      buildEnvDetailMeta('状态', status),
    );
  }
  if (codeEl) {
    codeEl.textContent = JSON.stringify(item?.details || {}, null, 2);
  }

  modal.hidden = false;
  modal.classList.add('is-open');
  closeBtn?.focus();
}

function buildEnvDetailMeta(label, value) {
  const wrap = document.createElement('div');
  wrap.className = 'env-detail-modal__meta-item';
  const key = document.createElement('span');
  key.textContent = label;
  const val = document.createElement('strong');
  val.textContent = String(value);
  wrap.append(key, val);
  return wrap;
}

function ensureEnvironmentCheckDetailModal() {
  let modal = $('#environmentCheckDetailModal');
  if (modal) return modal;

  modal = document.createElement('div');
  modal.id = 'environmentCheckDetailModal';
  modal.className = 'env-detail-modal';
  modal.hidden = true;
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', 'environmentCheckDetailTitle');

  const panel = document.createElement('section');
  panel.className = 'env-detail-modal__panel';

  const header = document.createElement('header');
  header.className = 'env-detail-modal__header';
  const titleWrap = document.createElement('div');
  const eyebrow = document.createElement('div');
  eyebrow.className = 'eyebrow';
  eyebrow.textContent = '检测详情';
  const title = document.createElement('h3');
  title.id = 'environmentCheckDetailTitle';
  title.setAttribute('data-env-detail-title', '');
  title.textContent = '检测详情';
  titleWrap.append(eyebrow, title);
  const closeBtn = document.createElement('button');
  closeBtn.type = 'button';
  closeBtn.className = 'icon-btn env-detail-modal__close';
  closeBtn.setAttribute('aria-label', '关闭详情');
  closeBtn.setAttribute('data-env-detail-close', '');
  closeBtn.textContent = '×';
  header.append(titleWrap, closeBtn);

  const body = document.createElement('div');
  body.className = 'env-detail-modal__body';
  const badge = document.createElement('span');
  badge.className = 'env-check-badge';
  badge.setAttribute('data-env-detail-badge', '');
  const message = document.createElement('p');
  message.className = 'env-detail-modal__message';
  message.setAttribute('data-env-detail-message', '');
  const meta = document.createElement('div');
  meta.className = 'env-detail-modal__meta';
  meta.setAttribute('data-env-detail-meta', '');
  const codeTitle = document.createElement('h4');
  codeTitle.textContent = '原始细节';
  const code = document.createElement('pre');
  code.className = 'env-detail-modal__code';
  code.setAttribute('data-env-detail-code', '');
  body.append(badge, message, meta, codeTitle, code);

  panel.append(header, body);
  modal.appendChild(panel);
  document.body.appendChild(modal);

  const close = () => {
    modal.classList.remove('is-open');
    modal.hidden = true;
  };
  closeBtn.addEventListener('click', close);
  modal.addEventListener('click', (event) => {
    if (event.target === modal) close();
  });
  modal.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') close();
  });

  return modal;
}

function setupQuestionTypeUI() {
  const modeEl = $('#questionTypeMode');
  const typeEls = Array.from($$('input[name="questionTypeOption"]') || []);
  const weightFields = [
    { type: '简答题', el: $('#weightShort') },
    { type: '单选题', el: $('#weightChoice') },
    { type: '判断题', el: $('#weightJudge') },
    { type: '计算题', el: $('#weightCalc') },
  ];

  function getMode() {
    return (modeEl?.value || 'mixed').trim();
  }

  function syncWeightFields() {
    const mixed = getMode() === 'mixed';
    weightFields.forEach(({ el }) => {
      if (!el) return;
      el.disabled = !mixed;
      if (!mixed) el.value = '';
    });
  }

  function enforceFixedSingleSelection(preferredEl) {
    if (getMode() !== 'fixed') return;
    if (preferredEl && preferredEl.checked) {
      typeEls.forEach((cb) => {
        if (cb !== preferredEl) cb.checked = false;
      });
      return;
    }
    const checked = typeEls.filter((cb) => cb.checked);
    if (checked.length > 1) {
      checked.slice(1).forEach((cb) => (cb.checked = false));
    }
  }

  typeEls.forEach((cb) => {
    cb.addEventListener('change', () => enforceFixedSingleSelection(cb));
  });

  weightFields.forEach(({ type, el }) => {
    if (!el) return;
    el.addEventListener('input', () => {
      if (getMode() !== 'mixed') return;
      const v = el.value.trim();
      if (!v) return;
      const num = Number(v);
      if (Number.isNaN(num) || num <= 0) return;
      const cb = typeEls.find((x) => x.value === type);
      if (cb) cb.checked = true;
    });
  });

  if (modeEl) {
    modeEl.addEventListener('change', () => {
      enforceFixedSingleSelection(null);
      syncWeightFields();
    });
  }

  enforceFixedSingleSelection(null);
  syncWeightFields();
}

function setupEvaluationUI() {
  const includeEl = $('#includeEvaluation');
  const filterEl = $('#filterByThreshold');
  const thresholdEl = $('#scoreThreshold');
  const evalMethodEl = $('#evaluationMethod');
  const hypothesisModeEl = $('#faithfulnessHypothesisMode');
  const hypothesisConcurrencyEl = $('#faithfulnessHypothesisMaxConcurrency');
  const hypothesisModeRow = hypothesisModeEl ? hypothesisModeEl.closest('label') : null;
  const hypothesisConcurrencyRow = hypothesisConcurrencyEl
    ? hypothesisConcurrencyEl.closest('label')
    : null;

  function sync() {
    const enabled = !!includeEl?.checked;
    const method = evalMethodEl?.value || 'llm';
    const useUnsupervisedSuite = enabled && method === 'unsupervised_f1';
    if (!enabled && filterEl) filterEl.checked = false;
    if (filterEl) filterEl.disabled = !enabled;
    if (thresholdEl) thresholdEl.disabled = !enabled;
    if (evalMethodEl) evalMethodEl.disabled = !enabled;
    if (hypothesisModeRow) hypothesisModeRow.style.display = useUnsupervisedSuite ? '' : 'none';
    if (hypothesisConcurrencyRow) {
      hypothesisConcurrencyRow.style.display = useUnsupervisedSuite ? '' : 'none';
    }
  }

  if (includeEl) includeEl.addEventListener('change', sync);
  if (evalMethodEl) evalMethodEl.addEventListener('change', sync);
  sync();
}

function setupChunkingModeUI() {
  const modeEl = $('#chunkingSplitType');
  const chunkSizeEl = $('#chunkSize');
  const summaryEl = $('#chunkingModeSummary');
  const panels = Array.from($$('.chunking-mode-panel') || []);

  const minEl = $('#chunkingTextSplitMinLength');
  const maxEl = $('#chunkingTextSplitMaxLength');
  const textSeparatorEl = $('#chunkingSeparator');
  const textOverlapEl = $('#chunkingChunkOverlap');
  const tokenOverlapEl = $('#chunkingChunkOverlapToken');
  const recursiveSeparatorsEl = $('#chunkingSeparators');
  const recursiveOverlapEl = $('#chunkingChunkOverlapRecursive');
  const codeLanguageEl = $('#chunkingSplitLanguage');
  const codeOverlapEl = $('#chunkingChunkOverlapCode');
  const customSeparatorEl = $('#chunkingCustomSeparator');

  function getChunkSize() {
    const raw = Number(chunkSizeEl?.value || '600');
    if (!Number.isFinite(raw) || raw <= 0) return 600;
    return Math.max(1, Math.floor(raw));
  }

  function getDefaultOverlap(size) {
    return Math.min(Math.max(0, Math.floor(size / 6)), Math.max(0, size - 1));
  }

  function normalizeMode() {
    const raw = String(modeEl?.value || '').trim().toLowerCase();
    return raw || 'markdown';
  }

  function applyDynamicPlaceholders() {
    const chunkSize = getChunkSize();
    const markdownMin = Math.max(120, Math.floor(chunkSize * 0.75));
    const markdownMax = Math.max(markdownMin, chunkSize);
    const overlap = getDefaultOverlap(chunkSize);

    if (minEl) minEl.placeholder = `默认 ${markdownMin}`;
    if (maxEl) maxEl.placeholder = `默认 ${markdownMax}`;
    if (textSeparatorEl) textSeparatorEl.placeholder = '默认 \\n\\n';
    if (textOverlapEl) textOverlapEl.placeholder = `默认 ${overlap}`;
    if (tokenOverlapEl) tokenOverlapEl.placeholder = `默认 ${overlap}`;
    if (recursiveSeparatorsEl) recursiveSeparatorsEl.placeholder = '默认 |, ##, >, -；支持逗号、换行或 JSON 数组';
    if (recursiveOverlapEl) recursiveOverlapEl.placeholder = `默认 ${overlap}`;
    if (codeLanguageEl) codeLanguageEl.placeholder = '默认 js；例如 python / java / markdown';
    if (codeOverlapEl) codeOverlapEl.placeholder = `默认 ${overlap}`;
    if (customSeparatorEl) customSeparatorEl.placeholder = '默认 ---';
  }

  function buildSummary(mode) {
    if (mode === 'text') {
      return '当前模式：按你指定的文本分隔符切分；适合固定段落边界明确、但没有标题层级的普通文本。未填写时使用默认分隔符。';
    }
    if (mode === 'token') {
      return '当前模式：按 token 长度切分；适合只关注模型上下文长度控制、不依赖自然结构边界的场景。未填写时使用默认重叠长度。';
    }
    if (mode === 'recursive') {
      return '当前模式：按分隔符递归切分；适合半结构化文本。会优先尝试你提供的分隔符列表，未填写时使用默认分隔符集合。';
    }
    if (mode === 'code') {
      return '当前模式：按代码结构切分；适合源码、脚本或 Markdown 代码型内容。未填写时默认按 js 规则处理。';
    }
    if (mode === 'custom') {
      return '当前模式：按自定义分隔符硬切；适合源文本里已经有稳定标记位的场景。未填写时默认分隔符是 ---。';
    }
    return '当前模式：按标题和大纲切分；适合政策、制度、合同、报告这类有明确层级结构的文本。未填写时自动使用默认长度参数。';
  }

  function sync() {
    if (!modeEl) return;
    const mode = normalizeMode();
    if (modeEl.value !== mode) modeEl.value = mode;
    panels.forEach((panel) => {
      panel.style.display = panel.dataset.chunkingMode === mode ? '' : 'none';
    });
    if (summaryEl) {
      summaryEl.textContent = buildSummary(mode);
    }
    applyDynamicPlaceholders();
  }

  if (modeEl) modeEl.addEventListener('change', sync);
  if (chunkSizeEl) chunkSizeEl.addEventListener('input', sync);
  sync();
}

function setupDocumentProcessingModeUI() {
  const modeEl = $('#pipelineProcessingMode');
  const hintEl = $('#pipelineProcessingModeHint');
  const integratedOptions = $('#integratedDocumentOptions');
  const ocrTimeoutField = $('#ocrTimeoutField');

  function sync() {
    const mode = String(modeEl?.value || 'standard').trim();
    const integrated = mode === 'integrated';
    if (integratedOptions) integratedOptions.hidden = !integrated;
    if (ocrTimeoutField) ocrTimeoutField.style.display = integrated ? 'none' : '';
    if (hintEl) {
      hintEl.textContent = integrated
        ? '使用 unified 内置的 dw 文档解析、图片理解和回填逻辑，再进入 hao 问答流水线。'
        : '使用当前激活的 OCR 配置解析 PDF、图片、OFD、DOCX、DOC 后进入 hao 问答流水线。';
    }
  }

  if (modeEl) modeEl.addEventListener('change', sync);
  sync();
}

async function handlePipelineSubmit(e) {
  e.preventDefault();
  const base = getApiBaseUrl();
  const fileInput = $('#pipelineFileInput');
  const statusEl = $('#pipelineStatus');
  const submitBtn =
    (e && e.submitter) ||
    (pipelineForm ? pipelineForm.querySelector('button[type="submit"]') : null);
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    if (statusEl) statusEl.textContent = '请先选择要上传的文件';
    return;
  }

  try {
    setBtnLoading(submitBtn, true);
    const formData = new FormData();
    Array.from(fileInput.files).forEach((f) => formData.append('files', f));
    const processingMode = String($('#pipelineProcessingMode')?.value || 'standard').trim();
    const useIntegratedPipeline = processingMode === 'integrated';
    const url =
      base +
      (useIntegratedPipeline
        ? '/batch-upload-integrated-document-pipeline'
        : '/batch-upload-complete-pipeline-with-evaluation');

    const qaPerChunk = $('#qaPerChunk')?.value || '1';
    const augmentPerQa = $('#augmentPerQa')?.value || '0';
    const chunkSize = $('#chunkSize')?.value || '600';
    const ocrTimeoutSeconds = $('#ocrTimeoutSeconds')?.value || '';
    const integratedOcrEnabled = $('#integratedOcrEnabled')?.checked !== false;
    const integratedOcrFailFast = $('#integratedOcrFailFast')?.checked === true;
    const integratedRemoveWatermark = $('#integratedRemoveWatermark')?.checked === true;
    const integratedWatermarkDpi = $('#integratedWatermarkDpi')?.value || '200';
    const integratedDocxStrategy = $('#integratedDocxStrategy')?.value || 'pdf';
    const imageContextSummaryMode = $('#imageContextSummaryMode')?.value || 'lightweight';
    const imageFitCheckEnabled = $('#imageFitCheckEnabled')?.checked !== false;
    const imageFitMinScore = $('#imageFitMinScore')?.value || '0.65';
    const qaDetailMode = $('#qaDetailMode')?.value || 'point';
    const knowledgeClassifier = $('#knowledgeClassifier')?.value || 'doc_level3_rule';
    const useCategoryPromptTemplates = $('#useCategoryPromptTemplates')?.checked !== false;
    const promptLanguage = $('#promptLanguage')?.value || 'auto';
    const questionTypeMode = $('#questionTypeMode')?.value || 'mixed';
    const questionTypeOptions = $$('input[name="questionTypeOption"]:checked');
    const selectedTypes = Array.from(questionTypeOptions || []).map((el) => el.value);
    const selectedTypeSet = new Set(selectedTypes);
    const questionTypes = selectedTypes.length ? selectedTypes.join(',') : '简答题';
    const weightFields = [
      { key: '简答题', el: $('#weightShort') },
      { key: '单选题', el: $('#weightChoice') },
      { key: '判断题', el: $('#weightJudge') },
      { key: '计算题', el: $('#weightCalc') },
    ];
    const weightObj = {};
    weightFields.forEach(({ key, el }) => {
      if (!el) return;
      if (!selectedTypeSet.has(key)) return;
      const v = el.value.trim();
      if (v === '') return;
      const num = Number(v);
      if (!Number.isNaN(num)) {
        weightObj[key] = num;
      }
    });
    const questionTypeWeights =
      Object.keys(weightObj).length > 0 ? JSON.stringify(weightObj) : '';
    const fewShotExamples = collectFewShotExamples();
    const includeEvaluation = $('#includeEvaluation')?.checked;
    const evaluationMethod = $('#evaluationMethod')?.value || 'llm';
    const faithfulnessHypothesisMode = $('#faithfulnessHypothesisMode')?.value || 'llm';
    const faithfulnessHypothesisMaxConcurrency =
      $('#faithfulnessHypothesisMaxConcurrency')?.value || '';
    const filterByThreshold = includeEvaluation ? $('#filterByThreshold')?.checked : false;
    const scoreThreshold = $('#scoreThreshold')?.value || '0.7';
    const enableVectorStorage = $('#enableVectorStorage')?.checked;
    const enableChunkStorage = $('#enableChunkStorage')?.checked;
    const chunkStorageFailFast = $('#chunkStorageFailFast')?.checked;
    const chunkingPrefixMaxDepth = $('#chunkingPrefixMaxDepth')?.value || '4';
    const chunkingSplitType = ($('#chunkingSplitType')?.value || 'markdown').trim() || 'markdown';
    const chunkingTextSplitMinLength = $('#chunkingTextSplitMinLength')?.value || '';
    const chunkingTextSplitMaxLength = $('#chunkingTextSplitMaxLength')?.value || '';
    const chunkingSeparator = $('#chunkingSeparator')?.value || '';
    const chunkingSeparators = $('#chunkingSeparators')?.value || '';
    const chunkingSplitLanguage = $('#chunkingSplitLanguage')?.value || '';
    const chunkingCustomSeparator = $('#chunkingCustomSeparator')?.value || '';
    const chunkingManualSplitPoints = $('#chunkingManualSplitPoints')?.value || '';
    const chunkingOverlapByMode = {
      text: $('#chunkingChunkOverlap')?.value || '',
      token: $('#chunkingChunkOverlapToken')?.value || '',
      recursive: $('#chunkingChunkOverlapRecursive')?.value || '',
      code: $('#chunkingChunkOverlapCode')?.value || '',
    };
    const chunkingChunkOverlap = chunkingOverlapByMode[chunkingSplitType] || '';
    const syncMode = $('#syncMode')?.checked;
    const maxConcurrency = $('#maxConcurrency')?.value || '';
    const evalMaxConcurrency = $('#evalMaxConcurrency')?.value || '';
    const chunkMaxConcurrency = $('#chunkMaxConcurrency')?.value || '';
    const augmentMaxConcurrency = $('#augmentMaxConcurrency')?.value || '';
    const saveModeEl = $('#saveMode');

    formData.append('qa_per_chunk', qaPerChunk);
    formData.append('augment_per_qa', augmentPerQa);
    formData.append('chunk_size', chunkSize);
    if (!useIntegratedPipeline && String(ocrTimeoutSeconds).trim()) {
      formData.append('ocr_timeout_seconds', String(ocrTimeoutSeconds).trim());
    }
    if (useIntegratedPipeline) {
      formData.append('ocr_enabled', integratedOcrEnabled ? 'true' : 'false');
      formData.append('ocr_fail_fast', integratedOcrFailFast ? 'true' : 'false');
      formData.append('remove_watermark', integratedRemoveWatermark ? 'true' : 'false');
      formData.append('watermark_dpi', String(integratedWatermarkDpi || '200'));
      formData.append('docx_strategy', integratedDocxStrategy);
      formData.append('image_context_summary_mode', imageContextSummaryMode);
      formData.append('image_fit_check_enabled', imageFitCheckEnabled ? 'true' : 'false');
      formData.append('image_fit_min_score', String(imageFitMinScore || '0.65'));
    }
    formData.append('qa_detail_mode', qaDetailMode);
    formData.append('knowledge_classifier', knowledgeClassifier);
    formData.append('use_category_prompt_templates', useCategoryPromptTemplates ? 'true' : 'false');
    formData.append('prompt_language', promptLanguage);
    formData.append('question_type_mode', questionTypeMode);
    if (questionTypes.trim()) {
      formData.append('question_types', questionTypes.trim());
    }
    if (questionTypeWeights.trim()) {
      formData.append('question_type_weights', questionTypeWeights.trim());
    }
    if (fewShotExamples && fewShotExamples.length) {
      formData.append('few_shot_examples', JSON.stringify(fewShotExamples));
    }
    formData.append('include_evaluation', includeEvaluation ? 'true' : 'false');
    formData.append('evaluation_method', evaluationMethod);
    if (evaluationMethod === 'unsupervised_f1') {
      formData.append('faithfulness_hypothesis_mode', faithfulnessHypothesisMode);
      if (String(faithfulnessHypothesisMaxConcurrency).trim()) {
        formData.append(
          'faithfulness_hypothesis_max_concurrency',
          String(faithfulnessHypothesisMaxConcurrency).trim(),
        );
      }
    }
    formData.append('filter_by_threshold', filterByThreshold ? 'true' : 'false');
    formData.append('score_threshold', scoreThreshold);
    formData.append('enable_vector_storage', enableVectorStorage ? 'true' : 'false');
    formData.append('enable_chunk_storage', enableChunkStorage ? 'true' : 'false');
    formData.append('chunk_storage_fail_fast', chunkStorageFailFast ? 'true' : 'false');
    formData.append('chunking_prefix_max_depth', String(chunkingPrefixMaxDepth || '4'));
    formData.append('chunking_split_type', chunkingSplitType);
    if (chunkingSplitType === 'markdown') {
      if (chunkingTextSplitMinLength.trim()) {
        formData.append('chunking_text_split_min_length', chunkingTextSplitMinLength.trim());
      }
      if (chunkingTextSplitMaxLength.trim()) {
        formData.append('chunking_text_split_max_length', chunkingTextSplitMaxLength.trim());
      }
    } else if (chunkingSplitType === 'text') {
      if (chunkingSeparator.trim()) {
        formData.append('chunking_separator', chunkingSeparator);
      }
      if (chunkingChunkOverlap.trim()) {
        formData.append('chunking_chunk_overlap', chunkingChunkOverlap.trim());
      }
    } else if (chunkingSplitType === 'token') {
      if (chunkingChunkOverlap.trim()) {
        formData.append('chunking_chunk_overlap', chunkingChunkOverlap.trim());
      }
    } else if (chunkingSplitType === 'recursive') {
      if (chunkingSeparators.trim()) {
        formData.append('chunking_separators', chunkingSeparators);
      }
      if (chunkingChunkOverlap.trim()) {
        formData.append('chunking_chunk_overlap', chunkingChunkOverlap.trim());
      }
    } else if (chunkingSplitType === 'code') {
      if (chunkingSplitLanguage.trim()) {
        formData.append('chunking_split_language', chunkingSplitLanguage.trim());
      }
      if (chunkingChunkOverlap.trim()) {
        formData.append('chunking_chunk_overlap', chunkingChunkOverlap.trim());
      }
    } else if (chunkingSplitType === 'custom') {
      if (chunkingCustomSeparator.trim()) {
        formData.append('chunking_custom_separator', chunkingCustomSeparator);
      }
    }
    if (chunkingManualSplitPoints.trim()) {
      formData.append('chunking_manual_split_points', chunkingManualSplitPoints);
    }
    formData.append('sync_mode', syncMode ? 'true' : 'false');
    formData.append('save_mode', saveModeEl?.value || 'separate');
    if (chunkMaxConcurrency.trim()) {
      formData.append('chunk_max_concurrency', chunkMaxConcurrency.trim());
    }
    if (augmentMaxConcurrency.trim()) {
      formData.append('augment_max_concurrency', augmentMaxConcurrency.trim());
    }
    if (evalMaxConcurrency.trim()) {
      formData.append('eval_max_concurrency', evalMaxConcurrency.trim());
    }
    if (maxConcurrency.trim()) {
      formData.append('max_concurrency', maxConcurrency.trim());
    }

    if (statusEl) statusEl.textContent = '正在提交任务…';
    const resp = await fetch(url, { method: 'POST', body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      if (statusEl) {
        statusEl.textContent = '调用失败：' + (detail || resp.statusText);
      }
      return;
    }
    updatePipelineStatusView(data);
    if (data.task_id) {
      applyPipelineStatus(data, { base, taskId: data.task_id });
      startTaskPolling(base, data.task_id);
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = '调用出错：' + String(err);
  } finally {
    setBtnLoading(submitBtn, false);
  }
}

function updatePipelineStatusView(status) {
  const statusEl = $('#pipelineStatus');
  if (!statusEl) return;
  statusEl.textContent = JSON.stringify(status, null, 2);
}

function updatePipelineTaskHint(text) {
  const hintEl = $('#pipelineTaskHint');
  if (!hintEl) return;
  hintEl.textContent =
    String(text || '').trim() ||
    '页面会记住最近查看或正在运行的流水线任务。上方“终止当前/指定任务”会优先使用这里填写的 task_id；如果任务已经结束或变成僵尸记录，可直接在下方任务列表里删除记录。';
}

function clearPipelineOutputsView() {
  lastCsvPath = null;
  lastPipelineOutputs = [];
  const panel = document.querySelector('#pipelineOutputsList');
  if (panel) panel.innerHTML = '';
}

function applyPipelineStatus(status, { base = '', taskId = '' } = {}) {
  const normalizedTaskId = String(taskId || status?.task_id || '').trim();
  if (!normalizedTaskId) return;
  setTaskSelection(normalizedTaskId, { active: !isTaskTerminal(status?.status) });
  updatePipelineStatusView(status);
  rememberTask({ task_id: normalizedTaskId, ...status });
  renderPipelineTaskHistory();

  const outputs = Array.isArray(status?.outputs) ? status.outputs : [];
  if (outputs.length && base) {
    handlePipelineOutputs(base, outputs);
  } else {
    clearPipelineOutputsView();
  }

  const statusText = String(status?.status || '').trim() || 'unknown';
  const msg = String(status?.message || '').trim();
  updatePipelineTaskHint(
    `当前查看 task_id=${normalizedTaskId}（${statusText}${msg ? `，${msg}` : ''}）。上方“终止当前/指定任务”会优先使用这个 task_id。`,
  );
}

async function loadTaskStatusById(taskId, { resumePolling = true, silent = false } = {}) {
  const normalized = String(taskId || '').trim();
  if (!normalized) {
    if (!silent) notify('请先输入 task_id', 'warning');
    return;
  }
  const base = getApiBaseUrl();
  const statusEl = $('#pipelineStatus');
  if (!resumePolling && currentTaskPoller) {
    clearInterval(currentTaskPoller);
    currentTaskPoller = null;
  }
  try {
    if (!silent) updatePipelineTaskHint(`正在加载 task_id=${normalized}…`);
    const result = await fetchTaskStatusMaybe(base, normalized);
    if (!result.found) {
      removeRecentTask(normalized);
      renderPipelineTaskHistory();
      if (!silent) {
        if (statusEl) statusEl.textContent = `任务不存在：${normalized}`;
        updatePipelineTaskHint(`task_id=${normalized} 不存在，已从本地最近任务列表移除。`);
      }
      return null;
    }
    const data = result.data;
    applyPipelineStatus(data, { base, taskId: normalized });
    if (!isTaskTerminal(data?.status) && resumePolling) {
      startTaskPolling(base, normalized);
    } else if (isTaskTerminal(data?.status) && currentTaskPoller) {
      clearInterval(currentTaskPoller);
      currentTaskPoller = null;
    }
  } catch (err) {
    if (!silent) {
      const statusEl = $('#pipelineStatus');
      if (statusEl) statusEl.textContent = '查询任务失败：' + String(err);
      updatePipelineTaskHint(`查询 task_id=${normalized} 失败：${String(err)}`);
    }
    throw err;
  }
}

function startTaskPolling(base, taskId) {
  const statusEl = $('#pipelineStatus');
  if (!taskId) return;
  setTaskSelection(taskId, { active: true });
  const intervalMs = 2000;
  if (currentTaskPoller) {
    clearInterval(currentTaskPoller);
    currentTaskPoller = null;
  }
  const timerId = setInterval(async () => {
    try {
      const result = await fetchTaskStatusMaybe(base, taskId);
      if (!result.found) {
        clearInterval(timerId);
        currentTaskPoller = null;
        removeRecentTask(taskId);
        renderPipelineTaskHistory();
        if (statusEl) {
          statusEl.textContent += `\n任务不存在，已从最近任务列表移除：${taskId}`;
        }
        updatePipelineTaskHint(`task_id=${taskId} 不存在，已从本地最近任务列表移除。`);
        return;
      }
      const data = result.data;
      applyPipelineStatus(data, { base, taskId });
      if (
        data &&
        data.status &&
        ['completed', 'failed', 'canceled'].includes(data.status)
      ) {
        clearInterval(timerId);
        currentTaskPoller = null;
      }
    } catch (err) {
      clearInterval(timerId);
      currentTaskPoller = null;
      if (statusEl) statusEl.textContent += '\n轮询异常：' + String(err);
    }
  }, intervalMs);
  currentTaskPoller = timerId;
}

async function handleCancelTask(explicitTaskId) {
  const base = getApiBaseUrl();
  const statusEl = $('#pipelineStatus');
  const inputId = explicitTaskId || $('#taskIdInput')?.value.trim();
  const taskId = inputId || lastTaskId;
  if (!taskId) {
    if (statusEl) statusEl.textContent = '请先填入任务ID或执行过一次任务再终止';
    return;
  }
  if (currentTaskPoller) {
    clearInterval(currentTaskPoller);
    currentTaskPoller = null;
  }
  try {
    if (statusEl) statusEl.textContent = `正在终止任务 ${taskId}…`;
    const resp = await fetch(
      `${base}/cancel-task/${encodeURIComponent(taskId)}`,
      { method: 'POST' },
    );
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      const message = String(detail || resp.statusText || '终止失败');
      const extra =
        message.includes('不存在') || message.includes('已结束') || message.includes('已完成')
          ? '；如果只是清理历史记录，可直接点任务列表里的“删除记录”'
          : '';
      if (statusEl) statusEl.textContent = '终止失败：' + message + extra;
      return;
    }
    if (statusEl) {
      statusEl.textContent = `终止请求已发送：${JSON.stringify(data)}`;
    }
    rememberTask({
      task_id: taskId,
      status: 'canceled',
      message: String(data?.status || 'cancel_requested'),
      updated_at: new Date().toISOString(),
    });
    renderPipelineTaskHistory();
    updatePipelineTaskHint(`已发送终止请求：task_id=${taskId}`);
    await loadTaskStatusById(taskId, { resumePolling: false, silent: true }).catch(() => {});
  } catch (err) {
    if (statusEl) statusEl.textContent = '终止出错：' + String(err);
  }
}

function handlePipelineOutputs(base, outputs) {
  if (!Array.isArray(outputs) || !outputs.length) return;
  lastPipelineOutputs = outputs.slice();
  const usable = outputs.filter(
    (o) =>
      o &&
      (o.consolidated_json ||
        o.consolidated_csv ||
        String(o.history_source || '').trim().toLowerCase() === 'milvus'),
  );
  const first = usable[0] || null;
  if (first) {
    lastCsvPath = first.consolidated_csv || null;
    if (first.consolidated_json) {
      loadConsolidatedFromServer(base, first.consolidated_json);
    } else if (String(first.history_source || '') === 'milvus') {
      loadPipelineMilvusHistory(base, first.milvus_task_id || lastTaskId, {
        sourceFile: first.mode === 'separate' ? first.source_file || '' : '',
      });
    }
  } else {
    lastCsvPath = null;
    lastTaskQaItems = [];
    updateTaskQaFileFilter([]);
    renderMeta({
      outputs: outputs.map((output) => ({
        source_file: output?.source_file || output?.core_file || output?.filename || '',
        history_source: output?.history_source || '',
        artifacts_deleted: output?.artifacts_deleted || false,
        artifacts_expire_at: output?.artifacts_expire_at || null,
      })),
      message: '当前任务的临时 JSON/CSV 已不可用。已入库结果请到管理页查询，未入库且已过期的结果无法恢复。',
    });
    const qaResults = $('#qaResults');
    if (qaResults) {
      qaResults.textContent =
        '当前任务没有可直接预览的 JSON/CSV 结果。已入库请到管理页查询，未入库且已清理的结果无法再下载。';
    }
  }
  renderPipelineOutputsList(base, outputs);
}

async function loadPipelineMilvusHistory(base, taskId, { sourceFile = '' } = {}) {
  const normalizedTaskId = String(taskId || '').trim();
  if (!normalizedTaskId) {
    renderMeta({ error: 'Milvus task_id 不存在，无法加载数据库历史' });
    return;
  }
  try {
    if (sourceFile) {
      const params = new URLSearchParams();
      params.set('original_filename', sourceFile);
      params.set('task_id', normalizedTaskId);
      params.set('page', '1');
      params.set('page_size', '200');
      params.set('include_details', 'true');
      const data = await fetchJson(`${base}/file-qa?${params.toString()}`);
      renderMeta({
        files: data.files || [],
        counts: data.counts || {},
        category_distribution: data.category_distribution || {},
        pagination: data.pagination || {},
        filters: data.filters || {},
        history_source: 'milvus',
      });
      renderQaResults({ items: normalizeItems(data.items || []) }, true);
      return;
    }
    const params = new URLSearchParams();
    params.set('only_filtered', 'false');
    params.set('min_avg_score', '0');
    params.set('page', '1');
    params.set('page_size', '200');
    params.set('include_raw_responses', 'false');
    const data = await fetchJson(`${base}/task-qa/${encodeURIComponent(normalizedTaskId)}?${params.toString()}`);
    const items = normalizeItems(data.items || []);
    lastTaskQaItems = items;
    updateTaskQaFileFilter(
      Array.from(
        new Set(
          items
            .map((item) => item.original_filename || '')
            .filter((name) => String(name || '').trim()),
        ),
      ),
    );
    renderMeta({
      task_info: data.task_info || {},
      model_info: data.model_info || {},
      counts: data.counts || {},
      category_distribution: data.category_distribution || {},
      filter_info: data.filter_info || {},
      pagination: data.pagination || {},
      history_source: 'milvus',
    });
    renderQaResults({ items }, true);
  } catch (err) {
    renderMeta({ error: `加载数据库历史失败：${String(err)}` });
  }
}

function renderPipelineOutputsList(base, outputs) {
  const panel = document.querySelector('#pipelineOutputsList');
  if (!panel) return;
  panel.innerHTML = '';
  if (!Array.isArray(outputs) || !outputs.length) return;

  const list = document.createElement('ul');
  list.className = 'pipeline-output-list';

  outputs.forEach((o, idx) => {
    if (!o || typeof o !== 'object') return;
    const li = document.createElement('li');
    li.className = 'pipeline-output-item';

    const label = document.createElement('span');
    const parts = [];
    const sourceName =
      o.core_file ||
      o.source_file ||
      o.filename ||
      (Array.isArray(o.source_files) && o.source_files.length
        ? `合并 ${o.source_files.length} 个文件`
        : '');
    if (sourceName) {
      parts.push(`源文件: ${sourceName}`);
    }
    if (typeof o.qa_pairs === 'number') {
      parts.push(`问答数: ${o.qa_pairs}`);
    }
    const timing = o.timing || {};
    const tParts = [];
    if (typeof timing.ocr_seconds === 'number') tParts.push(`OCR ${fmtSeconds(timing.ocr_seconds)}`);
    if (typeof timing.generation_seconds === 'number')
      tParts.push(`生成 ${fmtSeconds(timing.generation_seconds)}`);
    if (typeof timing.unsupervised_seconds === 'number')
      tParts.push(`无监督评估 ${fmtSeconds(timing.unsupervised_seconds)}`);
    if (typeof timing.evaluation_seconds === 'number')
      tParts.push(`评估 ${fmtSeconds(timing.evaluation_seconds)}`);
    if (tParts.length) parts.push(`耗时: ${tParts.join(' / ')}`);
    label.textContent = parts.join(' | ') || `文件 #${idx + 1}`;
    li.appendChild(label);

    let actionCount = 0;
    if (o.consolidated_json) {
      const btnJson = document.createElement('button');
      btnJson.type = 'button';
      btnJson.textContent = '查看 JSON';
      btnJson.addEventListener('click', () => {
        loadConsolidatedFromServer(base, o.consolidated_json);
      });
      li.appendChild(btnJson);
      actionCount += 1;
    }

    if (String(o.history_source || '').trim().toLowerCase() === 'milvus') {
      const btnMilvus = document.createElement('button');
      btnMilvus.type = 'button';
      btnMilvus.textContent = '查看入库记录';
      btnMilvus.addEventListener('click', () => {
        loadPipelineMilvusHistory(base, o.milvus_task_id || lastTaskId, {
          sourceFile: o.mode === 'separate' ? o.source_file || '' : '',
        });
      });
      li.appendChild(btnMilvus);
      actionCount += 1;
    }

    if (o.consolidated_csv) {
      const btnCsv = document.createElement('button');
      btnCsv.type = 'button';
      btnCsv.textContent = '下载 CSV';
      btnCsv.addEventListener('click', () => {
        const parts2 = String(o.consolidated_csv).split('/');
        const fileName = parts2[parts2.length - 1];
        const dlPath = `outputs/${fileName}`;
        const url = `${base}/download/${dlPath}`;
        window.open(url, '_blank');
      });
      li.appendChild(btnCsv);
      actionCount += 1;
    }

    const hint = document.createElement('div');
    hint.className = 'muted';
    if (String(o.history_source || '').trim().toLowerCase() === 'milvus') {
      hint.textContent = '该结果已入库；如需筛选、审核或精确查看，请到管理页按 task_id / 原文件名查询。';
    } else if (actionCount === 0 && o.artifacts_deleted) {
      hint.textContent = '临时 JSON/CSV 已被清理；如果没有入库，就不能再从这里下载。';
    } else if (actionCount === 0) {
      hint.textContent = '当前没有可直接查看的文件结果；如需查库，请到管理页按 task_id / 原文件名查询。';
    }
    if (hint.textContent) {
      li.appendChild(hint);
    }

    list.appendChild(li);
  });

  panel.appendChild(list);
}

async function hydratePipelineRuntime() {
  renderPipelineTaskHistory();
  const recentTasks = await refreshPipelineHistory({ silent: true });
  const page = getRuntimePageState();
  const rememberedId =
    String(page.activeTaskId || '').trim() ||
    String(page.selectedTaskId || '').trim() ||
    String($('#taskIdInput')?.value || '').trim();
  const recentList = Array.isArray(recentTasks) ? recentTasks : [];
  const existingIds = new Set(recentList.map((task) => String(task.task_id || '').trim()).filter(Boolean));
  const targetId = rememberedId && existingIds.has(rememberedId)
    ? rememberedId
    : (recentList[0] && String(recentList[0].task_id || '').trim()) || '';
  if (!targetId) {
    updatePipelineTaskHint();
    return;
  }
  try {
    await loadTaskStatusById(targetId, { resumePolling: true, silent: true });
  } catch {
    renderPipelineTaskHistory();
  }
}

function fmtSeconds(val) {
  const n = Number(val);
  if (Number.isNaN(n)) return '';
  if (n >= 10) return n.toFixed(1) + 's';
  return n.toFixed(2) + 's';
}

async function loadConsolidatedFromServer(base, jsonPath) {
  if (!jsonPath) return;
  try {
    // 后端 download 接口约定：如果路径以 "outputs/" 开头，则只取 basename
    // consolidated_json 里通常是 "qa/outputs/xxx.json"，这里统一转换一下
    const parts = String(jsonPath).split('/');
    const fileName = parts[parts.length - 1];
    const dlPath = `outputs/${fileName}`;
    const resp = await fetch(`${base}/download/${dlPath}`);
    if (!resp.ok) {
      console.error('加载合并结果失败', resp.status, resp.statusText);
      renderMeta({ error: `下载合并结果失败：${resp.status} ${resp.statusText}` });
      return;
    }
    const text = await resp.text();
    try {
      const data = JSON.parse(text);
      renderMetaFromConsolidated(data);
      renderFromConsolidated(data);
    } catch (parseErr) {
      console.error('解析合并 JSON 失败', parseErr);
      renderMeta({ error: '合并结果文件不是有效 JSON，无法预览' });
    }
  } catch (err) {
    console.error('解析合并 JSON 失败', err);
    renderMeta({ error: '合并结果加载异常' });
  }
}
