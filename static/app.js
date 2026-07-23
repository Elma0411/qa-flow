window.__QA_UI_APPJS_READY__ = true;
window.__QA_UI_APPJS_VERSION__ = '2026-07-03-6';

let currentDwJobPoller = null;
const MODULE_SETTINGS_CACHE_KEY = 'qa_flow_module_settings_v1';
const WORKSPACE_MODAL_SIZE_KEY = 'qa_flow_workspace_modal_size_v1';
const MODULE_SECRET_FIELD_IDS = new Set(['cfgKey', 'dwVlmApiKey', 'integratedVlmApiKey']);
let activeSettingsModule = null;
const settingsModules = new Map();
let pipelineTaskActiveTab = 'status';
let pipelineRawJsonOpen = false;
let pipelineOutputsAutoShownForTaskId = '';
const pipelineTaskFirstSeenAtMs = Object.create(null);
let latestPipelineStatus = null;
let currentConsolidatedJsonPath = '';

const qaReviewState = {
  enabled: false,
  taskId: '',
  sourceFile: '',
  jsonPath: '',
  expireAt: null,
  output: null,
  items: [],
  selectedIds: new Set(),
  selectAllTask: false,
};

const DOC_STAGE_ORDER = [
  'doc_input',
  'doc_text_read',
  'dw_input',
  'dw_text_read',
  'input',
  'format_routing',
  'format_conversion',
  'doc_ocr',
  'dw_ocr',
  'ocr',
  'watermark',
  'ocr_predict',
  'page_collect',
  'image_extract',
  'markdown_merge',
  'image_replacement',
  'ocr_output',
  'doc_marker',
  'dw_marker',
  'doc_pre_chunking',
  'dw_pre_chunking',
  'doc_chunk_summary',
  'dw_chunk_summary',
  'image_classification',
  'doc_image_analysis',
  'dw_image_analysis',
  'image_analysis',
  'doc_placement',
  'dw_placement',
  'text_integration',
  'document_output',
  'doc_handoff',
  'dw_handoff',
  'knowledge_tagging',
  'chunking',
  'chunk_storage',
  'qa_generation',
  'qa_augmentation',
  'unsupervised_evaluation',
  'evaluation',
  'completed',
  'doc_error',
  'dw_error',
  'error',
];

const DOC_STAGE_LABELS = {
  doc_input: '输入准备',
  doc_text_read: '文本读取',
  dw_input: '输入准备',
  dw_text_read: '文本读取',
  input: '输入保存',
  format_routing: '格式识别',
  format_conversion: '格式转换',
  doc_ocr: 'OCR/抽取',
  dw_ocr: 'OCR/抽取',
  ocr: 'OCR',
  watermark: '去水印',
  ocr_predict: 'OCR 模型',
  page_collect: '页面块收集',
  image_extract: '图片提取',
  markdown_merge: 'Markdown 合并',
  image_replacement: '高质量裁图',
  ocr_output: 'OCR 输出',
  doc_marker: '图片标记',
  dw_marker: '图片标记',
  doc_pre_chunking: '预切块',
  dw_pre_chunking: '预切块',
  doc_chunk_summary: 'chunk 摘要',
  dw_chunk_summary: 'chunk 摘要',
  image_classification: '图片分类',
  doc_image_analysis: '图片理解',
  dw_image_analysis: '图片理解',
  image_analysis: '图片理解',
  doc_placement: '契合度判断',
  dw_placement: '契合度判断',
  text_integration: '文本整合',
  document_output: '结果输出',
  doc_handoff: '移交问答',
  dw_handoff: '移交问答',
  knowledge_tagging: '知识分类',
  chunking: '文本切分',
  chunk_storage: '保存 chunk 溯源索引',
  qa_generation: '问答生成',
  qa_augmentation: '问句增广',
  unsupervised_evaluation: '无监督评估',
  evaluation: '评估过滤',
  completed: '完成',
  doc_error: '错误',
  dw_error: '错误',
  error: '错误',
};

const DOCUMENT_PARSE_STAGE_KEYS = [
  'doc_input',
  'doc_text_read',
  'dw_input',
  'dw_text_read',
  'input',
  'format_routing',
  'format_conversion',
  'doc_ocr',
  'dw_ocr',
  'ocr',
  'doc_marker',
  'dw_marker',
  'doc_pre_chunking',
  'dw_pre_chunking',
  'doc_chunk_summary',
  'dw_chunk_summary',
  'image_classification',
  'doc_image_analysis',
  'dw_image_analysis',
  'image_analysis',
  'doc_placement',
  'dw_placement',
  'text_integration',
  'document_output',
  'doc_handoff',
  'dw_handoff',
];
const DOCUMENT_PARSE_FALLBACK_STAGE_KEYS = [
  'watermark',
  'ocr_predict',
  'page_collect',
  'image_extract',
  'markdown_merge',
  'image_replacement',
  'ocr_output',
];
const EVALUATION_STAGE_KEYS = ['unsupervised_evaluation', 'evaluation'];
const STORAGE_OUTPUT_STAGE_KEYS = ['chunk_storage'];

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
setupFormPresentation();
applyCompactFieldCopy();
ui().enhanceFileInputs?.();
setupDwDocumentPanel();
setupModuleSettingsUI();
setupWorkbenchRedesign();
const cancelTaskBtn = $('#cancelTaskBtn');
if (cancelTaskBtn) {
  cancelTaskBtn.addEventListener('click', handleCancelCurrentTask);
}
const cancelTaskByIdBtn = $('#btnCancelTaskById');
if (cancelTaskByIdBtn) {
  cancelTaskByIdBtn.addEventListener('click', handleCancelInputTask);
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
hydrateDwRuntime().catch(() => {});

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
  const unsupervisedBatchSizeEl = $('#unsupervisedBatchSize');
  const faithfulnessNliModelEl = $('#faithfulnessNliModel');
  const answerabilityQaModelEl = $('#answerabilityQaModel');
  const coverageEmbeddingModelEl = $('#coverageEmbeddingModel');
  const hypothesisModeEl = $('#faithfulnessHypothesisMode');
  const hypothesisConcurrencyEl = $('#faithfulnessHypothesisMaxConcurrency');
  const unsupervisedRows = [
    unsupervisedBatchSizeEl,
    faithfulnessNliModelEl,
    answerabilityQaModelEl,
    coverageEmbeddingModelEl,
    hypothesisModeEl,
    hypothesisConcurrencyEl,
  ]
    .map((el) => (el ? el.closest('label') : null))
    .filter(Boolean);

  function sync() {
    const enabled = !!includeEl?.checked;
    const method = evalMethodEl?.value || 'llm';
    const useUnsupervisedSuite = enabled && method === 'unsupervised_f1';
    if (!enabled && filterEl) filterEl.checked = false;
    if (filterEl) filterEl.disabled = !enabled;
    if (thresholdEl) thresholdEl.disabled = !enabled;
    if (evalMethodEl) evalMethodEl.disabled = !enabled;
    unsupervisedRows.forEach((row) => {
      row.style.display = useUnsupervisedSuite ? '' : 'none';
    });
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
        ? '使用内置文档解析、图片理解和回填逻辑，再进入问答流水线。'
        : '使用当前激活的 OCR 配置解析 PDF、图片、OFD、DOCX、DOC 后进入问答流水线。';
    }
  }

  if (modeEl) modeEl.addEventListener('change', sync);
  sync();
}

function setupFormPresentation() {
  const labels = Array.from($$('label') || []);
  labels.forEach((label) => {
    if (!label || !label.querySelector) return;
    if (label.querySelector('input[type="checkbox"]')) {
      label.classList.add('checkbox-field');
    }
    if (label.querySelector('input[type="file"]')) {
      label.classList.add('file-field');
    }
  });

  const forms = Array.from($$('main form') || []);
  forms.forEach((form) => {
    if (!form.classList.contains('admin-form-stack') && !form.classList.contains('admin-grid-form')) {
      form.classList.add('console-form');
    }
  });

  const detailsList = Array.from($$('main details') || []);
  detailsList.forEach((details) => {
    details.classList.add('console-details');
  });
}

function stripFieldKeyText(text) {
  return String(text || '')
    .replace(/[（(][a-zA-Z0-9_., /\-]+[）)]/g, '')
    .replace(/\b[a-zA-Z][a-zA-Z0-9_]{2,}\b/g, (word) => {
      const keep = new Set(['API', 'Base', 'URL', 'JSON', 'OCR', 'VLM', 'LLM', 'DOCX', 'DOC', 'PDF']);
      return keep.has(word) ? word : '';
    })
    .replace(/\s+/g, ' ')
    .replace(/\s+([，。；：])/g, '$1')
    .trim();
}

function compactLabelCopy(label) {
  if (!label || label.dataset.copyCompacted === '1') return;
  label.dataset.copyCompacted = '1';
  const controls = Array.from(label.querySelectorAll('input, select, textarea'));
  if (!controls.length) return;
  const small = label.querySelector('small');
  const controlSet = new Set(controls);
  const textNodes = [];
  label.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE && String(node.textContent || '').trim()) {
      textNodes.push(node);
    }
  });
  if (!textNodes.length) return;
  const original = textNodes.map((node) => node.textContent || '').join(' ').trim();
  const primaryId = controls.find((control) => control.id)?.id || '';
  const friendlyById = {
    qaTotalLimit: '总题数上限',
    qaTotalLimitScope: '上限范围',
    llmMaxConcurrentRequests: 'LLM/VLM 请求并发',
    chunkMaxAttempts: 'unit 尝试次数',
    chunkMaxConcurrency: 'unit 并发',
    maxConcurrency: '文件并发',
    evalMaxConcurrency: '评估并发',
    augmentMaxConcurrency: '增广并发',
    scoreThreshold: '过滤阈值',
    ocrTimeoutSeconds: 'OCR 超时',
  };
  const compact = friendlyById[primaryId] || stripFieldKeyText(original);
  if (!compact || compact.length < 2) return;
  textNodes.forEach((node, idx) => {
    node.textContent = idx === 0 ? compact + ' ' : '';
  });
  controls.forEach((control) => {
    if (!controlSet.has(control)) return;
    if (control.id && !control.dataset.fieldKey) control.dataset.fieldKey = control.id;
  });
  if (small && small.textContent) {
    const short = String(small.textContent || '').replace(/\s+/g, ' ').trim();
    if (short.length > 64) small.textContent = short.slice(0, 62) + '...';
  }
}

function applyCompactFieldCopy(root) {
  const scope = root && root.querySelectorAll ? root : document;
  Array.from(scope.querySelectorAll('label') || []).forEach(compactLabelCopy);
}

// ---------- 模块化参数抽屉 ----------

function nodeForField(id) {
  const el = document.getElementById(String(id || ''));
  if (!el) return null;
  return el.closest('label, .inline-checkbox-group, details, .llm-config-split, .form-grid') || el;
}

function closestNodeForSelector(selector, closestSelector) {
  const el = document.querySelector(selector);
  if (!el) return null;
  return closestSelector ? el.closest(closestSelector) : el;
}

function resolveModuleNode(def) {
  if (!def) return null;
  if (def.heading) {
    const heading = document.createElement('div');
    heading.className = 'settings-section-title';
    heading.textContent = String(def.heading || '');
    return heading;
  }
  if (def.node) return def.node;
  if (def.field) return nodeForField(def.field);
  if (def.selector && def.closest) return closestNodeForSelector(def.selector, def.closest);
  if (def.selector) return document.querySelector(def.selector);
  return null;
}

function readModuleCache() {
  try {
    const raw = window.localStorage.getItem(MODULE_SETTINGS_CACHE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeModuleCache(cache) {
  try {
    window.localStorage.setItem(MODULE_SETTINGS_CACHE_KEY, JSON.stringify(cache || {}));
  } catch {
    // ignore
  }
}

function readFieldValue(el) {
  if (!el) return null;
  const type = String(el.type || '').toLowerCase();
  if (type === 'checkbox') return !!el.checked;
  if (String(el.tagName || '').toLowerCase() === 'select' || el.value !== undefined) {
    return String(el.value ?? '');
  }
  return null;
}

function readFieldDefault(el) {
  if (!el) return null;
  const type = String(el.type || '').toLowerCase();
  const tag = String(el.tagName || '').toLowerCase();
  if (type === 'checkbox') return !!el.defaultChecked;
  if (tag === 'select') {
    const selected = Array.from(el.options || []).find((option) => option.defaultSelected);
    return selected ? String(selected.value ?? '') : String(el.options?.[0]?.value ?? '');
  }
  if (tag === 'textarea' || tag === 'input') return String(el.defaultValue ?? '');
  return readFieldValue(el);
}

function applyFieldValue(el, value) {
  if (!el || value === undefined || value === null) return;
  const type = String(el.type || '').toLowerCase();
  if (type === 'checkbox') {
    el.checked = !!value;
  } else {
    el.value = String(value);
  }
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
}

function collectModuleFields(module) {
  const fields = [];
  module.nodes.forEach((node) => {
    if (!node || !node.querySelectorAll) return;
    node.querySelectorAll('input[id], select[id], textarea[id]').forEach((el) => {
      if (!el.id || String(el.type || '').toLowerCase() === 'file') return;
      fields.push(el);
    });
  });
  return Array.from(new Map(fields.map((el) => [el.id, el])).values());
}

function moduleValues(module, { defaults = false } = {}) {
  const values = {};
  module.fields.forEach((el) => {
    if (!el.id || MODULE_SECRET_FIELD_IDS.has(el.id)) return;
    values[el.id] = defaults ? readFieldDefault(el) : readFieldValue(el);
  });
  return values;
}

function applyModuleValues(module, values) {
  if (!module || !values || typeof values !== 'object') return;
  module.fields.forEach((el) => {
    if (!el.id || MODULE_SECRET_FIELD_IDS.has(el.id)) return;
    if (!Object.prototype.hasOwnProperty.call(values, el.id)) return;
    applyFieldValue(el, values[el.id]);
    persistUiField(el);
  });
}

function saveModuleValues(module) {
  if (!module) return;
  if (!module.cacheKey) {
    updateModuleCard(module);
    return;
  }
  const cache = readModuleCache();
  cache[module.cacheKey] = {
    updated_at: new Date().toISOString(),
    values: moduleValues(module),
  };
  writeModuleCache(cache);
  updateModuleCard(module);
}

function restoreModuleDefaults(module, options = {}) {
  if (!module) return;
  applyModuleValues(module, moduleValues(module, { defaults: true }));
  module.fields.forEach((el) => {
    if (!el.id || !MODULE_SECRET_FIELD_IDS.has(el.id)) return;
    applyFieldValue(el, readFieldDefault(el));
  });
  saveModuleValues(module);
  if (!options.silent) notify(`已恢复 ${module.title} 默认值`, 'success');
}

function restoreModuleCache(modules) {
  const cache = readModuleCache();
  modules.forEach((module) => {
    const entry = cache[module.cacheKey];
    if (!entry || !entry.values || typeof entry.values !== 'object') return;
    applyModuleValues(module, entry.values);
  });
}

function workspaceModalBounds() {
  const maxWidth = Math.max(320, window.innerWidth - 32);
  const maxHeight = Math.max(360, window.innerHeight - 32);
  const minWidth = Math.min(maxWidth, Math.max(320, Math.min(720, window.innerWidth - 32)));
  const minHeight = Math.min(maxHeight, Math.max(320, Math.min(480, window.innerHeight - 32)));
  return { minWidth, minHeight, maxWidth, maxHeight };
}

function clampWorkspaceModalSize(size) {
  const bounds = workspaceModalBounds();
  const width = Math.min(bounds.maxWidth, Math.max(bounds.minWidth, Number(size?.width) || bounds.maxWidth));
  const height = Math.min(bounds.maxHeight, Math.max(bounds.minHeight, Number(size?.height) || bounds.maxHeight));
  return { width, height };
}

function defaultWorkspaceModalSize() {
  return clampWorkspaceModalSize({
    width: Math.min(1240, window.innerWidth - 32),
    height: Math.min(900, Math.round(window.innerHeight * 0.92)),
  });
}

function readWorkspaceModalSize() {
  try {
    const raw = window.localStorage.getItem(WORKSPACE_MODAL_SIZE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === 'object' ? clampWorkspaceModalSize(parsed) : null;
  } catch {
    return null;
  }
}

function writeWorkspaceModalSize(size) {
  try {
    window.localStorage.setItem(WORKSPACE_MODAL_SIZE_KEY, JSON.stringify(clampWorkspaceModalSize(size)));
  } catch {
    // ignore
  }
}

function applyWorkspaceModalSize(drawer, size) {
  if (!drawer) return;
  const next = clampWorkspaceModalSize(size || readWorkspaceModalSize() || defaultWorkspaceModalSize());
  drawer.style.width = `${Math.round(next.width)}px`;
  drawer.style.height = `${Math.round(next.height)}px`;
  drawer.style.maxHeight = 'none';
}

function clearWorkspaceModalSizeStyle(drawer) {
  if (!drawer) return;
  drawer.style.width = '';
  drawer.style.height = '';
  drawer.style.maxHeight = '';
}

function setupWorkspaceModalResizer(drawer) {
  const handle = $('#moduleSettingsResizeHandle');
  if (!drawer || !handle || handle.dataset.resizeBound === '1') return;
  handle.dataset.resizeBound = '1';

  const finishResize = () => {
    drawer.classList.remove('is-resizing');
    document.body.classList.remove('workspace-modal-resizing');
    writeWorkspaceModalSize({
      width: drawer.getBoundingClientRect().width,
      height: drawer.getBoundingClientRect().height,
    });
  };

  handle.addEventListener('pointerdown', (e) => {
    if (!drawer.classList.contains('settings-modal--workspace')) return;
    e.preventDefault();
    const start = drawer.getBoundingClientRect();
    const startX = e.clientX;
    const startY = e.clientY;
    drawer.classList.add('is-resizing');
    document.body.classList.add('workspace-modal-resizing');
    if (typeof handle.setPointerCapture === 'function') {
      try {
        handle.setPointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    }

    const onMove = (evt) => {
      applyWorkspaceModalSize(drawer, {
        width: start.width + (evt.clientX - startX),
        height: start.height + (evt.clientY - startY),
      });
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      finishResize();
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
  });

  handle.addEventListener('dblclick', () => {
    try {
      window.localStorage.removeItem(WORKSPACE_MODAL_SIZE_KEY);
    } catch {
      // ignore
    }
    applyWorkspaceModalSize(drawer, defaultWorkspaceModalSize());
  });

  handle.addEventListener('keydown', (e) => {
    if (!drawer.classList.contains('settings-modal--workspace')) return;
    const step = e.shiftKey ? 96 : 32;
    const rect = drawer.getBoundingClientRect();
    let width = rect.width;
    let height = rect.height;
    if (e.key === 'ArrowRight') width += step;
    else if (e.key === 'ArrowLeft') width -= step;
    else if (e.key === 'ArrowDown') height += step;
    else if (e.key === 'ArrowUp') height -= step;
    else if (e.key === 'Home') {
      try {
        window.localStorage.removeItem(WORKSPACE_MODAL_SIZE_KEY);
      } catch {
        // ignore
      }
      applyWorkspaceModalSize(drawer, defaultWorkspaceModalSize());
      e.preventDefault();
      return;
    } else {
      return;
    }
    e.preventDefault();
    applyWorkspaceModalSize(drawer, { width, height });
    writeWorkspaceModalSize({ width: drawer.getBoundingClientRect().width, height: drawer.getBoundingClientRect().height });
  });

  window.addEventListener('resize', () => {
    if (!drawer.classList.contains('settings-modal--workspace') || drawer.hidden) return;
    applyWorkspaceModalSize(drawer, {
      width: drawer.getBoundingClientRect().width,
      height: drawer.getBoundingClientRect().height,
    });
  });
}

function createSettingsDrawer() {
  let overlay = $('#moduleSettingsOverlay');
  let drawer = $('#moduleSettingsDrawer');
  if (overlay && drawer) return { overlay, drawer };

  overlay = document.createElement('div');
  overlay.id = 'moduleSettingsOverlay';
  overlay.className = 'drawer-overlay module-settings-overlay';
  overlay.hidden = true;
  document.body.appendChild(overlay);

  drawer = document.createElement('aside');
  drawer.id = 'moduleSettingsDrawer';
  drawer.className = 'settings-modal';
  drawer.setAttribute('role', 'dialog');
  drawer.setAttribute('aria-modal', 'true');
  drawer.setAttribute('aria-labelledby', 'moduleSettingsTitle');
  drawer.setAttribute('aria-hidden', 'true');
  drawer.hidden = true;
  drawer.innerHTML = [
    '<div class="settings-modal-header">',
    '<div>',
    '<h2 class="settings-modal-title" id="moduleSettingsTitle">参数配置</h2>',
    '<p class="settings-modal-desc" id="moduleSettingsDesc"></p>',
    '</div>',
    '<button type="button" class="icon-btn settings-modal-close" id="moduleSettingsClose" aria-label="关闭配置" title="关闭">',
    '<span aria-hidden="true">×</span>',
    '</button>',
    '</div>',
    '<div class="settings-modal-body" id="moduleSettingsBody"></div>',
    '<div class="settings-modal-footer">',
    '<button type="button" class="secondary" id="moduleSettingsReset">恢复默认</button>',
    '<button type="button" class="secondary" id="moduleSettingsCloseSecondary">取消</button>',
    '<button type="button" id="moduleSettingsApply">保存</button>',
    '</div>',
    '<button type="button" class="settings-modal-resize-handle" id="moduleSettingsResizeHandle" aria-label="拖动调整弹窗大小，双击恢复默认尺寸" title="拖动调整大小，双击恢复默认尺寸">',
    '<span aria-hidden="true"></span>',
    '</button>',
  ].join('');
  document.body.appendChild(drawer);
  setupWorkspaceModalResizer(drawer);

  overlay.addEventListener('click', closeSettingsDrawer);
  $('#moduleSettingsClose')?.addEventListener('click', closeSettingsDrawer);
  $('#moduleSettingsCloseSecondary')?.addEventListener('click', closeSettingsDrawer);
  $('#moduleSettingsApply')?.addEventListener('click', () => {
    if (activeSettingsModule) saveActiveSettingsSession(activeSettingsModule);
    closeSettingsDrawer();
  });
  $('#moduleSettingsReset')?.addEventListener('click', () => {
    if (activeSettingsModule) restoreActiveSettingsSession(activeSettingsModule);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && activeSettingsModule) closeSettingsDrawer();
  });
  return { overlay, drawer };
}

function saveActiveSettingsSession(module) {
  if (!module) return;
  if (Array.isArray(module.groupedModules) && module.groupedModules.length) {
    module.groupedModules.forEach((item) => saveModuleValues(item));
    notify(`已保存 ${module.title}`, 'success');
    return;
  }
  saveModuleValues(module);
}

function restoreActiveSettingsSession(module) {
  if (!module) return;
  if (Array.isArray(module.groupedModules) && module.groupedModules.length) {
    module.groupedModules.forEach((item) => restoreModuleDefaults(item, { silent: true }));
    notify(`已恢复 ${module.title} 默认值`, 'success');
    return;
  }
  restoreModuleDefaults(module);
}

function groupModuleNodes(module) {
  const groups = [];
  let current = { title: '配置项', nodes: [] };
  (module.nodes || []).forEach((node) => {
    if (node && node.classList && node.classList.contains('settings-section-title')) {
      if (current.nodes.length) groups.push(current);
      current = { title: String(node.textContent || '配置项').trim() || '配置项', nodes: [node] };
      return;
    }
    current.nodes.push(node);
  });
  if (current.nodes.length) groups.push(current);
  return groups.length ? groups : [{ title: '配置项', nodes: module.nodes || [] }];
}

function renderSettingsModalBody(module, body) {
  if (Array.isArray(module?.groupedModules) && module.groupedModules.length) {
    renderGroupedSettingsModalBody(module, body);
    return;
  }
  const groups = groupModuleNodes(module);
  body.replaceChildren();
  if (groups.length <= 1) {
    const panel = document.createElement('div');
    panel.className = 'settings-tab-panel is-active';
    groups[0].nodes.forEach((node) => panel.appendChild(node));
    body.appendChild(panel);
    applyCompactFieldCopy(panel);
    return;
  }

  const layout = document.createElement('div');
  layout.className = 'settings-tab-layout';
  const nav = document.createElement('div');
  nav.className = 'settings-tab-nav';
  nav.setAttribute('role', 'tablist');
  const panels = document.createElement('div');
  panels.className = 'settings-tab-panels';

  groups.forEach((group, index) => {
    const tabId = `settingsTab${index}`;
    const panelId = `settingsPanel${index}`;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `settings-tab-btn${index === 0 ? ' is-active' : ''}`;
    button.textContent = group.title;
    button.setAttribute('role', 'tab');
    button.setAttribute('id', tabId);
    button.setAttribute('aria-controls', panelId);
    button.setAttribute('aria-selected', index === 0 ? 'true' : 'false');
    const panel = document.createElement('div');
    panel.className = `settings-tab-panel${index === 0 ? ' is-active' : ''}`;
    panel.id = panelId;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', tabId);
    group.nodes.forEach((node) => panel.appendChild(node));
    button.addEventListener('click', () => {
      nav.querySelectorAll('.settings-tab-btn').forEach((btn) => {
        btn.classList.toggle('is-active', btn === button);
        btn.setAttribute('aria-selected', btn === button ? 'true' : 'false');
      });
      panels.querySelectorAll('.settings-tab-panel').forEach((item) => {
        item.classList.toggle('is-active', item === panel);
      });
    });
    nav.appendChild(button);
    panels.appendChild(panel);
  });

  layout.append(nav, panels);
  body.appendChild(layout);
  applyCompactFieldCopy(layout);
}

function renderGroupedSettingsModalBody(session, body) {
  const modules = (session.groupedModules || []).filter((module) => module && module.nodes && module.nodes.length);
  body.replaceChildren();
  if (!modules.length) {
    const empty = document.createElement('div');
    empty.className = 'settings-empty';
    empty.textContent = '暂无可配置参数。';
    body.appendChild(empty);
    return;
  }

  const layout = document.createElement('div');
  layout.className = 'settings-tab-layout settings-tab-layout--grouped';
  const nav = document.createElement('div');
  nav.className = 'settings-tab-nav';
  nav.setAttribute('role', 'tablist');
  const panels = document.createElement('div');
  panels.className = 'settings-tab-panels';

  modules.forEach((module, index) => {
    const tabId = `settingsGroupTab${index}`;
    const panelId = `settingsGroupPanel${index}`;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `settings-tab-btn${index === 0 ? ' is-active' : ''}`;
    button.textContent = module.title || `配置 ${index + 1}`;
    button.setAttribute('role', 'tab');
    button.setAttribute('id', tabId);
    button.setAttribute('aria-controls', panelId);
    button.setAttribute('aria-selected', index === 0 ? 'true' : 'false');

    const panel = document.createElement('div');
    panel.className = `settings-tab-panel${index === 0 ? ' is-active' : ''}`;
    panel.id = panelId;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', tabId);

    const head = document.createElement('div');
    head.className = 'settings-panel-head';
    const title = document.createElement('h3');
    title.textContent = module.title || `配置 ${index + 1}`;
    const desc = document.createElement('p');
    desc.textContent = module.description || '';
    head.append(title, desc);
    panel.appendChild(head);
    module.nodes.forEach((node) => panel.appendChild(node));

    button.addEventListener('click', () => {
      nav.querySelectorAll('.settings-tab-btn').forEach((btn) => {
        btn.classList.toggle('is-active', btn === button);
        btn.setAttribute('aria-selected', btn === button ? 'true' : 'false');
      });
      panels.querySelectorAll('.settings-tab-panel').forEach((item) => {
        item.classList.toggle('is-active', item === panel);
      });
    });

    nav.appendChild(button);
    panels.appendChild(panel);
  });

  layout.append(nav, panels);
  body.appendChild(layout);
  applyCompactFieldCopy(layout);
}

function openSettingsDrawer(module) {
  if (!module) return;
  if (activeSettingsModule && activeSettingsModule !== module) closeSettingsDrawer();
  const { overlay, drawer } = createSettingsDrawer();
  const body = $('#moduleSettingsBody');
  const title = $('#moduleSettingsTitle');
  const desc = $('#moduleSettingsDesc');
  if (!body) return;

  activeSettingsModule = module;
  renderSettingsModalBody(module, body);
  if (title) title.textContent = module.title;
  if (desc) desc.textContent = module.description || '';
  drawer.classList.toggle('settings-modal--workspace', module.workspaceModal === true);
  if (module.workspaceModal === true) applyWorkspaceModalSize(drawer);
  else clearWorkspaceModalSizeStyle(drawer);

  overlay.hidden = false;
  drawer.hidden = false;
  requestAnimationFrame(() => {
    overlay.classList.add('is-open');
    drawer.classList.add('is-open');
    drawer.setAttribute('aria-hidden', 'false');
    document.body.classList.add('drawer-open');
  });
}

function closeSettingsDrawer() {
  const module = activeSettingsModule;
  const overlay = $('#moduleSettingsOverlay');
  const drawer = $('#moduleSettingsDrawer');
  if (!module || !overlay || !drawer) return;
  if (Array.isArray(module.groupedModules) && module.groupedModules.length) {
    module.groupedModules.forEach((item) => {
      if (!item || !item.bank || !Array.isArray(item.nodes)) return;
      item.nodes.forEach((node) => item.bank.appendChild(node));
    });
  } else {
    module.nodes.forEach((node) => module.bank.appendChild(node));
  }
  activeSettingsModule = null;
  overlay.classList.remove('is-open');
  drawer.classList.remove('is-open');
  drawer.classList.remove('settings-modal--workspace');
  drawer.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('drawer-open');
  window.setTimeout(() => {
    if (!activeSettingsModule) {
      overlay.hidden = true;
      drawer.hidden = true;
    }
  }, 180);
}

function openAppModal(module) {
  openSettingsDrawer(module);
}

function closeAppModal() {
  closeSettingsDrawer();
}

function modulesByKeys(keys) {
  return (keys || []).map((key) => settingsModules.get(key)).filter(Boolean);
}

function openSettingsGroup(title, description, moduleKeys) {
  const modules = modulesByKeys(moduleKeys);
  if (!modules.length) return;
  openSettingsDrawer({
    title,
    description,
    groupedModules: modules,
    nodes: [],
    fields: [],
    bank: null,
  });
}

function openPipelineSettingsModal() {
  openSettingsGroup(
    '任务设置',
    '按生成流程分区配置文档解析、切分、生成、评估、性能和输出。保存后只影响后续提交的任务。',
    [
      'pipeline.document',
      'pipeline.chunking',
      'pipeline.generation',
      'pipeline.evaluation',
      'pipeline.performance',
      'pipeline.output',
    ],
  );
}

function createModuleCard(module) {
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'module-settings-card';
  card.setAttribute('aria-label', `打开${module.title}配置`);
  card.innerHTML = [
    '<span class="module-card-icon" aria-hidden="true"></span>',
    '<span class="module-card-copy">',
    '<span class="module-card-kicker"></span>',
    '<strong class="module-card-title"></strong>',
    '<span class="module-card-summary"></span>',
    '</span>',
  ].join('');
  card.querySelector('.module-card-icon').textContent = module.icon || '';
  card.querySelector('.module-card-kicker').textContent = module.kicker || '配置';
  card.querySelector('.module-card-title').textContent = module.title;
  card.addEventListener('click', () => openSettingsDrawer(module));
  module.card = card;
  updateModuleCard(module);
  return card;
}

function updateModuleCard(module) {
  if (!module || !module.card) return;
  const summary = module.card.querySelector('.module-card-summary');
  if (summary) summary.textContent = module.summary ? module.summary() : '点击配置';
}

function registerModuleConsole(options) {
  const host = options.host;
  const after = options.after;
  if (!host || !after) return [];
  const shell = document.createElement('div');
  shell.className = 'module-settings-shell';
  const title = document.createElement('div');
  title.className = 'module-settings-shell-title';
  title.textContent = options.title || '参数模块';
  const grid = document.createElement('div');
  grid.className = 'module-settings-grid';
  shell.append(title, grid);
  after.insertAdjacentElement('afterend', shell);

  const registered = [];
  options.modules.forEach((def) => {
    const nodes = (def.nodes || []).map(resolveModuleNode).filter(Boolean);
    const uniqueNodes = Array.from(new Set(nodes));
    if (!uniqueNodes.length) return;

    const bank = document.createElement('div');
    bank.className = 'module-field-bank';
    bank.hidden = true;
    bank.dataset.moduleKey = def.key;
    host.appendChild(bank);
    uniqueNodes.forEach((node) => bank.appendChild(node));

    const module = {
      ...def,
      cacheKey: `${options.scope}.${def.key}`,
      bank,
      nodes: uniqueNodes,
      fields: [],
      card: null,
    };
    module.fields = collectModuleFields(module);
    module.fields.forEach((el) => {
      const onChange = () => saveModuleValues(module);
      el.addEventListener('change', onChange);
      const type = String(el.type || '').toLowerCase();
      if (type === 'text' || type === 'number' || type === 'search' || String(el.tagName || '').toLowerCase() === 'textarea') {
        el.addEventListener('input', onChange);
      }
    });
    settingsModules.set(module.cacheKey, module);
    grid.appendChild(createModuleCard(module));
    registered.push(module);
  });

  restoreModuleCache(registered);
  registered.forEach(updateModuleCard);
  return registered;
}

function checkedQuestionTypesSummary() {
  const types = Array.from($$('input[name="questionTypeOption"]:checked') || []).map((el) => el.value);
  return types.length ? types.join('/') : '简答题';
}

function pipelineEvaluationSummary() {
  if ($('#includeEvaluation')?.checked === false) return '关闭';
  const method = $('#evaluationMethod')?.value || 'llm';
  const filter = $('#filterByThreshold')?.checked ? `过滤 ${$('#scoreThreshold')?.value || 0.7}` : '不过滤';
  return `${method} / ${filter}`;
}

function pipelineStorageSummary() {
  const vector = $('#enableVectorStorage')?.checked ? '自动入库' : '人工审阅';
  const chunks = $('#enableChunkStorage')?.checked ? '保存溯源' : '不保存溯源';
  const wait = $('#syncMode')?.checked ? '等待完成' : '异步返回';
  return `${vector} / ${chunks} / ${wait}`;
}

function setupPipelineModuleConsole() {
  const form = $('#pipelineForm');
  const modeNode = nodeForField('pipelineProcessingMode');
  const qaNode = nodeForField('qaTotalLimit');
  const runActions = $('#cancelTaskBtn')?.closest('.actions-row');
  if (!form || !modeNode || !qaNode) return;
  modeNode.insertAdjacentElement('afterend', qaNode);
  registerModuleConsole({
    scope: 'pipeline',
    host: form,
    after: qaNode,
    title: '任务设置',
    modules: [
      {
        key: 'document',
        icon: 'D',
        kicker: '文档',
        title: '文档解析',
        description: '配置 OCR、一体流程文档解析、图片理解与 VLM 覆盖参数。',
        nodes: [
          { selector: '#ocrTimeoutField' },
          { selector: '#integratedDocumentOptions' },
        ],
        summary: () => {
          const mode = $('#pipelineProcessingMode')?.value === 'integrated' ? '一体流程' : '标准 OCR';
          const image = $('#integratedEnableImageAnalysis')?.checked === false ? '图片关' : '图片开';
          return `${mode} / ${image}`;
        },
      },
      {
        key: 'chunking',
        icon: 'C',
        kicker: '切块',
        title: '切分',
        description: '设置 chunk 大小、切分模式、标题前缀和手工切分点。',
        nodes: [
          { selector: '#chunkingSplitType', closest: 'details' },
        ],
        summary: () => `${$('#chunkingSplitType')?.value || 'markdown'} / ${$('#chunkSize')?.value || 600} 字`,
      },
      {
        key: 'generation',
        icon: 'Q',
        kicker: '生成',
        title: '问答生成',
        description: '配置 QA 粒度、分类器、题型、few-shot、增广条数和尝试次数。',
        nodes: [
          { field: 'qaTotalLimitScope' },
          { field: 'augmentPerQa' },
          { field: 'qaDetailMode' },
          { field: 'knowledgeClassifier' },
          { field: 'useCategoryPromptTemplates' },
          { field: 'promptLanguage' },
          { field: 'questionTypeMode' },
          { selector: 'input[name="questionTypeOption"]', closest: '.inline-checkbox-group' },
          { selector: 'input[name="qtWeight"]', closest: '.inline-checkbox-group' },
          { selector: '#fewShotList', closest: '.inline-checkbox-group' },
          { field: 'chunkMaxAttempts' },
        ],
        summary: () => `${checkedQuestionTypesSummary()} / 上限 ${$('#qaTotalLimit')?.value || 20} / 尝试 ${$('#chunkMaxAttempts')?.value || 2}`,
      },
      {
        key: 'retrieval',
        icon: 'R',
        kicker: '检索',
        title: '检索证据',
        description: '配置候选问题检索 evidence 的排序模式、topK、轻量重排、权重和系统允许的最大证据范围。',
        nodes: [
          { field: 'retrievalMode' },
          { field: 'semanticTopK' },
          { field: 'rerankTopN' },
          { field: 'hybridWeightDense' },
          { field: 'hybridWeightLexical' },
          { field: 'retrievalStructureWeight' },
          { field: 'answerScopePolicy' },
        ],
        summary: () => `${$('#retrievalMode')?.value || 'hybrid'} / topK ${$('#semanticTopK')?.value || 3} / ${$('#answerScopePolicy')?.value || 'source_primary'}`,
      },
      {
        key: 'evaluation',
        icon: 'E',
        kicker: '评估',
        title: '评估过滤',
        description: '配置问答评估、无监督评估、忠实度陈述句生成和分数过滤。',
        nodes: [
          { field: 'includeEvaluation' },
          { field: 'evaluationMethod' },
          { field: 'unsupervisedBatchSize' },
          { field: 'faithfulnessNliModel' },
          { field: 'answerabilityQaModel' },
          { field: 'coverageEmbeddingModel' },
          { field: 'faithfulnessHypothesisMode' },
          { field: 'faithfulnessHypothesisMaxConcurrency' },
          { field: 'filterByThreshold' },
          { field: 'scoreThreshold' },
        ],
        summary: () => pipelineEvaluationSummary(),
      },
      {
        key: 'performance',
        icon: 'P',
        kicker: '性能',
        title: '性能并发',
        description: '配置文件级、chunk 级、评估、增广和 LLM/VLM API 请求并发。',
        nodes: [
          { field: 'maxConcurrency' },
          { field: 'evalMaxConcurrency' },
          { field: 'chunkMaxConcurrency' },
          { field: 'llmMaxConcurrentRequests' },
          { field: 'augmentMaxConcurrency' },
        ],
        summary: () => {
          const chunk = String($('#chunkMaxConcurrency')?.value || '').trim() || '8';
          const llm = String($('#llmMaxConcurrentRequests')?.value || '').trim() || 'Docker 默认';
          return `chunk ${chunk} / LLM ${llm}`;
        },
      },
      {
        key: 'output',
        icon: 'O',
        kicker: '输出',
        title: '存储输出',
        description: '配置 QA 自动入库、chunk 溯源索引、同步等待和批量保存方式。',
        nodes: [
          { field: 'enableVectorStorage' },
          { field: 'enableChunkStorage' },
          { field: 'chunkStorageFailFast' },
          { field: 'syncMode' },
          { field: 'saveMode' },
        ],
        summary: () => pipelineStorageSummary(),
      },
    ],
  });
  if (runActions) form.appendChild(runActions);
}

function setupConfigSectionModules() {
  const llmSection = $('#cfgName')?.closest('section');
  const envPanel = llmSection?.querySelector('.env-check-panel');
  if (llmSection && envPanel) {
    registerModuleConsole({
      scope: 'llm',
      host: llmSection,
      after: envPanel,
      title: 'LLM 设置',
      modules: [
        {
          key: 'saved',
          icon: 'L',
          kicker: '模型',
          title: 'LLM 配置',
          description: '新增、编辑、激活或删除后端 LLM 配置。API Key 不会写入本地缓存。',
          nodes: [{ selector: '#cfgName', closest: '.llm-config-split' }],
          summary: () => String($('#cfgActive')?.textContent || '').trim() || '选择或新增模型配置',
        },
        {
          key: 'debug',
          icon: 'T',
          kicker: '测试',
          title: 'LLM 响应测试',
          description: '发送一条轻量测试请求，查看模型连通性和原始返回。',
          nodes: [{ selector: '#llmDebugPrompt', closest: 'details' }],
          summary: () => `超时 ${$('#llmDebugTimeoutSeconds')?.value || 30}s / ${$('#llmDebugResponseFormat')?.value || 'json_object'}`,
        },
      ],
    });
  }

  const ocrSection = $('#ocrCfgName')?.closest('section');
  const ocrAfter = ocrSection?.querySelector('.form-grid');
  if (ocrSection && ocrAfter) {
    registerModuleConsole({
      scope: 'ocr',
      host: ocrSection,
      after: ocrAfter,
      title: 'OCR 设置',
      modules: [
        {
          key: 'saved',
          icon: 'O',
          kicker: 'OCR',
          title: 'OCR 配置',
          description: '新增、编辑、激活、删除或测试后端 OCR 配置。',
          nodes: [{ selector: '#ocrCfgName', closest: '.llm-config-split' }],
          summary: () => String($('#ocrCfgActive')?.textContent || '').trim() || `${$('#ocrCfgProvider')?.value || 'batch_ocr'} / ${$('#ocrCfgTimeoutSeconds')?.value || 600}s`,
        },
      ],
    });
  }
}

function setupDwModuleConsole() {
  const form = $('#dwJobForm');
  const fileNode = nodeForField('dwFileInput');
  if (!form || !fileNode) return;
  registerModuleConsole({
    scope: 'document_worker',
    host: form,
    after: fileNode,
    title: '文档解析参数',
    modules: [
      {
        key: 'output',
        icon: 'F',
        kicker: '格式',
        title: '输出与格式',
        description: '配置文档解析输出格式和 DOCX/DOC 处理策略。',
        nodes: [
          { field: 'dwOutputFormat' },
          { field: 'dwDocxStrategy' },
        ],
        summary: () => `${$('#dwOutputFormat')?.value || 'text'} / ${$('#dwDocxStrategy')?.value || 'pdf'}`,
      },
      {
        key: 'image',
        icon: 'I',
        kicker: '图片',
        title: '图片理解',
        description: '控制图片理解、图片分类、去水印、高质量裁图和 VLM API 开关。',
        nodes: [
          { field: 'dwEnableImageAnalysis' },
          { field: 'dwEnableClassification' },
          { field: 'dwClassificationThreshold' },
          { field: 'dwRemoveWatermark' },
          { field: 'dwWatermarkDpi' },
          { field: 'dwReplaceImages' },
          { field: 'dwUseApi' },
        ],
        summary: () => {
          const image = $('#dwEnableImageAnalysis')?.checked ? '图片理解开' : '图片理解关';
          const api = $('#dwUseApi')?.checked ? 'API 开' : 'API 关';
          return `${image} / ${api} / 阈值 ${$('#dwClassificationThreshold')?.value || 0.9}`;
        },
      },
      {
        key: 'vlm',
        icon: 'V',
        kicker: 'VLM',
        title: 'VLM 覆盖参数',
        description: '按本次文档解析任务覆盖 VLM API Base、模型、类型和版本。API Key 不会写入本地缓存。',
        nodes: [{ selector: '#dwVlmApiBase', closest: 'details' }],
        summary: () => String($('#dwVlmModelName')?.value || '').trim() || '使用后端默认 VLM',
      },
    ],
  });
}

function setupModuleSettingsUI() {
  createSettingsDrawer();
  setupConfigSectionModules();
  setupDwModuleConsole();
  setupPipelineModuleConsole();
  const refreshCards = () => settingsModules.forEach((module) => updateModuleCard(module));
  document.addEventListener('change', refreshCards);
  document.addEventListener('input', refreshCards);
}

function sectionByField(id) {
  return document.getElementById(String(id || ''))?.closest('section.card') || null;
}

function setSectionMeta(section, meta) {
  if (!section || !meta) return;
  section.classList.add('workbench-section');
  if (meta.key) section.dataset.sectionKey = meta.key;
  const h2 = section.querySelector('h2');
  if (h2 && meta.title) h2.textContent = meta.title;
  const hint = section.querySelector(':scope > .hint, :scope > p');
  if (hint && meta.description) hint.textContent = meta.description;
}

function makeStatusDot(type) {
  const dot = document.createElement('span');
  dot.className = `status-dot status-dot--${type || 'idle'}`;
  dot.setAttribute('aria-hidden', 'true');
  return dot;
}

function createSummaryItem(label, getValue, options = {}) {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'summary-item';
  item.innerHTML = [
    '<span class="summary-label"></span>',
    '<strong class="summary-value"></strong>',
    '<span class="summary-hint"></span>',
  ].join('');
  item.querySelector('.summary-label').textContent = label;
  item.querySelector('.summary-hint').textContent = options.hint || '';
  const render = () => {
    const value = typeof getValue === 'function' ? getValue() : getValue;
    item.querySelector('.summary-value').textContent = String(value || '未设置');
  };
  render();
  item.addEventListener('click', () => {
    if (options.moduleKey) {
      const module = settingsModules.get(options.moduleKey);
      if (module) openSettingsDrawer(module);
    }
    if (options.target) {
      const target = typeof options.target === 'function' ? options.target() : options.target;
      if (target && typeof target.scrollIntoView === 'function') {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  });
  item.render = render;
  return item;
}

function createSummaryChip(label, getValue, options = {}) {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'summary-chip';
  item.innerHTML = [
    '<span class="summary-chip-label"></span>',
    '<strong class="summary-chip-value"></strong>',
  ].join('');
  item.querySelector('.summary-chip-label').textContent = label;
  const render = () => {
    const value = typeof getValue === 'function' ? getValue() : getValue;
    item.querySelector('.summary-chip-value').textContent = String(value || '未设置');
  };
  render();
  item.addEventListener('click', () => {
    if (options.moduleKey) {
      const module = settingsModules.get(options.moduleKey);
      if (module) openSettingsDrawer(module);
    }
    if (options.openPipelineSettings) openPipelineSettingsModal();
  });
  item.render = render;
  return item;
}

function observeSummarySources(refreshSummary) {
  if (typeof refreshSummary !== 'function' || typeof MutationObserver !== 'function') return;
  ['cfgActive', 'ocrCfgActive'].forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    const observer = new MutationObserver(refreshSummary);
    observer.observe(node, { childList: true, subtree: true, characterData: true });
  });
}

function moveSectionAfter(section, anchor) {
  if (!section || !anchor || !anchor.parentNode) return;
  anchor.insertAdjacentElement('afterend', section);
}

function setSectionTitle(section, title, description) {
  if (!section) return;
  const heading = section.querySelector('h2, h3');
  if (heading && title) heading.textContent = title;
  const hint = section.querySelector(':scope > .hint, :scope > p');
  if (hint && description !== undefined) hint.textContent = description;
}

function createPanelShell(title, description, className) {
  const section = document.createElement('section');
  section.className = className || 'card';
  const head = document.createElement('div');
  head.className = 'section-head';
  const copy = document.createElement('div');
  const h2 = document.createElement('h2');
  h2.textContent = title || '';
  const p = document.createElement('p');
  p.className = 'hint';
  p.textContent = description || '';
  copy.append(h2, p);
  head.appendChild(copy);
  section.appendChild(head);
  return section;
}

function activateTaskTab(name) {
  const root = $('#pipelineTaskWorkspace');
  if (!root) return;
  const target = String(name || 'status');
  pipelineTaskActiveTab = target;
  root.querySelectorAll('.task-tab-btn').forEach((btn) => {
    const active = btn.dataset.taskTab === target;
    btn.classList.toggle('is-active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  root.querySelectorAll('.task-tab-panel').forEach((panel) => {
    panel.classList.toggle('is-active', panel.dataset.taskPanel === target);
  });
}

function setupTaskWorkspace(pipelineSection) {
  if (!pipelineSection || $('#pipelineTaskWorkspace')) return null;
  const taskConsole = pipelineSection.querySelector('.task-console');
  const outputsPanel = $('#pipelineOutputsPanel');
  const statusPanel = $('#pipelineStatus');
  if (!taskConsole || !outputsPanel || !statusPanel) return null;

  const workspace = createPanelShell(
    '当前任务',
    '提交、恢复或查询任务后，在这里看进度、耗时、输出文件和历史记录。',
    'card current-task-card',
  );
  workspace.id = 'pipelineTaskWorkspace';
  const head = workspace.querySelector('.section-head');
  const tabs = document.createElement('div');
  tabs.className = 'task-tab-nav';
  tabs.setAttribute('role', 'tablist');
  [
    ['status', '进度耗时'],
    ['outputs', '输出文件'],
    ['history', '任务历史'],
  ].forEach(([key, label], index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `task-tab-btn${index === 0 ? ' is-active' : ''}`;
    btn.dataset.taskTab = key;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', index === 0 ? 'true' : 'false');
    btn.textContent = label;
    btn.addEventListener('click', () => activateTaskTab(key));
    tabs.appendChild(btn);
  });
  if (head) head.appendChild(tabs);

  const panels = document.createElement('div');
  panels.className = 'task-tab-panels';

  const statusWrap = document.createElement('div');
  statusWrap.className = 'task-tab-panel is-active';
  statusWrap.dataset.taskPanel = 'status';
  const statusEmpty = document.createElement('div');
  statusEmpty.className = 'current-task-empty';
  statusEmpty.textContent = '等待提交任务。开始执行后会显示进度、阶段耗时和 generation unit 明细。';
  statusPanel.appendChild(statusEmpty);
  statusWrap.appendChild(statusPanel);

  const outputsWrap = document.createElement('div');
  outputsWrap.className = 'task-tab-panel';
  outputsWrap.dataset.taskPanel = 'outputs';
  setSectionTitle(outputsPanel, '输出文件', '查看合并 JSON、下载 CSV 或跳转入库记录。');
  outputsWrap.appendChild(outputsPanel);

  const historyWrap = document.createElement('div');
  historyWrap.className = 'task-tab-panel';
  historyWrap.dataset.taskPanel = 'history';
  historyWrap.appendChild(taskConsole);

  panels.append(statusWrap, outputsWrap, historyWrap);
  workspace.appendChild(panels);
  pipelineSection.insertAdjacentElement('afterend', workspace);
  return workspace;
}

function setupReviewWorkspace({ anchor, chunkSection, qaResultsSection }) {
  if (!anchor || $('#pipelineReviewWorkspace')) return null;
  const review = createPanelShell(
    '结果检查',
    '围绕当前任务做溯源检查和 QA 预览。没有任务结果时这里保持安静。',
    'card review-workspace is-empty',
  );
  review.id = 'pipelineReviewWorkspace';
  const body = document.createElement('div');
  body.className = 'review-grid';

  if (chunkSection) {
    setSectionTitle(chunkSection, 'Chunk 溯源', '按当前 task_id 查看树结构，点击 leaf chunk 检查正文和对应 QA。');
    chunkSection.classList.add('review-panel', 'review-panel--chunk');
    body.appendChild(chunkSection);
  }
  if (qaResultsSection) {
    setSectionTitle(qaResultsSection, 'QA 预览', '');
    qaResultsSection.classList.add('review-panel', 'review-panel--qa');
    body.appendChild(qaResultsSection);
  }

  const empty = document.createElement('div');
  empty.className = 'review-empty';
  empty.textContent = '任务完成或加载结果后会在这里显示溯源和问答预览。';
  review.append(empty, body);
  anchor.insertAdjacentElement('afterend', review);
  const refresh = () => refreshReviewWorkspaceState();
  ['qaResults', 'chunkTree'].forEach((id) => {
    const node = document.getElementById(id);
    if (!node || typeof MutationObserver !== 'function') return;
    const observer = new MutationObserver(refresh);
    observer.observe(node, { childList: true, subtree: true, characterData: true });
  });
  ['taskIdInput', 'chunkTaskId'].forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    node.addEventListener('input', refresh);
    node.addEventListener('change', refresh);
  });
  refreshReviewWorkspaceState();
  return review;
}

function refreshReviewWorkspaceState() {
  const review = $('#pipelineReviewWorkspace');
  if (!review) return;
  const selectedTask = String($('#taskIdInput')?.value || $('#chunkTaskId')?.value || lastTaskId || '').trim();
  const qaText = String($('#qaResults')?.textContent || '').trim();
  const chunkText = String($('#chunkTree')?.textContent || '').trim();
  const hasQa = !!qaText && qaText !== '没有结果';
  const hasChunk = !!chunkText;
  review.classList.toggle('is-empty', !(selectedTask || hasQa || hasChunk));
}

function makeToolCard(title, description, actionLabel, onClick, options = {}) {
  const card = document.createElement('article');
  card.className = 'tool-card';
  const icon = document.createElement('span');
  icon.className = 'tool-card-icon';
  icon.textContent = options.icon || title.slice(0, 1);
  const copy = document.createElement('div');
  copy.className = 'tool-card-copy';
  const h3 = document.createElement('h3');
  h3.textContent = title;
  const p = document.createElement('p');
  p.textContent = description || '';
  copy.append(h3, p);
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = options.primary ? '' : 'secondary';
  btn.textContent = actionLabel || '打开';
  btn.addEventListener('click', onClick);
  card.append(icon, copy, btn);
  return card;
}

function openSectionModal(title, description, section, options = {}) {
  if (!section) return;
  const module = {
    title,
    description,
    nodes: [section],
    fields: [],
    bank: options.bank || $('#utilitySectionBank') || document.body,
    workspaceModal: true,
  };
  openSettingsDrawer(module);
}

function setupUtilityWorkspace({ anchor, llmSection, ocrSection, dwSection, tagSection, adminLinkSection, localJsonSection }) {
  if (!anchor || $('#utilityWorkspace')) return null;
  const tools = createPanelShell(
    '工具箱',
    '低频配置和独立调试工具收在这里，不干扰日常生成任务。',
    'card utility-workspace',
  );
  tools.id = 'utilityWorkspace';
  const bank = document.createElement('div');
  bank.id = 'utilitySectionBank';
  bank.className = 'utility-section-bank';
  bank.setAttribute('aria-hidden', 'true');

  [llmSection, ocrSection, dwSection, tagSection, adminLinkSection, localJsonSection].forEach((section) => {
    if (section) bank.appendChild(section);
  });

  const grid = document.createElement('div');
  grid.className = 'tool-grid';
  grid.appendChild(makeToolCard('环境检测', '检查 API、LLM、OCR、Milvus、CUDA 和关键目录。', '开始检测', () => {
    openSectionModal('连接与模型', '运行环境检测，维护 API 地址和 LLM 配置。', llmSection, { bank });
    window.setTimeout(() => $('#btnEnvironmentCheck')?.click(), 120);
  }, { icon: 'E', primary: true }));
  grid.appendChild(makeToolCard('LLM 配置', '新增、激活、删除模型配置，或发送调试请求。', '配置', () => {
    openSettingsGroup('LLM 配置', '管理后端 LLM 配置和轻量调试请求。', ['llm.saved', 'llm.debug']);
  }, { icon: 'L' }));
  grid.appendChild(makeToolCard('OCR 配置', '维护 OCR 服务协议、地址、字段名和测试请求。', '配置', () => {
    openSettingsGroup('OCR 配置', '管理后端 OCR 配置。', ['ocr.saved']);
  }, { icon: 'O' }));
  grid.appendChild(makeToolCard('单独文档解析', '只跑文档解析、图片理解和文本整合，不进入 QA 生成。', '打开', () => {
    openSectionModal('文档解析', '单独执行 OCR、图片理解和文本整合。', dwSection, { bank });
  }, { icon: 'D' }));
  grid.appendChild(makeToolCard('知识分类测试', '粘贴文本验证三级知识分类结果。', '打开', () => {
    openSectionModal('三级知识分类', '用本地分类器直接预测文本标签。', tagSection, { bank });
  }, { icon: 'K' }));
  grid.appendChild(makeToolCard('本地 JSON 预览', '选择合并输出 JSON，在页面内预览 QA。', '打开', () => {
    openSectionModal('本地 JSON 预览', '解析本地合并输出文件并刷新 QA 预览。', localJsonSection, { bank });
  }, { icon: 'J' }));

  if (adminLinkSection) {
    grid.appendChild(makeToolCard('QA 管理', '进入列表筛选、语义检索、批量评估和软删除。', '进入管理', () => {
      window.location.href = '/ui/admin.html';
    }, { icon: 'A' }));
  }

  tools.appendChild(grid);
  tools.appendChild(bank);
  anchor.insertAdjacentElement('afterend', tools);
  return tools;
}

function setupWorkbenchHero() {
  const main = document.querySelector('main');
  const pipelineSection = sectionByField('pipelineFileInput');
  const llmSection = sectionByField('cfgName');
  const ocrSection = sectionByField('ocrCfgName');
  const dwSection = sectionByField('dwFileInput');
  if (!main || !pipelineSection) return;

  const hero = document.createElement('section');
  hero.className = 'pipeline-workbench';
  hero.innerHTML = [
    '<div class="workbench-main">',
    '<div class="workbench-title-row">',
    '<div>',
    '<p class="workbench-kicker">QA Flow</p>',
    '<h2>运行任务</h2>',
    '<p>上传文件，选择流程和目标数量，然后启动生成。高级参数进入任务设置。</p>',
    '</div>',
    '<div class="workbench-title-actions">',
    '<button type="button" class="secondary workbench-settings-btn" id="openPipelineSettingsBtn">任务设置</button>',
    '<button type="button" class="secondary compact" id="quickEnvCheckBtn">环境检测</button>',
    '</div>',
    '</div>',
    '<div class="workbench-drop" id="pipelineDropZone"></div>',
    '<div class="workbench-core" id="pipelineCoreFields"></div>',
    '<div class="summary-chip-row" id="workbenchSummary"></div>',
    '<div class="workbench-actions" id="pipelineHeroActions"></div>',
    '</div>',
  ].join('');
  main.insertBefore(hero, pipelineSection);

  const drop = $('#pipelineDropZone');
  const fileNode = nodeForField('pipelineFileInput');
  if (drop && fileNode) {
    drop.appendChild(fileNode);
    const uploadHint = document.createElement('div');
    uploadHint.className = 'workbench-upload-hint';
    uploadHint.textContent = '支持单文件或批量上传';
    drop.appendChild(uploadHint);
  }

  const core = $('#pipelineCoreFields');
  const modeNode = nodeForField('pipelineProcessingMode');
  const qaNode = nodeForField('qaTotalLimit');
  if (core && modeNode) core.appendChild(modeNode);
  if (core && qaNode) core.appendChild(qaNode);

  const pipelineShell = pipelineSection.querySelector('.module-settings-shell');
  if (pipelineShell) {
    pipelineShell.classList.add('workbench-settings-strip', 'is-hidden-on-workbench');
  }

  const heroActions = $('#pipelineHeroActions');
  const originalActions = $('#cancelTaskBtn')?.closest('.actions-row');
  if (heroActions && originalActions) {
    heroActions.appendChild(originalActions);
    const submit = $('#pipelineSubmitBtn') || heroActions.querySelector('button[type="submit"]');
    if (submit) {
      submit.type = 'button';
      submit.removeAttribute('form');
      if (!submit.dataset.directSubmitBound) {
        submit.dataset.directSubmitBound = 'true';
        submit.addEventListener('click', () => {
          handlePipelineSubmit({
            preventDefault() {},
            submitter: submit,
          });
        });
      }
    }
  }

  const summary = $('#workbenchSummary');
  const items = [];
  if (summary) {
    items.push(createSummaryChip('LLM', () => String($('#cfgActive')?.textContent || '').trim() || '未激活', { moduleKey: 'llm.saved' }));
    items.push(createSummaryChip('OCR', () => String($('#ocrCfgActive')?.textContent || '').trim() || `${$('#ocrCfgProvider')?.value || 'batch_ocr'}`, { moduleKey: 'ocr.saved' }));
    items.push(createSummaryChip('流程', () => $('#pipelineProcessingMode')?.value === 'integrated' ? '一体流程' : '标准 OCR', { moduleKey: 'pipeline.document' }));
    items.push(createSummaryChip('切分', () => `${$('#chunkingSplitType')?.value || 'markdown'} / ${$('#chunkSize')?.value || 600}`, { moduleKey: 'pipeline.chunking' }));
    items.push(createSummaryChip('生成', () => `${checkedQuestionTypesSummary()} / 上限 ${$('#qaTotalLimit')?.value || 20}`, { moduleKey: 'pipeline.generation' }));
    items.push(createSummaryChip('检索', () => `${$('#retrievalMode')?.value || 'hybrid'} / topK ${$('#semanticTopK')?.value || 3}`, { moduleKey: 'pipeline.retrieval' }));
    items.push(createSummaryChip('评估', () => pipelineEvaluationSummary(), { moduleKey: 'pipeline.evaluation' }));
    items.push(createSummaryChip('并发', () => `unit ${$('#chunkMaxConcurrency')?.value || '8'} / API ${$('#llmMaxConcurrentRequests')?.value || '默认'}`, { moduleKey: 'pipeline.performance' }));
    items.push(createSummaryChip('存储', () => pipelineStorageSummary(), { moduleKey: 'pipeline.output' }));
    items.forEach((item) => summary.appendChild(item));
  }

  const refreshSummary = () => items.forEach((item) => {
    if (item && typeof item.render === 'function') item.render();
  });
  document.addEventListener('change', refreshSummary);
  document.addEventListener('input', refreshSummary);
  observeSummarySources(refreshSummary);

  $('#openPipelineSettingsBtn')?.addEventListener('click', () => {
    openPipelineSettingsModal();
  });
  $('#quickEnvCheckBtn')?.addEventListener('click', () => {
    openSectionModal('连接与模型', '检查运行环境，维护 API 地址和模型配置。', llmSection, {
      bank: $('#utilitySectionBank') || llmSection?.parentElement || document.body,
    });
    window.setTimeout(() => $('#btnEnvironmentCheck')?.click(), 120);
  });

  setSectionMeta(llmSection, {
    key: 'connections',
    title: '连接与模型',
    description: '管理后端 API 地址、环境检测、LLM 和 OCR 配置。',
  });
  setSectionMeta(ocrSection, {
    key: 'ocr',
    title: 'OCR 配置',
    description: '维护 OCR 服务地址和协议。',
  });
  setSectionMeta(dwSection, {
    key: 'document',
    title: '文档解析',
    description: '单独执行 OCR、图片理解和文本整合。',
  });
  setSectionMeta(pipelineSection, {
    key: 'pipeline',
    title: '任务状态',
    description: '查看流水线任务、输出和调试耗时。',
  });

  [llmSection, ocrSection, dwSection, pipelineSection].forEach((section) => {
    if (section) section.classList.add('workbench-card');
  });
  if (pipelineSection) pipelineSection.classList.add('pipeline-status-card');
}

function setupWorkbenchSectionOrder() {
  const main = document.querySelector('main');
  const hero = $('.pipeline-workbench');
  if (!main || !hero) return;
  const pipelineSection = sectionByField('taskIdInput') || sectionByField('pipelineFileInput');
  const llmSection = sectionByField('cfgName');
  const ocrSection = sectionByField('ocrCfgName');
  const dwSection = sectionByField('dwFileInput');
  const chunkSection = sectionByField('chunkTaskId');
  const tagSection = sectionByField('knowledgeTagText');
  const localJsonSection = sectionByField('localJsonInput');
  const qaResultsSection = $('#qaResults')?.closest('section.card');
  const adminLinkSection = Array.from(document.querySelectorAll('section.card') || []).find((section) => {
    const h2 = section.querySelector('h2');
    return h2 && String(h2.textContent || '').includes('QA 数据管理');
  });

  const taskWorkspace = setupTaskWorkspace(pipelineSection);
  if (pipelineSection) {
    pipelineSection.classList.add('pipeline-storage-section');
  }
  const reviewWorkspace = setupReviewWorkspace({
    anchor: taskWorkspace || hero,
    chunkSection,
    qaResultsSection,
  });
  setupUtilityWorkspace({
    anchor: reviewWorkspace || taskWorkspace || hero,
    llmSection,
    ocrSection,
    dwSection,
    tagSection,
    adminLinkSection,
    localJsonSection,
  });

  if (taskWorkspace) moveSectionAfter(taskWorkspace, hero);
}

function setupWorkbenchRedesign() {
  setupWorkbenchHero();
  setupWorkbenchSectionOrder();
  document.body.classList.add('qa-workbench-redesign');
}

async function handlePipelineSubmit(e) {
  e.preventDefault();
  const base = getApiBaseUrl();
  const fileInput = $('#pipelineFileInput');
  const statusEl = $('#pipelineStatus');
  const submitBtn =
    (e && e.submitter) ||
    $('#pipelineSubmitBtn') ||
    (pipelineForm ? pipelineForm.querySelector('button[type="submit"]') : null);
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    if (statusEl) {
      statusEl.textContent = '请先选择要上传的文件';
      activateTaskTab('status');
    }
    notify('请先选择要上传的文件', 'warning');
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

    const qaTotalLimit = $('#qaTotalLimit')?.value || '20';
    const qaTotalLimitScope = $('#qaTotalLimitScope')?.value || 'per_file';
    const augmentPerQa = $('#augmentPerQa')?.value || '0';
    const chunkSize = $('#chunkSize')?.value || '600';
    const ocrTimeoutSeconds = $('#ocrTimeoutSeconds')?.value || '';
    const integratedOcrEnabled = $('#integratedOcrEnabled')?.checked !== false;
    const integratedOcrFailFast = $('#integratedOcrFailFast')?.checked === true;
    const integratedRemoveWatermark = $('#integratedRemoveWatermark')?.checked === true;
    const integratedReplaceImages = $('#integratedReplaceImages')?.checked !== false;
    const integratedWatermarkDpi = $('#integratedWatermarkDpi')?.value || '200';
    const integratedDocxStrategy = $('#integratedDocxStrategy')?.value || 'pdf';
    const imageContextSummaryMode = $('#imageContextSummaryMode')?.value || 'lightweight';
    const integratedEnableImageAnalysis = $('#integratedEnableImageAnalysis')?.checked !== false;
    const integratedImageAnalysisUseApi = true;
    const integratedEnableImageClassification = $('#integratedEnableImageClassification')?.checked === true;
    const integratedClassificationThreshold = $('#integratedClassificationThreshold')?.value || '0';
    const integratedVlmApiBase = $('#integratedVlmApiBase')?.value || '';
    const integratedVlmModelName = $('#integratedVlmModelName')?.value || '';
    const integratedVlmApiKey = $('#integratedVlmApiKey')?.value || '';
    const integratedVlmApiType = $('#integratedVlmApiType')?.value || '';
    const integratedVlmModelVersion = $('#integratedVlmModelVersion')?.value || '';
    const imageFitCheckEnabled = $('#imageFitCheckEnabled')?.checked !== false;
    const imageFitMinScore = $('#imageFitMinScore')?.value || '0.65';
    const docMaxConcurrency = $('#docMaxConcurrency')?.value || '';
    const ocrMaxConcurrency = $('#ocrMaxConcurrency')?.value || '';
    const imageAnalysisMaxConcurrency = $('#imageAnalysisMaxConcurrency')?.value || '';
    const imageFitMaxConcurrency = $('#imageFitMaxConcurrency')?.value || '';
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
    const unsupervisedBatchSize = $('#unsupervisedBatchSize')?.value || '';
    const faithfulnessNliModel = $('#faithfulnessNliModel')?.value || '';
    const answerabilityQaModel = $('#answerabilityQaModel')?.value || '';
    const coverageEmbeddingModel = $('#coverageEmbeddingModel')?.value || '';
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
    const llmMaxConcurrentRequests = $('#llmMaxConcurrentRequests')?.value || '';
    const chunkMaxAttempts = $('#chunkMaxAttempts')?.value || '2';
    const retrievalMode = $('#retrievalMode')?.value || 'hybrid';
    const semanticTopK = $('#semanticTopK')?.value || '3';
    const rerankTopN = $('#rerankTopN')?.value || '12';
    const hybridWeightDense = $('#hybridWeightDense')?.value || '0.68';
    const hybridWeightLexical = $('#hybridWeightLexical')?.value || '0.24';
    const retrievalStructureWeight = $('#retrievalStructureWeight')?.value || '0.08';
    const answerScopePolicy = $('#answerScopePolicy')?.value || 'source_primary';
    const augmentMaxConcurrency = $('#augmentMaxConcurrency')?.value || '';
    const saveModeEl = $('#saveMode');

    formData.append('qa_total_limit', qaTotalLimit);
    formData.append('qa_total_limit_scope', qaTotalLimitScope);
    formData.append('augment_per_qa', augmentPerQa);
    formData.append('chunk_size', chunkSize);
    if (!useIntegratedPipeline && String(ocrTimeoutSeconds).trim()) {
      formData.append('ocr_timeout_seconds', String(ocrTimeoutSeconds).trim());
    }
    if (useIntegratedPipeline) {
      formData.append('ocr_enabled', integratedOcrEnabled ? 'true' : 'false');
      formData.append('ocr_fail_fast', integratedOcrFailFast ? 'true' : 'false');
      formData.append('remove_watermark', integratedRemoveWatermark ? 'true' : 'false');
      formData.append('replace_images', integratedReplaceImages ? 'true' : 'false');
      formData.append('watermark_dpi', String(integratedWatermarkDpi || '200'));
      formData.append('docx_strategy', integratedDocxStrategy);
      formData.append('image_context_summary_mode', imageContextSummaryMode);
      formData.append('enable_image_analysis', integratedEnableImageAnalysis ? 'true' : 'false');
      formData.append('image_analysis_use_api', integratedImageAnalysisUseApi ? 'true' : 'false');
      formData.append('enable_image_classification', integratedEnableImageClassification ? 'true' : 'false');
      formData.append('classification_confidence_threshold', String(integratedClassificationThreshold || '0'));
      if (String(integratedVlmApiBase).trim()) formData.append('vlm_api_base', integratedVlmApiBase.trim());
      if (String(integratedVlmModelName).trim()) formData.append('vlm_model_name', integratedVlmModelName.trim());
      if (String(integratedVlmApiKey).trim()) formData.append('vlm_api_key', integratedVlmApiKey.trim());
      if (String(integratedVlmApiType).trim()) formData.append('vlm_api_type', integratedVlmApiType.trim());
      if (String(integratedVlmModelVersion).trim()) formData.append('vlm_model_version', integratedVlmModelVersion.trim());
      formData.append('image_fit_check_enabled', imageFitCheckEnabled ? 'true' : 'false');
      formData.append('image_fit_min_score', String(imageFitMinScore || '0.65'));
      if (String(docMaxConcurrency).trim()) formData.append('doc_max_concurrency', String(docMaxConcurrency).trim());
      if (String(ocrMaxConcurrency).trim()) formData.append('ocr_max_concurrency', String(ocrMaxConcurrency).trim());
      if (String(imageAnalysisMaxConcurrency).trim()) {
        formData.append('image_analysis_max_concurrency', String(imageAnalysisMaxConcurrency).trim());
      }
      if (String(imageFitMaxConcurrency).trim()) {
        formData.append('image_fit_max_concurrency', String(imageFitMaxConcurrency).trim());
      }
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
      if (String(unsupervisedBatchSize).trim()) {
        formData.append('unsupervised_batch_size', String(unsupervisedBatchSize).trim());
      }
      if (String(faithfulnessNliModel).trim()) {
        formData.append('faithfulness_nli_model', String(faithfulnessNliModel).trim());
      }
      if (String(answerabilityQaModel).trim()) {
        formData.append('answerability_qa_model', String(answerabilityQaModel).trim());
      }
      if (String(coverageEmbeddingModel).trim()) {
        formData.append('coverage_embedding_model', String(coverageEmbeddingModel).trim());
      }
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
    if (llmMaxConcurrentRequests.trim()) {
      formData.append('llm_max_concurrent_requests', llmMaxConcurrentRequests.trim());
    }
    if (chunkMaxAttempts.trim()) {
      formData.append('chunk_max_attempts', chunkMaxAttempts.trim());
    }
    formData.append('retrieval_mode', retrievalMode);
    if (semanticTopK.trim()) {
      formData.append('semantic_top_k', semanticTopK.trim());
    }
    if (rerankTopN.trim()) {
      formData.append('rerank_top_n', rerankTopN.trim());
    }
    if (hybridWeightDense.trim()) {
      formData.append('hybrid_weight_dense', hybridWeightDense.trim());
    }
    if (hybridWeightLexical.trim()) {
      formData.append('hybrid_weight_lexical', hybridWeightLexical.trim());
    }
    if (retrievalStructureWeight.trim()) {
      formData.append('retrieval_structure_weight', retrievalStructureWeight.trim());
    }
    formData.append('answer_scope_policy', answerScopePolicy);
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
    if (data.task_id) {
      pipelineOutputsAutoShownForTaskId = '';
      pipelineRawJsonOpen = false;
      applyPipelineStatus(data, { base, taskId: data.task_id, activateStatus: true });
      startTaskPolling(base, data.task_id);
    } else {
      updatePipelineStatusView(data, { activateStatus: true });
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = '调用出错：' + String(err);
  } finally {
    setBtnLoading(submitBtn, false);
  }
}

function capturePipelineStatusViewport(statusEl) {
  const scroller = document.scrollingElement || document.documentElement;
  const rawPre = statusEl ? statusEl.querySelector('.pipeline-debug-raw pre') : null;
  return {
    windowX: Number(window.scrollX || window.pageXOffset || 0),
    windowY: Number(scroller ? scroller.scrollTop : window.scrollY || window.pageYOffset || 0),
    rawPreScrollTop: rawPre ? Number(rawPre.scrollTop || 0) : 0,
    rawPreScrollLeft: rawPre ? Number(rawPre.scrollLeft || 0) : 0,
  };
}

function restorePipelineStatusViewport(snapshot) {
  if (!snapshot) return;
  const statusEl = $('#pipelineStatus');
  const rawPre = statusEl ? statusEl.querySelector('.pipeline-debug-raw pre') : null;
  if (rawPre) {
    rawPre.scrollTop = snapshot.rawPreScrollTop || 0;
    rawPre.scrollLeft = snapshot.rawPreScrollLeft || 0;
  }
  try {
    window.scrollTo(snapshot.windowX || 0, snapshot.windowY || 0);
  } catch {
    // Scroll restoration is best-effort; never break status rendering.
  }
}

function updatePipelineStatusView(status, options = {}) {
  const statusEl = $('#pipelineStatus');
  if (!statusEl) return;
  try {
    const viewportSnapshot = capturePipelineStatusViewport(statusEl);
    const currentRaw = statusEl.querySelector('.pipeline-debug-raw');
    if (currentRaw) pipelineRawJsonOpen = Boolean(currentRaw.open);
    statusEl.textContent = '';
    statusEl.classList.add('pipeline-debug-panel');
    statusEl.appendChild(renderPipelineDebugStatus(status, { rawOpen: pipelineRawJsonOpen }));
    if (options.activateStatus) activateTaskTab('status');
    refreshReviewWorkspaceState();
    restorePipelineStatusViewport(viewportSnapshot);
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => restorePipelineStatusViewport(viewportSnapshot));
    }
  } catch (err) {
    statusEl.textContent = JSON.stringify(status, null, 2);
  }
}

function asNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function firstNumber() {
  for (let i = 0; i < arguments.length; i += 1) {
    const n = asNumber(arguments[i]);
    if (n !== null) return n;
  }
  return null;
}

function getStageExtra(fileEntry, stageName) {
  const stages = fileEntry && fileEntry.stages && typeof fileEntry.stages === 'object'
    ? fileEntry.stages
    : {};
  const stage = stages[stageName] && typeof stages[stageName] === 'object'
    ? stages[stageName]
    : {};
  return {
    ...stage,
    ...(stage.extra && typeof stage.extra === 'object' ? stage.extra : {}),
  };
}

function parseTimestampMs(value) {
  if (value === null || value === undefined || value === '') return null;
  const raw = String(value).trim();
  if (!raw) return null;
  const numeric = Number(raw);
  if (Number.isFinite(numeric) && /^\d+(\.\d+)?$/.test(raw)) {
    return raw.length > 11 ? numeric : numeric * 1000;
  }
  const parsed = Date.parse(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

function pipelineFirstSeenStartMs(status) {
  const safeStatus = status && typeof status === 'object' ? status : {};
  const explicit = parseTimestampMs(safeStatus.started_at || safeStatus.created_at);
  if (explicit !== null) return explicit;
  const key = String(safeStatus.task_id || '__current_pipeline_task');
  if (!pipelineTaskFirstSeenAtMs[key]) {
    pipelineTaskFirstSeenAtMs[key] = Date.now();
  }
  return pipelineTaskFirstSeenAtMs[key];
}

function deriveLiveElapsedSeconds(status) {
  const safeStatus = status && typeof status === 'object' ? status : {};
  const startMs = pipelineFirstSeenStartMs(safeStatus);
  if (startMs === null) return null;
  const endMs = isTaskTerminal(safeStatus.status)
    ? parseTimestampMs(safeStatus.finished_at || safeStatus.completed_at || safeStatus.updated_at)
    : Date.now();
  if (endMs === null) return null;
  return Math.max(0, (endMs - startMs) / 1000);
}

function collectFileProgressEntries(status) {
  const progress = status && status.file_progress && typeof status.file_progress === 'object'
    ? status.file_progress
    : {};
  return Object.keys(progress).map((filename) => ({
    filename,
    entry: progress[filename] || {},
  }));
}

function pipelineStatusLabel(status) {
  const key = String(status || 'unknown').trim().toLowerCase();
  const labels = {
    queued: '排队中',
    pending: '等待中',
    processing: '运行中',
    running: '运行中',
    completed: '已完成',
    succeeded: '已完成',
    failed: '失败',
    canceled: '已终止',
    cancelled: '已终止',
  };
  return labels[key] || String(status || 'unknown');
}

function translatePipelineMessage(message) {
  const text = String(message || '').trim();
  if (!text) return '';
  let match = text.match(/^batch done:\s*Batch finished:\s*(\d+)\s+success,\s*(\d+)\s+failed$/i);
  if (match) return `任务完成：${match[1]} 成功，${match[2]} 失败`;
  match = text.match(/^Batch finished:\s*(\d+)\s+success,\s*(\d+)\s+failed$/i);
  if (match) return `任务完成：${match[1]} 成功，${match[2]} 失败`;
  match = text.match(/^batch done:\s*Batch finished with partial failures\s*\((\d+)\s+files?\)$/i);
  if (match) return `任务部分完成：${match[1]} 个文件失败`;
  match = text.match(/^Batch finished with partial failures\s*\((\d+)\s+files?\)$/i);
  if (match) return `任务部分完成：${match[1]} 个文件失败`;
  if (/^Batch started$/i.test(text)) return '批量流水线开始';
  return text;
}

function stageOrderIndex(stage) {
  const index = DOC_STAGE_ORDER.indexOf(stage);
  return index < 0 ? 999 : index;
}

function stageCandidateText(candidate) {
  if (!candidate) return '';
  const translated = translatePipelineMessage(candidate.message);
  if (translated) return translated;
  return DOC_STAGE_LABELS[candidate.stage] || candidate.stage || '';
}

function pipelineCurrentStageText(status) {
  const safeStatus = status && typeof status === 'object' ? status : {};
  const entries = collectFileProgressEntries(safeStatus);
  const candidates = [];
  for (let i = 0; i < entries.length; i += 1) {
    const entry = entries[i].entry || {};
    const stages = entry.stages && typeof entry.stages === 'object' ? entry.stages : {};
    Object.keys(stages).forEach((name) => {
      const item = stages[name] || {};
      const state = String(item.state || '').trim().toLowerCase();
      candidates.push({
        stage: name,
        state,
        message: item.message,
        updatedMs: parseTimestampMs(item.updated_at || item.completed_at || item.started_at),
        order: stageOrderIndex(name),
      });
    });
    if (entry.message) {
      candidates.push({
        stage: '',
        state: String(entry.status || '').trim().toLowerCase(),
        message: entry.message,
        updatedMs: parseTimestampMs(entry.updated_at || entry.completed_at),
        order: 1000,
      });
    }
  }
  const active = candidates.filter((candidate) => {
    return ['processing', 'running', 'waiting', 'queued', 'pending'].includes(candidate.state);
  });
  const sortLatest = (a, b) => {
    const left = a.updatedMs === null ? -1 : a.updatedMs;
    const right = b.updatedMs === null ? -1 : b.updatedMs;
    if (right !== left) return right - left;
    return b.order - a.order;
  };
  active.sort(sortLatest);
  const activeText = stageCandidateText(active[0]);
  if (activeText) return activeText;
  candidates.sort(sortLatest);
  const candidateText = stageCandidateText(candidates[0]);
  if (candidateText) return candidateText;
  return translatePipelineMessage(safeStatus.message);
}

function pipelineProgressPercent(status) {
  const safeStatus = status && typeof status === 'object' ? status : {};
  const raw = firstNumber(safeStatus.progress_percent, safeStatus.progress, safeStatus.percent);
  if (raw === null) return null;
  const percent = raw <= 1 ? raw * 100 : raw;
  return Math.max(0, Math.min(100, percent));
}

function collectOutputTimings(status) {
  const outputs = status && Array.isArray(status.outputs) ? status.outputs : [];
  return outputs
    .map((item) => (item && typeof item.timing === 'object' ? item.timing : null))
    .filter(Boolean);
}

function sumTiming(timings, key) {
  let total = 0;
  let found = false;
  timings.forEach((timing) => {
    const n = asNumber(timing && timing[key]);
    if (n === null) return;
    total += n;
    found = true;
  });
  return found ? total : null;
}

function isPlainObject(value) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value));
}

function hasObjectKeys(value) {
  return isPlainObject(value) && Object.keys(value).length > 0;
}

function findGenerationExtra(status) {
  const entries = collectFileProgressEntries(status);
  for (let i = 0; i < entries.length; i += 1) {
    const extra = getStageExtra(entries[i].entry, 'qa_generation');
    if (extra && Object.keys(extra).length) return extra;
  }
  return {};
}

function sumStageElapsed(fileEntries, stageNames) {
  const wanted = new Set(stageNames || []);
  let total = 0;
  let found = false;
  fileEntries.forEach(({ entry }) => {
    const stages = entry && entry.stages && typeof entry.stages === 'object' ? entry.stages : {};
    Object.keys(stages).forEach((stageName) => {
      if (!wanted.has(stageName)) return;
      const extra = getStageExtra(entry, stageName);
      const elapsed = firstNumber(extra.elapsed_seconds);
      if (elapsed === null) return;
      total += elapsed;
      found = true;
    });
  });
  return found ? total : null;
}

function sumNumbers() {
  let total = 0;
  let found = false;
  for (let i = 0; i < arguments.length; i += 1) {
    const value = arguments[i];
    if (Array.isArray(value)) {
      value.forEach((item) => {
        const n = asNumber(item);
        if (n === null) return;
        total += n;
        found = true;
      });
      continue;
    }
    const n = asNumber(value);
    if (n === null) continue;
    total += n;
    found = true;
  }
  return found ? total : null;
}

function normalizeGenerationWallDetail(detail, explicitWallDetail) {
  if (!isPlainObject(detail)) return {};
  const normalized = { ...detail };
  const total = firstNumber(normalized.document_total_seconds);
  const mainSum = sumNumbers(
    normalized.candidate_question_seconds,
    normalized.retrieval_seconds,
    normalized.answer_generation_seconds,
    normalized.validation_and_bookkeeping_seconds,
    normalized.scheduler_gap_seconds,
  );
  if (!explicitWallDetail && total !== null && mainSum !== null && mainSum > total + 0.1) {
    return {
      document_total_seconds: total,
      scheduler_gap_seconds: total,
      chunks_total: normalized.chunks_total,
      chunks_completed: normalized.chunks_completed,
      generation_units_total: normalized.generation_units_total,
      generation_units_completed: normalized.generation_units_completed,
      qa_generated: normalized.qa_generated,
      qa_total_limit: normalized.qa_total_limit,
      qa_total_limit_scope: normalized.qa_total_limit_scope,
      wall_detail_unavailable: true,
    };
  }
  return normalized;
}

function deriveGenerationTimingViews(generationExtra, outputTimings) {
  const progressTiming = isPlainObject(generationExtra.generation_timing)
    ? generationExtra.generation_timing
    : {};
  let wallDetail = {};
  let cumulativeDetail = {};
  let explicitWallDetail = false;
  let outputChunkDetails = [];
  let outputUnitDetails = [];

  if (hasObjectKeys(generationExtra.generation_wall_detail)) {
    wallDetail = generationExtra.generation_wall_detail;
    explicitWallDetail = true;
  } else if (hasObjectKeys(progressTiming.generation_wall_detail)) {
    wallDetail = progressTiming.generation_wall_detail;
    explicitWallDetail = true;
  }

  if (hasObjectKeys(generationExtra.generation_cumulative_detail)) {
    cumulativeDetail = generationExtra.generation_cumulative_detail;
  } else if (hasObjectKeys(progressTiming.generation_cumulative_detail)) {
    cumulativeDetail = progressTiming.generation_cumulative_detail;
  }

  outputTimings.forEach((timing) => {
    if (Array.isArray(timing.generation_unit_details)) {
      outputUnitDetails = outputUnitDetails.concat(timing.generation_unit_details);
    }
    if (Array.isArray(timing.generation_chunk_details)) {
      outputChunkDetails = outputChunkDetails.concat(timing.generation_chunk_details);
    }
    if (!hasObjectKeys(wallDetail) && hasObjectKeys(timing.generation_wall_detail)) {
      wallDetail = timing.generation_wall_detail;
      explicitWallDetail = true;
    }
    if (!hasObjectKeys(cumulativeDetail) && hasObjectKeys(timing.generation_cumulative_detail)) {
      cumulativeDetail = timing.generation_cumulative_detail;
    }
    if (!hasObjectKeys(wallDetail) && hasObjectKeys(timing.generation_detail)) {
      wallDetail = timing.generation_detail;
    }
  });

  if (!hasObjectKeys(wallDetail) && hasObjectKeys(progressTiming)) {
    wallDetail = progressTiming;
    explicitWallDetail = Boolean(progressTiming.scheduler_gap_seconds !== undefined);
  }

  const progressUnitDetails = Array.isArray(generationExtra.generation_unit_details)
    ? generationExtra.generation_unit_details
    : [];
  const progressChunkDetails = Array.isArray(generationExtra.generation_chunk_details)
    ? generationExtra.generation_chunk_details
    : [];
  const unitDetails = progressUnitDetails.length
    ? progressUnitDetails
    : (outputUnitDetails.length ? outputUnitDetails : (progressChunkDetails.length ? progressChunkDetails : outputChunkDetails));

  return {
    wallDetail: normalizeGenerationWallDetail(wallDetail, explicitWallDetail),
    cumulativeDetail,
    unitDetails,
    chunkDetails: progressChunkDetails.length ? progressChunkDetails : outputChunkDetails,
  };
}

function derivePipelineTiming(status) {
  const outputTimings = collectOutputTimings(status);
  const generationExtra = findGenerationExtra(status);
  const fileEntries = collectFileProgressEntries(status);
  const generationViews = deriveGenerationTimingViews(generationExtra, outputTimings);
  const generationTiming = generationViews.wallDetail || {};

  let documentSeconds = sumStageElapsed(fileEntries, DOCUMENT_PARSE_STAGE_KEYS);
  if (documentSeconds === null) {
    documentSeconds = sumStageElapsed(fileEntries, DOCUMENT_PARSE_FALLBACK_STAGE_KEYS);
  }
  documentSeconds = firstNumber(documentSeconds, sumTiming(outputTimings, 'ocr_seconds'));

  const generationSeconds = firstNumber(
    generationTiming.document_total_seconds,
    sumTiming(outputTimings, 'generation_seconds'),
    generationExtra.generation_seconds,
    generationExtra.elapsed_seconds,
  );

  const evaluationFromProgress = sumStageElapsed(fileEntries, EVALUATION_STAGE_KEYS);
  const evaluationSeconds = firstNumber(
    evaluationFromProgress,
    sumNumbers(sumTiming(outputTimings, 'unsupervised_seconds'), sumTiming(outputTimings, 'evaluation_seconds')),
  );
  const unsupervisedSeconds = firstNumber(
    sumTiming(outputTimings, 'unsupervised_seconds'),
    sumStageElapsed(fileEntries, ['unsupervised_evaluation']),
  );
  const llmEvaluationSeconds = firstNumber(
    sumTiming(outputTimings, 'evaluation_seconds'),
    sumStageElapsed(fileEntries, ['evaluation']),
  );
  const storageSeconds = sumStageElapsed(fileEntries, STORAGE_OUTPUT_STAGE_KEYS);
  const totalSeconds = deriveLiveElapsedSeconds(status);

  return {
    document_seconds: documentSeconds,
    ocr_seconds: documentSeconds,
    generation_seconds: generationSeconds,
    unsupervised_seconds: unsupervisedSeconds,
    evaluation_seconds: evaluationSeconds,
    llm_evaluation_seconds: llmEvaluationSeconds,
    storage_seconds: storageSeconds,
    total_seconds: totalSeconds,
    live_elapsed_seconds: totalSeconds,
    generation_detail: generationTiming,
    generation_cumulative_detail: generationViews.cumulativeDetail,
    generation_unit_details: generationViews.unitDetails,
    generation_chunk_details: generationViews.chunkDetails,
  };
}

function appendMetricChip(parent, label, value, emptyText) {
  const chip = document.createElement('div');
  chip.className = 'pipeline-debug-chip';
  const name = document.createElement('span');
  name.className = 'pipeline-debug-chip-label';
  name.textContent = label;
  const val = document.createElement('strong');
  val.textContent = value === null || value === undefined || value === ''
    ? (emptyText || '未记录')
    : fmtSeconds(value);
  chip.appendChild(name);
  chip.appendChild(val);
  parent.appendChild(chip);
}

function appendTextMetric(parent, label, value) {
  const item = document.createElement('div');
  item.className = 'pipeline-debug-kv';
  const name = document.createElement('span');
  name.textContent = label;
  const val = document.createElement('strong');
  val.textContent = value === null || value === undefined || value === '' ? '未记录' : String(value);
  item.appendChild(name);
  item.appendChild(val);
  parent.appendChild(item);
}

function debugCountText(value, emptyText = '等待') {
  const n = asNumber(value);
  if (n === null) return emptyText;
  return String(n);
}

function debugSecondsText(value, emptyText = '未返回') {
  const n = asNumber(value);
  if (n === null) return emptyText;
  return fmtSeconds(n);
}

function translatedDropReason(reason) {
  const labels = {
    ok: '通过',
    missing_items: '模型没有返回可用条目',
    missing_question: '缺少问题',
    missing_answer: '缺少答案',
    missing_answer_explanation: '缺少答案解释',
    missing_source_fact_text: '缺少来源事实',
    question_mismatch: '答案阶段返回的问题与候选问题不一致',
    duplicate_question: '重复问题',
    empty_or_duplicate_question: '空问题/重复问题',
    invalid_json: 'JSON 无效',
    normalize_failed: '格式归一失败',
    source_fact_missing: '来源事实缺失',
    source_fact_not_grounded: '来源事实未命中原文',
    source_fact_not_grounded_in_chunk: '来源事实未能在检索证据中定位',
    source_fact_not_anchored_to_source_chunk: '来源事实没有锚定到主来源块',
    source_fact_too_short: '来源事实过短',
    point_source_fact_multiline: '单点模式下来源事实不应跨多行',
    point_source_fact_has_semicolon: '单点模式下来源事实包含分号，像是多个事实',
    point_source_fact_multi_sentence: '单点模式下来源事实包含多个句子',
    summary_source_fact_not_compound: '总结模式下来源事实过于单一，不像多信息点汇总',
    summary_source_fact_segments_insufficient: '总结模式下来源事实片段不足，至少需要两个可定位的信息点',
    summary_source_fact_segment_not_grounded_in_chunk: '总结模式下可定位的来源事实片段不足',
    summary_source_fact_not_grounded_in_chunk: '总结模式下来源事实没有命中证据',
    summary_question_not_grouped: '历史规则：总结题形态不符合要求',
    summary_question_too_shallow_list: '总结模式下候选问题只是泛化浅层清单题',
    ambiguous_reference_question: '问题里存在未消解的指代',
    ambiguous_reference_answer: '答案里存在未消解的指代',
    ambiguous_reference_answer_explanation: '答案解释里存在未消解的指代',
    ambiguous_reference_source_fact_text: '来源事实里存在未消解的指代',
    mcq_invalid_options: '单选题选项无效',
    mcq_options_not_4: '单选题不是 4 个选项',
    mcq_invalid_correct_option: '单选题正确选项无效',
    judge_invalid_answer: '判断题答案无效',
  };
  return labels[reason] || reason;
}

function chunkStatusText(chunk) {
  if (chunk && chunk.error) return '失败';
  if (asNumber(chunk && chunk.valid_items) !== null) return '完成';
  return '等待';
}

function chunkReasonParts(chunk) {
  const parts = [];
  const stats = chunk && chunk.dropped_reason_stats && typeof chunk.dropped_reason_stats === 'object'
    ? chunk.dropped_reason_stats
    : {};
  Object.keys(stats).forEach((key) => {
    parts.push(`${translatedDropReason(key)}：${stats[key]}`);
  });
  if (chunk && chunk.error) parts.push(`错误：${chunk.error}`);
  return parts;
}

function pipelineDebugEventLabel(event) {
  const labels = {
    candidate_question_llm_call: '候选题生成',
    evidence_answer_llm_call: '答案生成',
  };
  return labels[event] || event || '未知事件';
}

function formatExpireAt(ts) {
  if (ts === null || ts === undefined || ts === '') return '';
  const n = Number(ts);
  if (!Number.isFinite(n) || n <= 0) return '';
  try {
    return new Date(n * 1000).toLocaleString();
  } catch {
    return '';
  }
}

function findPipelineOutputForReview(taskId, sourceFile, jsonPath) {
  const outputs = Array.isArray(lastPipelineOutputs) ? lastPipelineOutputs : [];
  const wantedTask = String(taskId || '').trim();
  const wantedSource = String(sourceFile || '').trim();
  const wantedJsonBase = String(jsonPath || '').split('/').pop();
  const isMatch = (output) => {
    if (!output || typeof output !== 'object') return false;
    const outputJsonBase = String(output.consolidated_json || '').split('/').pop();
    if (wantedJsonBase && outputJsonBase && wantedJsonBase === outputJsonBase) return true;
    const candidates = [
      output.source_file,
      output.core_file,
      output.filename,
    ].map((value) => String(value || '').trim());
    if (Array.isArray(output.source_files)) {
      output.source_files.forEach((value) => candidates.push(String(value || '').trim()));
    }
    if (wantedSource && candidates.includes(wantedSource)) return true;
    if (!wantedSource && wantedTask && output.task_id && String(output.task_id) === wantedTask) return true;
    return false;
  };
  return outputs.find(isMatch) || null;
}

function isReviewOutputEligible(output) {
  if (!output || typeof output !== 'object') return false;
  if (!output.consolidated_json) return false;
  if (output.manual_ingest_select_all) return false;
  const expireAt = Number(output.artifacts_expire_at || latestPipelineStatus?.artifacts_expire_at || 0);
  if (Number.isFinite(expireAt) && expireAt > 0 && expireAt * 1000 <= Date.now()) return false;
  return true;
}

function updateQaReviewContext(json, items) {
  const task = json && typeof json === 'object' ? json.task || {} : {};
  const taskId = String(task.task_id || lastTaskId || '').trim();
  const sourceFile = String(task.original_filename || '').trim();
  const jsonPath = String(currentConsolidatedJsonPath || '').trim();
  const output = findPipelineOutputForReview(taskId, sourceFile, jsonPath);
  const previousKey = `${qaReviewState.taskId}|${qaReviewState.sourceFile}|${qaReviewState.jsonPath}`;
  const nextKey = `${taskId}|${sourceFile}|${jsonPath}`;
  qaReviewState.taskId = taskId;
  qaReviewState.sourceFile = sourceFile;
  qaReviewState.jsonPath = jsonPath;
  qaReviewState.output = output;
  qaReviewState.expireAt = Number(output?.artifacts_expire_at || latestPipelineStatus?.artifacts_expire_at || 0) || null;
  qaReviewState.items = Array.isArray(items) ? items.slice() : [];
  qaReviewState.enabled = Boolean(taskId && qaReviewState.items.length && isReviewOutputEligible(output));
  if (previousKey !== nextKey) {
    qaReviewState.selectedIds = new Set();
    qaReviewState.selectAllTask = false;
  }
  updateQaReviewToolbar();
}

function qaReviewItemId(item) {
  return String(item && item.id || '').trim();
}

function updateQaReviewCheckboxes() {
  document.querySelectorAll('[data-qa-review-checkbox]').forEach((node) => {
    const id = String(node.getAttribute('data-qa-id') || '').trim();
    node.checked = qaReviewState.selectAllTask || qaReviewState.selectedIds.has(id);
  });
}

function selectedQaCount() {
  if (qaReviewState.selectAllTask) {
    return qaReviewState.items.filter((item) => qaReviewItemId(item)).length;
  }
  return qaReviewState.selectedIds.size;
}

function updateQaReviewToolbar() {
  const qaResults = $('#qaResults');
  if (!qaResults) return;
  let toolbar = $('#qaReviewToolbar');
  if (!qaReviewState.enabled) {
    if (toolbar) toolbar.remove();
    return;
  }
  if (!toolbar) {
    toolbar = document.createElement('div');
    toolbar.id = 'qaReviewToolbar';
    toolbar.className = 'qa-review-toolbar';
  }
  toolbar.replaceChildren();

  const copy = document.createElement('div');
  copy.className = 'qa-review-copy';
  const title = document.createElement('strong');
  title.textContent = '人工审阅入库';
  const desc = document.createElement('span');
  const expireText = formatExpireAt(qaReviewState.expireAt);
  desc.textContent = expireText
    ? `未自动入库的结果可勾选后写入向量库；临时文件保留到 ${expireText}`
    : '未自动入库的结果可勾选后写入向量库。';
  copy.append(title, desc);

  const actions = document.createElement('div');
  actions.className = 'qa-review-actions';
  const count = document.createElement('span');
  count.className = 'qa-review-count';
  count.textContent = `已选 ${selectedQaCount()} / ${qaReviewState.items.length}`;
  const selectAll = document.createElement('button');
  selectAll.type = 'button';
  selectAll.className = 'secondary';
  selectAll.textContent = '一键全选当前任务';
  selectAll.addEventListener('click', () => {
    qaReviewState.selectAllTask = true;
    qaReviewState.selectedIds = new Set(
      qaReviewState.items.map((item) => qaReviewItemId(item)).filter(Boolean),
    );
    updateQaReviewCheckboxes();
    updateQaReviewToolbar();
  });
  const clear = document.createElement('button');
  clear.type = 'button';
  clear.className = 'secondary';
  clear.textContent = '清空选择';
  clear.addEventListener('click', () => {
    qaReviewState.selectAllTask = false;
    qaReviewState.selectedIds.clear();
    updateQaReviewCheckboxes();
    updateQaReviewToolbar();
  });
  const ingest = document.createElement('button');
  ingest.type = 'button';
  ingest.textContent = '入库所选 QA';
  ingest.disabled = selectedQaCount() <= 0;
  ingest.addEventListener('click', ingestSelectedQa);
  actions.append(count, selectAll, clear, ingest);
  toolbar.append(copy, actions);
  if (qaResults.firstChild !== toolbar) {
    qaResults.insertBefore(toolbar, qaResults.firstChild);
  }
}

async function ingestSelectedQa() {
  if (!qaReviewState.enabled || !qaReviewState.taskId) {
    notify('当前没有可人工入库的任务结果', 'warning');
    return;
  }
  if (!qaReviewState.selectAllTask && qaReviewState.selectedIds.size <= 0) {
    notify('请先勾选要入库的 QA', 'warning');
    return;
  }
  const base = getApiBaseUrl();
  const body = {
    source_file: qaReviewState.sourceFile || undefined,
    select_all_task: Boolean(qaReviewState.selectAllTask),
    selected_ids: qaReviewState.selectAllTask ? [] : Array.from(qaReviewState.selectedIds),
  };
  try {
    const data = await fetchJson(
      `${base}/pipeline-tasks/${encodeURIComponent(qaReviewState.taskId)}/ingest-selected-qa`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
    );
    const stored = data && typeof data.stored_count === 'number' ? data.stored_count : 0;
    notify(`人工入库完成：写入 ${stored} 条记录`, 'success');
    qaReviewState.selectAllTask = false;
    qaReviewState.selectedIds.clear();
    await loadTaskStatusById(qaReviewState.taskId, { resumePolling: false, silent: true }).catch(() => {});
    updateQaReviewContext({ task: { task_id: qaReviewState.taskId, original_filename: qaReviewState.sourceFile } }, qaReviewState.items);
  } catch (err) {
    notify(`人工入库失败：${String(err.message || err)}`, 'error');
  }
}

function ensureRawResponseModal() {
  let overlay = $('#rawResponseOverlay');
  let modal = $('#rawResponseModal');
  if (overlay && modal) return { overlay, modal };
  overlay = document.createElement('div');
  overlay.id = 'rawResponseOverlay';
  overlay.className = 'raw-response-overlay';
  overlay.hidden = true;
  modal = document.createElement('section');
  modal.id = 'rawResponseModal';
  modal.className = 'raw-response-modal';
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-modal', 'true');
  modal.setAttribute('aria-labelledby', 'rawResponseModalTitle');
  overlay.addEventListener('click', closeRawResponseModal);
  document.body.append(overlay, modal);
  return { overlay, modal };
}

function closeRawResponseModal() {
  const overlay = $('#rawResponseOverlay');
  const modal = $('#rawResponseModal');
  if (overlay) overlay.hidden = true;
  if (modal) {
    modal.classList.remove('is-open');
    modal.replaceChildren();
  }
}

function appendRawRecordDetail(parent, label, value) {
  if (value === null || value === undefined || value === '') return;
  const detail = document.createElement('details');
  detail.className = 'raw-response-detail';
  const summary = document.createElement('summary');
  summary.textContent = label;
  const pre = document.createElement('pre');
  pre.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  detail.append(summary, pre);
  parent.appendChild(detail);
}

function renderRawResponseRecords(container, data, chunkIndex) {
  const records = Array.isArray(data?.records) ? data.records : [];
  container.replaceChildren();
  const head = document.createElement('div');
  head.className = 'raw-response-head';
  const title = document.createElement('h3');
  title.id = 'rawResponseModalTitle';
  title.textContent = `chunk ${chunkIndex} 的模型原始响应`;
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'secondary';
  close.textContent = '关闭';
  close.addEventListener('click', closeRawResponseModal);
  head.append(title, close);
  container.appendChild(head);

  const hint = document.createElement('div');
  hint.className = 'raw-response-hint';
  const expireText = formatExpireAt(data?.artifacts_expire_at);
  hint.textContent = expireText
    ? `仅从后端受限接口读取，不写入浏览器缓存；调试文件预计保留到 ${expireText}。`
    : '仅从后端受限接口读取，不写入浏览器缓存。';
  container.appendChild(hint);

  if (!records.length) {
    const empty = document.createElement('div');
    empty.className = 'pipeline-debug-empty';
    empty.textContent = '这个 chunk 暂无模型原始响应，可能任务尚未执行到该阶段，或调试文件已清理。';
    container.appendChild(empty);
    return;
  }

  const list = document.createElement('div');
  list.className = 'raw-response-list';
  records.forEach((record, idx) => {
    const card = document.createElement('article');
    card.className = 'raw-response-card';
    const event = String(record.event || '');
    const summary = document.createElement('div');
    summary.className = 'raw-response-summary';
    const titleLine = document.createElement('strong');
    titleLine.textContent = `${idx + 1}. ${pipelineDebugEventLabel(event)}`;
    const meta = document.createElement('span');
    const validCount = record.items_validated_count !== undefined ? `有效 ${record.items_validated_count}` : '';
    const rawCount = record.items_raw_count !== undefined ? `原始 ${record.items_raw_count}` : '';
    meta.textContent = [
      record.model ? `模型 ${record.model}` : '',
      record.prompt_template_key ? `模板 ${record.prompt_template_key}` : '',
      rawCount,
      validCount,
      record.dropped_reason ? `丢弃：${translatedDropReason(record.dropped_reason)}` : '',
      record.parse_error ? `解析错误：${record.parse_error}` : '',
    ].filter(Boolean).join(' | ');
    summary.append(titleLine, meta);
    card.appendChild(summary);

    const compact = document.createElement('div');
    compact.className = 'raw-response-compact';
    if (record.candidate && typeof record.candidate === 'object') {
      appendTextMetric(compact, '候选问题', record.candidate.question || '');
      appendTextMetric(compact, '题型', record.candidate.question_type || '');
    }
    if (record.dropped_validation_reasons && typeof record.dropped_validation_reasons === 'object') {
      appendTextMetric(
        compact,
        '候选题丢弃',
        Object.keys(record.dropped_validation_reasons)
          .map((key) => `${translatedDropReason(key)}：${record.dropped_validation_reasons[key]}`)
          .join('；') || '无',
      );
    }
    if (record._debug_file) appendTextMetric(compact, '调试文件', record._debug_file);
    if (compact.childElementCount) card.appendChild(compact);

    appendRawRecordDetail(card, 'system_prompt', record.system_prompt);
    appendRawRecordDetail(card, 'user_content', record.user_content);
    appendRawRecordDetail(card, 'raw_response', record.raw_response);
    appendRawRecordDetail(card, 'retrieval_trace', record.generation_unit?.retrieval_trace);
    appendRawRecordDetail(card, 'candidate', record.candidate);
    list.appendChild(card);
  });
  container.appendChild(list);
}

async function openChunkDebugRawModal(chunkIndex) {
  const taskId = String(latestPipelineStatus?.task_id || lastTaskId || '').trim();
  if (!taskId) {
    notify('请先加载一个流水线任务', 'warning');
    return;
  }
  const { overlay, modal } = ensureRawResponseModal();
  overlay.hidden = false;
  modal.classList.add('is-open');
  modal.replaceChildren();
  const loading = document.createElement('div');
  loading.className = 'raw-response-loading';
  loading.textContent = '正在读取模型原始响应…';
  modal.appendChild(loading);
  try {
    const params = new URLSearchParams();
    if (chunkIndex !== undefined && chunkIndex !== null) params.set('chunk_index', String(chunkIndex));
    const data = await fetchJson(
      `${getApiBaseUrl()}/pipeline-tasks/${encodeURIComponent(taskId)}/debug-jsonl?${params.toString()}`,
    );
    renderRawResponseRecords(modal, data, chunkIndex);
  } catch (err) {
    modal.replaceChildren();
    const head = document.createElement('div');
    head.className = 'raw-response-head';
    const title = document.createElement('h3');
    title.id = 'rawResponseModalTitle';
    title.textContent = '模型原始响应不可用';
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'secondary';
    close.textContent = '关闭';
    close.addEventListener('click', closeRawResponseModal);
    head.append(title, close);
    const empty = document.createElement('div');
    empty.className = 'pipeline-debug-empty';
    empty.textContent = String(err.message || err);
    modal.append(head, empty);
  }
}

window.qaFlowReview = {
  setContext(json, items) {
    updateQaReviewContext(json, items);
  },
  isEnabled() {
    return Boolean(qaReviewState.enabled);
  },
  isSelected(id) {
    const normalized = String(id || '').trim();
    return Boolean(normalized && (qaReviewState.selectAllTask || qaReviewState.selectedIds.has(normalized)));
  },
  setSelected(id, checked) {
    const normalized = String(id || '').trim();
    if (!normalized) return;
    if (checked) {
      qaReviewState.selectedIds.add(normalized);
    } else {
      qaReviewState.selectAllTask = false;
      qaReviewState.selectedIds.delete(normalized);
    }
    updateQaReviewToolbar();
  },
  afterRender() {
    updateQaReviewToolbar();
    updateQaReviewCheckboxes();
  },
};

function appendChunkTableCell(row, text, className = '') {
  const cell = document.createElement('td');
  if (className) cell.className = className;
  cell.textContent = text;
  row.appendChild(cell);
  return cell;
}

function renderPipelineDebugStatus(status, options = {}) {
  const root = document.createElement('div');
  root.className = 'pipeline-debug';
  const safeStatus = status && typeof status === 'object' ? status : {};
  const timing = derivePipelineTiming(safeStatus);
  const statusKey = String(safeStatus.status || 'unknown').trim().toLowerCase();
  const stageText = pipelineCurrentStageText(safeStatus);
  const progressPercent = pipelineProgressPercent(safeStatus);

  const header = document.createElement('div');
  header.className = 'pipeline-debug-header';
  const headerTop = document.createElement('div');
  headerTop.className = 'pipeline-debug-header-top';
  const title = document.createElement('div');
  title.className = 'pipeline-debug-title';
  title.textContent = '流水线调试视图';
  const state = document.createElement('span');
  state.className = 'pipeline-state-pill';
  state.dataset.status = statusKey;
  state.textContent = pipelineStatusLabel(statusKey);
  headerTop.append(title, state);
  const subtitle = document.createElement('div');
  subtitle.className = 'pipeline-debug-subtitle';
  subtitle.textContent = [
    safeStatus.task_id ? '任务 ' + safeStatus.task_id : '',
  ].filter(Boolean).join(' | ') || '等待任务状态';
  const stage = document.createElement('div');
  stage.className = 'pipeline-debug-stage';
  stage.textContent = stageText ? `当前阶段：${stageText}` : '当前阶段：等待后端状态更新';
  const progress = document.createElement('div');
  progress.className = 'pipeline-debug-progress';
  if (progressPercent === null && !isTaskTerminal(statusKey)) {
    progress.classList.add('is-indeterminate');
  }
  const progressBar = document.createElement('span');
  progressBar.className = 'pipeline-debug-progress-bar';
  progressBar.style.width = progressPercent === null ? '36%' : `${progressPercent}%`;
  progress.appendChild(progressBar);
  header.append(headerTop, subtitle, stage, progress);
  root.appendChild(header);

  const major = document.createElement('section');
  major.className = 'pipeline-debug-section';
  const majorTitle = document.createElement('h4');
  majorTitle.textContent = '大流程耗时';
  major.appendChild(majorTitle);
  const chips = document.createElement('div');
  chips.className = 'pipeline-debug-chip-grid';
  appendMetricChip(chips, '实时运行', timing.live_elapsed_seconds, '等待开始');
  appendMetricChip(chips, '文档解析', timing.document_seconds);
  appendMetricChip(chips, '问答生成', timing.generation_seconds);
  appendMetricChip(chips, '评估', timing.evaluation_seconds);
  appendMetricChip(chips, '存储输出', timing.storage_seconds);
  appendMetricChip(chips, '总耗时', timing.total_seconds);
  major.appendChild(chips);
  root.appendChild(major);

  const detail = timing.generation_detail || {};
  const generation = document.createElement('section');
  generation.className = 'pipeline-debug-section';
  const generationTitle = document.createElement('h4');
  generationTitle.textContent = 'QA 生成墙钟细分';
  generation.appendChild(generationTitle);
  if (detail.wall_detail_unavailable) {
    const legacyNotice = document.createElement('div');
    legacyNotice.className = 'pipeline-debug-empty';
    legacyNotice.textContent = '这个任务创建时还没有记录墙钟细分；新任务会显示可相加的小阶段耗时。';
    generation.appendChild(legacyNotice);
  }
  const genGrid = document.createElement('div');
  genGrid.className = 'pipeline-debug-chip-grid';
  const generationBreakdownTotal = sumNumbers(
    detail.candidate_question_seconds,
    detail.retrieval_seconds,
    detail.answer_generation_seconds,
    detail.validation_and_bookkeeping_seconds,
    detail.scheduler_gap_seconds,
  );
  appendMetricChip(genGrid, 'QA 生成合计', firstNumber(detail.document_total_seconds, timing.generation_seconds));
  appendMetricChip(genGrid, '候选题生成', firstNumber(detail.candidate_question_seconds));
  appendMetricChip(genGrid, '检索', firstNumber(detail.retrieval_seconds));
  appendMetricChip(genGrid, '答案生成', firstNumber(detail.answer_generation_seconds));
  appendMetricChip(genGrid, '校验/丢弃', firstNumber(detail.validation_and_bookkeeping_seconds));
  appendMetricChip(genGrid, '调度/等待', firstNumber(detail.scheduler_gap_seconds));
  appendMetricChip(genGrid, '细分合计', generationBreakdownTotal);
  generation.appendChild(genGrid);
  const genMeta = document.createElement('div');
  genMeta.className = 'pipeline-debug-kv-grid';
  appendTextMetric(genMeta, 'chunk 总数', firstNumber(detail.chunks_total, safeStatus.chunk_count));
  appendTextMetric(genMeta, 'unit 总数', firstNumber(detail.generation_units_total, safeStatus.generation_units_total));
  appendTextMetric(genMeta, '已完成 unit', firstNumber(detail.generation_units_completed, safeStatus.generation_units_done));
  appendTextMetric(genMeta, '生成 QA 数', firstNumber(detail.qa_generated));
  appendTextMetric(genMeta, '总题数上限', firstNumber(detail.qa_total_limit, safeStatus.qa_total_limit));
  appendTextMetric(genMeta, '上限范围', detail.qa_total_limit_scope || safeStatus.qa_total_limit_scope || 'per_file');
  appendTextMetric(genMeta, 'unit 最大尝试次数', safeStatus.chunk_max_attempts);
  appendTextMetric(genMeta, 'LLM/VLM API 请求并发', safeStatus.llm_max_concurrent_requests || 'Docker 环境默认');
  const retrievalConfig = safeStatus.retrieval_config || {};
  appendTextMetric(genMeta, '检索模式', retrievalConfig.retrieval_mode || 'hybrid');
  appendTextMetric(genMeta, 'evidence topK', retrievalConfig.semantic_top_k ?? '3');
  appendTextMetric(genMeta, '轻量重排候选', retrievalConfig.rerank_top_n ?? '12');
  appendTextMetric(genMeta, '前端范围策略', retrievalConfig.answer_scope_policy || 'source_primary');
  generation.appendChild(genMeta);
  root.appendChild(generation);

  const units = (timing.generation_unit_details || timing.generation_chunk_details || []).slice().sort((a, b) => {
    return Number(a && (a.unit_index || a.anchor_chunk_index || a.chunk_index) || 0)
      - Number(b && (b.unit_index || b.anchor_chunk_index || b.chunk_index) || 0);
  });
  const chunkSection = document.createElement('section');
  chunkSection.className = 'pipeline-debug-section';
  const chunkTitle = document.createElement('h4');
  chunkTitle.textContent = 'generation unit 明细';
  chunkSection.appendChild(chunkTitle);
  if (!units.length) {
    const empty = document.createElement('div');
    empty.className = 'pipeline-debug-empty';
    empty.textContent = '生成阶段完成一个 generation unit 后会显示明细。';
    chunkSection.appendChild(empty);
  } else {
    const tableWrap = document.createElement('div');
    tableWrap.className = 'pipeline-debug-chunk-table-wrap';
    const table = document.createElement('table');
    table.className = 'pipeline-debug-chunk-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    [
      'unit',
      '类型',
      '模式',
      '来源 chunk',
      '状态',
      '候选题',
      '进入答案',
      '有效 QA',
      '总耗时',
      '候选',
      '检索',
      '答案',
      '丢弃或错误',
      '原始响应',
    ].forEach((label) => {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    units.forEach((chunk) => {
      const row = document.createElement('tr');
      const ct = chunk.timing && typeof chunk.timing === 'object' ? chunk.timing : {};
      appendChunkTableCell(row, debugCountText(chunk.unit_index || chunk.chunk_index, '?'), 'mono');
      appendChunkTableCell(row, chunk.unit_type || 'leaf');
      appendChunkTableCell(row, chunk.qa_mode || 'point');
      appendChunkTableCell(
        row,
        Array.isArray(chunk.source_chunk_indexes) && chunk.source_chunk_indexes.length
          ? chunk.source_chunk_indexes.join(',')
          : debugCountText(chunk.anchor_chunk_index || chunk.chunk_index, '?'),
        'mono',
      );
      const stateCell = appendChunkTableCell(row, chunkStatusText(chunk));
      stateCell.dataset.state = chunkStatusText(chunk);
      appendChunkTableCell(row, debugCountText(chunk.candidate_questions));
      appendChunkTableCell(row, debugCountText(chunk.candidates_considered));
      appendChunkTableCell(row, debugCountText(chunk.valid_items, '未返回'));
      appendChunkTableCell(row, debugSecondsText(ct.chunk_total_seconds), 'mono');
      appendChunkTableCell(row, debugSecondsText(ct.candidate_question_seconds), 'mono');
      appendChunkTableCell(row, debugSecondsText(ct.retrieval_seconds), 'mono');
      appendChunkTableCell(row, debugSecondsText(ct.answer_generation_seconds), 'mono');
      const reasonCell = document.createElement('td');
      reasonCell.className = 'pipeline-debug-reason-cell';
      const reasons = chunkReasonParts(chunk);
      if (!reasons.length) {
        reasonCell.textContent = '无';
      } else {
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = reasons.length === 1 ? '查看 1 项' : `查看 ${reasons.length} 项`;
        const body = document.createElement('div');
        body.textContent = reasons.join('；');
        details.append(summary, body);
        reasonCell.appendChild(details);
      }
      row.appendChild(reasonCell);
      const actionCell = document.createElement('td');
      actionCell.className = 'pipeline-debug-action-cell';
      const rawBtn = document.createElement('button');
      rawBtn.type = 'button';
      rawBtn.className = 'secondary compact';
      rawBtn.textContent = '查看';
      rawBtn.addEventListener('click', () => openChunkDebugRawModal(chunk.anchor_chunk_index || chunk.chunk_index));
      actionCell.appendChild(rawBtn);
      row.appendChild(actionCell);
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    chunkSection.appendChild(tableWrap);
  }
  root.appendChild(chunkSection);

  const raw = document.createElement('details');
  raw.className = 'pipeline-debug-raw';
  raw.open = Boolean(options.rawOpen);
  raw.addEventListener('toggle', () => {
    pipelineRawJsonOpen = Boolean(raw.open);
  });
  const rawSummary = document.createElement('summary');
  rawSummary.textContent = '原始 JSON';
  const rawPre = document.createElement('pre');
  rawPre.textContent = JSON.stringify(safeStatus, null, 2);
  raw.appendChild(rawSummary);
  raw.appendChild(rawPre);
  root.appendChild(raw);

  return root;
}

function updatePipelineTaskHint(text) {
  const hintEl = $('#pipelineTaskHint');
  if (!hintEl) return;
  hintEl.textContent =
    String(text || '').trim() ||
    '页面会记住最近查看或正在运行的流水线任务。“终止当前任务”使用当前选中的 task_id；输入 task_id 后可点“终止此 task_id”。如果任务已经结束或变成僵尸记录，可直接在下方任务列表里删除记录。';
}

function clearPipelineOutputsView() {
  lastCsvPath = null;
  lastPipelineOutputs = [];
  currentConsolidatedJsonPath = '';
  qaReviewState.enabled = false;
  qaReviewState.items = [];
  qaReviewState.selectedIds.clear();
  qaReviewState.selectAllTask = false;
  updateQaReviewToolbar();
  const panel = document.querySelector('#pipelineOutputsList');
  if (panel) panel.innerHTML = '';
}

function applyPipelineStatus(status, { base = '', taskId = '', activateStatus = false } = {}) {
  const normalizedTaskId = String(taskId || status?.task_id || '').trim();
  if (!normalizedTaskId) return;
  latestPipelineStatus = status && typeof status === 'object' ? status : null;
  setTaskSelection(normalizedTaskId, { active: !isTaskTerminal(status?.status) });
  updatePipelineStatusView(status, { activateStatus });
  rememberTask({ task_id: normalizedTaskId, ...status });
  renderPipelineTaskHistory();

  const outputs = Array.isArray(status?.outputs) ? status.outputs : [];
  if (outputs.length && base) {
    const shouldAutoShowOutputs =
      !activateStatus &&
      isTaskTerminal(status?.status) &&
      pipelineTaskActiveTab === 'status' &&
      !pipelineRawJsonOpen &&
      pipelineOutputsAutoShownForTaskId !== normalizedTaskId;
    handlePipelineOutputs(base, outputs, {
      autoShow: shouldAutoShowOutputs,
      taskId: normalizedTaskId,
    });
  } else {
    clearPipelineOutputsView();
  }
  renderDwIntegratedProgress(status);

  const statusText = String(status?.status || '').trim() || 'unknown';
  const msg = translatePipelineMessage(status?.message);
  updatePipelineTaskHint(
    `当前查看 task_id=${normalizedTaskId}（${statusText}${msg ? `，${msg}` : ''}）。“终止当前任务”会使用这个 task_id；也可在输入框粘贴其他 task_id 后点“终止此 task_id”。`,
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
    applyPipelineStatus(data, { base, taskId: normalized, activateStatus: !silent });
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

function resolveCurrentTaskIdForCancel() {
  const page = getRuntimePageState();
  return String(page.activeTaskId || page.selectedTaskId || lastTaskId || $('#taskIdInput')?.value || '').trim();
}

function handleCancelCurrentTask() {
  const taskId = resolveCurrentTaskIdForCancel();
  return handleCancelTask(taskId);
}

function handleCancelInputTask() {
  const taskId = String($('#taskIdInput')?.value || '').trim();
  return handleCancelTask(taskId);
}

async function handleCancelTask(explicitTaskId) {
  const base = getApiBaseUrl();
  const statusEl = $('#pipelineStatus');
  const inputId = typeof explicitTaskId === 'string' ? explicitTaskId.trim() : '';
  const taskId = inputId || lastTaskId;
  if (!taskId) {
    if (statusEl) statusEl.textContent = '请先填入 task_id，或先恢复/查询一个当前任务再终止';
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

function handlePipelineOutputs(base, outputs, options = {}) {
  if (!Array.isArray(outputs) || !outputs.length) return;
  lastPipelineOutputs = outputs.slice();
  const usable = outputs.filter(
    (o) =>
      o &&
      (o.consolidated_json ||
        o.consolidated_csv ||
        String(o.history_source || '').trim().toLowerCase() === 'milvus' ||
        o.milvus_task_id ||
        o.task_id),
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
  if (options.autoShow) {
    pipelineOutputsAutoShownForTaskId = String(options.taskId || lastTaskId || '');
    activateTaskTab('outputs');
  }
  refreshReviewWorkspaceState();
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
      currentConsolidatedJsonPath = '';
      qaReviewState.enabled = false;
      qaReviewState.items = [];
      qaReviewState.selectedIds.clear();
      qaReviewState.selectAllTask = false;
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
    currentConsolidatedJsonPath = '';
    qaReviewState.enabled = false;
    qaReviewState.items = [];
    qaReviewState.selectedIds.clear();
    qaReviewState.selectAllTask = false;
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

    const label = document.createElement('div');
    label.className = 'pipeline-output-title';
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
    label.textContent = parts.join(' | ') || `文件 #${idx + 1}`;
    li.appendChild(label);

    const timing = o.timing || {};
    const timingRow = document.createElement('div');
    timingRow.className = 'pipeline-output-timing';
    const outputTimingValues = [
      ['文档解析', timing.ocr_seconds],
      ['问答生成', timing.generation_seconds],
      ['评估', sumNumbers(timing.unsupervised_seconds, timing.evaluation_seconds)],
    ];
    let outputTimingTotal = 0;
    let outputTimingFound = false;
    outputTimingValues.forEach((pair) => {
      const value = asNumber(pair[1]);
      if (value === null) return;
      outputTimingTotal += value;
      outputTimingFound = true;
      appendMetricChip(timingRow, pair[0], value);
    });
    if (outputTimingFound) {
      appendMetricChip(timingRow, '总耗时', outputTimingTotal);
      li.appendChild(timingRow);
    }

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

    const milvusTaskId = o.milvus_task_id || o.task_id || lastTaskId;
    if (milvusTaskId) {
      const btnMilvus = document.createElement('button');
      btnMilvus.type = 'button';
      btnMilvus.textContent = '查看入库记录';
      btnMilvus.addEventListener('click', () => {
        loadPipelineMilvusHistory(base, milvusTaskId, {
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

function setupDwDocumentPanel() {
  const form = $('#dwJobForm');
  if (form) form.addEventListener('submit', handleDwJobSubmit);

  const loadBtn = $('#dwLoadJobBtn');
  if (loadBtn) {
    loadBtn.addEventListener('click', () => {
      loadDwJobById($('#dwJobIdInput')?.value || '', { resumePolling: true, silent: false }).catch(() => {});
    });
  }

  const jobInput = $('#dwJobIdInput');
  if (jobInput) {
    jobInput.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      e.preventDefault();
      loadDwJobById($('#dwJobIdInput')?.value || '', { resumePolling: true, silent: false }).catch(() => {});
    });
  }

  const restoreBtn = $('#dwRestoreJobBtn');
  if (restoreBtn) {
    restoreBtn.addEventListener('click', () => {
      const page = getRuntimePageState();
      const rememberedId =
        String(page.activeDwJobId || '').trim() ||
        String(page.selectedDwJobId || '').trim() ||
        String($('#dwJobIdInput')?.value || '').trim();
      if (!rememberedId) {
        notify('当前没有可恢复的 document_job_id', 'warning');
        return;
      }
      loadDwJobById(rememberedId, { resumePolling: true, silent: false }).catch(() => {});
    });
  }

  const cancelBtn = $('#dwCancelJobBtn');
  if (cancelBtn) cancelBtn.addEventListener('click', handleCancelDwJob);

  const syncBtn = $('#dwSyncToIntegratedBtn');
  if (syncBtn) syncBtn.addEventListener('click', syncDwOptionsToIntegratedPipeline);
}

function normalizeDwJobRecord(job) {
  if (!job || typeof job !== 'object') return null;
  const jobId = String(job.job_id || '').trim();
  if (!jobId) return null;
  return {
    job_id: jobId,
    status: String(job.status || ''),
    message: String(job.message || ''),
    input_filename: String(job.input_filename || ''),
    updated_at: job.updated_at ?? job.finished_at ?? job.created_at ?? '',
    output_format: String(job.output_format || job.params?.output_format || ''),
  };
}

function getRecentDwJobs() {
  const page = getRuntimePageState();
  return Array.isArray(page.recentDwJobs) ? page.recentDwJobs : [];
}

function replaceRecentDwJobs(jobs) {
  const normalized = Array.isArray(jobs)
    ? jobs
        .map((job) => normalizeDwJobRecord(job))
        .filter(Boolean)
        .slice(0, 12)
    : [];
  mutateRuntimePage((page) => {
    const ids = new Set(normalized.map((job) => job.job_id));
    page.recentDwJobs = normalized;
    if (page.selectedDwJobId && !ids.has(String(page.selectedDwJobId || ''))) {
      page.selectedDwJobId = normalized[0] ? normalized[0].job_id : '';
    }
    if (page.activeDwJobId && !ids.has(String(page.activeDwJobId || ''))) {
      page.activeDwJobId = '';
    }
  });
  return normalized;
}

function removeRecentDwJob(jobId) {
  const normalized = String(jobId || '').trim();
  if (!normalized) return getRecentDwJobs();
  return mutateRuntimePage((page) => {
    const prev = Array.isArray(page.recentDwJobs) ? page.recentDwJobs : [];
    page.recentDwJobs = prev.filter((it) => String(it.job_id || '') !== normalized);
    if (String(page.selectedDwJobId || '') === normalized) {
      page.selectedDwJobId = page.recentDwJobs[0] ? String(page.recentDwJobs[0].job_id || '') : '';
    }
    if (String(page.activeDwJobId || '') === normalized) {
      page.activeDwJobId = '';
    }
  }).recentDwJobs || [];
}

function rememberDwJob(job) {
  const record = normalizeDwJobRecord(job);
  if (!record) return getRecentDwJobs();
  return mutateRuntimePage((page) => {
    const prev = Array.isArray(page.recentDwJobs) ? page.recentDwJobs : [];
    const merged = [record, ...prev.filter((it) => String(it.job_id || '') !== record.job_id)];
    page.recentDwJobs = merged.slice(0, 12);
    page.selectedDwJobId = record.job_id;
    page.activeDwJobId = isTaskTerminal(record.status)
      ? (page.activeDwJobId === record.job_id ? '' : page.activeDwJobId || '')
      : record.job_id;
  }).recentDwJobs || [];
}

function setDwJobSelection(jobId, { active = null } = {}) {
  const normalized = String(jobId || '').trim();
  const input = $('#dwJobIdInput');
  if (input && input.value !== normalized) {
    input.value = normalized;
    persistUiField(input);
  }
  mutateRuntimePage((page) => {
    page.selectedDwJobId = normalized;
    if (active === true) page.activeDwJobId = normalized;
    else if (active === false && page.activeDwJobId === normalized) page.activeDwJobId = '';
  });
}

function renderDwJobHistory() {
  const wrap = $('#dwJobHistory');
  if (!wrap) return;
  const jobs = getRecentDwJobs();
  wrap.replaceChildren();
  if (!jobs.length) {
    const empty = document.createElement('div');
    empty.className = 'task-history-empty';
    empty.textContent = '暂无最近文档任务。提交一次文档解析或输入 document_job_id 查询后，这里会保留最近记录。';
    wrap.appendChild(empty);
    return;
  }

  jobs.forEach((job) => {
    const row = document.createElement('div');
    row.className = 'task-history-item';

    const main = document.createElement('div');
    main.className = 'task-history-main';

    const title = document.createElement('div');
    title.className = 'task-history-title';
    title.appendChild(buildDwStatusPill(job.status || 'unknown'));

    const name = document.createElement('span');
    name.className = 'task-history-name';
    name.textContent = job.input_filename || '文档解析';
    title.appendChild(name);

    const id = document.createElement('div');
    id.className = 'task-history-id';
    id.textContent = job.job_id;

    const meta = document.createElement('div');
    meta.className = 'task-history-meta';
    meta.textContent = [
      job.output_format ? `输出: ${job.output_format}` : '',
      job.updated_at ? `更新: ${fmtRuntimeTime(job.updated_at)}` : '',
      job.message || '',
    ].filter(Boolean).join(' | ');

    main.append(title, id, meta);

    const actions = document.createElement('div');
    actions.className = 'task-history-actions';
    const viewBtn = document.createElement('button');
    viewBtn.type = 'button';
    viewBtn.className = 'secondary';
    viewBtn.textContent = '查看';
    viewBtn.addEventListener('click', () => {
      loadDwJobById(job.job_id, { resumePolling: true, silent: false }).catch(() => {});
    });
    actions.appendChild(viewBtn);

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'secondary';
    removeBtn.textContent = '移除';
    removeBtn.addEventListener('click', () => {
      removeRecentDwJob(job.job_id);
      renderDwJobHistory();
    });
    actions.appendChild(removeBtn);

    row.append(main, actions);
    wrap.appendChild(row);
  });
}

async function refreshDwJobHistory({ silent = false } = {}) {
  const base = getApiBaseUrl();
  try {
    const data = await fetchJson(`${base}/document-processing/jobs?limit=20`);
    const jobs = replaceRecentDwJobs(Array.isArray(data?.jobs) ? data.jobs : []);
    renderDwJobHistory();
    if (!silent) updateDwJobHint(`已刷新最近文档任务，共 ${jobs.length} 条。`);
    return jobs;
  } catch (err) {
    renderDwJobHistory();
    if (!silent) updateDwJobHint(`刷新最近文档任务失败：${String(err)}`);
    return getRecentDwJobs();
  }
}

function updateDwJobHint(text) {
  const hint = $('#dwJobHint');
  if (!hint) return;
  hint.textContent =
    String(text || '').trim() ||
    '页面会记住最近查看或正在运行的文档任务；一体流程的文档预处理进度会显示在下方。';
}

function collectDwJobFormData() {
  const fileInput = $('#dwFileInput');
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    throw new Error('请先选择要解析的文档');
  }
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  fd.append('output_format', $('#dwOutputFormat')?.value || 'text');
  fd.append('docx_strategy', $('#dwDocxStrategy')?.value || 'pdf');
  fd.append('enable_image_analysis', $('#dwEnableImageAnalysis')?.checked !== false ? 'true' : 'false');
  fd.append('enable_classification', $('#dwEnableClassification')?.checked ? 'true' : 'false');
  fd.append('classification_confidence_threshold', $('#dwClassificationThreshold')?.value || '0.9');
  fd.append('remove_watermark', $('#dwRemoveWatermark')?.checked ? 'true' : 'false');
  fd.append('watermark_dpi', $('#dwWatermarkDpi')?.value || '200');
  fd.append('replace_images', $('#dwReplaceImages')?.checked !== false ? 'true' : 'false');
  fd.append('use_api', $('#dwUseApi')?.checked !== false ? 'true' : 'false');

  [
    ['vlm_api_base', '#dwVlmApiBase'],
    ['vlm_model_name', '#dwVlmModelName'],
    ['vlm_api_key', '#dwVlmApiKey'],
    ['vlm_api_type', '#dwVlmApiType'],
    ['vlm_model_version', '#dwVlmModelVersion'],
  ].forEach(([key, selector]) => {
    const value = String($(selector)?.value || '').trim();
    if (value) fd.append(key, value);
  });
  return fd;
}

async function handleDwJobSubmit(e) {
  e.preventDefault();
  const btn = e?.submitter || $('#dwJobForm button[type="submit"]');
  const base = getApiBaseUrl();
  try {
    setBtnLoading(btn, true);
    renderDwJobStatus({ status: 'submitting', message: '正在提交文档任务…' }, { base });
    const data = await fetchJson(`${base}/document-processing/jobs`, {
      method: 'POST',
      body: collectDwJobFormData(),
    });
    applyDwJobStatus(data, { base, jobId: data.job_id });
    if (data.job_id) startDwJobPolling(base, data.job_id);
    notify('文档任务已提交', 'success');
  } catch (err) {
    renderDwJobStatus({ status: 'failed', message: String(err) }, { base });
    notify(`提交文档任务失败：${String(err)}`, 'error');
  } finally {
    setBtnLoading(btn, false);
  }
}

async function fetchDwJobMaybe(base, jobId) {
  const resp = await fetch(`${base}/document-processing/jobs/${encodeURIComponent(jobId)}`);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = { raw: text };
  }
  if (resp.status === 404) return { found: false, data };
  if (!resp.ok) {
    const detail = data && (data.detail || data.message || data.error);
    throw new Error(detail || text || resp.statusText || `HTTP ${resp.status}`);
  }
  return { found: true, data };
}

async function loadDwJobById(jobId, { resumePolling = true, silent = false } = {}) {
  const normalized = String(jobId || '').trim();
  if (!normalized) {
    if (!silent) notify('请先输入 document_job_id', 'warning');
    return null;
  }
  const base = getApiBaseUrl();
  if (!resumePolling && currentDwJobPoller) {
    clearInterval(currentDwJobPoller);
    currentDwJobPoller = null;
  }
  try {
    if (!silent) updateDwJobHint(`正在加载 document_job_id=${normalized}…`);
    const result = await fetchDwJobMaybe(base, normalized);
    if (!result.found) {
      removeRecentDwJob(normalized);
      renderDwJobHistory();
      if (!silent) updateDwJobHint(`document_job_id=${normalized} 不存在，已从最近任务移除。`);
      return null;
    }
    const data = result.data;
    applyDwJobStatus(data, { base, jobId: normalized });
    if (!isTaskTerminal(data?.status) && resumePolling) {
      startDwJobPolling(base, normalized);
    } else if (isTaskTerminal(data?.status) && currentDwJobPoller) {
      clearInterval(currentDwJobPoller);
      currentDwJobPoller = null;
    }
    return data;
  } catch (err) {
    if (!silent) updateDwJobHint(`查询 document_job_id=${normalized} 失败：${String(err)}`);
    throw err;
  }
}

function startDwJobPolling(base, jobId) {
  const normalized = String(jobId || '').trim();
  if (!normalized) return;
  setDwJobSelection(normalized, { active: true });
  if (currentDwJobPoller) {
    clearInterval(currentDwJobPoller);
    currentDwJobPoller = null;
  }
  currentDwJobPoller = setInterval(async () => {
    try {
      const result = await fetchDwJobMaybe(base, normalized);
      if (!result.found) {
        clearInterval(currentDwJobPoller);
        currentDwJobPoller = null;
        removeRecentDwJob(normalized);
        renderDwJobHistory();
        updateDwJobHint(`document_job_id=${normalized} 不存在，已从最近任务移除。`);
        return;
      }
      const data = result.data;
      applyDwJobStatus(data, { base, jobId: normalized });
      if (isTaskTerminal(data?.status)) {
        clearInterval(currentDwJobPoller);
        currentDwJobPoller = null;
      }
    } catch (err) {
      clearInterval(currentDwJobPoller);
      currentDwJobPoller = null;
      updateDwJobHint(`文档任务轮询异常：${String(err)}`);
    }
  }, 2000);
}

async function handleCancelDwJob() {
  const page = getRuntimePageState();
  const jobId =
    String($('#dwJobIdInput')?.value || '').trim() ||
    String(page.activeDwJobId || '').trim() ||
    String(page.selectedDwJobId || '').trim();
  if (!jobId) {
    notify('请先输入 document_job_id 或提交过一次文档任务', 'warning');
    return;
  }
  const base = getApiBaseUrl();
  try {
    const data = await fetchJson(`${base}/document-processing/jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: 'POST',
    });
    if (data?.job) applyDwJobStatus(data.job, { base, jobId });
    notify(data?.canceled ? '已请求取消文档任务' : '当前文档任务不可取消', data?.canceled ? 'success' : 'warning');
  } catch (err) {
    notify(`取消文档任务失败：${String(err)}`, 'error');
  }
}

function applyDwJobStatus(job, { base = '', jobId = '' } = {}) {
  const normalizedJobId = String(jobId || job?.job_id || '').trim();
  if (normalizedJobId) setDwJobSelection(normalizedJobId, { active: !isTaskTerminal(job?.status) });
  rememberDwJob({ job_id: normalizedJobId, ...job });
  renderDwJobHistory();
  renderDwJobStatus(job, { base, jobId: normalizedJobId });

  const statusText = String(job?.status || 'unknown');
  const msg = String(job?.message || '').trim();
  updateDwJobHint(
    `当前查看 document_job_id=${normalizedJobId || '-'}（${statusText}${msg ? `，${msg}` : ''}）。`,
  );
}

function renderDwJobStatus(job, { base = '', jobId = '' } = {}) {
  const statusEl = $('#dwJobStatus');
  if (statusEl) statusEl.textContent = JSON.stringify(job || {}, null, 2);
  renderDwProgressSummary(job);
  renderDwProgressFiles($('#dwProgressFiles'), job?.file_progress || {});
  renderDwOutputLinks(job, { base, jobId });
}

function renderDwProgressSummary(job) {
  const wrap = $('#dwProgressSummary');
  if (!wrap) return;
  wrap.replaceChildren();
  if (!job || typeof job !== 'object') {
    wrap.appendChild(buildDwSummaryChip('状态', '未加载'));
    return;
  }
  wrap.appendChild(buildDwSummaryChip('状态', job.status || 'unknown'));
  if (job.input_filename) wrap.appendChild(buildDwSummaryChip('文件', job.input_filename));
  if (job.output_format || job.params?.output_format) {
    wrap.appendChild(buildDwSummaryChip('输出', job.output_format || job.params.output_format));
  }
  const metrics = collectDwProgressMetrics(job.file_progress || {});
  if (metrics.total_pages !== undefined) wrap.appendChild(buildDwSummaryChip('页数', metrics.total_pages));
  if (metrics.total_images !== undefined) wrap.appendChild(buildDwSummaryChip('图片', metrics.total_images));
  if (metrics.analyzed_images !== undefined) wrap.appendChild(buildDwSummaryChip('已理解', metrics.analyzed_images));
  if (metrics.chunks !== undefined) wrap.appendChild(buildDwSummaryChip('chunk', metrics.chunks));
  if (metrics.accepted_images !== undefined) wrap.appendChild(buildDwSummaryChip('回填图片', metrics.accepted_images));
  if (job.updated_at) wrap.appendChild(buildDwSummaryChip('更新', fmtRuntimeTime(job.updated_at)));
}

function collectDwProgressMetrics(fileProgress) {
  const metrics = {};
  Object.values(fileProgress || {}).forEach((fileEntry) => {
    const stages = fileEntry && typeof fileEntry === 'object' ? fileEntry.stages || {} : {};
    Object.values(stages || {}).forEach((stage) => {
      const extra = stage && typeof stage === 'object' ? stage.extra || {} : {};
      ['total_pages', 'total_images', 'analyzed_images', 'chunks', 'accepted_images'].forEach((key) => {
        if (extra[key] === undefined || extra[key] === null || extra[key] === '') return;
        metrics[key] = extra[key];
      });
    });
  });
  return metrics;
}

function buildDwSummaryChip(label, value) {
  const chip = document.createElement('span');
  chip.className = 'dw-summary-chip';
  chip.textContent = `${label}: ${String(value ?? '-')}`;
  return chip;
}

function renderDwOutputLinks(job, { base = '', jobId = '' } = {}) {
  const wrap = $('#dwOutputLinks');
  if (!wrap) return;
  wrap.replaceChildren();
  const normalizedJobId = String(jobId || job?.job_id || '').trim();
  const files = job?.files && typeof job.files === 'object' ? job.files : {};
  if (!normalizedJobId || !Object.keys(files).length) return;
  const linkDefs = [
    ['text', '下载整合文本'],
    ['markdown', '下载整合 Markdown'],
    ['ocr_markdown', '下载 OCR Markdown'],
    ['summary', '下载 OCR summary'],
    ['image_analysis_summary', '下载图片理解 summary'],
  ];
  linkDefs.forEach(([key, label]) => {
    if (!files[key]) return;
    const a = document.createElement('a');
    a.href = `${base}/document-processing/jobs/${encodeURIComponent(normalizedJobId)}/download?file_key=${encodeURIComponent(key)}`;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = label;
    wrap.appendChild(a);
  });
}

function renderDwProgressFiles(container, fileProgress, { onlyIntegratedDw = false } = {}) {
  if (!container) return;
  container.replaceChildren();
  const entries = Object.entries(fileProgress || {});
  const rendered = [];
  entries.forEach(([filename, fileEntry]) => {
    const stages = fileEntry && typeof fileEntry === 'object' ? fileEntry.stages || {} : {};
    const stageEntries = Object.entries(stages).filter(([stage]) => {
      if (!onlyIntegratedDw) return true;
      const normalizedStage = String(stage || '');
      return normalizedStage.startsWith('doc_') || normalizedStage.startsWith('dw_') || normalizedStage === 'image_classification';
    });
    if (!stageEntries.length) return;
    rendered.push(buildDwFileProgressItem(filename, fileEntry, stageEntries));
  });
  if (!rendered.length) {
    const empty = document.createElement('div');
    empty.className = 'task-history-empty';
    empty.textContent = onlyIntegratedDw ? '当前任务暂无文档预处理进度。' : '暂无文档解析进度。';
    container.appendChild(empty);
    return;
  }
  rendered.forEach((el) => container.appendChild(el));
}

function buildDwFileProgressItem(filename, fileEntry, stageEntries) {
  const item = document.createElement('article');
  item.className = 'dw-file-progress-item';

  const head = document.createElement('div');
  head.className = 'dw-file-progress-head';
  head.appendChild(buildDwStatusPill(fileEntry?.status || 'processing'));
  const name = document.createElement('div');
  name.className = 'dw-file-progress-name';
  name.textContent = String(filename || 'upload');
  head.appendChild(name);
  item.appendChild(head);

  const list = document.createElement('div');
  list.className = 'dw-stage-list';
  stageEntries
    .sort(([left], [right]) => {
      const leftIndex = DOC_STAGE_ORDER.indexOf(left);
      const rightIndex = DOC_STAGE_ORDER.indexOf(right);
      return (leftIndex < 0 ? 999 : leftIndex) - (rightIndex < 0 ? 999 : rightIndex);
    })
    .forEach(([stage, stageEntry]) => {
      list.appendChild(buildDwStageRow(stage, stageEntry || {}));
    });
  item.appendChild(list);
  return item;
}

function buildDwStageRow(stage, entry) {
  const row = document.createElement('div');
  row.className = 'dw-stage-row';

  const title = document.createElement('div');
  title.className = 'dw-stage-title';
  const chip = document.createElement('span');
  chip.className = 'dw-stage-chip';
  chip.dataset.state = String(entry.state || 'processing').toLowerCase();
  chip.textContent = String(entry.state || 'processing');
  title.appendChild(chip);
  const name = document.createElement('span');
  name.className = 'dw-stage-name';
  name.textContent = DOC_STAGE_LABELS[stage] || stage;
  title.appendChild(name);

  const body = document.createElement('div');
  body.className = 'dw-stage-message';
  body.textContent = String(entry.message || '');
  const extraText = formatDwStageExtra(entry.extra || {});
  if (extraText) {
    const extra = document.createElement('div');
    extra.className = 'dw-stage-extra';
    extra.textContent = extraText;
    body.appendChild(extra);
  }

  row.append(title, body);
  return row;
}

function buildDwStatusPill(status) {
  const pill = document.createElement('span');
  pill.className = 'status-pill';
  pill.dataset.status = String(status || 'unknown').toLowerCase();
  pill.textContent = String(status || 'unknown');
  return pill;
}

function formatDwStageExtra(extra) {
  if (!extra || typeof extra !== 'object') return '';
  const labels = {
    sub_stage: '子阶段',
    total_pages: '页',
    total_images: '图片',
    analyzed_images: '已理解',
    failed_images: '失败图片',
    success_count: '成功',
    failed_count: '失败',
    chunks: 'chunk',
    accepted_images: '接受图片',
    checked_images: '判断图片',
    ocr_seconds: 'OCR 秒',
    image_analysis_seconds: '图片理解秒',
    elapsed_seconds: '耗时秒',
    processing_time: '处理秒',
    error: '错误',
    image_id: '图片 ID',
    image_index: '序号',
    output_chars: '输出字符',
    prompt_key: 'prompt',
    mode: '模式',
    docx_strategy: 'DOCX',
    requires_ocr: '需 OCR',
    enabled: '启用',
    threshold: '阈值',
    classification_errors: '分类错误',
  };
  const parts = [];
  Object.entries(labels).forEach(([key, label]) => {
    if (extra[key] === undefined || extra[key] === null || extra[key] === '') return;
    parts.push(`${label}: ${shortDwValue(extra[key])}`);
  });
  return parts.join(' | ');
}

function shortDwValue(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(Math.round(value * 100) / 100) : String(value);
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  return text.length > 220 ? `${text.slice(0, 220)}…` : text;
}

function renderDwIntegratedProgress(status) {
  const wrap = $('#dwIntegratedProgress');
  if (!wrap) return;
  renderDwProgressFiles(wrap, status?.file_progress || {}, { onlyIntegratedDw: true });
}

function syncDwOptionsToIntegratedPipeline() {
  const modeEl = $('#pipelineProcessingMode');
  if (modeEl) {
    modeEl.value = 'integrated';
    modeEl.dispatchEvent(new Event('change', { bubbles: true }));
    persistUiField(modeEl);
  }
  const mappings = [
    ['#dwRemoveWatermark', '#integratedRemoveWatermark', 'checked'],
    ['#dwReplaceImages', '#integratedReplaceImages', 'checked'],
    ['#dwEnableImageAnalysis', '#integratedEnableImageAnalysis', 'checked'],
    ['#dwEnableClassification', '#integratedEnableImageClassification', 'checked'],
    ['#dwClassificationThreshold', '#integratedClassificationThreshold', 'value'],
    ['#dwVlmApiBase', '#integratedVlmApiBase', 'value'],
    ['#dwVlmModelName', '#integratedVlmModelName', 'value'],
    ['#dwVlmApiKey', '#integratedVlmApiKey', 'value'],
    ['#dwVlmApiType', '#integratedVlmApiType', 'value'],
    ['#dwVlmModelVersion', '#integratedVlmModelVersion', 'value'],
    ['#dwWatermarkDpi', '#integratedWatermarkDpi', 'value'],
    ['#dwDocxStrategy', '#integratedDocxStrategy', 'value'],
  ];
  mappings.forEach(([sourceSel, targetSel, prop]) => {
    const source = $(sourceSel);
    const target = $(targetSel);
    if (!source || !target) return;
    if (prop === 'checked') target.checked = source.checked;
    else if (prop === 'value') {
      const value = source.value === 'auto' && targetSel === '#integratedDocxStrategy' ? 'pdf' : source.value;
      target.value = value;
    }
    target.dispatchEvent(new Event('change', { bubbles: true }));
    persistUiField(target);
  });
  const details = $('#integratedDocumentOptions');
  if (details) details.open = true;
  notify('已同步 dw 参数到完整流水线；请在完整流水线区域选择文件并提交。', 'success');
}

async function hydrateDwRuntime() {
  renderDwJobHistory();
  renderDwIntegratedProgress(null);
  const recentJobs = await refreshDwJobHistory({ silent: true });
  const page = getRuntimePageState();
  const rememberedId =
    String(page.activeDwJobId || '').trim() ||
    String(page.selectedDwJobId || '').trim() ||
    String($('#dwJobIdInput')?.value || '').trim();
  const recentList = Array.isArray(recentJobs) ? recentJobs : [];
  const existingIds = new Set(recentList.map((job) => String(job.job_id || '').trim()).filter(Boolean));
  const targetId = rememberedId && existingIds.has(rememberedId)
    ? rememberedId
    : (recentList[0] && String(recentList[0].job_id || '').trim()) || '';
  if (!targetId) {
    updateDwJobHint();
    renderDwJobStatus(null, {});
    return;
  }
  try {
    await loadDwJobById(targetId, { resumePolling: true, silent: true });
  } catch {
    renderDwJobHistory();
  }
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
    currentConsolidatedJsonPath = String(jsonPath || '');
    // 后端 download 接口约定：如果路径以 "outputs/" 开头，则只取 basename
    // consolidated_json 里通常是 "qa/outputs/xxx.json"，这里统一转换一下
    const parts = String(jsonPath).split('/');
    const fileName = parts[parts.length - 1];
    const dlPath = `outputs/${fileName}`;
    const resp = await fetch(`${base}/download/${dlPath}`);
    if (!resp.ok) {
      console.error('加载合并结果失败', resp.status, resp.statusText);
      currentConsolidatedJsonPath = '';
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
      currentConsolidatedJsonPath = '';
      renderMeta({ error: '合并结果文件不是有效 JSON，无法预览' });
    }
  } catch (err) {
    console.error('解析合并 JSON 失败', err);
    currentConsolidatedJsonPath = '';
    renderMeta({ error: '合并结果加载异常' });
  }
}
