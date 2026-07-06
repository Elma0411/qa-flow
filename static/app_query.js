// ---------- Chunk 树与溯源（doc_tree_chunks） ----------

const btnLoadChunkDocs = $('#btnLoadChunkDocs');
const btnLoadChunkTree = $('#btnLoadChunkTree');
const chunkDocSelect = $('#chunkDocSelect');
const chunkDebugState = {
  currentChunkId: '',
  payload: null,
  qaItems: [],
  selectedQaId: '',
  activeView: 'chunk',
};

if (btnLoadChunkDocs) btnLoadChunkDocs.addEventListener('click', handleLoadChunkDocs);
if (btnLoadChunkTree) btnLoadChunkTree.addEventListener('click', handleLoadChunkTree);
const chunkQaDetailOverlay = $('#chunkQaDetailOverlay');
const chunkQaDetailModal = $('#chunkQaDetailModal');
const chunkQaDetailModalClose = $('#chunkQaDetailModalClose');
if (chunkQaDetailOverlay) chunkQaDetailOverlay.addEventListener('click', closeQaDetailModal);
if (chunkQaDetailModalClose) chunkQaDetailModalClose.addEventListener('click', closeQaDetailModal);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeQaDetailModal();
});
if (chunkDocSelect) {
  chunkDocSelect.addEventListener('change', () => {
    // 用户切换文档时自动刷新树（避免多点一次）
    handleLoadChunkTree().catch(() => {});
  });
}

$$('[data-debug-view]').forEach((button) => {
  button.addEventListener('click', () => {
    const view = String(button.dataset.debugView || '').trim() || 'chunk';
    switchChunkDebugView(view);
  });
});

function setChunkOutputs(detailText, qaText) {
  const detailEl = $('#chunkDetailOutput');
  const qaEl = $('#chunkQaOutput');
  if (detailEl) detailEl.textContent = detailText || '';
  if (qaEl) qaEl.textContent = qaText || '';
}

function fmtDebugTs(ts) {
  if (!ts && ts !== 0) return '';
  try {
    const raw = String(ts).trim();
    const date = /^\d+$/.test(raw)
      ? new Date(raw.length > 11 ? Number(raw) : Number(raw) * 1000)
      : new Date(raw);
    return Number.isNaN(date.getTime()) ? '' : date.toLocaleString();
  } catch {
    return '';
  }
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function toScoreText(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : '—';
}

function createDebugEmpty(text) {
  const box = document.createElement('div');
  box.className = 'chunk-debug-empty';
  box.textContent = text;
  return box;
}

function createKvGrid(rows) {
  const grid = document.createElement('div');
  grid.className = 'kv';
  rows.forEach(([label, value]) => {
    if (value === undefined || value === null || value === '') return;
    const row = document.createElement('div');
    row.className = 'kv-row';
    const keyEl = document.createElement('div');
    keyEl.className = 'kv-k';
    keyEl.textContent = String(label);
    const valEl = document.createElement('div');
    valEl.className = 'kv-v';
    valEl.textContent = typeof value === 'string' ? value : String(value);
    row.appendChild(keyEl);
    row.appendChild(valEl);
    grid.appendChild(row);
  });
  return grid;
}

function createCodeSection(title, text) {
  const section = document.createElement('section');
  section.className = 'chunk-debug-section';
  const heading = document.createElement('h4');
  heading.className = 'chunk-debug-section-title';
  heading.textContent = title;
  const body = document.createElement('div');
  body.className = 'chunk-debug-code';
  body.textContent = String(text || '').trim() || '—';
  section.appendChild(heading);
  section.appendChild(body);
  return section;
}

function createPillRow(items) {
  const row = document.createElement('div');
  row.className = 'chunk-debug-pill-row';
  items.forEach((item) => {
    const text = String(item || '').trim();
    if (!text) return;
    const pill = document.createElement('span');
    pill.className = 'chunk-debug-pill';
    pill.textContent = text;
    row.appendChild(pill);
  });
  return row;
}

function createSmallActionButton(text, onClick) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'secondary chunk-debug-mini-action';
  button.textContent = text;
  button.addEventListener('click', onClick);
  return button;
}

function openQaDetailModal(item) {
  const modal = $('#chunkQaDetailModal');
  const overlay = $('#chunkQaDetailOverlay');
  const body = $('#chunkQaDetailModalBody');
  const title = $('#chunkQaDetailModalTitle');
  if (!modal || !overlay || !body) return;
  body.innerHTML = '';
  body.appendChild(buildQaDetailCard(item, { modal: true }));
  if (title) {
    title.textContent = String(item && item.question || '').trim() || 'QA 详情';
  }
  overlay.hidden = false;
  modal.hidden = false;
  overlay.classList.add('is-open');
  modal.classList.add('is-open');
  modal.setAttribute('aria-hidden', 'false');
  document.body.classList.add('drawer-open');
}

function closeQaDetailModal() {
  const modal = $('#chunkQaDetailModal');
  const overlay = $('#chunkQaDetailOverlay');
  if (!modal || !overlay || modal.hidden) return;
  overlay.classList.remove('is-open');
  modal.classList.remove('is-open');
  modal.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('drawer-open');
  window.setTimeout(() => {
    overlay.hidden = true;
    modal.hidden = true;
  }, 160);
}

function switchChunkDebugView(view) {
  chunkDebugState.activeView = view === 'qa' ? 'qa' : 'chunk';
  $$('[data-debug-view]').forEach((button) => {
    const active = String(button.dataset.debugView || '') === chunkDebugState.activeView;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  $$('[data-debug-view-panel]').forEach((panel) => {
    const active = String(panel.dataset.debugViewPanel || '') === chunkDebugState.activeView;
    panel.classList.toggle('is-active', active);
  });
}

function setChunkDebugStatus(text) {
  const el = $('#chunkDebugStatus');
  if (el) el.textContent = String(text || '');
}

function setChunkDebugSummary(text) {
  const el = $('#chunkDebugSummary');
  if (el) el.textContent = String(text || '');
}

function resetChunkDebugPanel(message = '请选择左侧 leaf chunk') {
  chunkDebugState.currentChunkId = '';
  chunkDebugState.payload = null;
  chunkDebugState.qaItems = [];
  chunkDebugState.selectedQaId = '';
  setChunkDebugSummary(message);
  setChunkDebugStatus('等待加载');
  const chunkView = $('#chunkDebugChunkView');
  const qaList = $('#chunkDebugQaList');
  const qaDetail = $('#chunkDebugQaDetail');
  if (chunkView) {
    chunkView.innerHTML = '';
    chunkView.appendChild(createDebugEmpty(message));
  }
  if (qaList) {
    qaList.innerHTML = '';
    qaList.appendChild(createDebugEmpty('暂无 QA'));
  }
  if (qaDetail) {
    qaDetail.innerHTML = '';
    qaDetail.appendChild(createDebugEmpty('请选择一条 QA'));
  }
}

function renderChunkDebugSummary(chunk, qaItems) {
  const parts = [];
  const chunkIndex = chunk && chunk.chunk_index != null ? `Chunk #${chunk.chunk_index}` : 'Chunk';
  parts.push(chunkIndex);
  if (chunk && chunk.index_path) parts.push(`路径 ${chunk.index_path}`);
  if (chunk && chunk.title_path) parts.push(String(chunk.title_path));
  parts.push(`QA ${qaItems.length} 条`);
  setChunkDebugSummary(parts.join(' ｜ '));
}

function renderChunkView(payload) {
  const container = $('#chunkDebugChunkView');
  if (!container) return;
  container.innerHTML = '';
  const chunk = (payload && payload.chunk) || {};
  const qaItems = safeArray(payload && payload.qa && payload.qa.items);
  if (!chunk || !Object.keys(chunk).length) {
    container.appendChild(createDebugEmpty('未找到 chunk 详情'));
    return;
  }

  const card = document.createElement('div');
  card.className = 'chunk-debug-card';

  const title = document.createElement('h3');
  title.className = 'chunk-debug-title';
  title.textContent = 'Chunk 信息';
  card.appendChild(title);

  const kpi = document.createElement('div');
  kpi.className = 'chunk-debug-kpi-grid';
  [
    ['Chunk', chunk.chunk_index != null ? `#${chunk.chunk_index}` : '—'],
    ['层级', chunk.level != null ? String(chunk.level) : '—'],
    ['QA 数', String(qaItems.length)],
    ['过滤后', String(qaItems.filter((it) => !!it.filtered).length)],
  ].forEach(([label, value]) => {
    const cell = document.createElement('div');
    cell.className = 'chunk-debug-kpi';
    const labelEl = document.createElement('div');
    labelEl.className = 'chunk-debug-kpi-label';
    labelEl.textContent = label;
    const valEl = document.createElement('div');
    valEl.className = 'chunk-debug-kpi-value';
    valEl.textContent = value;
    cell.appendChild(labelEl);
    cell.appendChild(valEl);
    kpi.appendChild(cell);
  });
  card.appendChild(kpi);

  card.appendChild(
    createKvGrid([
      ['chunk_id', chunk.chunk_id],
      ['doc_id', chunk.doc_id],
      ['task_id', chunk.task_id],
      ['文件', chunk.original_filename],
      ['index_path', chunk.index_path],
      ['title_path', chunk.title_path],
      ['parent_index_path', chunk.parent_index_path],
      ['root_index_path', chunk.root_index_path],
      ['创建时间', fmtDebugTs(chunk.created_at)],
    ]),
  );
  card.appendChild(createCodeSection('正文', chunk.text));

  const qaSection = document.createElement('section');
  qaSection.className = 'chunk-debug-section';
  const qaHeading = document.createElement('h4');
  qaHeading.className = 'chunk-debug-section-title';
  qaHeading.textContent = '该块关联 QA';
  qaSection.appendChild(qaHeading);
  if (!qaItems.length) {
    qaSection.appendChild(createDebugEmpty('该 chunk 暂无 QA'));
  } else {
    const list = document.createElement('div');
    list.className = 'chunk-debug-trace-list';
    qaItems.forEach((item, index) => {
      const block = document.createElement('div');
      block.className = 'chunk-debug-trace-item';
      const head = document.createElement('div');
      head.className = 'chunk-debug-trace-head';
      const headTitle = document.createElement('div');
      headTitle.className = 'chunk-debug-trace-title';
      headTitle.textContent = `${index + 1}. ${String(item.question || '').trim() || '未命名问题'}`;
      const score = document.createElement('div');
      score.className = 'chunk-debug-trace-score';
      score.textContent = `平均分 ${toScoreText(item.average_score)}`;
      head.appendChild(headTitle);
      head.appendChild(score);
      block.appendChild(head);
      const meta = document.createElement('div');
      meta.className = 'chunk-debug-trace-meta';
      meta.textContent = [
        item.question_type ? `题型 ${item.question_type}` : '',
        item.knowledge_category ? `分类 ${item.knowledge_category}` : '',
        item.filtered ? '已过滤' : '保留',
        getEvidenceTraces(item).length
          ? `证据块 ${getEvidenceTraces(item).length} 个`
          : '无补充证据',
      ]
        .filter(Boolean)
        .join(' ｜ ');
      block.appendChild(meta);
      const evidenceDisclosure = renderEvidenceDisclosure(item);
      if (evidenceDisclosure) block.appendChild(evidenceDisclosure);
      const actions = document.createElement('div');
      actions.className = 'chunk-debug-item-actions';
      actions.appendChild(createSmallActionButton('打开完整详情', () => openQaDetailModal(item)));
      block.appendChild(actions);
      list.appendChild(block);
    });
    qaSection.appendChild(list);
  }
  card.appendChild(qaSection);
  container.appendChild(card);
}

function renderQaList(qaItems) {
  const list = $('#chunkDebugQaList');
  if (!list) return;
  list.innerHTML = '';
  if (!qaItems.length) {
    list.appendChild(createDebugEmpty('该 chunk 暂无 QA'));
    return;
  }
  qaItems.forEach((item, index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chunk-debug-qa-item';
    if (String(item.id || '') === String(chunkDebugState.selectedQaId || '')) {
      btn.classList.add('is-active');
    }
    const title = document.createElement('div');
    title.className = 'chunk-debug-qa-item-title';
    title.textContent = `${index + 1}. ${String(item.question || '').trim() || '未命名问题'}`;
    const meta = document.createElement('div');
    meta.className = 'chunk-debug-qa-item-meta';
    meta.textContent = [
      item.question_type ? `题型 ${item.question_type}` : '',
      item.knowledge_category ? `分类 ${item.knowledge_category}` : '',
      `平均分 ${toScoreText(item.average_score)}`,
      item.filtered ? '已过滤' : '保留',
    ]
      .filter(Boolean)
      .join(' ｜ ');
    btn.appendChild(title);
    btn.appendChild(meta);
    btn.addEventListener('click', () => {
      chunkDebugState.selectedQaId = String(item.id || '');
      renderQaList(chunkDebugState.qaItems);
      renderQaDetail(item);
    });
    btn.addEventListener('dblclick', () => openQaDetailModal(item));
    list.appendChild(btn);
  });
}

function renderTraceSection(title, traces) {
  const section = document.createElement('section');
  section.className = 'chunk-debug-section';
  const heading = document.createElement('h4');
  heading.className = 'chunk-debug-section-title';
  heading.textContent = title;
  section.appendChild(heading);
  const items = safeArray(traces);
  if (!items.length) {
    section.appendChild(createDebugEmpty('无数据'));
    return section;
  }
  const list = document.createElement('div');
  list.className = 'chunk-debug-trace-list';
  items.forEach((trace) => {
    const block = document.createElement('div');
    block.className = 'chunk-debug-trace-item';
    const head = document.createElement('div');
    head.className = 'chunk-debug-trace-head';
    const left = document.createElement('div');
    left.className = 'chunk-debug-trace-title';
    left.textContent = [
      trace.role ? `[${trace.role}]` : '',
      trace.chunk_index != null ? `Chunk #${trace.chunk_index}` : '',
      trace.title_path || '',
    ]
      .filter(Boolean)
      .join(' ');
    const right = document.createElement('div');
    right.className = 'chunk-debug-trace-score';
    right.textContent =
      trace.score !== undefined && trace.score !== null ? `综合 ${Number(trace.score).toFixed(4)}` : '';
    head.appendChild(left);
    head.appendChild(right);
    block.appendChild(head);
    const meta = document.createElement('div');
    meta.className = 'chunk-debug-trace-meta';
    meta.textContent = [
      trace.chunk_id ? `id ${trace.chunk_id}` : '',
      trace.parent_index_path ? `父路径 ${trace.parent_index_path}` : '',
      trace.final_rank ? `最终排序 ${trace.final_rank}` : '',
      trace.dense_rank ? `向量排序 ${trace.dense_rank}` : '',
      trace.lexical_rank ? `词项排序 ${trace.lexical_rank}` : '',
      trace.dense_score !== undefined && trace.dense_score !== null ? `向量 ${Number(trace.dense_score).toFixed(4)}` : '',
      trace.lexical_score !== undefined && trace.lexical_score !== null ? `词项 ${Number(trace.lexical_score).toFixed(4)}` : '',
      trace.structure_score !== undefined && trace.structure_score !== null ? `结构 ${Number(trace.structure_score).toFixed(4)}` : '',
      trace.must_term_coverage !== undefined && trace.must_term_coverage !== null ? `关键词覆盖 ${Number(trace.must_term_coverage).toFixed(2)}` : '',
      trace.rerank_score !== undefined && trace.rerank_score !== null ? `rerank ${Number(trace.rerank_score).toFixed(4)}` : '',
      trace.score_gap_top1_top2 !== undefined && trace.score_gap_top1_top2 !== null ? `top1-top2 差 ${Number(trace.score_gap_top1_top2).toFixed(4)}` : '',
      trace.same_parent ? '同章节' : '',
      trace.adjacent ? '相邻块' : '',
      trace.is_source_chunk ? '主来源块' : '',
    ]
      .filter(Boolean)
      .join(' ｜ ');
    if (meta.textContent) block.appendChild(meta);
    list.appendChild(block);
  });
  section.appendChild(list);
  return section;
}

function getEvidenceTraces(item, options = {}) {
  const includeRaw = !!options.includeRaw;
  const rawEvidenceIds = safeArray(item.evidence_chunk_ids);
  const retrievalTrace = item.retrieval_trace && typeof item.retrieval_trace === 'object'
    ? item.retrieval_trace
    : {};
  const selectedEvidence = safeArray(retrievalTrace.selected_evidence);
  const rawSemanticHits = safeArray(retrievalTrace.raw_semantic_hits);
  const evidenceMap = new Map();

  const traces = includeRaw ? [...selectedEvidence, ...rawSemanticHits] : selectedEvidence;
  traces.forEach((trace) => {
    const chunkId = String(trace && trace.chunk_id || '').trim();
    if (!chunkId || evidenceMap.has(chunkId)) return;
    evidenceMap.set(chunkId, trace);
  });

  rawEvidenceIds.forEach((chunkId) => {
    const key = String(chunkId || '').trim();
    if (!key || evidenceMap.has(key)) return;
    evidenceMap.set(key, { chunk_id: key });
  });

  return [...evidenceMap.values()];
}

function createEvidenceCard(trace) {
  const itemEl = document.createElement('div');
  itemEl.className = 'chunk-debug-evidence-item';

  const head = document.createElement('div');
  head.className = 'chunk-debug-evidence-head';

  const title = document.createElement('div');
  title.className = 'chunk-debug-evidence-title';
  title.textContent = [
    trace.role ? `[${trace.role}]` : '',
    trace.chunk_index != null ? `Chunk #${trace.chunk_index}` : `Chunk ${trace.chunk_id || ''}`,
  ]
    .filter(Boolean)
    .join(' ');

  const score = document.createElement('div');
  score.className = 'chunk-debug-trace-score';
  score.textContent =
    trace.score !== undefined && trace.score !== null ? `综合 ${Number(trace.score).toFixed(4)}` : '';

  head.appendChild(title);
  head.appendChild(score);
  itemEl.appendChild(head);

  const meta = document.createElement('div');
  meta.className = 'chunk-debug-evidence-meta';
  meta.textContent = [
    trace.chunk_id ? `id ${trace.chunk_id}` : '',
    trace.title_path ? `标题 ${trace.title_path}` : '',
    trace.parent_index_path ? `父路径 ${trace.parent_index_path}` : '',
    trace.retrieval_rank ? `排序 ${trace.retrieval_rank}` : '',
    trace.dense_score !== undefined && trace.dense_score !== null ? `向量 ${Number(trace.dense_score).toFixed(4)}` : '',
    trace.lexical_score !== undefined && trace.lexical_score !== null ? `词项 ${Number(trace.lexical_score).toFixed(4)}` : '',
    trace.structure_score !== undefined && trace.structure_score !== null ? `结构 ${Number(trace.structure_score).toFixed(4)}` : '',
    trace.is_source_chunk ? '主来源块' : '',
  ]
    .filter(Boolean)
    .join(' ｜ ');
  if (meta.textContent) itemEl.appendChild(meta);

  return itemEl;
}

function renderEvidenceDisclosure(item) {
  const traces = getEvidenceTraces(item);
  if (!traces.length) return null;

  const details = document.createElement('details');
  details.className = 'chunk-debug-evidence-disclosure';
  const summary = document.createElement('summary');
  summary.textContent = `展开证据块详情（${traces.length}）`;
  details.appendChild(summary);

  const list = document.createElement('div');
  list.className = 'chunk-debug-evidence-list';
  traces.forEach((trace) => list.appendChild(createEvidenceCard(trace)));
  details.appendChild(list);
  return details;
}

function renderEvidenceSection(item) {
  const traces = getEvidenceTraces(item);
  if (!traces.length) return null;

  const section = document.createElement('section');
  section.className = 'chunk-debug-section';
  const heading = document.createElement('h4');
  heading.className = 'chunk-debug-section-title';
  heading.textContent = `最终使用证据块（${traces.length}）`;
  section.appendChild(heading);

  const list = document.createElement('div');
  list.className = 'chunk-debug-evidence-list';
  traces.forEach((trace) => list.appendChild(createEvidenceCard(trace)));
  section.appendChild(list);
  return section;
}

function translateRetrievalMode(mode) {
  const value = String(mode || '').trim().toLowerCase();
  if (value === 'semantic') return 'semantic（仅向量语义排序）';
  if (value === 'hybrid') return 'hybrid（向量 + 词项 + 结构加权）';
  return value || '';
}

function translateAnswerScope(scope) {
  const value = String(scope || '').trim().toLowerCase();
  if (value === 'source_primary') return 'source_primary（主来源块优先）';
  if (value === 'same_section') return 'same_section（允许同章节补证据）';
  if (value === 'cross_chunk') return 'cross_chunk（允许跨 chunk 证据）';
  return value || '';
}

function formatAnswerScopeDecision(decision) {
  if (!decision || typeof decision !== 'object') return '';
  const reason = String(decision.reason || '').trim();
  const reasonCode = String(decision.reason_code || '').trim();
  const parts = [];
  if (reason) parts.push(reason);
  if (reasonCode) parts.push(`规则：${reasonCode}`);
  return parts.join(' ｜ ');
}

function buildQaDetailCard(item, options = {}) {
  if (!item || !Object.keys(item).length) {
    return createDebugEmpty('请选择一条 QA');
  }

  const card = document.createElement('div');
  card.className = options.modal ? 'chunk-debug-card chunk-debug-card-modal' : 'chunk-debug-card';

  const title = document.createElement('h3');
  title.className = 'chunk-debug-title';
  title.textContent = 'QA 详情';
  card.appendChild(title);

  if (!options.modal) {
    const actions = document.createElement('div');
    actions.className = 'chunk-debug-detail-actions';
    actions.appendChild(createSmallActionButton('打开大窗查看', () => openQaDetailModal(item)));
    card.appendChild(actions);
  }

  card.appendChild(
    createPillRow([
      item.question_type ? `题型：${item.question_type}` : '',
      item.knowledge_category ? `分类：${item.knowledge_category}` : '',
      `平均分：${toScoreText(item.average_score)}`,
      item.filtered ? '状态：已过滤' : '状态：保留',
      item.is_augmented ? '增广项' : '主项',
    ]),
  );

  card.appendChild(
    createKvGrid([
      ['qa_id', item.id],
      ['task_id', item.task_id],
      ['文件', item.original_filename],
      ['source_chunk_id', item.source_chunk_id],
      ['source_chunk_index', item.source_chunk_index != null ? `#${item.source_chunk_index}` : ''],
      ['source_chunk_title_path', item.source_chunk_title_path],
      ['source', item.source],
      ['source_anchor_text', item.source_anchor_text],
      ['qa_generation_unit_id', item.qa_generation_unit_id],
      ['创建时间', fmtDebugTs(item.created_at)],
    ]),
  );

  card.appendChild(createCodeSection('问题', item.question));
  card.appendChild(createCodeSection('答案', item.answer));
  if (item.answer_explanation) {
    card.appendChild(createCodeSection('答案解释', item.answer_explanation));
  }
  if (item.source_fact_text) {
    card.appendChild(createCodeSection('来源事实片段', item.source_fact_text));
  }
  if (item.qa_generation_unit_text) {
    card.appendChild(createCodeSection('出题单元', item.qa_generation_unit_text));
  }
  const retrievalTrace = item.retrieval_trace && typeof item.retrieval_trace === 'object'
    ? item.retrieval_trace
    : {};
  const evidenceSection = renderEvidenceSection(item);
  if (evidenceSection) {
    card.appendChild(evidenceSection);
  }
  if (retrievalTrace.query || retrievalTrace.retrieval_query) {
    const weightSummary = [
      retrievalTrace.hybrid_weight_dense !== undefined && retrievalTrace.hybrid_weight_dense !== null ? `向量 ${retrievalTrace.hybrid_weight_dense}` : '',
      retrievalTrace.hybrid_weight_lexical !== undefined && retrievalTrace.hybrid_weight_lexical !== null ? `词项 ${retrievalTrace.hybrid_weight_lexical}` : '',
      retrievalTrace.structure_weight !== undefined && retrievalTrace.structure_weight !== null ? `结构 ${retrievalTrace.structure_weight}` : '',
    ].filter(Boolean).join(' ｜ ');
    card.appendChild(
      createKvGrid([
        ['原始问题', retrievalTrace.query],
        ['检索查询', retrievalTrace.retrieval_query],
        ['必含术语', Array.isArray(retrievalTrace.must_have_terms) ? retrievalTrace.must_have_terms.join('、') : retrievalTrace.must_have_terms],
        ['模型建议范围', translateAnswerScope(retrievalTrace.answer_scope_hint || item.answer_scope_hint)],
        ['系统最终范围', translateAnswerScope(retrievalTrace.effective_answer_scope || retrievalTrace.answer_scope || item.effective_answer_scope || item.answer_scope)],
        ['前端范围策略', translateAnswerScope(retrievalTrace.answer_scope_policy)],
        ['范围裁决原因', formatAnswerScopeDecision(retrievalTrace.answer_scope_decision || item.answer_scope_decision)],
        ['检索模式', translateRetrievalMode(retrievalTrace.retrieval_mode)],
        ['evidence 数量', retrievalTrace.semantic_top_k],
        ['轻量重排候选数', retrievalTrace.rerank_top_n],
        ['权重', weightSummary],
        ['最大上下文字符', retrievalTrace.max_unit_chars],
      ]),
    );
  }

  if (safeArray(retrievalTrace.selected_evidence).length) {
    card.appendChild(renderTraceSection('最终选中证据', retrievalTrace.selected_evidence));
  }
  if (safeArray(retrievalTrace.raw_semantic_hits).length) {
    card.appendChild(renderTraceSection('语义召回候选', retrievalTrace.raw_semantic_hits));
  }

  if (item.evaluation || item.unsupervised_evaluation) {
    const evaluationWrap = document.createElement('div');
    evaluationWrap.className = 'evaluation';
    const heading = document.createElement('h4');
    heading.textContent = '评分详情';
    evaluationWrap.appendChild(heading);
    if (item.evaluation && item.evaluation.llm && item.evaluation.llm.scores) {
      evaluationWrap.appendChild(
        renderScoreGroup('LLM', item.evaluation.llm.scores, item.evaluation.llm.reasons || {}),
      );
    }
    if (item.evaluation && item.evaluation.local && item.evaluation.local.scores) {
      evaluationWrap.appendChild(renderScoreGroup('Local', item.evaluation.local.scores));
    }
    if (item.unsupervised_evaluation && item.unsupervised_evaluation.scores) {
      evaluationWrap.appendChild(
        renderScoreGroup('Unsupervised', item.unsupervised_evaluation.scores),
      );
    }
    card.appendChild(evaluationWrap);
  }

  return card;
}

function renderQaDetail(item) {
  const detail = $('#chunkDebugQaDetail');
  if (!detail) return;
  detail.innerHTML = '';
  detail.appendChild(buildQaDetailCard(item));
}

function renderChunkDebugPanel(payload) {
  const chunk = (payload && payload.chunk) || {};
  const qaItems = safeArray(payload && payload.qa && payload.qa.items);
  chunkDebugState.payload = payload;
  chunkDebugState.qaItems = qaItems;
  if (!qaItems.some((item) => String(item.id || '') === String(chunkDebugState.selectedQaId || ''))) {
    const firstPrimary = qaItems.find((item) => item && item.is_primary) || qaItems[0] || null;
    chunkDebugState.selectedQaId = firstPrimary ? String(firstPrimary.id || '') : '';
  }
  renderChunkDebugSummary(chunk, qaItems);
  renderChunkView(payload);
  renderQaList(qaItems);
  const selected =
    qaItems.find((item) => String(item.id || '') === String(chunkDebugState.selectedQaId || '')) ||
    qaItems[0] ||
    null;
  renderQaDetail(selected);
}

function renderChunkTree(tree) {
  const container = $('#chunkTree');
  if (!container) return;
  container.innerHTML = '';
  if (!tree) {
    container.textContent = '暂无数据';
    return;
  }

  const buildNode = (node, depth = 0) => {
    const children = Array.isArray(node.children) ? node.children : [];
    const chunks = Array.isArray(node.chunks) ? node.chunks : [];
    const title = String(node.title || '').trim() || 'Untitled';
    const indexPath = String(node.index_path || '').trim();

    const wrapper = document.createElement('div');

    const label = indexPath ? `${title} (${indexPath})` : title;
    const details = document.createElement('details');
    if (depth <= 1) details.open = true;
    const summary = document.createElement('summary');
    summary.textContent = label;
    details.appendChild(summary);

    if (chunks.length) {
      const leafWrap = document.createElement('div');
      chunks.forEach((ch) => {
        const chunkId = ch.chunk_id;
        const chunkIndex = ch.chunk_index;
        if (!chunkId) return;
        const row = document.createElement('div');
        row.className = 'chunk-tree-leaf';
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = `Chunk #${chunkIndex}`;
        btn.addEventListener('click', () => loadChunkDetailAndQa(String(chunkId)));
        row.appendChild(btn);
        const meta = document.createElement('span');
        meta.className = 'muted';
        meta.textContent = String(ch.title_path || '');
        row.appendChild(meta);
        leafWrap.appendChild(row);
      });
      details.appendChild(leafWrap);
    }

    if (children.length) {
      children.forEach((child) => {
        details.appendChild(buildNode(child, depth + 1));
      });
    }

    wrapper.appendChild(details);
    return wrapper;
  };

  container.appendChild(buildNode(tree, 0));
}

async function handleLoadChunkDocs() {
  const base = getApiBaseUrl();
  const taskId = ($('#chunkTaskId')?.value || '').trim() || lastTaskId;
  if (!taskId) {
    notify('请先输入 task_id 或先执行一次流水线', 'warning');
    return;
  }
  const select = $('#chunkDocSelect');
  setChunkOutputs('加载中…', '');
  resetChunkDebugPanel('加载文档列表中…');
  try {
    const data = await fetchJson(`${base}/doc-chunks/by-task/${encodeURIComponent(taskId)}`);
    if (!select) return;
    const docs = Array.isArray(data.docs) ? data.docs : [];
    select.innerHTML = '';
    docs.forEach((d) => {
      const opt = document.createElement('option');
      opt.value = String(d.doc_id || '');
      const name = String(d.original_filename || d.doc_id || '');
      const count = d.chunk_count != null ? ` (${d.chunk_count} chunks)` : '';
      opt.textContent = name + count;
      select.appendChild(opt);
    });
    if (docs.length) {
      select.value = String(docs[0].doc_id || '');
      persistUiField(select);
      await handleLoadChunkTree();
      setChunkOutputs('', '');
    } else {
      setChunkOutputs('该 task_id 暂无入库的 chunks（请确认 enable_chunk_storage=True 且 Milvus 可用）', '');
      resetChunkDebugPanel('该任务暂无已入库 chunk');
    }
  } catch (err) {
    setChunkOutputs('加载文档列表失败：' + String(err), '');
    resetChunkDebugPanel('加载文档列表失败');
  }
}

async function handleLoadChunkTree() {
  const base = getApiBaseUrl();
  const docId = ($('#chunkDocSelect')?.value || '').trim();
  if (!docId) return;
  setChunkOutputs('加载树中…', '');
  resetChunkDebugPanel('加载树中…');
  try {
    const taskId = ($('#chunkTaskId')?.value || '').trim() || lastTaskId || '';
    const qs = new URLSearchParams();
    qs.set('doc_id', docId);
    if (taskId) qs.set('task_id', taskId);
    const data = await fetchJson(`${base}/doc-chunks/tree?${qs.toString()}`);
    renderChunkTree(data.tree);
    setChunkOutputs('', '');
    resetChunkDebugPanel('请选择左侧 leaf chunk');
  } catch (err) {
    renderChunkTree(null);
    setChunkOutputs('加载树失败：' + String(err), '');
    resetChunkDebugPanel('加载树失败');
  }
}

async function loadChunkDetailAndQa(chunkId) {
  const base = getApiBaseUrl();
  const onlyFiltered = ($('#chunkQaOnlyFiltered')?.value || 'false') === 'true';
  chunkDebugState.currentChunkId = String(chunkId || '');
  setChunkDebugStatus('加载 chunk 调试数据…');
  setChunkOutputs('加载 chunk 调试数据…', '');
  try {
    const qs = new URLSearchParams();
    qs.set('page', '1');
    qs.set('page_size', '100');
    qs.set('only_filtered', onlyFiltered ? 'true' : 'false');
    const payload = await fetchJson(
      `${base}/doc-chunks/${encodeURIComponent(chunkId)}/debug?${qs.toString()}`,
    );
    setChunkOutputs(
      JSON.stringify(payload.chunk || {}, null, 2),
      JSON.stringify(((payload.qa || {}).items || []), null, 2),
    );
    renderChunkDebugPanel(payload);
    setChunkDebugStatus(`已加载 ${safeArray(payload.qa && payload.qa.items).length} 条 QA`);
  } catch (err) {
    setChunkOutputs('加载 chunk 调试数据失败：' + String(err), '');
    resetChunkDebugPanel('加载 chunk 调试数据失败');
    setChunkDebugStatus('加载失败');
  }
}

// 下载当前任务的 CSV
const downloadCsvBtn = $('#downloadCsvBtn');
if (downloadCsvBtn) {
  downloadCsvBtn.addEventListener('click', () => {
    const base = getApiBaseUrl();
    const taskId = $('#taskIdInput')?.value.trim();
    const fileFilterEl = $('#taskQaFileFilter');
    const selectedFile =
      (fileFilterEl && fileFilterEl.value && fileFilterEl.value.trim()) || '';

    // 如果同时指定了任务 ID 和“只看某个文件”，优先：按任务 + 文件名下载对应 CSV
    if (taskId && selectedFile) {
      downloadCsvForTaskFile(base, taskId, selectedFile);
      return;
    }

    if (
      taskId &&
      Array.isArray(lastPipelineOutputs) &&
      lastPipelineOutputs.some(
        (item) =>
          String(item && item.milvus_task_id || '') === String(taskId) &&
          String(item && item.history_source || '') === 'milvus',
      ) &&
      !lastPipelineOutputs.some((item) => item && item.consolidated_csv)
    ) {
      notify('当前任务结果已入库，临时 CSV 已删除，请改用数据库历史视图查看。', 'warning');
      return;
    }

    // 其次：如果用户指定了任务 ID，则按 task_id 下载“该任务的最新 CSV”
    if (taskId) {
      const url = `${base}/task-csv/${encodeURIComponent(taskId)}`;
      window.open(url, '_blank');
      return;
    }

    // 否则退回到当前前端流水线任务的 CSV（lastCsvPath）
    if (!lastCsvPath) {
      notify('当前没有可用的任务 CSV，请先执行流水线或在上方输入任务 ID。', 'warning');
      return;
    }
    const parts = String(lastCsvPath).split('/');
    const fileName = parts[parts.length - 1];
    const dlPath = `outputs/${fileName}`;
    const url = `${base}/download/${dlPath}`;
    window.open(url, '_blank');
  });
}

// ---------- 数据库 / 本地查询 ----------

const taskQaForm = $('#taskQaForm');
if (taskQaForm) {
  taskQaForm.addEventListener('submit', handleTaskQaQuery);
}

const taskQaFileFilter = $('#taskQaFileFilter');
if (taskQaFileFilter) {
  taskQaFileFilter.addEventListener('change', () => {
    const fileName = taskQaFileFilter.value.trim();
    const filtered = filterItemsByFile(lastTaskQaItems || [], fileName);
    renderQaResults({ items: filtered }, true);
  });
}

async function handleTaskQaQuery(e) {
  e.preventDefault();
  const base = getApiBaseUrl();
  const taskId = $('#taskIdInput')?.value.trim();
  if (!taskId) {
    notify('请输入任务ID', 'warning');
    return;
  }
  const onlyFiltered = $('#taskQaOnlyFiltered')?.checked;
  const minScore = $('#taskQaMinScore')?.value || '0';
  const page = $('#taskQaPage')?.value || '1';
  const pageSize = $('#taskQaPageSize')?.value || '20';
  const includeRaw = $('#taskQaIncludeRaw')?.checked;

  const params = new URLSearchParams();
  params.set('only_filtered', onlyFiltered ? 'true' : 'false');
  params.set('min_avg_score', minScore);
  params.set('page', page);
  params.set('page_size', pageSize);
  params.set('include_raw_responses', includeRaw ? 'true' : 'false');

  try {
    const resp = await fetch(
      `${base}/task-qa/${encodeURIComponent(taskId)}?${params.toString()}`,
    );
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      renderMeta({ error: detail || resp.statusText });
      $('#qaResults').textContent = '查询失败';
      return;
    }
    const items = normalizeItems(data.items || []);
    lastTaskQaItems = items;
    // 根据返回结果里的 original_filename 去重生成文件列表
    const fileSet = new Set();
    items.forEach((it) => {
      if (it.original_filename) {
        fileSet.add(it.original_filename);
      }
    });
    updateTaskQaFileFilter(Array.from(fileSet));
    const fileFilterEl = $('#taskQaFileFilter');
    const currentFile =
      (fileFilterEl && fileFilterEl.value && fileFilterEl.value.trim()) || '';
    const filteredItems = filterItemsByFile(items, currentFile);
    const meta = {
      task_info: data.task_info || {},
      model_info: data.model_info || {},
      counts: data.counts || {},
      category_distribution: data.category_distribution || {},
      filter_info: data.filter_info || {},
      pagination: data.pagination || {},
    };
    renderMeta(meta);
    renderQaResults({ items: filteredItems }, true);
  } catch (err) {
    renderMeta({ error: String(err) });
    $('#qaResults').textContent = '查询异常';
  }
}

const fileQaForm = $('#fileQaForm');
if (fileQaForm) {
  fileQaForm.addEventListener('submit', handleFileQaQuery);
}

async function handleFileQaQuery(e) {
  e.preventDefault();
  const base = getApiBaseUrl();
  const name = $('#fileQaName')?.value.trim();
  if (!name) {
    notify('请输入原始文件名', 'warning');
    return;
  }
  const page = $('#fileQaPage')?.value || '1';
  const pageSize = $('#fileQaPageSize')?.value || '20';
  const includeDetails = $('#fileQaIncludeDetails')?.checked;
  const taskId = $('#fileQaTaskId')?.value.trim();

  const params = new URLSearchParams();
  params.set('original_filename', name);
  params.set('page', page);
  params.set('page_size', pageSize);
  params.set('include_details', includeDetails ? 'true' : 'false');
  if (taskId) {
    params.set('task_id', taskId);
  }

  try {
    const resp = await fetch(`${base}/file-qa?${params.toString()}`);
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      renderMeta({ error: detail || resp.statusText });
      $('#qaResults').textContent = '查询失败';
      return;
    }
    const meta = {
      files: data.files || [],
      counts: data.counts || {},
      category_distribution: data.category_distribution || {},
      pagination: data.pagination || {},
      filters: data.filters || {},
    };
    renderMeta(meta);
    renderQaResults({ items: normalizeItems(data.items || []) }, true);
  } catch (err) {
    renderMeta({ error: String(err) });
    $('#qaResults').textContent = '查询异常';
  }
}

const localQueryForm = $('#localQueryForm');
if (localQueryForm) {
  localQueryForm.addEventListener('submit', handleLocalQuery);
}

async function handleLocalQuery(e) {
  e.preventDefault();
  const base = getApiBaseUrl();
  const text = $('#localQueryText')?.value.trim();
  if (!text) {
    notify('请输入查询文本', 'warning');
    return;
  }
  const taskId = $('#localQueryTaskId')?.value.trim();
  const topK = $('#localQueryTopK')?.value || '10';
  const onlyFiltered = $('#localQueryOnlyFiltered')?.checked;
  const minScore = $('#localQueryMinScore')?.value || '0';

  const formData = new FormData();
  formData.append('query_text', text);
  if (taskId) formData.append('task_id', taskId);
  formData.append('top_k', topK);
  formData.append('only_filtered', onlyFiltered ? 'true' : 'false');
  formData.append('min_avg_score', minScore);
  formData.append('include_raw_responses', 'false');

  try {
    const resp = await fetch(base + '/query-qa', {
      method: 'POST',
      body: formData,
    });
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      renderMeta({ error: detail || resp.statusText });
      $('#qaResults').textContent = '查询失败';
      return;
    }
    const meta = {
      query: data.query,
      filters: data.filters || {},
      counts: { total: data.total, returned: data.returned },
      message: data.message,
    };
    renderMeta(meta);
    renderQaResults({ items: normalizeItems(data.results || []) }, true);
  } catch (err) {
    renderMeta({ error: String(err) });
    $('#qaResults').textContent = '查询异常';
  }
}

function updateTaskQaFileFilter(files) {
  const select = $('#taskQaFileFilter');
  if (!select) return;
  lastTaskQaFiles = Array.isArray(files) ? files : [];
  const prev = select.value;
  select.innerHTML = '';
  const optAll = document.createElement('option');
  optAll.value = '';
  optAll.textContent = '全部文件';
  select.appendChild(optAll);
  lastTaskQaFiles.forEach((name) => {
    const opt = document.createElement('option');
    opt.value = String(name);
    opt.textContent = String(name);
    select.appendChild(opt);
  });
  if (prev && lastTaskQaFiles.includes(prev)) {
    select.value = prev;
  }
}

function filterItemsByFile(items, filename) {
  if (!filename) return items;
  return items.filter(
    (it) => (it.original_filename || '') === filename,
  );
}

// 覆盖按任务 + 文件名下载 CSV 的实现，改为调用后端专门端点
async function downloadCsvForTaskFile(base, taskId, originalFilename) {
  const url = `${base}/task-file-csv/${encodeURIComponent(
    taskId,
  )}?original_filename=${encodeURIComponent(originalFilename)}`;
  const match = Array.isArray(lastPipelineOutputs)
    ? lastPipelineOutputs.find((item) => String(item && item.source_file || '') === String(originalFilename || '').trim())
    : null;
  if (match && String(match.history_source || '') === 'milvus' && !match.consolidated_csv) {
    notify('该文件结果已入库，临时 CSV 已删除，请改用“查看入库记录”。', 'warning');
    return;
  }
  window.open(url, '_blank');
}
function setupFewShotUI() {
  const list = $('#fewShotList');
  const addBtn = $('#addFewShotBtn');
  if (!list || !addBtn) return;

  const addRow = (q = '', a = '') => {
    const row = document.createElement('div');
    row.className = 'fewshot-row';
    const qInput = document.createElement('input');
    qInput.type = 'text';
    qInput.placeholder = '示例问题';
    qInput.value = q;
    qInput.className = 'fewshot-q';
    const aInput = document.createElement('input');
    aInput.type = 'text';
    aInput.placeholder = '示例答案';
    aInput.value = a;
    aInput.className = 'fewshot-a';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = '删除';
    btn.addEventListener('click', () => {
      row.remove();
    });
    row.appendChild(qInput);
    row.appendChild(aInput);
    row.appendChild(btn);
    list.appendChild(row);
  };

  addBtn.addEventListener('click', () => addRow());

  // 默认放一行空示例，便于直接填写
  if (!list.children.length) {
    addRow();
  }
}

function collectFewShotExamples() {
  const rows = $$('.fewshot-row');
  const samples = [];
  rows.forEach((row) => {
    const q = row.querySelector('.fewshot-q')?.value.trim() || '';
    const a = row.querySelector('.fewshot-a')?.value.trim() || '';
    if (q && a) {
      samples.push({ question: q, answer: a });
    }
  });
  return samples;
}
