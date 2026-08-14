function normalizeItems(items) {
  if (!Array.isArray(items)) return [];
  return items.map((it) => {
    const theme = it.theme || it.knowledge_category || '';
    const themeReason = it.theme_reason || it.knowledge_category_reason || '';
    return { ...it, theme, theme_reason: themeReason };
  });
}

function displayedUnsupervisedScores(item) {
  const raw = item?.unsupervised_evaluation?.scores || item?.unsupervised_scores || {};
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.filterUnsupervisedScores === 'function') {
    return apiuseUi.filterUnsupervisedScores(raw);
  }
  const source = raw && typeof raw === 'object' ? raw : {};
  const filtered = {};
  const validUnitScore = (value) => {
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 && number <= 1 ? number : 0;
  };
  ['faithfulness', 'answerability', 'coverage_score'].forEach((key) => {
    filtered[key] = validUnitScore(source[key]);
  });
  if (source.answerability === undefined || source.answerability === null || source.answerability === '') {
    filtered.answerability = validUnitScore(source.p);
  }
  filtered.average_score =
    (filtered.faithfulness + filtered.answerability + filtered.coverage_score) / 3;
  return filtered;
}

// 本地 JSON 选择并解析
const localJsonInput = $('#localJsonInput');
if (localJsonInput) {
  localJsonInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      try {
        if (typeof currentConsolidatedJsonPath !== 'undefined') currentConsolidatedJsonPath = '';
      } catch (ignore) {
        // ignore
      }
      renderMetaFromConsolidated(data);
      renderFromConsolidated(data);
    } catch (err) {
      $('#qaResults').textContent = '解析失败：' + String(err);
    }
  });
}

function renderQaResults(data, includeDetails) {
  const container = document.createElement('div');
  const items = data.items || [];
  if (!items.length) {
    container.textContent = '没有结果';
    $('#qaResults').innerHTML = '';
    $('#qaResults').appendChild(container);
    if (window.qaFlowReview && typeof window.qaFlowReview.afterRender === 'function') {
      window.qaFlowReview.afterRender([]);
    }
    return;
  }
  const reviewHook = window.qaFlowReview && typeof window.qaFlowReview === 'object'
    ? window.qaFlowReview
    : null;
  const canReview = Boolean(reviewHook && typeof reviewHook.isEnabled === 'function' && reviewHook.isEnabled());
  items.forEach((it) => {
    const card = document.createElement('div');
    card.className = 'qa-card';

    const qaId = String(it.id || '').trim();
    if (canReview && qaId) {
      const reviewRow = document.createElement('label');
      reviewRow.className = 'qa-review-select';
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.setAttribute('data-qa-review-checkbox', 'true');
      checkbox.setAttribute('data-qa-id', qaId);
      checkbox.checked = typeof reviewHook.isSelected === 'function' && reviewHook.isSelected(qaId);
      checkbox.addEventListener('change', () => {
        if (typeof reviewHook.setSelected === 'function') {
          reviewHook.setSelected(qaId, checkbox.checked);
        }
      });
      const labelText = document.createElement('span');
      labelText.textContent = '选择入库';
      reviewRow.append(checkbox, labelText);
      card.appendChild(reviewRow);
    }

    const header = document.createElement('div');
    header.className = 'qa-header';
    header.innerHTML = `<strong>Q:</strong> ${escapeHtml(it.question || '')}`;

    const hasMcqOptions =
      (it.question_type === '单选题' || it.question_type === '选择题') &&
      Array.isArray(it.options) &&
      it.options.length === 4;

    let optionTextBlock = null;
    if (hasMcqOptions) {
      const optLetters = ['A', 'B', 'C', 'D'];
      const options = document.createElement('div');
      options.className = 'qa-options';
      options.innerHTML = it.options
        .map(
          (opt, idx) =>
            `<div class="qa-option-row"><strong>${optLetters[idx]}.</strong> ${escapeHtml(
              opt || '',
            )}</div>`,
        )
        .join('');
      optionTextBlock = options;
    }

    const answer = document.createElement('div');
    answer.className = 'qa-answer';
    const answerText =
      hasMcqOptions && it.correct_option
        ? `${it.correct_option}. ${it.answer || ''}`
        : it.answer || '';
    answer.innerHTML = `<strong>A:</strong> ${escapeHtml(answerText)}`;

    const meta = document.createElement('div');
    meta.className = 'qa-meta';
    const simCount = Array.isArray(it.similar_questions) ? it.similar_questions.length : 0;
    const scores = displayedUnsupervisedScores(it);
    const faithfulness =
      scores && typeof scores.faithfulness === 'number' ? scores.faithfulness : null;
    const answerability =
      scores && typeof scores.answerability === 'number' ? scores.answerability : null;
    const coverageScore =
      scores && typeof scores.coverage_score === 'number' ? scores.coverage_score : null;
    const averageScore =
      scores && typeof scores.average_score === 'number' ? scores.average_score : null;
    const metricParts = [];
    if (typeof faithfulness === 'number') metricParts.push(`faithfulness: ${faithfulness.toFixed(2)}`);
    if (typeof answerability === 'number') metricParts.push(`answerability: ${answerability.toFixed(2)}`);
    if (typeof coverageScore === 'number') metricParts.push(`coverage_score: ${coverageScore.toFixed(2)}`);
    if (typeof averageScore === 'number') metricParts.push(`平均分: ${averageScore.toFixed(2)}`);
    const metricPart = metricParts.length ? ` | ${metricParts.join(' | ')}` : '';
    meta.textContent = `主题: ${it.theme || '未分类'}${metricPart ? ` | ${metricPart}` : ''} | 文件: ${it.original_filename || ''}${simCount ? ` | 相似问: ${simCount}` : ''}`;

    const more = document.createElement('details');
    more.className = 'more';
    const sum = document.createElement('summary');
    sum.textContent = '更多';
    more.appendChild(sum);

    const detail = renderDetailPanel(it, includeDetails);
    more.appendChild(detail);

    card.appendChild(header);
    if (optionTextBlock) card.appendChild(optionTextBlock);
    card.appendChild(answer);
    card.appendChild(meta);
    card.appendChild(more);
    container.appendChild(card);
  });
  $('#qaResults').innerHTML = '';
  $('#qaResults').appendChild(container);
  if (reviewHook && typeof reviewHook.afterRender === 'function') {
    reviewHook.afterRender(items);
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderMeta(obj) {
  $('#metaInfo').textContent = JSON.stringify(obj || {}, null, 2);
}

function renderMetaFromConsolidated(json) {
  const unsupervisedScores = displayedUnsupervisedScores({
    unsupervised_scores: json.unsupervised_scores || {},
  });
  const meta = {
    task: json.task || {},
    model: json.model || {},
    counts: json.counts || {},
    timing: json.timing || {},
    unsupervised_scores: unsupervisedScores,
    theme_distribution: json.theme_distribution || {},
  };
  renderMeta(meta);
}

function renderFromConsolidated(json) {
  const includeDetails = true;
  const items = normalizeItems(Array.isArray(json.items) ? json.items : []);
  if (window.qaFlowReview && typeof window.qaFlowReview.setContext === 'function') {
    window.qaFlowReview.setContext(json, items);
  }
  if (json.task && json.task.task_id) {
    lastTaskId = json.task.task_id;
    const taskInput = $('#taskIdInput');
    if (taskInput && !taskInput.value) {
      taskInput.value = lastTaskId;
      persistUiField(taskInput);
    }
  }
  if (json.task && json.task.original_filename) {
    const fileInput = $('#fileQaName');
    if (fileInput && !fileInput.value) {
      fileInput.value = json.task.original_filename;
      persistUiField(fileInput);
    }
  }
  renderQaResults({ items }, includeDetails);
}

function renderDetailPanel(it, includeDetails) {
  const panel = document.createElement('div');
  panel.className = 'detail-panel';

  const unsupScores = displayedUnsupervisedScores(it);
  const pSingle =
    unsupScores && typeof unsupScores.answerability === 'number' ? unsupScores.answerability : null;
  const averageScore =
    unsupScores && typeof unsupScores.average_score === 'number' ? unsupScores.average_score : null;

  const badges = document.createElement('div');
  badges.className = 'badges';
  badges.innerHTML = `
    <span class="badge theme">${escapeHtml(it.theme || '未分类')}</span>
    ${
      typeof unsupScores.faithfulness === 'number'
        ? `<span class="badge faith">faithfulness ${unsupScores.faithfulness.toFixed(
            2,
          )}</span>`
        : ''
    }
    ${
      typeof pSingle === 'number'
        ? `<span class="badge faith">answerability ${pSingle.toFixed(2)}</span>`
        : ''
    }
    ${
      typeof unsupScores.coverage_score === 'number'
        ? `<span class="badge score">coverage_score ${unsupScores.coverage_score.toFixed(
            2,
          )}</span>`
        : ''
    }
    ${
      typeof averageScore === 'number'
        ? `<span class="badge score">平均分 ${averageScore.toFixed(2)}</span>`
        : ''
    }
    ${
      it.original_filename
        ? `<span class="badge file">${escapeHtml(it.original_filename)}</span>`
        : ''
    }
  `;
  panel.appendChild(badges);

  const kv = document.createElement('div');
  kv.className = 'kv';
  addKv(kv, 'ID', it.id);
  addKv(kv, '任务ID', it.task_id);
  addKv(kv, '创建时间', fmtTs(it.created_at));
  if (it.theme_reason) addKv(kv, '主题理由', it.theme_reason);
  if (it.source) addKv(kv, '来源', it.source);
  if (it.chunk_index !== undefined && it.chunk_index !== null) {
    addKv(kv, 'Chunk', it.chunk_index);
  }
  if (it.source_fact_text) addKv(kv, '来源片段', it.source_fact_text);
  if (it.question_type) addKv(kv, '题型', it.question_type);
  if (Array.isArray(it.options) && it.options.length) {
    const optLetters = ['A', 'B', 'C', 'D'];
    addKv(
      kv,
      '选项',
      it.options
        .map((opt, idx) => `${optLetters[idx] || ''}. ${opt || ''}`.trim())
        .join('\n'),
    );
  }
  if (it.correct_option) addKv(kv, '正确选项', it.correct_option);
  if (it.difficulty_level) addKv(kv, '难度', it.difficulty_level);
  if (it.answer_explanation) addKv(kv, '答案解析', it.answer_explanation);
  if (it.evaluation_method) addKv(kv, '评估方式', it.evaluation_method);
  if (typeof unsupScores.faithfulness === 'number') {
    addKv(kv, 'faithfulness', unsupScores.faithfulness.toFixed(4));
  }
  if (typeof unsupScores.answerability === 'number') {
    addKv(kv, 'answerability', unsupScores.answerability.toFixed(4));
  }
  if (typeof unsupScores.coverage_score === 'number') {
    addKv(kv, 'coverage_score', unsupScores.coverage_score.toFixed(4));
  }
  if (typeof averageScore === 'number') {
    addKv(kv, '平均分', averageScore.toFixed(4));
  }
  if (Array.isArray(it.similar_questions) && it.similar_questions.length) {
    const lines = it.similar_questions
      .map((sq, idx) => {
        if (!sq) return '';
        if (typeof sq === 'string') return `${idx + 1}. ${sq}`;
        return `${idx + 1}. ${(sq.question || '').toString()}`.trim();
      })
      .filter((x) => x && x.trim());
    if (lines.length) {
      addKv(kv, `增广问句（${lines.length}）`, lines.join('\n'));
    }
  }
  panel.appendChild(kv);

  if (includeDetails && (it.evaluation || it.unsupervised_evaluation)) {
    const ev = document.createElement('div');
    ev.className = 'evaluation';
    const title = document.createElement('h4');
    title.textContent = '评分详情';
    ev.appendChild(title);

    if (it.evaluation) {
      if (it.evaluation.llm && it.evaluation.llm.scores) {
        ev.appendChild(
          renderScoreGroup(
            'LLM',
            it.evaluation.llm.scores,
            it.evaluation.llm.reasons || {},
          ),
        );
      }
      if (it.evaluation.local && it.evaluation.local.scores) {
        ev.appendChild(renderScoreGroup('Local', it.evaluation.local.scores));
      }
    }

    if (it.unsupervised_evaluation && it.unsupervised_evaluation.scores) {
      const s = it.unsupervised_evaluation.scores || {};
      const apiuseUi = ui();
      const filtered = apiuseUi && typeof apiuseUi.filterUnsupervisedScores === 'function'
        ? apiuseUi.filterUnsupervisedScores(s)
        : {
            faithfulness: s.faithfulness,
            answerability: s.answerability,
            coverage_score: s.coverage_score,
          };
      ev.appendChild(renderScoreGroup('Unsupervised', filtered));
    }
    panel.appendChild(ev);
  }

  return panel;
}

function addKv(container, label, value) {
  if (value === undefined || value === null || value === '') return;
  const row = document.createElement('div');
  row.className = 'kv-row';
  const k = document.createElement('div');
  k.className = 'kv-k';
  k.textContent = String(label);
  const v = document.createElement('div');
  v.className = 'kv-v';
  v.textContent = typeof value === 'string' ? value : String(value);
  row.appendChild(k);
  row.appendChild(v);
  container.appendChild(row);
}

function renderScoreGroup(label, scoresObj, reasonsObj = {}) {
  const apiuseUi = ui();
  if (apiuseUi && typeof apiuseUi.createScoreGroup === 'function') {
    const group = apiuseUi.createScoreGroup(label, scoresObj, reasonsObj, {
      digits: 2,
      reasonPlacement: 'row',
      barValue: (metric, score) => (String(metric || '').endsWith('_100') ? score / 100 : score),
    });
    if (group) return group;
  }

  const wrap = document.createElement('div');
  wrap.className = 'score-group';
  return wrap;
}

function fmtTs(ts) {
  if (!ts) return '';
  try {
    const d = new Date(Number(ts) * (String(ts).length > 11 ? 1 : 1000));
    return isNaN(d.getTime()) ? '' : d.toLocaleString();
  } catch {
    return '';
  }
}
