// ---------- 三级知识分类（本地模型） ----------

const knowledgeTagForm = $('#knowledgeTagForm');
if (knowledgeTagForm) knowledgeTagForm.addEventListener('submit', handleKnowledgeTagPredict);

async function handleKnowledgeTagPredict(ev) {
  ev.preventDefault();
  const base = getApiBaseUrl();
  const text = $('#knowledgeTagText')?.value || '';
  const filename = $('#knowledgeTagFilename')?.value || '';
  const classifierMode = $('#knowledgeTagClassifier')?.value || 'doc_level3_rule';
  const output = $('#knowledgeTagOutput');
  if (output) output.textContent = '预测中…';
  try {
    const resp = await fetch(`${base}/knowledge-tagging/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, filename, classifier_mode: classifierMode }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      if (output) output.textContent = '预测失败：' + (detail || resp.statusText);
      return;
    }
    if (output) {
      output.textContent = JSON.stringify(data, null, 2);
    }
  } catch (err) {
    if (output) output.textContent = '预测异常：' + String(err);
  }
}

// ---------- LLM 配置管理 ----------

const cfgSelect = $('#cfgSelect');
const btnCfgSave = $('#btnCfgSave');
const btnCfgActivate = $('#btnCfgActivate');
const btnCfgDelete = $('#btnCfgDelete');
const btnCfgRefresh = $('#btnCfgRefresh');
const btnLlmDebug = $('#btnLlmDebug');

if (btnCfgSave) btnCfgSave.addEventListener('click', handleCfgSave);
if (btnCfgActivate) btnCfgActivate.addEventListener('click', handleCfgActivate);
if (btnCfgDelete) btnCfgDelete.addEventListener('click', handleCfgDelete);
if (btnCfgRefresh) btnCfgRefresh.addEventListener('click', () => refreshConfigStore());
if (cfgSelect) cfgSelect.addEventListener('change', fillConfigFormFromSelect);
if (btnLlmDebug) btnLlmDebug.addEventListener('click', handleLlmDebug);

refreshConfigStore().catch(() => {});

function setCfgStatus(text, isError = false) {
  const el = $('#cfgStatus');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = isError ? '#b91c1c' : '';
}

async function refreshConfigStore() {
  const base = getApiBaseUrl();
  setCfgStatus('加载中…', false);
  try {
    const data = await fetchJson(`${base}/llm-configs`);
    llmConfigStore = data || { active: '', profiles: {} };
    renderConfigStore();
    setCfgStatus('', false);
  } catch (err) {
    console.warn('加载 LLM 配置失败', err);
    setCfgStatus(`加载失败（${base}）：${String(err)}`, true);
  }
}

function renderConfigStore() {
  const select = $('#cfgSelect');
  const activeLabel = $('#cfgActive');
  if (select) {
    select.innerHTML = '';
    const profiles = llmConfigStore.profiles || {};
    Object.keys(profiles).forEach((name) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (llmConfigStore.active === name) opt.selected = true;
      select.appendChild(opt);
    });
  }
  if (activeLabel) {
    activeLabel.textContent = llmConfigStore.active
      ? `当前激活: ${llmConfigStore.active}`
      : '未激活配置';
  }
}

function fillConfigFormFromSelect() {
  const name = cfgSelect?.value;
  if (!name) return;
  const profile = (llmConfigStore.profiles || {})[name];
  if (!profile) return;
  $('#cfgName').value = profile.name || name;
  $('#cfgKey').value = profile.api_key || '';
  $('#cfgBaseUrl').value = profile.base_url || '';
  $('#cfgModel').value = profile.model || '';
  if ($('#cfgApiType')) $('#cfgApiType').value = profile.api_type || 'openai';
  if ($('#cfgModelVersion')) $('#cfgModelVersion').value = profile.model_version || '';
}

async function handleCfgSave() {
  const name = $('#cfgName')?.value.trim();
  const apiKey = $('#cfgKey')?.value.trim();
  const baseUrl = $('#cfgBaseUrl')?.value.trim();
  const model = $('#cfgModel')?.value.trim();
  const apiType = $('#cfgApiType')?.value.trim() || 'openai';
  const modelVersion = $('#cfgModelVersion')?.value.trim() || '';
  if (!name || !apiKey || !baseUrl || !model) {
    notify('请填写名称、API Key、Base URL、模型', 'warning');
    return;
  }
  setBtnLoading(btnCfgSave, true);
  try {
    const base = getApiBaseUrl();
    const data = await fetchJson(`${base}/llm-configs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        api_key: apiKey,
        base_url: baseUrl,
        model,
        api_type: apiType,
        model_version: modelVersion,
      }),
    });
    llmConfigStore = data;
    renderConfigStore();
    setCfgStatus('保存成功', false);
    notify('保存成功', 'success');
  } catch (err) {
    setCfgStatus(`保存失败：${String(err)}`, true);
    notify('保存失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnCfgSave, false);
  }
}

async function handleCfgActivate() {
  const name = $('#cfgName')?.value.trim() || cfgSelect?.value;
  if (!name) {
    notify('请先选择或输入要激活的配置名称', 'warning');
    return;
  }
  setBtnLoading(btnCfgActivate, true);
  try {
    const base = getApiBaseUrl();
    const data = await fetchJson(`${base}/llm-configs/${encodeURIComponent(name)}/activate`, {
      method: 'POST',
    });
    llmConfigStore = data;
    renderConfigStore();
    setCfgStatus(`已激活配置：${name}`, false);
    notify('已激活配置：' + name, 'success');
  } catch (err) {
    setCfgStatus(`激活失败：${String(err)}`, true);
    notify('激活失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnCfgActivate, false);
  }
}

async function handleCfgDelete() {
  const name = $('#cfgName')?.value.trim() || cfgSelect?.value;
  if (!name) {
    notify('请先选择或输入要删除的配置名称', 'warning');
    return;
  }
  if (!confirm(`确定删除配置 ${name} 吗？`)) return;
  setBtnLoading(btnCfgDelete, true);
  try {
    const base = getApiBaseUrl();
    const data = await fetchJson(`${base}/llm-configs/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    llmConfigStore = data;
    renderConfigStore();
    setCfgStatus(`已删除配置：${name}`, false);
    notify('已删除配置：' + name, 'success');
  } catch (err) {
    setCfgStatus(`删除失败：${String(err)}`, true);
    notify('删除失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnCfgDelete, false);
  }
}

async function handleLlmDebug() {
  const base = getApiBaseUrl();
  const output = $('#llmDebugOutput');
  const prompt = $('#llmDebugPrompt')?.value || '';
  const systemPrompt = $('#llmDebugSystemPrompt')?.value || 'You are a helpful assistant.';
  const responseFormat = $('#llmDebugResponseFormat')?.value || 'json_object';
  const timeoutSeconds = parseInt($('#llmDebugTimeoutSeconds')?.value || '30', 10) || 30;
  const maxTokens = parseInt($('#llmDebugMaxTokens')?.value || '256', 10) || 256;

  if (output) output.textContent = '测试中…';
  try {
    const data = await fetchJson(`${base}/llm-debug/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        system_prompt: systemPrompt,
        response_format: responseFormat,
        timeout_seconds: timeoutSeconds,
        max_tokens: maxTokens,
      }),
    });
    if (output) output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    if (output) output.textContent = '测试失败：' + String(err);
  }
}

// ---------- OCR 配置管理 ----------

const ocrCfgSelect = $('#ocrCfgSelect');
const btnOcrCfgSave = $('#btnOcrCfgSave');
const btnOcrCfgActivate = $('#btnOcrCfgActivate');
const btnOcrCfgDelete = $('#btnOcrCfgDelete');
const btnOcrCfgRefresh = $('#btnOcrCfgRefresh');
const btnOcrCfgTest = $('#btnOcrCfgTest');
const ocrCfgProvider = $('#ocrCfgProvider');

if (btnOcrCfgSave) btnOcrCfgSave.addEventListener('click', handleOcrCfgSave);
if (btnOcrCfgActivate) btnOcrCfgActivate.addEventListener('click', handleOcrCfgActivate);
if (btnOcrCfgDelete) btnOcrCfgDelete.addEventListener('click', handleOcrCfgDelete);
if (btnOcrCfgRefresh) btnOcrCfgRefresh.addEventListener('click', () => refreshOcrConfigStore());
if (btnOcrCfgTest) btnOcrCfgTest.addEventListener('click', handleOcrCfgTest);
if (ocrCfgSelect) ocrCfgSelect.addEventListener('change', fillOcrConfigFormFromSelect);
if (ocrCfgProvider) ocrCfgProvider.addEventListener('change', syncOcrProviderUi);

refreshOcrConfigStore().catch(() => {});
syncOcrProviderUi();

function setOcrCfgStatus(text, isError = false) {
  const el = $('#ocrCfgStatus');
  if (!el) return;
  el.textContent = text || '';
  el.style.color = isError ? '#b91c1c' : '';
}

async function refreshOcrConfigStore() {
  const base = getApiBaseUrl();
  setOcrCfgStatus('加载中…', false);
  try {
    const data = await fetchJson(`${base}/ocr-configs`);
    ocrConfigStore = data || { active: '', profiles: {} };
    renderOcrConfigStore();
    setOcrCfgStatus('', false);
  } catch (err) {
    console.warn('加载 OCR 配置失败', err);
    setOcrCfgStatus(`加载失败（${base}）：${String(err)}`, true);
  }
}

function renderOcrConfigStore() {
  const select = $('#ocrCfgSelect');
  const activeLabel = $('#ocrCfgActive');
  if (select) {
    select.innerHTML = '';
    const profiles = ocrConfigStore.profiles || {};
    Object.keys(profiles).forEach((name) => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (ocrConfigStore.active === name) opt.selected = true;
      select.appendChild(opt);
    });
  }
  if (activeLabel) {
    activeLabel.textContent = ocrConfigStore.active
      ? `当前激活: ${ocrConfigStore.active}`
      : '未激活配置';
  }
}

function parseJsonObject(text) {
  const raw = String(text || '').trim();
  if (!raw) return {};
  const obj = JSON.parse(raw);
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new Error('必须是 JSON 对象，例如 {"enable_image_analysis":"true"}');
  }
  return obj;
}

function syncOcrProviderUi() {
  const provider = $('#ocrCfgProvider')?.value || 'batch_ocr';
  const batchRow = $('#ocrCfgBatchField')?.closest('label');
  if (batchRow) batchRow.style.display = provider === 'batch_ocr' ? '' : 'none';
  const responseModeEl = $('#ocrCfgResponseMode');
  if (responseModeEl) {
    if (provider === 'batch_ocr') {
      responseModeEl.value = 'structured_json';
    } else if (!responseModeEl.value || responseModeEl.value === 'structured_json') {
      responseModeEl.value = 'text';
    }
  }
}

function fillOcrConfigFormFromSelect() {
  const name = ocrCfgSelect?.value;
  if (!name) return;
  const profile = (ocrConfigStore.profiles || {})[name];
  if (!profile) return;
  const provider = profile.provider || 'batch_ocr';
  const request = profile.request || {};
  const response = profile.response || {};

  $('#ocrCfgName').value = profile.name || name;
  $('#ocrCfgProvider').value = provider;
  $('#ocrCfgPostUrl').value = profile.post_url || '';
  $('#ocrCfgTimeoutSeconds').value = String(profile.timeout_seconds || 600);
  $('#ocrCfgBatchField').value = request.batch_field || 'files';
  $('#ocrCfgFileField').value = request.file_field || 'file';
  try {
    $('#ocrCfgExtraFields').value = JSON.stringify(request.extra_form_fields || {}, null, 2);
  } catch (_) {
    $('#ocrCfgExtraFields').value = '{}';
  }
  $('#ocrCfgResponseMode').value =
    response.mode || (provider === 'batch_ocr' ? 'structured_json' : 'text');
  syncOcrProviderUi();
}

async function handleOcrCfgSave() {
  const name = $('#ocrCfgName')?.value.trim();
  const provider = $('#ocrCfgProvider')?.value || 'batch_ocr';
  const postUrl = $('#ocrCfgPostUrl')?.value.trim();
  const timeoutSeconds = parseInt($('#ocrCfgTimeoutSeconds')?.value || '600', 10) || 600;
  const batchField = $('#ocrCfgBatchField')?.value.trim() || 'files';
  const fileField = $('#ocrCfgFileField')?.value.trim() || 'file';
  const responseMode =
    $('#ocrCfgResponseMode')?.value ||
    (provider === 'batch_ocr' ? 'structured_json' : 'text');
  if (!name || !postUrl) {
    notify('请填写配置名称与 POST 地址', 'warning');
    return;
  }

  let extraFields = {};
  try {
    extraFields = parseJsonObject($('#ocrCfgExtraFields')?.value || '{}');
  } catch (err) {
    notify('extra_form_fields 解析失败：' + String(err), 'error');
    return;
  }

  setBtnLoading(btnOcrCfgSave, true);
  try {
    const base = getApiBaseUrl();
    const payload = {
      name,
      provider,
      post_url: postUrl,
      timeout_seconds: timeoutSeconds,
      request: {
        batch_field: batchField,
        file_field: fileField,
        extra_form_fields: extraFields,
      },
      response: { mode: responseMode },
    };
    const data = await fetchJson(`${base}/ocr-configs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    ocrConfigStore = data;
    renderOcrConfigStore();
    setOcrCfgStatus('保存成功', false);
    notify('保存成功', 'success');
  } catch (err) {
    setOcrCfgStatus(`保存失败：${String(err)}`, true);
    notify('保存失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnOcrCfgSave, false);
  }
}

async function handleOcrCfgActivate() {
  const name = $('#ocrCfgName')?.value.trim() || ocrCfgSelect?.value;
  if (!name) {
    notify('请先选择或输入要激活的配置名称', 'warning');
    return;
  }
  setBtnLoading(btnOcrCfgActivate, true);
  try {
    const base = getApiBaseUrl();
    const data = await fetchJson(`${base}/ocr-configs/${encodeURIComponent(name)}/activate`, {
      method: 'POST',
    });
    ocrConfigStore = data;
    renderOcrConfigStore();
    setOcrCfgStatus(`已激活配置：${name}`, false);
    notify('已激活配置：' + name, 'success');
  } catch (err) {
    setOcrCfgStatus(`激活失败：${String(err)}`, true);
    notify('激活失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnOcrCfgActivate, false);
  }
}

async function handleOcrCfgDelete() {
  const name = $('#ocrCfgName')?.value.trim() || ocrCfgSelect?.value;
  if (!name) {
    notify('请先选择或输入要删除的配置名称', 'warning');
    return;
  }
  if (!confirm(`确定删除配置 ${name} 吗？`)) return;
  setBtnLoading(btnOcrCfgDelete, true);
  try {
    const base = getApiBaseUrl();
    const data = await fetchJson(`${base}/ocr-configs/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    ocrConfigStore = data;
    renderOcrConfigStore();
    setOcrCfgStatus(`已删除配置：${name}`, false);
    notify('已删除配置：' + name, 'success');
  } catch (err) {
    setOcrCfgStatus(`删除失败：${String(err)}`, true);
    notify('删除失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnOcrCfgDelete, false);
  }
}

async function handleOcrCfgTest() {
  const name = $('#ocrCfgName')?.value.trim() || ocrCfgSelect?.value;
  if (!name) {
    notify('请先选择或输入要测试的配置名称', 'warning');
    return;
  }
  setBtnLoading(btnOcrCfgTest, true);
  try {
    const base = getApiBaseUrl();
    setOcrCfgStatus('测试中…', false);
    const data = await fetchJson(`${base}/ocr-configs/${encodeURIComponent(name)}/test`, {
      method: 'POST',
    });
    if (data && data.ok) {
      setOcrCfgStatus(`可达（HTTP ${data.status_code}）`, false);
      notify(`可达（HTTP ${data.status_code}）`, 'success');
    } else {
      const msg = (data && (data.error || data.detail)) || 'unknown';
      setOcrCfgStatus(`不可达：${msg}`, true);
      notify('不可达：' + msg, 'error');
    }
  } catch (err) {
    setOcrCfgStatus(`测试失败：${String(err)}`, true);
    notify('测试失败：' + String(err), 'error');
  } finally {
    setBtnLoading(btnOcrCfgTest, false);
  }
}
