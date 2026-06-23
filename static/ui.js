(() => {
  const THEME_KEY = "apiuse_theme_v1";

  const ICON_SUN = `
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3v2.25M12 18.75V21M4.22 4.22l1.59 1.59M18.19 18.19l1.59 1.59M3 12h2.25M18.75 12H21M4.22 19.78l1.59-1.59M18.19 5.81l1.59-1.59"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
      />
      <path
        d="M15.5 12a3.5 3.5 0 1 1-7 0 3.5 3.5 0 0 1 7 0Z"
        stroke="currentColor"
        stroke-width="1.8"
      />
    </svg>
  `;

  const ICON_MOON = `
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M21 14.3A7.5 7.5 0 0 1 9.7 3a6.2 6.2 0 0 0 8.3 8.3A7.5 7.5 0 0 1 21 14.3Z"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linejoin="round"
      />
    </svg>
  `;

  const ICON_X = `
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M7 7l10 10M17 7L7 17"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
      />
    </svg>
  `;

  function ready(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  function systemPrefersDark() {
    try {
      return !!window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch {
      return false;
    }
  }

  function getStoredTheme() {
    try {
      const v = String(localStorage.getItem(THEME_KEY) || "").trim().toLowerCase();
      if (v === "light" || v === "dark") return v;
    } catch {
      // ignore
    }
    return null;
  }

  function getEffectiveTheme() {
    const forced = String(document.documentElement.dataset.theme || "").trim().toLowerCase();
    if (forced === "light" || forced === "dark") return forced;
    return systemPrefersDark() ? "dark" : "light";
  }

  function updateThemeToggle() {
    const btn = document.getElementById("themeToggle");
    if (!btn) return;

    const effective = getEffectiveTheme();
    const next = effective === "dark" ? "light" : "dark";

    btn.innerHTML = next === "dark" ? ICON_MOON : ICON_SUN;
    btn.setAttribute(
      "aria-label",
      next === "dark" ? "切换到深色主题（右键恢复跟随系统）" : "切换到浅色主题（右键恢复跟随系统）",
    );
    btn.setAttribute("title", btn.getAttribute("aria-label") || "");
    btn.setAttribute("aria-pressed", String(effective === "dark"));
  }

  function setTheme(theme) {
    const t = String(theme || "").trim().toLowerCase();
    if (t === "light" || t === "dark") {
      document.documentElement.dataset.theme = t;
      try {
        localStorage.setItem(THEME_KEY, t);
      } catch {
        // ignore
      }
      updateThemeToggle();
      return;
    }

    // auto
    try {
      localStorage.removeItem(THEME_KEY);
    } catch {
      // ignore
    }
    delete document.documentElement.dataset.theme;
    updateThemeToggle();
  }

  function ensureToastRegion() {
    let region = document.getElementById("toastRegion");
    if (region) return region;
    region = document.createElement("div");
    region.id = "toastRegion";
    region.className = "toast-region";
    region.setAttribute("role", "status");
    region.setAttribute("aria-live", "polite");
    region.setAttribute("aria-relevant", "additions");
    document.body.appendChild(region);
    return region;
  }

  function removeToast(el) {
    if (!el) return;
    try {
      const id = parseInt(String(el.dataset.timer || "0"), 10);
      if (id) window.clearTimeout(id);
    } catch {
      // ignore
    }
    el.classList.add("is-leaving");
    let delay = 180;
    try {
      delay =
        window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 180;
    } catch {
      delay = 180;
    }
    window.setTimeout(() => {
      try {
        el.remove();
      } catch {
        // ignore
      }
    }, delay);
  }

  function toast(message, opts = {}) {
    const raw = String(message || "").trim();
    if (!raw) return null;

    const msg = raw.length > 900 ? raw.slice(0, 900) + "…" : raw;
    const type = String(opts.type || "info").trim().toLowerCase();
    const title = String(opts.title || "").trim();
    const duration =
      typeof opts.duration === "number" && Number.isFinite(opts.duration)
        ? opts.duration
        : type === "error"
          ? 6000
          : 3200;

    const region = ensureToastRegion();
    const el = document.createElement("div");
    el.className = `toast toast--${type}`;
    el.tabIndex = 0;

    const dot = document.createElement("div");
    dot.className = "toast__dot";

    const content = document.createElement("div");
    content.className = "toast__content";

    const titleEl = document.createElement("div");
    titleEl.className = "toast__title";
    titleEl.textContent =
      title ||
      (type === "success"
        ? "已完成"
        : type === "warning"
          ? "提示"
          : type === "error"
            ? "出错了"
            : "通知");

    const msgEl = document.createElement("div");
    msgEl.className = "toast__msg";
    msgEl.textContent = msg;

    content.appendChild(titleEl);
    content.appendChild(msgEl);

    const close = document.createElement("button");
    close.type = "button";
    close.className = "toast__close";
    close.setAttribute("aria-label", "关闭通知");
    close.innerHTML = ICON_X;
    close.addEventListener("click", () => removeToast(el));

    el.appendChild(dot);
    el.appendChild(content);
    el.appendChild(close);
    region.appendChild(el);

    const timer = window.setTimeout(() => removeToast(el), Math.max(1200, Math.min(15000, duration)));
    el.dataset.timer = String(timer);

    el.addEventListener("mouseenter", () => {
      const id = parseInt(String(el.dataset.timer || "0"), 10);
      if (id) window.clearTimeout(id);
      el.dataset.timer = "0";
    });
    el.addEventListener("mouseleave", () => {
      if (String(el.dataset.timer || "0") !== "0") return;
      const t2 = window.setTimeout(() => removeToast(el), Math.max(1200, Math.min(15000, duration)));
      el.dataset.timer = String(t2);
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Escape") removeToast(el);
    });

    return el;
  }

  function notify(message, type = "info") {
    const text = String(message ?? "").trim();
    if (!text) return null;
    return toast(text, { type });
  }

  function setButtonLoading(button, isLoading) {
    if (!button) return;
    const btn = button;
    if (isLoading) {
      btn.classList.add("is-loading");
      btn.disabled = true;
      return;
    }
    btn.classList.remove("is-loading");
    btn.disabled = false;
  }

  const DEFAULT_UI_CACHE_KEY = "apiuse_ui_cache_v1";

  function loadUiCache(cacheKey = DEFAULT_UI_CACHE_KEY) {
    try {
      const raw = localStorage.getItem(String(cacheKey || DEFAULT_UI_CACHE_KEY));
      const parsed = raw ? JSON.parse(raw) : null;
      return parsed && typeof parsed === "object" ? parsed : { shared: {}, pages: {} };
    } catch {
      return { shared: {}, pages: {} };
    }
  }

  function saveUiCache(cache, cacheKey = DEFAULT_UI_CACHE_KEY) {
    try {
      localStorage.setItem(
        String(cacheKey || DEFAULT_UI_CACHE_KEY),
        JSON.stringify(cache || { shared: {}, pages: {} }),
      );
    } catch {
      // ignore
    }
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

  function toExcludeSet(excludeIds) {
    if (excludeIds instanceof Set) return excludeIds;
    if (Array.isArray(excludeIds)) return new Set(excludeIds.map((x) => String(x || "")));
    return new Set();
  }

  function restoreUiCache(options = {}) {
    const cacheKey = options.cacheKey || DEFAULT_UI_CACHE_KEY;
    const pageKey = String(options.pageKey || "").trim();
    const apiInputId = String(options.apiInputId || "apiBaseUrl").trim();
    const excludeIds = toExcludeSet(options.excludeIds);
    const cache = loadUiCache(cacheKey);
    const shared = cache.shared && typeof cache.shared === "object" ? cache.shared : {};
    const pages = cache.pages && typeof cache.pages === "object" ? cache.pages : {};
    const page = pageKey && pages[pageKey] && typeof pages[pageKey] === "object" ? pages[pageKey] : {};

    const apiInput = document.getElementById(apiInputId);
    if (apiInput && shared.apiBaseUrl) applyCacheValue(apiInput, shared.apiBaseUrl);

    Object.keys(page).forEach((id) => {
      if (!id || excludeIds.has(id)) return;
      const el = document.getElementById(id);
      if (!el) return;
      if (String(el.type || "").toLowerCase() === "file") return;
      if (el.dataset && String(el.dataset.noCache || "").toLowerCase() === "true") return;
      applyCacheValue(el, page[id]);
    });
  }

  function persistUiField(el, options = {}) {
    if (!el || !el.id) return;
    const id = String(el.id || "");
    const cacheKey = options.cacheKey || DEFAULT_UI_CACHE_KEY;
    const pageKey = String(options.pageKey || "").trim();
    const apiInputId = String(options.apiInputId || "apiBaseUrl").trim();
    const excludeIds = toExcludeSet(options.excludeIds);
    if (!id || excludeIds.has(id)) return;
    if (String(el.type || "").toLowerCase() === "file") return;
    if (el.dataset && String(el.dataset.noCache || "").toLowerCase() === "true") return;

    const cache = loadUiCache(cacheKey);
    if (!cache.shared || typeof cache.shared !== "object") cache.shared = {};
    if (!cache.pages || typeof cache.pages !== "object") cache.pages = {};
    if (pageKey) {
      if (!cache.pages[pageKey] || typeof cache.pages[pageKey] !== "object") {
        cache.pages[pageKey] = {};
      }
    }

    const value = readCacheValue(el);
    if (id === apiInputId) {
      cache.shared.apiBaseUrl = value;
    } else if (pageKey) {
      cache.pages[pageKey][id] = value;
    }
    saveUiCache(cache, cacheKey);
  }

  function bindUiCache(options = {}) {
    const cacheKey = options.cacheKey || DEFAULT_UI_CACHE_KEY;
    const pageKey = String(options.pageKey || "").trim();
    const apiInputId = String(options.apiInputId || "apiBaseUrl").trim();
    const excludeIds = toExcludeSet(options.excludeIds);
    const elements = document.querySelectorAll("input[id], textarea[id], select[id]");
    elements.forEach((el) => {
      const id = String(el.id || "");
      if (!id || excludeIds.has(id)) return;
      if (String(el.type || "").toLowerCase() === "file") return;
      if (el.dataset && String(el.dataset.noCache || "").toLowerCase() === "true") return;

      const type = String(el.type || "").toLowerCase();
      const onChange = () => persistUiField(el, { cacheKey, pageKey, excludeIds, apiInputId });
      el.addEventListener("change", onChange);
      if (
        type === "text" ||
        type === "number" ||
        type === "search" ||
        String(el.tagName || "").toLowerCase() === "textarea"
      ) {
        el.addEventListener("input", onChange);
      }
    });
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

  function getApiBaseUrl(options = {}) {
    const inputSelector = String(options.inputSelector || "#apiBaseUrl");
    const input = document.querySelector(inputSelector);
    const raw = input && input.value ? input.value.trim() : "";
    const origin =
      String(options.fallbackOrigin || window.location.origin || "").replace(/\/+$/, "");
    const val = raw || origin;
    return normalizeApiBaseUrl(val, origin).replace(/\/+$/, "");
  }

  function initApiBaseUrl(options = {}) {
    const inputSelector = String(options.inputSelector || "#apiBaseUrl");
    const input = document.querySelector(inputSelector);
    if (!input) return;
    const raw = input.value ? input.value.trim() : "";
    const origin = String(window.location.origin || "").replace(/\/+$/, "");
    if (!origin) return;
    if (!raw || raw === "http://localhost:12000") input.value = origin;
  }

  function clamp01(v) {
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isFinite(n)) return 0;
    if (n <= 0) return 0;
    if (n >= 1) return 1;
    return n;
  }

  function fmtScore(v, digits = 4) {
    const n = typeof v === "number" ? v : Number(v);
    if (!Number.isFinite(n)) return (0).toFixed(digits);
    return n.toFixed(digits);
  }

  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    if (attrs && typeof attrs === "object") {
      Object.entries(attrs).forEach(([k, v]) => {
        if (v === null || v === undefined) return;
        if (k === "className") node.className = String(v);
        else if (k === "text") node.textContent = String(v);
        else if (k === "html") node.innerHTML = String(v);
        else if (k.startsWith("data-")) node.setAttribute(k, String(v));
        else if (k === "role" || k.startsWith("aria-")) node.setAttribute(k, String(v));
        else if (k === "title") node.setAttribute("title", String(v));
        else node[k] = v;
      });
    }
    const list = Array.isArray(children) ? children : [children];
    list.forEach((c) => {
      if (c === null || c === undefined) return;
      if (typeof c === "string" || typeof c === "number") node.appendChild(document.createTextNode(String(c)));
      else node.appendChild(c);
    });
    return node;
  }

  function getUnsupervisedEval(item) {
    if (!item || typeof item !== "object") return null;
    const ue = item.unsupervised_evaluation;
    if (!ue || typeof ue !== "object") return null;
    const scores = ue.scores && typeof ue.scores === "object" ? ue.scores : {};
    const meta = ue.meta && typeof ue.meta === "object" ? ue.meta : {};
    return { method: ue.method, scores, meta };
  }

  function getScore(scores, key, fallbacks = []) {
    if (!scores || typeof scores !== "object") return 0;
    const keys = [key, ...(Array.isArray(fallbacks) ? fallbacks : [])];
    for (const k of keys) {
      const v = scores[k];
      const n = typeof v === "number" ? v : Number(v);
      if (Number.isFinite(n)) return n;
    }
    return 0;
  }

  function finiteNumberOrNull(v) {
    const n = typeof v === "number" ? v : Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function scoreTier(v) {
    const n = clamp01(v);
    if (n >= 0.85) return { key: "good", label: "高" };
    if (n >= 0.6) return { key: "ok", label: "中" };
    return { key: "bad", label: "低" };
  }

  function nliLabelZh(label) {
    const l = String(label || "").toLowerCase();
    if (l === "entailment") return "支持";
    if (l === "contradiction") return "冲突";
    if (l === "neutral") return "不确定";
    return String(label || "");
  }

  function createScoreGroup(title, scores, reasons = {}, options = {}) {
    if (!scores || typeof scores !== "object") return null;
    const sortKeys = options.sortKeys === true;
    const digits = Number.isFinite(Number(options.digits)) ? Number(options.digits) : 2;
    const reasonPlacement = options.reasonPlacement === "row" ? "row" : "group";
    const barValue =
      typeof options.barValue === "function"
        ? options.barValue
        : (_metric, rawScore) => rawScore;

    const keys = Object.keys(scores);
    if (!keys.length) return null;
    if (sortKeys) keys.sort();

    const group = el("div", { className: "score-group" }, [
      el("div", { className: "score-group-title", text: String(title || "") }),
    ]);

    keys.forEach((metric) => {
      const raw = scores[metric];
      const score = typeof raw === "number" ? raw : Number(raw);
      if (!Number.isFinite(score)) return;

      const row = el("div", { className: "score-row" }, [
        el("div", { className: "metric", text: metric, title: metric }),
        (() => {
          const bar = el("div", { className: "score-bar" });
          const fill = el("div", { className: "score-bar-fill" });
          fill.style.width = `${clamp01(barValue(metric, score)) * 100}%`;
          bar.appendChild(fill);
          return bar;
        })(),
        el("div", { className: "score-num", text: score.toFixed(digits) }),
      ]);

      const reasonRaw = reasons && typeof reasons === "object" ? reasons[metric] : "";
      const reasonText =
        typeof reasonRaw === "string"
          ? reasonRaw
          : reasonRaw && typeof reasonRaw === "object" && typeof reasonRaw.reasons === "string"
            ? reasonRaw.reasons
            : "";

      if (reasonText && reasonText.trim()) {
        const reasonEl = el("div", { className: "reason", text: reasonText.trim() });
        if (reasonPlacement === "row") row.appendChild(reasonEl);
        else group.appendChild(reasonEl);
      }

      group.appendChild(row);
    });

    return group.childElementCount > 1 ? group : null;
  }

  function renderScoreRow(metricName, v) {
    const clamped = clamp01(v);
    const row = el("div", { className: "score-row" }, [
      el("div", { className: "metric", text: metricName }),
      (() => {
        const bar = el("div", { className: "score-bar" });
        const fill = el("div", { className: "score-bar-fill" });
        fill.style.width = `${(clamped * 100).toFixed(1)}%`;
        bar.appendChild(fill);
        return bar;
      })(),
      el("div", { className: "score-num", text: fmtScore(clamped, 4) }),
    ]);
    return row;
  }

  function renderHighlightExcerpt(text, opts = {}) {
    const o = opts && typeof opts === "object" ? opts : {};
    const start = o.start ?? null;
    const end = o.end ?? null;
    const needle = String(o.needle || "");
    const radius = o.radius ?? 90;
    const raw = String(text || "");
    if (!raw.trim()) return el("div", { className: "explain-excerpt muted", text: "（无上下文内容）" });

    let s = typeof start === "number" ? start : null;
    let e = typeof end === "number" ? end : null;

    if ((s === null || e === null) && needle) {
      const idx = raw.indexOf(String(needle));
      if (idx >= 0) {
        s = idx;
        e = idx + String(needle).length;
      }
    }

    if (s === null || e === null || s < 0 || e <= s || s >= raw.length) {
      const take = Math.min(260, raw.length);
      const excerpt = raw.slice(0, take);
      const more = raw.length > take ? "…" : "";
      return el("div", { className: "explain-excerpt" }, [el("span", { text: excerpt + more })]);
    }

    const r = Math.max(20, Math.min(220, Number(radius) || 90));
    const left = Math.max(0, s - r);
    const right = Math.min(raw.length, e + r);
    const prefix = left > 0 ? "…" : "";
    const suffix = right < raw.length ? "…" : "";

    const wrap = el("div", { className: "explain-excerpt" });
    wrap.appendChild(document.createTextNode(prefix + raw.slice(left, s)));
    wrap.appendChild(el("span", { className: "hl", text: raw.slice(s, e) }));
    wrap.appendChild(document.createTextNode(raw.slice(e, right) + suffix));
    return wrap;
  }

  function resolveSpanText(text, opts = {}) {
    const o = opts && typeof opts === "object" ? opts : {};
    const raw = String(text || "");
    const startNum = finiteNumberOrNull(o.start);
    const endNum = finiteNumberOrNull(o.end);
    if (startNum !== null && endNum !== null) {
      const s = Math.max(0, Math.trunc(startNum));
      const e = Math.min(raw.length, Math.trunc(endNum));
      if (e > s && s < raw.length) return raw.slice(s, e);
    }
    return String(o.needle || "").trim();
  }

  function coverageWeightModeZh(mode) {
    const raw = String(mode || "").trim().toLowerCase();
    if (raw === "centered_sparse_v1") return "居中稀疏归一化";
    if (raw === "relevance_normalized_fallback_v1") return "线性归一化回退";
    if (raw === "uniform_fallback_v1") return "均匀权重回退";
    if (raw === "question_conditioned_centered_sparse_v2") return "问题条件化 + 居中稀疏权重";
    return raw || "未知";
  }

  function renderCoverageUnitMetric(label, value) {
    return el("div", { className: "coverage-unit-metric" }, [
      el("div", { className: "coverage-unit-metric-label", text: label }),
      el("div", { className: "coverage-unit-metric-value", text: fmtScore(value, 4) }),
    ]);
  }

  function renderCoverageUnitMetricGroup(title, formulaText, metrics) {
    const items = Array.isArray(metrics) ? metrics.filter((it) => it && it.value != null && Number.isFinite(Number(it.value))) : [];
    if (!items.length) return null;
    const body = el("div", { className: "coverage-unit-group-grid" });
    items.forEach((it) => {
      body.appendChild(renderCoverageUnitMetric(String(it.label || ""), Number(it.value)));
    });
    return el("div", { className: "coverage-unit-group" }, [
      el("div", { className: "coverage-unit-group-title", text: title }),
      formulaText ? el("div", { className: "coverage-unit-group-formula", text: formulaText }) : null,
      body,
    ]);
  }

  function renderCoverageUnitCard(unit, kind) {
    const item = unit && typeof unit === "object" ? unit : {};
    const text = String(item.text || "").trim();
    const metrics = el("div", { className: "coverage-unit-metrics" });
    const mode = String(kind || "").trim().toLowerCase();

    if (mode === "relevant") {
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 1 步：原始信号", "先看这条题和该信息点是否相关、是否匹配。", [
          { label: "问题相关（q）", value: item.question_score },
          { label: "答案匹配（a）", value: item.answer_anchor },
          { label: "原始相似（base）", value: item.qa_score_base },
          { label: "标定分位（cal）", value: item.qa_score_calibrated },
        ]),
      );
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 2 步：中间计算", "先算 qa = sqrt(base × cal)，再算 rel = (q + a + qa) / 3。", [
          { label: "单题覆盖（qa）", value: item.qa_score },
          { label: "综合相关（rel）", value: item.relevance },
          { label: "高于均值部分", value: item.centered_relevance },
        ]),
      );
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 3 步：权重结果", "只保留高于均值的部分，再归一化得到权重 w。", [
          { label: "权重（w）", value: item.weight },
        ]),
      );
    } else if (mode === "support") {
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 1 步：原始信号", "先看问题、答案和完整 QA 对该信息点的基础匹配。", [
          { label: "问题相关（q）", value: item.question_score },
          { label: "答案匹配（a）", value: item.answer_anchor },
          { label: "原始相似（base）", value: item.qa_score_base },
          { label: "标定分位（cal）", value: item.qa_score_calibrated },
        ]),
      );
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 2 步：中间计算", "先算 qa = sqrt(base × cal)，再算 rel = (q + a + qa) / 3。", [
          { label: "单题覆盖（qa）", value: item.p ?? item.qa_score },
          { label: "综合相关（rel）", value: item.relevance },
          { label: "高于均值部分", value: item.centered_relevance },
        ]),
      );
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 3 步：最终结果", "由高于均值部分归一化得到 w，最后 c = w × qa。", [
          { label: "权重（w）", value: item.weight },
          { label: "直接贡献（c）", value: item.contribution },
        ]),
      );
    } else if (mode === "worst") {
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 1 步：先找组内最像的题", "对这个信息点，在同组所有 QA 里找原始相似度最高的那条，得到 sim。", [
          { label: "原始相似（sim）", value: item.sim_max },
        ]),
      );
      if (item.p_calibrated != null) {
        metrics.appendChild(
          renderCoverageUnitMetricGroup("第 2 步：再做负样本标定", "把 sim 放进负样本分布，得到标定分位 cal = F_neg(sim)。1-cal 越大，越像随机错配。", [
            { label: "标定分位（cal）", value: item.p_calibrated },
            { label: "随机错配余量", value: 1 - clamp01(item.p_calibrated) },
          ]),
        );
      }
      metrics.appendChild(
        renderCoverageUnitMetricGroup("第 3 步：得到组级覆盖", item.p_calibrated != null ? "最后用 p_best = sqrt(sim × cal) 得到这个信息点在整组里的最佳覆盖。" : "最终 p_best 表示这个信息点在整组里的最佳覆盖。", [
          { label: "组内最佳覆盖（p_best）", value: item.p },
        ]),
      );
    }

    return el("div", { className: "coverage-unit-card" }, [
      metrics,
      el("div", { className: "coverage-unit-text" }, [
        el("div", { className: "coverage-unit-text-title", text: "信息点内容" }),
        el("div", { text }),
      ]),
    ]);
  }

  function renderCoverageUnitList(items, kind) {
    const wrap = el("div", { className: "coverage-unit-list" });
    (Array.isArray(items) ? items : []).slice(0, 6).forEach((unit) => {
      wrap.appendChild(renderCoverageUnitCard(unit, kind));
    });
    return wrap;
  }

  function coverageUnitKey(unit) {
    const text = String(unit?.text || "")
      .replace(/\s+/g, " ")
      .trim();
    return text || "";
  }

  function mergeCoverageUnitsForExplain(relevantUnits, supportUnits) {
    const relevant = Array.isArray(relevantUnits) ? relevantUnits : [];
    const support = Array.isArray(supportUnits) ? supportUnits : [];
    const merged = [];
    const seen = new Set();
    const supportMap = new Map();

    support.forEach((unit) => {
      const key = coverageUnitKey(unit);
      if (!key) return;
      supportMap.set(key, unit);
    });

    support.forEach((unit) => {
      const key = coverageUnitKey(unit);
      if (!key || seen.has(key)) return;
      const fromRelevant = relevant.find((it) => coverageUnitKey(it) === key) || {};
      merged.push({ ...fromRelevant, ...unit, _in_relevant: true, _in_support: true });
      seen.add(key);
    });

    relevant.forEach((unit) => {
      const key = coverageUnitKey(unit);
      if (!key || seen.has(key)) return;
      merged.push({ ...unit, _in_relevant: true, _in_support: supportMap.has(key) });
      seen.add(key);
    });

    return merged.slice(0, 6);
  }

  function renderUnsupervisedExplain(item, opts = {}) {
    const options = opts && typeof opts === "object" ? opts : {};
    const includeQa = options.includeQa !== false;
    const includeRaw = options.includeRaw !== false;

    const ue = getUnsupervisedEval(item);
    if (!ue) {
      return el("div", { className: "muted", text: "该条记录没有 unsupervised_evaluation 字段，无法展示解释。" });
    }

    const scores = ue.scores || {};
    const meta = ue.meta || {};

    const faith = getScore(scores, "faithfulness");
    const ans = getScore(scores, "answerability", ["p"]);
    const cov = getScore(scores, "coverage_score");
    const covSelf = getScore(scores, "coverage_self");
    const covGroup = getScore(scores, "coverage_recall_soft", ["r_soft"]);
    const f1 = getScore(scores, "unsupervised_f1", ["f1"]);
    const fullCtx = String(item?.source_fact_text || item?.context || item?.source || "");
    const answerText = String(item?.answer || "").trim();

    const wrap = el("div", { className: "explain-wrap" });

    if (includeQa) {
      wrap.appendChild(
        el("div", { className: "explain-qa" }, [
          el("div", { className: "explain-qa-row" }, [
            el("div", { className: "explain-qa-k", text: "问" }),
            el("div", { className: "explain-qa-v", text: String(item?.question || "") }),
          ]),
          el("div", { className: "explain-qa-row" }, [
            el("div", { className: "explain-qa-k", text: "答" }),
            el("div", { className: "explain-qa-v", text: String(item?.answer || "") }),
          ]),
        ]),
      );
    }

    const grid = el("div", { className: "score-grid" });
    const cards = [
      { k: "faithfulness", name: "忠实度 (F)", v: faith },
      { k: "answerability", name: "可回答性 (P)", v: ans },
      { k: "coverage_score", name: "Coverage", v: cov },
      { k: "unsupervised_f1", name: "无监督 F1", v: f1 },
    ];
    cards.forEach((c) => {
      const tier = scoreTier(c.v);
      const card = el("div", { className: `score-card tier-${tier.key}` }, [
        el("div", { className: "score-name", text: c.name }),
        el("div", { className: "score-val", text: fmtScore(c.v, 4) }),
        el("div", { className: "score-tip", text: `等级：${tier.label}` }),
      ]);
      grid.appendChild(card);
    });
    wrap.appendChild(grid);

    // --- Faithfulness ---
    const faithMeta =
      (meta && typeof meta === "object" && meta.faithfulness && typeof meta.faithfulness === "object"
        ? meta.faithfulness
        : null) || (String(ue.method || "").trim() === "nli_faithfulness_v1" ? meta : null);

    const faithSec = el("section", { className: "explain-section" }, [
      el("h3", { className: "explain-h", text: "1) 忠实度（Faithfulness）" }),
      renderScoreRow("F = faithfulness（与上下文一致的概率）", faith),
    ]);

    if (faithMeta && typeof faithMeta === "object") {
      const expected = String(faithMeta.expected_label || "").toLowerCase();
      const pred = String(faithMeta.pred_label || "").toLowerCase();
      const expectedZh = expected ? nliLabelZh(expected) : "";
      const predZh = pred ? nliLabelZh(pred) : "";
      const strategy = String(faithMeta.strategy || "").toLowerCase();
      const hypoModeRaw = String(faithMeta.hypothesis_mode || "").trim().toLowerCase();
      const hypoError = String(faithMeta.hypothesis_error || "").trim();
      const hypo = String(faithMeta.hypothesis || "").trim();
      const worst = Array.isArray(faithMeta.worst_clauses) ? faithMeta.worst_clauses : [];
      const clauses = Array.isArray(faithMeta.clauses) ? faithMeta.clauses : [];

      const summaryLineParts = [];
      if (predZh) summaryLineParts.push(`模型判断：${predZh}`);
      if (expectedZh && expected !== pred && expectedZh !== predZh) summaryLineParts.push(`（期望：${expectedZh}）`);
      if (strategy) summaryLineParts.push(`策略：${strategy}`);
      if (hypoModeRaw && hypoModeRaw !== "llm") {
        const srcZh = hypoModeRaw === "fallback_answer_sentences" ? "答案分句兜底" : hypoModeRaw;
        summaryLineParts.push(`来源：${srcZh}`);
      }
      faithSec.appendChild(el("p", { className: "hint", text: summaryLineParts.join(" ") }));
      if (!clauses.length) {
        faithSec.appendChild(
          el("div", {
            className: "coverage-note muted",
            text:
              "说明：下面“证据候选”的分数不是最终忠实度分，只是词面匹配排序分。系统会先挑出 top 若干候选单元，把它们拼接成一个证据片段（premise_excerpt），再用这个片段和上面的陈述句做 1 次 NLI。最终忠实度 F 看的是后面 NLI 概率里的 entailment。",
          }),
        );
      }

      if (hypoModeRaw && hypoModeRaw !== "llm" && hypoError) {
        const shortErr = hypoError.length > 220 ? hypoError.slice(0, 220) + "…" : hypoError;
        faithSec.appendChild(el("p", { className: "muted", text: `LLM 改写失败，已启用兜底：${shortErr}` }));
      }

      if (clamp01(faith) < 0.85) {
        const reason =
          pred === "contradiction"
            ? "陈述句与上下文可能存在冲突。"
            : pred === "neutral"
              ? "上下文对陈述句没有明确支持（不确定）。"
              : "支持概率偏低。";
        faithSec.appendChild(
          el("p", {
            className: "muted",
            text: `可能扣分原因：${reason} 建议优先核对答案中的关键事实是否能在上下文中逐句找到依据。`,
          }),
        );
      }

      if (hypo) {
        faithSec.appendChild(el("div", { className: "explain-kicker", text: "用于判定的陈述句（hypothesis）" }));
        faithSec.appendChild(el("pre", { text: hypo }));
      } else if (clauses.length) {
        faithSec.appendChild(
          el("div", {
            className: "explain-kicker",
            text: "子陈述得分明细（每条子陈述单独选证据 → 单独做 NLI）",
          }),
        );

        const sorted = clauses
          .map((c, i) => ({ c, i }))
          .sort((a, b) => clamp01(a?.c?.p_expected) - clamp01(b?.c?.p_expected));

        sorted.forEach(({ c }, rank) => {
          const clauseText = String(c?.text || "").trim();
          const pExp = clamp01(c?.p_expected);
          const clausePred = String(c?.pred_label || "").toLowerCase();
          const clausePredZh = clausePred ? nliLabelZh(clausePred) : "";

          const short = clauseText.length > 80 ? clauseText.slice(0, 80) + "…" : clauseText;
          const summary = el("summary", {}, [
            el("span", { className: "mono", text: fmtScore(pExp, 4) }),
            el("span", { text: clausePredZh ? `  预测：${clausePredZh}` : "" }),
            el("span", { className: "muted", text: short ? `  ${short}` : "" }),
          ]);

          const clauseDet = el("details", { className: "explain-clause", open: rank === 0 }, [summary]);

          if (clauseText) {
            clauseDet.appendChild(el("div", { className: "explain-kicker", text: "陈述句（该子陈述）" }));
            clauseDet.appendChild(el("pre", { text: clauseText }));
          }

          const premiseExcerpt = String(c?.premise_excerpt || "").trim();
          const premiseText = premiseExcerpt || fullCtx;

          const premiseSel =
            c?.premise_select && typeof c.premise_select === "object" ? c.premise_select : null;
          const evidenceUnits = premiseSel && Array.isArray(premiseSel.evidence_units) ? premiseSel.evidence_units : [];

          const needleCandidates = [];
          if (answerText) needleCandidates.push(answerText);
          if (clauseText) needleCandidates.push(clauseText);
          const topEv = evidenceUnits.length ? String(evidenceUnits[0]?.text || "").trim() : "";
          if (topEv) needleCandidates.push(topEv);
          let needle = "";
          for (const cand of needleCandidates) {
            if (!cand) continue;
            if (premiseText && premiseText.includes(cand)) {
              needle = cand;
              break;
            }
          }

          if (premiseText) {
            clauseDet.appendChild(el("div", { className: "explain-kicker", text: "上下文证据片段（该子陈述）" }));
            clauseDet.appendChild(renderHighlightExcerpt(premiseText, { needle }));
          }

          if (evidenceUnits.length) {
            clauseDet.appendChild(el("div", { className: "explain-kicker", text: "证据候选（按词面匹配排序）" }));
            const ul = el("ul", { className: "explain-list" });
            evidenceUnits.slice(0, 6).forEach((u) => {
              const t = String(u?.text || "").trim();
              const s = typeof u?.lex_score === "number" ? u.lex_score : Number(u?.lex_score);
              ul.appendChild(el("li", {}, [el("span", { className: "mono", text: fmtScore(s, 4) }), el("span", { text: t })]));
            });
            clauseDet.appendChild(ul);
          }

          const probs = c?.probs && typeof c.probs === "object" ? c.probs : {};
          const pEnt = clamp01(probs.entailment);
          const pCon = clamp01(probs.contradiction);
          const pNeu = clamp01(probs.neutral);
          clauseDet.appendChild(
            el("details", { className: "explain-tech" }, [
              el("summary", { text: "该子陈述的 NLI 概率" }),
              el("div", { className: "kv" }, [
                el("div", { className: "kv-row" }, [
                  el("div", { className: "kv-k", text: "entailment" }),
                  el("div", { className: "kv-v mono", text: fmtScore(pEnt, 4) }),
                ]),
                el("div", { className: "kv-row" }, [
                  el("div", { className: "kv-k", text: "contradiction" }),
                  el("div", { className: "kv-v mono", text: fmtScore(pCon, 4) }),
                ]),
                el("div", { className: "kv-row" }, [
                  el("div", { className: "kv-k", text: "neutral" }),
                  el("div", { className: "kv-v mono", text: fmtScore(pNeu, 4) }),
                ]),
              ]),
            ]),
          );

          faithSec.appendChild(clauseDet);
        });
      } else if (worst.length) {
        faithSec.appendChild(el("div", { className: "explain-kicker", text: "最弱的陈述句（最可能导致扣分）" }));
        const ul = el("ul", { className: "explain-list" });
        worst.slice(0, 3).forEach((w) => {
          const t = String(w?.text || "").trim();
          const pExp = w?.p_expected;
          ul.appendChild(
            el("li", {}, [
              el("span", { className: "mono", text: fmtScore(pExp, 4) }),
              el("span", { text: t }),
            ]),
          );
        });
        faithSec.appendChild(ul);
      }

      if (!clauses.length) {
        const premiseExcerpt = String(faithMeta.premise_excerpt || "").trim();
        const premiseText = premiseExcerpt || fullCtx;
        const needleCandidates = [];
        if (hypo) needleCandidates.push(hypo);
        if (answerText) needleCandidates.push(answerText);
        let needle = "";
        for (const c of needleCandidates) {
          if (!c) continue;
          if (premiseText && premiseText.includes(c)) {
            needle = c;
            break;
          }
        }
        if (premiseText) {
          faithSec.appendChild(el("div", { className: "explain-kicker", text: "上下文证据片段（用于判定/便于人工核对）" }));
          faithSec.appendChild(renderHighlightExcerpt(premiseText, { needle }));
        }

        const premiseSel =
          faithMeta.premise_select && typeof faithMeta.premise_select === "object" ? faithMeta.premise_select : null;
        const evidenceUnits = premiseSel && Array.isArray(premiseSel.evidence_units) ? premiseSel.evidence_units : [];
        if (evidenceUnits.length) {
          faithSec.appendChild(el("div", { className: "explain-kicker", text: "证据候选（按词面匹配排序）" }));
          const ul = el("ul", { className: "explain-list" });
          evidenceUnits.slice(0, 6).forEach((u) => {
            const t = String(u?.text || "").trim();
            const s = typeof u?.lex_score === "number" ? u.lex_score : Number(u?.lex_score);
            ul.appendChild(
              el("li", {}, [el("span", { className: "mono", text: fmtScore(s, 4) }), el("span", { text: t })])
            );
          });
          faithSec.appendChild(ul);
        }
      }

      const probs = faithMeta.probs && typeof faithMeta.probs === "object" ? faithMeta.probs : {};
      const pEnt = clamp01(probs.entailment);
      const pCon = clamp01(probs.contradiction);
      const pNeu = clamp01(probs.neutral);
      const tech = el("details", { className: "explain-tech" }, [
        el("summary", { text: "技术细节（NLI 概率）" }),
        el("div", { className: "kv" }, [
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "entailment" }), el("div", { className: "kv-v mono", text: fmtScore(pEnt, 4) })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "contradiction" }), el("div", { className: "kv-v mono", text: fmtScore(pCon, 4) })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "neutral" }), el("div", { className: "kv-v mono", text: fmtScore(pNeu, 4) })]),
        ]),
      ]);
      faithSec.appendChild(tech);
    } else {
      faithSec.appendChild(
        el("p", {
          className: "muted",
          text: "缺少 faithfulness 的解释字段（可能是旧数据，或评估时裁剪了 meta）。",
        }),
      );
    }
    wrap.appendChild(faithSec);

    // --- Answerability ---
    const ansMeta = meta && typeof meta === "object" && meta.answerability && typeof meta.answerability === "object" ? meta.answerability : null;

    const ansSec = el("section", { className: "explain-section" }, [
      el("h3", { className: "explain-h", text: "2) 可回答性（Answerability / Precision P）" }),
      renderScoreRow("P = answerability（从上下文可抽取答案的概率）", ans),
      el("p", { className: "muted", text: "注意：P 只判断“问题是否可从上下文回答”，不检查你提供的 answer 是否正确。" }),
    ]);
    if (ansMeta && typeof ansMeta === "object") {
      const pNoAns = clamp01(ansMeta.p_no_answer);
      const bestSpanStart = finiteNumberOrNull(ansMeta.best_span_start);
      const bestSpanEnd = finiteNumberOrNull(ansMeta.best_span_end);
      const bestSpan = resolveSpanText(fullCtx, {
        start: bestSpanStart,
        end: bestSpanEnd,
        needle: ansMeta.best_span,
      });
      const bestSpanCharLength =
        finiteNumberOrNull(ansMeta.best_span_char_length) ??
        (bestSpan ? bestSpan.length : 0);
      const windowCount = finiteNumberOrNull(ansMeta.window_count);
      const bestWindowIndex = finiteNumberOrNull(ansMeta.best_window_index);
      const maxLength = finiteNumberOrNull(ansMeta.max_length);
      const docStride = finiteNumberOrNull(ansMeta.doc_stride);
      const maxAnswerLength = finiteNumberOrNull(ansMeta.max_answer_length);
      const nBest = finiteNumberOrNull(ansMeta.n_best);
      ansSec.appendChild(el("p", { className: "hint", text: `拒答概率 p_no_answer=${fmtScore(pNoAns, 4)}（越高越不可回答）` }));
      if (clamp01(ans) < 0.85 || pNoAns >= 0.4) {
        ansSec.appendChild(
          el("p", {
            className: "muted",
            text: "可能扣分原因：模型更倾向拒答（或抽取片段不稳定）。建议检查问题是否超出上下文信息，或补充更直接包含答案的证据句。",
          }),
        );
      }
      const splitFacts = [];
      if (maxLength !== null) splitFacts.push(`单个窗口最多 ${Math.trunc(maxLength)} 个 token`);
      if (docStride !== null) splitFacts.push(`相邻窗口重叠 ${Math.trunc(docStride)} 个 token`);
      if (maxAnswerLength !== null) splitFacts.push(`单个答案片段最长 ${Math.trunc(maxAnswerLength)} 个 token`);
      if (windowCount !== null) splitFacts.push(`这条上下文实际切成了 ${Math.trunc(windowCount)} 个重叠窗口`);
      ansSec.appendChild(
        el("p", {
          className: "muted",
          text:
            `best span 不是按自然段人工截出来的，而是抽取式 QA 模型在滑窗后的上下文里预测“起点 token + 终点 token”得到的片段。` +
            (splitFacts.length ? splitFacts.join("；") + "。" : ""),
        }),
      );
      if (bestSpan) {
        ansSec.appendChild(el("div", { className: "explain-kicker", text: "模型抽取出的完整最佳片段（best_span）" }));
        ansSec.appendChild(el("div", { className: "answerability-span-full" }, [el("span", { text: bestSpan })]));
        ansSec.appendChild(
          el("div", { className: "answerability-span-stats" }, [
            el("div", { className: "answerability-span-stat", text: `起点字符：${bestSpanStart !== null ? Math.trunc(bestSpanStart) : "-"}` }),
            el("div", { className: "answerability-span-stat", text: `终点字符：${bestSpanEnd !== null ? Math.trunc(bestSpanEnd) : "-"}` }),
            el("div", { className: "answerability-span-stat", text: `片段长度：${Math.trunc(bestSpanCharLength)} 字符` }),
            el(
              "div",
              {
                className: "answerability-span-stat",
                text:
                  windowCount !== null && bestWindowIndex !== null
                    ? `最佳窗口：第 ${Math.trunc(bestWindowIndex) + 1} / ${Math.trunc(windowCount)} 个`
                    : "最佳窗口：当前数据未记录",
              },
            ),
          ]),
        );
        ansSec.appendChild(el("div", { className: "explain-kicker", text: "该片段在原文中的位置（高亮定位）" }));
        ansSec.appendChild(
          renderHighlightExcerpt(fullCtx, {
            start: bestSpanStart,
            end: bestSpanEnd,
            needle: bestSpan,
            radius: 140,
          }),
        );
      } else {
        ansSec.appendChild(el("p", { className: "muted", text: "未找到明显的答案片段（best_span 为空）。" }));
      }

      const tech = el("details", { className: "explain-tech" }, [
        el("summary", { text: "技术细节（分数与切窗参数）" }),
        el("div", { className: "kv" }, [
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "gap（最佳片段分数 - 拒答分数）" }), el("div", { className: "kv-v mono", text: fmtScore(ansMeta.gap, 4) })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "score_best（最佳片段原始分）" }), el("div", { className: "kv-v mono", text: fmtScore(ansMeta.score_best, 4) })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "score_null（拒答原始分）" }), el("div", { className: "kv-v mono", text: fmtScore(ansMeta.score_null, 4) })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "判定模式" }), el("div", { className: "kv-v", text: String(ansMeta.score_mode_effective || ansMeta.score_mode || "") })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "切窗总数" }), el("div", { className: "kv-v", text: windowCount !== null ? String(Math.trunc(windowCount)) : "-" })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "最佳窗口序号" }), el("div", { className: "kv-v", text: bestWindowIndex !== null ? `第 ${Math.trunc(bestWindowIndex) + 1} 个` : "-" })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "max_length（单窗 token 上限）" }), el("div", { className: "kv-v", text: maxLength !== null ? String(Math.trunc(maxLength)) : "-" })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "doc_stride（窗口重叠 token）" }), el("div", { className: "kv-v", text: docStride !== null ? String(Math.trunc(docStride)) : "-" })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "max_answer_length（答案最长 token）" }), el("div", { className: "kv-v", text: maxAnswerLength !== null ? String(Math.trunc(maxAnswerLength)) : "-" })]),
          el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "n_best（起止候选数量）" }), el("div", { className: "kv-v", text: nBest !== null ? String(Math.trunc(nBest)) : "-" })]),
        ]),
      ]);
      ansSec.appendChild(tech);
    } else {
      ansSec.appendChild(el("p", { className: "muted", text: "缺少 answerability 的解释字段（可能是旧数据，或评估时裁剪了 meta）。" }));
    }
    wrap.appendChild(ansSec);

    // --- Coverage recall ---
    const covMeta = meta && typeof meta === "object" && meta.coverage_recall && typeof meta.coverage_recall === "object" ? meta.coverage_recall : null;
    const covSec = el("section", { className: "explain-section" }, [
      el("h3", { className: "explain-h", text: "3) 覆盖（Coverage）" }),
      renderScoreRow("Coverage = coverage_score（最终 Coverage 分数）", cov),
      renderScoreRow("CoverageSelf = coverage_self（本条 QA 对自身相关信息点的覆盖）", covSelf),
      renderScoreRow("R_group = coverage_recall_soft（同一段 context 的组级覆盖召回）", covGroup),
      el("p", { className: "hint", text: "提示：最终 Coverage 分数使用 Coverage = sqrt(R_group × CoverageSelf)。CoverageSelf 先算每个 unit 的相关性，再用“居中稀疏归一化”把高于平均值的相关性转成权重，最后对覆盖强度做加权平均。" }),
    ]);
      if (covMeta && typeof covMeta === "object") {
        const unitsTotal = covMeta.units_total != null ? String(covMeta.units_total) : "";
        const unitsCovered = covMeta.units_covered;
        const unitsCoveredSoft = covMeta.units_covered_soft;
        const qaTotal = covMeta.qa_total != null ? String(covMeta.qa_total) : "";
        const qaEffective = covMeta.qa_effective != null ? String(covMeta.qa_effective) : "";
        const unitType = String(covMeta.unit_type || "");
        const mappingName = String(covMeta.similarity_mapping || "");
        const baseMappingName = String(covMeta.base_similarity_mapping || "");
      const sigCenter = covMeta.sigmoid_center;
      const sigTemp = covMeta.sigmoid_temperature;
      const sigSrc = String(covMeta.sigmoid_center_source || "");
      const negSamplesTotal = covMeta.neg_samples_total;
      const negSamplesPerGroup = covMeta.neg_samples_per_group;
      const negValidGroups = covMeta.neg_valid_groups;
      const negSamplesExpected = covMeta.neg_samples_total_expected;
      const negSampleFormula = String(covMeta.neg_sample_formula || "");
      const weightMode = String(covMeta.coverage_weight_mode_effective || covMeta.coverage_item_mode || "");
      const relevanceMean = covMeta.relevance_mean;
      const centeredMass = covMeta.centered_relevance_mass;
      const coverageSupportUnits = Array.isArray(covMeta.coverage_support_units) ? covMeta.coverage_support_units : [];
      const relevantUnits = Array.isArray(covMeta.question_relevant_units) ? covMeta.question_relevant_units : [];
      const worstUnits = Array.isArray(covMeta.worst_units) ? covMeta.worst_units : [];
      const visibleSupportContribution = coverageSupportUnits.reduce((sum, unit) => {
        const contribution = Number(unit && unit.contribution);
        return Number.isFinite(contribution) ? sum + contribution : sum;
      }, 0);
      const hiddenSupportContribution = Math.max(0, Number(covSelf || 0) - visibleSupportContribution);
      const unitsTotalNum = Number(covMeta.units_total || 0);
      const unitsCoveredSoftNum = Number(covMeta.units_covered_soft || 0);
      let unitsLine = unitsTotal ? `信息点数量：${unitsTotal}（unit_type=${unitType || "?"}）` : "信息点数量未知";
      if (unitsTotal && unitsCovered != null) unitsLine += `；估计覆盖：${String(unitsCovered)}/${unitsTotal}`;
      else if (unitsTotal && unitsCoveredSoft != null) unitsLine += `；估计覆盖：${fmtScore(unitsCoveredSoft, 1)}/${unitsTotal}`;
      covSec.appendChild(el("p", { className: "hint", text: unitsLine }));
      if (qaTotal || qaEffective) {
        covSec.appendChild(
          el("p", {
            className: "hint",
            text: `同组问答数：${qaEffective || qaTotal}${qaTotal && qaEffective && qaTotal !== qaEffective ? `（原始 ${qaTotal}，有效 ${qaEffective}）` : ""}。R_group 是这组问答共享的组级分，不是这条题单独算出来的分。`,
          }),
        );
      }
      if (unitsTotalNum > 0 && Number.isFinite(unitsCoveredSoftNum)) {
        covSec.appendChild(
          el("div", {
            className: "coverage-note",
          }, [
            el("div", { className: "coverage-note-title", text: "R_group 怎么来的" }),
            el("div", {
              text: `先对每个信息点 unit，看同组所有 QA 里谁覆盖它最好，得到该点的组内最佳覆盖 p_best；再把所有 unit 的 p_best 求平均。`,
            }),
            el("div", {
              className: "mono",
              text: `这次：units_covered_soft = Σ p_best = ${fmtScore(unitsCoveredSoftNum, 4)}`,
            }),
            el("div", {
              className: "mono",
              text: `这次：R_group = units_covered_soft / units_total = ${fmtScore(unitsCoveredSoftNum, 4)} / ${unitsTotalNum} = ${fmtScore(covGroup, 4)}`,
            }),
          ]),
        );
      }
      covSec.appendChild(
        el("div", {
          className: "coverage-note",
        }, [
          el("div", { className: "coverage-note-title", text: "CoverageSelf 怎么来的" }),
          el("div", {
            text: "对这条 QA 的每个相关信息点，先算原始信号 q / a / base / cal，再推出 qa、rel、权重 w，最后把每个点的直接贡献 c = w × qa 相加。",
          }),
          el("div", {
            className: "mono",
            text: `这次：CoverageSelf = Σ(w × qa) = ${fmtScore(covSelf, 4)}`,
          }),
        ]),
      );
      covSec.appendChild(
        el("div", {
          className: "coverage-note",
        }, [
          el("div", { className: "coverage-note-title", text: "最终 Coverage 怎么来的" }),
          el("div", { text: "最终 Coverage 不是简单平均，而是把组级覆盖和单题覆盖做几何平均融合。" }),
          el("div", {
            className: "mono",
            text: `这次：Coverage = sqrt(R_group × CoverageSelf) = sqrt(${fmtScore(covGroup, 4)} × ${fmtScore(covSelf, 4)}) = ${fmtScore(cov, 4)}`,
          }),
        ]),
      );

      if (clamp01(covSelf) <= 0.35 && clamp01(covGroup) >= 0.6) {
        covSec.appendChild(
          el("p", {
            className: "muted",
            text: "CoverageSelf 偏低通常表示：系统认为这条题真正相关的那些 units 上，它自己的覆盖强度还不够稳。即使本组整体 R_group 不低，也可能说明“这组题覆盖得还行，但这条题自身没有覆盖好它该负责的点”。",
          }),
        );
      }

      if (mappingName || baseMappingName || sigCenter != null || sigTemp != null || negSamplesTotal != null) {
        const kv = el("div", { className: "kv" });
        if (mappingName) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "覆盖映射" }), el("div", { className: "kv-v mono", text: mappingName })]));
        if (baseMappingName) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "基础映射" }), el("div", { className: "kv-v mono", text: baseMappingName })]));
        if (weightMode) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "权重方式" }), el("div", { className: "kv-v", text: coverageWeightModeZh(weightMode) })]));
        if (relevanceMean != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "平均相关性" }), el("div", { className: "kv-v mono", text: fmtScore(relevanceMean, 4) })]));
        if (centeredMass != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "高于均值总量" }), el("div", { className: "kv-v mono", text: fmtScore(centeredMass, 4) })]));
        if (negSamplesTotal != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "负样本总数" }), el("div", { className: "kv-v mono", text: String(negSamplesTotal) })]));
        if (negSamplesPerGroup != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "每组上限" }), el("div", { className: "kv-v mono", text: String(negSamplesPerGroup) })]));
        if (negValidGroups != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "有效分组数" }), el("div", { className: "kv-v mono", text: String(negValidGroups) })]));
        if (sigCenter != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "sigmoid 中心" }), el("div", { className: "kv-v mono", text: fmtScore(sigCenter, 4) + (sigSrc ? ` (${sigSrc})` : "") })]));
        if (sigTemp != null) kv.appendChild(el("div", { className: "kv-row" }, [el("div", { className: "kv-k", text: "sigmoid 温度" }), el("div", { className: "kv-v mono", text: fmtScore(sigTemp, 4) })]));
        const tech = el("details", { className: "explain-tech" }, [
          el("summary", { text: "技术口径（映射 / 权重 / 负样本）" }),
          kv,
          negSamplesTotal != null && negSamplesPerGroup != null && negValidGroups != null
            ? el("div", {
                className: "coverage-note muted",
                text:
                  negSamplesExpected != null && negSampleFormula
                    ? `负样本总数 ${negSamplesTotal} 来自本次 ${negValidGroups} 个有效 context 组，按 ${negSampleFormula} 累加得到；在这次结果里就是把每组的 min(${negSamplesPerGroup}, units_total) 相加。`
                    : `负样本总数 ${negSamplesTotal} 是本次全部有效 context 组共同构成的负样本池大小。`,
              })
            : null,
        ]);
        covSec.appendChild(tech);
      }

      const mergedCoverageUnits = mergeCoverageUnitsForExplain(relevantUnits, coverageSupportUnits);
      if (mergedCoverageUnits.length) {
        covSec.appendChild(el("div", { className: "explain-kicker", text: "本题最相关的信息点（已合并相关性与直接贡献）" }));
        covSec.appendChild(
          el("p", {
            className: "muted",
            text:
              "这块已经把原来的 question_relevant_units 和 coverage_support_units 合并展示，避免同一个信息点重复出现。相关性（rel）表示这个信息点和本题有多相关；直接贡献（c）= 权重（w）× QA 覆盖（qa），表示它实际给 CoverageSelf 加了多少分。",
          }),
        );
        covSec.appendChild(
          el("p", {
            className: "muted",
            text:
              hiddenSupportContribution > 0.0005
                ? `这里只展示贡献最高的 ${coverageSupportUnits.length} 个信息点，它们的直接贡献合计是 ${fmtScore(visibleSupportContribution, 4)}；CoverageSelf 实际会把全部信息点的贡献都加总，所以还有 ${fmtScore(hiddenSupportContribution, 4)} 来自未展示的信息点。`
                : `这里只展示贡献最高的 ${coverageSupportUnits.length} 个信息点；当前可见贡献已经基本接近 CoverageSelf 总和。`,
          }),
        );
        covSec.appendChild(
          el("p", {
            className: "muted",
            text:
              "字段说明：相关性=综合相关程度；高于均值部分=相关性减去本题平均相关性后的正值；问题相关=unit 与问题的语义接近程度；答案匹配=答案语义相似和字面锚点匹配二者取大；QA 覆盖=完整 QA 对该 unit 的最终覆盖分；基础相似=原始语义相似度；负样本分位=这个相似度超过了多少比例的错配样本。",
          }),
        );
        covSec.appendChild(renderCoverageUnitList(mergedCoverageUnits, "support"));
      } else if (relevantUnits.length) {
        covSec.appendChild(el("div", { className: "explain-kicker", text: "本题最相关的信息点（question_relevant_units）" }));
        covSec.appendChild(renderCoverageUnitList(relevantUnits, "relevant"));
      } else if (coverageSupportUnits.length) {
        covSec.appendChild(el("div", { className: "explain-kicker", text: "本条 QA 的主要贡献点（coverage_support_units）" }));
        covSec.appendChild(renderCoverageUnitList(coverageSupportUnits, "support"));
      } else {
        covSec.appendChild(el("p", { className: "muted", text: "本条没有明显的 CoverageSelf 主要支撑点。这通常表示：虽然系统能找出相关 units，但这条题在这些 units 上的覆盖强度整体不高。" }));
      }

      const lowThreshold = mappingName.includes("sigmoid") || mappingName.includes("neg_cdf") ? 0.75 : 0.6;
      if (clamp01(covGroup) < lowThreshold) {
        covSec.appendChild(
          el("p", {
            className: "muted",
            text: "组级可能扣分原因：本段里仍有较多信息点没有被本组问答覆盖。建议围绕下方“组缺口（最未覆盖的信息点）”补充问答对。",
          }),
        );
      }
      if (worstUnits.length) {
        const pNote = mappingName.includes("sigmoid")
          ? "p≈覆盖概率，越低越像没被提到"
          : mappingName.includes("neg_cdf")
            ? "sim=组内最佳原始相似度；cal=把 sim 放进负样本分布后的标定分位；p=sqrt(sim×cal)；1-cal 越大，越像随机错配"
            : "p 越低越像没被提到";
        covSec.appendChild(el("div", { className: "explain-kicker", text: `组缺口：最未覆盖的信息点（${pNote}）` }));
        covSec.appendChild(
          el("p", {
            className: "muted",
            text: "这里展示的是整组里 p_best 最低的几个信息点。它们不是单条 QA 的扣分项，而是告诉你：对这些点来说，同组所有 QA 里表现最好的一条题也只覆盖到了这个程度。",
          }),
        );
        covSec.appendChild(
          el("p", {
            className: "muted",
            text: "计算顺序是：先在同组所有 QA 里取该信息点的最大原始相似度 sim，再做 cal = F_neg(sim)，最后得到 p_best = sqrt(sim × cal)。所有信息点的 p_best 加总就是 units_covered_soft，再除以 units_total 得到 R_group。",
          }),
        );
        covSec.appendChild(renderCoverageUnitList(worstUnits, "worst"));
      } else {
        covSec.appendChild(el("p", { className: "muted", text: "没有 worst_units 诊断信息（可能信息点过少或未能计算）。" }));
      }
    } else {
      covSec.appendChild(
        el("p", {
          className: "muted",
          text: "缺少 coverage_recall 的解释字段（可能是旧数据，或评估时裁剪了 meta）。",
        }),
      );
    }
    wrap.appendChild(covSec);

    // --- F1 ---
    const suiteMeta = meta && typeof meta === "object" && meta.suite && typeof meta.suite === "object" ? meta.suite : null;
    const f1Sec = el("section", { className: "explain-section" }, [
      el("h3", { className: "explain-h", text: "4) 无监督 F1（综合得分）" }),
      renderScoreRow("F1 = 2PR/(P+R)（组级）", f1),
      el("p", { className: "hint", text: "提示：F1 是组级指标（同一段 context 内每条 QA 相同），由 P 与 R 共同决定。" }),
    ]);
    if (suiteMeta && typeof suiteMeta === "object") {
      const pGroup = suiteMeta.p_group;
      const rGroup = suiteMeta.r_group;
      const f1Group = suiteMeta.f1_group;
      const mode = String(suiteMeta.precision_mode || "answerability");
      const def = String(suiteMeta.precision_definition || "");
      const groupSize = suiteMeta.group_size != null ? String(suiteMeta.group_size) : "";

      const bottleneck = clamp01(pGroup) <= clamp01(rGroup) ? "P" : "R";
      f1Sec.appendChild(
        el("p", {
          className: "hint",
          text: `本组：P=${fmtScore(pGroup, 4)}，R=${fmtScore(rGroup, 4)}，F1=${fmtScore(f1Group, 4)}（瓶颈：${bottleneck}）`,
        }),
      );
      if (clamp01(f1) < 0.85) {
        f1Sec.appendChild(
          el("p", {
            className: "muted",
            text: `可能扣分原因：${bottleneck} 偏低会显著拉低 F1。建议优先提升较低的一项（提高可回答性或补齐覆盖召回）。`,
          }),
        );
      }
      f1Sec.appendChild(
        el("p", {
          className: "muted",
          text: `P 的计算方式：precision_mode=${mode}${def ? `（${def}）` : ""}${groupSize ? `；group_size=${groupSize}` : ""}`,
        }),
      );
    } else {
      f1Sec.appendChild(el("p", { className: "muted", text: "缺少 suite 的解释字段（可能是旧数据，或评估时裁剪了 meta）。" }));
    }
    wrap.appendChild(f1Sec);

    if (includeRaw) {
      const raw = {
        id: item?.id || "",
        group_id: item?.group_id || item?.source || "",
        original_filename: item?.original_filename || "",
        question: item?.question || "",
        answer: item?.answer || "",
        unsupervised_evaluation: item?.unsupervised_evaluation || null,
      };
      const details = el("details", { className: "explain-tech" }, [
        el("summary", { text: "调试：原始 JSON（节选）" }),
        el("pre", { text: JSON.stringify(raw, null, 2) }),
      ]);
      wrap.appendChild(details);
    }

    return wrap;
  }

  function initTheme() {
    const stored = getStoredTheme();
    if (stored) document.documentElement.dataset.theme = stored;
    updateThemeToggle();

    const toggle = document.getElementById("themeToggle");
    if (toggle) {
      toggle.addEventListener("click", () => {
        const effective = getEffectiveTheme();
        setTheme(effective === "dark" ? "light" : "dark");
      });
      toggle.addEventListener("contextmenu", (ev) => {
        ev.preventDefault();
        setTheme("auto");
        toast("主题已恢复为跟随系统", { type: "info" });
      });
    }

    try {
      if (!stored && window.matchMedia) {
        const mq = window.matchMedia("(prefers-color-scheme: dark)");
        const handler = () => {
          if (!getStoredTheme()) updateThemeToggle();
        };
        if (typeof mq.addEventListener === "function") mq.addEventListener("change", handler);
        else if (typeof mq.addListener === "function") mq.addListener(handler);
      }
    } catch {
      // ignore
    }
  }

  function initFormAttrs() {
    const controls = document.querySelectorAll("input[id], textarea[id], select[id]");
    controls.forEach((el) => {
      const id = String(el.id || "").trim();
      if (!id) return;
      if (!el.getAttribute("name")) el.setAttribute("name", id);

      const tag = String(el.tagName || "").toLowerCase();
      if (tag === "input") {
        const type = String(el.type || "").toLowerCase();
        if (type === "file") return;
        if (!el.getAttribute("autocomplete")) el.setAttribute("autocomplete", "off");
        return;
      }
      if (tag === "textarea") {
        if (!el.getAttribute("autocomplete")) el.setAttribute("autocomplete", "off");
      }
    });
  }

  function enhanceTableWrapScroll() {
    const wraps = Array.from(document.querySelectorAll(".table-wrap"));
    wraps.forEach((wrap) => {
      if (!wrap || !(wrap instanceof HTMLElement)) return;
      if (wrap.dataset && wrap.dataset.topScrollEnhanced === "1") return;
      if (!wrap.parentNode) return;

      const top = document.createElement("div");
      top.className = "table-scroll-top is-hidden";
      top.setAttribute("aria-hidden", "true");

      const inner = document.createElement("div");
      inner.className = "table-scroll-top-inner";
      top.appendChild(inner);

      // Preserve spacing by moving the wrapper's margin-top onto the top scrollbar.
      try {
        const cs = window.getComputedStyle(wrap);
        const mt = cs && cs.marginTop ? cs.marginTop : "0px";
        if (mt && mt !== "0px") {
          top.style.marginTop = mt;
          wrap.style.marginTop = "0px";
        }
      } catch {
        // ignore
      }

      wrap.parentNode.insertBefore(top, wrap);
      wrap.dataset.topScrollEnhanced = "1";

      let syncing = false;
      let rafId = 0;

      const scheduleSync = () => {
        if (rafId) return;
        rafId = window.requestAnimationFrame(() => {
          rafId = 0;
          const scrollW = wrap.scrollWidth || 0;
          const clientW = wrap.clientWidth || 0;
          const hasOverflow = scrollW > clientW + 1;
          inner.style.width = `${Math.max(1, scrollW)}px`;
          top.classList.toggle("is-hidden", !hasOverflow);
          wrap.classList.toggle("has-top-scroll", hasOverflow);
          if (hasOverflow) {
            syncing = true;
            top.scrollLeft = wrap.scrollLeft;
            syncing = false;
          }
        });
      };

      wrap.addEventListener(
        "scroll",
        () => {
          if (syncing) return;
          syncing = true;
          top.scrollLeft = wrap.scrollLeft;
          syncing = false;
        },
        { passive: true },
      );

      top.addEventListener(
        "scroll",
        () => {
          if (syncing) return;
          syncing = true;
          wrap.scrollLeft = top.scrollLeft;
          syncing = false;
        },
        { passive: true },
      );

      if (window.ResizeObserver) {
        try {
          const ro = new window.ResizeObserver(() => scheduleSync());
          ro.observe(wrap);
          const table = wrap.querySelector("table");
          if (table) ro.observe(table);
        } catch {
          // ignore
        }
      }

      if (window.MutationObserver) {
        try {
          const mo = new window.MutationObserver(() => scheduleSync());
          mo.observe(wrap, { childList: true, subtree: true });
        } catch {
          // ignore
        }
      }

      window.addEventListener("resize", scheduleSync, { passive: true });
      scheduleSync();
    });
  }

  window.apiuseUi = window.apiuseUi || {};
  window.apiuseUi.notify = notify;
  window.apiuseUi.toast = toast;
  window.apiuseUi.setTheme = setTheme;
  window.apiuseUi.getEffectiveTheme = getEffectiveTheme;
  window.apiuseUi.setButtonLoading = setButtonLoading;
  window.apiuseUi.loadUiCache = loadUiCache;
  window.apiuseUi.saveUiCache = saveUiCache;
  window.apiuseUi.readCacheValue = readCacheValue;
  window.apiuseUi.applyCacheValue = applyCacheValue;
  window.apiuseUi.restoreUiCache = restoreUiCache;
  window.apiuseUi.persistUiField = persistUiField;
  window.apiuseUi.bindUiCache = bindUiCache;
  window.apiuseUi.normalizeApiBaseUrl = normalizeApiBaseUrl;
  window.apiuseUi.getApiBaseUrl = getApiBaseUrl;
  window.apiuseUi.initApiBaseUrl = initApiBaseUrl;
  window.apiuseUi.createScoreGroup = createScoreGroup;
  window.apiuseUi.renderUnsupervisedExplain = renderUnsupervisedExplain;
  window.apiuseUi.enhanceTableWrapScroll = enhanceTableWrapScroll;

  ready(() => {
    initTheme();
    initFormAttrs();
    enhanceTableWrapScroll();
  });
})();
