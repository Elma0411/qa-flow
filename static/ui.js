(() => {
  const THEME_KEY = "apiuse_theme_v1";
  const DISPLAYED_UNSUPERVISED_METRICS = Object.freeze([
    "faithfulness",
    "answerability",
    "coverage_score",
  ]);

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

  function inferApiBaseUrl() {
    const origin = String(window.location.origin || "").replace(/\/+$/, "");
    const pathname = String(window.location.pathname || "");
    const match = pathname.match(/^(.*)\/ui(?:\/|$)/);
    const proxyPrefix = match ? String(match[1] || "") : "";
    return `${origin}${proxyPrefix}`.replace(/\/+$/, "");
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
      String(options.fallbackOrigin || inferApiBaseUrl()).replace(/\/+$/, "");
    const val = raw || origin;
    return normalizeApiBaseUrl(val, origin).replace(/\/+$/, "");
  }

  function initApiBaseUrl(options = {}) {
    const inputSelector = String(options.inputSelector || "#apiBaseUrl");
    const input = document.querySelector(inputSelector);
    if (!input) return;
    const raw = input.value ? input.value.trim() : "";
    const browserOrigin = String(window.location.origin || "").replace(/\/+$/, "");
    const origin = inferApiBaseUrl();
    if (!origin) return;
    if (!raw || raw === "http://localhost:12000" || raw === browserOrigin) input.value = origin;
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
    const scores = ue.scores ?? {};
    const meta = ue.meta && typeof ue.meta === "object" ? ue.meta : {};
    return { method: ue.method, scores, meta };
  }

  function filterUnsupervisedScores(scores) {
    if (typeof scores === "string") {
      try {
        scores = JSON.parse(scores);
      } catch {
        scores = {};
      }
    }
    if (!scores || typeof scores !== "object") scores = {};
    const filtered = {};
    const validUnitScore = (raw) => {
      const value = typeof raw === "number" ? raw : Number(raw);
      return Number.isFinite(value) && value >= 0 && value <= 1 ? value : 0;
    };
    DISPLAYED_UNSUPERVISED_METRICS.forEach((key) => {
      filtered[key] = validUnitScore(scores[key]);
    });
    // Older records used `p` for answerability. Keep the displayed name stable
    // without exposing the legacy metric key.
    if (scores.answerability === undefined || scores.answerability === null || scores.answerability === "") {
      filtered.answerability = validUnitScore(scores.p);
    }
    const total = DISPLAYED_UNSUPERVISED_METRICS.reduce((sum, key) => sum + filtered[key], 0);
    filtered.average_score = total / DISPLAYED_UNSUPERVISED_METRICS.length;
    return filtered;
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

  function scoreTier(v) {
    const n = clamp01(v);
    if (n >= 0.85) return { key: "good", label: "高" };
    if (n >= 0.6) return { key: "ok", label: "中" };
    return { key: "bad", label: "低" };
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

  function renderUnsupervisedExplain(item, opts = {}) {
    const options = opts && typeof opts === "object" ? opts : {};
    const includeQa = options.includeQa !== false;
    const includeRaw = options.includeRaw !== false;

    const ue = getUnsupervisedEval(item);
    if (!ue) {
      return el("div", { className: "muted", text: "该条记录没有 unsupervised_evaluation 字段，无法展示解释。" });
    }

    const scores = filterUnsupervisedScores(ue.scores);
    const faith = getScore(scores, "faithfulness");
    const ans = getScore(scores, "answerability");
    const cov = getScore(scores, "coverage_score");
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
    [
      { name: "faithfulness（忠实度）", value: faith },
      { name: "answerability（可回答性）", value: ans },
      { name: "coverage_score（覆盖）", value: cov },
      { name: "平均分（三项均值）", value: getScore(scores, "average_score") },
    ].forEach(({ name, value }) => {
      const tier = scoreTier(value);
      grid.appendChild(
        el("div", { className: `score-card tier-${tier.key}` }, [
          el("div", { className: "score-name", text: name }),
          el("div", { className: "score-val", text: fmtScore(value, 4) }),
          el("div", { className: "score-tip", text: `等级：${tier.label}` }),
        ]),
      );
    });
    wrap.appendChild(grid);

    if (includeRaw) {
      const evaluation = item?.unsupervised_evaluation;
      const raw = {
        id: item?.id || "",
        group_id: item?.group_id || item?.source || "",
        original_filename: item?.original_filename || "",
        question: item?.question || "",
        answer: item?.answer || "",
        unsupervised_evaluation:
          evaluation && typeof evaluation === "object"
            ? {
                method: evaluation.method,
                scores: filterUnsupervisedScores(evaluation.scores),
              }
            : null,
      };
      wrap.appendChild(
        el("details", { className: "explain-tech" }, [
          el("summary", { text: "调试：原始 JSON（节选）" }),
          el("pre", { text: JSON.stringify(raw, null, 2) }),
        ]),
      );
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

  function initAppShell() {
    const header = document.querySelector("body > header");
    const main = document.querySelector("main");
    if (!header || !main || document.querySelector(".app-workbar")) return;

    document.body.classList.add("app-shell", "qa-workbench-redesign");
    const titleText = String(header.querySelector("h1")?.textContent || document.title || "QA Flow").trim();
    const activeNav = header.querySelector(".nav a.active");
    const currentSection = String(activeNav?.textContent || "").trim();
    const themeToggle = document.getElementById("themeToggle");

    const workbar = document.createElement("div");
    workbar.className = "app-workbar";

    const titleWrap = document.createElement("div");
    titleWrap.className = "app-workbar-title";
    const section = document.createElement("div");
    section.className = "app-workbar-section";
    section.textContent = currentSection || "QA Flow";
    const title = document.createElement("h2");
    title.textContent = titleText;
    titleWrap.append(section, title);

    const actions = document.createElement("div");
    actions.className = "app-workbar-actions";
    const status = document.createElement("span");
    status.className = "app-runtime-pill";
    status.textContent = "Docker runtime";
    actions.appendChild(status);
    if (themeToggle) actions.appendChild(themeToggle);

    workbar.append(titleWrap, actions);
    main.insertAdjacentElement("beforebegin", workbar);
  }

  function initAdminFilterModal() {
    const form = document.getElementById("filterForm");
    const section = form ? form.closest("section") : null;
    if (!form || !section || document.getElementById("adminFiltersModal")) return;

    const launch = document.createElement("div");
    launch.className = "filter-launch-bar";
    const copy = document.createElement("div");
    copy.className = "filter-launch-copy";
    copy.textContent = "筛选条件已收进 Filters，列表和语义检索共用同一组条件。";
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.id = "btnOpenAdminFilters";
    openBtn.textContent = "筛选条件";
    launch.append(copy, openBtn);

    const title = section.querySelector("h2");
    if (title) title.insertAdjacentElement("afterend", launch);
    else section.prepend(launch);

    const overlay = document.createElement("div");
    overlay.id = "adminFiltersOverlay";
    overlay.className = "drawer-overlay";
    overlay.hidden = true;

    const modal = document.createElement("aside");
    modal.id = "adminFiltersModal";
    modal.className = "settings-modal";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", "adminFiltersTitle");
    modal.setAttribute("aria-hidden", "true");
    modal.hidden = true;

    const header = document.createElement("div");
    header.className = "settings-modal-header";
    const headCopy = document.createElement("div");
    const modalTitle = document.createElement("h2");
    modalTitle.id = "adminFiltersTitle";
    modalTitle.className = "settings-modal-title";
    modalTitle.textContent = "筛选条件";
    const desc = document.createElement("p");
    desc.className = "settings-modal-desc";
    desc.textContent = "按任务、文件、题型、分数、上架状态和语义检索条件筛选 QA 数据。";
    headCopy.append(modalTitle, desc);
    const closeTop = document.createElement("button");
    closeTop.type = "button";
    closeTop.className = "icon-btn settings-modal-close";
    closeTop.setAttribute("aria-label", "关闭筛选");
    closeTop.innerHTML = '<span aria-hidden="true">×</span>';
    header.append(headCopy, closeTop);

    const body = document.createElement("div");
    body.className = "settings-modal-body";
    body.appendChild(form);

    const footer = document.createElement("div");
    footer.className = "settings-modal-footer";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "secondary";
    close.textContent = "取消";
    const apply = document.createElement("button");
    apply.type = "button";
    apply.textContent = "应用";
    footer.append(close, apply);

    modal.append(header, body, footer);
    document.body.append(overlay, modal);

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
    const closeModal = () => {
      overlay.classList.remove("is-open");
      modal.classList.remove("is-open");
      modal.setAttribute("aria-hidden", "true");
      document.body.classList.remove("drawer-open");
      window.setTimeout(() => {
        overlay.hidden = true;
        modal.hidden = true;
      }, 180);
    };

    openBtn.addEventListener("click", open);
    closeTop.addEventListener("click", closeModal);
    close.addEventListener("click", closeModal);
    overlay.addEventListener("click", closeModal);
    apply.addEventListener("click", () => {
      const activeButton = document.querySelector('[data-query-mode-panel]:not([hidden]) button[id^="btnRun"]');
      if (activeButton instanceof HTMLButtonElement) activeButton.click();
      closeModal();
    });
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

  function fileInputSummary(input) {
    const files = input && input.files ? Array.from(input.files) : [];
    if (!files.length) return "未选择文件";
    if (files.length === 1) return files[0].name || "已选择 1 个文件";
    return `已选择 ${files.length} 个文件`;
  }

  function enhanceFileInputs(root = document) {
    const scope = root && root.querySelectorAll ? root : document;
    Array.from(scope.querySelectorAll('input[type="file"]') || []).forEach((input) => {
      if (!input || input.dataset.fileEnhanced === "1") return;
      input.dataset.fileEnhanced = "1";
      input.classList.add("file-input-native");
      const picker = document.createElement("span");
      picker.className = "file-picker-ui";
      picker.innerHTML = [
        '<span class="file-picker-button">选择文件</span>',
        '<span class="file-picker-name">未选择文件</span>',
      ].join("");
      input.insertAdjacentElement("afterend", picker);
      const name = picker.querySelector(".file-picker-name");
      const sync = () => {
        if (name) name.textContent = fileInputSummary(input);
      };
      picker.addEventListener("click", (event) => {
        event.preventDefault();
        input.click();
      });
      input.addEventListener("change", sync);
      sync();
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
  window.apiuseUi.inferApiBaseUrl = inferApiBaseUrl;
  window.apiuseUi.normalizeApiBaseUrl = normalizeApiBaseUrl;
  window.apiuseUi.getApiBaseUrl = getApiBaseUrl;
  window.apiuseUi.initApiBaseUrl = initApiBaseUrl;
  window.apiuseUi.createScoreGroup = createScoreGroup;
  window.apiuseUi.filterUnsupervisedScores = filterUnsupervisedScores;
  window.apiuseUi.renderUnsupervisedExplain = renderUnsupervisedExplain;
  window.apiuseUi.enhanceTableWrapScroll = enhanceTableWrapScroll;
  window.apiuseUi.enhanceFileInputs = enhanceFileInputs;

  ready(() => {
    initAppShell();
    initAdminFilterModal();
    initTheme();
    initFormAttrs();
    enhanceFileInputs();
    enhanceTableWrapScroll();
  });
})();
