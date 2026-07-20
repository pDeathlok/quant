const state = {
  payload: null,
  activePage: window.location.hash === "#similar" ? "similar" : window.location.hash === "#chan" ? "chan" : window.location.hash === "#long" ? "long" : window.location.hash === "#byd" ? "byd" : window.location.hash === "#cb-allotment" ? "cbAllotment" : window.location.hash === "#cb" ? "cb" : "short",
  selectedStrategies: new Set(),
  selectedSymbol: null,
  query: "",
  signalDate: "",
  calendar: null,
  calendarOpen: false,
  calendarMonth: "",
  refreshPollId: null,
  latestRefreshStatus: null,
  longPayload: null,
  longVariant: "tea",
  longLoading: false,
  chanPayload: null,
  chanLoading: false,
  chanSelectedSymbol: null,
  bydPayload: null,
  bydLoading: false,
  bydLastNoticeKey: "",
  cbPayload: null,
  cbLoading: false,
  selectedCbStrategy: null,
  selectedCbCode: null,
  cbAllotmentPayload: null,
  cbAllotmentLoading: false,
  cbAllotmentError: "",
  cbAllotmentSort: null,
  cbAllotmentStatusFilters: new Set(),
  similarPayload: null,
  similarLoading: false,
  similarError: "",
  similarSelectedSymbol: null,
  loading: false,
};

const API_BASE = "/api";
const REFRESH_STATUS_STORAGE_KEY = "quant.selector.latestRefreshStatus";
const BYD_HOLDING_STORAGE_KEY = "quant.byd.holding.v1";
const WORKSPACE_TAB_ORDER_STORAGE_KEY = "quant.workspaceTabOrder.v1";
const BYD_HOLDING_INPUT_IDS = [
  "bydSharesInput",
  "bydCostInput",
];
const REFRESH_SCOPE_LABELS = {
  all: "全部",
  short: "短线",
  chan: "缠论",
  long: "长线",
  cb: "可转债",
  cbAllotment: "配债股",
  byd: "BYD",
  similar: "自选池",
};
const REFRESH_BUTTON_LABELS = {
  refreshAllButton: "更新全部",
  refreshLatestButton: "更新本页",
  longRefreshLatestButton: "更新本页",
  chanRefreshLatestButton: "更新本页",
  cbRefreshLatestButton: "更新本页",
  cbAllotmentRefreshLatestButton: "更新本页",
  bydRefreshLatestButton: "更新本页",
  similarRefreshLatestButton: "更新本页",
};

const WORKSPACE_TABS = [
  { key: "short", label: "短线策略", description: "每日选股 / 交易计划", panelId: "shortPage" },
  { key: "chan", label: "缠论策略", description: "三买模型 / T+1 计划", panelId: "chanPage" },
  { key: "long", label: "长线策略", description: "组合候选 / 仓位择时", panelId: "longPage" },
  { key: "cb", label: "可转债策略", description: "低位候选 / 分批计划", panelId: "cbPage" },
  { key: "cbAllotment", label: "配债股", description: "发行流程 / 关键日期", panelId: "cbAllotmentPage" },
  { key: "byd", label: "BYD 做T", description: "盘前计划 / 正T优先", panelId: "bydPage" },
  { key: "similar", label: "自选池", description: "相似走势 / 策略联动", panelId: "similarPage" },
];
let focusWorkspaceTabAfterRender = false;
let workspaceTabDragKey = "";
let suppressWorkspaceTabClick = false;

function normalizeWorkspaceTabOrder(value) {
  const knownKeys = WORKSPACE_TABS.map((tab) => tab.key);
  const supplied = Array.isArray(value) ? value.filter((key) => knownKeys.includes(key)) : [];
  return [...new Set([...supplied, ...knownKeys])];
}

function loadWorkspaceTabOrder() {
  try {
    return normalizeWorkspaceTabOrder(JSON.parse(localStorage.getItem(WORKSPACE_TAB_ORDER_STORAGE_KEY) || "[]"));
  } catch (_error) {
    return normalizeWorkspaceTabOrder([]);
  }
}

let workspaceTabOrder = loadWorkspaceTabOrder();

function orderedWorkspaceTabs() {
  const tabByKey = new Map(WORKSPACE_TABS.map((tab) => [tab.key, tab]));
  return workspaceTabOrder.map((key) => tabByKey.get(key)).filter(Boolean);
}

function syncWorkspaceTabDomOrder(focusKey = "") {
  document.querySelectorAll(".workspace-tabs").forEach((nav) => {
    workspaceTabOrder.forEach((key) => {
      const button = nav.querySelector(`.page-tab[data-page="${key}"]`);
      if (button) nav.appendChild(button);
    });
  });
  if (focusKey) {
    document.querySelector(`.page-view.active .page-tab[data-page="${focusKey}"]`)?.focus({ preventScroll: true });
  }
}

function saveWorkspaceTabOrder(order, focusKey = "") {
  workspaceTabOrder = normalizeWorkspaceTabOrder(order);
  try {
    localStorage.setItem(WORKSPACE_TAB_ORDER_STORAGE_KEY, JSON.stringify(workspaceTabOrder));
  } catch (_error) {
    // The current session still keeps the chosen order when storage is unavailable.
  }
  syncWorkspaceTabDomOrder(focusKey);
}

function reorderWorkspaceTab(sourceKey, targetKey, placeAfter = false) {
  if (!sourceKey || !targetKey || sourceKey === targetKey) return false;
  const nextOrder = workspaceTabOrder.filter((key) => key !== sourceKey);
  const targetIndex = nextOrder.indexOf(targetKey);
  if (targetIndex < 0) return false;
  nextOrder.splice(targetIndex + (placeAfter ? 1 : 0), 0, sourceKey);
  if (nextOrder.join("|") === workspaceTabOrder.join("|")) return false;
  saveWorkspaceTabOrder(nextOrder, sourceKey);
  return true;
}

function moveWorkspaceTabByOffset(key, offset) {
  const currentIndex = workspaceTabOrder.indexOf(key);
  const nextIndex = Math.max(0, Math.min(workspaceTabOrder.length - 1, currentIndex + offset));
  if (currentIndex < 0 || nextIndex === currentIndex) return false;
  const nextOrder = [...workspaceTabOrder];
  nextOrder.splice(currentIndex, 1);
  nextOrder.splice(nextIndex, 0, key);
  saveWorkspaceTabOrder(nextOrder, key);
  return true;
}

function clearWorkspaceTabDropTargets() {
  document.querySelectorAll(".page-tab.tab-drop-before, .page-tab.tab-drop-after").forEach((button) => {
    button.classList.remove("tab-drop-before", "tab-drop-after");
  });
}

function setBydHoldingSaveStatus(message, tone = "") {
  const status = document.querySelector("#bydHoldingSaveStatus");
  if (!status) return;
  status.textContent = message;
  status.classList.toggle("saved", tone === "saved");
  status.classList.toggle("error", tone === "error");
}

function saveBydHoldingInputs() {
  const values = Object.fromEntries(BYD_HOLDING_INPUT_IDS.map((id) => [
    id,
    document.querySelector(`#${id}`)?.value ?? "",
  ]));
  const shares = Number(values.bydSharesInput);
  const cost = Number(values.bydCostInput);
  if (!Number.isFinite(shares) || shares < 0 || !Number.isFinite(cost) || cost <= 0) {
    setBydHoldingSaveStatus("持仓股数或成本价无效，未保存", "error");
    return false;
  }
  try {
    localStorage.setItem(BYD_HOLDING_STORAGE_KEY, JSON.stringify({
      version: 3,
      savedAt: new Date().toISOString(),
      values,
    }));
    setBydHoldingSaveStatus(`持仓已永久保存 · ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`, "saved");
    return true;
  } catch {
    setBydHoldingSaveStatus("浏览器禁止本地存储，持仓未保存", "error");
    return false;
  }
}

function restoreBydHoldingInputs() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(BYD_HOLDING_STORAGE_KEY) || "null");
  } catch {
    localStorage.removeItem(BYD_HOLDING_STORAGE_KEY);
    setBydHoldingSaveStatus("已忽略损坏的持仓记录，请重新填写", "error");
    return;
  }
  if (!saved?.values) return;
  const values = saved.values;
  BYD_HOLDING_INPUT_IDS.forEach((id) => {
    const input = document.querySelector(`#${id}`);
    if (input && values[id] !== undefined) input.value = values[id];
  });
  setBydHoldingSaveStatus("已恢复永久保存的持仓和成本", "saved");
}

const longStrategies = [
  {
    key: "tea",
    name: "茶大长线趋势网格",
    version: "core14_soft_plus",
    tag: "当前主推",
    summaryLabel: "主推荐",
    description: "独立于旧长线策略，使用茶大体系的年线趋势、核心底仓、行业上限、软刹车和日线做T。",
    note: "当前主推：不使用成长卫星仓，保留约 14 个核心候选，适合长线每日观察。",
    metrics: { annual: "7.14%", drawdown: "-32.02%", sharpe: "0.608", weight: "49.27%" },
    yearly: [
      { year: "2013", ret: "1.95%", dd: "-7.68%", weight: "30.0%" },
      { year: "2014", ret: "40.87%", dd: "-4.62%", weight: "44.0%" },
      { year: "2015", ret: "1.47%", dd: "-27.96%", weight: "46.2%" },
      { year: "2016", ret: "-6.16%", dd: "-9.83%", weight: "51.7%" },
      { year: "2017", ret: "30.02%", dd: "-6.44%", weight: "76.6%" },
      { year: "2018", ret: "-7.14%", dd: "-13.13%", weight: "30.2%" },
      { year: "2019", ret: "9.02%", dd: "-16.13%", weight: "61.5%" },
      { year: "2020", ret: "28.01%", dd: "-13.31%", weight: "62.1%" },
      { year: "2021", ret: "6.83%", dd: "-14.16%", weight: "39.9%" },
      { year: "2022", ret: "-12.54%", dd: "-14.53%", weight: "26.6%" },
      { year: "2023", ret: "11.45%", dd: "-7.23%", weight: "42.8%" },
      { year: "2024", ret: "3.56%", dd: "-10.20%", weight: "55.5%" },
      { year: "2025", ret: "4.99%", dd: "-7.77%", weight: "63.9%" },
      { year: "2026 YTD", ret: "-3.21%", dd: "-12.69%", weight: "69.8%" },
    ],
  },
  {
    key: "tea_safe",
    name: "茶大长线稳健网格",
    version: "core14_soft_spread",
    tag: "保守备选",
    summaryLabel: "稳健备选",
    description: "同样使用茶大独立逻辑，但 risk_on/neutral 仓位更保守，回撤略低。",
    note: "适合更看重回撤控制的账户；收益略低于主推版本。",
    metrics: { annual: "6.67%", drawdown: "-30.10%", sharpe: "0.607", weight: "45.84%" },
    yearly: [
      { year: "2013", ret: "1.84%", dd: "-7.22%", weight: "28.1%" },
      { year: "2014", ret: "37.53%", dd: "-4.27%", weight: "40.9%" },
      { year: "2015", ret: "1.53%", dd: "-26.23%", weight: "43.0%" },
      { year: "2016", ret: "-5.73%", dd: "-9.21%", weight: "48.0%" },
      { year: "2017", ret: "27.74%", dd: "-5.99%", weight: "71.2%" },
      { year: "2018", ret: "-6.62%", dd: "-12.23%", weight: "28.1%" },
      { year: "2019", ret: "8.48%", dd: "-15.04%", weight: "57.3%" },
      { year: "2020", ret: "25.89%", dd: "-12.40%", weight: "57.7%" },
      { year: "2021", ret: "6.27%", dd: "-13.24%", weight: "37.1%" },
      { year: "2022", ret: "-11.68%", dd: "-13.58%", weight: "24.6%" },
      { year: "2023", ret: "10.61%", dd: "-6.77%", weight: "39.8%" },
      { year: "2024", ret: "3.33%", dd: "-9.52%", weight: "51.7%" },
      { year: "2025", ret: "4.63%", dd: "-7.28%", weight: "59.4%" },
      { year: "2026 YTD", ret: "-2.91%", dd: "-11.83%", weight: "65.0%" },
    ],
  },
  {
    key: "v44",
    name: "旧防守中性长期组合",
    version: "v44",
    tag: "旧策略对照",
    summaryLabel: "旧策略",
    description: "旧长线策略分支，保留作为对照。",
    note: "后续茶大策略以 tea/tea_safe 为主。",
    metrics: { annual: "5.62%", drawdown: "-21.80%", sharpe: "0.616", weight: "42.11%" },
    yearly: [
      { year: 2013, ret: "-3.13%", dd: "-4.33%", weight: "12.4%" },
      { year: 2014, ret: "40.84%", dd: "-4.17%", weight: "31.7%" },
      { year: 2015, ret: "16.72%", dd: "-21.75%", weight: "47.3%" },
      { year: 2016, ret: "-0.97%", dd: "-6.02%", weight: "46.9%" },
      { year: 2017, ret: "14.39%", dd: "-4.26%", weight: "64.5%" },
      { year: 2018, ret: "-7.79%", dd: "-14.24%", weight: "38.1%" },
      { year: 2019, ret: "2.13%", dd: "-11.15%", weight: "54.3%" },
      { year: 2020, ret: "5.83%", dd: "-8.90%", weight: "48.9%" },
      { year: 2021, ret: "12.74%", dd: "-7.59%", weight: "46.8%" },
      { year: 2022, ret: "-6.52%", dd: "-8.43%", weight: "26.5%" },
      { year: 2023, ret: "3.58%", dd: "-5.67%", weight: "29.5%" },
      { year: 2024, ret: "10.02%", dd: "-11.71%", weight: "41.3%" },
      { year: 2025, ret: "0.67%", dd: "-7.24%", weight: "52.4%" },
      { year: "2026 YTD", ret: "-2.37%", dd: "-7.68%", weight: "55.9%" },
    ],
  },
  {
    key: "v43",
    name: "核心质量长期组合",
    version: "v43",
    tag: "收益候选",
    summaryLabel: "参考策略",
    description: "完全去掉自动成长仓，只在市场趋势框架下配置核心质量股票，牛市弹性更强。",
    note: "年化更高，但中性/弱市暴露高于 v44；适合风险承受能力稍高的账户参考。",
    metrics: { annual: "6.12%", drawdown: "-23.97%", sharpe: "0.607", weight: "48.76%" },
    yearly: [
      { year: 2013, ret: "-3.13%", dd: "-4.33%", weight: "12.4%" },
      { year: 2014, ret: "40.84%", dd: "-4.17%", weight: "31.7%" },
      { year: 2015, ret: "15.67%", dd: "-23.92%", weight: "50.5%" },
      { year: 2016, ret: "-0.85%", dd: "-6.02%", weight: "47.8%" },
      { year: 2017, ret: "20.11%", dd: "-4.44%", weight: "70.3%" },
      { year: 2018, ret: "-10.62%", dd: "-17.64%", weight: "47.3%" },
      { year: 2019, ret: "4.87%", dd: "-11.30%", weight: "60.5%" },
      { year: 2020, ret: "4.68%", dd: "-9.42%", weight: "52.3%" },
      { year: 2021, ret: "13.69%", dd: "-10.08%", weight: "61.3%" },
      { year: 2022, ret: "-8.96%", dd: "-11.03%", weight: "35.8%" },
      { year: 2023, ret: "4.10%", dd: "-7.61%", weight: "42.8%" },
      { year: 2024, ret: "11.77%", dd: "-10.48%", weight: "50.9%" },
      { year: 2025, ret: "3.35%", dd: "-8.22%", weight: "61.9%" },
      { year: "2026 YTD", ret: "-1.34%", dd: "-8.80%", weight: "65.7%" },
    ],
  },
  {
    key: "v34",
    name: "PIT 成长防守长期组合",
    version: "v34",
    tag: "成长对照",
    summaryLabel: "成长对照",
    description: "保留少量成长仓和 PIT 股票池约束，用于观察成长股进入机制对组合的影响。",
    note: "前复权下综合表现不如 v44/v43，但适合作为成长仓策略继续迭代的对照组。",
    metrics: { annual: "5.78%", drawdown: "-27.19%", sharpe: "0.554", weight: "47.86%" },
    yearly: [
      { year: 2013, ret: "-0.79%", dd: "-2.76%", weight: "13.4%" },
      { year: 2014, ret: "36.47%", dd: "-5.41%", weight: "32.9%" },
      { year: 2015, ret: "2.99%", dd: "-27.11%", weight: "43.4%" },
      { year: 2016, ret: "-1.84%", dd: "-4.71%", weight: "34.5%" },
      { year: 2017, ret: "9.24%", dd: "-4.33%", weight: "74.7%" },
      { year: 2018, ret: "-7.46%", dd: "-15.60%", weight: "39.4%" },
      { year: 2019, ret: "11.88%", dd: "-9.58%", weight: "62.9%" },
      { year: 2020, ret: "14.10%", dd: "-10.03%", weight: "58.1%" },
      { year: 2021, ret: "12.15%", dd: "-13.71%", weight: "57.4%" },
      { year: 2022, ret: "-8.96%", dd: "-11.03%", weight: "35.8%" },
      { year: 2023, ret: "4.29%", dd: "-7.15%", weight: "43.5%" },
      { year: 2024, ret: "13.79%", dd: "-10.48%", weight: "52.2%" },
      { year: 2025, ret: "4.44%", dd: "-8.36%", weight: "64.2%" },
      { year: "2026 YTD", ret: "-2.43%", dd: "-9.31%", weight: "68.6%" },
    ],
  },
];

const fmtPct = (value, digits = 2) => {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return `${Number(value).toFixed(digits)}%`;
};
const fmtRate = (value, digits = 1) => {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(digits)}%`;
};
const fmtPrice = (value) => {
  if (value == null || value === "" || Number.isNaN(Number(value))) return "-";
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric.toFixed(2) : "-";
};
const t1ReturnBucketOrder = (bucket) => {
  const text = String(bucket || "");
  const knownOrder = {
    "大跌<=-3%": 0,
    "小跌-3~-1%": 1,
    "震荡-1~1%": 2,
    "小涨1~3%": 3,
    "大涨>=3%": 4,
  };
  if (Object.prototype.hasOwnProperty.call(knownOrder, text)) return knownOrder[text];
  const numbers = text.match(/-?\d+(?:\.\d+)?/g)?.map(Number).filter(Number.isFinite) || [];
  if (!numbers.length) return Number.POSITIVE_INFINITY;
  return Math.min(...numbers);
};
const t1VolumeBucketOrder = (bucket) => {
  const text = String(bucket || "");
  const knownOrder = {
    "缩量<0.8": 0,
    "平量0.8~1.3": 1,
    "放量1.3~2": 2,
    "巨量>2": 3,
  };
  if (Object.prototype.hasOwnProperty.call(knownOrder, text)) return knownOrder[text];
  return text;
};
const sortT1ScenarioRows = (rows) => [...(rows || [])].sort((a, b) => {
  const returnDiff = t1ReturnBucketOrder(a.t1_return_bucket) - t1ReturnBucketOrder(b.t1_return_bucket);
  if (returnDiff !== 0) return returnDiff;
  const volumeA = t1VolumeBucketOrder(a.t1_volume_bucket);
  const volumeB = t1VolumeBucketOrder(b.t1_volume_bucket);
  if (typeof volumeA === "number" && typeof volumeB === "number") return volumeA - volumeB;
  return String(volumeA).localeCompare(String(volumeB), "zh-Hans-CN");
});
const fmtWeight = (value) => {
  if (value == null || value === "" || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
};
const fmtMoney = (value) => {
  if (value == null || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
};
const fmtRange = (range) => {
  if (!range) return "-";
  if (range.label) return range.label;
  const low = fmtPrice(range.low);
  const high = fmtPrice(range.high);
  return low === high ? low : `${low}-${high}`;
};
const hashPage = () => window.location.hash === "#similar" ? "similar" : window.location.hash === "#chan" ? "chan" : window.location.hash === "#long" ? "long" : window.location.hash === "#byd" ? "byd" : window.location.hash === "#cb-allotment" ? "cbAllotment" : window.location.hash === "#cb" ? "cb" : "short";
const formatBuyPlanText = (text) => {
  if (!text) return "";
  let normalized = String(text)
    .replaceAll("开盘执行条件：", "T+1 开盘执行条件：")
    .replaceAll("符合信号后 T+1 开盘执行，不满足则空仓。", "不满足 T+1 开盘条件则空仓观察。");
  normalized = normalized.replace(/T\+1 T\+1 开盘执行条件：/g, "T+1 开盘执行条件：");
  return normalized.replace(/T\+1 开盘执行条件：([^。；]*信号日收盘位置[^。；]*)([。；])/g, (_match, conditions, suffix) => {
    const parts = conditions.split(/[，,]/).map((part) => part.trim()).filter(Boolean);
    const signalParts = parts.filter((part) => part.includes("信号日") || part.includes("收盘位置"));
    const openParts = parts.filter((part) => !signalParts.includes(part));
    const signalText = signalParts.length ? `信号确认条件（T日收盘后已确认）：${signalParts.join("；")}；` : "";
    const openText = `T+1 开盘执行条件：${openParts.join("；") || "T+1 开盘观察"}`;
    return `${signalText}${openText}${suffix}`;
  });
};
const pageHash = (page) => page === "similar" ? "#similar" : page === "chan" ? "#chan" : page === "long" ? "#long" : page === "byd" ? "#byd" : page === "cbAllotment" ? "#cb-allotment" : page === "cb" ? "#cb" : "#short";
const currentLongStrategy = () => longStrategies.find((item) => item.key === state.longVariant) || longStrategies[0];

function ensureWorkspaceTabs() {
  document.querySelectorAll(".workspace-tabs").forEach((nav, navIndex) => {
    if (nav.dataset.workspaceTabsReady === "true") return;
    nav.dataset.workspaceTabsReady = "true";
    nav.setAttribute("role", "tablist");
    nav.setAttribute("aria-orientation", "horizontal");
    nav.innerHTML = orderedWorkspaceTabs().map((tab) => `
      <button
        id="workspace-tab-${navIndex}-${tab.key}"
        class="page-tab"
        type="button"
        role="tab"
        data-page="${tab.key}"
        aria-controls="${tab.panelId}"
        aria-selected="false"
        aria-keyshortcuts="Alt+ArrowLeft Alt+ArrowRight"
        draggable="true"
        tabindex="-1"
        title="按住拖动排序；Alt + 左右方向键也可调整"
      >
        <strong>${tab.label}</strong>
        <span>${tab.description}</span>
      </button>
    `).join("");
  });
  if (!document.querySelector("#chanDateSlot")) {
    const actions = document.querySelector("#chanPage .toolbar-actions");
    if (actions) {
      const slot = document.createElement("div");
      slot.id = "chanDateSlot";
      actions.insertBefore(slot, actions.firstElementChild || null);
    }
  }
}

function revealActiveWorkspaceTab() {
  const activePage = document.querySelector(".page-view.active");
  const nav = activePage?.querySelector(".workspace-tabs");
  const activeTab = nav?.querySelector(".page-tab.active");
  if (!nav || !activeTab || nav.scrollWidth <= nav.clientWidth) return;
  const navRect = nav.getBoundingClientRect();
  const tabRect = activeTab.getBoundingClientRect();
  const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  if (tabRect.left < navRect.left) {
    nav.scrollBy({ left: tabRect.left - navRect.left - 8, behavior });
  } else if (tabRect.right > navRect.right) {
    nav.scrollBy({ left: tabRect.right - navRect.right + 8, behavior });
  }
}

function loadActivePageData() {
  if (state.activePage === "long" && !state.longPayload && !state.longLoading) {
    loadLongStockPool().catch(showError);
  } else if (state.activePage === "chan" && !state.chanPayload && !state.chanLoading) {
    loadChanModelStrategy().catch(showError);
  } else if (state.activePage === "cb" && !state.cbPayload && !state.cbLoading) {
    loadConvertibleBondPlan().catch(showError);
  } else if (state.activePage === "cbAllotment" && !state.cbAllotmentPayload && !state.cbAllotmentLoading) {
    loadConvertibleBondAllotments().catch(showError);
  } else if (state.activePage === "byd" && !state.bydPayload && !state.bydLoading) {
    loadBydMinuteStrategy().catch(showError);
  } else if (state.activePage === "similar" && !state.similarPayload && !state.similarLoading) {
    loadSimilarPatterns().catch(showError);
  } else if (state.activePage === "short" && !state.payload && !state.loading) {
    loadSelector({ latest: true }).catch(showError);
  }
}

function longPositionPlan(item) {
  const target = Number(item.target_weight || 0);
  const first = Number(item.first_tranche_weight || 0);
  if (target > 0) {
    return `
      <strong>目标 ${fmtWeight(target)}</strong>
      <em>首批 ${fmtWeight(first)}</em>
      <em>建仓 <= ${fmtPrice(item.price_levels?.entry_target_price)}</em>
    `;
  }
  if (item.state === "BUILDING") {
    return `
      <strong>待入池</strong>
      <em>当前目标 0%</em>
      <em>观察建仓 <= ${fmtPrice(item.price_levels?.entry_target_price)}</em>
    `;
  }
  if (item.state === "REDUCE") {
    return `
      <strong>降仓观察</strong>
      <em>未纳入本期目标仓</em>
      <em>跌破 ${fmtPrice(item.price_levels?.reduce_ma60_price)} 降风险</em>
    `;
  }
  if (item.state === "EXIT") {
    return `
      <strong>不建仓</strong>
      <em>目标 0%</em>
      <em>清仓线 ${fmtPrice(item.price_levels?.exit_ma120_price)}</em>
    `;
  }
  return `
    <strong>观察</strong>
    <em>等待席位/价格确认</em>
    <em>建仓 <= ${fmtPrice(item.price_levels?.entry_target_price)}</em>
  `;
}

function analystCoverageText(item) {
  const reports = Number(item.analyst_report_count_180d || 0);
  const orgs = Number(item.analyst_org_count_180d || 0);
  const forwardYears = Number(item.analyst_forward_years_180d || 0);
  if (!reports) {
    return { main: "近180日无结构化预测", sub: "使用财务/估值/趋势因子" };
  }
  const orgText = orgs <= 1 ? "一致预期" : `${orgs}家机构`;
  const yearText = forwardYears ? `未来${forwardYears}年` : "无前瞻年度";
  return { main: `${reports}条 · ${orgText}`, sub: `${yearText} · 成长 ${Number(item.analyst_forward_growth_score || 0).toFixed(1)}` };
}

async function fetchJson(path, options = {}) {
  const { timeoutMs = 0, ...fetchOptions } = options;
  const controller = timeoutMs > 0 && !fetchOptions.signal ? new AbortController() : null;
  const timeoutId = controller ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      cache: "no-store",
      ...fetchOptions,
      ...(controller ? { signal: controller.signal } : {}),
      headers: {
        ...(fetchOptions.body ? { "Content-Type": "application/json" } : {}),
        ...(fetchOptions.headers || {}),
      },
    });
  } catch (error) {
    if (controller?.signal.aborted) {
      throw new Error(`${path} 加载超时，请稍后重试`);
    }
    throw error;
  } finally {
    if (timeoutId !== null) window.clearTimeout(timeoutId);
  }
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch {
      detail = "";
    }
    throw new Error(`${path} 加载失败: ${response.status}${detail ? ` · ${detail}` : ""}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function selectedStrategyParam() {
  return Array.from(state.selectedStrategies).join(",");
}

function calendarDay(date) {
  return (state.calendar?.days || []).find((item) => item.date === date) || null;
}

function applySelectableSignalDate(date) {
  if (!date) return "";
  const day = calendarDay(date);
  if (day?.disabled) {
    const fallback = day.effective_signal_date || state.calendar?.latest_signal_date || "";
    setRefreshMessage(`${date} 为休市日，已切换到最近可用交易日 ${fallback}`);
    return fallback;
  }
  return date;
}

function setRefreshMessage(message) {
  document.querySelectorAll(".refresh-status").forEach((el) => {
    el.textContent = message || "";
  });
}

function workspaceRequestOptions(options = {}) {
  return { timeoutMs: options.refresh ? 120000 : 15000 };
}

function renderShortPage() {
  renderHeader();
  renderStrategyFilters();
  renderStockRows();
  renderStockDetail();
  renderNotes();
  renderCalendar();
  renderDateStatus();
}

async function loadSelector(options = {}) {
  state.loading = true;
  renderShortPage();
  if (options.latest && state.calendar?.latest_signal_date) {
    state.signalDate = state.calendar.latest_signal_date;
  }
  state.signalDate = applySelectableSignalDate(state.signalDate);
  const requestedSignalDate = state.signalDate;
  const query = new URLSearchParams();
  const params = selectedStrategyParam();
  if (params) query.set("strategies", params);
  if (!params) query.set("include_extended", "true");
  if (state.signalDate) query.set("signal_date", state.signalDate);
  if (options.refresh) query.set("refresh", "true");
  const suffix = query.toString();
  const path = suffix ? `/selector/stocks?${suffix}` : "/selector/stocks";
  try {
    state.payload = await fetchJson(path, workspaceRequestOptions(options));
    state.signalDate = requestedSignalDate || state.payload.signal_date || state.signalDate;
    const dateInput = document.querySelector("#signalDateInput");
    if (dateInput) {
      dateInput.value = state.signalDate || state.payload.signal_date || "";
    }
    if (!state.selectedSymbol && state.payload.stocks.length) {
      state.selectedSymbol = state.payload.stocks[0].symbol;
    }
    if (state.selectedSymbol && !state.payload.stocks.some((item) => item.symbol === state.selectedSymbol)) {
      state.selectedSymbol = state.payload.stocks[0]?.symbol || null;
    }
  } catch (error) {
    showError(error);
    throw error;
  } finally {
    state.loading = false;
    renderShortPage();
  }
}

async function loadSimilarPatterns(options = {}) {
  state.similarLoading = true;
  state.similarError = "";
  renderSimilarPatternsPage();
  try {
    const watchlistPayload = await fetchJson("/similar-patterns/watchlist");
    state.similarPayload = {
      ...(state.similarPayload || {}),
      generated_at: state.similarPayload?.generated_at || watchlistPayload.updated_at,
      watchlist: watchlistPayload.stocks || [],
      results: state.similarPayload?.results || [],
      config: state.similarPayload?.config || {},
    };
    renderSimilarPatternsPage();
    const path = options.refresh ? "/similar-patterns/analysis?refresh=true" : "/similar-patterns/analysis";
    const analysisPayload = await fetchJson(path, workspaceRequestOptions(options));
    const latestWatchlist = state.similarPayload?.watchlist || watchlistPayload.stocks || analysisPayload.watchlist || [];
    state.similarPayload = {
      ...analysisPayload,
      watchlist: latestWatchlist,
    };
    const results = state.similarPayload.results || [];
    if (!state.similarSelectedSymbol && results.length) {
      state.similarSelectedSymbol = results[0].target?.symbol || null;
    }
    if (state.similarSelectedSymbol && !results.some((item) => item.target?.symbol === state.similarSelectedSymbol)) {
      state.similarSelectedSymbol = results[0]?.target?.symbol || null;
    }
  } catch (error) {
    state.similarError = error.message || "自选池分析加载失败";
    showError(error);
    throw error;
  } finally {
    state.similarLoading = false;
    renderSimilarPatternsPage();
  }
}

async function addSimilarWatchSymbol(symbol, options = {}) {
  await fetchJson("/similar-patterns/watchlist", {
    method: "POST",
    body: JSON.stringify({ symbol, note: options.note || "" }),
  });
  state.similarPayload = null;
  if (options.refresh !== false) await loadSimilarPatterns({ refresh: true });
}

async function removeSimilarWatchSymbol(symbol) {
  await fetchJson(`/similar-patterns/watchlist/${encodeURIComponent(symbol)}`, { method: "DELETE" });
  if (state.similarSelectedSymbol === symbol) state.similarSelectedSymbol = null;
  state.similarPayload = null;
  await loadSimilarPatterns({ refresh: true });
}

async function saveSimilarWatchNote(symbol, content) {
  const payload = await fetchJson(`/similar-patterns/watchlist/${encodeURIComponent(symbol)}/note`, {
    method: "PUT",
    body: JSON.stringify({ content }),
  });
  const saved = (payload.stocks || []).find((item) => item.symbol === symbol);
  const current = (state.similarPayload?.watchlist || []).find((item) => item.symbol === symbol);
  if (saved && current) Object.assign(current, saved);
  renderSimilarPatternsPage();
  return saved;
}

function applySimilarWatchlistPayload(payload) {
  if (!state.similarPayload) return;
  state.similarPayload.watchlist = payload.stocks || [];
  renderSimilarPatternsPage();
}

async function saveSimilarWatchOrder(symbols) {
  const payload = await fetchJson("/similar-patterns/watchlist/order", {
    method: "PUT",
    body: JSON.stringify({ symbols }),
  });
  applySimilarWatchlistPayload(payload);
  return payload;
}

async function setSimilarWatchPinned(symbol, pinned) {
  const payload = await fetchJson(`/similar-patterns/watchlist/${encodeURIComponent(symbol)}/pin`, {
    method: "PUT",
    body: JSON.stringify({ pinned }),
  });
  applySimilarWatchlistPayload(payload);
  return payload;
}

const similarActionClass = (action) => {
  const text = String(action || "");
  if (text.includes("卖出") || text.includes("减仓")) return "similar-action-sell";
  if (text.includes("低吸") || text.includes("持有/可")) return "similar-action-buy";
  return "";
};

function similarForecastLabel(horizon) {
  if (horizon === "next_1d") return "次日";
  if (horizon === "next_1m") return "1个月";
  if (horizon === "next_3m") return "3个月";
  return horizon || "-";
}

function selectedSimilarResult() {
  const results = state.similarPayload?.results || [];
  return results.find((item) => item.target?.symbol === state.similarSelectedSymbol) || results[0] || null;
}

function similarResultForSymbol(symbol) {
  return (state.similarPayload?.results || []).find((item) => item.target?.symbol === symbol) || null;
}

function similarForecastRow(item, horizon) {
  const optimized = (item?.optimized_forecast || []).find((row) => row.horizon === horizon);
  return optimized || (item?.forecast || []).find((row) => row.horizon === horizon) || {};
}

function similarDecision(item, horizon) {
  return (item?.decisions || []).find((row) => row.horizon === horizon) || {};
}

function similarSignalLabel(signal) {
  if (signal === "bullish") return "看涨";
  if (signal === "bearish") return "看跌";
  return "观望";
}

function similarProbabilitySourceLabel(source) {
  const labels = {
    raw_baseline: "原始相似样本",
    event_dedupe: "事件去重",
    nonlinear: "非线性权重",
    regime_industry: "市场/行业状态权重",
    recency: "时间衰减",
    full_weighting: "完整组合权重",
    optimized: "完整组合权重",
    calibrated: "全池滚动校准",
  };
  return labels[source] || source || "统一策略";
}

function watchlistStrategyHits(item) {
  return item?.strategy_hits || [];
}

function watchlistStrategyBadges(item, includeDetail = false) {
  const hits = watchlistStrategyHits(item);
  if (!hits.length) return `<span class="strategy-hit-empty">仅自选观察</span>`;
  return `<div class="strategy-hit-list">${hits.map((hit) => `
    <span class="strategy-hit-badge ${hit.strategy_key || ""}" title="${hit.detail || hit.strategy_label || ""}">
      <strong>${hit.strategy_label || hit.strategy_key || "策略"}</strong>
      ${includeDetail ? `<em>${hit.detail || "已命中"}</em>` : ""}
    </span>
  `).join("")}</div>`;
}

function strategyResonanceLabel(item) {
  const count = watchlistStrategyHits(item).length;
  if (count >= 2) return `${count} 策略共振`;
  if (count === 1) return "策略命中";
  return "观察中";
}

function compactWatchlistDate(value) {
  const digits = String(value || "").replaceAll("-", "").slice(0, 8);
  if (!/^\d{8}$/.test(digits)) return "";
  return `${Number(digits.slice(4, 6))}.${Number(digits.slice(6, 8))}`;
}

function watchlistSourceNote(dateValue, sourceText) {
  const dateLabel = compactWatchlistDate(dateValue);
  return [dateLabel, String(sourceText || "").trim()].filter(Boolean).join(" ");
}

function xueqiuStockUrl(symbol) {
  const normalized = String(symbol || "").trim().toUpperCase();
  const match = normalized.match(/^(\d{6})(?:\.(SH|SZ|BJ))?$/);
  if (!match) return "";
  const code = match[1];
  const market = match[2]
    || (/^[69]/.test(code) ? "SH" : /^[48]/.test(code) ? "BJ" : "SZ");
  return `https://xueqiu.com/S/${market}${code}`;
}

function fallbackSimilarityScore(row, config) {
  const current = Number(row.forecast_weight);
  const ceiling = Number(config?.similarity_score_ceiling);
  const contrast = Number(config?.similarity_score_contrast);
  if (Number.isFinite(current) && Number.isFinite(ceiling) && ceiling > 0 && Number.isFinite(contrast) && contrast > 0) {
    const normalized = Math.max(0, Math.min(1, current / ceiling));
    return (Math.log1p(contrast * normalized) / Math.log1p(contrast) * 100).toFixed(1);
  }
  if (row.similarity_score != null && !Number.isNaN(Number(row.similarity_score))) {
    return Number(row.similarity_score).toFixed(1);
  }
  return "-";
}

function renderSimilarPatternsPage() {
  const payload = state.similarPayload;
  const meta = document.querySelector("#similarMeta");
  const watchCount = document.querySelector("#similarWatchCount");
  const watchlist = document.querySelector("#similarWatchlist");
  const overview = document.querySelector("#similarSelectedOverview");
  const scenarioRows = document.querySelector("#similarScenarioRows");
  const modelList = document.querySelector("#similarSellModel");
  const topRows = document.querySelector("#similarTopRows");
  const detailTitle = document.querySelector("#similarDetailTitle");
  const detailMeta = document.querySelector("#similarDetailMeta");
  const modelMeta = document.querySelector("#similarModelMeta");
  const signalSummary = document.querySelector("#similarSignalSummary");
  if (!meta || !watchlist || !overview || !scenarioRows || !modelList || !topRows) return;

  if (state.similarLoading && !payload) {
    meta.textContent = "正在计算自选池相似走势";
    if (signalSummary) signalSummary.innerHTML = `<article class="similar-summary-card loading"><span>策略共振</span><strong>分析中</strong><em>正在汇总其他策略命中</em></article>`;
    overview.innerHTML = `<article class="panel"><p class="subline">正在扫描历史相似片段...</p></article>`;
    return;
  }
  if (state.similarError && !payload) {
    meta.textContent = "自选池加载失败";
    if (signalSummary) signalSummary.innerHTML = `
      <article class="similar-summary-card error">
        <span>数据状态</span>
        <strong>加载失败</strong>
        <em>${state.similarError}</em>
      </article>
    `;
    watchlist.innerHTML = `<tr><td colspan="8" class="empty-cell">${state.similarError} · <button type="button" data-similar-retry>重新加载</button></td></tr>`;
    overview.innerHTML = "";
    return;
  }
  if (!payload) {
    meta.textContent = "切到自选池后加载";
    if (signalSummary) signalSummary.innerHTML = "";
    watchlist.innerHTML = `<tr><td colspan="8" class="empty-cell">切到自选池后加载</td></tr>`;
    overview.innerHTML = "";
    return;
  }

  const watch = payload.watchlist || [];
  const results = payload.results || [];
  const resonantResults = results.filter((item) => watchlistStrategyHits(item).length > 0);
  const totalStrategyHits = resonantResults.reduce((total, item) => total + watchlistStrategyHits(item).length, 0);
  const bullishCount = results.filter((item) => similarDecision(item, "next_1d").signal === "bullish").length;
  const rankedWatch = [...watch];
  meta.textContent = state.similarError
    ? `分析加载失败，笔记仍可编辑 · 自选 ${watch.length} 只`
    : state.similarLoading
      ? `自选 ${watch.length} 只 · 正在加载分析，笔记可先编辑`
      : `更新于 ${payload.generated_at || "-"} · 全池统一策略 · T+1 观望区 ${payload.config?.signal_bearish_max ?? 45}%～${payload.config?.signal_bullish_min ?? 55}% · 自选 ${watch.length} 只`;
  if (watchCount) watchCount.textContent = `${watch.length} 只股票`;
  if (signalSummary) {
    signalSummary.innerHTML = `
      <article class="similar-summary-card primary">
        <span>多策略确认</span>
        <strong>${resonantResults.length} <small>/ ${watch.length} 只</small></strong>
        <em>${resonantResults.length ? `累计 ${totalStrategyHits} 次策略命中` : "暂未命中其他策略"}</em>
      </article>
      <article class="similar-summary-card">
        <span>T+1 看涨</span>
        <strong>${bullishCount} <small>只</small></strong>
        <em>来自相似走势统一模型</em>
      </article>
      <article class="similar-summary-card">
        <span>当前自选</span>
        <strong>${watch.length} <small>只</small></strong>
        <em>优先查看带“策略命中”的股票</em>
      </article>
    `;
  }
  watchlist.innerHTML = rankedWatch.map((item) => {
    const result = similarResultForSymbol(item.symbol);
    const target = result?.target || {};
    const snapshot = result?.latest_snapshot || {};
    const nextDay = similarForecastRow(result, "next_1d");
    const nextMonth = similarForecastRow(result, "next_1m");
    const nextDayDecision = similarDecision(result, "next_1d");
    const selectedClass = item.symbol === state.similarSelectedSymbol ? "active" : "";
    const strategyHits = watchlistStrategyHits(result);
    const resonanceClass = strategyHits.length ? "has-strategy-hit" : "";
    const xueqiuUrl = xueqiuStockUrl(item.symbol);
    return `
      <tr
        class="similar-watch-row ${selectedClass} ${resonanceClass} ${item.pinned ? "is-pinned" : ""}"
        data-similar-symbol="${item.symbol}"
        data-watchlist-symbol="${item.symbol}"
        data-watchlist-name="${escapeHtml(item.name || item.symbol)}"
        data-watchlist-note="${escapeHtml(item.note || "")}"
        data-watchlist-pinned="${item.pinned ? "true" : "false"}"
        draggable="true"
        tabindex="0"
        title="按住拖动排序；右键可置顶"
      >
        <td class="similar-stock-cell">
          <strong>${item.name || item.symbol}${item.pinned ? `<span class="watchlist-pin-badge">置顶</span>` : ""}</strong>
          <span>${item.symbol} · ${item.industry || "-"}</span>
          <em>${target.target_date || "-"}</em>
        </td>
        <td class="similar-resonance-cell">
          <span class="similar-resonance-label">${strategyResonanceLabel(result)}</span>
          ${watchlistStrategyBadges(result)}
        </td>
        <td class="similar-market-cell">
          <strong>${fmtPrice(snapshot.close)}</strong>
          <span>20日 ${fmtPct(snapshot.ret_20d)}</span>
          <em>回撤 ${fmtPct(snapshot.drawdown_60d)} · 量比 ${snapshot.vol_ratio20 ?? "-"}</em>
        </td>
        <td class="similar-decision-cell">
          <strong class="similar-signal-${nextDayDecision.signal || "observe"}">${similarSignalLabel(nextDayDecision.signal)}</strong>
          <span>上涨概率 ${fmtPct(nextDay.selected_up_probability ?? nextDay.calibrated_up_probability ?? nextDay.up_probability)}</span>
          <em>${result?.optimization_summary?.effective_sample_size ?? result?.scan_summary?.matched_cases ?? "-"} 个有效样本</em>
        </td>
        <td class="similar-medium-cell">
          <strong>${fmtPct(nextMonth.selected_up_probability ?? nextMonth.calibrated_up_probability ?? nextMonth.up_probability)}</strong>
          <span>1月上涨概率</span>
          <em>收益中位 ${fmtPct(nextMonth.median)}</em>
        </td>
        <td class="similar-note-cell">
          <button type="button" class="similar-note-button" data-similar-note="${item.symbol}" aria-label="编辑 ${escapeHtml(item.name || item.symbol)} 的笔记">
            <strong>${item.note ? "查看 / 编辑" : "写笔记"}</strong>
            <span>${item.note ? escapeHtml(item.note) : "记录操作计划、关注价位等"}</span>
          </button>
        </td>
        <td class="similar-xueqiu-cell">
          ${xueqiuUrl ? `
            <a class="xueqiu-stock-link" data-similar-xueqiu href="${escapeHtml(xueqiuUrl)}" target="_blank" rel="noopener noreferrer" aria-label="在雪球查看 ${escapeHtml(item.name || item.symbol)}">
              雪球
            </a>
          ` : `<span class="xueqiu-stock-unavailable">—</span>`}
        </td>
        <td class="similar-row-actions"><button type="button" data-similar-remove="${item.symbol}">删除</button></td>
      </tr>
    `;
  }).join("") || `<tr><td colspan="8" class="empty-cell">暂无自选股票，请在上方输入股票代码加入</td></tr>`;

  const selected = selectedSimilarResult();
  overview.innerHTML = selected ? (() => {
    const item = selected;
    const target = item.target || {};
    const snapshot = item.latest_snapshot || {};
    const forecast = item.optimized_forecast?.length ? item.optimized_forecast : (item.forecast || []);
    const nextDecision = similarDecision(item, "next_1d");
    const validation = (item.validation_summary || []).find((row) => row.horizon === "next_1d") || {};
    const strategyHits = watchlistStrategyHits(item);
    return `
      <article class="similar-card active ${strategyHits.length ? "has-strategy-hit" : ""}">
        <div class="similar-card-head">
          <div>
            <span class="similar-detail-kicker">当前决策</span>
            <strong>${target.name || target.symbol}</strong>
            <span>${target.symbol} · ${target.target_date || "-"}</span>
          </div>
          <span class="similar-resonance-label">${strategyResonanceLabel(item)}</span>
        </div>
        <div class="similar-decision-banner similar-signal-${nextDecision.signal || "observe"}">
          <strong>T+1 ${similarSignalLabel(nextDecision.signal)}</strong>
          <span>${item.market_regime || "neutral"} / 行业 ${item.industry_regime || "neutral"}${nextDecision.risk_gate === "blocked" ? ` · 风险闸门：${(nextDecision.reasons || []).join("、")}` : ""}</span>
        </div>
        <div class="watchlist-strategy-summary ${strategyHits.length ? "has-hits" : ""}">
          <div><span>策略依据</span><strong>${strategyHits.length ? `${strategyHits.length} 个其他策略同时命中` : "暂无其他策略确认"}</strong></div>
          ${watchlistStrategyBadges(item, true)}
        </div>
        <div class="similar-snapshot">
          <div><span>收盘</span><strong>${fmtPrice(snapshot.close)}</strong></div>
          <div><span>20日涨跌</span><strong>${fmtPct(snapshot.ret_20d)}</strong></div>
          <div><span>60日回撤</span><strong>${fmtPct(snapshot.drawdown_60d)}</strong></div>
          <div><span>量比</span><strong>${snapshot.vol_ratio20 ?? "-"}</strong></div>
        </div>
        <div class="similar-forecast">
          ${forecast.map((row) => {
            const rowDecision = similarDecision(item, row.horizon);
            return `
            <div class="similar-forecast-row">
              <span>${similarForecastLabel(row.horizon)} 选择上涨 ${fmtPct(row.selected_up_probability ?? row.calibrated_up_probability ?? row.up_probability)} · ${similarProbabilitySourceLabel(row.probability_source)}</span>
              <strong>${rowDecision.bearish_max ?? "-"}/${rowDecision.bullish_min ?? "-"} · 中位 ${fmtPct(row.median)}</strong>
            </div>
          `}).join("")}
        </div>
        <div class="similar-validation-strip">
          <span>2025+ T+1 验证</span>
          <strong>本股覆盖 ${fmtPct(validation.selected_model_coverage ?? validation.coverage)} · 命中 ${fmtPct(validation.selected_model_accuracy ?? validation.direction_accuracy)}</strong>
          <small>全池命中 ${fmtPct(validation.global_model_accuracy)} · ${similarProbabilitySourceLabel(validation.selected_model)}</small>
        </div>
      </article>
    `;
  })() : `<article class="panel"><p class="subline">暂无分析结果，点击刷新分析</p></article>`;

  if (!selected) {
    if (detailTitle) detailTitle.textContent = "选择股票";
    if (detailMeta) detailMeta.textContent = "查看 T+1 量价情景";
    scenarioRows.innerHTML = `<tr><td colspan="7" class="empty-cell">暂无数据</td></tr>`;
    modelList.innerHTML = `<p class="subline">暂无模型结果</p>`;
    topRows.innerHTML = `<tr><td colspan="8" class="empty-cell">暂无数据</td></tr>`;
    return;
  }

  if (detailTitle) detailTitle.textContent = `${selected.target?.name || selected.target?.symbol} T+1 情景`;
  if (detailMeta) detailMeta.textContent = `${selected.target?.symbol} · ${selected.scan_summary?.matched_cases || 0} 个相似阶段`;
  scenarioRows.innerHTML = sortT1ScenarioRows(selected.t1_scenario_plan).map((row) => `
    <tr>
      <td>${row.t1_return_bucket}</td>
      <td>${row.t1_volume_bucket}</td>
      <td>${row.sample_count}</td>
      <td>${fmtPct(row.hit_up_3d_prob)}</td>
      <td>${fmtPct(row.hit_down_3d_prob)}</td>
      <td>${fmtPct(row.median_fwd_20d)}</td>
      <td class="${similarActionClass(row.action)}">${row.action}</td>
    </tr>
  `).join("") || `<tr><td colspan="7" class="empty-cell">暂无情景样本</td></tr>`;

  const summary = selected.sell_model_summary || {};
  if (modelMeta) modelMeta.textContent = `${summary.status || "-"} · 样本 ${summary.sample_count || 0} · 止盈 ${fmtPct(summary.take_profit_3d)} / 止损 ${fmtPct(summary.stop_loss_3d)}`;
  modelList.innerHTML = (selected.sell_model_plan || []).map((row) => `
    <div class="similar-model-item">
      <div>
        <strong>${row.scenario}</strong>
        <span>T+1 ${fmtPct(row.t1_return)} · 量比 ${row.t1_volume_ratio}</span>
      </div>
      <div>
        <span>上冲 ${fmtPct(row.model_hit_up_3d_prob)} / 下破 ${fmtPct(row.model_hit_down_3d_prob)}</span>
        <strong class="${similarActionClass(row.recommendation)}">${row.recommendation}</strong>
      </div>
    </div>
  `).join("") || `<p class="subline">模型样本不足</p>`;

  const topCases = selected.top_cases || [];
  topRows.innerHTML = topCases.map((row) => `
    <tr>
      <td>${row.rank}</td>
      <td>${row.name || row.symbol}<br><span class="subline">${row.symbol}</span></td>
      <td>${row.industry || "-"}</td>
      <td>${row.date || "-"}</td>
      <td><strong>${fallbackSimilarityScore(row, payload.config)}</strong></td>
      <td>${fmtPct(row.fwd_1d)}</td>
      <td>${fmtPct(row.fwd_20d)}</td>
      <td>${fmtPct(row.fwd_60d)}</td>
    </tr>
  `).join("") || `<tr><td colspan="8" class="empty-cell">暂无相似阶段</td></tr>`;
}

function similarWatchItem(symbol) {
  return (state.similarPayload?.watchlist || []).find((item) => item.symbol === symbol) || null;
}

function positionSimilarNoteTooltip(clientX, clientY) {
  const tooltip = document.querySelector("#similarNoteTooltip");
  if (!tooltip || tooltip.hidden) return;
  const gap = 12;
  const width = tooltip.offsetWidth || 320;
  const height = tooltip.offsetHeight || 100;
  tooltip.style.left = `${Math.max(gap, Math.min(clientX + gap, window.innerWidth - width - gap))}px`;
  tooltip.style.top = `${Math.max(gap, Math.min(clientY + gap, window.innerHeight - height - gap))}px`;
}

function showSimilarNoteTooltip(row, clientX, clientY) {
  const tooltip = document.querySelector("#similarNoteTooltip");
  const item = similarWatchItem(row?.dataset.similarSymbol);
  if (!tooltip || !item?.note) return;
  tooltip.querySelector("strong").textContent = `${item.name || item.symbol} · 笔记与操作计划`;
  tooltip.querySelector("p").textContent = item.note;
  tooltip.hidden = false;
  row.setAttribute("aria-describedby", "similarNoteTooltip");
  positionSimilarNoteTooltip(clientX, clientY);
}

function hideSimilarNoteTooltip(row = null) {
  const tooltip = document.querySelector("#similarNoteTooltip");
  if (tooltip) tooltip.hidden = true;
  row?.removeAttribute("aria-describedby");
}

async function loadLongStockPool(options = {}) {
  const requestedVariant = state.longVariant;
  state.longLoading = true;
  renderLongStockPool();
  const query = new URLSearchParams();
  query.set("variant", requestedVariant);
  if (state.signalDate) query.set("signal_date", state.signalDate);
  if (options.refresh) query.set("refresh", "true");
  try {
    const payload = await fetchJson(`/long/stock-pool?${query.toString()}`, workspaceRequestOptions(options));
    if (state.longVariant === requestedVariant) {
      state.longPayload = payload;
    }
  } catch (error) {
    showError(error);
    throw error;
  } finally {
    if (state.longVariant === requestedVariant) {
      state.longLoading = false;
      renderLongStockPool();
    }
  }
}

async function loadBydMinuteStrategy(options = {}) {
  state.bydLoading = true;
  renderBydPage();
  const shares = Number(document.querySelector("#bydSharesInput")?.value || 10000);
  const cost = Number(document.querySelector("#bydCostInput")?.value || 110.6061);
  const query = new URLSearchParams();
  query.set("shares", String(shares));
  query.set("cost", String(cost));
  if (options.refresh) query.set("refresh", "true");
  try {
    state.bydPayload = await fetchJson(`/byd/daily-plan?${query.toString()}`, workspaceRequestOptions(options));
    maybeShowBydTradeToast(state.bydPayload, options);
  } catch (error) {
    showError(error);
    throw error;
  } finally {
    state.bydLoading = false;
    renderBydPage();
  }
}

async function loadConvertibleBondPlan(options = {}) {
  state.cbLoading = true;
  renderConvertibleBondPage();
  const query = new URLSearchParams();
  if (state.signalDate) query.set("trade_date", state.signalDate);
  query.set("limit", "18");
  if (options.refresh) query.set("refresh", "true");
  try {
    state.cbPayload = await fetchJson(`/convertible-bonds/plan?${query.toString()}`, workspaceRequestOptions(options));
    const plans = state.cbPayload.strategy_plans || [];
    if (!state.selectedCbStrategy && plans.length) {
      state.selectedCbStrategy = "all";
    }
    const groups = cbCandidateGroups(state.selectedCbStrategy || "all");
    if (!state.selectedCbCode && groups.length) {
      state.selectedCbCode = groups[0].ts_code;
    }
    if (state.selectedCbCode && !groups.some((item) => item.ts_code === state.selectedCbCode)) {
      state.selectedCbCode = groups[0]?.ts_code || null;
    }
  } catch (error) {
    showError(error);
    throw error;
  } finally {
    state.cbLoading = false;
    renderConvertibleBondPage();
  }
}

async function loadConvertibleBondAllotments(options = {}) {
  state.cbAllotmentLoading = true;
  state.cbAllotmentError = "";
  renderConvertibleBondAllotments();
  const query = new URLSearchParams();
  query.set("limit", "120");
  query.set("include_listed_days", "180");
  query.set("stage_scope", "pipeline");
  if (options.refresh) query.set("refresh", "true");
  try {
    state.cbAllotmentPayload = await fetchJson(
      `/convertible-bonds/allotments?${query.toString()}`,
      workspaceRequestOptions(options)
    );
  } catch (error) {
    state.cbAllotmentError = error.message;
    showError(error);
    throw error;
  } finally {
    state.cbAllotmentLoading = false;
    renderConvertibleBondAllotments();
  }
}

function activeCbPlan() {
  const payload = state.cbPayload;
  if (!payload) return null;
  const plans = payload.strategy_plans || [];
  if (!plans.length) return payload;
  if (state.selectedCbStrategy === "all") return payload;
  const selected = plans.find((plan) => plan.strategy?.key === state.selectedCbStrategy);
  return selected || plans[0];
}

function cbStrategyPlans() {
  const payload = state.cbPayload;
  if (!payload) return [];
  return payload.strategy_plans?.length ? payload.strategy_plans : [payload];
}

function cbCandidateGroups(filterKey = "all") {
  const groups = new Map();
  cbStrategyPlans().forEach((plan) => {
    const strategy = plan.strategy || {};
    const include = filterKey === "all" || strategy.key === filterKey;
    (plan.candidates || []).forEach((candidate) => {
      const code = candidate.ts_code;
      if (!code) return;
      if (!groups.has(code)) {
        groups.set(code, {
          ...candidate,
          strategies: [],
          allStrategies: [],
          strategyCandidates: [],
        });
      }
      const group = groups.get(code);
      const hit = {
        key: strategy.key,
        name: strategy.name || strategy.key || "-",
        style: strategy.style || "策略",
        candidate,
        plan,
      };
      group.allStrategies.push(hit);
      if (include) {
        group.strategies.push(hit);
        group.strategyCandidates.push(candidate);
      }
    });
  });
  return Array.from(groups.values())
    .filter((group) => group.strategies.length)
    .map((group) => {
      const primary = group.strategyCandidates[0] || group.allStrategies[0]?.candidate || group;
      return {
        ...primary,
        strategies: group.strategies,
        allStrategies: group.allStrategies,
      };
    });
}

function bydNoticeKey(item, payload) {
  return [
    payload?.generated_at || "",
    item?.action || "",
    item?.kind || "",
    item?.price_line || "",
    item?.shares_delta || "",
  ].join("|");
}

function dismissBydTradeToast(markSeen = false) {
  const toast = document.querySelector("#bydTradeToast");
  if (!toast) return;
  if (markSeen) {
    const key = toast.dataset.noticeKey || "";
    if (key) state.bydLastNoticeKey = key;
  }
  toast.classList.remove("show", "sell", "buy");
  toast.innerHTML = "";
  toast.dataset.noticeKey = "";
}

function maybeShowBydTradeToast(payload, options = {}) {
  if (state.activePage !== "byd" || !payload) return;
  const toast = document.querySelector("#bydTradeToast");
  if (!toast) return;
  const triggered = (payload.alerts || []).filter((item) => item.triggered && Math.abs(Number(item.shares_delta || 0)) > 0);
  if (!triggered.length) {
    if (options.refresh) dismissBydTradeToast(false);
    return;
  }
  const item = triggered[0];
  const key = bydNoticeKey(item, payload);
  if (key === state.bydLastNoticeKey && !options.forceNotice) return;
  const isSell = item.action === "SELL";
  const rangeText = fmtRange(item.price_range);
  const direction = isSell ? "卖出提醒" : "买回提醒";
  const sharesText = `${Number(item.shares_delta) > 0 ? "+" : ""}${item.shares_delta} 股`;
  toast.dataset.noticeKey = key;
  toast.classList.toggle("sell", isSell);
  toast.classList.toggle("buy", !isSell);
  toast.classList.add("show");
  toast.innerHTML = `
    <div class="byd-toast-copy">
      <span>${direction} · ${payload.minute?.asof || payload.generated_at || "-"}</span>
      <strong>${item.title || direction}</strong>
      <p>${isSell ? "卖出区间" : "买回区间"} <b>${rangeText}</b> · 建议 ${sharesText}</p>
      <em>${item.detail || ""}</em>
    </div>
    <div class="byd-toast-actions">
      <button type="button" data-byd-toast-action="seen">已记录</button>
      <button type="button" data-byd-toast-action="close" aria-label="关闭 BYD 交易提醒">关闭</button>
    </div>
  `;
}

async function loadCalendar() {
  state.calendar = await fetchJson("/selector/calendar?start=2020-01-01", { timeoutMs: 15000 });
  const latestPageDate = state.activePage === "chan"
    ? (state.calendar.latest_chan_snapshot_date || state.calendar.latest_chan_signal_date)
    : state.calendar.latest_signal_date;
  if (!state.signalDate && latestPageDate) {
    state.signalDate = latestPageDate;
    const input = document.querySelector("#signalDateInput");
    if (input) input.value = state.signalDate;
  }
  if (!state.calendarMonth) {
    state.calendarMonth = (state.signalDate || latestPageDate || state.calendar.latest_signal_date || "").slice(0, 7);
  }
  renderCalendar();
  renderDateStatus();
}

async function loadChanModelStrategy(options = {}) {
  state.chanLoading = true;
  renderChanModelPage();
  const query = new URLSearchParams({ top_n: String(options.topN || 20) });
  if (!state.signalDate) {
    state.signalDate = state.calendar?.latest_chan_snapshot_date
      || state.calendar?.latest_chan_signal_date
      || state.calendar?.latest_signal_date
      || "";
  }
  state.signalDate = applySelectableSignalDate(state.signalDate);
  if (state.signalDate) query.set("signal_date", state.signalDate);
  if (options.refresh) query.set("refresh", "true");
  try {
    state.chanPayload = await fetchJson(`/chan/strategy-plan?${query.toString()}`, workspaceRequestOptions(options));
    state.signalDate = state.signalDate || state.chanPayload?.signal_date || "";
    const dateInput = document.querySelector("#signalDateInput");
    if (dateInput) dateInput.value = state.signalDate || state.chanPayload?.signal_date || "";
    const rows = state.chanPayload?.candidates || [];
    if (!rows.some((item) => item.symbol === state.chanSelectedSymbol)) {
      state.chanSelectedSymbol = rows[0]?.symbol || null;
    }
  } finally {
    state.chanLoading = false;
    renderChanModelPage();
  }
}

function chanCandidates() {
  return state.chanPayload?.candidates || [];
}

function selectedChanCandidate() {
  return chanCandidates().find((item) => item.symbol === state.chanSelectedSymbol) || chanCandidates()[0] || null;
}

function chanRuleLabel(item) {
  if (!item?.rule_name && !item?.rule_id) return "-";
  return item.rule_name || (item.rule_id === "chan_model_primary" ? "主策略" : "扩容策略");
}

function chanMetricCard(label, value, subtext = "") {
  return `
    <article class="signal-card chan-mini-card">
      <span>${label}</span>
      <strong>${value}</strong>
      ${subtext ? `<p>${subtext}</p>` : ""}
    </article>
  `;
}

function renderChanModelPage() {
  const payload = state.chanPayload;
  const rows = chanCandidates();
  const selected = selectedChanCandidate();
  const meta = document.querySelector("#chanPlanMeta");
  const count = document.querySelector("#chanCandidateCount");
  const metrics = document.querySelector("#chanMetrics");
  const tbody = document.querySelector("#chanRows");
  const detail = document.querySelector("#chanDetail");
  const detailTitle = document.querySelector("#chanDetailTitle");
  const detailMeta = document.querySelector("#chanDetailMeta");
  const notes = document.querySelector("#chanNotes");
  if (meta) {
    meta.textContent = state.chanLoading
      ? "正在加载缠论模型候选"
      : payload
      ? `${payload.signal_date || "-"} 收盘后分析 · T+1 开盘条件触发买入`
      : "等待加载";
  }
  if (count) count.textContent = state.chanLoading ? "加载中" : `${rows.length} 只候选`;
  if (metrics) {
    const primary = payload?.oot_primary || {};
    const expanded = payload?.oot_expanded || {};
    metrics.innerHTML = payload ? `
      <div class="chan-performance-title">
        <strong>历史样本表现</strong>
        <span>OOT · 持有 10 日</span>
      </div>
      <div class="chan-performance-item">
        <span>主策略</span>
        <strong>${fmtPct(primary.avg_return_10d)}</strong>
        <em>胜率 ${fmtRate(primary.win_rate_10d)} · PF ${Number(primary.profit_factor_10d || 0).toFixed(2)}</em>
      </div>
      <div class="chan-performance-item">
        <span>扩容策略</span>
        <strong>${fmtPct(expanded.avg_return_10d)}</strong>
        <em>胜率 ${fmtRate(expanded.win_rate_10d)} · PF ${Number(expanded.profit_factor_10d || 0).toFixed(2)}</em>
      </div>
    ` : `
      <div class="chan-performance-title"><strong>历史样本表现</strong><span>${state.chanLoading ? "加载中" : "-"}</span></div>
    `;
  }
  if (tbody) {
    tbody.innerHTML = rows.length ? rows.map((item) => `
      <tr class="${item.symbol === state.chanSelectedSymbol ? "selected-row" : ""}" data-chan-symbol="${item.symbol}" data-watchlist-symbol="${item.symbol}" data-watchlist-name="${item.name || ""}" data-watchlist-note="${escapeHtml(watchlistSourceNote(payload?.signal_date || item.date, `触发 ${chanRuleLabel(item)}`))}" tabindex="0">
        <td><strong>${item.symbol}</strong><span>${item.name || ""}</span></td>
        <td><span class="tag ${item.rule_id === "chan_model_primary" ? "strong" : ""}">${chanRuleLabel(item)}</span></td>
        <td>${Number(item.rank_score || 0).toFixed(3)}</td>
        <td>
          <div class="score-stack">
            <strong>${Number(item.pred_good || 0).toFixed(3)}</strong>
            <em>大赚 ${Number(item.pred_big10 || 0).toFixed(3)}</em>
          </div>
        </td>
        <td>${Number(item.chan_score || 0).toFixed(1)}</td>
        <td>${fmtPct(item.entry_gap_pct, 1)}</td>
        <td>${fmtWeight(item.position_pct)}</td>
      </tr>
    `).join("") : `<tr><td colspan="7" class="empty-cell">${state.chanLoading ? "正在加载候选" : "暂无缠论模型候选"}</td></tr>`;
  }
  if (detailTitle) detailTitle.textContent = selected ? `${selected.symbol} ${selected.name || ""}` : "选择候选";
  if (detailMeta) detailMeta.textContent = selected ? `${chanRuleLabel(selected)} · ${selected.signal_name || "缠论结构信号"}` : "";
  if (detail) {
    detail.innerHTML = selected ? `
      <div class="signal-cards chan-detail-cards">
        <article class="signal-card">
          <div class="signal-card-head">
            <div>
              <span>${selected.industry || "未分类行业"}</span>
              <h4>${selected.rule_description || selected.structure_note || "模型筛选候选"}</h4>
            </div>
            <span>排名 ${Number(selected.rank_score || 0).toFixed(3)}</span>
          </div>
          <div class="metric-strip">
            <span>10日预测胜率 ${fmtRate(selected.pred_win10)}</span>
            <span>好交易概率 ${fmtRate(selected.pred_good)}</span>
            <span>大收益概率 ${fmtRate(selected.pred_big10)}</span>
            <span>最大仓位 ${fmtWeight(selected.position_pct)}</span>
          </div>
        </article>
        <div class="chan-mini-grid">
          ${chanMetricCard("信号收盘", fmtPrice(selected.close), "T日收盘后确认")}
          ${chanMetricCard("中枢区间", `${fmtPrice(selected.center_low)}-${fmtPrice(selected.center_high)}`, `宽度 ${fmtRate(selected.center_width)}`)}
          ${chanMetricCard("T+1 开盘跳空", fmtPct(selected.entry_gap_pct, 1), "高开过大按计划降仓或放弃")}
          ${chanMetricCard("笔振幅", fmtRate(selected.stroke_amplitude), "过滤过热结构")}
        </div>
        <article class="signal-card">
          <div class="signal-grid">
            <div>
              <span>买入计划</span>
              <p>${formatBuyPlanText(selected.buy_plan || "T+1 开盘满足条件后买入；不满足则空仓观察。")}</p>
            </div>
            <div>
              <span>卖出计划</span>
              <p>${selected.sell_plan || "T+2 起根据止盈、止损和持仓天数规则处理。"}</p>
            </div>
            <div>
              <span>结构备注</span>
              <p>${selected.structure_note || selected.signal_name || "等待更多结构信息。"}</p>
            </div>
          </div>
        </article>
      </div>
    ` : `<div class="empty-state">${state.chanLoading ? "正在加载缠论策略候选" : "暂无候选"}</div>`;
  }
  if (notes) notes.innerHTML = (payload?.notes || []).map((item) => `<li>${item}</li>`).join("");
}

function monthOffset(month, offset) {
  const [year, monthValue] = month.split("-").map(Number);
  const date = new Date(Date.UTC(year, monthValue - 1 + offset, 1));
  return date.toISOString().slice(0, 7);
}

function monthLabel(month) {
  const [year, monthValue] = month.split("-");
  return `${year}年${monthValue}月`;
}

function filteredStocks() {
  const rows = state.payload?.stocks || [];
  const q = state.query.trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((item) => (
    item.symbol.toLowerCase().includes(q)
    || (item.name || "").toLowerCase().includes(q)
    || (item.industry || "").toLowerCase().includes(q)
    || item.matched_families.join(",").toLowerCase().includes(q)
  ));
}

function selectedStock() {
  return (state.payload?.stocks || []).find((item) => item.symbol === state.selectedSymbol) || filteredStocks()[0] || null;
}

function renderHeader() {
  const actualDate = state.payload?.signal_date || "";
  const requestedDate = state.signalDate || actualDate;
  const dateText = requestedDate && actualDate && requestedDate !== actualDate
    ? `选择 ${requestedDate}，实际使用本地最新可用 ${actualDate} 收盘后选股`
    : (actualDate ? `${actualDate} 收盘后选股` : "");
  document.querySelector("#generatedAt").textContent = state.loading ? "正在加载" : (state.payload ? `更新于 ${state.payload.generated_at}` : "加载中");
  document.querySelector("#signalDate").textContent = state.payload ? dateText : "";
  document.querySelector("#executionDate").textContent = state.payload?.execution_date || "";
  if (state.loading) {
    document.querySelector("#stockCount").textContent = "加载中";
  } else {
    const shown = filteredStocks().length;
    const total = Number(state.payload?.total_stock_count || shown);
    const complete = Number(state.payload?.complete_stock_count || total);
    const limit = Number(state.payload?.display_limit || shown);
    document.querySelector("#stockCount").textContent = complete > total
      ? `展示 ${shown} / 可操作 ${total} / 完整 ${complete} 只`
      : total > shown || total > limit
      ? `展示 ${shown} / 可操作 ${total} 只`
      : `${shown} 只股票`;
  }
}

function renderCalendar() {
  const wrap = document.querySelector("#dateCalendar");
  const panel = document.querySelector("#calendarPanel");
  const toggle = document.querySelector("#calendarToggle");
  const monthLabelEl = document.querySelector("#calendarMonthLabel");
  if (!wrap || !state.calendar) return;
  if (panel) panel.classList.toggle("open", state.calendarOpen);
  if (toggle) toggle.textContent = state.calendarOpen ? "收起" : "选择日期";
  const selected = state.signalDate || state.payload?.signal_date || state.calendar.latest_signal_date || "";
  const currentMonth = state.calendarMonth || selected.slice(0, 7);
  if (monthLabelEl) monthLabelEl.textContent = monthLabel(currentMonth);
  const monthDays = (state.calendar.days || []).filter((day) => day.date.startsWith(currentMonth));
  const firstDay = monthDays[0]?.date || `${currentMonth}-01`;
  const leadingBlanks = new Date(`${firstDay}T00:00:00`).getDay();
  const blanks = Array.from({ length: leadingBlanks }, () => `<span class="calendar-empty"></span>`).join("");
  wrap.innerHTML = blanks + monthDays.map((day) => {
    const active = day.date === selected || (!state.signalDate && day.date === state.payload?.signal_date);
    const status = state.activePage === "chan" && day.is_open
      ? (day.has_chan_model_strategy || day.has_chan_model_strategy_snapshot ? "ready" : "open_missing_data")
      : day.status;
    const label = state.activePage === "chan" && day.is_open
      ? (day.has_chan_model_strategy || day.has_chan_model_strategy_snapshot ? "缠论已生成" : "缠论无候选")
      : day.label;
    return `
      <button
        class="calendar-day ${status} ${active ? "selected" : ""}"
        data-date="${day.date}"
        type="button"
        ${day.disabled ? "disabled" : ""}
        title="${day.date} · ${label}${day.effective_signal_date && day.effective_signal_date !== day.date ? ` · 实际使用 ${day.effective_signal_date}` : ""}"
      >
        <span>${day.date.slice(5)}</span>
      </button>
    `;
  }).join("");
  wrap.querySelectorAll("button[data-date]:not(:disabled)").forEach((button) => {
    button.addEventListener("click", async () => {
      state.signalDate = applySelectableSignalDate(button.dataset.date);
      state.calendarOpen = false;
      state.selectedSymbol = null;
      const input = document.querySelector("#signalDateInput");
      if (input) input.value = state.signalDate;
      renderDateStatus();
      renderCalendar();
      if (state.activePage === "long") {
        state.longPayload = null;
        await loadLongStockPool().catch(showError);
      } else if (state.activePage === "chan") {
        state.chanPayload = null;
        state.chanSelectedSymbol = null;
        await loadChanModelStrategy().catch(showError);
      } else if (state.activePage === "cb") {
        state.cbPayload = null;
        await loadConvertibleBondPlan().catch(showError);
      } else {
        await loadSelector().catch(showError);
      }
    });
  });
}

function renderDateStatus() {
  const control = document.querySelector("#sharedDateControl");
  const statusEl = document.querySelector("#dateStatus");
  const input = document.querySelector("#signalDateInput");
  if (!control || !statusEl || !input) return;
  const selected = state.signalDate || state.payload?.signal_date || state.calendar?.latest_signal_date || "";
  input.value = selected || "";
  const day = calendarDay(selected);
  control.classList.remove("date-ready", "date-missing", "date-closed");
  input.classList.remove("date-ready", "date-missing", "date-closed");
  if (!day) {
    statusEl.textContent = state.activePage === "long"
      ? (selected ? `${selected} 长线股票池可按最近可用截面生成` : "选择长线股票池日期")
      : state.activePage === "chan"
      ? (selected ? `${selected} 缠论策略将按该日模型候选生成` : "选择缠论策略日期")
      : state.activePage === "cb"
      ? (selected ? `${selected} 可转债将使用本地最近可用行情` : "选择可转债计划日期")
      : "选择已生成快照的交易日";
    return;
  }
  if (day.status === "open_missing_data") {
    control.classList.add("date-missing");
    input.classList.add("date-missing");
    statusEl.textContent = state.activePage === "long"
      ? `${selected} 长线将使用该日之前最近可用截面`
      : state.activePage === "chan"
      ? `${selected} 缠论策略将按该日模型候选生成`
      : state.activePage === "cb"
      ? `${selected} 可转债将回看最近可用行情`
      : `${selected} 为交易日，但暂未生成策略快照；将回看 ${day.effective_signal_date || "最近可用日"}`;
  } else if (day.status === "closed") {
    control.classList.add("date-closed");
    input.classList.add("date-closed");
    statusEl.textContent = state.activePage === "long"
      ? `${selected} 为休市日，长线将回看最近可用交易日`
      : state.activePage === "chan"
      ? `${selected} 为休市日，缠论策略将回看最近交易日`
      : state.activePage === "cb"
      ? `${selected} 为休市日，可转债将回看最近交易日`
      : `${selected} 为休市日`;
  } else {
    control.classList.add("date-ready");
    input.classList.add("date-ready");
    if (state.activePage === "long") {
      statusEl.textContent = day.has_long_stock_pool_snapshot
        ? `${selected} 已有长线股票池快照`
        : `${selected} 长线将使用该日之前最近可用截面`;
    } else if (state.activePage === "chan") {
      statusEl.textContent = day.has_chan_model_strategy
        ? `${selected} 已有缠论模型策略候选`
        : day.has_chan_model_strategy_snapshot
        ? `${selected} 已完成策略回溯，暂无符合规则的候选`
        : `${selected} 暂无缠论模型候选`;
    } else if (state.activePage === "cb") {
      statusEl.textContent = `${selected} 可生成可转债低位网格计划`;
    } else {
      statusEl.textContent = `${selected} 已有策略快照`;
    }
  }
}

function renderStrategyFilters() {
  const wrap = document.querySelector("#strategyFilters");
  wrap.innerHTML = (state.payload?.available_strategies || []).map((item) => {
    const active = state.selectedStrategies.has(item.key);
    return `
      <button class="filter-chip ${active ? "active" : ""}" data-strategy="${item.key}" type="button" aria-pressed="${active}">
        <strong>${item.label}</strong>
        <span>${item.status}</span>
      </button>
    `;
  }).join("");
  wrap.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      const key = button.dataset.strategy;
      if (state.selectedStrategies.has(key)) {
        state.selectedStrategies.delete(key);
      } else {
        state.selectedStrategies.add(key);
      }
      await loadSelector().catch(showError);
    });
  });
}

function renderStockRows() {
  const rows = filteredStocks();
  const body = document.querySelector("#stockRows");
  if (state.loading) {
    body.innerHTML = `<tr><td colspan="10" class="empty-cell">正在加载股票池...</td></tr>`;
    document.querySelector("#stockCount").textContent = "加载中";
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="10" class="empty-cell">当前筛选条件下没有股票</td></tr>`;
    document.querySelector("#stockCount").textContent = "0 只股票";
    return;
  }
  body.innerHTML = rows.map((item) => `
    <tr class="${item.symbol === state.selectedSymbol ? "selected-row" : ""}" data-symbol="${item.symbol}" data-watchlist-symbol="${item.symbol}" data-watchlist-name="${item.name || ""}" data-watchlist-note="${escapeHtml(watchlistSourceNote(item.date || state.payload?.signal_date, `触发 ${(item.matched_families || []).join(" / ")} 策略`))}" tabindex="0">
      <td>
        <strong class="copyable-symbol">${item.symbol}</strong>
        <span>${item.name || ""}</span>
      </td>
      <td>${item.industry || "-"}</td>
      <td>${fmtPrice(item.close)}</td>
      <td>${item.matched_count}</td>
      <td>${item.matched_families.map((family) => `<span class="tag">${family}</span>`).join("")}</td>
      <td>
        <span class="score-stack">
          <strong>${Number(item.opportunity_score ?? item.selector_score ?? 0).toFixed(1)}</strong>
          <em>${item.score_band || ""} ${item.score_percentile_label || ""}</em>
        </span>
      </td>
      <td>
        <span class="score-stack">
          <strong>${Number(item.holding_score ?? 0).toFixed(1)}</strong>
          <em>${item.score_risk_note || ""}</em>
        </span>
      </td>
      <td>${fmtPct(item.best_avg_return_pct)}</td>
      <td>${Number(item.best_profit_factor || 0).toFixed(2)}</td>
      <td>${item.date || ""}</td>
    </tr>
  `).join("");
  body.querySelectorAll("tr[data-symbol]").forEach((row) => {
    row.addEventListener("click", (event) => {
      const selection = window.getSelection?.().toString();
      if (selection) return;
      state.selectedSymbol = row.dataset.symbol;
      render();
    });
  });
}

function renderStockDetail() {
  const stock = selectedStock();
  const title = document.querySelector("#detailTitle");
  const meta = document.querySelector("#detailMeta");
  const body = document.querySelector("#signalCards");
  if (!stock) {
    title.textContent = "未选择股票";
    meta.textContent = "";
    body.innerHTML = `<div class="empty-state">${state.loading ? "正在加载股票池..." : "请先选择一只股票"}</div>`;
    return;
  }
  title.textContent = `${stock.symbol} ${stock.name || ""}`;
  meta.textContent = `行业 ${stock.industry || "-"} · 收盘 ${fmtPrice(stock.close)} · 买入分 ${Number(stock.opportunity_score ?? stock.selector_score ?? 0).toFixed(1)}（${stock.score_band || "-"}，${stock.score_percentile_label || "-"}） · 持有分 ${Number(stock.holding_score ?? 0).toFixed(1)} · ${stock.score_usage_hint || ""} · ${stock.score_risk_note || ""} · 命中 ${stock.matched_count} 个策略组 · ${stock.matched_families.join(" / ")} · ${stock.rank_reason || ""}`;
  body.innerHTML = stock.signals.map((signal) => {
    const metrics = signal.metrics || {};
    return `
      <article class="signal-card">
        <div class="signal-card-head">
          <div>
            <span class="tag strong">${signal.strategy_group_label || signal.strategy_group || signal.strategy_family}</span>
            <h4>${signal.strategy_name}</h4>
          </div>
          <span>${signal.timeframe}</span>
        </div>
        <div class="signal-grid">
          <div>
            <span>选股逻辑</span>
            <p>${signal.logic}</p>
          </div>
          <div>
            <span>命中原因</span>
            <p>${signal.reason}</p>
          </div>
          <div>
            <span>T+1执行</span>
            <p>${formatBuyPlanText(signal.buy_plan)}</p>
          </div>
          <div>
            <span>卖出策略</span>
            <p>${signal.sell_plan}</p>
          </div>
        </div>
        <div class="metric-strip">
          <span>${signal.metrics_text}</span>
          <span>胜率 ${fmtRate(metrics.win_rate)}</span>
          <span>回撤 ${fmtPct(metrics.max_drawdown_pct)}</span>
          <span>PF ${Number(metrics.profit_factor || 0).toFixed(2)}</span>
        </div>
      </article>
    `;
  }).join("");
}

function renderNotes() {
  document.querySelector("#notes").innerHTML = (state.payload?.notes || []).map((item) => `<li>${item}</li>`).join("");
}

function renderPageShell() {
  ensureWorkspaceTabs();
  const shortPage = document.querySelector("#shortPage");
  const chanPage = document.querySelector("#chanPage");
  const longPage = document.querySelector("#longPage");
  const bydPage = document.querySelector("#bydPage");
  const similarPage = document.querySelector("#similarPage");
  const cbPage = document.querySelector("#cbPage");
  const cbAllotmentPage = document.querySelector("#cbAllotmentPage");
  const filterSection = document.querySelector("#strategyFilterSection");
  const longStrategySection = document.querySelector("#longStrategySection");
  const cbStrategySection = document.querySelector("#cbStrategySection");
  const clearButton = document.querySelector("#clearFilters");
  const dateControl = document.querySelector("#sharedDateControl");
  const longDateSlot = document.querySelector("#longDateSlot");
  const cbDateSlot = document.querySelector("#cbDateSlot");
  const chanDateSlot = document.querySelector("#chanDateSlot");
  const shortActions = document.querySelector("#searchInput")?.parentElement;
  const searchInput = document.querySelector("#searchInput");
  shortPage?.classList.toggle("active", state.activePage === "short");
  chanPage?.classList.toggle("active", state.activePage === "chan");
  longPage?.classList.toggle("active", state.activePage === "long");
  bydPage?.classList.toggle("active", state.activePage === "byd");
  similarPage?.classList.toggle("active", state.activePage === "similar");
  cbPage?.classList.toggle("active", state.activePage === "cb");
  cbAllotmentPage?.classList.toggle("active", state.activePage === "cbAllotment");
  filterSection?.classList.toggle("hidden", state.activePage !== "short");
  clearButton?.classList.toggle("hidden", state.activePage !== "short");
  longStrategySection?.classList.toggle("hidden", state.activePage !== "long");
  cbStrategySection?.classList.toggle("hidden", state.activePage !== "cb");
  if (dateControl && longDateSlot && cbDateSlot && chanDateSlot && shortActions && searchInput) {
    if (state.activePage === "long") {
      longDateSlot.appendChild(dateControl);
    } else if (state.activePage === "chan") {
      chanDateSlot.appendChild(dateControl);
    } else if (state.activePage === "cb") {
      cbDateSlot.appendChild(dateControl);
    } else if (dateControl.parentElement !== shortActions) {
      shortActions.insertBefore(dateControl, searchInput);
    }
  }
  document.querySelectorAll(".page-tab").forEach((button) => {
    const active = button.dataset.page === state.activePage;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  WORKSPACE_TABS.forEach((tab) => {
    const panel = document.querySelector(`#${tab.panelId}`);
    panel?.setAttribute("role", "tabpanel");
    panel?.setAttribute("aria-label", tab.label);
    panel?.setAttribute("aria-hidden", String(tab.key !== state.activePage));
  });
  window.requestAnimationFrame(() => {
    revealActiveWorkspaceTab();
    if (!focusWorkspaceTabAfterRender) return;
    document.querySelector(".page-view.active .page-tab.active")?.focus({ preventScroll: true });
    focusWorkspaceTabAfterRender = false;
  });
}

function renderLongOverview() {
  const strategy = currentLongStrategy();
  const title = document.querySelector("#longHeroTitle");
  const description = document.querySelector("#longHeroDescription");
  const tag = document.querySelector("#longHeroTag");
  const note = document.querySelector("#longHeroNote");
  const summaryLabel = document.querySelector("#longSummaryLabel");
  const summaryName = document.querySelector("#longSummaryName");
  if (tag) tag.textContent = strategy.tag;
  if (title) title.textContent = `${strategy.version} ${strategy.name}`;
  if (description) description.textContent = strategy.description;
  if (note) note.textContent = strategy.note;
  if (summaryLabel) summaryLabel.textContent = strategy.summaryLabel;
  if (summaryName) summaryName.textContent = strategy.name;
  document.querySelector("#longMetricAnnual").textContent = strategy.metrics.annual;
  document.querySelector("#longMetricDrawdown").textContent = strategy.metrics.drawdown;
  document.querySelector("#longMetricSharpe").textContent = strategy.metrics.sharpe;
  document.querySelector("#longMetricWeight").textContent = strategy.metrics.weight;
}

function renderLongStrategies() {
  const body = document.querySelector("#longStrategyRows");
  if (!body) return;
  const strategy = currentLongStrategy();
  const head = document.querySelector(".long-table thead");
  if (head) {
    head.innerHTML = `
      <tr>
        <th>年份</th>
        <th>${strategy.name}</th>
      </tr>
    `;
  }
  body.innerHTML = strategy.yearly.map((row) => {
    const negative = String(row.ret).startsWith("-");
    return `
      <tr>
        <td><strong>${row.year}</strong></td>
        <td>
          <span class="yearly-cell">
            <strong class="${negative ? "negative" : "positive"}">${row.ret}</strong>
            <em>回撤 ${row.dd} · 仓位 ${row.weight}</em>
          </span>
        </td>
      </tr>
    `;
  }).join("");
}

function stateLabel(stateName) {
  const labels = {
    WATCH: "观察",
    BUILDING: "建仓",
    CORE: "核心",
    T_ACTIVE: "做T",
    REDUCE: "降仓",
    EXIT: "清仓",
    COOLDOWN: "冷却",
  };
  return labels[stateName] || stateName || "-";
}

function renderLongStockPool() {
  const body = document.querySelector("#longStockRows");
  const meta = document.querySelector("#longPoolMeta");
  const counts = document.querySelector("#longStateCounts");
  if (!body || !meta || !counts) return;
  document.querySelectorAll(".long-variant-button").forEach((button) => {
    const active = button.dataset.longVariant === state.longVariant;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  if (state.longLoading) {
    meta.textContent = "正在生成长线股票池...";
    counts.innerHTML = "";
    body.innerHTML = `<tr><td colspan="9" class="empty-cell">正在按策略状态机生成股票池...</td></tr>`;
    return;
  }
  const payload = state.longPayload;
  if (!payload) {
    meta.textContent = "等待加载";
    counts.innerHTML = "";
    body.innerHTML = `<tr><td colspan="9" class="empty-cell">切到长线策略后加载股票池</td></tr>`;
    return;
  }
  meta.textContent = `${payload.variant_name} · 信号日 ${payload.signal_date} · 市场 ${payload.market_regime} · ${payload.stocks.length} 只`;
  counts.innerHTML = Object.entries(payload.state_counts || {}).map(([key, value]) => (
    `<span class="state-pill ${key}">${stateLabel(key)} ${value}</span>`
  )).join("");
  if (!payload.stocks.length) {
    body.innerHTML = `<tr><td colspan="9" class="empty-cell">当前没有长线候选股票</td></tr>`;
    return;
  }
  body.innerHTML = payload.stocks.map((item) => `
    <tr data-watchlist-symbol="${item.ts_code}" data-watchlist-name="${item.name || ""}" data-watchlist-note="${escapeHtml(watchlistSourceNote(payload.signal_date, `${payload.variant_name} · ${stateLabel(item.state)}`))}" tabindex="0">
      <td><span class="state-pill ${item.state}">${stateLabel(item.state)}</span></td>
      <td>
        <strong>${item.ts_code}</strong>
        <span>${item.name || ""}</span>
      </td>
      <td>${item.industry || "-"}</td>
      <td>
        <span class="score-stack left">
          <strong>${item.action || "-"}</strong>
          <em>${item.t_action || "HOLD"} · ${item.t_profile || "-"}</em>
        </span>
      </td>
      <td>
        <span class="score-stack">
          ${longPositionPlan(item)}
        </span>
      </td>
      <td>
        <span class="score-stack">
          <strong>${Number(item.long_score || 0).toFixed(1)}</strong>
          <em>成长 ${Number(item.growth_score || 0).toFixed(1)} · 趋势 ${Number(item.trend_score || 0).toFixed(1)}</em>
        </span>
      </td>
      <td>
        <span class="score-stack price-plan">
          <strong>${fmtPrice(item.close)}</strong>
          <em>积极建仓 <= ${fmtPrice(item.price_levels?.entry_aggressive_price)}</em>
          <em>做T低吸 ${item.price_levels?.t_buy_text || "-"}</em>
          <em>做T高抛 ${item.price_levels?.t_sell_text || "-"}</em>
          <em>降仓 < ${fmtPrice(item.price_levels?.reduce_ma60_price)} · 清仓 < ${fmtPrice(item.price_levels?.exit_ma120_price)}</em>
        </span>
      </td>
      <td>
        ${(() => {
          const coverage = analystCoverageText(item);
          return `
        <span class="score-stack">
          <strong>${coverage.main}</strong>
          <em>${coverage.sub}</em>
        </span>
          `;
        })()}
      </td>
      <td class="reason-cell">${item.reason || item.t_reason || "-"}</td>
    </tr>
  `).join("");
}

function renderConvertibleBondPage() {
  const rootPayload = state.cbPayload;
  const payload = activeCbPlan();
  const rows = document.querySelector("#cbRows");
  const meta = document.querySelector("#cbPlanMeta");
  const metrics = document.querySelector("#cbMetrics");
  const strategyWrap = document.querySelector("#cbStrategyCards");
  const detailTitle = document.querySelector("#cbDetailTitle");
  const detailMeta = document.querySelector("#cbDetailMeta");
  const detail = document.querySelector("#cbPlanDetail");
  if (!rows || !meta || !metrics || !detail) return;
  const generatedAt = document.querySelector("#generatedAt");
  if (state.cbLoading) {
    if (generatedAt) generatedAt.textContent = "正在加载可转债计划";
    meta.textContent = "正在读取最近可用的可转债计划快照...";
    metrics.innerHTML = "";
    if (strategyWrap) strategyWrap.innerHTML = "";
    rows.innerHTML = `<tr><td colspan="11" class="empty-cell">正在读取低位网格策略候选...</td></tr>`;
    detail.innerHTML = `<div class="empty-state">正在读取分批买入和网格计划...</div>`;
    return;
  }
  if (!rootPayload || !payload) {
    if (generatedAt && state.activePage === "cb") generatedAt.textContent = "等待可转债计划";
    meta.textContent = "等待加载";
    metrics.innerHTML = "";
    if (strategyWrap) strategyWrap.innerHTML = "";
    rows.innerHTML = `<tr><td colspan="11" class="empty-cell">切到可转债页面后加载候选计划</td></tr>`;
    detail.innerHTML = `<div class="empty-state">选择一只可转债查看操作计划</div>`;
    return;
  }
  const plans = cbStrategyPlans();
  const primaryPlan = plans[0] || payload;
  const market = payload.market_state || primaryPlan.market_state || {};
  if (!state.selectedCbStrategy && plans.length) state.selectedCbStrategy = "all";
  const activeStrategyKey = state.selectedCbStrategy || "all";
  const candidates = cbCandidateGroups(activeStrategyKey);
  if (generatedAt && state.activePage === "cb") generatedAt.textContent = `更新于 ${rootPayload.generated_at || ""}`;
  if (strategyWrap) {
    const totalGroups = cbCandidateGroups("all");
    const allCard = `
        <button class="cb-strategy-card ${activeStrategyKey === "all" ? "active" : ""}" type="button" data-cb-strategy="all" aria-pressed="${activeStrategyKey === "all"}">
          <span>全策略</span>
          <strong>全部策略命中</strong>
          <em>${totalGroups.length} 只去重候选 · 点击转债查看全部计划</em>
        </button>
      `;
    strategyWrap.innerHTML = allCard + plans.map((plan) => {
      const strategy = plan.strategy || {};
      const bt2024 = strategy.backtest?.from_2024 || {};
      const active = strategy.key === activeStrategyKey;
      return `
        <button class="cb-strategy-card ${active ? "active" : ""}" type="button" data-cb-strategy="${strategy.key || ""}" aria-pressed="${active}">
          <span>${strategy.style || "策略"}</span>
          <strong>${strategy.name || strategy.key || "-"}</strong>
          <em>${(plan.candidates || []).length} 只 · 2024+ 年化 ${fmtRate(bt2024.annual_return, 2)} · Sharpe ${bt2024.sharpe == null ? "-" : Number(bt2024.sharpe).toFixed(2)} · 回撤 ${fmtRate(bt2024.max_drawdown, 2)}</em>
        </button>
      `;
    }).join("");
    strategyWrap.querySelectorAll("[data-cb-strategy]").forEach((el) => {
      el.addEventListener("click", () => {
        state.selectedCbStrategy = el.dataset.cbStrategy;
        const groups = cbCandidateGroups(state.selectedCbStrategy);
        state.selectedCbCode = groups?.[0]?.ts_code || null;
        renderConvertibleBondPage();
      });
    });
  }
  const strategyName = activeStrategyKey === "all" ? "全部可转债策略" : (payload.strategy?.name || "可转债策略");
  const snapshotNote = rootPayload.cache?.stale
    ? ` · 最近可用快照 ${rootPayload.cache.snapshot_date || payload.trade_date || rootPayload.trade_date}`
    : "";
  meta.textContent = `${strategyName} · 信号日 ${payload.trade_date || rootPayload.trade_date} · ${market.entry_permission || "-"}${snapshotNote}`;
  metrics.innerHTML = `
    <div><span>去重候选</span><strong>${candidates.length} 只</strong></div>
    <div><span>当前策略</span><strong>${activeStrategyKey === "all" ? `${plans.length} 套` : (payload.strategy?.name || "-")}</strong></div>
    <div><span>2024+回撤</span><strong class="negative">${activeStrategyKey === "all" ? "-" : fmtRate(payload.strategy?.backtest?.from_2024?.max_drawdown, 2)}</strong></div>
    <div><span>双低中位</span><strong>${Number(market.median_double_low || 0).toFixed(1)}</strong></div>
    <div><span>20日趋势</span><strong class="${Number(market.trend_20d || 0) < 0 ? "negative" : "positive"}">${fmtRate(market.trend_20d, 3)}</strong></div>
    <div><span>趋势广度</span><strong>${fmtRate(market.trend_breadth, 1)}</strong></div>
    <div><span>执行状态</span><strong>${market.entry_permission || "-"}</strong><em>${market.existing_grid_permission || ""}</em></div>
  `;
  if (!candidates.length) {
    rows.innerHTML = `<tr><td colspan="11" class="empty-cell">当前没有满足低位网格条件的可转债</td></tr>`;
    detail.innerHTML = `<div class="empty-state">暂无可执行候选，等待低位条件或市场状态恢复</div>`;
    return;
  }
  rows.innerHTML = candidates.map((item) => {
    const plan = item.operation_plan || {};
    const active = item.ts_code === state.selectedCbCode;
    return `
      <tr class="${active ? "selected-row" : ""}" data-cb-code="${item.ts_code}">
        <td><strong>${item.ts_code}</strong></td>
        <td>${item.bond_name || "-"}</td>
        <td><strong>${item.stock_code || item.stk_code || "-"}</strong><span>${item.stock_name || "-"}</span></td>
        <td>${fmtPrice(item.close)}</td>
        <td>${Number(item.premium_rate || 0).toFixed(1)}%</td>
        <td>${Number(item.double_low || 0).toFixed(1)}</td>
        <td>${fmtRate(item.price_position_252, 1)}</td>
        <td><strong>${item.risk_label || "-"}</strong><span>${plan.grid_step_pct || 0}% 网格</span></td>
        <td><strong>${plan.max_parts || 0}份</strong><span>首批 ${plan.first_buy_parts || 0}份</span></td>
        <td>${(item.allStrategies || item.strategies || []).map((hit) => `<span class="strategy-hit">${hit.name}</span>`).join("")}</td>
        <td>${item.action || "-"}</td>
      </tr>
    `;
  }).join("");
  rows.querySelectorAll("[data-cb-code]").forEach((el) => {
    el.addEventListener("click", () => {
      state.selectedCbCode = el.dataset.cbCode;
      renderConvertibleBondPage();
    });
  });
  const selected = candidates.find((item) => item.ts_code === state.selectedCbCode) || candidates[0];
  if (!selected) return;
  state.selectedCbCode = selected.ts_code;
  if (detailTitle) detailTitle.textContent = `${selected.ts_code} ${selected.bond_name || ""}`;
  if (detailMeta) {
    detailMeta.textContent = `${selected.stock_code || selected.stk_code || "-"} ${selected.stock_name || "-"} · ${selected.risk_label || "标准"} · 价格 ${fmtPrice(selected.close)} · 溢价 ${Number(selected.premium_rate || 0).toFixed(1)}% · 双低 ${Number(selected.double_low || 0).toFixed(1)}`;
  }
  detail.innerHTML = (selected.allStrategies || selected.strategies || []).map((hit) => {
    const item = hit.candidate || selected;
    const plan = item.operation_plan || {};
    return `
      <section class="cb-strategy-plan-block">
        <div class="cb-strategy-plan-head">
          <span>${hit.style}</span>
          <strong>${hit.name}</strong>
          <em>${item.action || "-"}</em>
        </div>
        <section class="cb-plan-summary">
          <article>
            <span>策略动作</span>
            <strong>${item.action || "-"}</strong>
            <p>${item.execution_enabled ? "可按计划分批执行" : "市场过滤为弱市，未成交网格暂停；已有实际持仓才继续管理"}</p>
          </article>
          <article>
            <span>份数口径</span>
            <strong>${plan.unit_definition || payload.unit_definition}</strong>
            <p>最大 ${plan.max_parts || 0} 份；首批 ${plan.first_buy_parts || 0} 份</p>
          </article>
          <article>
            <span>网格大小</span>
            <strong>${plan.grid_step_pct || 0}%</strong>
            <p>约 ${fmtPrice(plan.grid_step_price)} 元/格，以当前参考价动态折算</p>
          </article>
          <article>
            <span>趋势过滤</span>
            <strong>${item.trend_strength == null ? "-" : Number(item.trend_strength).toFixed(0)} / ${item.six_sword_daily == null ? "-" : Number(item.six_sword_daily).toFixed(0)}</strong>
            <p>5日 ${fmtPct(item.return_5d, 2)} · 1日 ${fmtPct(item.return_1d, 2)} · 60日位置 ${fmtRate(item.price_position_60d, 1)}</p>
          </article>
        </section>
        <section class="cb-grid-columns">
          <div>
            <h4>买入网格</h4>
            ${(plan.buy_levels || []).map((level) => `
              <article class="cb-level-card">
                <span>第 ${level.level} 格 · ${level.trigger_pct}%</span>
                <strong>${fmtPrice(level.trigger_price)}</strong>
                <p>买入 ${level.buy_parts} 份，持仓增至 ${level.target_total_parts} 份</p>
                <em>${level.condition}</em>
              </article>
            `).join("") || `<div class="empty-state">没有加仓网格</div>`}
          </div>
          <div>
            <h4>止盈网格</h4>
            ${(plan.sell_levels || []).map((level) => `
              <article class="cb-level-card sell">
                <span>第 ${level.level} 档 · +${level.trigger_pct}%</span>
                <strong>${fmtPrice(level.trigger_price)}</strong>
                <p>卖出 ${level.sell_parts} 份，剩余 ${level.target_remaining_parts} 份</p>
                <em>${level.condition}</em>
              </article>
            `).join("") || `<div class="empty-state">没有止盈网格</div>`}
          </div>
          <div>
            <h4>风控</h4>
            ${(plan.risk_controls || []).map((risk) => `
              <article class="cb-level-card risk">
                <span>${risk.name}</span>
                <strong>${risk.trigger_price ? fmtPrice(risk.trigger_price) : "事件触发"}</strong>
                <p>${risk.action}</p>
              </article>
            `).join("") || `<div class="empty-state">没有风控规则</div>`}
          </div>
        </section>
      </section>
    `;
  }).join("");
}

function allotmentStatusClass(stage) {
  if (["registered", "exchange_approved"].includes(stage)) return "ready";
  if (["accepted", "shareholder_approved", "board_plan"].includes(stage)) return "pipeline";
  if (stage === "issuing") return "watching";
  if (stage === "delisted") return "delisted";
  return "listed";
}

const allotmentSortConfig = {
  rights_value_pct: { type: "number", defaultDirection: "desc" },
  kdj_daily_j: { type: "number", defaultDirection: "desc" },
  kdj_weekly_j: { type: "number", defaultDirection: "desc" },
  kdj_monthly_j: { type: "number", defaultDirection: "desc" },
  announce_date: { type: "date", defaultDirection: "asc" },
  record_date: { type: "date", defaultDirection: "asc" },
  pay_date: { type: "date", defaultDirection: "asc" },
  issue_date: { type: "date", defaultDirection: "asc" },
};
const allotmentStageFlow = [
  { stage: "board_plan", status: "董事会预案" },
  { stage: "shareholder_approved", status: "股东大会通过" },
  { stage: "accepted", status: "交易所受理" },
  { stage: "exchange_approved", status: "上市委通过" },
  { stage: "registered", status: "同意注册" },
  { stage: "issuing", status: "发行公告" },
];

function allotmentSortValue(item, key, type) {
  const value = item?.[key];
  if (value === null || value === undefined || value === "") return null;
  if (type === "date") {
    const time = new Date(value).getTime();
    return Number.isFinite(time) ? time : null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function sortedAllotmentRecords(records) {
  const sort = state.cbAllotmentSort;
  if (!sort?.key) return records;
  const config = allotmentSortConfig[sort.key];
  if (!config) return records;
  const direction = sort.direction === "asc" ? 1 : -1;
  return [...records].sort((a, b) => {
    const av = allotmentSortValue(a, sort.key, config.type);
    const bv = allotmentSortValue(b, sort.key, config.type);
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    if (av === bv) return 0;
    return av > bv ? direction : -direction;
  });
}

function allotmentStatusOptions(records) {
  const counts = records.reduce((map, item) => {
    const stage = item.stage || "";
    if (!stage) return map;
    map.set(stage, (map.get(stage) || 0) + 1);
    return map;
  }, new Map());
  const ordered = allotmentStageFlow
    .filter((item) => counts.has(item.stage))
    .map((item) => ({ ...item, count: counts.get(item.stage) || 0 }));
  const configured = new Set(ordered.map((item) => item.stage));
  const extras = [...counts.keys()]
    .filter((stage) => !configured.has(stage))
    .sort()
    .map((stage) => {
      const sample = records.find((item) => item.stage === stage);
      return {
        stage,
        status: sample?.status || stage || "未知状态",
        count: counts.get(stage) || 0,
      };
    });
  return [...ordered, ...extras];
}

function filteredAllotmentRecords(records) {
  const selected = state.cbAllotmentStatusFilters;
  if (!selected?.size) return records;
  return records.filter((item) => selected.has(item.stage));
}

function renderAllotmentStatusFilters(options) {
  const container = document.querySelector("#allotmentStatusFilters");
  if (!container) return;
  const validStages = new Set(options.map((item) => item.stage));
  state.cbAllotmentStatusFilters = new Set(
    [...state.cbAllotmentStatusFilters].filter((stage) => validStages.has(stage))
  );
  const selected = state.cbAllotmentStatusFilters;
  container.innerHTML = `
    <span>状态筛选</span>
    <div class="allotment-status-filter-options">
      <label class="allotment-status-filter-option">
        <input type="checkbox" data-allotment-status-filter="all" ${selected.size === 0 ? "checked" : ""} />
        <span>全部</span>
      </label>
      ${options.map((item) => `
        <label class="allotment-status-filter-option">
          <input type="checkbox" data-allotment-status-filter="${item.stage}" ${selected.has(item.stage) ? "checked" : ""} />
          <span>${item.status}${item.count == null ? "" : ` ${item.count}`}</span>
        </label>
      `).join("")}
    </div>
  `;
  container.querySelectorAll("[data-allotment-status-filter]").forEach((input) => {
    input.addEventListener("change", () => {
      const stage = input.dataset.allotmentStatusFilter;
      if (stage === "all") {
        state.cbAllotmentStatusFilters.clear();
      } else if (input.checked) {
        state.cbAllotmentStatusFilters.add(stage);
      } else {
        state.cbAllotmentStatusFilters.delete(stage);
      }
      renderConvertibleBondAllotments();
    });
  });
}

function bindAllotmentSortHeaders() {
  document.querySelectorAll("[data-allotment-sort]").forEach((button) => {
    const key = button.dataset.allotmentSort;
    const config = allotmentSortConfig[key];
    const active = state.cbAllotmentSort?.key === key;
    const direction = active ? state.cbAllotmentSort.direction : config?.defaultDirection;
    button.classList.toggle("active", active);
    button.setAttribute("aria-sort", active ? (direction === "asc" ? "ascending" : "descending") : "none");
    button.dataset.direction = active ? direction : "";
    button.onclick = () => {
      const current = state.cbAllotmentSort;
      const nextDirection = current?.key === key
        ? (current.direction === "asc" ? "desc" : "asc")
        : config.defaultDirection;
      state.cbAllotmentSort = { key, direction: nextDirection };
      renderConvertibleBondAllotments();
    };
  });
}

function renderConvertibleBondAllotments() {
  const summary = document.querySelector("#allotmentSummary");
  const rows = document.querySelector("#allotmentRows");
  if (!summary || !rows) return;
  const generatedAt = document.querySelector("#generatedAt");
  bindAllotmentSortHeaders();
  if (state.cbAllotmentLoading) {
    if (generatedAt) generatedAt.textContent = "正在加载配债股数据";
    summary.innerHTML = `
      <span>数据状态</span>
      <strong>正在加载</strong>
      <em>正在读取可转债基础资料和发行配售数据</em>
      <button id="cbAllotmentRefreshButton" class="secondary-button" type="button" disabled>刷新中</button>
    `;
    rows.innerHTML = `<tr><td colspan="16" class="empty-cell">正在加载配债股跟踪数据...</td></tr>`;
    return;
  }
  const payload = state.cbAllotmentPayload;
  if (!payload) {
    const errorMessage = state.cbAllotmentError;
    if (generatedAt && state.activePage === "cbAllotment") {
      generatedAt.textContent = errorMessage ? "配债股数据加载失败" : "等待加载配债股数据";
    }
    summary.innerHTML = `
      <span>数据状态</span>
      <strong>${errorMessage ? "加载失败" : "等待加载"}</strong>
      <em>${errorMessage || "切到配债股页面后读取后端数据"}</em>
      <button id="cbAllotmentRefreshButton" class="secondary-button" type="button">${errorMessage ? "重试" : "加载数据"}</button>
    `;
    rows.innerHTML = `<tr><td colspan="16" class="empty-cell">${errorMessage || "切到配债股页面后加载后端数据"}</td></tr>`;
    document.querySelector("#cbAllotmentRefreshButton")?.addEventListener("click", () => {
      loadConvertibleBondAllotments().catch(showError);
    });
    return;
  }
  if (generatedAt && state.activePage === "cbAllotment") generatedAt.textContent = `更新于 ${payload.generated_at || ""}`;
  const source = payload.data_sources || {};
  const basicRows = source.basic?.rows ?? 0;
  const issueRows = source.issue?.rows ?? 0;
  const issueAvailable = Boolean(source.issue?.available);
  const allRecords = payload.records || [];
  const statusOptions = allotmentStatusOptions(allRecords);
  renderAllotmentStatusFilters(statusOptions);
  const filteredRecords = filteredAllotmentRecords(allRecords);
  summary.innerHTML = `
    <span>数据状态</span>
    <strong>${filteredRecords.length} / ${allRecords.length} 条前置阶段</strong>
    <em>六阶段队列 ${source.pipeline?.rows ?? 0} 行 · 发行明细 ${source.cninfo_issue?.rows ?? 0} 行 · cb_issue ${issueAvailable ? `${issueRows} 行` : "未可用"}</em>
    <button id="cbAllotmentRefreshButton" class="secondary-button" type="button">刷新配债数据</button>
  `;
  document.querySelector("#cbAllotmentRefreshButton")?.addEventListener("click", async () => {
    const button = document.querySelector("#cbAllotmentRefreshButton");
    button.disabled = true;
    button.textContent = "刷新中";
    try {
      await loadConvertibleBondAllotments({ refresh: true });
    } catch (error) {
      showError(error);
    }
  });
  const records = sortedAllotmentRecords(filteredRecords);
  if (!records.length) {
    rows.innerHTML = `<tr><td colspan="16" class="empty-cell">当前状态下暂无配债股；可切换状态或点击刷新配债数据</td></tr>`;
    return;
  }
  rows.innerHTML = records.map((item) => `
    <tr data-watchlist-symbol="${item.stock_code || ""}" data-watchlist-name="${item.stock_name || ""}" data-watchlist-note="${escapeHtml(watchlistSourceNote(item.stock_price_date || item.announce_date || payload.asof, `配债股${item.status ? ` · ${item.status}` : ""}`))}" tabindex="0">
      <td><span class="allotment-status ${allotmentStatusClass(item.stage)}">${item.status || "-"}</span></td>
      <td><strong>${item.stock_code || "--"}</strong><em>${item.stock_name || "--"}</em></td>
      <td><strong>${fmtPrice(item.stock_price)}</strong><em>${item.stock_price_date || "--"}</em></td>
      <td>${fmtPrice(item.kdj_daily_j)}</td>
      <td>${fmtPrice(item.kdj_weekly_j)}</td>
      <td>${fmtPrice(item.kdj_monthly_j)}</td>
      <td>${fmtPct(item.rights_value_pct)}</td>
      <td>${item.shares_for_one_lot || item.shares_for_10_bonds || "--"}</td>
      <td>${item.announce_date || "--"}</td>
      <td>${item.record_date || "--"}</td>
      <td>${item.pay_date || "--"}</td>
      <td>${item.issue_date || "--"}</td>
      <td><strong>${item.bond_code || "--"}</strong><em>${item.bond_name || item.announcement_title || "--"} · ${item.rating || "未评级"}</em></td>
      <td><strong>${item.allot_code || "--"}</strong><em>${item.allot_name || "等待公告"}</em></td>
      <td><strong>${item.list_date || "--"}</strong><em>转股 ${item.convert_start_date || "--"} 至 ${item.convert_end_date || "--"}</em></td>
      <td><strong>${item.allotment_note || "--"}</strong><em>${item.risk_note || "--"}</em></td>
    </tr>
  `).join("");
}

function renderBydPage() {
  const payload = state.bydPayload;
  const status = document.querySelector("#bydDataStatus");
  const subline = document.querySelector("#bydSubline");
  const primary = document.querySelector("#bydPrimaryAction");
  const metrics = document.querySelector("#bydPositionMetrics");
  const stageLabel = document.querySelector("#bydStageLabel");
  const ladder = document.querySelector("#bydPriceLadder");
  const intradayZones = document.querySelector("#bydIntradayZones");
  const alerts = document.querySelector("#bydAlerts");
  const alertCount = document.querySelector("#bydAlertCount");
  const playbook = document.querySelector("#bydPlaybook");
  const validationStatus = document.querySelector("#bydValidationStatus");
  const validationDecision = document.querySelector("#bydValidationDecision");
  const validationMetrics = document.querySelector("#bydValidationMetrics");
  if (!primary || !metrics || !ladder || !alerts || !playbook) return;
  if (state.bydLoading) {
    if (status) status.textContent = "正在刷新日线计划...";
    primary.innerHTML = `<div class="empty-state">正在生成 BYD 日线做T计划...</div>`;
    metrics.innerHTML = "";
    ladder.innerHTML = "";
    if (intradayZones) intradayZones.innerHTML = "";
    alerts.innerHTML = "";
    playbook.innerHTML = "";
    if (validationStatus) validationStatus.textContent = "正在读取回测结论";
    if (validationMetrics) validationMetrics.innerHTML = "";
    return;
  }
  if (!payload) {
    if (status) status.textContent = "等待加载";
    primary.innerHTML = `<div class="empty-state">切到 BYD 做T页面后加载日线计划</div>`;
    metrics.innerHTML = "";
    ladder.innerHTML = "";
    if (intradayZones) intradayZones.innerHTML = "";
    alerts.innerHTML = "";
    playbook.innerHTML = "";
    if (validationStatus) validationStatus.textContent = "等待加载";
    if (validationMetrics) validationMetrics.innerHTML = "";
    return;
  }
  const holding = payload.holding || {};
  const minute = payload.minute || {};
  const plan = payload.daily_t_plan || payload.planned_t || {};
  const positive = plan.positive || {};
  const reverse = plan.reverse || {};
  const inventory = plan.inventory || {};
  const action = payload.primary_action || {};
  const validation = payload.validation || plan.validation || {};
  if (status) status.textContent = `${payload.data_status || "-"} · ${payload.generated_at || ""}`;
  if (stageLabel) stageLabel.textContent = payload.stage?.label || "-";
  if (subline) {
    subline.textContent = `${holding.shares || 0} 股 · 成本 ${fmtPrice(holding.cost)} · 日线收盘 ${fmtPrice(plan.reference_close || minute.last)} · 数据日 ${plan.signal_date || minute.asof || ""}`;
  }
  if (validationStatus) {
    validationStatus.textContent = `${validation.label || "未提供验证结果"} · ${validation.period || ""}`;
    validationStatus.classList.toggle("positive", Boolean(validation.execution_enabled));
    validationStatus.classList.toggle("negative", validation.execution_enabled === false);
  }
  if (validationDecision) validationDecision.textContent = validation.decision || "没有验证结论，不启用新开T仓。";
  if (validationMetrics) {
    validationMetrics.innerHTML = (validation.held_out_results || []).map((item) => `
      <article>
        <span>${item.name || "候选"}</span>
        <strong class="${Number(item.net_pnl || 0) >= 0 ? "positive" : "negative"}">${fmtMoney(item.net_pnl)}</strong>
        <em>${item.cycles || 0} 笔 · 胜率 ${fmtPct(Number(item.win_rate || 0) * 100, 1)} · 盈利因子 ${item.profit_factor == null ? "∞（无亏损）" : Number(item.profit_factor).toFixed(2)}</em>
      </article>
    `).join("");
  }
  const actionClass = positive.execution_enabled ? "triggered" : "waiting";
  primary.innerHTML = `
    <div class="byd-action-card ${actionClass}">
      <span>${action.action || "-"}</span>
      <strong>${action.title || "-"}</strong>
      <p>${action.detail || ""}</p>
      <div>
        <em>建议股数 ${Number(action.shares_delta || 0) > 0 ? "+" : ""}${action.shares_delta || 0}</em>
        <em>买入 ${fmtPrice(positive.buy_price)}</em>
        <em>目标 ${fmtPrice(positive.target_price)}</em>
      </div>
    </div>
  `;
  metrics.innerHTML = `
    <div><span>持仓</span><strong>${holding.shares || 0} 股</strong></div>
    <div><span>库存比例</span><strong>${fmtPct((holding.inventory_ratio || 0) * 100, 1)}</strong></div>
    <div><span>浮动盈亏</span><strong class="${Number(holding.unrealized_pnl || 0) < 0 ? "negative" : "positive"}">${fmtMoney(holding.unrealized_pnl)}</strong></div>
    <div><span>盈亏率</span><strong class="${Number(holding.unrealized_pnl_pct || 0) < 0 ? "negative" : "positive"}">${fmtPct(holding.unrealized_pnl_pct, 2)}</strong></div>
    <div><span>合理收盘仓</span><strong>${inventory.reasonable_min_shares || 8000}-${inventory.full_shares || 10000} 股</strong></div>
    <div><span>盘中上限</span><strong>${inventory.intraday_max_shares || 12000} 股</strong></div>
  `;
  if (intradayZones) {
    intradayZones.innerHTML = `
      <article class="${positive.execution_enabled ? "buyback-zone" : "no-sell-zone"}">
        <span>正T优先 · ${positive.status || "等待"}</span>
        <strong>买 ${fmtPrice(positive.buy_price)} → 卖 ${fmtPrice(positive.target_price)}</strong>
        <em>${positive.entry_rule || "买点不到不买"} ${positive.exit_rule || "目标不到尾盘退出"}</em>
      </article>
      <article class="no-sell-zone">
        <span>反T · ${reverse.status || "暂停"}</span>
        <strong>观察卖 ${fmtPrice(reverse.sell_price)} → 买回 ${fmtPrice(reverse.buyback_price)}</strong>
        <em>${reverse.reason || "反T未通过验证，不执行"}</em>
      </article>
      <article class="basis-zone">
        <span>计划口径</span>
        <strong>${plan.signal_date || "-"} 收盘后生成</strong>
        <em>${plan.basis || "盘前固定，盘中不更新"}</em>
      </article>
      <article class="basis-zone">
        <span>仓位处理放在T计划之后</span>
        <strong>${holding.shares || 0} / ${inventory.full_shares || 10000} 股</strong>
        <em>${inventory.note || "完成T后再处理仓位偏离"}</em>
      </article>
    `;
  }
  ladder.innerHTML = `
    <div><span>日线参考收盘</span><strong>${fmtPrice(plan.reference_close)}</strong><em>${plan.signal_date || "-"}</em></div>
    <div><span>正T买入点</span><strong>${fmtPrice(positive.buy_price)}</strong><em>到价买 ${positive.shares || 0} 股；不到不买</em></div>
    <div><span>正T目标价</span><strong>${fmtPrice(positive.target_price)}</strong><em>成交后立即准备卖出同等股数</em></div>
    <div><span>正T兜底退出</span><strong>14:50 后</strong><em>目标未到则直接卖出完成T</em></div>
    <div><span>反T观察卖点</span><strong>${fmtPrice(reverse.sell_price)}</strong><em>当前不执行</em></div>
    <div><span>反T观察买回</span><strong>${fmtPrice(reverse.buyback_price)}</strong><em>未通过盈利闸门</em></div>
  `;
  if (alertCount) alertCount.textContent = positive.execution_enabled ? "1 条正T可执行" : "今日等待";
  alerts.innerHTML = [positive, reverse].map((item, index) => `
    <article class="byd-alert ${item.execution_enabled ? "triggered" : "research-only"}">
      <div>
        <span>${item.execution_enabled ? "可执行" : "观察"}</span>
        <strong>${index === 0 ? "正T计划" : "反T计划"} · ${item.status || "-"}</strong>
        <p>${index === 0 ? `${item.entry_rule || ""} ${item.exit_rule || ""}` : item.reason || ""}</p>
      </div>
      <aside>
        <strong>${index === 0 ? fmtPrice(item.buy_price) : fmtPrice(item.sell_price)}</strong>
        <em>${item.shares || 0} 股</em>
      </aside>
    </article>
  `).join("");
  playbook.innerHTML = (payload.playbook || []).map((item) => `<li>${item}</li>`).join("");
}

function render() {
  renderPageShell();
  if (state.activePage === "short") {
    renderShortPage();
  } else if (state.activePage === "long") {
    renderCalendar();
    renderDateStatus();
    renderLongOverview();
    renderLongStrategies();
    renderLongStockPool();
  } else if (state.activePage === "chan") {
    renderCalendar();
    renderDateStatus();
    renderChanModelPage();
  } else if (state.activePage === "cb") {
    renderCalendar();
    renderDateStatus();
    renderConvertibleBondPage();
  } else if (state.activePage === "cbAllotment") {
    renderConvertibleBondAllotments();
  } else if (state.activePage === "byd") {
    renderBydPage();
  } else if (state.activePage === "similar") {
    renderSimilarPatternsPage();
  }
}

function showError(error) {
  document.querySelector("#detailTitle").textContent = error.message;
  setRefreshMessage(error.message);
}

document.querySelector("#searchInput").addEventListener("input", (event) => {
  state.query = event.target.value;
  const rows = filteredStocks();
  if (rows.length && !rows.some((item) => item.symbol === state.selectedSymbol)) {
    state.selectedSymbol = rows[0].symbol;
  }
  renderStockRows();
  renderStockDetail();
  renderHeader();
});

document.querySelector("#clearFilters").addEventListener("click", async () => {
  state.selectedStrategies.clear();
  state.query = "";
  document.querySelector("#searchInput").value = "";
  await loadSelector().catch(showError);
});

document.querySelector("#signalDateInput").addEventListener("click", () => {
  state.calendarOpen = true;
  renderCalendar();
});

document.querySelector("#calendarToggle").addEventListener("click", (event) => {
  event.stopPropagation();
  state.calendarOpen = !state.calendarOpen;
  renderCalendar();
});

document.querySelector("#prevMonthButton").addEventListener("click", (event) => {
  event.stopPropagation();
  state.calendarMonth = monthOffset(state.calendarMonth || state.signalDate.slice(0, 7), -1);
  renderCalendar();
});

document.querySelector("#nextMonthButton").addEventListener("click", (event) => {
  event.stopPropagation();
  state.calendarMonth = monthOffset(state.calendarMonth || state.signalDate.slice(0, 7), 1);
  renderCalendar();
});

document.addEventListener("click", (event) => {
  const control = document.querySelector("#sharedDateControl");
  if (!state.calendarOpen || control?.contains(event.target)) return;
  state.calendarOpen = false;
  renderCalendar();
});

document.querySelector("#reloadButton").addEventListener("click", async () => {
  const button = document.querySelector("#reloadButton");
  button.disabled = true;
  button.textContent = "刷新中";
  const latestRefreshRunning = state.latestRefreshStatus?.status === "running" || state.latestRefreshStatus?.status === "queued";
  try {
    if (!latestRefreshRunning) {
      setRefreshMessage("正在重算当前日期股票池，并写入快照缓存...");
    }
    if (state.activePage === "long") {
      await loadLongStockPool({ refresh: true });
    } else if (state.activePage === "chan") {
      await loadChanModelStrategy({ refresh: true });
    } else {
      await loadSelector({ refresh: true });
    }
    if (!latestRefreshRunning) {
      setRefreshMessage("当前日期股票池已重算完成");
    } else {
      setRefreshStatus(state.latestRefreshStatus);
    }
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新当前日期";
  }
});

function setRefreshStatus(status) {
  const statusEls = document.querySelectorAll(".refresh-status");
  const stepsEls = document.querySelectorAll(".progress-steps");
  if (!status) {
    state.latestRefreshStatus = null;
    statusEls.forEach((el) => { el.textContent = ""; });
    stepsEls.forEach((el) => { el.innerHTML = ""; });
    localStorage.removeItem(REFRESH_STATUS_STORAGE_KEY);
    return;
  }
  state.latestRefreshStatus = status;
  const time = status.finished_at || status.started_at || "";
  const percent = Number(status.percent || 0);
  const scopeLabel = status.scope_label || REFRESH_SCOPE_LABELS[status.scope] || "";
  localStorage.setItem(REFRESH_STATUS_STORAGE_KEY, JSON.stringify(status));
  const statusHtml = `
    <div class="refresh-status-line">
      <span>${scopeLabel ? `${scopeLabel} · ` : ""}${status.message || status.status}${time ? ` · ${time}` : ""}</span>
      <strong>${percent}%</strong>
    </div>
    <div class="refresh-progress-track">
      <div class="refresh-progress-fill ${status.status === "running" || status.status === "queued" ? "active" : ""}" style="width: ${Math.max(0, Math.min(100, percent))}%"></div>
    </div>
  `;
  const stepsHtml = (status.steps || []).map((step) => `
    <span class="progress-step ${step.status || "pending"}">${step.label}</span>
  `).join("");
  statusEls.forEach((el) => { el.innerHTML = statusHtml; });
  stepsEls.forEach((el) => { el.innerHTML = stepsHtml; });
}

function setRefreshButtonRunning(isRunning) {
  document.querySelectorAll("#refreshAllButton, [data-refresh-scope]").forEach((button) => {
    button.disabled = isRunning;
    button.textContent = isRunning ? "更新中" : (REFRESH_BUTTON_LABELS[button.id] || "更新本页");
  });
}

async function reloadAfterRefresh(status) {
  const scope = status?.scope || "all";
  const shouldReloadShort = scope === "all" || scope === "short";
  const shouldReloadChan = scope === "all" || scope === "chan";
  const shouldReloadLong = scope === "all" || scope === "long";
  const shouldReloadCb = scope === "all" || scope === "cb";
  const shouldReloadAllotment = scope === "all" || scope === "cbAllotment";
  const shouldReloadByd = scope === "all" || scope === "byd";

  if (shouldReloadShort) {
    state.signalDate = "";
    state.selectedSymbol = null;
    await loadCalendar().catch(showError);
  }
  if (shouldReloadChan) state.chanPayload = null;
  if (shouldReloadLong) state.longPayload = null;
  if (shouldReloadCb) state.cbPayload = null;
  if (shouldReloadAllotment) state.cbAllotmentPayload = null;
  if (shouldReloadByd) state.bydPayload = null;

  if (state.activePage === "long" && shouldReloadLong) {
    await loadLongStockPool();
  } else if (state.activePage === "chan" && shouldReloadChan) {
    await loadChanModelStrategy();
  } else if (state.activePage === "cb" && shouldReloadCb) {
    await loadConvertibleBondPlan();
  } else if (state.activePage === "cbAllotment" && shouldReloadAllotment) {
    await loadConvertibleBondAllotments();
  } else if (state.activePage === "byd" && shouldReloadByd) {
    await loadBydMinuteStrategy();
  } else if (state.activePage === "short" && shouldReloadShort) {
    await loadSelector();
  }
}

function startRefreshPolling() {
  if (state.refreshPollId) clearInterval(state.refreshPollId);
  state.refreshPollId = setInterval(() => {
    pollLatestRefresh().catch(showError);
  }, 5000);
}

function stopRefreshPolling() {
  if (state.refreshPollId) clearInterval(state.refreshPollId);
  state.refreshPollId = null;
}

async function pollLatestRefresh() {
  const status = await fetchJson("/selector/refresh-latest/status");
  setRefreshStatus(status);
  if (status.status === "success") {
    stopRefreshPolling();
    await reloadAfterRefresh(status);
    setRefreshButtonRunning(false);
  } else if (status.status === "failed") {
    stopRefreshPolling();
    setRefreshButtonRunning(false);
    showError(new Error(status.error || "刷新任务失败"));
  }
}

async function restoreLatestRefreshStatus() {
  const cached = localStorage.getItem(REFRESH_STATUS_STORAGE_KEY);
  if (cached) {
    try {
      const cachedStatus = JSON.parse(cached);
      setRefreshStatus(cachedStatus);
      if (cachedStatus.status === "running" || cachedStatus.status === "queued") {
        setRefreshButtonRunning(true);
        startRefreshPolling();
      }
    } catch {
      localStorage.removeItem(REFRESH_STATUS_STORAGE_KEY);
    }
  }
  const status = await fetchJson("/selector/refresh-latest/status");
  setRefreshStatus(status);
  if (status.status === "running" || status.status === "queued") {
    setRefreshButtonRunning(true);
    startRefreshPolling();
  } else if (status.status === "success" || status.status === "failed" || status.status === "idle") {
    setRefreshButtonRunning(false);
  }
}

async function startLatestDataRefresh(scope = "all") {
  setRefreshButtonRunning(true);
  try {
    const status = await fetchJson("/selector/refresh-latest", {
      method: "POST",
      body: JSON.stringify({ scope }),
    });
    setRefreshStatus(status);
    startRefreshPolling();
    await pollLatestRefresh();
  } catch (error) {
    setRefreshButtonRunning(false);
    showError(error);
  }
}

document.querySelector("#refreshAllButton")?.addEventListener("click", () => {
  startLatestDataRefresh("all");
});

ensureWorkspaceTabs();

document.querySelectorAll("[data-refresh-scope]").forEach((button) => {
  button.addEventListener("click", () => {
    startLatestDataRefresh(button.dataset.refreshScope || "all");
  });
});

document.querySelectorAll(".page-tab").forEach((button) => {
  button.addEventListener("click", (event) => {
    if (suppressWorkspaceTabClick) {
      event.preventDefault();
      suppressWorkspaceTabClick = false;
      return;
    }
    hideWatchlistContextMenu();
    const nextPage = button.dataset.page || "short";
    const nextHash = pageHash(nextPage);
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
      return;
    }
    state.activePage = nextPage;
    renderPageShell();
    loadActivePageData();
  });
});

document.querySelectorAll(".workspace-tabs").forEach((nav) => {
  nav.addEventListener("dragstart", (event) => {
    const button = event.target.closest(".page-tab");
    if (!button) return;
    workspaceTabDragKey = button.dataset.page || "";
    button.classList.add("dragging");
    nav.classList.add("is-reordering");
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", workspaceTabDragKey);
    }
  });

  nav.addEventListener("dragover", (event) => {
    const target = event.target.closest(".page-tab");
    if (!target || !workspaceTabDragKey || target.dataset.page === workspaceTabDragKey) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
    clearWorkspaceTabDropTargets();
    const placeAfter = event.clientX >= target.getBoundingClientRect().left + target.getBoundingClientRect().width / 2;
    target.classList.add(placeAfter ? "tab-drop-after" : "tab-drop-before");
  });

  nav.addEventListener("dragleave", (event) => {
    if (!nav.contains(event.relatedTarget)) clearWorkspaceTabDropTargets();
  });

  nav.addEventListener("drop", (event) => {
    const target = event.target.closest(".page-tab");
    if (!target || !workspaceTabDragKey) return;
    event.preventDefault();
    const placeAfter = event.clientX >= target.getBoundingClientRect().left + target.getBoundingClientRect().width / 2;
    const moved = reorderWorkspaceTab(workspaceTabDragKey, target.dataset.page || "", placeAfter);
    suppressWorkspaceTabClick = moved;
    if (moved) window.setTimeout(() => { suppressWorkspaceTabClick = false; }, 250);
    clearWorkspaceTabDropTargets();
  });

  nav.addEventListener("dragend", () => {
    document.querySelectorAll(".page-tab.dragging").forEach((button) => button.classList.remove("dragging"));
    document.querySelectorAll(".workspace-tabs.is-reordering").forEach((item) => item.classList.remove("is-reordering"));
    clearWorkspaceTabDropTargets();
    workspaceTabDragKey = "";
  });

  nav.addEventListener("keydown", (event) => {
    if (!event.target.matches(".page-tab")) return;
    if (event.altKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault();
      moveWorkspaceTabByOffset(event.target.dataset.page || "", event.key === "ArrowLeft" ? -1 : 1);
      return;
    }
    const tabs = [...nav.querySelectorAll(".page-tab")];
    const currentIndex = tabs.indexOf(event.target);
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % tabs.length;
    if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = tabs.length - 1;
    if (nextIndex === currentIndex) return;
    event.preventDefault();
    focusWorkspaceTabAfterRender = true;
    tabs[nextIndex].click();
  });
});

window.addEventListener("hashchange", () => {
  hideWatchlistContextMenu();
  state.activePage = hashPage();
  renderPageShell();
  loadActivePageData();
});

document.querySelector("#chanRefreshButton")?.addEventListener("click", async () => {
  const button = document.querySelector("#chanRefreshButton");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadChanModelStrategy({ refresh: true });
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新计划";
  }
});

document.querySelector("#cbRefreshButton")?.addEventListener("click", async () => {
  const button = document.querySelector("#cbRefreshButton");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadConvertibleBondPlan({ refresh: true });
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新计划";
  }
});

document.querySelector("#cbAllotmentRefreshButton")?.addEventListener("click", async () => {
  const button = document.querySelector("#cbAllotmentRefreshButton");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadConvertibleBondAllotments({ refresh: true });
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新配债数据";
  }
});

document.querySelectorAll(".long-variant-button").forEach((button) => {
  button.addEventListener("click", () => {
    state.longVariant = button.dataset.longVariant || "tea";
    state.longPayload = null;
    renderLongOverview();
    renderLongStrategies();
    renderLongStockPool();
    loadLongStockPool().catch(showError);
  });
});

document.querySelector("#longPoolRefresh").addEventListener("click", async () => {
  const button = document.querySelector("#longPoolRefresh");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadLongStockPool({ refresh: true });
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新股票池";
  }
});

document.querySelector("#bydRefreshButton")?.addEventListener("click", async () => {
  if (!saveBydHoldingInputs()) return;
  const button = document.querySelector("#bydRefreshButton");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadBydMinuteStrategy({ refresh: true });
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新日线计划";
  }
});

document.querySelector("#bydTradeToast")?.addEventListener("click", (event) => {
  const action = event.target?.dataset?.bydToastAction;
  if (!action) return;
  dismissBydTradeToast(action === "seen");
});

BYD_HOLDING_INPUT_IDS.map((id) => document.querySelector(`#${id}`)).filter(Boolean).forEach((input) => {
  input.addEventListener("change", () => {
    if (!saveBydHoldingInputs()) return;
    if (state.activePage === "byd") {
      loadBydMinuteStrategy().catch(showError);
    }
  });
});

let watchlistContextTarget = null;
let watchlistToastTimer = null;
let similarWatchDragSymbol = "";
let suppressSimilarWatchRowClick = false;

function hideWatchlistContextMenu() {
  const menu = document.querySelector("#watchlistContextMenu");
  if (menu) menu.hidden = true;
  watchlistContextTarget = null;
}

function showWatchlistToast(message, tone = "success") {
  const toast = document.querySelector("#watchlistToast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.toggle("error", tone === "error");
  toast.classList.add("show");
  window.clearTimeout(watchlistToastTimer);
  watchlistToastTimer = window.setTimeout(() => toast.classList.remove("show"), 2400);
}

function openWatchlistContextMenu(target, clientX, clientY) {
  const menu = document.querySelector("#watchlistContextMenu");
  if (!menu || !target?.dataset.watchlistSymbol) return;
  watchlistContextTarget = {
    symbol: target.dataset.watchlistSymbol,
    name: target.dataset.watchlistName || target.dataset.watchlistSymbol,
    note: target.dataset.watchlistNote || "",
    inWatchlist: Boolean(target.closest("#similarWatchlist")),
    pinned: target.dataset.watchlistPinned === "true",
  };
  const addButton = menu.querySelector("[data-watchlist-context-add]");
  const pinButton = menu.querySelector("[data-watchlist-context-pin]");
  if (addButton) addButton.hidden = watchlistContextTarget.inWatchlist;
  if (pinButton) {
    pinButton.hidden = !watchlistContextTarget.inWatchlist;
    pinButton.textContent = watchlistContextTarget.pinned ? "取消置顶" : "置顶";
  }
  menu.hidden = false;
  const width = menu.offsetWidth || 150;
  const height = menu.offsetHeight || 44;
  menu.style.left = `${Math.max(8, Math.min(clientX, window.innerWidth - width - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(clientY, window.innerHeight - height - 8))}px`;
  menu.querySelector("button:not([hidden])")?.focus();
}

document.addEventListener("contextmenu", (event) => {
  const target = event.target.closest("[data-watchlist-symbol]");
  const canAdd = ["short", "chan", "long", "cbAllotment"].includes(state.activePage);
  const canManage = state.activePage === "similar" && Boolean(target?.closest("#similarWatchlist"));
  if (!target || (!canAdd && !canManage)) return;
  event.preventDefault();
  openWatchlistContextMenu(target, event.clientX, event.clientY);
});

document.addEventListener("keydown", (event) => {
  const target = event.target.closest?.("[data-watchlist-symbol]");
  if (target && (event.key === "ContextMenu" || (event.shiftKey && event.key === "F10"))) {
    event.preventDefault();
    const rect = target.getBoundingClientRect();
    openWatchlistContextMenu(target, rect.left + 24, rect.top + 24);
    return;
  }
  if (event.key === "Escape") hideWatchlistContextMenu();
});

document.querySelector("[data-watchlist-context-add]")?.addEventListener("click", async () => {
  const target = watchlistContextTarget;
  if (!target) return;
  const button = document.querySelector("[data-watchlist-context-add]");
  button.disabled = true;
  try {
    await addSimilarWatchSymbol(target.symbol, { refresh: false, note: target.note });
    showWatchlistToast(`${target.name} 已加入自选池${target.note ? "并记录来源" : ""}`);
  } catch (error) {
    showWatchlistToast(error.message || "加入自选池失败", "error");
  } finally {
    button.disabled = false;
    hideWatchlistContextMenu();
  }
});

document.querySelector("[data-watchlist-context-pin]")?.addEventListener("click", async () => {
  const target = watchlistContextTarget;
  if (!target?.inWatchlist) return;
  const button = document.querySelector("[data-watchlist-context-pin]");
  button.disabled = true;
  try {
    await setSimilarWatchPinned(target.symbol, !target.pinned);
    showWatchlistToast(`${target.name} 已${target.pinned ? "取消置顶" : "置顶"}`);
  } catch (error) {
    showWatchlistToast(error.message || "置顶状态保存失败", "error");
  } finally {
    button.disabled = false;
    hideWatchlistContextMenu();
  }
});

document.addEventListener("click", (event) => {
  if (!event.target.closest("#watchlistContextMenu")) hideWatchlistContextMenu();
});
window.addEventListener("resize", hideWatchlistContextMenu);
window.addEventListener("scroll", hideWatchlistContextMenu, true);

document.querySelector("#similarAddForm")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.querySelector("#similarSymbolInput");
  const symbol = input?.value?.trim();
  if (!symbol) return;
  const button = event.submitter;
  if (button) button.disabled = true;
  try {
    await addSimilarWatchSymbol(symbol);
    if (input) input.value = "";
  } catch (error) {
    showError(error);
  } finally {
    if (button) button.disabled = false;
  }
});

document.querySelector("#similarRefreshButton")?.addEventListener("click", async () => {
  const button = document.querySelector("#similarRefreshButton");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadSimilarPatterns({ refresh: true });
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新分析";
  }
});

function openSimilarNoteDialog(symbol) {
  const item = (state.similarPayload?.watchlist || []).find((row) => row.symbol === symbol);
  const dialog = document.querySelector("#similarNoteDialog");
  const form = document.querySelector("#similarNoteForm");
  const textarea = document.querySelector("#similarNoteInput");
  if (!item || !dialog || !form || !textarea) return;
  form.dataset.similarNoteSymbol = symbol;
  document.querySelector("#similarNoteTitle").textContent = `${item.name || symbol} · 笔记与操作计划`;
  document.querySelector("#similarNoteMeta").textContent = item.note_updated_at
    ? `${symbol} · 上次保存 ${item.note_updated_at}`
    : `${symbol} · 尚未保存笔记`;
  textarea.value = item.note || "";
  dialog.showModal();
  window.setTimeout(() => textarea.focus(), 0);
}

function closeSimilarNoteDialog() {
  const dialog = document.querySelector("#similarNoteDialog");
  if (dialog?.open) dialog.close();
  const form = document.querySelector("#similarNoteForm");
  if (form) delete form.dataset.similarNoteSymbol;
}

document.querySelector("#similarNoteCancel")?.addEventListener("click", closeSimilarNoteDialog);
document.querySelector("#similarNoteCancelBottom")?.addEventListener("click", closeSimilarNoteDialog);
document.querySelector("#similarNoteDialog")?.addEventListener("click", (event) => {
  if (event.target === event.currentTarget) closeSimilarNoteDialog();
});
document.querySelector("#similarNoteForm")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const symbol = event.currentTarget.dataset.similarNoteSymbol;
  if (!symbol) return;
  const button = document.querySelector("#similarNoteSave");
  const content = document.querySelector("#similarNoteInput")?.value || "";
  button.disabled = true;
  button.textContent = "保存中";
  try {
    await saveSimilarWatchNote(symbol, content);
    closeSimilarNoteDialog();
    showWatchlistToast(content.trim() ? "笔记已保存" : "笔记已清空");
  } catch (error) {
    showWatchlistToast(error.message || "笔记保存失败", "error");
  } finally {
    button.disabled = false;
    button.textContent = "保存笔记";
  }
});

document.querySelector("#similarPage")?.addEventListener("click", (event) => {
  if (event.target.closest("[data-similar-xueqiu]")) return;
  const retryButton = event.target.closest("[data-similar-retry]");
  if (retryButton) {
    loadSimilarPatterns().catch(showError);
    return;
  }
  const removeButton = event.target.closest("[data-similar-remove]");
  if (removeButton) {
    removeSimilarWatchSymbol(removeButton.dataset.similarRemove).catch(showError);
    return;
  }
  const noteButton = event.target.closest("[data-similar-note]");
  if (noteButton) {
    openSimilarNoteDialog(noteButton.dataset.similarNote);
    return;
  }
  const card = event.target.closest("[data-similar-symbol]");
  if (!card) return;
  if (suppressSimilarWatchRowClick) {
    suppressSimilarWatchRowClick = false;
    return;
  }
  state.similarSelectedSymbol = card.dataset.similarSymbol;
  renderSimilarPatternsPage();
});

function clearSimilarWatchDropTargets() {
  document.querySelectorAll(".similar-watch-row.watch-drop-before, .similar-watch-row.watch-drop-after").forEach((row) => {
    row.classList.remove("watch-drop-before", "watch-drop-after");
  });
}

function reorderSimilarWatchRows(sourceSymbol, targetSymbol, placeAfter = false) {
  const watch = state.similarPayload?.watchlist || [];
  const source = watch.find((item) => item.symbol === sourceSymbol);
  const target = watch.find((item) => item.symbol === targetSymbol);
  if (!source || !target || sourceSymbol === targetSymbol) return false;
  if (Boolean(source.pinned) !== Boolean(target.pinned)) {
    showWatchlistToast("置顶股票和普通股票请在各自区域内排序", "error");
    return false;
  }
  const order = watch.map((item) => item.symbol).filter((symbol) => symbol !== sourceSymbol);
  const targetIndex = order.indexOf(targetSymbol);
  order.splice(targetIndex + (placeAfter ? 1 : 0), 0, sourceSymbol);
  suppressSimilarWatchRowClick = true;
  saveSimilarWatchOrder(order)
    .then(() => showWatchlistToast("自选池顺序已保存"))
    .catch((error) => showWatchlistToast(error.message || "自选池排序保存失败", "error"));
  window.setTimeout(() => { suppressSimilarWatchRowClick = false; }, 300);
  return true;
}

document.querySelector("#similarWatchlist")?.addEventListener("dragstart", (event) => {
  const row = event.target.closest(".similar-watch-row");
  if (!row) return;
  similarWatchDragSymbol = row.dataset.similarSymbol || "";
  row.classList.add("dragging");
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", similarWatchDragSymbol);
  }
});

document.querySelector("#similarWatchlist")?.addEventListener("dragover", (event) => {
  const row = event.target.closest(".similar-watch-row");
  if (!row || !similarWatchDragSymbol || row.dataset.similarSymbol === similarWatchDragSymbol) return;
  event.preventDefault();
  if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
  clearSimilarWatchDropTargets();
  const placeAfter = event.clientY >= row.getBoundingClientRect().top + row.getBoundingClientRect().height / 2;
  row.classList.add(placeAfter ? "watch-drop-after" : "watch-drop-before");
});

document.querySelector("#similarWatchlist")?.addEventListener("drop", (event) => {
  const row = event.target.closest(".similar-watch-row");
  if (!row || !similarWatchDragSymbol) return;
  event.preventDefault();
  const placeAfter = event.clientY >= row.getBoundingClientRect().top + row.getBoundingClientRect().height / 2;
  reorderSimilarWatchRows(similarWatchDragSymbol, row.dataset.similarSymbol || "", placeAfter);
  clearSimilarWatchDropTargets();
});

document.querySelector("#similarWatchlist")?.addEventListener("dragend", () => {
  document.querySelectorAll(".similar-watch-row.dragging").forEach((row) => row.classList.remove("dragging"));
  clearSimilarWatchDropTargets();
  similarWatchDragSymbol = "";
});

document.querySelector("#similarWatchlist")?.addEventListener("keydown", (event) => {
  const row = event.target.closest(".similar-watch-row");
  if (!row || !event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
  const rows = [...document.querySelectorAll("#similarWatchlist .similar-watch-row")];
  const currentIndex = rows.indexOf(row);
  const nextIndex = currentIndex + (event.key === "ArrowUp" ? -1 : 1);
  const target = rows[nextIndex];
  if (!target) return;
  event.preventDefault();
  const moved = reorderSimilarWatchRows(
    row.dataset.similarSymbol || "",
    target.dataset.similarSymbol || "",
    event.key === "ArrowDown",
  );
  if (moved) window.setTimeout(() => document.querySelector(`#similarWatchlist [data-similar-symbol="${row.dataset.similarSymbol}"]`)?.focus(), 0);
});

document.querySelector("#similarPage")?.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest("[data-similar-symbol]");
  if (!row || event.target.closest("button, input, a")) return;
  event.preventDefault();
  state.similarSelectedSymbol = row.dataset.similarSymbol;
  renderSimilarPatternsPage();
});

document.querySelector("#similarPage")?.addEventListener("mouseover", (event) => {
  const row = event.target.closest("[data-similar-symbol]");
  if (!row || row.contains(event.relatedTarget)) return;
  showSimilarNoteTooltip(row, event.clientX, event.clientY);
});

document.querySelector("#similarPage")?.addEventListener("mousemove", (event) => {
  if (!event.target.closest("[data-similar-symbol]")) return;
  positionSimilarNoteTooltip(event.clientX, event.clientY);
});

document.querySelector("#similarPage")?.addEventListener("mouseout", (event) => {
  const row = event.target.closest("[data-similar-symbol]");
  if (!row || row.contains(event.relatedTarget)) return;
  hideSimilarNoteTooltip(row);
});

document.querySelector("#similarPage")?.addEventListener("focusin", (event) => {
  const row = event.target.closest("[data-similar-symbol]");
  if (!row) return;
  const rect = row.getBoundingClientRect();
  showSimilarNoteTooltip(row, rect.left + Math.min(rect.width / 2, 280), rect.top + 24);
});

document.querySelector("#similarPage")?.addEventListener("focusout", (event) => {
  const row = event.target.closest("[data-similar-symbol]");
  if (!row || row.contains(event.relatedTarget)) return;
  hideSimilarNoteTooltip(row);
});

document.querySelector("#chanPage")?.addEventListener("click", (event) => {
  const row = event.target.closest("[data-chan-symbol]");
  if (!row) return;
  state.chanSelectedSymbol = row.dataset.chanSymbol;
  renderChanModelPage();
});

const initialDateInput = document.querySelector("#signalDateInput");
if (initialDateInput) initialDateInput.value = "";
state.signalDate = "";
restoreBydHoldingInputs();
renderPageShell();
renderLongStrategies();
renderLongStockPool();
restoreLatestRefreshStatus().catch(showError);
loadCalendar().catch(showError);
loadActivePageData();
