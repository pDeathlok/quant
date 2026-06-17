const state = {
  payload: null,
  activePage: window.location.hash === "#long" ? "long" : window.location.hash === "#byd" ? "byd" : window.location.hash === "#cb-allotment" ? "cbAllotment" : window.location.hash === "#cb" ? "cb" : "short",
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
  bydPayload: null,
  bydLoading: false,
  bydLastNoticeKey: "",
  cbPayload: null,
  cbLoading: false,
  selectedCbStrategy: null,
  selectedCbCode: null,
  cbAllotmentPayload: null,
  cbAllotmentLoading: false,
  cbAllotmentSort: null,
  cbAllotmentStatusFilters: new Set(),
  loading: false,
};

const API_BASE = "/api";
const REFRESH_STATUS_STORAGE_KEY = "quant.selector.latestRefreshStatus";

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
const pageHash = (page) => page === "long" ? "#long" : page === "byd" ? "#byd" : page === "cbAllotment" ? "#cb-allotment" : page === "cb" ? "#cb" : "#short";
const currentLongStrategy = () => longStrategies.find((item) => item.key === state.longVariant) || longStrategies[0];

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
  const response = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
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

async function loadSelector(options = {}) {
  state.loading = true;
  render();
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
    state.payload = await fetchJson(path);
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
    render();
  }
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
    const payload = await fetchJson(`/long/stock-pool?${query.toString()}`);
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
  const shares = Number(document.querySelector("#bydSharesInput")?.value || 10500);
  const cost = Number(document.querySelector("#bydCostInput")?.value || 110.6061);
  const soldTodayShares = Number(document.querySelector("#bydSoldTodaySharesInput")?.value || 0);
  const soldTodayPrice = Number(document.querySelector("#bydSoldTodayPriceInput")?.value || 0);
  const openTShares = Number(document.querySelector("#bydOpenTInput")?.value || 0);
  const openTPrice = Number(document.querySelector("#bydOpenTPriceInput")?.value || 0);
  const query = new URLSearchParams();
  query.set("shares", String(shares));
  query.set("cost", String(cost));
  if (soldTodayShares > 0) query.set("sold_today_shares", String(soldTodayShares));
  if (soldTodayPrice > 0) query.set("sold_today_price", String(soldTodayPrice));
  if (openTShares > 0) query.set("open_t_shares", String(openTShares));
  if (openTPrice > 0) query.set("open_t_price", String(openTPrice));
  if (options.refresh) query.set("refresh", "true");
  try {
    state.bydPayload = await fetchJson(`/byd/daily-plan?${query.toString()}`);
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
    state.cbPayload = await fetchJson(`/convertible-bonds/plan?${query.toString()}`);
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
  renderConvertibleBondAllotments();
  const query = new URLSearchParams();
  query.set("limit", "120");
  query.set("include_listed_days", "180");
  query.set("stage_scope", "pipeline");
  if (options.refresh) query.set("refresh", "true");
  try {
    state.cbAllotmentPayload = await fetchJson(`/convertible-bonds/allotments?${query.toString()}`);
  } catch (error) {
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
  state.calendar = await fetchJson("/selector/calendar?start=2020-01-01");
  if (!state.signalDate && state.calendar.latest_signal_date) {
    state.signalDate = state.calendar.latest_signal_date;
    const input = document.querySelector("#signalDateInput");
    if (input) input.value = state.signalDate;
  }
  if (!state.calendarMonth) {
    state.calendarMonth = (state.signalDate || state.calendar.latest_signal_date || "").slice(0, 7);
  }
  renderCalendar();
  renderDateStatus();
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
    return `
      <button
        class="calendar-day ${day.status} ${active ? "selected" : ""}"
        data-date="${day.date}"
        type="button"
        ${day.disabled ? "disabled" : ""}
        title="${day.date} · ${day.label}${day.effective_signal_date && day.effective_signal_date !== day.date ? ` · 实际使用 ${day.effective_signal_date}` : ""}"
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
      : state.activePage === "cb"
      ? `${selected} 可转债将回看最近可用行情`
      : `${selected} 为交易日，但暂未生成策略快照；将回看 ${day.effective_signal_date || "最近可用日"}`;
  } else if (day.status === "closed") {
    control.classList.add("date-closed");
    input.classList.add("date-closed");
    statusEl.textContent = state.activePage === "long"
      ? `${selected} 为休市日，长线将回看最近可用交易日`
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
      <button class="filter-chip ${active ? "active" : ""}" data-strategy="${item.key}" type="button">
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
    <tr class="${item.symbol === state.selectedSymbol ? "selected-row" : ""}" data-symbol="${item.symbol}">
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
          <strong>${Number(item.opportunity_score ?? item.selector_score ?? 0).toFixed(2)}</strong>
          <em>${item.score_band || ""} ${item.score_percentile_label || ""}</em>
        </span>
      </td>
      <td>
        <span class="score-stack">
          <strong>${Number(item.holding_score ?? 0).toFixed(2)}</strong>
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
  meta.textContent = `行业 ${stock.industry || "-"} · 收盘 ${fmtPrice(stock.close)} · 冲高分 ${Number(stock.opportunity_score ?? stock.selector_score ?? 0).toFixed(2)}（${stock.score_band || "-"}，${stock.score_percentile_label || "-"}） · 持有参考 ${Number(stock.holding_score ?? 0).toFixed(2)} · ${stock.score_usage_hint || ""} · ${stock.score_risk_note || ""} · 命中 ${stock.matched_count} 个策略组 · ${stock.matched_families.join(" / ")} · ${stock.rank_reason || ""}`;
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
  const shortPage = document.querySelector("#shortPage");
  const longPage = document.querySelector("#longPage");
  const bydPage = document.querySelector("#bydPage");
  const cbPage = document.querySelector("#cbPage");
  const cbAllotmentPage = document.querySelector("#cbAllotmentPage");
  const filterSection = document.querySelector("#strategyFilterSection");
  const longStrategySection = document.querySelector("#longStrategySection");
  const cbStrategySection = document.querySelector("#cbStrategySection");
  const clearButton = document.querySelector("#clearFilters");
  const dateControl = document.querySelector("#sharedDateControl");
  const longDateSlot = document.querySelector("#longDateSlot");
  const cbDateSlot = document.querySelector("#cbDateSlot");
  const shortActions = document.querySelector("#searchInput")?.parentElement;
  const searchInput = document.querySelector("#searchInput");
  shortPage?.classList.toggle("active", state.activePage === "short");
  longPage?.classList.toggle("active", state.activePage === "long");
  bydPage?.classList.toggle("active", state.activePage === "byd");
  cbPage?.classList.toggle("active", state.activePage === "cb");
  cbAllotmentPage?.classList.toggle("active", state.activePage === "cbAllotment");
  filterSection?.classList.toggle("hidden", state.activePage !== "short");
  clearButton?.classList.toggle("hidden", state.activePage !== "short");
  longStrategySection?.classList.toggle("hidden", state.activePage !== "long");
  cbStrategySection?.classList.toggle("hidden", state.activePage !== "cb");
  if (dateControl && longDateSlot && cbDateSlot && shortActions && searchInput) {
    if (state.activePage === "long") {
      longDateSlot.appendChild(dateControl);
    } else if (state.activePage === "cb") {
      cbDateSlot.appendChild(dateControl);
    } else if (dateControl.parentElement !== shortActions) {
      shortActions.insertBefore(dateControl, searchInput);
    }
  }
  document.querySelectorAll(".page-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === state.activePage);
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
    button.classList.toggle("active", button.dataset.longVariant === state.longVariant);
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
    <tr>
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
    meta.textContent = "正在生成可转债计划...";
    metrics.innerHTML = "";
    if (strategyWrap) strategyWrap.innerHTML = "";
    rows.innerHTML = `<tr><td colspan="11" class="empty-cell">正在按低位网格策略筛选可转债...</td></tr>`;
    detail.innerHTML = `<div class="empty-state">正在生成分批买入和网格计划...</div>`;
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
        <button class="cb-strategy-card ${activeStrategyKey === "all" ? "active" : ""}" type="button" data-cb-strategy="all">
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
        <button class="cb-strategy-card ${active ? "active" : ""}" type="button" data-cb-strategy="${strategy.key || ""}">
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
  meta.textContent = `${strategyName} · 信号日 ${payload.trade_date || rootPayload.trade_date} · ${market.entry_permission || "-"}`;
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
    rows.innerHTML = `<tr><td colspan="16" class="empty-cell">切到配债股页面后加载后端数据</td></tr>`;
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
    <tr>
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
  if (!primary || !metrics || !ladder || !alerts || !playbook) return;
  if (state.bydLoading) {
    if (status) status.textContent = "正在刷新日线计划...";
    primary.innerHTML = `<div class="empty-state">正在生成 BYD 日线做T计划...</div>`;
    metrics.innerHTML = "";
    ladder.innerHTML = "";
    if (intradayZones) intradayZones.innerHTML = "";
    alerts.innerHTML = "";
    playbook.innerHTML = "";
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
    return;
  }
  const holding = payload.holding || {};
  const minute = payload.minute || {};
  const levels = payload.daily_levels || {};
  const intraday = payload.intraday_levels || {};
  const plan = payload.planned_t || {};
  const dynamic = payload.intraday_dynamic || {};
  const action = payload.primary_action || {};
  const triggered = (payload.alerts || []).filter((item) => item.triggered);
  if (status) status.textContent = `${payload.data_status || "-"} · ${payload.generated_at || ""}`;
  if (stageLabel) stageLabel.textContent = payload.stage?.label || "-";
  if (subline) {
    subline.textContent = `${holding.shares || 0} 股 · 成本 ${fmtPrice(holding.cost)} · 最新 ${fmtPrice(minute.last)} · ${minute.asof || ""}`;
  }
  const actionClass = triggered.length ? "triggered" : "waiting";
  primary.innerHTML = `
    <div class="byd-action-card ${actionClass}">
      <span>${action.action || "-"}</span>
      <strong>${action.title || "-"}</strong>
      <p>${action.detail || ""}</p>
      <div>
        <em>建议股数 ${Number(action.shares_delta || 0) > 0 ? "+" : ""}${action.shares_delta || 0}</em>
        <em>箱体位置 ${fmtPct(payload.range_position_pct, 1)}</em>
        <em>日线计划</em>
      </div>
    </div>
  `;
  metrics.innerHTML = `
    <div><span>持仓</span><strong>${holding.shares || 0} 股</strong></div>
    <div><span>库存比例</span><strong>${fmtPct((holding.inventory_ratio || 0) * 100, 1)}</strong></div>
    <div><span>浮动盈亏</span><strong class="${Number(holding.unrealized_pnl || 0) < 0 ? "negative" : "positive"}">${fmtMoney(holding.unrealized_pnl)}</strong></div>
    <div><span>盈亏率</span><strong class="${Number(holding.unrealized_pnl_pct || 0) < 0 ? "negative" : "positive"}">${fmtPct(holding.unrealized_pnl_pct, 2)}</strong></div>
    <div><span>阶段目标</span><strong>${payload.stage?.goal_shares || "-"} 股</strong></div>
    <div><span>模式</span><strong>${payload.stage?.mode || "-"}</strong></div>
  `;
  if (intradayZones) {
    const sellZones = plan.sell_zones || [];
    const buybackZones = plan.buyback_zones || [];
    if (sellZones.length) {
      intradayZones.innerHTML = `
        <article class="basis-zone">
          <span>计划依据</span>
          <strong>${fmtPrice(plan.reference_close)}</strong>
          <em>ATR ${fmtPrice(plan.atr14)} · ${fmtPct(plan.atr14_pct, 2)}</em>
        </article>
        <article class="no-sell-zone">
          <span>${plan.no_sell_zone?.label || "低位不追卖区"}</span>
          <strong>${fmtRange(plan.no_sell_zone?.range)}</strong>
          <em>${plan.no_sell_zone?.action || "等待反弹"}</em>
        </article>
        ${sellZones.map((zone) => `
          <article class="sell-zone">
            <span>${zone.label}</span>
            <strong>${fmtRange(zone.range)}</strong>
            <em>${Number(zone.shares) > 0 ? "+" : ""}${zone.shares} 股 · ${zone.condition}</em>
          </article>
        `).join("")}
        ${buybackZones.map((zone) => `
          <article class="buyback-zone">
            <span>${zone.label}</span>
            <strong>${fmtRange(zone.buy_range)}</strong>
            <em>买回 ${zone.shares} 股 · 对应卖出 ${fmtRange(zone.sell_range)}</em>
          </article>
        `).join("")}
        <article class="risk-zone">
          <span>${plan.risk_zone?.label || "下破风控区"}</span>
          <strong>${fmtRange(plan.risk_zone?.range)}</strong>
          <em>${plan.risk_zone?.action || "风控减仓"}</em>
        </article>
      `;
    } else if (dynamic.available) {
      intradayZones.innerHTML = `
        <article>
          <span>日内状态</span>
          <strong>${dynamic.state || "-"}</strong>
          <em>区间位置 ${fmtPct(dynamic.current_position_pct, 1)}</em>
        </article>
        <article>
          <span>今日低位区</span>
          <strong>${fmtRange(dynamic.low_zone)}</strong>
          <em>低位不追卖，只等反抽</em>
        </article>
        <article>
          <span>反抽卖出一档</span>
          <strong>${fmtPrice(dynamic.rebound_sell_1)}</strong>
          <em>从低点反抽 38.2%</em>
        </article>
        <article>
          <span>反抽卖出二档</span>
          <strong>${fmtPrice(dynamic.rebound_sell_2)}</strong>
          <em>从低点反抽 50%</em>
        </article>
        <article>
          <span>反抽卖出三档</span>
          <strong>${fmtPrice(dynamic.rebound_sell_3)}</strong>
          <em>从低点反抽 61.8%</em>
        </article>
        <article>
          <span>今日高位区</span>
          <strong>${fmtRange(dynamic.high_zone)}</strong>
          <em>靠近这里优先降仓</em>
        </article>
      `;
    } else {
      intradayZones.innerHTML = `
        <article>
          <span>计划区间</span>
          <strong>暂无计划数据</strong>
          <em>当前只能使用前复权日线兜底价位</em>
        </article>
      `;
    }
  }
  ladder.innerHTML = `
    <div><span>下破风险线</span><strong>${fmtPrice(levels.support_break)}</strong><em>跌破卖 1000-1500 股</em></div>
    <div><span>箱体下沿</span><strong>${fmtPrice(levels.range_low)}</strong><em>满仓不买，降仓后才买回</em></div>
    <div><span>兜底一档</span><strong>${fmtPrice(intraday.micro_sell_1)}</strong><em>无计划数据时使用</em></div>
    <div><span>兜底二档</span><strong>${fmtPrice(intraday.micro_sell_2)}</strong><em>无计划数据时使用</em></div>
    <div><span>兜底三档</span><strong>${fmtPrice(intraday.micro_sell_3)}</strong><em>无计划数据时使用</em></div>
    <div><span>中位减仓线</span><strong>${fmtPrice(levels.mid_trim)}</strong><em>高仓先卖 300-500 股</em></div>
    <div><span>中上沿线</span><strong>${fmtPrice(levels.weak_trim)}</strong><em>动能转弱卖 400-700 股</em></div>
    <div><span>强减仓线</span><strong>${fmtPrice(levels.strong_trim)}</strong><em>上沿卖 600-1000 股</em></div>
    <div><span>箱体上沿</span><strong>${fmtPrice(levels.range_high)}</strong><em>保留突破观察，不机械卖飞</em></div>
  `;
  if (alertCount) alertCount.textContent = `${triggered.length} 条触发 / ${(payload.alerts || []).length} 条计划`;
  alerts.innerHTML = (payload.alerts || []).map((item) => `
    <article class="byd-alert ${item.triggered ? "triggered" : ""}">
      <div>
        <span>${item.action}</span>
        <strong>${item.title}</strong>
        <p>${item.detail}</p>
      </div>
      <aside>
        <strong>${fmtRange(item.price_range) || fmtPrice(item.price_line)}</strong>
        <em>${Number(item.shares_delta) > 0 ? "+" : ""}${item.shares_delta} 股</em>
      </aside>
    </article>
  `).join("");
  playbook.innerHTML = (payload.playbook || []).map((item) => `<li>${item}</li>`).join("");
}

function render() {
  renderPageShell();
  if (state.activePage === "short") {
    renderHeader();
    renderStrategyFilters();
    renderStockRows();
    renderStockDetail();
    renderNotes();
  }
  renderCalendar();
  renderDateStatus();
  renderLongOverview();
  renderLongStrategies();
  renderLongStockPool();
  renderConvertibleBondPage();
  renderConvertibleBondAllotments();
  renderBydPage();
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
  localStorage.setItem(REFRESH_STATUS_STORAGE_KEY, JSON.stringify(status));
  const statusHtml = `
    <div class="refresh-status-line">
      <span>${status.message || status.status}${time ? ` · ${time}` : ""}</span>
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
  document.querySelectorAll("#refreshLatestButton, #longRefreshLatestButton").forEach((button) => {
    button.disabled = isRunning;
    button.textContent = isRunning ? "更新中" : "一键更新数据";
  });
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
    state.signalDate = "";
    state.selectedSymbol = null;
    state.longPayload = null;
    await loadCalendar().catch(showError);
    if (state.activePage === "long") {
      await loadLongStockPool();
    } else if (state.activePage === "cb") {
      state.cbPayload = null;
      await loadConvertibleBondPlan();
    } else {
      await loadSelector();
    }
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

async function startLatestDataRefresh() {
  setRefreshButtonRunning(true);
  try {
    const status = await fetchJson("/selector/refresh-latest", { method: "POST" });
    setRefreshStatus(status);
    startRefreshPolling();
    await pollLatestRefresh();
  } catch (error) {
    setRefreshButtonRunning(false);
    showError(error);
  }
}

document.querySelectorAll("#refreshLatestButton, #longRefreshLatestButton").forEach((button) => {
  button.addEventListener("click", startLatestDataRefresh);
});

document.querySelectorAll(".page-tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.activePage = button.dataset.page || "short";
    const nextHash = pageHash(state.activePage);
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
    }
    render();
    if (state.activePage === "long" && !state.longPayload && !state.longLoading) {
      loadLongStockPool().catch(showError);
    } else if (state.activePage === "cb" && !state.cbPayload && !state.cbLoading) {
      loadConvertibleBondPlan().catch(showError);
    } else if (state.activePage === "cbAllotment" && !state.cbAllotmentPayload && !state.cbAllotmentLoading) {
      loadConvertibleBondAllotments().catch(showError);
    } else if (state.activePage === "byd" && !state.bydPayload && !state.bydLoading) {
      loadBydMinuteStrategy().catch(showError);
    } else if (state.activePage === "short" && !state.payload && !state.loading) {
      loadSelector({ latest: true }).catch(showError);
    }
  });
});

window.addEventListener("hashchange", () => {
  state.activePage = window.location.hash === "#long" ? "long" : window.location.hash === "#byd" ? "byd" : window.location.hash === "#cb-allotment" ? "cbAllotment" : window.location.hash === "#cb" ? "cb" : "short";
  render();
  if (state.activePage === "long" && !state.longPayload && !state.longLoading) {
    loadLongStockPool().catch(showError);
  } else if (state.activePage === "cb" && !state.cbPayload && !state.cbLoading) {
    loadConvertibleBondPlan().catch(showError);
  } else if (state.activePage === "cbAllotment" && !state.cbAllotmentPayload && !state.cbAllotmentLoading) {
    loadConvertibleBondAllotments().catch(showError);
  } else if (state.activePage === "byd" && !state.bydPayload && !state.bydLoading) {
    loadBydMinuteStrategy().catch(showError);
  } else if (state.activePage === "short" && !state.payload && !state.loading) {
    loadSelector({ latest: true }).catch(showError);
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

document.querySelectorAll("#bydSharesInput, #bydCostInput, #bydSoldTodaySharesInput, #bydSoldTodayPriceInput, #bydOpenTInput, #bydOpenTPriceInput").forEach((input) => {
  input.addEventListener("change", () => {
    if (state.activePage === "byd") {
      loadBydMinuteStrategy().catch(showError);
    }
  });
});

const initialDateInput = document.querySelector("#signalDateInput");
if (initialDateInput) initialDateInput.value = "";
state.signalDate = "";
renderPageShell();
renderLongStrategies();
renderLongStockPool();
restoreLatestRefreshStatus().catch(showError);
loadCalendar().catch(showError).finally(() => {
  if (state.activePage === "long") {
    loadLongStockPool().catch(showError);
  } else if (state.activePage === "cb") {
    loadConvertibleBondPlan().catch(showError);
  } else if (state.activePage === "cbAllotment") {
    loadConvertibleBondAllotments().catch(showError);
  } else if (state.activePage === "byd") {
    loadBydMinuteStrategy().catch(showError);
  } else {
    loadSelector({ latest: true }).catch(showError);
  }
});
