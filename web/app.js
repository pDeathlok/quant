const state = {
  payload: null,
  selectedStrategies: new Set(),
  selectedSymbol: null,
  query: "",
  signalDate: "",
  refreshPollId: null,
  loading: false,
};

const API_BASE = "/api";

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

async function loadSelector() {
  state.loading = true;
  render();
  const query = new URLSearchParams();
  const params = selectedStrategyParam();
  if (params) query.set("strategies", params);
  if (state.signalDate) query.set("signal_date", state.signalDate);
  const suffix = query.toString();
  const path = suffix ? `/selector/stocks?${suffix}` : "/selector/stocks";
  try {
    state.payload = await fetchJson(path);
  } finally {
    state.loading = false;
  }
  state.signalDate = state.payload.signal_date || state.signalDate;
  const dateInput = document.querySelector("#signalDateInput");
  if (dateInput && state.payload.signal_date) {
    dateInput.value = state.payload.signal_date;
  }
  if (!state.selectedSymbol && state.payload.stocks.length) {
    state.selectedSymbol = state.payload.stocks[0].symbol;
  }
  if (state.selectedSymbol && !state.payload.stocks.some((item) => item.symbol === state.selectedSymbol)) {
    state.selectedSymbol = state.payload.stocks[0]?.symbol || null;
  }
  render();
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
  document.querySelector("#generatedAt").textContent = state.loading ? "正在加载" : (state.payload ? `更新于 ${state.payload.generated_at}` : "加载中");
  document.querySelector("#signalDate").textContent = state.payload ? `${state.payload.signal_date} 收盘后选股` : "";
  document.querySelector("#executionDate").textContent = state.payload?.execution_date || "";
  document.querySelector("#stockCount").textContent = state.loading ? "加载中" : `${filteredStocks().length} 只股票`;
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
    body.innerHTML = `<tr><td colspan="9" class="empty-cell">正在加载股票池...</td></tr>`;
    document.querySelector("#stockCount").textContent = "加载中";
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="9" class="empty-cell">当前筛选条件下没有股票</td></tr>`;
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
      <td>${Number(item.selector_score || 0).toFixed(2)}</td>
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
  meta.textContent = `行业 ${stock.industry || "-"} · 收盘 ${fmtPrice(stock.close)} · 综合分 ${Number(stock.selector_score || 0).toFixed(2)} · 命中 ${stock.matched_count} 个策略 · ${stock.matched_families.join(" / ")} · ${stock.rank_reason || ""}`;
  body.innerHTML = stock.signals.map((signal) => {
    const metrics = signal.metrics || {};
    return `
      <article class="signal-card">
        <div class="signal-card-head">
          <div>
            <span class="tag strong">${signal.strategy_family}</span>
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
            <span>买入观察</span>
            <p>${signal.buy_plan}</p>
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

function render() {
  renderHeader();
  renderStrategyFilters();
  renderStockRows();
  renderStockDetail();
  renderNotes();
}

function showError(error) {
  document.querySelector("#detailTitle").textContent = error.message;
  document.querySelector("#refreshStatus").textContent = error.message;
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

document.querySelector("#signalDateInput").addEventListener("change", async (event) => {
  state.signalDate = event.target.value;
  state.selectedSymbol = null;
  await loadSelector().catch(showError);
});

document.querySelector("#reloadButton").addEventListener("click", async () => {
  const button = document.querySelector("#reloadButton");
  button.disabled = true;
  button.textContent = "刷新中";
  try {
    await loadSelector();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "刷新当前日期";
  }
});

function setRefreshStatus(status) {
  const el = document.querySelector("#refreshStatus");
  if (!status) {
    el.textContent = "";
    return;
  }
  const time = status.finished_at || status.started_at || "";
  el.textContent = `${status.message || status.status}${time ? ` · ${time}` : ""}`;
}

async function pollLatestRefresh() {
  const status = await fetchJson("/selector/refresh-latest/status");
  setRefreshStatus(status);
  if (status.status === "success") {
    clearInterval(state.refreshPollId);
    state.refreshPollId = null;
    state.signalDate = "";
    state.selectedSymbol = null;
    await loadSelector();
    document.querySelector("#refreshLatestButton").disabled = false;
    document.querySelector("#refreshLatestButton").textContent = "更新最新数据";
  } else if (status.status === "failed") {
    clearInterval(state.refreshPollId);
    state.refreshPollId = null;
    document.querySelector("#refreshLatestButton").disabled = false;
    document.querySelector("#refreshLatestButton").textContent = "更新最新数据";
    showError(new Error(status.error || "刷新任务失败"));
  }
}

document.querySelector("#refreshLatestButton").addEventListener("click", async () => {
  const button = document.querySelector("#refreshLatestButton");
  button.disabled = true;
  button.textContent = "更新中";
  try {
    const status = await fetchJson("/selector/refresh-latest", { method: "POST" });
    setRefreshStatus(status);
    if (state.refreshPollId) clearInterval(state.refreshPollId);
    state.refreshPollId = setInterval(() => {
      pollLatestRefresh().catch(showError);
    }, 5000);
    await pollLatestRefresh();
  } catch (error) {
    button.disabled = false;
    button.textContent = "更新最新数据";
    showError(error);
  }
});

loadSelector().catch(showError);
