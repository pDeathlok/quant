from pathlib import Path

from fastapi.testclient import TestClient

from quant.webapp.app import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ENTRY_JS = (PROJECT_ROOT / "web" / "app.js").read_text(encoding="utf-8")
CORE_JS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted((PROJECT_ROOT / "web" / "core").glob("*.js"))
)
APP_JS = f"{APP_ENTRY_JS}\n{CORE_JS}"
STYLES_CSS = (PROJECT_ROOT / "web" / "styles.css").read_text(encoding="utf-8")
INDEX_HTML = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")


def test_frontend_entry_uses_es_modules_for_shared_core() -> None:
    assert 'from "./core/api-client.js"' in APP_ENTRY_JS
    assert 'from "./core/formatters.js"' in APP_ENTRY_JS
    assert 'type="module"' in INDEX_HTML
    assert "export function createApiClient(apiBase)" in CORE_JS


def test_frontend_core_modules_are_served_by_fastapi() -> None:
    client = TestClient(app)

    index_response = client.get("/")
    formatter_response = client.get("/core/formatters.js")
    api_client_response = client.get("/core/api-client.js")

    assert index_response.status_code == 200
    assert 'type="module"' in index_response.text
    assert formatter_response.status_code == 200
    assert "export const fmtPct" in formatter_response.text
    assert api_client_response.status_code == 200
    assert "export function createApiClient" in api_client_response.text


def test_static_assets_use_cache_policy_gzip_and_module_preload() -> None:
    client = TestClient(app)

    index_response = client.get("/", headers={"Accept-Encoding": "identity"})
    plain_app = client.get("/app.js", headers={"Accept-Encoding": "identity"})
    compressed_app = client.get("/app.js", headers={"Accept-Encoding": "gzip"})
    stylesheet = client.get("/styles.css", headers={"Accept-Encoding": "gzip"})

    assert index_response.headers["cache-control"] == "no-cache"
    assert plain_app.headers["cache-control"] == "public, max-age=3600"
    assert stylesheet.headers["cache-control"] == "public, max-age=3600"
    assert "content-encoding" not in plain_app.headers
    assert compressed_app.headers["content-encoding"] == "gzip"
    assert compressed_app.headers["vary"] == "Accept-Encoding"
    assert compressed_app.content == plain_app.content
    assert '<link rel="modulepreload" href="/core/api-client.js" />' in INDEX_HTML
    assert '<link rel="modulepreload" href="/core/formatters.js" />' in INDEX_HTML


def test_workspace_tabs_use_one_eight_item_contract() -> None:
    expected_keys = ["short", "chan", "long", "cb", "cbAllotment", "byd", "similar", "plans"]

    assert "const WORKSPACE_TABS = [" in APP_JS
    assert all(f'key: "{key}"' in APP_JS for key in expected_keys)
    assert "ensureWorkspaceTabs" in APP_JS
    assert "ensureChanTabs" not in APP_JS
    assert "grid-template-columns: repeat(8, minmax(0, 1fr));" in STYLES_CSS


def test_operation_plans_tab_supports_durable_crud_and_horizon_filters() -> None:
    assert 'id="plansPage" class="main page-view"' in INDEX_HTML
    assert 'data-plan-filter="tomorrow"' in INDEX_HTML
    assert 'data-plan-filter="long_term"' in INDEX_HTML
    assert 'id="operationPlanForm"' in INDEX_HTML
    assert 'fetchJson("/operation-plans")' in APP_JS
    assert 'method: id ? "PUT" : "POST"' in APP_JS
    assert 'method: "DELETE"' in APP_JS


def test_operation_plans_support_editable_persistent_checklists() -> None:
    assert 'id="operationPlanChecklist"' in INDEX_HTML
    assert 'id="operationPlanChecklistAdd"' in INDEX_HTML
    assert "function renderOperationPlanChecklistEditor(items = [])" in APP_JS
    assert 'data-operation-plan-checklist-toggle' in APP_JS
    assert 'checklist: operationPlanChecklistItemsFromForm()' in APP_JS
    assert 'class="operation-plan-checklist"' in APP_JS
    assert ".operation-plan-checklist-item.is-completed" in STYLES_CSS


def test_new_tomorrow_operation_plan_defaults_to_next_local_date() -> None:
    assert "function nextDayDateInputValue(referenceDate = new Date())" in APP_JS
    assert "nextDay.setDate(nextDay.getDate() + 1);" in APP_JS
    assert "return localDateInputValue(nextDay);" in APP_JS
    assert 'dateInput.dataset.defaulted = "true";' in APP_JS
    assert "setDefaultOperationPlanDate();" in APP_JS
    assert 'document.querySelector("#operationPlanHorizon")?.addEventListener("change"' in APP_JS
    assert 'if (event.currentTarget.value === "tomorrow")' in APP_JS
    assert "if (!dateInput.value) setDefaultOperationPlanDate();" in APP_JS
    assert APP_JS.rindex("resetOperationPlanForm();") < APP_JS.rindex("loadActivePageData();")


def test_refresh_status_uses_one_global_surface_with_readable_progress() -> None:
    assert INDEX_HTML.count('class="refresh-status') == 1
    assert INDEX_HTML.count('class="progress-steps') == 1
    assert "function formatRefreshStatusMessage(status, scopeLabel" in APP_JS
    assert "正在扫描相似走势" in APP_JS
    assert "${statusMessage}" in APP_JS
    assert "20260803-refresh-status-v1" in INDEX_HTML


def test_watchlist_is_default_workspace_for_root_url() -> None:
    assert 'const DEFAULT_WORKSPACE_PAGE = "similar";' in APP_JS
    assert "activePage: workspacePageFromHash(window.location.hash)" in APP_JS
    assert 'id="shortPage" class="main page-view"' in INDEX_HTML
    assert 'id="similarPage" class="main page-view active"' in INDEX_HTML
    assert "default-similar-v1" in INDEX_HTML


def test_workspace_tabs_expose_accessible_keyboard_state() -> None:
    assert 'role="tab"' in APP_JS
    assert 'aria-selected="false"' in APP_JS
    assert 'button.setAttribute("aria-selected", String(active));' in APP_JS
    assert 'event.key === "ArrowRight"' in APP_JS
    assert 'event.key === "ArrowLeft"' in APP_JS
    assert 'event.key === "Home"' in APP_JS
    assert 'event.key === "End"' in APP_JS
    assert "focusWorkspaceTabAfterRender" in APP_JS


def test_workspace_tabs_support_persistent_reordering() -> None:
    assert 'const WORKSPACE_TAB_ORDER_STORAGE_KEY = "quant.workspaceTabOrder.v1"' in APP_JS
    assert "function normalizeWorkspaceTabOrder(value)" in APP_JS
    assert "function reorderWorkspaceTab(sourceKey, targetKey, placeAfter = false)" in APP_JS
    assert "localStorage.setItem(WORKSPACE_TAB_ORDER_STORAGE_KEY" in APP_JS
    assert 'draggable="true"' in APP_JS
    assert 'event.altKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")' in APP_JS
    assert '.workspace-tabs .page-tab[draggable="true"]' in STYLES_CSS
    assert "20260720-allotment-divider-v10" in INDEX_HTML
    assert "20260720-watchlist-reorder-v5" in INDEX_HTML


def test_workspace_tabs_scroll_consistently_on_narrow_screens() -> None:
    assert "grid-auto-flow: column;" in STYLES_CSS
    assert "grid-auto-columns: minmax(148px, 1fr);" in STYLES_CSS
    assert "overflow-x: auto;" in STYLES_CSS
    assert "scroll-snap-type: inline proximity;" in STYLES_CSS
    assert "prefers-reduced-motion: reduce" in STYLES_CSS


def test_workspace_grid_children_can_shrink_on_mobile() -> None:
    assert ".main > * {" in STYLES_CSS
    shrink_rule = STYLES_CSS.split(".main > * {", 1)[1].split("}", 1)[0]
    assert "min-width: 0;" in shrink_rule
    assert ".chan-performance {\n    grid-template-columns: 1fr;\n  }" in STYLES_CSS
    assert ".chan-performance-item {\n    border-left: 0;" in STYLES_CSS
    assert "20260730-grid-shrink-v2" in INDEX_HTML


def test_long_workspace_exposes_blood_chip_daily_iteration_as_sub_strategy() -> None:
    assert 'data-long-variant="blood_chip"' in INDEX_HTML
    assert 'id="bloodChipLongContent"' in INDEX_HTML
    assert 'id="bloodChipCandidateRows"' in INDEX_HTML
    assert 'id="bloodChipPositionRows"' in INDEX_HTML
    assert 'id="bloodChipIteration"' in INDEX_HTML
    assert 'class="long-variant-mobile-control"' in INDEX_HTML
    assert 'requestedVariant === "blood_chip"' in APP_JS
    assert '`/long/blood-chip?${query.toString()}`' in APP_JS
    assert "function renderBloodChipLongPlan()" in APP_JS
    assert ".blood-chip-summary-grid" in STYLES_CSS
    assert "首仓 20%" in INDEX_HTML
    assert "第二段 30%" in INDEX_HTML
    assert "第三段 50%" in INDEX_HTML
    assert "row.stage_label" in APP_JS
    assert "row.next_addition_fraction" in APP_JS
    assert "iteration.advanced_positions" in APP_JS
    assert "iteration.ready_additions" in APP_JS
    assert "fmtRate(row.deployed_fraction" in APP_JS
    assert "fmtRate(row.next_addition_fraction" in APP_JS
    assert "fmtRate(row.current_residual_return_3d" in APP_JS
    assert "fmtRate(validation.total_return" in APP_JS
    assert ".long-variant-mobile-control" in STYLES_CSS


def test_blood_chip_tables_link_each_stock_to_xueqiu_in_a_new_tab() -> None:
    assert "function bloodChipXueqiuLink(row)" in APP_JS
    assert "xueqiuStockUrl(row.ts_code)" in APP_JS
    assert 'data-blood-chip-xueqiu' in APP_JS
    assert 'target="_blank" rel="noopener noreferrer"' in APP_JS
    assert "雪球 ↗" in APP_JS
    assert "20260810-blood-chip-xueqiu-v1" in INDEX_HTML


def test_short_strategy_stocks_link_to_xueqiu_in_a_new_tab() -> None:
    assert "xueqiuStockUrl(item.symbol)" in APP_JS
    assert 'data-short-xueqiu' in APP_JS
    assert 'target="_blank" rel="noopener noreferrer"' in APP_JS
    assert "雪球 ↗" in APP_JS
    assert "<th>雪球</th>" in INDEX_HTML
    assert "20260811-short-xueqiu-v1" in INDEX_HTML


def test_chan_mobile_toolbar_keeps_refresh_buttons_readable() -> None:
    assert ".chan-toolbar .toolbar-actions {\n    display: grid;" in STYLES_CSS
    assert ".chan-toolbar #chanDateSlot {\n    grid-column: 1 / -1;" in STYLES_CSS
    assert ".chan-toolbar .toolbar-actions > button {\n    margin-top: 0;" in STYLES_CSS
    assert "20260730-mobile-toolbar-v1" in INDEX_HTML


def test_watchlist_stocks_have_persistent_note_editor() -> None:
    assert 'data-similar-note="${item.symbol}"' in APP_JS
    assert 'id="similarNoteDialog"' in INDEX_HTML
    assert 'id="similarNoteInput"' in INDEX_HTML
    assert "/note`, {" in APP_JS
    assert 'method: "PUT"' in APP_JS
    assert 'await fetchJson("/similar-patterns/watchlist?include_scores=false")' in APP_JS
    assert "分析加载失败，笔记仍可编辑" in APP_JS
    assert ".similar-note-dialog" in STYLES_CSS


def test_watchlist_mutations_keep_previous_analysis_during_background_refresh() -> None:
    remove_function = APP_JS.split("async function removeSimilarWatchSymbol(symbol) {", 1)[1].split(
        "async function saveSimilarWatchNote",
        1,
    )[0]

    assert "similarRefreshPromise: null" in APP_JS
    assert "similarPendingRemovals: new Set()" in APP_JS
    assert "similarOrderSaving: false" in APP_JS
    assert "function mergeSimilarPayloadWithWatchlist" in APP_JS
    assert "function mergeWatchlistProfiles(currentWatchlist, incomingWatchlist)" in APP_JS
    assert "function enrichWatchlistProfiles(currentWatchlist, enrichedWatchlist)" in APP_JS
    assert "if (state.similarPendingRemovals.has(symbol)) return;" in APP_JS
    assert 'showWatchlistToast(`${displayName} 已从自选池删除`);' in APP_JS
    assert "loadSimilarPatterns" not in remove_function
    assert "/similar-patterns/analysis?refresh=true" not in APP_JS
    assert 'data-refresh-scope="similar"' in INDEX_HTML
    assert 'const shouldReloadSimilar = scope === "all" || scope === "similar";' in APP_JS
    assert "state.similarPayload = null;" not in APP_JS


def test_watchlist_add_starts_debounced_background_analysis_and_score_refresh() -> None:
    assert 'await fetchJson("/similar-patterns/watchlist?include_scores=false")' in APP_JS
    add_function = APP_JS.split("async function addSimilarWatchSymbol(symbol, options = {}) {", 1)[1].split(
        "async function removeSimilarWatchSymbol",
        1,
    )[0]
    assert "loadSimilarPatterns" not in add_function
    assert "scheduleSimilarAnalysisRefresh();" in add_function
    assert 'fetchJson("/similar-patterns/watchlist?include_scores=true")' in APP_JS
    assert 'startLatestDataRefresh("similar")' in APP_JS
    assert "similarAutoRefreshPending" in APP_JS
    assert "评分与分析正在后台更新" in APP_JS
    assert "正在后台更新分析，当前结果继续保留" in APP_JS
    assert 'previousStatus !== status.status || previousScope !== status.scope' in APP_JS
    assert "if (addButton) addButton.disabled = state.similarLoading;" not in APP_JS


def test_watchlist_new_stock_selection_does_not_fall_back_to_another_stock_result() -> None:
    selected_result = APP_JS.split("function selectedSimilarResult() {", 1)[1].split(
        "function similarResultForSymbol", 1
    )[0]
    loader = APP_JS.split("async function loadSimilarPatternsOnce() {", 1)[1].split(
        "function loadSimilarPatterns()", 1
    )[0]

    assert "if (!state.similarSelectedSymbol) return null;" in selected_result
    assert "|| results[0] || null" not in selected_result
    assert "watchlistSymbols.has(state.similarSelectedSymbol)" in loader
    assert "selectedWatchItem.name || selectedWatchItem.symbol" in APP_JS
    assert "分析结果待更新" in APP_JS
    assert "等待后台分析结果" in APP_JS


def test_selector_filters_coalesce_rapid_clicks_and_ignore_stale_responses() -> None:
    assert "selectorFilterReloadTimer = window.setTimeout" in APP_JS
    assert "}, 150);" in APP_JS
    assert "const requestId = ++state.selectorRequestId;" in APP_JS
    assert "if (requestId !== state.selectorRequestId) return;" in APP_JS


def test_direct_refreshes_and_watchlist_order_are_single_flight() -> None:
    assert "options.refresh ? 0 : 15000" in APP_JS
    assert "function runDirectWorkspaceRefresh(key, operation)" in APP_JS
    assert 'runDirectWorkspaceRefresh("chan"' in APP_JS
    assert 'runDirectWorkspaceRefresh("cb"' in APP_JS
    assert 'runDirectWorkspaceRefresh("long"' in APP_JS
    assert 'runDirectWorkspaceRefresh("byd"' in APP_JS
    assert "if (state.similarOrderSaving)" in APP_JS


def test_strategy_watchlist_add_includes_default_source_note() -> None:
    assert "function compactWatchlistDate(value)" in APP_JS
    assert "function watchlistSourceNote(dateValue, sourceText)" in APP_JS
    assert 'body: JSON.stringify({ symbol, note: options.note || "" })' in APP_JS
    assert 'data-watchlist-note=' in APP_JS
    assert 'note: target.dataset.watchlistNote || ""' in APP_JS
    assert 'note: target.note' in APP_JS
    assert "触发 ${(item.matched_families || []).join(\" / \")} 策略" in APP_JS
    assert "配债股${item.status ?" in APP_JS


def test_allotment_watchlist_note_includes_one_lot_shares_and_rights_value() -> None:
    assert "function allotmentWatchlistSourceNote(item, payload)" in APP_JS
    assert "const oneLotShares = item.shares_for_one_lot ?? item.shares_for_10_bonds;" in APP_JS
    assert "`一手股数 ${oneLotText}`" in APP_JS
    assert '`含权量 ${rightsValueText === "-" ? "待计算" : rightsValueText}`' in APP_JS
    assert 'data-watchlist-note="${escapeHtml(allotmentWatchlistSourceNote(item, payload))}"' in APP_JS
    assert "20260731-allotment-watchlist-note-v1" in INDEX_HTML


def test_allotment_kdj_values_use_signed_number_formatter() -> None:
    assert "const fmtNumber = (value, digits = 2) => {" in APP_JS
    assert "return Number.isFinite(numeric) ? numeric.toFixed(digits) : \"-\";" in APP_JS
    assert "${fmtNumber(item.kdj_daily_j)}" in APP_JS
    assert "${fmtNumber(item.kdj_weekly_j)}" in APP_JS
    assert "${fmtNumber(item.kdj_monthly_j)}" in APP_JS
    assert "${fmtPrice(item.kdj_weekly_j)}" not in APP_JS
    assert "${fmtPrice(item.kdj_monthly_j)}" not in APP_JS


def test_allotment_refresh_updates_market_inputs_and_exposes_quality() -> None:
    assert 'cbAllotmentRefreshButton: "更新行情与配债"' in APP_JS
    assert 'startLatestDataRefresh("cbAllotment");' in APP_JS
    assert "loadConvertibleBondAllotments({ refresh: true })" not in APP_JS
    loader = APP_JS.split("async function loadConvertibleBondAllotments", 1)[1].split(
        "function activeCbPlan", 1
    )[0]
    assert 'query.set("limit"' not in loader
    assert 'query.set("include_listed_days"' not in loader
    assert 'query.set("stage_scope", "pipeline")' in loader
    assert "20260809-allotment-cache-contract-v1" in INDEX_HTML
    assert "qualityMetrics.stock_daily_match" in APP_JS
    assert "qualityMetrics.kdj_weekly_j" in APP_JS
    assert "qualityMetrics.kdj_monthly_j" in APP_JS
    assert "20260730-allotment-refresh-quality-v1" in INDEX_HTML


def test_allotment_stocks_link_to_xueqiu_in_a_new_tab() -> None:
    assert "function allotmentStockCell(item)" in APP_JS
    assert "xueqiuStockUrl(stockCode)" in APP_JS
    assert 'class="allotment-xueqiu-link"' in APP_JS
    assert "data-allotment-xueqiu" in APP_JS
    assert 'target="_blank"' in APP_JS
    assert 'rel="noopener noreferrer"' in APP_JS
    assert "雪球 ↗" in APP_JS
    assert ".allotment-xueqiu-link" in STYLES_CSS
    assert "20260730-allotment-xueqiu-v1" in INDEX_HTML


def test_xueqiu_market_prefix_handles_beijing_920_codes_first() -> None:
    assert '/^(?:920|[48])/.test(code) ? "BJ"' in APP_JS
    assert 'https://xueqiu.com/S/${market}${code}' in APP_JS
    assert "20260730-xueqiu-bj920-v2" in INDEX_HTML


def test_watchlist_note_is_visible_on_stock_hover_and_keyboard_focus() -> None:
    assert 'id="similarNoteTooltip"' in INDEX_HTML
    assert 'role="tooltip"' in INDEX_HTML
    assert 'addEventListener("mouseover"' in APP_JS
    assert 'addEventListener("mousemove"' in APP_JS
    assert 'addEventListener("focusin"' in APP_JS
    assert 'row.setAttribute("aria-describedby", "similarNoteTooltip")' in APP_JS
    assert ".similar-note-tooltip" in STYLES_CSS


def test_watchlist_stocks_link_to_xueqiu_in_a_new_tab() -> None:
    assert "function xueqiuStockUrl(symbol)" in APP_JS
    assert "https://xueqiu.com/S/${market}${code}" in APP_JS
    assert 'data-similar-xueqiu' in APP_JS
    assert 'target="_blank"' in APP_JS
    assert 'rel="noopener noreferrer"' in APP_JS
    assert 'event.target.closest("button, input, a")' in APP_JS
    assert "<th>雪球</th>" in INDEX_HTML
    assert ".xueqiu-stock-link" in STYLES_CSS


def test_watchlist_reuses_selector_buy_and_hold_scores() -> None:
    assert "<th>买入分</th>" in INDEX_HTML
    assert "<th>持有分</th>" in INDEX_HTML
    assert "item.opportunity_score ?? item.buy_score" in APP_JS
    assert "item.holding_score ?? item.hold_score" in APP_JS
    assert 'class="similar-score-cell"' in APP_JS
    assert 'colspan="11"' in INDEX_HTML


def test_watchlist_rows_support_persistent_drag_order_and_pin_menu() -> None:
    assert 'await fetchJson("/similar-patterns/watchlist/order", {' in APP_JS
    assert '/pin`, {' in APP_JS
    assert 'data-watchlist-pinned="${item.pinned ? "true" : "false"}"' in APP_JS
    assert 'title="按住拖动排序；右键可置顶、设置提醒或删除"' in APP_JS
    assert "function reorderSimilarWatchRows(sourceSymbol, targetSymbol, placeAfter = false)" in APP_JS
    assert 'event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)' in APP_JS
    assert "data-watchlist-context-pin" in INDEX_HTML
    assert "data-watchlist-context-delete" in INDEX_HTML
    assert "watchlist-context-danger" in INDEX_HTML
    assert 'document.querySelector("[data-watchlist-context-delete]")' in APP_JS
    assert "watchlist-fast-delete-v1" in INDEX_HTML
    assert ".similar-stock-cell .watchlist-pin-badge" in STYLES_CSS
    assert "20260720-watchlist-reorder-v5" in INDEX_HTML


def test_watchlist_context_menu_opens_multi_condition_alert_dialog() -> None:
    assert "data-watchlist-context-alert" in INDEX_HTML
    assert 'id="watchlistAlertDialog"' in INDEX_HTML
    assert 'id="watchlistAlertReminders"' in INDEX_HTML
    assert "data-watchlist-alert-add-reminder" in INDEX_HTML
    assert "data-alert-reminder-note" in APP_JS
    assert "data-alert-condition-kind" in APP_JS
    assert "data-alert-condition-conjunction" in APP_JS
    assert "data-alert-condition-operator" in APP_JS
    assert "data-alert-condition-add" in APP_JS
    assert "AND · 并且" in APP_JS
    assert "OR · 或者" in APP_JS
    assert "大于 ＞" in APP_JS
    assert "等于 ＝" in APP_JS
    assert "小于 ＜" in APP_JS
    assert 'kdj_daily_j: { label: "日线J", unit: "", source: "snapshot" }' in APP_JS
    assert "/alerts`, {" in APP_JS
    assert "function evaluateWatchlistAlert(item" in APP_JS
    assert "watchlistAlertBell(item, alertState)" in APP_JS
    assert "data-watchlist-alert-bell" in APP_JS
    assert "openWatchlistAlertDialog(alertBell.dataset.watchlistAlertBell)" in APP_JS
    assert "announceTriggeredWatchlistAlerts" not in APP_JS
    assert "提醒已触发：" not in APP_JS
    assert ".watchlist-alert-dialog" in STYLES_CSS
    assert ".watchlist-alert-reminder-note" in STYLES_CSS
    assert ".watchlist-alert-condition-actions" in STYLES_CSS
    assert ".watchlist-alert-bell.triggered" in STYLES_CSS
    assert ".watchlist-alert-bell-body" in STYLES_CSS
    assert "grid-column: span 2;" in STYLES_CSS
    assert "min-height: 42px;" in STYLES_CSS
    assert "watchlist-alerts-v5" in INDEX_HTML
    assert "watchlist-alerts-v4" in INDEX_HTML


def test_triggered_watchlist_alert_highlights_matching_condition_rows_only() -> None:
    assert 'id="watchlistAlertHitSummary"' not in INDEX_HTML
    assert "matchedConditionIndexes" not in APP_JS
    assert "watchlistAlertMatchedLineLabel(reminder)" not in APP_JS
    assert 'data-alert-condition-status="${presentation.tone}"' in APP_JS
    assert "第 ${index + 1} 行" in APP_JS
    assert 'label: "命中"' in APP_JS
    assert "命中条件行会用橙色高亮" in APP_JS
    assert ".watchlist-alert-hit-summary" not in STYLES_CSS
    assert ".watchlist-alert-reminder.is-triggered" not in STYLES_CSS
    assert ".watchlist-alert-condition.is-hit" in STYLES_CSS
    assert "watchlist-alert-hit-lines-v3" in INDEX_HTML


def test_similar_cases_explain_and_display_forecast_weight_ranking() -> None:
    assert "综合相似度、行业与市场匹配和时间衰减；按预测权重降序，采用全局统一尺度" in INDEX_HTML
    assert "<th>原始相似度</th>" not in INDEX_HTML
    assert "Math.log1p(contrast * normalized) / Math.log1p(contrast) * 100" in APP_JS
    assert '<td>${row.similarity ?? "-"}</td>' not in APP_JS


def test_long_analyst_coverage_uses_honest_labels_and_missing_state() -> None:
    assert '`${institutions}家机构 · ${researchReports}份研报`' in APP_JS
    assert '`一致预期 · 覆盖${consensusReports}份研报`' in APP_JS
    assert '`${dataPoints}项预测数据`' in APP_JS
    assert '`覆盖未来${forwardYears}年`' in APP_JS
    assert '"成长评分 暂无"' in APP_JS
    assert 'Number(item.analyst_forward_growth_score || 0).toFixed(1)' not in APP_JS
    assert "function analystForecastRows(item)" in APP_JS
    assert "function analystForecastCell(item, horizon)" in APP_JS
    assert 'EPS均值 ${numberText(forecast.eps_mean, 3, "元")}' in APP_JS
    assert 'EPS标准差 ${numberText(forecast.eps_std, 3, "元")}' in APP_JS
    assert '股价均值 ${numberText(forecast.price_mean, 2, "元")}' in APP_JS
    assert '股价标准差 ${numberText(forecast.price_std, 2, "元")}' in APP_JS
    assert "EPS×预测PE隐含股价" in APP_JS


def test_long_page_focuses_on_good_stocks_and_good_prices() -> None:
    assert "function recommendationBadge(item, includeDays = false)" in APP_JS
    assert 'RECOMMENDED: "推荐"' in APP_JS
    assert 'WATCH: "观察"' in APP_JS
    assert "function priceScoreDetail(item)" in APP_JS
    assert 'item.price_state === "WAIT_HISTORY"' in APP_JS
    assert 'item.price_state === "WAIT_STABILITY"' in APP_JS
    assert "metric(item.pr_from_pe, 3)" in APP_JS
    assert "metric(item.pr_from_pb, 3)" in APP_JS
    assert "percentile(item.roe_hist_percentile, item.roe_history_points)" in APP_JS
    assert 'item.display_reason || item.reason || "-"' in APP_JS
    assert "<th>推荐程度</th>" in INDEX_HTML
    for sort_key in [
        "good_stock_score",
        "price_score",
        "pe_ttm",
        "pb",
        "pr_from_pe",
        "pr_from_pb",
    ]:
        assert f'data-long-sort="{sort_key}"' in INDEX_HTML
        assert f'data-long-sort-header="{sort_key}"' in INDEX_HTML
    assert "function sortedLongStocks(stocks)" in APP_JS
    assert "function toggleLongSort(key)" in APP_JS
    assert 'pr_from_pe: "PR-PE"' in APP_JS
    assert 'pr_from_pb: "PR-PB"' in APP_JS
    assert 'header.setAttribute("aria-sort"' in APP_JS
    assert "metric(item.price_score)" in APP_JS
    assert "个月样本" in APP_JS
    assert "个股自身历史归一化（最多7年）· 跨日可比" not in APP_JS
    assert "非单日横截面排名" not in APP_JS
    assert "好股票 + 价格分 ≥ 60 + 长期价格结构通过" in INDEX_HTML
    for header in [
        "ROE",
        "当年E EPS / 股价",
        "次年E EPS / 股价",
        "后年E EPS / 股价",
    ]:
        assert f"<th>{header}</th>" in INDEX_HTML
    assert 'data-long-xueqiu' in APP_JS
    assert "xueqiuStockUrl(item.ts_code)" in APP_JS
    assert "雪球 ↗" in APP_JS
    assert 'target="_blank" rel="noopener noreferrer"' in APP_JS


def test_long_page_highlights_good_price_thresholds() -> None:
    assert "function longSignalClass(kind, value)" in APP_JS
    assert 'kind === "price-score" && numeric >= 80' in APP_JS
    assert 'kind === "pr" && numeric < 1' in APP_JS
    assert "numeric <= 10" in APP_JS
    assert "numeric <= 20" in APP_JS
    assert 'longSignalClass("price-score", item.price_score)' in APP_JS
    assert 'longSignalClass("pr", item.pr_from_pe)' in APP_JS
    assert 'longSignalClass("pr", item.pr_from_pb)' in APP_JS
    for percentile_key in [
        "pe_hist_percentile",
        "pb_hist_percentile",
        "pr_pe_hist_percentile",
        "pr_pb_hist_percentile",
    ]:
        assert f'longSignalClass("valuation-percentile", item.{percentile_key})' in APP_JS
    assert 'class="long-price-signal-legend"' in INDEX_HTML
    for signal_class in ["price-score", "pr-under-one", "percentile-20", "percentile-10"]:
        assert f".long-price-signal.{signal_class}" in STYLES_CSS
    assert "long-value-alerts-v1" in INDEX_HTML


def test_long_page_explains_price_score_bands_with_backtest_evidence() -> None:
    assert "价格分分档与历史回测" in INDEX_HTML
    assert 'id="longPriceBandRows"' in INDEX_HTML
    assert "验证期12月收益" in INDEX_HTML
    assert "样本外12月收益" in INDEX_HTML
    assert "样本外平均回撤" in INDEX_HTML
    assert "function renderLongPriceScoreBacktest()" in APP_JS
    assert "backtest.conclusion" in APP_JS
    assert "持有期存在重叠" in APP_JS
    assert "卖出、减仓与持仓管理移至自选池" in INDEX_HTML
    assert "PR-PE 与 PR-PB 同时保留" in INDEX_HTML
    assert "最多 7 年月末历史归一化" in INDEX_HTML


def test_byd_holding_inputs_are_restored_and_persisted() -> None:
    assert 'const BYD_HOLDING_STORAGE_KEY = "quant.byd.holding.v1"' in APP_JS
    assert "function saveBydHoldingInputs()" in APP_JS
    assert "function restoreBydHoldingInputs()" in APP_JS
    assert "version: 3" in APP_JS
    assert '"bydSharesInput"' in APP_JS
    assert '"bydCostInput"' in APP_JS
    assert "已恢复永久保存的持仓和成本" in APP_JS
    for removed in [
        "bydSoldTodaySharesInput",
        "bydSoldTodayPriceInput",
        "bydBoughtTodaySharesInput",
        "bydBoughtTodayPriceInput",
        "bydOpenTInput",
        "bydOpenTPriceInput",
        "bydOpenPositiveInput",
        "bydOpenPositivePriceInput",
        "sold_today_shares",
        "bought_today_shares",
        "open_t_shares",
        "open_positive_shares",
    ]:
        assert removed not in APP_JS
        assert removed not in INDEX_HTML
    assert APP_JS.rindex("restoreBydHoldingInputs();") < APP_JS.rindex("loadActivePageData();")
    assert 'id="bydHoldingSaveStatus"' in INDEX_HTML


def test_byd_page_shows_historical_validation_gate() -> None:
    assert 'id="bydValidationStatus"' in INDEX_HTML
    assert 'id="bydValidationMetrics"' in INDEX_HTML
    assert "横盘期历史验证" in INDEX_HTML
    assert "正T / 反T 分开计划" in INDEX_HTML
    assert "payload.daily_t_plan" in APP_JS
    assert "正T优先" in APP_JS
    assert "反T计划" in APP_JS
    assert "validation.held_out_results" in APP_JS
    assert ".byd-validation-metrics" in STYLES_CSS
    assert ".byd-alert.research-only" in STYLES_CSS


def test_strategy_workspace_summaries_use_compact_layouts() -> None:
    assert ".long-rule-panel {\n  display: none;" in STYLES_CSS
    assert "grid-template-columns: repeat(7, minmax(0, 1fr));" in STYLES_CSS
    assert ".byd-validation-panel {\n  display: grid;" in STYLES_CSS
    assert ".similar-summary-card {\n  align-items: center;\n  display: grid;" in STYLES_CSS
    assert "grid-auto-flow: column;" in STYLES_CSS
    assert "20260720-allotment-divider-v10" in INDEX_HTML


def test_allotment_workspace_header_uses_compact_layout() -> None:
    assert "/* Compact allotment header:" in STYLES_CSS
    assert "border-bottom: 4px solid var(--accent);" in STYLES_CSS
    assert ".cb-allotment-toolbar .eyebrow," in STYLES_CSS
    assert "grid-template-columns: auto auto minmax(0, 1fr) auto;" in STYLES_CSS


def test_active_workspace_tab_has_stable_red_indicator() -> None:
    assert '.workspace-tabs .page-tab[aria-selected="true"]::after' in STYLES_CSS
    assert "background: var(--accent);" in STYLES_CSS
    assert "20260720-allotment-divider-v10" in INDEX_HTML
