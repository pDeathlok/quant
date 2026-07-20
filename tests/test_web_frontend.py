from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_JS = (PROJECT_ROOT / "web" / "app.js").read_text(encoding="utf-8")
STYLES_CSS = (PROJECT_ROOT / "web" / "styles.css").read_text(encoding="utf-8")
INDEX_HTML = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")


def test_workspace_tabs_use_one_seven_item_contract() -> None:
    expected_keys = ["short", "chan", "long", "cb", "cbAllotment", "byd", "similar"]

    assert "const WORKSPACE_TABS = [" in APP_JS
    assert all(f'key: "{key}"' in APP_JS for key in expected_keys)
    assert "ensureWorkspaceTabs" in APP_JS
    assert "ensureChanTabs" not in APP_JS
    assert "grid-template-columns: repeat(7, minmax(0, 1fr));" in STYLES_CSS


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


def test_watchlist_stocks_have_persistent_note_editor() -> None:
    assert 'data-similar-note="${item.symbol}"' in APP_JS
    assert 'id="similarNoteDialog"' in INDEX_HTML
    assert 'id="similarNoteInput"' in INDEX_HTML
    assert "/note`, {" in APP_JS
    assert 'method: "PUT"' in APP_JS
    assert 'await fetchJson("/similar-patterns/watchlist")' in APP_JS
    assert "分析加载失败，笔记仍可编辑" in APP_JS
    assert ".similar-note-dialog" in STYLES_CSS


def test_strategy_watchlist_add_includes_default_source_note() -> None:
    assert "function compactWatchlistDate(value)" in APP_JS
    assert "function watchlistSourceNote(dateValue, sourceText)" in APP_JS
    assert 'body: JSON.stringify({ symbol, note: options.note || "" })' in APP_JS
    assert 'data-watchlist-note=' in APP_JS
    assert 'note: target.dataset.watchlistNote || ""' in APP_JS
    assert 'note: target.note' in APP_JS
    assert "触发 ${(item.matched_families || []).join(\" / \")} 策略" in APP_JS
    assert "配债股${item.status ?" in APP_JS


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


def test_watchlist_rows_support_persistent_drag_order_and_pin_menu() -> None:
    assert 'await fetchJson("/similar-patterns/watchlist/order", {' in APP_JS
    assert '/pin`, {' in APP_JS
    assert 'data-watchlist-pinned="${item.pinned ? "true" : "false"}"' in APP_JS
    assert 'title="按住拖动排序；右键可置顶"' in APP_JS
    assert "function reorderSimilarWatchRows(sourceSymbol, targetSymbol, placeAfter = false)" in APP_JS
    assert 'event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)' in APP_JS
    assert "data-watchlist-context-pin" in INDEX_HTML
    assert ".similar-stock-cell .watchlist-pin-badge" in STYLES_CSS
    assert "20260720-watchlist-reorder-v5" in INDEX_HTML


def test_similar_cases_explain_and_display_forecast_weight_ranking() -> None:
    assert "综合相似度、行业与市场匹配和时间衰减；按预测权重降序，采用全局统一尺度" in INDEX_HTML
    assert "<th>原始相似度</th>" not in INDEX_HTML
    assert "Math.log1p(contrast * normalized) / Math.log1p(contrast) * 100" in APP_JS
    assert '<td>${row.similarity ?? "-"}</td>' not in APP_JS


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
    assert ".cb-allotment-toolbar .refresh-status:not(:has(.refresh-progress-fill.active))" in STYLES_CSS


def test_active_workspace_tab_has_stable_red_indicator() -> None:
    assert '.workspace-tabs .page-tab[aria-selected="true"]::after' in STYLES_CSS
    assert "background: var(--accent);" in STYLES_CSS
    assert "20260720-allotment-divider-v10" in INDEX_HTML
