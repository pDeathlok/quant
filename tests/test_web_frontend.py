from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_JS = (PROJECT_ROOT / "web" / "app.js").read_text(encoding="utf-8")
STYLES_CSS = (PROJECT_ROOT / "web" / "styles.css").read_text(encoding="utf-8")


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


def test_workspace_tabs_scroll_consistently_on_narrow_screens() -> None:
    assert "grid-auto-flow: column;" in STYLES_CSS
    assert "grid-auto-columns: minmax(148px, 1fr);" in STYLES_CSS
    assert "overflow-x: auto;" in STYLES_CSS
    assert "scroll-snap-type: inline proximity;" in STYLES_CSS
    assert "prefers-reduced-motion: reduce" in STYLES_CSS
