from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY, Layer
from quant.application.page_refresh_coverage import PAGE_REFRESH_CONTRACTS
from quant.application.refresh_contracts import (
    REFRESH_SCOPE_LABELS,
    REFRESH_SCOPE_STEPS,
    build_progress_steps,
    normalize_refresh_scope,
)
from quant.routine.default_operations import DEFAULT_DAILY_OPERATION_REGISTRY


class WorkspaceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.pages = {}
        self.page = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "main" and "page-view" in attrs.get("class", "").split():
            page_id = attrs["id"]
            assert page_id.endswith("Page")
            self.page = page_id[:-4]
            assert self.page not in self.pages
            self.pages[self.page] = []
        if "data-refresh-scope" in attrs:
            assert self.page is not None
            self.pages[self.page].append(attrs["data-refresh-scope"])

    def handle_endtag(self, tag):
        if tag == "main":
            self.page = None


def test_every_frontend_strategy_page_has_a_registered_refresh_scope():
    parser = WorkspaceParser()
    parser.feed((Path(__file__).resolve().parents[1] / "web/index.html").read_text())
    # These pages edit user-authored records, not daily market products.
    manual_pages = {"reviews", "plans"}
    on_demand_pages = {page.tab for page in PAGE_REFRESH_CONTRACTS if page.policy == "on_demand"}
    assert on_demand_pages == {"futures"}
    assert manual_pages <= parser.pages.keys()
    assert on_demand_pages <= parser.pages.keys()
    strategy_pages = parser.pages.keys() - manual_pages - on_demand_pages
    assert strategy_pages
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    all_required = set(registry.required_node_ids("all"))
    for page, scopes in parser.pages.items():
        if page in manual_pages | on_demand_pages:
            assert not scopes
            continue
        assert scopes, f"{page} lacks a refresh button and daily dependency contract"
        assert set(scopes) == {page}, f"{page} refreshes a different workspace"
        for scope in scopes:
            assert normalize_refresh_scope(scope) == scope
            assert scope in registry.scope_roots
            assert set(registry.required_node_ids(scope)) <= all_required
            assert all(registry.nodes[root].layer == Layer.PRODUCT for root in registry.scope_roots[scope])
            DEFAULT_DAILY_OPERATION_REGISTRY.validate_against_dependencies(
                registry, node_ids=registry.required_node_ids(scope),
            )


@pytest.mark.parametrize("scope", sorted(DEFAULT_DAILY_DEPENDENCY_REGISTRY.scope_roots))
def test_progress_contract_covers_registry_steps_once(scope):
    expected = DEFAULT_DAILY_DEPENDENCY_REGISTRY.progress_steps(scope)
    steps = build_progress_steps(scope)
    keys = [step["key"] for step in steps]
    assert tuple(keys) == expected
    assert REFRESH_SCOPE_STEPS[scope] == keys
    assert len(keys) == len(set(keys))
    assert scope in REFRESH_SCOPE_LABELS
