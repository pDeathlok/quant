from __future__ import annotations

import ast
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.application.page_refresh_coverage import (
    PAGE_REFRESH_CONTRACTS,
    LONG_VARIANT_EXPOSURE,
    SHORT_STRATEGY_MEMBERS,
    page_refresh_coverage_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_frontend_tab_has_explicit_refresh_policy() -> None:
    source = (ROOT / "web/app.js").read_text()
    tabs = source.split("const WORKSPACE_TABS = [", 1)[1].split("]", 1)[0]
    assert set(re.findall(r'key: "([^"]+)"', tabs)) == {
        page.tab for page in PAGE_REFRESH_CONTRACTS
    }
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    all_nodes = set(registry.required_node_ids("all"))
    for page in PAGE_REFRESH_CONTRACTS:
        if page.policy == "user_crud":
            assert page.tab in {"plans", "reviews"}
            assert not page.products
            continue
        if page.policy == "on_demand":
            assert page.tab == "futures"
            assert not page.products
            assert page.tab not in registry.scope_roots
            continue
        assert set(page.products) <= all_nodes
        assert set(page.products) <= set(registry.required_node_ids(page.tab))
        assert all(registry.nodes[node].final_gate for node in page.products)


def test_futures_preview_has_an_independent_refresh_api() -> None:
    from quant.webapp.api import router

    routes = {route.path: route.methods for route in router.routes}
    assert "POST" in routes["/index-futures/refresh"]
    assert "GET" in routes["/index-futures/refresh/status"]
    report = page_refresh_coverage_report(
        DEFAULT_DAILY_DEPENDENCY_REGISTRY, "all", "2026-11-13", {},
    )
    page = next(page for page in report["pages"] if page["tab"] == "futures")
    assert page["policy"] == "on_demand"
    assert page["in_scope"] is False
    assert page["hard_gate_registered"] is False


def test_short_strategy_groups_and_members_match_service_contract() -> None:
    tree = ast.parse((ROOT / "src/quant/webapp/services.py").read_text())
    groups = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "STRATEGY_GROUPS"
                          for target in node.targets))
    assert {item["key"]: tuple(item["members"]) for item in groups} == SHORT_STRATEGY_MEMBERS


def test_cb_substrategy_inventory_matches_builder() -> None:
    from quant.routine.convertible_bond_grid_plan import convertible_bond_grid_strategy_configs

    contract = next(page for page in PAGE_REFRESH_CONTRACTS if page.tab == "cb")
    assert contract.strategies == tuple(config.name for config, _ in convertible_bond_grid_strategy_configs())


def test_chan_substrategy_inventory_matches_builder() -> None:
    from quant.strategies.custom.chan_model import CHAN_MODEL_RULES

    contract = next(page for page in PAGE_REFRESH_CONTRACTS if page.tab == "chan")
    assert contract.strategies == tuple(rule.id for rule in CHAN_MODEL_RULES)


def test_long_visible_buttons_do_not_include_historical_catalog_variants() -> None:
    class Buttons(HTMLParser):
        variants: list[str]

        def __init__(self) -> None:
            super().__init__()
            self.variants = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            attributes = dict(attrs)
            if tag == "button" and "data-long-variant" in attributes:
                assert "disabled" not in attributes and "hidden" not in attributes
                self.variants.append(attributes["data-long-variant"])

    parser = Buttons()
    parser.feed((ROOT / "web/index.html").read_text())
    assert set(parser.variants) == set(LONG_VARIANT_EXPOSURE["visible_buttons"])
    assert len(parser.variants) == 4  # Desktop and mobile controls.
    page = next(page for page in PAGE_REFRESH_CONTRACTS if page.tab == "long")
    assert set(page.strategies) == set(LONG_VARIANT_EXPOSURE["visible_buttons"]) | set(
        LONG_VARIANT_EXPOSURE["daily_compatibility_outputs"],
    )
    assert set(page.strategies).isdisjoint(LONG_VARIANT_EXPOSURE["catalog_only_not_daily"])


def test_long_and_transition_variants_all_have_required_evidence() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    for tab in ("long", "byd"):
        page = next(page for page in PAGE_REFRESH_CONTRACTS if page.tab == tab)
        evidence = registry.nodes[page.products[0]].freshness.evidence
        variants = {spec.expected_value for spec in evidence if spec.adapter == "result_variant"}
        if tab == "long":
            assert any(spec.locator == "long_stock_pool.blood_chip" and spec.required for spec in evidence)
            variants.add("blood_chip")
        assert variants == set(page.strategies)
        assert all(spec.required for spec in evidence)


@pytest.mark.parametrize("status", ("success", "degraded", "failed", "not_observed"))
def test_etf_is_explicitly_optional_and_preserves_source_lag(status: str) -> None:
    result = {"status": status, "signal_date": "2026-11-13",
              "source_dates": {"fd_share": "2026-11-11", "amount": "2026-11-13"}}
    results = {} if status == "not_observed" else {"short_etf_flow": result}
    report = page_refresh_coverage_report(DEFAULT_DAILY_DEPENDENCY_REGISTRY, "all", "2026-11-13", results)
    flow = report["optional_dependencies"]["short_etf_flow"]
    assert flow["status"] == status
    assert flow["hard_gate"] is False
    assert flow["require_all_feature_values"] is False
    assert flow["source_max_lag_sessions"] == {"fd_share": 2, "unit_nav": 2, "adj_factor": 2, "amount": 0}
    assert "short_etf_rank.left" in flow["consumers"]
    assert "short_etf_rank.right" in flow["consumers"]
    if results:
        assert flow["source_dates"] == result["source_dates"]
    assert all("etf" not in node for node in DEFAULT_DAILY_DEPENDENCY_REGISTRY.required_node_ids("all"))


def test_optional_report_distinguishes_stale_and_out_of_scope() -> None:
    registry = DEFAULT_DAILY_DEPENDENCY_REGISTRY
    results = {"short_etf_flow": {"status": "success", "signal_date": "2026-11-12"}}
    report = page_refresh_coverage_report(registry, "short", "2026-11-13", results)
    assert report["optional_dependencies"]["short_etf_flow"]["status"] == "stale_or_undated"
    report = page_refresh_coverage_report(registry, "long", "2026-11-13", results)
    assert report["optional_dependencies"]["short_etf_flow"]["status"] == "out_of_scope"
