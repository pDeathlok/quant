from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant.research.good_company_deep_evaluation import (
    EvaluationPaths,
    build_universe,
    evaluate_companies,
    flatten_evaluation,
    json_safe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS = EvaluationPaths(
    raw_dir=PROJECT_ROOT / "data/raw",
    broad_shortlist=PROJECT_ROOT / "reports/good_company_screen_20260809/industry_shortlist.csv",
    niche_watchlist=PROJECT_ROOT / "reports/good_company_screen_20260809/niche_capability_watchlist.csv",
    daily_basic_snapshot=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/tushare/tushare_daily_basic_20260807.parquet",
    governance_dir=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/tushare_governance",
    mcp_overrides_path=PROJECT_ROOT / "reports/good_company_deep_20260809/sources/mcp_overrides.json",
)


@pytest.fixture(scope="module")
def evaluations() -> list[dict]:
    _, items = evaluate_companies(
        PATHS,
        analysis_cutoff="2026-08-09T19:46:24+08:00",
        target_date="2027-08-09",
    )
    return json_safe(items)


def test_universe_is_exact_union_of_48_and_75() -> None:
    universe = build_universe(PATHS)
    assert len(universe) == 112
    assert universe["ts_code"].nunique() == 112
    assert int(universe["in_broad_48"].sum()) == 48
    assert int(universe["in_niche_75"].sum()) == 75
    assert int((universe["in_broad_48"] & universe["in_niche_75"]).sum()) == 11
    assert universe["scarcity_hypothesis"].str.len().gt(10).all()


def test_every_company_has_complete_contract_and_valid_json(evaluations: list[dict]) -> None:
    assert len(evaluations) == 112
    assert len({item["identity"]["ts_code"] for item in evaluations}) == 112
    json.dumps(evaluations, ensure_ascii=False, allow_nan=False)
    for item in evaluations:
        assert set(item) == {
            "identity", "cutoff", "market", "financials", "forecast", "gqs",
            "valuation", "research", "evidence",
        }
        assert item["cutoff"]["analysis_cutoff"] == "2026-08-09T19:46:24+08:00"
        assert item["cutoff"]["target_date"] == "2027-08-09"
        assert item["market"]["current_price"] > 0
        assert 0 <= item["gqs"]["gqs_r"] <= 100
        assert -5 <= item["gqs"]["forward_adjustment"] <= 5
        assert 0 <= item["gqs"]["gqs_f"] <= 100
        assert 0 <= item["gqs"]["coverage_ratio"] <= 1
        assert len(item["research"]["thesis_pillars"]) == 3
        assert item["research"]["strongest_counterargument"]


def test_gqs_module_points_respect_weights(evaluations: list[dict]) -> None:
    limits = {
        "a_customer_business": 10,
        "b_scarcity_moat": 20,
        "c_growth_reinvestment": 10,
        "d_returns_profitability": 20,
        "e_cash_accounting": 15,
        "f_resilience_risk": 10,
        "g_governance_allocation": 15,
    }
    for item in evaluations:
        for field, maximum in limits.items():
            score = item["gqs"][field]
            assert score is None or 0 <= score <= maximum


def test_scenario_target_and_space_are_reproducible(evaluations: list[dict]) -> None:
    available = 0
    unavailable = 0
    for item in evaluations:
        valuation = item["valuation"]
        price = item["market"]["current_price"]
        if valuation["status"] == "unavailable":
            unavailable += 1
            assert valuation["missing_reasons"]
            assert valuation["bear"] is None
            continue
        available += 1
        for scenario in ("bear", "base", "bull"):
            values = valuation[scenario]
            assert values["target_price"] > 0
            assert values["price_upside"] == pytest.approx(
                values["target_price"] / price - 1,
                abs=0.001,
            )
            assert values["total_return"] == pytest.approx(
                values["price_upside"] + values["dividend_return"],
                abs=0.0002,
            )
    assert available + unavailable == 112
    assert available >= 90


def test_calibration_companies_use_distinct_industry_routes(evaluations: list[dict]) -> None:
    by_code = {item["identity"]["ts_code"]: item for item in evaluations}
    assert by_code["002142.SZ"]["valuation"]["method_primary"] == "P/B–ROE"
    assert by_code["601899.SH"]["valuation"]["method_primary"] == "中周期盈利×P/E"
    assert by_code["002595.SZ"]["valuation"]["method_primary"] == "FY2027稀释EPS×目标P/E"
    assert by_code["603605.SH"]["valuation"]["method_primary"] == "FY2027稀释EPS×目标P/E"
    assert by_code["002142.SZ"]["valuation"]["base"]["price_upside"] < 0.60


def test_flatten_preserves_sortable_dashboard_fields(evaluations: list[dict]) -> None:
    row = flatten_evaluation(evaluations[0])
    required = {
        "ts_code", "name", "industry", "gqs_r", "gqs_f", "coverage_ratio",
        "score_a", "score_b", "score_c", "score_d", "score_e", "score_f", "score_g",
        "bear_target", "base_target", "bull_target", "base_upside", "technical_state",
    }
    assert required <= set(row)
