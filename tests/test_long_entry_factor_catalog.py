from __future__ import annotations

import sys
from pathlib import Path


RESEARCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "research"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from build_long_entry_factor_catalog import (  # noqa: E402
    SELECTOR_PRODUCTION_FEATURES,
    build_catalog,
    read_literal_list,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_catalog_covers_all_previously_used_project_and_selector_factors() -> None:
    catalog = build_catalog()
    by_factor = {record.factor: record for record in catalog}
    project_factors = read_literal_list(
        PROJECT_ROOT / "src/quant/features/variable_library.py",
        "PROJECT_FACTOR_COLUMNS",
    )

    assert len(project_factors) == 147
    assert len(SELECTOR_PRODUCTION_FEATURES) == 49
    assert set(project_factors) <= set(by_factor)
    assert set(SELECTOR_PRODUCTION_FEATURES) <= set(by_factor)
    assert all("B1_production" in by_factor[factor].used_by for factor in project_factors)
    assert all(
        "selector_buy_hold_production" in by_factor[factor].used_by
        for factor in SELECTOR_PRODUCTION_FEATURES
    )


def test_catalog_preserves_dual_pr_and_multiwindow_weekly_candidates() -> None:
    by_factor = {record.factor: record for record in build_catalog()}

    assert by_factor["pr_pe"].status == "phase1_core"
    assert by_factor["pr_pb"].status == "phase1_core"
    assert by_factor["pe_hist_pct_2y"].status == "phase1_core"
    assert by_factor["pe_hist_pct_10y"].status == "phase1_core"
    assert by_factor["pr_pe_pct_short_long_gap_2y_7y"].status == "phase1_core"
    assert by_factor["price_hist_pct_5y"].role == "timing;drawdown_risk"
    assert by_factor["analyst_forward_y0_eps_std_180d"].point_in_time_rule == (
        "report_date_lte_signal_date_180d_window_deduplicated"
    )
    assert by_factor["analyst_forward_y2_price_mean_180d"].used_by == (
        "long_analyst_experiments;long_page_display"
    )


def test_catalog_blocks_hindsight_and_admits_backfilled_history_sources() -> None:
    by_factor = {record.factor: record for record in build_catalog()}

    for factor in ["actual_eps", "actual_revenue", "actual_net_profit", "future_return_52w"]:
        assert by_factor[factor].status == "excluded"
        assert by_factor[factor].point_in_time_rule == "never_feature"

    for factor in [
        "large_net_5d_ratio",
        "top_list_count_20d",
        "holder_net_change_ratio_90d",
    ]:
        assert by_factor[factor].status == "phase1_candidate"
        assert "2013" in by_factor[factor].availability


def test_catalog_has_unique_factor_names_and_known_statuses() -> None:
    catalog = build_catalog()
    names = [record.factor for record in catalog]

    assert len(names) == len(set(names))
    assert {record.status for record in catalog} == {
        "phase1_core",
        "phase1_candidate",
        "transform_required",
        "phase2_research",
        "shadow_only",
        "benchmark_only",
        "excluded",
    }
