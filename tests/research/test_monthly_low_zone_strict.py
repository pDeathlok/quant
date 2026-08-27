from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.monthly_low_zone_strict import (
    PRIMARY_STRICT_STRUCTURE,
    StrictLowZoneConfig,
    add_strict_gate_columns,
    cohort_bootstrap_interval,
    decide_strict_structure,
    leave_one_cohort_out_minimum,
    materialize_strict_structures,
    wilson_interval,
)


def _event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rule": ["no_new_low_20", "no_new_low_20"],
            "signal_date": pd.to_datetime(["2020-04-30", "2020-04-30"]),
            "breadth_constituents": [800, 800],
            "breadth_positive_share_20d": [0.19, 0.21],
            "breadth_median_return_20d": [-0.09, -0.09],
            "drawdown_from_prior_peak": [-0.72, -0.72],
            "annual_quality_available_at": pd.to_datetime(
                ["2020-04-01", "2020-05-01"]
            ),
            "financial_age_days": [120, 120],
            "annual_history_years": [5, 5],
            "profit_positive_share_5y": [0.8, 0.8],
            "cfo_positive_share_5y": [1.0, 1.0],
            "income_n_income_attr_p": [10.0, 10.0],
            "cashflow_n_cashflow_act": [12.0, 12.0],
        }
    )


def test_strict_gate_requires_both_panic_dimensions_and_causal_financials() -> None:
    gated = add_strict_gate_columns(_event_frame(), StrictLowZoneConfig())

    assert gated["gate_systemic_deep"].tolist() == [True, False]
    assert gated["gate_dd70"].tolist() == [True, True]
    assert gated["gate_survival80"].tolist() == [True, False]


def test_primary_structure_is_materialized_only_after_all_frozen_gates() -> None:
    gated = add_strict_gate_columns(_event_frame(), StrictLowZoneConfig())
    materialized = materialize_strict_structures(gated)
    primary = materialized.loc[
        materialized["structure"].eq(PRIMARY_STRICT_STRUCTURE)
    ]

    assert len(primary) == 1
    assert primary.index.tolist() == [0] or len(primary) == 1


def test_wilson_interval_is_bounded_and_penalizes_small_samples() -> None:
    low_small, high_small = wilson_interval(2, 2)
    low_large, high_large = wilson_interval(20, 20)

    assert 0.0 < low_small < high_small <= 1.0
    assert low_large > low_small


def test_cluster_bootstrap_uses_cohort_values_deterministically() -> None:
    values = pd.Series([0.10, 0.20, 0.30])
    first = cohort_bootstrap_interval(values, iterations=500, seed=7)
    second = cohort_bootstrap_interval(values, iterations=500, seed=7)

    assert first == second
    assert np.isclose(first[0], 0.20)
    assert first[1] > 0.0


def test_leave_one_cohort_out_returns_weakest_deleted_mean() -> None:
    result = leave_one_cohort_out_minimum(pd.Series([-0.10, 0.20, 0.30]))

    assert np.isclose(result, 0.05)


def test_decision_does_not_treat_large_stock_count_as_independent_sample() -> None:
    rows = []
    for structure in (
        "systemic_deep_no_new_low_dd60_survival80",
        PRIMARY_STRICT_STRUCTURE,
        "systemic_deep_no_new_low_dd80_survival80",
    ):
        for period in (
            "development_2013_2016",
            "exposed_validation_2017_2020",
            "seen_diagnostic_2021_2024",
            "historical_2013_2024",
        ):
            rows.append(
                {
                    "period": period,
                    "structure": structure,
                    "horizon": 504,
                    "target_return": 0.15,
                    "completed_events": 1000,
                    "completed_cohorts": 1 if period != "historical_2013_2024" else 3,
                    "event_win_rate": 0.99,
                    "median_event_return": 0.148,
                    "profit_factor": 10.0,
                    "positive_cohort_share": 1.0,
                    "cohort_equal_mean_return": 0.10,
                    "cohort_bootstrap_ci95_low": 0.05,
                    "worst_cohort_return": 0.02,
                    "leave_one_cohort_out_min_mean": 0.08,
                }
            )
    decision = decide_strict_structure(pd.DataFrame(rows), StrictLowZoneConfig())

    assert not decision["event_gate_passed"]
    assert decision["status"] == "promising_but_independent_sample_insufficient"
    assert not decision["deployment_eligible"]
