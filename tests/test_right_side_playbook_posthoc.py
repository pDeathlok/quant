from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.research.right_side_playbook_posthoc import (
    MonthBlockBootstrapSpec,
    attach_signal_identities,
    capital_curve_feasibility,
    compare_shared_to_static_by_signal,
    paired_monthly_stability,
    summarize_outcomes_by_fold_action,
    summarize_selections_by_arm_action,
    summarize_selections_by_signal,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS


def _selections() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in range(1, 5):
        for position in range(10):
            event_id = f"B-{month}-{position}"
            for arm, action, net in (
                ("static_per_signal", "no_trade", 0.0),
                (
                    "shared_playbook_model",
                    "next_open__expiry_t5_close",
                    0.01 if position < 7 else -0.005,
                ),
            ):
                rows.append(
                    {
                        "arm": arm,
                        "fold": "B",
                        "event_id": event_id,
                        "symbol": "000001.SZ",
                        "date": pd.Timestamp(2025, month, position + 1),
                        "playbook_id": action,
                        "planned_playbook_id": action,
                        "entry_date": pd.Timestamp(2025, month, position + 2),
                        "exit_date": pd.Timestamp(2025, month, position + 4),
                        "net_return": net,
                        "mae": -0.01 if action != "no_trade" else 0.0,
                        "round_trip_cost": 0.0015 if action != "no_trade" else 0.0,
                        "eligible": True,
                        "mature": True,
                        "execution_status": (
                            "executed" if action != "no_trade" else "no_trade"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month in range(1, 5):
        for position in range(10):
            row: dict[str, object] = {
                "fold": "B",
                "event_id": f"B-{month}-{position}",
            }
            row.update({signal: True for signal in RIGHT_SIDE_SIGNALS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_month_block_stability_is_paired_deterministic_and_positive() -> None:
    spec = MonthBlockBootstrapSpec(iterations=500, random_seed=7)
    monthly, first = paired_monthly_stability(_selections(), bootstrap_spec=spec)
    _, second = paired_monthly_stability(_selections(), bootstrap_spec=spec)

    assert len(monthly) == 4
    assert first == second
    assert first["paired_events"] == 40
    assert first["negative_month_ratio"] == pytest.approx(0.0)
    assert first["bootstrap"]["ci_low"] > 0.0
    assert first["month_cluster_sign_flip"][
        "one_sided_p_value_shared_gt_static"
    ] == pytest.approx(1.0 / 16.0)
    assert first["month_cluster_sign_flip"]["two_sided_p_value"] == pytest.approx(
        2.0 / 16.0
    )
    assert first["monthly_direction_sign_test"][
        "one_sided_p_value_shared_positive"
    ] == pytest.approx(1.0 / 16.0)


def test_action_signal_and_capital_summaries_keep_event_semantics() -> None:
    selections = _selections()
    action = summarize_selections_by_arm_action(selections)
    joined = attach_signal_identities(selections, _events())
    signal = summarize_selections_by_signal(joined)
    delta = compare_shared_to_static_by_signal(signal)
    capital = capital_curve_feasibility(selections)

    shared_executed = action.loc[
        action["arm"].eq("shared_playbook_model") & action["stage"].eq("executed")
    ].iloc[0]
    assert shared_executed["events"] == 40
    assert len(signal) == 2 * len(RIGHT_SIDE_SIGNALS)
    assert len(delta) == len(RIGHT_SIDE_SIGNALS)
    assert delta["delta_average_event_net_return"].gt(0).all()
    assert capital["can_build_assumption_bound_occupancy_backtest"]
    assert capital["unconstrained_occupancy_envelope"][
        "maximum_raw_concurrent_candidates"
    ] > 0
    assert not capital["is_true_capital_curve_now"]
    assert not capital["production_ready"]


def test_outcome_fold_action_summary_preserves_unknown_tail() -> None:
    outcomes = pd.DataFrame(
        {
            "fold": ["A", "A", "B", "B"],
            "event_id": ["A1", "A1", "B1", "B1"],
            "playbook_id": ["trade", "no_trade", "trade", "no_trade"],
            "entry_mode": ["next_open", "no_trade", "next_open", "no_trade"],
            "exit_policy_id": ["expiry", "no_trade", "expiry", "no_trade"],
            "eligible": [True, True, True, True],
            "mature": [True, True, False, True],
            "net_return": [0.01, 0.0, np.nan, 0.0],
            "mae": [-0.02, 0.0, np.nan, 0.0],
            "round_trip_cost_bps": [15.0, 0.0, 15.0, 0.0],
        }
    )
    summary = summarize_outcomes_by_fold_action(outcomes)
    b_trade = summary.loc[
        summary["fold"].eq("B") & summary["playbook_id"].eq("trade")
    ].iloc[0]

    assert b_trade["known_return_events"] == 0
    assert b_trade["known_return_coverage"] == pytest.approx(0.0)
    contaminated = outcomes.copy()
    contaminated.loc[0, "fold"] = "C"
    with pytest.raises(ValueError, match="exactly A/B"):
        summarize_outcomes_by_fold_action(contaminated)
