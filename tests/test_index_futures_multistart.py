"""Temporal boundaries, comparable horizons, funding denominator and switching."""
import numpy as np
import pandas as pd

from test_index_futures_funded import row
from test_index_futures_risk_paths import scenario
import evaluate_index_futures_multistart as research


def reference_windows():
    records = []
    for horizon in (122, 244):
        for i in range(6):
            records.append(dict(start=f"20210{i+1}04", end=f"20220{i+1}01", horizon=horizon,
                total_return=.1*horizon/122, max_drawdown=-.2, forced=0, trades=3*horizon/122,
                active_days=30*horizon/122, signal_opportunities=10, blocked_opportunities=2))
    return pd.DataFrame(records)


def scores():
    return {"base":dict(ready=True, forced=0, median=.10, q25=.04, dd90=.20,
                       trades=4, active=30, block=.2, score=-.03, daily_pnl=200),
            "aux55":dict(ready=True, forced=0, median=.15, q25=.08, dd90=.18,
                        trades=4, active=30, block=.1, score=.025, daily_pnl=400)}


def test_future_window_and_too_old_start_excluded_before_scoring():
    frame = reference_windows()
    extra = frame.iloc[[0, 0]].copy()
    extra.loc[extra.index[0], "total_return"] = 1e9
    extra = extra.reset_index(drop=True)
    extra.loc[0, "end"] = "20220704"
    extra.loc[1, "start"] = "20180101"
    before = research.past_windows(frame, "20220630")
    after = research.past_windows(pd.concat([frame, extra], ignore_index=True), "20220630")
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    assert research.summarize_policy(before, "20220630") == research.summarize_policy(after, "20220630")


def test_horizon_normalization_keeps_same_daily_profit_comparable():
    result = research.summarize_policy(reference_windows(), "20220630")
    assert result["ready"]
    assert np.isclose(result["median"], .1)
    assert np.isclose(result["q25"], .1)
    assert np.isclose(result["score"], 0)
    assert np.isclose(result["block"], .2)


def test_recent_matured_windows_get_more_weight_without_future_ends():
    sample = reference_windows()
    sample.loc[sample.end.eq("20220101"), "total_return"] = -1
    sample.loc[sample.end.eq("20220601"), "total_return"] = .8
    result = research.summarize_policy(sample, "20220630")
    assert result["median"] > 0
    assert result["last_history_end"] == "20220601"


def test_missing_horizon_and_no_signals_do_not_pass_eligibility():
    data = reference_windows()
    assert not research.summarize_policy(data.iloc[:-1], "20220630")["ready"]
    data[["signal_opportunities", "blocked_opportunities"]] = 0
    assert not research.summarize_policy(data, "20220630")["ready"]


def test_funding_ratio_denominator_is_only_qualified_entry_opportunities():
    trace = pd.DataFrame(dict(raw_candidates=[0, 2, 1, 0], affordable_candidates=[0, 1, 0, 0]))
    assert research.blocked_counts(trace) == dict(signal_opportunities=2, blocked_opportunities=1, blocked_ratio=.5)
    empty = research.blocked_counts(trace.iloc[[0]])
    assert empty["signal_opportunities"] == 0 and np.isnan(empty["blocked_ratio"])


def test_funding_ablation_rejects_candidate_worse_than_base():
    values = scores()
    values["aux55"]["block"] = .3
    assert research.select_policy(values, "base", "multi_tail", initial=True)[0] == "aux55"
    assert research.select_policy(values, "base", "multi_funded", initial=True)[0] == "base"


def test_switch_checks_current_cash_and_expected_edge_in_currency():
    values = scores()
    selected, reason, detail = research.select_policy(values, "base", "multi_funded", rows=[row()], positions={}, equity=500000)
    assert selected == "aux55" and reason == "passed_robust_and_cash_hurdles"
    assert detail["historical_quarter_edge"] == 12200
    assert detail["required_edge"] >= .005*500000
    assert research.select_policy(values, "base", "multi_funded", rows=[row()], positions={}, equity=1)[1] == "no_currently_affordable_contract"
    values["aux55"]["daily_pnl"] = 201
    assert research.select_policy(values, "base", "multi_funded", rows=[row()], positions={})[1] == "insufficient_net_advantage"


def test_forced_or_zero_coverage_candidates_cannot_be_selected():
    for field, value in (("forced", 1), ("trades", 0), ("active", 0), ("median", 0)):
        values = scores()
        values["aux55"][field] = value
        assert research.select_policy(values, "base", "multi_funded", initial=True)[0] == "base"


def test_prefix_liquidation_matches_direct_short_horizon_with_one_closing_fee():
    groups, opens, closes = scenario([1000]*9, [1000]*8)
    full, _, _ = research.risk_replay(groups, opens, closes, .0003, .15, entry_calendar={"20220104"})
    short, _, _ = research.risk_replay(groups[:3], opens, closes, .0003, .15, entry_calendar={"20220104"})
    prefix = research.liquidated_prefix(full, 3, opens, .0003)
    pd.testing.assert_frame_equal(prefix, short)
