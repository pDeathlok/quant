"""Time boundaries, inherited positions, switch costs, and policy-selection guards."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("adaptation", SCRIPTS / "compare_index_futures_adaptation.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def sample_groups():
    dates = pd.bdate_range("2022-03-28", periods=12).strftime("%Y%m%d").tolist()
    groups = []
    for i in range(10):
        groups.append([SimpleNamespace(entry_date=dates[i], exit_date=dates[i+1], trade_date="20220325",
                                      symbol="IF", ts_code="IF2206", entry_open=1000., exit_open=1000.,
                                      entry_settle=1000., multiplier=300, f_volatility_20=.01,
                                      etf_ret_1=.01, long_60=.8, short_60=.1, long_20=.8, short_20=.1,
                                      win_long=.8, win_short=.2, flat=0., long_IF=1., short_IF=-1.)])
    return groups


def test_only_mature_labels_enter_training():
    panel = pd.DataFrame(dict(trade_date=["20200101", "20220301", "20220302", "20220303"],
                              maturity=["20200130", "20220330", "20220331", "20220401"],
                              etf_ret_1=[.01]*4))
    actual = research.mature_training(panel, "20220331", "maturity")
    assert actual.maturity.tolist() == ["20200130", "20220330"]


def test_validation_excludes_unavailable_next_open_returns():
    frame = pd.DataFrame(dict(entry_date=["20220329", "20220330", "20220331"],
                              exit_date=["20220330", "20220331", "20220401"]))
    history = research.history_before(frame, "20220331")
    assert history.exit_date.tolist() == ["20220330", "20220331"]


def test_switch_closes_position_and_charges_both_round_trips():
    groups = sample_groups()
    day = groups[4][0].entry_date
    fixed, _ = research.replay(groups, .0003, .15)
    changed, events = research.replay(groups, .0003, .15, schedule={day: "confidence60"})
    assert changed.iloc[4].positions == ""
    assert changed.iloc[5].positions == "IF2206:+1"
    assert len(events) == 1 and events[0]["closed_for_switch"]
    assert np.isclose(fixed.fees.sum(), 180.)
    assert np.isclose(changed.fees.sum(), 360.)
    assert np.isclose(changed.equity.iloc[-1], 500000-360)
    opens = {(r[0].entry_date, "IF2206"): 1000. for r in groups}
    opens[groups[-1][0].exit_date, "IF2206"] = 1000.
    trades = research.trade_episodes(changed, opens, .0003)
    assert len(trades) == 2 and np.isclose(trades.pnl.sum(), -360.)


def test_unchanged_policy_does_not_create_turnover():
    groups = sample_groups()
    fixed, _ = research.replay(groups, .0003, .15)
    unchanged, events = research.replay(groups, .0003, .15, schedule={groups[4][0].entry_date: "base"})
    pd.testing.assert_frame_equal(fixed, unchanged)
    assert events == []


def score(**extra):
    return dict(dict(days=500, trades=10, active_days=200, forced=0, total_return=.10,
                     weighted_mean=.001, max_drawdown=-.2, weighted_sortino=1.), **extra)


def test_high_score_does_not_override_risk_and_coverage_guards():
    scores = {"base": score(), "aux55": score(max_drawdown=-.3, weighted_sortino=10),
              "confidence60": score(trades=2, active_days=20, weighted_sortino=100),
              "opposing_exit": score(weighted_sortino=2)}
    selected, _ = research.choose_policy(scores, "base")
    assert selected == "opposing_exit"


def test_no_eligible_candidate_retains_current_rule():
    scores = {name: score(weighted_mean=-.001) for name in research.POLICIES}
    selected, reason = research.choose_policy(scores, "aux55")
    assert selected == "aux55"
    assert reason == "insufficient_evidence_keep_current"


def test_stability_counts_flat_months_and_unrecovered_drawdown():
    ledger = pd.DataFrame(dict(exit_date=["20220131", "20220228", "20220331"],
                               equity=[510000., 490000., 490000.], pnl=[10000., -20000., 0.],
                               forced_exit=[False]*3, settlement_breach=[False]*3))
    stats = research.stability(ledger)
    assert stats["positive_month_ratio"] == 1/3
    assert stats["max_underwater_sessions"] == 2
