"""Causal loss marks, symmetric confirmation, rolls and year-end execution."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import evaluate_index_futures_risk_paths as research
from test_index_futures_funded import row


def scenario(prices, closes, side=1, roll=False):
    groups, opening, closing = [], {}, {}
    for i, price in enumerate(prices[:-1]):
        r = row(price=price, entry=f"202201{4+i:02}", exit=f"202201{5+i:02}")
        r.trade_date = f"202201{3+i:02}"
        r.exit_open = prices[i+1]
        if side < 0:
            r.long_60, r.short_60 = .1, .8
            r.long_20, r.short_20 = .1, .8
        if roll and i >= 1:
            r.ts_code = "IF2207"
        groups.append([r])
        opening[r.entry_date, r.ts_code] = price
        opening[r.exit_date, r.ts_code] = prices[i+1]
        closing[r.trade_date, "IF2206"] = closes[i]
        closing[r.trade_date, "IF2207"] = closes[i]
    return groups, opening, closing


def test_previous_close_mark_cancels_open_gap_and_contains_roll_costs():
    # Trade has already earned 10k after fees at yesterday's close of 1100.
    a = research.previous_close_pnl(510000, 500000, (1, 330000), 1100, 300, .0003)
    b = research.previous_close_pnl(540000, 500000, (1, 360000), 1100, 300, .0003)
    assert np.isclose(a, 9901) and a == b
    assert np.isclose(research.previous_close_pnl(509800, 500000, (1, 330000), 1100, 300, .0003), a-200)


def test_roll_preserves_trade_loss_basis_and_charges_both_legs():
    groups, opens, closes = scenario([1000, 1010, 1020, 1030], [1000, 1005, 1015], roll=True)
    ledger, _, trace = research.risk_replay(groups, opens, closes, .0003, .15, variant="loss_exit", trace=True)
    assert ledger.positions.tolist() == ["IF2206:+1", "IF2207:+1", "IF2207:+1"]
    # Entry fee is 1000*300*.0003 = 90; roll costs another 181.8.
    assert np.isclose(trace.close_pnl.iloc[1], 1319.55)
    assert np.isclose(trace.close_pnl.iloc[2], 4136.85)
    assert np.isclose(ledger.fees.iloc[1], 181.8)


def test_loss_exit_waits_for_previous_close_and_does_not_use_future_exit():
    groups, opens, closes = scenario([1000, 800, 900, 400], [1000, 990, 800])
    ledger, _, trace = research.risk_replay(groups, opens, closes, 0, .15, variant="loss_exit", trace=True)
    assert trace.action.tolist() == ["enter", "hold", "close_loss_exit"]
    assert ledger.positions.tolist() == ["IF2206:+1", "IF2206:+1", ""]
    # Day 2's opening gap already exceeded 5%, but yesterday's close had not.
    assert trace.close_pnl.iloc[1] == -3000


def test_confirmed_signal_exit_is_symmetric_and_requires_two_closes():
    for side in (1, -1):
        groups, opens, closes = scenario([1000]*5, [1000]*4, side)
        for i in (1, 2, 3):
            r = groups[i][0]
            r.short_20, r.long_20 = (.7, .2) if side > 0 else (.2, .7)
        ledger, _, trace = research.risk_replay(groups, opens, closes, 0, .15, variant="signal_exit", trace=True)
        assert trace.action.iloc[1] == "hold"
        assert trace.action.iloc[2] == "confirmed_opposing_exit"
        assert ledger.gross_lots.tolist() == [1, 1, 0, 0]


def test_risk_filter_can_choose_lower_exposure_and_is_side_symmetric():
    a, b = row("IF", 5000, .9), row("IH", 2000, .8)
    groups = [[a, b]]
    ledger, _, trace = research.risk_replay(groups, {}, {}, 0, .15, variant="entry_risk", trace=True)
    assert ledger.positions.tolist() == ["IH2206:+1"]
    assert trace.affordable_candidates.iloc[0] == 2 and trace.risk_candidates.iloc[0] == 1
    for r in (a, b):
        r.long_60, r.short_60 = r.short_60, r.long_60
    ledger, _, _ = research.risk_replay(groups, {}, {}, 0, .15, variant="entry_risk")
    assert ledger.positions.tolist() == ["IH2206:-1"]


def test_reserve_exit_is_distinct_from_forced_margin_exit():
    groups, opens, closes = scenario([5000, 4500, 4500], [5000, 4500])
    ledger, _, trace = research.risk_replay(groups, opens, closes, 0, .25, capital=550000, trace=True)
    assert trace.action.tolist() == ["enter", "reserve_exit"]
    assert ledger.gross_lots.tolist() == [1, 0]
    assert not ledger.forced_exit.any()


def test_minimum_equity_matches_admission_boundary():
    for direction in (-1, 1):
        r = row(price=4500)
        boundary = research.minimum_equity(r, direction, .0006, .25)
        assert research.affordable(r, direction, boundary+.01, .0006, .25)
        assert not research.affordable(r, direction, boundary-.01, .0006, .25)


def test_first_year_closes_once_and_preserves_causal_prefix():
    groups, opens, closes = scenario([1000]*4, [1000]*3)
    groups[1][0].exit_date = "20230103"
    groups[2][0].entry_date, groups[2][0].exit_date = "20230103", "20230104"
    groups[2][0].quarter = "2023Q1"
    for day in ("20230103", "20230104"):
        opens[day, "IF2206"] = 1000
    ledger, _, _ = research.risk_replay(groups, opens, closes, .0003, .15)
    prefix = research.first_year(ledger, "20220101", opens, .0003)
    direct, _, _ = research.risk_replay(groups[:1], opens, closes, .0003, .15)
    assert np.isclose(prefix.pnl.sum(), direct.pnl.sum())
    assert np.isclose(prefix.fees.sum(), 180)
    # Full sample is already liquidated; do not charge another closing fee.
    whole = research.first_year(direct, "20220101", opens, .0003)
    assert whole.fees.sum() == direct.fees.sum()
