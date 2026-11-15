"""Capital constraints, cost-aware switch hurdles, and cumulative drawdown."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("funded", SCRIPTS / "refine_index_futures_adaptation.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def row(symbol="IF", price=1000., confidence=.8, entry="20220104", exit="20220105"):
    values = dict(entry_date=entry, exit_date=exit, trade_date="20211231", quarter="2022Q1",
                  symbol=symbol, ts_code=symbol+"2206", entry_open=price, exit_open=price,
                  entry_settle=price, multiplier=300 if symbol in ("IF", "IH") else 200,
                  f_volatility_20=.01, etf_ret_1=.01, long_60=confidence, short_60=.1,
                  long_20=.8, short_20=.1, win_long=.8, win_short=.2, flat=0.)
    for product in ("IF", "IH", "IC", "IM"):
        values[f"long_{product}"] = float(product == symbol)
        values[f"short_{product}"] = -float(product == symbol)
    return SimpleNamespace(**values)


def scores(**aux):
    base = dict(days=500, trades=10, active_days=200, forced=0, total_return=.2,
                weighted_mean=.001, max_drawdown=-.2, weighted_sortino=1., weighted_pnl=100.)
    candidate = dict(base, weighted_sortino=1.4, weighted_pnl=150., max_drawdown=-.15)
    candidate.update(aux)
    return {"base": base, "aux55": candidate}


def test_skip_unaffordable_top_signal_and_take_affordable_second():
    rows = [row("IM", 8000, .9), row("IH", 2000, .8)]
    old, _ = research.funded_replay([rows], .0003, .15, capital=200000, funded=False)
    new, _ = research.funded_replay([rows], .0003, .15, capital=200000, funded=True)
    assert old.iloc[0].positions == ""
    assert new.iloc[0].positions == "IH2206:+1"
    assert np.isclose(new.fees.sum(), 360.)


def test_opening_cost_is_reserved_before_admission():
    target = row(price=1000)
    # At zero fee, equity exactly meets the stress reserve inequality:
    # .8*equity = .05*300000 + .15*.95*300000.
    boundary = (15000+42750)/.8
    assert research.affordable(target, 1, boundary, 0., .15)
    assert not research.affordable(target, 1, boundary, .0003, .15)


def test_selector_sees_current_open_equity_not_future_exit_pnl():
    first = row()
    first.exit_open = 1100.
    second = row(price=1100., entry="20220105", exit="20220106")
    second.exit_open = 10000.
    seen = []
    def select(day, positions, equity):
        seen.append((day, equity))
        return ("long_IF", .5, 1, "both", None)
    research.simulate([[first], [second]], "flat", .5, 1, "both", 1, .0003,
                      capital=500000, lot_cap=1, state_policy_selector=select)
    assert seen == [("20220104", 500000), ("20220105", 529910)]


def test_small_risk_adjusted_advantage_does_not_trigger_switch():
    selected, reason, _ = research.decide(scores(weighted_sortino=1.1), "base", [row()], {},
                                          500000, .0003, .15, True)
    assert selected == "base" and reason == "insufficient_sortino_advantage"


def test_historical_edge_must_cover_equity_and_cost_hurdles():
    selected, reason, detail = research.decide(scores(weighted_pnl=120), "base", [row()], {},
                                               500000, .0003, .15, True)
    assert selected == "base" and reason == "insufficient_net_advantage"
    assert detail["historical_quarter_edge"] == 1220
    assert detail["required_edge"] == 2500
    selected, reason, detail = research.decide(scores(), "base", [row()], {"IF2206": (1, 300000)},
                                               500000, .0003, .15, True)
    assert selected == "aux55" and reason == "passed_capital_and_switch_hurdles"
    assert np.isclose(detail["estimated_switch_cost"], 180)


def test_switch_is_blocked_when_no_contract_is_affordable_now():
    selected, reason, _ = research.decide(scores(), "base", [row()], {}, 10000, .0003, .15, False)
    assert selected == "base" and reason == "no_currently_affordable_contract"


def test_capital_quality_uses_actual_validation_capital():
    ledger = pd.DataFrame(dict(exit_date=["20220105", "20220106"], equity=[210000., 220000.],
                               pnl=[10000., 10000.], gross_lots=[1, 1], forced_exit=[False, False]))
    result = research.capital_quality(ledger, "20220401", 200000, 1)
    assert np.isclose(result["total_return"], .10)
    assert result["max_drawdown"] == 0
    assert np.isclose(result["weighted_pnl"], 10000)


def test_cumulative_year_table_retains_earlier_drawdown_and_original_capital():
    ledger = pd.DataFrame(dict(entry_date=["20220104", "20221229", "20231228", "20241230"],
                               exit_date=["20220105", "20221230", "20231229", "20241231"],
                               equity=[600000., 400000., 450000., 700000.],
                               pnl=[100000., -200000., 50000., 250000.], fees=[1., 1., 1., 1.],
                               forced_exit=[False]*4, settlement_breach=[False]*4))
    annual = pd.DataFrame(research.cumulative_years(ledger))
    np.testing.assert_allclose(annual.cumulative_return, [-.2, -.1, .4])
    np.testing.assert_allclose(annual.cumulative_drawdown, [-1/3]*3)
    assert annual.cumulative_fees.tolist() == [2., 3., 4.]
