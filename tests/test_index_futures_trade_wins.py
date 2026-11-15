"""Guard label chronology, roll costs and completed-trade annual attribution."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "research"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("trade_wins", SCRIPTS / "evaluate_index_futures_trade_wins.py")
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_label_counts_roll_cost_not_price_level_jump():
    frame = pd.DataFrame(dict(entry_date=["20240102", "20240103"],
                              exit_date=["20240103", "20240104"],
                              ts_code=["IF2401", "IF2402"],
                              entry_open=[100., 200.], exit_open=[101., 201.]))
    # Gross gain 2; fees 0.6% * (100 + 101 + 200 + 201) = 3.612.
    labelled = research.profit_labels(frame, days=2, fee=.006)
    assert labelled.iloc[0].profit_long == 0
    assert labelled.iloc[0].profit_short == 0
    assert labelled.iloc[0].profit_end == "20240104"
    assert pd.isna(labelled.iloc[1].profit_end)
    assert research.profit_labels(frame, days=2, fee=0).iloc[0].profit_long == 1


def test_label_rejects_calendar_gap():
    frame = pd.DataFrame(dict(entry_date=["20240102", "20240104"],
                              exit_date=["20240103", "20240105"],
                              ts_code=["IF2401", "IF2401"],
                              entry_open=[100., 101.], exit_open=[101., 102.]))
    assert research.profit_labels(frame, days=2).profit_end.isna().all()


def test_annual_equity_continues_but_trade_belongs_to_exit_year():
    ledger = pd.DataFrame(dict(exit_date=["20231229", "20240102"],
                               equity=[510000., 520000.], forced_exit=[False, False], fees=[1., 1.]))
    trades = pd.DataFrame(dict(entry=["20231228"], exit=["20240102"],
                               pnl=[20000.], side=["long"]))
    annual = pd.DataFrame(research.annual_stats(ledger, trades)).query("side == 'all'")
    assert annual.trades.tolist() == [0, 1]
    assert np.isclose(annual.iloc[1].total_return, 520000 / 510000 - 1)
    assert np.isnan(annual.iloc[0].win_rate)
    assert annual.iloc[1].win_rate == 1


def test_breakeven_is_not_a_win_and_payoff_differs_from_profit_factor():
    stats = research.trade_stats(pd.DataFrame(dict(pnl=[100., 100., -50., 0.])))
    assert stats["win_rate"] == .5
    assert stats["breakeven"] == 1
    assert stats["payoff_ratio"] == 2
    assert stats["profit_factor"] == 4
    assert stats["win_ci_low"] < .5 < stats["win_ci_high"]
