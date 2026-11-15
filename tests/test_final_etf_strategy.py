"""Causality and execution checks for the frozen final ETF research policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/research"))
import select_final_etf_strategy as strategy


def synthetic_panel() -> strategy.Panel:
    dates = pd.bdate_range("2020-01-01", "2022-12-30")
    rng = np.random.default_rng(19)
    p = pd.DataFrame(10*np.exp(np.cumsum(rng.normal(0, .008, (len(dates), 8)),axis=0)),
                     index=dates, columns=strategy.CODES)
    true = pd.DataFrame(True,index=dates,columns=strategy.CODES)
    amount = pd.DataFrame(1e9,index=dates,columns=strategy.CODES)
    nav_dates = pd.DataFrame({code: dates-pd.Timedelta(days=1) for code in strategy.CODES}, index=dates)
    return strategy.Panel(p.copy(),p.copy(),amount,true,true.copy(),p*0,nav_dates,nav_dates.copy(),amount.copy())


def test_nav_disclosure_must_precede_signal_and_prices_match_nav_date():
    dates = pd.to_datetime(["2021-01-04","2021-01-05","2021-01-06"])
    close = pd.Series([100.,110.,120.],index=dates)
    nav = pd.DataFrame({"nav_date":["20210104","20210105"],"ann_date":["20210104","20210106"],"unit_nav":[100.,100.]})
    known = strategy.known_nav_premium(nav,close)
    assert pd.isna(known.iloc[0].premium)
    assert known.iloc[1].premium == 0
    assert known.iloc[2].premium == 0  # Jan 6 announcement may occur after the close.


def test_lots_cash_and_volume_cap():
    panel = synthetic_panel(); i=panel.close.index.get_loc(strategy.START)
    targets={i-1:np.ones(8)/8}
    result, orders = strategy.simulate(panel,targets,initial=100_000)
    buy = orders[orders.side.eq("buy")]
    np.testing.assert_allclose(buy.shares/100,np.round(buy.shares/100),atol=1e-8)
    assert result.weights.sum(axis=1).max() <= 1+1e-10
    assert result.weights.min().min() >= 0
    assert (orders.notional <= .01*1e9).all()
    assert (orders.fee >= 5).all()
    assert (pd.to_datetime(orders.signal_date) < pd.to_datetime(orders.date)).all()


def test_unavailable_quote_cannot_execute():
    panel=synthetic_panel(); i=panel.close.index.get_loc(strategy.START)
    panel.valid.iloc[i,0]=False
    w=np.zeros(8);w[0]=1
    result,orders=strategy.simulate(panel,{i-1:w})
    assert orders.empty
    assert result.weights.sum().sum()==0
    assert len(result.skipped)==1


def test_execution_lag_does_not_capture_signal_to_fill_return():
    panel=synthetic_panel(); panel.close[:]=10;panel.adjusted[:]=10
    i=panel.close.index.get_loc(strategy.START)
    panel.close.iloc[i:,0]=20;panel.adjusted.iloc[i:,0]=20
    w=np.zeros(8);w[0]=1
    result,_=strategy.simulate(panel,{i-1:w})
    assert result.returns.iloc[i] < 0  # Only costs, no doubling before the purchase.


def test_liquidity_or_nav_rejection_goes_to_cash_without_reallocation():
    panel=synthetic_panel(); panel.eligible.loc[:,strategy.CODES[5]]=False
    targets,_=strategy.make_targets(panel,"defensive","fixed")
    for w in targets.values():
        assert w[5]==0
        np.testing.assert_allclose(w[:5],strategy.CONFIGS['defensive'][:5])
        assert abs(w.sum()-(1-1/30)) < 1e-10


def test_future_data_cannot_change_previous_targets():
    panel=synthetic_panel()
    before,_=strategy.make_targets(panel,"defensive","trendvol")
    cutoff=pd.Timestamp('2022-01-01')
    panel.adjusted.loc[panel.adjusted.index>=cutoff] *= 3
    after,_=strategy.make_targets(panel,"defensive","trendvol")
    for t in before:
        if panel.close.index[t]<cutoff: np.testing.assert_allclose(before[t],after[t],atol=1e-12)


def test_selection_prefers_simple_within_predeclared_tolerance():
    table=pd.DataFrame({'all_execution_cases_pass':[True,True], 'score':[.04,.042],
                        'complexity':[0,2],'annual_traded_notional':[1,2],
                        'max_drawdown':[-.06,-.04], 'worst_day':[-.03,-.02]},index=['fixed','complex'])
    assert strategy.select_candidate(table)==('fixed',True)
