"""Behavioral checks for causality, censoring and fill timing in scan research."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1]/"scripts/research/validate_scan_a_shares.py"
spec = importlib.util.spec_from_file_location("scan_validation",PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_future_split_does_not_rewrite_signal_history():
    dates=pd.date_range("2020-01-01",periods=3)
    f=pd.DataFrame({"date":dates,"close":[10.,11.,6.],"pre_close":[10.,10.,5.5],
                    "open":[10.,10.,5.5],"high":[10.,11.,6.],"low":[10.,10.,5.5]})
    short=module.continuous(f.iloc[:2])
    full=module.continuous(f)
    pd.testing.assert_frame_equal(short,full.iloc[:2])
    assert full.close.iloc[-1]==12.


def test_incomplete_future_is_censored_not_failure():
    x=module.forward_stats(10.,np.array([10.,30.]),3)
    assert x["mature"] is False
    assert x["peak_multiple"]==3.
    assert np.isnan(x["terminal_multiple"])


def fixture(prices: list[float], opens: list[float] | None = None):
    dates=pd.bdate_range("2020-01-01",periods=len(prices))
    c=pd.DataFrame({"000001.SZ":prices},index=dates)
    o=pd.DataFrame({"000001.SZ":opens or prices},index=dates)
    hi=np.maximum(c,o);lo=np.minimum(c,o)
    observed=pd.DataFrame(True,index=dates,columns=c.columns)
    atr=pd.DataFrame(.2,index=dates,columns=c.columns)
    signals=pd.DataFrame({"signal_date":[dates[0]],"symbol":["000001.SZ"]})
    return dates,c,o,hi,lo,observed,atr,signals


def test_stop_only_sells_next_open_and_allows_T_plus_1():
    dates,c,o,hi,lo,obs,atr,signals=fixture([10.,9.,8.7,8.6],[10.,10.,8.5,8.6])
    nav,trades=module.simulate(c,o,hi,lo,obs,atr,signals,module.ExitRule("test",rotation=True),{},str(dates[0].date()),str(dates[-1].date()))
    assert nav.positions.iloc[0]==0
    assert nav.positions.iloc[1]==1
    # Price close below stop on entry day, but no same-day exit. The next open
    # gapped down >4.8%, conservatively blocked; exit at the following open.
    assert trades.entry_date.iloc[0]==dates[1]
    assert trades.exit_date.iloc[0]==dates[3]
    assert nav.blocked_sell_cumulative.iloc[-1]>=1


def test_no_entry_on_large_gap():
    dates,c,o,hi,lo,obs,atr,signals=fixture([10.,11.,11.],[10.,11.,11.])
    nav,trades=module.simulate(c,o,hi,lo,obs,atr,signals,module.ExitRule("test"),{},str(dates[0].date()),str(dates[-1].date()))
    assert nav.positions.max()==0
    assert nav.blocked_buy_cumulative.iloc[-1]==1
    assert trades.empty


def test_later_bars_cannot_change_existing_features():
    dates=pd.bdate_range("2020-01-01",periods=150)
    c=pd.DataFrame({"x":np.arange(150)+100.},index=dates)
    amount=pd.DataFrame(60000.,index=dates,columns=["x"])
    obs=pd.DataFrame(True,index=dates,columns=["x"])
    before=module.features(c.iloc[:130],amount.iloc[:130],obs.iloc[:130],c.iloc[:130]+1,c.iloc[:130]-1)
    c.iloc[130:]*=1000
    after=module.features(c,amount,obs,c+1,c-1)
    for key in before:
        pd.testing.assert_frame_equal(before[key],after[key].iloc[:130])


def test_delisted_holding_not_marked_forever_at_old_price():
    dates,c,o,hi,lo,obs,atr,signals=fixture([10.,10.,10.,10.])
    nav,trades=module.simulate(c,o,hi,lo,obs,atr,signals,module.ExitRule("test",rotation=True),{"000001.SZ":dates[2]},str(dates[0].date()),str(dates[-1].date()))
    assert nav.delist_writeoffs.iloc[-1]==1
    assert trades.net_multiple.iloc[0]==0
    assert nav.nav.iloc[-1]<.98


def test_cash_plus_positions_reconciles_to_known_buy_hold_return():
    dates,c,o,hi,lo,obs,atr,signals=fixture([10.,20.,20.],[10.,10.,20.])
    nav,trades=module.simulate(c,o,hi,lo,obs,atr,signals,module.ExitRule("test",rotation=True),{},str(dates[0].date()),str(dates[-1].date()),cost=0.)
    # One slot gets 2.5% initial equity at 10, doubles at first entry-day close.
    assert abs(nav.nav.iloc[-1]-1.025)<1e-12
    assert abs(nav.cash.iloc[-1]-.975)<1e-12
    assert trades.reason.iloc[-1]=="open_at_end"
    assert np.isnan(trades.net_multiple.iloc[-1])


def test_bull_denominator_does_not_require_scanner_trend_gate():
    dates=pd.bdate_range("2020-01-01",periods=150)
    c=pd.DataFrame({"x":np.arange(150)[::-1]+100.},index=dates)
    amount=pd.DataFrame(60000.,index=dates,columns=["x"])
    obs=pd.DataFrame(True,index=dates,columns=["x"])
    result=module.features(c,amount,obs,c+1,c-1)
    assert result["base_eligible"].iloc[-1,0]
    assert not result["eligible"].iloc[-1,0]


def test_future_name_announcement_cannot_change_previous_status():
    dates=pd.bdate_range("2020-01-01",periods=10)
    base=pd.DataFrame({"ts_code":["x"],"name":["正常股"],"start_date":["20190101"],"end_date":[None],"ann_date":["20181231"]})
    future=pd.DataFrame({"ts_code":["x"],"name":["*ST异常"],"start_date":["20200103"],"end_date":[None],"ann_date":["20200110"]})
    known_before,st_before=module.historical_status(base,dates,["x"])
    known_after,st_after=module.historical_status(pd.concat([base,future]),dates,["x"])
    cut=pd.Timestamp("2020-01-09")
    pd.testing.assert_frame_equal(known_before.loc[:cut],known_after.loc[:cut])
    pd.testing.assert_frame_equal(st_before.loc[:cut],st_after.loc[:cut])
    assert st_after.loc["2020-01-10","x"]


def test_unknown_historical_status_cannot_buy():
    dates,c,o,hi,lo,obs,atr,signals=fixture([10.,10.,10.])
    allowed=obs&False
    nav,trades=module.simulate(c,o,hi,lo,obs,atr,signals,module.ExitRule("test"),{},str(dates[0].date()),str(dates[-1].date()),entry_allowed=allowed)
    assert nav.positions.max()==0
    assert trades.empty
