"""Trend-state timing, symmetric gates, and independent entry/exit effects."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research"))
from index_futures_regime import trend_states, entry_allowed, exit_required
from evaluate_index_futures_risk_paths import risk_replay
from test_index_futures_funded import row
from test_index_futures_risk_paths import scenario


def bars(values):
    close=np.asarray(values,dtype=float)
    return pd.DataFrame(dict(trade_date=pd.bdate_range("20190101",periods=len(close)).strftime("%Y%m%d"),
                             close=close,high=close+1,low=close-1))


def test_states_are_prefix_invariant_and_warmup_never_backfilled():
    data=bars(np.arange(400)+100)
    first=trend_states(data.iloc[:300])
    changed=data.copy()
    changed.loc[300:,["close","high","low"]]*=10
    pd.testing.assert_frame_equal(first,trend_states(changed).iloc[:300])
    assert first.state.iloc[:219].isna().all()
    assert first.state.iloc[219:].eq(1).all()
    assert trend_states(bars(600-np.arange(400))).state.iloc[219:].eq(-1).all()


def test_flat_slope_and_buffer_are_transition_not_bull_or_bear():
    assert trend_states(bars([100]*300)).state.iloc[-1]==0
    assert trend_states(bars(100+np.arange(300)*.001)).state.iloc[-1]==0


def test_entry_and_exit_are_side_symmetric_and_missing_forecast_is_not_exit():
    for direction in (-1,1):
        assert entry_allowed(direction,direction)
        assert entry_allowed(0,direction)
        assert not entry_allowed(-direction,direction)
        assert exit_required(0,direction,-.001)
        assert exit_required(-direction,direction,0)
        assert not exit_required(direction,direction,-.1)
        assert not exit_required(-direction,direction,.01)
        assert not exit_required(-direction,direction,np.nan)


def test_entry_filter_can_substitute_another_index_without_forcing_cash():
    a,b=row("IF",1000,.9),row("IH",1000,.8)
    state={(a.trade_date,"IF"):-1,(b.trade_date,"IH"):1}
    ledger,_,trace=risk_replay([[a,b]],{}, {},0,.15,trace=True,regime_mode="entry",market_regime=state)
    assert ledger.positions.iloc[0]=="IH2206:+1"
    assert trace.regime_rejected.iloc[0]==1 and trace.regime_substitution.iloc[0]
    assert trace.raw_candidates.iloc[0]==2 and trace.affordable_candidates.iloc[0]==2


@pytest.mark.parametrize("side",[-1,1])
def test_exit_requires_continuation_evidence_and_does_not_reverse(side):
    groups,opens,closes=scenario([1000]*6,[1000]*5,side)
    states={(g[0].trade_date,"IF"):0 for g in groups}
    entry=groups[0][0].entry_date
    forecasts={(entry,g[0].entry_date,"IF"):SimpleNamespace(direction=side,pred_net_5_0=net)
               for g,net in zip(groups,[.01,.01,-.01,-.01,-.01])}
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,regime_mode="exit",
        market_regime=states,continuation_lookup=forecasts)
    assert trace.action.tolist()==["enter","hold","regime_exit","entry_schedule_wait","entry_schedule_wait"]
    assert ledger.gross_lots.tolist()==[1,1,0,0,0]


def test_entry_only_does_not_close_existing_position_when_regime_turns():
    groups,opens,closes=scenario([1000]*5,[1000]*4)
    states={(g[0].trade_date,"IF"):(1 if i==0 else -1) for i,g in enumerate(groups)}
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,regime_mode="entry",market_regime=states)
    assert ledger.gross_lots.eq(1).all()
    assert trace.action.tolist()==["enter","hold","hold","hold"]


def test_opportunity_attribution_accounts_for_exit_fees_and_other_positions():
    from report_index_futures_reentry_cash import attribution
    a=pd.DataFrame(dict(entry_date=["a","b","c"],gross_lots=[1,1,1],pnl=[100.,-40.,20.]))
    b=pd.DataFrame(dict(entry_date=["a","b","c"],gross_lots=[0,0,1],pnl=[-2.,0.,30.]))
    got=attribution(a,b)
    assert got["missed_positive_pnl"]==100
    assert got["avoided_negative_pnl"]==40
    assert got["new_flat_fees"]==-2
    assert got["other_path_change"]==10
    assert got["net_pnl_change"]==-52
