"""Causal labels, evidence fallback, deadline locking and executable giveback."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research"))
from index_futures_state_rules import evidence_admits,holding_limit,profit_giveback
from index_futures_state_models import entry_path,add_regime_features
from evaluate_index_futures_risk_paths import risk_replay
from test_index_futures_risk_paths import scenario


def test_evidence_permits_countertrend_without_hard_state_ban():
    assert evidence_admits(.6,.02,.01)
    assert not evidence_admits(.54,.02,.01)
    assert not evidence_admits(.7,.004,.01)
    assert not evidence_admits(.7,.005,.01)
    assert evidence_admits(np.nan,np.nan,np.nan)
    with pytest.raises(ValueError):evidence_admits(np.nan,.01,.01)


def test_symmetric_deadlines_and_trailing_rules():
    for side in (1,-1):
        assert holding_limit(side,side)==20
        assert holding_limit(0,side)==10
        assert holding_limit(-side,side)==5
        assert not profit_giveback(side,side,24999,1000,500000)
        assert not profit_giveback(side,side,30000,16000,500000)
        assert profit_giveback(0,side,30000,20000,500000)
        assert not profit_giveback(0,side,30000,20000,500000,False)


def test_entry_label_includes_both_roll_legs_and_does_not_use_later_prices():
    groups,_,_=scenario([1000,1010,1020,1030,1040,1050,9000],[1000]*6,roll=True)
    rows=[g[0] for g in groups]
    closes={(r.entry_date,r.ts_code):r.entry_open for r in rows}
    a=entry_path(rows,0,1,.001,closes)
    assert a['maturity']==rows[4].exit_date
    # 50 points gross, opening1000 + roll1010*2 + final1050 fees, multiplier cancels.
    assert np.isclose(a['net'],(50-.001*(1000+2020+1050))/1000)
    assert a['risk']>0
    rows[-1].exit_open=-100000
    assert entry_path(rows,0,1,.001,closes)==a
    assert entry_path(rows,3,1,.001,closes) is None


def test_entry_adverse_label_includes_roll_open_fee_before_recovery():
    groups,_,_=scenario([1000,1000,1020,1030,1040,1050],[1000]*5,roll=True)
    rows=[g[0] for g in groups]
    closes={(r.entry_date,r.ts_code):r.exit_open for r in rows}
    target=entry_path(rows,0,1,.001,closes)
    # Flat first overnight, then roll charges old close + new open, and a
    # hypothetical new close: total4 price units before the day's recovery.
    assert np.isclose(target['risk'],.004)


def test_model_filter_really_allows_an_inverse_regime_entry():
    groups,opens,closes=scenario([1000]*3,[1000]*2)
    r=groups[0][0]
    states={(g[0].trade_date,'IF'):-1 for g in groups}
    lookup={(r.entry_date,'IF',1,0,'interaction'):(.7,.02,.01)}
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,
        evidence_model=3,evidence_lookup=lookup,market_regime=states)
    assert ledger.positions.iloc[0]=='IF2206:+1'
    assert trace.entry_probability.iloc[0]==.7 and trace.regime_state.iloc[0]==-1


def test_entry_deadline_cannot_be_extended_by_later_state_change_or_roll():
    groups,opens,closes=scenario([1000]*13,[1000]*12,roll=True)
    states={(g[0].trade_date,'IF'):(-1 if i==0 else 1) for i,g in enumerate(groups)}
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,
        soft_exit=4,market_regime=states,entry_calendar={groups[0][0].entry_date})
    assert ledger.gross_lots.iloc[:5].eq(1).all()
    assert trace.action.iloc[5]=='state_time_exit'
    assert ledger.gross_lots.iloc[5:].eq(0).all()


def test_trailing_uses_previous_close_not_future_open_and_includes_gap_loss():
    groups,opens,closes=scenario([1000,1120,800,800],[1000,1120,800])
    states={(g[0].trade_date,'IF'):1 for g in groups}
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,soft_exit=6,market_regime=states)
    # At the second open the following overnight loss is not yet known.
    assert trace.action.iloc[1]=='hold'
    assert trace.profit_peak.iloc[1]==36000
    assert trace.action.iloc[2]=='profit_giveback_exit'
    assert np.isclose(ledger.pnl.sum(),-60000) # Gap exceeds any promised profit lock.


def test_state_trail_tightens_without_resetting_peak():
    groups,opens,closes=scenario([1000,1120,1070,1070],[1000,1120,1070])
    states={(g[0].trade_date,'IF'):(1 if i<2 else 0) for i,g in enumerate(groups)}
    adaptive,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,soft_exit=6,market_regime=states)
    uniform,_,other=risk_replay(groups,opens,closes,0,.15,trace=True,soft_exit=5,market_regime=states)
    assert trace.action.iloc[2]=='profit_giveback_exit' and other.action.iloc[2]=='hold'
    assert trace.profit_peak.iloc[2]==36000


def test_state_model_exit_requires_two_finite_observations():
    groups,opens,closes=scenario([1000]*6,[1000]*5)
    states={(g[0].trade_date,'IF'):1 for g in groups}
    entry=groups[0][0].entry_date
    lookup={(entry,g[0].entry_date,'IF','interaction'):SimpleNamespace(direction=1,pred_net_5_0=n,pred_risk_5_0=0)
            for g,n in zip(groups,[0,-1,np.nan,-1,-1])}
    _,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,soft_exit=2,market_regime=states,soft_lookup=lookup)
    assert trace.action.tolist()==['enter','hold','hold','hold','state_model_exit']


def test_regime_feature_merge_is_point_in_time_and_has_only_predeclared_interactions():
    from index_futures_state_models import INTERACTIONS
    frame=pd.DataFrame([dict(trade_date='20240122',symbol='IM',**{f:1. for f in INTERACTIONS})])
    a=add_regime_features(frame,'trade_date')
    assert a.regime_bear.iloc[0]==1 and a.regime_bull.iloc[0]==0
    assert len(a.filter(regex='^interaction_').columns)==16
    assert a.interaction_regime_bear_f_momentum_5.iloc[0]==1
