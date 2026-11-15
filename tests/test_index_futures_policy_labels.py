"""Policy-label maturity, actual execution parity and past-only calibration."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
from index_futures_policy_labels import policy_exit_age,calibrate,affine_parameters
from index_futures_state_models import entry_path
from evaluate_index_futures_exit_value import exit_callback
from evaluate_index_futures_risk_paths import risk_replay
from test_index_futures_risk_paths import scenario


def forecasts(rows,values,side=1,bps=3):
    first=rows[0]
    return {(first.entry_date,r.entry_date,r.symbol):SimpleNamespace(direction=side,age=i,
        **{f'pred_net_5_{bps}':values[i-1],f'pred_risk_5_{bps}':.01},notional=300000)
        for i,r in enumerate(rows[1:],1)}


def test_policy_label_stops_at_causal_exit_and_ignores_prices_after_exit():
    groups,opens,closes=scenario([1000,1010,1020,1030,1040,9000],[1000]*5,roll=True)
    rows=[x[0] for x in groups];lookup=forecasts(rows,[-.01,np.nan,-.01,-.01])
    assert policy_exit_age(rows,0,1,3,10,0,lookup)==(4,'mean5')
    target=entry_path(rows,0,1,.0003,closes,4)
    rows[4].exit_open=-90000
    assert entry_path(rows,0,1,.0003,closes,4)==target
    assert policy_exit_age(rows,0,1,3,3,0,lookup)==(3,'deadline')


def test_native_policy_has_different_matched_target_and_censoring_is_not_zero():
    groups,_,_=scenario([1000]*7,[1000]*6)
    rows=[x[0] for x in groups];rows[2].short_20=.36
    lookup=forecasts(rows,[.01]*5)
    assert policy_exit_age(rows,0,1,3,5,1,lookup)==(2,'native_opposing')
    assert policy_exit_age(rows,0,1,3,5,0,lookup)==(5,'deadline')
    assert policy_exit_age(rows,0,1,3,10,0,lookup) is None


@pytest.mark.parametrize('limit,soft',[(5,8),(10,3),(20,0)])
@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('early',[False,True])
def test_matched_trade_net_equals_engine_with_rolls_and_actual_exit(limit,soft,native,early):
    prices=list(np.linspace(1000,1066,23))
    groups,opens,closes=scenario(prices,prices[:-1],roll=True)
    rows=[g[0] for g in groups]
    if native:rows[4].short_20=.4
    values=[-.01 if early and i>=2 else .01 for i in range(1,len(rows))]
    lookup=forecasts(rows,values)
    age,reason=policy_exit_age(rows,0,1,3,limit,int(native),lookup)
    label=entry_path(rows,0,1,.0003,closes,age)
    state={(r.trade_date,r.symbol):1 for r in rows}
    ledger,_,trace=risk_replay(groups,opens,closes,.0003,.15,capital=500000,trace=True,
        policy='opposing_exit' if native else 'base',soft_exit=soft,market_regime=state,
        entry_calendar={rows[0].entry_date},exit_decider=exit_callback(lookup,'mean5',.0003))
    assert ledger.gross_lots.sum()==age
    np.testing.assert_allclose(ledger.pnl.sum(),label['net']*rows[0].entry_open*300,atol=1e-6)
    assert trace.action.iloc[age] in ('learned_exit','native_opposing_exit','state_time_exit','holding_limit')


def calibration_frames():
    dates=pd.bdate_range('2023-01-01',periods=160)
    history=pd.DataFrame(dict(trade_date=dates.strftime('%Y%m%d'),maturity=(dates+pd.Timedelta(days=5)).strftime('%Y%m%d'),
        ready=True,quarter=dates.to_period('Q').astype(str),probability=.9,
        pred_net=.02,pred_risk=.005,net=np.tile([-.01,.005],80),risk=.02))
    test=history.iloc[-3:].copy();test['trade_date']='20240102'
    return history,test


def test_calibrator_uses_mature_past_only_and_corrects_known_bias():
    history,test=calibration_frames()
    a,info=calibrate(history,test,'20240101')
    assert info['calibration_ready'] and .45<a[0,0]<.55
    assert a[0,1]<.01 and a[0,2]>.01
    future=history.copy();future['trade_date']='20231229';future['maturity']='20240102'
    future['net']=1e9;future['risk']=1e9
    b,_=calibrate(pd.concat([history,future]),test,'20240101')
    np.testing.assert_array_equal(a,b)


def test_sparse_calibration_falls_back_and_affine_map_does_not_reverse_ranking():
    history,test=calibration_frames()
    a,info=calibrate(history.iloc[:25],test,'20240101')
    assert not info['calibration_ready']
    np.testing.assert_array_equal(a,test[['probability','pred_net','pred_risk']])
    x=np.linspace(-1,1,100)
    slope,_=affine_parameters(x,-x,np.ones(100))
    assert slope==0


def test_native_specific_evidence_respects_selected_policy():
    groups,opens,closes=scenario([1000]*4,[1000]*3)
    r=groups[0][0];key=(r.entry_date,'IF',1,0,'matched')
    lookup={(*key,0):(.8,.02,.01),(*key,1):(.4,-.01,.01)}
    states={(g[0].trade_date,'IF'):1 for g in groups}
    base,_,_=risk_replay(groups,opens,closes,0,.15,evidence_model=1,evidence_names=('matched',),
        evidence_lookup=lookup,market_regime=states,entry_calendar={r.entry_date})
    opposing,_,_=risk_replay(groups,opens,closes,0,.15,policy='opposing_exit',evidence_model=1,evidence_names=('matched',),
        evidence_lookup=lookup,market_regime=states,entry_calendar={r.entry_date})
    assert base.gross_lots.sum()>0 and opposing.gross_lots.sum()==0
