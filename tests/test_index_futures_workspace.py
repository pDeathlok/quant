"""Serving guards: closed-bar prices, policy/financing gates and drawdown origin."""
import sys
from datetime import date, datetime, timedelta
from subprocess import CalledProcessError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/research'))
import build_index_futures_signal as signal
from build_index_futures_signal import evaluate_candidates
from build_index_futures_workspace import recent_performance
from quant.webapp import index_futures as service


def predictions():
    return pd.DataFrame([dict(symbol='IF',ts_code='IF2612.CFX',trade_date='20261113',
        close=4000.,entry_open=999999.,multiplier=300,f_volatility_20=.01,etf_ret_1=.001,
        long_60=.58,short_60=.30,win_long=.6,win_short=.4,state=1,direction=side,
        probability=.65,pred_net=.03,pred_risk=.01,fallback='none',
        panic_common_source_date='20261112',panic_own_source_date='20261112') for side in (1,-1)])


def test_serving_uses_known_close_and_exact_minimum_equity():
    data=predictions()
    item=evaluate_candidates(data,'base',.0006,.25)[0]
    assert item['passes'] and item['price']==4000
    assert item['margin_cash']==300000
    minimum=item['minimum_equity']
    assert evaluate_candidates(data,'base',.0006,.25,minimum)[0]['passes']
    assert not evaluate_candidates(data,'base',.0006,.25,minimum-.01)[0]['passes']
    assert not evaluate_candidates(data,'confidence60',.0006,.25)[0]['passes']
    data.entry_open=-10  # A next-open column cannot change a close-based preview.
    assert evaluate_candidates(data,'base',.0006,.25)[0]==item


def test_missing_forecast_fails_closed_and_short_has_larger_stress_margin():
    data=predictions()
    long=evaluate_candidates(data,'base',.0006,.25)[0]
    data[['long_60','short_60']]=[.3,.58]
    short=evaluate_candidates(data,'base',.0006,.25)[0]
    assert short['side']=='short' and short['minimum_equity']>long['minimum_equity']
    assert short['holding_limit']==5
    data[['probability','pred_net','pred_risk']]=np.nan
    assert not evaluate_candidates(data,'base',.0006,.25)[0]['passes']


def test_recent_drawdown_includes_loss_on_first_day():
    ledger=pd.DataFrame(dict(entry_date=['20261102','20261103'],exit_date=['20261103','20261104'],
        equity=[450000.,475000.],pnl=[-50000.,25000.],gross_lots=[1,0]))
    result=recent_performance(ledger)[0]
    assert result['total_return']==pytest.approx(-.05)
    assert result['max_drawdown']==pytest.approx(-.10)
    assert result['net_pnl']==-25000 and result['flat_ratio']==.5


def test_refresh_failure_is_explicit_and_never_claims_success(monkeypatch,tmp_path):
    monkeypatch.setattr(service,'ROOT',tmp_path)
    def fail(*args,**kwargs): raise RuntimeError('private-provider-detail')
    monkeypatch.setattr(service.subprocess,'run',fail)
    service._refresh()
    assert service.refresh_status()['status']=='failed'
    assert 'private-provider-detail' not in service.refresh_status()['message']


def test_duplicate_refresh_is_coalesced(monkeypatch):
    monkeypatch.setattr(service,'_state',dict(status='running'))
    def fail(*args,**kwargs): raise AssertionError('must not start second thread')
    monkeypatch.setattr(service.threading,'Thread',fail)
    assert service.start_refresh()['status']=='running'


def test_missing_target_session_preserves_previous_signal(monkeypatch, tmp_path):
    current = tmp_path / 'signal.json'
    current.write_text('previous signal')
    monkeypatch.setattr(signal, 'LIVE', tmp_path)
    monkeypatch.setattr(signal, 'feature_panel', lambda: pd.DataFrame(
        {'symbol': ['IF', 'IH'], 'trade_date': ['20261116', '20261113']}))
    monkeypatch.setattr(signal.pd, 'read_parquet', lambda _: pd.DataFrame(
        {'cal_date': ['20261113', '20261116', '20261117']}))

    with pytest.raises(RuntimeError, match='available=20261113'):
        signal.build('20261116')
    assert current.read_text() == 'previous signal'


@pytest.mark.parametrize("publish", [True, False])
def test_daily_preview_waits_for_data_and_retries(monkeypatch, tmp_path, publish):
    day = date(2026, 11, 16)
    now = datetime(2026, 11, 16, 17, 55)
    calls = []

    def sleep(seconds):
        nonlocal now
        now += timedelta(seconds=seconds)

    def preview(root, python, trade_date):
        calls.append((now.strftime("%H:%M"), trade_date))
        if not publish or len(calls) == 1:
            raise CalledProcessError(1, "build_index_futures_signal")

    monkeypatch.setattr(service, "market_now", lambda: now)
    monkeypatch.setattr(service, "time", type("Clock", (), {"sleep": staticmethod(sleep)}))
    monkeypatch.setattr(service, "refresh_preview", preview)
    result = service.refresh_daily_preview(tmp_path, day, lambda _: None)
    assert calls[:2] == [("18:00", day), ("18:10", day)]
    if publish:
        assert result == {"status": "success", "signal_date": day.isoformat(), "attempts": 2}
    else:
        assert result["status"] == "failed"
        assert calls[-1] == ("19:00", day)
