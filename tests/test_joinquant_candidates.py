import numpy as np
import pandas as pd
import polars as pl
from quant.features.joinquant_candidates import price_factors, financial_events, align_events, PRICE


def bars(n=280):
    x=np.arange(n,dtype=float)
    return pl.DataFrame({'symbol':['X']*n,'date':pd.date_range('2020-01-01',periods=n),
                         'open':10+x/100,'close':10.1+x/100,'high':10.2+x/100,'low':9.9+x/100,
                         'volume':100+x,'ret':np.sin(x)/100,'turnover_rate':1+x/100})


def test_price_prefix_causality_and_hand_calculations():
    d=bars();full=price_factors(d);prefix=price_factors(d.head(260))
    np.testing.assert_allclose(full.head(260).select(*PRICE).to_numpy(),prefix.select(*PRICE).to_numpy(),equal_nan=True)
    t=d['turnover_rate'].to_numpy()
    assert np.isclose(full['jq_DAVOL20'][-1],t[-20:].mean()/t[-120:].mean())
    assert full['jq_arron_up_25'][-1]==100
    assert full['jq_arron_down_25'][-1]==4
    assert full['jq_MFI14'][-1]==100
    assert full['jq_VOL240'][:239].null_count()==239


def test_financial_visibility_and_cumulative_ttm():
    raw=pd.DataFrame({'ts_code':['X']*4,'end_date':['20231231','20230930','20240930','20240930'],
                      'ann_date':['20240301','20231020','20241020','20241120'],'report_type':['1']*4,'cash':[100,70,90,999]})
    e=financial_events(raw,['cash'],ttm=True)
    keys=pd.DataFrame({'symbol':['X']*3,'date':pd.to_datetime(['20241020','20241021','20241121'])})
    a=align_events(keys,e)
    assert a.cash.tolist()==[100,120,120]
