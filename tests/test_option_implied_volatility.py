import numpy as np
import pandas as pd
from quant.features.option_implied_volatility import black_price, implied_volatility, term_volatility, interpolate_30


def chain(sigma=.25,days=30,forward=5000,rate=.02):
    rows=[]
    for strike in np.arange(2500,7501,25):
        for cp in ('C','P'):
            price=black_price(forward,strike,days/365,rate,sigma,cp=='C')
            rows.append(dict(strike=float(strike),call_put=cp,close=price,vol=100,oi=100,low=price*.9,high=price*1.1))
    return pd.DataFrame(rows)


def test_black_inversion_and_no_arbitrage_bounds():
    for side in (True,False):
        p=black_price(5000,5100,30/365,.02,.25,side)
        assert abs(implied_volatility(p,5000,5100,30/365,.02,side)-.25)<1e-8
    assert np.isnan(implied_volatility(6000,5000,5000,.1,0,True))
    assert np.isnan(implied_volatility(0,5000,5000,.1,0,True))


def test_flat_surface_recovers_volatility_and_scale_invariance():
    f=chain();result=term_volatility(f,30,.02)
    assert result['reason']=='ok'
    assert abs(result['forward']-5000)<1e-8
    assert abs(np.sqrt(result['mf_variance'])-.25)<.001
    assert abs(np.sqrt(result['atm_variance'])-.25)<1e-8
    f[['strike','close','low','high']]*=2
    other=term_volatility(f,30,.02)
    assert np.isclose(other['mf_variance'],result['mf_variance'])


def test_total_variance_interpolation_and_no_extrapolation():
    f=pd.DataFrame(dict(days=[20,50],mf_variance=[.04,.09]))
    expected=(20*.04*20+50*.09*10)/30/30
    r=interpolate_30(f,'mf_variance')
    assert np.isclose((r['value']/100)**2,expected)
    assert np.isnan(interpolate_30(f.iloc[1:],'mf_variance')['value'])
    f.loc[0,'days']=6
    assert np.isnan(interpolate_30(f,'mf_variance')['value'])


def test_zero_volume_and_duplicate_chains_do_not_create_surface():
    f=chain();f['vol']=0
    assert np.isnan(term_volatility(f,30,.02)['mf_variance'])
    f=chain();f=pd.concat([f,f.iloc[:1]])
    assert term_volatility(f,30,.02)['reason']=='duplicate_strike_side'


def test_parity_corruption_rejected_without_repair():
    f=chain();f=f[f.strike.between(4950,5050)].copy()
    selected=f.call_put.eq('C')&f.strike.between(4950,5050)
    f.loc[selected,'close']+=np.array([100,0,100,0,100])
    f['high']=f.close*1.1
    result=term_volatility(f,30,.02)
    assert result['reason']=='parity_inconsistent'
    assert np.isnan(result['mf_variance']) and np.isnan(result['atm_variance'])


def test_atm_can_survive_missing_wings_but_variance_cannot():
    f=chain();f=f[f.strike.between(4950,5050)]
    result=term_volatility(f,30,.02)
    assert np.isnan(result['mf_variance'])
    assert abs(np.sqrt(result['atm_variance'])-.25)<1e-8
