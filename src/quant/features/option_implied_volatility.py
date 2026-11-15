"""Auditable daily-close volatility proxies, not an official quote-based VIX.

Only European unadjusted index options are accepted upstream. Rates are
explicit inputs. No temporal interpolation, forward filling or outcome tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr


def black_price(forward: float, strike: float, years: float, rate: float,
                sigma: float, call: bool) -> float:
    scale = sigma*np.sqrt(years)
    d1 = np.log(forward/strike)/scale+scale/2
    d2 = d1-scale
    if call:
        return float(np.exp(-rate*years)*(forward*ndtr(d1)-strike*ndtr(d2)))
    return float(np.exp(-rate*years)*(strike*ndtr(-d2)-forward*ndtr(-d1)))


def implied_volatility(price: float, forward: float, strike: float,
                       years: float, rate: float, call: bool) -> float:
    if min(price,forward,strike,years)<=0 or not np.isfinite([price,forward,strike,years,rate]).all():
        return np.nan
    discount = np.exp(-rate*years)
    lower = discount*max((forward-strike) if call else (strike-forward),0)
    upper = discount*(forward if call else strike)
    if not lower<price<upper: return np.nan
    try:
        return float(brentq(lambda v:black_price(forward,strike,years,rate,v,call)-price,1e-5,5,xtol=1e-10))
    except ValueError:
        return np.nan


def term_volatility(chain: pd.DataFrame, days: int, rate: float,
                    price_column: str = "close") -> dict:
    """One expiry, parity-median forward, OTM variance strip and near-ATM IV.

    Quality constants are fixed before strategy replay: 7..120 days; >=3 paired
    traded strikes; parity90-10 spread <=1% forward; ATM within5%, >=3 strikes;
    MF >=3 OTM strikes each wing, >=2% coverage each wing, gaps <=3 grid steps.
    Excluding zero-volume rows is a proxy for freshness, not a zero-bid rule.
    """
    result = dict(mf_variance=np.nan,atm_variance=np.nan,reason="invalid_input",
                  forward=np.nan,parity_spread=np.nan,liquid_options=0,put_strikes=0,call_strikes=0)
    if not 7<=days<=120 or not np.isfinite(rate): return result
    required = ["strike","call_put",price_column,"vol","oi","low","high"]
    frame = chain[required].copy()
    if frame.duplicated(["strike","call_put"]).any():
        result["reason"] = "duplicate_strike_side";return result
    years = days/365.
    all_strikes = np.sort(frame.strike.dropna().unique())
    tick_grid = np.median(np.diff(all_strikes)) if len(all_strikes)>1 else np.nan
    valid = frame["strike"].gt(0)&frame[price_column].gt(0)&frame.vol.gt(0)&frame.oi.gt(0)
    valid &= np.isfinite(frame[["strike",price_column,"vol","oi"]]).all(axis=1)
    if price_column=="close":
        valid &= frame.close.ge(frame.low-1e-8)&frame.close.le(frame.high+1e-8)&frame.low.ge(0)&frame.high.ge(frame.low)
    frame = frame[valid]
    result["liquid_options"] = len(frame)
    prices = frame.pivot(index="strike",columns="call_put",values=price_column).sort_index()
    if not {"C","P"}.issubset(prices.columns):
        result["reason"] = "missing_side";return result
    pairs = prices.dropna(subset=["C","P"]).copy()
    if len(pairs)<3:
        result["reason"] = "few_pairs";return result
    central = pairs.loc[(pairs.C-pairs.P).abs().sort_values(kind="stable").index[:5]]
    forwards = central.index.to_numpy()+np.exp(rate*years)*(central.C-central.P).to_numpy()
    forward = float(np.median(forwards))
    spread = float((np.quantile(forwards,.9)-np.quantile(forwards,.1))/forward)
    result.update(forward=forward,parity_spread=spread)
    if forward<=0 or spread>.01 or not np.isfinite(forward):
        result["reason"] = "parity_inconsistent";return result
    near = pairs[np.abs(np.log(pairs.index/forward))<=.05]
    near = near.iloc[np.argsort(np.abs(np.log(near.index/forward)),kind="stable")[:5]]
    ivs = [implied_volatility(r.C if k>=forward else r.P,forward,k,years,rate,k>=forward)
           for k,r in near.iterrows()]
    ivs = np.asarray(ivs);ivs=ivs[np.isfinite(ivs)]
    if len(ivs)>=3:
        result["atm_variance"] = float(np.median(ivs)**2)
    below = pairs.index[pairs.index<=forward]
    if not len(below):
        result["reason"] = "no_k0";return result
    k0 = float(max(below))
    # Use only observed OTM side at each strike, both sides at K0.
    puts = prices.loc[prices.index<k0,"P"].dropna()
    calls = prices.loc[prices.index>k0,"C"].dropna()
    result.update(put_strikes=len(puts),call_strikes=len(calls),k0=k0)
    if len(puts)<3 or len(calls)<3:
        result["reason"] = "few_wing_strikes";return result
    q = pd.concat([puts,pd.Series({k0:float(pairs.loc[k0,["C","P"]].mean())}),calls]).sort_index()
    strikes = q.index.to_numpy(dtype=float)
    if strikes[0]/forward>.98 or strikes[-1]/forward<1.02:
        result["reason"] = "narrow_wings";return result
    if not np.isfinite(tick_grid) or np.max(np.diff(strikes))>3*tick_grid+1e-8:
        result["reason"] = "sparse_strike_grid";return result
    # OTM bounds and gross monotonicity failures reveal stale/asynchronous closes.
    if (puts.diff().dropna()<-.4).any() or (calls.diff().dropna()>.4).any():
        result["reason"] = "nonmonotone_wing";return result
    if (puts.to_numpy()>np.exp(-rate*years)*puts.index).any() or (calls>np.exp(-rate*years)*forward).any():
        result["reason"] = "price_bound";return result
    dk = np.r_[strikes[1]-strikes[0],(strikes[2:]-strikes[:-2])/2,strikes[-1]-strikes[-2]]
    contributions = dk/strikes**2*np.exp(rate*years)*q.to_numpy()
    variance = 2/years*contributions.sum()-(forward/k0-1)**2/years
    result.update(min_moneyness=float(strikes[0]/forward),max_moneyness=float(strikes[-1]/forward),
                  tail_share=float((contributions[0]+contributions[-1])/contributions.sum()))
    if variance<=0 or not np.isfinite(variance):
        result["reason"] = "nonpositive_variance";return result
    result.update(mf_variance=float(variance),reason="ok")
    return result


def interpolate_30(terms: pd.DataFrame, field: str) -> dict:
    """Interpolate total variance, not volatility; no expiry extrapolation."""
    result = dict(value=np.nan,near_days=np.nan,far_days=np.nan,reason="no_bracket")
    valid = terms[terms[field].gt(0)&np.isfinite(terms[field])&terms.days.between(7,120)].sort_values("days")
    exact = valid[valid.days.eq(30)]
    if not exact.empty:
        row = exact.iloc[0]
        return dict(value=float(100*np.sqrt(row[field])),near_days=30,far_days=30,reason="ok")
    near,far = valid[valid.days.lt(30)],valid[valid.days.gt(30)]
    if near.empty or far.empty:return result
    a,b = near.iloc[-1],far.iloc[0]
    variance = (a.days*a[field]*(b.days-30)+b.days*b[field]*(30-a.days))/(b.days-a.days)/30
    return dict(value=float(100*np.sqrt(variance)),near_days=int(a.days),far_days=int(b.days),reason="ok")
