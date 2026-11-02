"""Causal screenshot-inspired first-wave features; research, not production gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.factors import KDJ
from quant.features.variable_library import build_continuous_ohlc

SCORE_COLUMNS = ["j_depth", "dry_volume", "impulse_energy", "wave_gain", "pullback_shape", "pullback_depth", "ma60_position"]


def confirmed_pivots(values: np.ndarray, kind: str, radius: int = 2) -> np.ndarray:
    """Pivot centers; caller MUST require center+radius <= information date."""
    result = []
    for i in range(radius, len(values)-radius):
        window = values[i-radius:i+radius+1]
        target = np.min(window) if kind == "low" else np.max(window)
        if values[i] == target and np.count_nonzero(window == target) == 1:
            result.append(i)
    return np.asarray(result, dtype=int)


def extract_wave_information(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Compute only signal-date information; no forward outcomes or learned weights."""
    f = daily.sort_values("date").reset_index(drop=True)
    if f.empty:
        return pd.DataFrame()
    price = build_continuous_ohlc(f)
    close, high, low, open_ = [price[x].to_numpy(dtype=float) for x in ["close", "high", "low", "open"]]
    volume = f.volume.to_numpy(dtype=float)
    dates = f.date.to_numpy()
    j = KDJ().compute(price).J.to_numpy()
    ma60 = price.close.rolling(60).mean().to_numpy()
    mean_volume = f.volume.shift(1).rolling(20).mean().to_numpy()
    ret = np.r_[np.nan, close[1:]/close[:-1]-1]
    impulse = (volume >= 1.6*mean_volume) & (ret >= .05) & (close > open_) & (volume > 0)
    lows = confirmed_pivots(low, "low")
    highs = confirmed_pivots(high, "high")
    starts: dict[int, tuple[int, int]] = {}
    for a in np.flatnonzero(impulse):
        if a < 30 or impulse[max(0,a-5):a].any():
            continue
        known_lows = lows[(lows >= a-30) & (lows+2 <= a)]
        if not len(known_lows):
            continue
        trough = int(known_lows[-1])
        previous_highs = highs[(highs < trough) & (highs >= trough-40)]
        prior_high = int(previous_highs[-1]) if len(previous_highs) else -1
        # Start near a confirmed trough and break prior resistance; reject late-wave re-acceleration.
        if close[a] > np.max(close[a-10:a]) and close[a]/low[trough]-1 <= .15 and a-trough <= 15:
            starts[int(a)] = (trough, prior_high)
    rows = []
    for t in np.flatnonzero((f.date >= start) & (f.date <= end) & (volume > 0) & (j < 13)):
        eligible = [a for a in starts if t-40 <= a < t]
        if not eligible:
            continue
        a = eligible[-1]
        trough, prior_high = starts[a]
        peak = a+int(np.argmax(high[a:t+1]))
        peak_confirmed = peak in highs and peak+2 <= t
        if not peak_confirmed or peak < a or peak-trough > 20:
            continue
        age = t-peak
        wave_prices = close[trough:peak+1]
        wave_efficiency = (close[peak]-close[trough]) / max(np.abs(np.diff(wave_prices)).sum(), 1e-12)
        pb = close[peak:t+1]
        decline_efficiency = (close[peak]-close[t]) / max(np.abs(np.diff(pb)).sum(), 1e-12)
        rebound = float(np.max(pb/np.minimum.accumulate(pb)-1))
        secondary_highs = highs[(highs > peak) & (highs+2 <= t)]
        wave_return = close[peak]/low[trough]-1
        higher_high = prior_high >= 0 and high[peak] > high[prior_high]
        higher_low = low[peak+1:t+1].min() > low[trough]
        n_structure = bool(higher_high and higher_low and wave_return >= .07 and wave_efficiency >= .35)
        straight = bool(2 <= age <= 10 and decline_efficiency >= .5 and rebound <= .05 and not len(secondary_highs))
        ratio = volume[t]/volume[a]
        shrink = volume[peak+1:t+1].mean()/volume[a]
        dry = bool(volume[t] <= 1.2*volume[a+1:t+1].min() and ratio < .8)
        bias = close[t]/ma60[t]-1
        draw = 1-close[t]/close[peak]
        energy = volume[a]/mean_volume[a]
        scores = [20 if j[t] <= 5 else 16 if j[t] <= 9 else 12,
                  15 if ratio < .3 else 11 if ratio < .5 else 6 if ratio < .8 else 0,
                  10 if energy >= 2.5 else 8 if energy >= 2 else 5,
                  10 if wave_return >= .15 else 7 if wave_return >= .10 else 4 if wave_return >= .07 else 0,
                  15 if shrink < .4 else 10 if shrink < .6 else 5 if shrink < .8 else 0,
                  12 if .10 <= draw <= .20 else 8 if .08 <= draw < .10 or .20 < draw <= .25 else 4 if .05 <= draw < .08 else 0,
                  10 if abs(bias) <= .05 else 6 if -.15 <= bias < -.05 else 0]
        row = dict(symbol=f.symbol.iloc[0], date=dates[t], trough_date=dates[trough], impulse_date=dates[a], peak_date=dates[peak], peak_known_at=dates[peak+2], prior_high_date=dates[prior_high] if prior_high >= 0 else pd.NaT,
                   j=j[t], wave_efficiency=wave_efficiency, decline_efficiency=decline_efficiency, rebound=rebound, wave_return=wave_return, drawdown=draw, impulse_multiple=energy, volume_ratio=ratio, pullback_volume_ratio=shrink, ma60_bias=bias, pullback_bars=age,
                   n_structure=n_structure, straight=straight, dry=dry, near_ma60=bool(-.15 <= bias <= .05), strict_gate=bool(n_structure and straight and dry and -.15 <= bias <= .05), soft_position_gate=bool(n_structure and straight and dry))
        row.update(dict(zip(SCORE_COLUMNS, scores)))
        rows.append(row)
    return pd.DataFrame(rows)
