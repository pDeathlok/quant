"""Array-based replay of the canonical 27 left-rule features.

This is a research execution optimization, checked against the canonical
calculator before use. It does not change windows, readiness or rule meanings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_COLUMNS, _base_state, _prepare_daily, _safe_div,
)


def replay_left_rules(daily: pd.DataFrame) -> pd.DataFrame:
    f = _prepare_daily(daily)
    s = _base_state(f)
    n = len(f)
    v = {c: np.full(n, np.nan) for c in LEFT_SIDE_RULE_FEATURE_COLUMNS}
    close, low, high, volume = (f[c].to_numpy(float) for c in ['close', 'low', 'high', 'volume'])
    pct, bbi, vr = (s[c].to_numpy(float) for c in ['pct_chg', 'bbi', 'volume_relative_prev5'])
    rise = s.is_rise.to_numpy(bool)
    event = rise & (pct >= 2.5) & (vr >= 1.8)
    ground = f.volume.rolling(60, min_periods=1).quantile(.25).to_numpy()
    v['ls_close_to_bbi'] = _safe_div(f.close, s.bbi).to_numpy() - 1
    v['ls_low_absorb_bbi'] = (np.isfinite(pct) & (pct >= -3) & (pct <= 1.5) & np.isfinite(bbi) & (close <= bbi * 1.03)).astype(float)
    v['ls_volume_ratio_prev'] = s.volume_ratio_prev.to_numpy()
    for i in range(15, n):
        candidates = np.flatnonzero(event[max(1, i-10):i])
        if len(candidates):
            j = max(1, i-10) + candidates[-1]
            if i-j >= 2:
                vals = [i-j, pct[j], vr[j], volume[i]/volume[j] if volume[j] else np.nan,
                        volume[i]/ground[i] if ground[i] else np.nan, close[i]/low[j]-1 if low[j] else np.nan,
                        float(volume[i] <= volume[j]*.75), float(volume[i] <= ground[i]), float(close[i] >= low[j]*.96)]
                cols = ['ls_days_since_yidong', 'ls_yidong_pct_chg', 'ls_yidong_volume_relative_prev5',
                        'ls_volume_to_yidong', 'ls_volume_to_ground_q25_60d', 'ls_close_to_yidong_low',
                        'ls_yidong_shrink_ok', 'ls_yidong_ground_ok', 'ls_yidong_pullback_ok']
                for c, value in zip(cols, vals): v[c][i] = value
        if i >= 23:
            # The first point in each original slice has no shifted predecessor.
            v['ls_nana_rise_volume_count_8d'][i] = np.sum(rise[i-13:i-6] & (volume[i-13:i-6] > volume[i-14:i-7]))
            v['ls_nana_shrink_count_7d'][i] = np.sum(volume[i-5:i+1] < volume[i-6:i])
            v['ls_nana_big_yin_count_15d'][i] = np.sum((close[i-14:i+1] < f.open.to_numpy()[i-14:i+1]) & (vr[i-14:i+1] >= 1.5) & (pct[i-14:i+1] <= -2))
        if i >= 29:
            peak = int(np.nanargmax(high[i-21:i+1]))
            trough = int(np.nanargmin(low[i-21:i-21+peak+1])) if peak > 2 else -1
            if 0 <= trough < peak < 20:
                lp, hp = low[i-21+trough], high[i-21+peak]
                up, down = (hp-lp)/lp if lp > 0 else np.nan, (hp-close[i])/hp if hp > 0 else np.nan
                vals = [peak-trough, 21-peak, up, down, (21-peak)/(peak-trough), down/up if np.isfinite(up) and up > 0 else np.nan]
                for c, value in zip(['ls_va_up_days','ls_va_down_days','ls_va_up_pct','ls_va_down_pct','ls_va_time_symmetry','ls_va_space_symmetry'], vals): v[c][i] = value
    out = pd.DataFrame(v)
    prevlow = f.low.shift(1).rolling(3, min_periods=3).min()
    out['ls_range3_pct'] = f.high.shift(1).rolling(3, min_periods=3).max() / prevlow.replace(0, np.nan) - 1
    out['ls_low_to_prev3_low_pct'] = _safe_div(f.low, prevlow) - 1
    washout = ~s.is_rise.astype(bool) & s.pct_chg.gt(-9.5) & s.volume_relative_prev5.ge(1.2) & f.low.lt(f.low.shift(1).rolling(10, min_periods=5).min())
    for key, flag in [('12', washout), ('15', washout & s.volume_relative_prev5.ge(1.5))]:
        out[f'ls_recent_washout{key}_3d'] = flag.shift(1).rolling(3, min_periods=1).max().fillna(0).gt(0)
    amp = _safe_div(f.high-f.low, f.low)*100
    vol20 = _safe_div(f.volume, f.volume.rolling(20, min_periods=5).mean())
    out['ls_small_reversal_loose'] = amp.lt(8) & s.pct_chg.gt(-2.5) & s.pct_chg.lt(3) & vol20.lt(1)
    out['ls_small_reversal_tight'] = amp.lt(7) & s.pct_chg.gt(-2) & s.pct_chg.lt(2.5) & vol20.lt(.9)
    for c in ['ls_yidong_shrink_ok', 'ls_yidong_ground_ok', 'ls_yidong_pullback_ok', 'ls_low_absorb_bbi']:
        out[c] = out[c].fillna(0).astype(bool)
    return out.loc[:, LEFT_SIDE_RULE_FEATURE_COLUMNS]
