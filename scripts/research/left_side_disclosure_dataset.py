#!/usr/bin/env python
"""Research-only faster rebuild; parity checked against the canonical rules.

The canonical implementation slices pandas frames for every bar. This version
uses identical rolling windows over arrays for the same 27 rule features.
Production factor and signal code is left untouched.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import quant.research.left_side_unified_features as left
import scripts.research.validate_unified_left_side_models as build


def fast_left_rules(daily: pd.DataFrame) -> pd.DataFrame:
    frame = left._prepare_daily(daily)
    state = left._base_state(frame)
    n = len(frame)
    pos = np.arange(n)
    values = {c: np.full(n, np.nan) for c in left.LEFT_SIDE_RULE_FEATURE_COLUMNS}
    volume = frame.volume.to_numpy()
    close = frame.close.to_numpy()
    low = frame.low.to_numpy()
    pct = state.pct_chg.to_numpy()
    bbi = state.bbi.to_numpy()
    volrel = state.volume_relative_prev5.to_numpy()
    rise = state.is_rise.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        values['ls_close_to_bbi'] = np.where(np.isfinite(bbi) & (bbi != 0), close/bbi-1, np.nan)
    absorb = np.isfinite(pct) & (pct >= -3) & (pct <= 1.5) & np.isfinite(bbi) & (close <= bbi*1.03)
    values['ls_low_absorb_bbi'] = absorb
    values['ls_volume_ratio_prev'] = state.volume_ratio_prev.to_numpy()
    event = rise & (pct >= 2.5) & (volrel >= 1.8) & (pos > 0)
    latest = np.maximum.accumulate(np.where(event, pos, -100))
    prior = np.r_[-100, latest[:-1]] if n else np.array([], dtype=int)
    age = pos-prior
    valid = (pos >= 15) & (age >= 2) & (age <= 10)
    today = pos[valid]; anchor = prior[valid]
    ground = frame.volume.rolling(60, min_periods=1).quantile(.25).to_numpy()
    values['ls_days_since_yidong'][valid] = age[valid]
    values['ls_yidong_pct_chg'][valid] = pct[anchor]
    values['ls_yidong_volume_relative_prev5'][valid] = volrel[anchor]
    with np.errstate(divide='ignore', invalid='ignore'):
        values['ls_volume_to_yidong'][valid] = np.where(volume[anchor] != 0, volume[today]/volume[anchor], np.nan)
        values['ls_volume_to_ground_q25_60d'][valid] = np.where(ground[today] != 0, volume[today]/ground[today], np.nan)
        values['ls_close_to_yidong_low'][valid] = np.where(low[anchor] != 0, close[today]/low[anchor]-1, np.nan)
    values['ls_yidong_shrink_ok'][valid] = volume[today] <= volume[anchor]*.75
    values['ls_yidong_ground_ok'][valid] = volume[today] <= ground[today]
    values['ls_yidong_pullback_ok'][valid] = close[today] >= low[anchor]*.96
    # The first bar of each sub-window has no predecessor within that window.
    rising = state.is_rise & frame.volume.gt(frame.volume.shift(1))
    shrinking = frame.volume.lt(frame.volume.shift(1))
    big_yin = frame.close.lt(frame.open) & state.volume_relative_prev5.ge(1.5) & state.pct_chg.le(-2)
    valid = pos >= 23
    values['ls_nana_rise_volume_count_8d'][valid] = rising.rolling(7).sum().shift(7).to_numpy()[valid]
    values['ls_nana_shrink_count_7d'][valid] = shrinking.rolling(6).sum().to_numpy()[valid]
    values['ls_nana_big_yin_count_15d'][valid] = big_yin.rolling(15).sum().to_numpy()[valid]
    if n >= 30:
        highs = np.lib.stride_tricks.sliding_window_view(frame.high.to_numpy(), 22)
        lows = np.lib.stride_tricks.sliding_window_view(low, 22)
        peak = np.where(np.isnan(highs), -np.inf, highs).argmax(axis=1)
        trough_candidates = np.where(np.arange(22)[None, :] <= peak[:, None], lows, np.nan)
        trough = np.where(np.isnan(trough_candidates), np.inf, trough_candidates).argmin(axis=1)
        valid = (pos[21:] >= 29) & (peak > 2) & (trough < peak) & (peak < 20)
        rows = np.flatnonzero(valid); at = rows+21
        up_days = peak[rows]-trough[rows]
        down_days = 21-peak[rows]
        bottom = lows[rows, trough[rows]]; top = highs[rows, peak[rows]]
        with np.errstate(divide='ignore', invalid='ignore'):
            up_pct = np.where(bottom > 0, (top-bottom)/bottom, np.nan)
            down_pct = np.where(top > 0, (top-close[at])/top, np.nan)
            symmetry = np.where(np.isfinite(up_pct) & (up_pct > 0), down_pct/up_pct, np.nan)
        for col, data in [('ls_va_up_days', up_days), ('ls_va_down_days', down_days),
                          ('ls_va_up_pct', up_pct), ('ls_va_down_pct', down_pct),
                          ('ls_va_time_symmetry', down_days/up_days), ('ls_va_space_symmetry', symmetry)]:
            values[col][at] = data
    result = pd.DataFrame(values, index=frame.index)
    previous_low3 = frame.low.shift(1).rolling(3, min_periods=3).min()
    result['ls_range3_pct'] = frame.high.shift(1).rolling(3, min_periods=3).max()/previous_low3.replace(0, np.nan)-1
    result['ls_low_to_prev3_low_pct'] = left._safe_div(frame.low, previous_low3)-1
    vol20 = left._safe_div(frame.volume, frame.volume.rolling(20, min_periods=5).mean())
    amplitude = left._safe_div(frame.high-frame.low, frame.low)*100
    washout12 = (~state.is_rise.astype(bool) & state.pct_chg.gt(-9.5)
                 & state.volume_relative_prev5.ge(1.2)
                 & frame.low.lt(frame.low.shift(1).rolling(10, min_periods=5).min()))
    washout15 = washout12 & state.volume_relative_prev5.ge(1.5)
    result['ls_recent_washout12_3d'] = washout12.shift(1).rolling(3, min_periods=1).max().fillna(0) > 0
    result['ls_recent_washout15_3d'] = washout15.shift(1).rolling(3, min_periods=1).max().fillna(0) > 0
    result['ls_small_reversal_loose'] = amplitude.lt(8) & state.pct_chg.gt(-2.5) & state.pct_chg.lt(3) & vol20.lt(1)
    result['ls_small_reversal_tight'] = amplitude.lt(7) & state.pct_chg.gt(-2) & state.pct_chg.lt(2.5) & vol20.lt(.9)
    for col in ['ls_yidong_shrink_ok', 'ls_yidong_ground_ok', 'ls_yidong_pullback_ok', 'ls_low_absorb_bbi']:
        result[col] = result[col].fillna(0).astype(bool)
    return result.loc[:, left.LEFT_SIDE_RULE_FEATURE_COLUMNS]


def parity() -> None:
    paths = build._daily_paths(ROOT / 'data/raw/daily_partitioned', '2014-01-01', '2026-10-01')
    symbols = ['000001.SZ', '000002.SZ', '300750.SZ', '600519.SH', '688981.SH', '920306.BJ']
    frames = [pd.read_parquet(p, filters=[('ts_code', 'in', symbols)]) for p in paths]
    daily = pd.concat(frames, ignore_index=True)
    results = []
    for symbol, group in daily.groupby('ts_code'):
        normalized = build.normalize_tushare_daily(group, symbol).sort_values('date').reset_index(drop=True)
        started = time.monotonic()
        expected = left.compute_left_side_rule_features(normalized)
        seconds = time.monotonic()-started
        started = time.monotonic(); actual = fast_left_rules(normalized)
        pd.testing.assert_frame_equal(expected, actual, rtol=1e-12, atol=1e-12)
        expected_flags = left.compute_left_side_signal_flags(normalized, rule_features=expected)
        actual_flags = left.compute_left_side_signal_flags(normalized, rule_features=actual)
        pd.testing.assert_frame_equal(expected_flags, actual_flags)
        result = {'symbol': symbol, 'bars': len(group), 'reference_seconds': seconds,
                  'fast_seconds': time.monotonic()-started, 'passed': True}
        print(result, flush=True); results.append(result)
    from scripts.research.validate_left_side_disclosure_gain import REPORT, save_json
    save_json(REPORT / 'rebuild_parity.json', results)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['parity', 'build'])
    args = parser.parse_args()
    if args.command == 'parity':
        parity(); return
    from scripts.research.validate_left_side_disclosure_gain import DATA
    if not (ROOT / 'reports/research/left_side_disclosure_gain_20261029/rebuild_parity.json').exists():
        raise RuntimeError('Run parity before using the accelerated research rebuild')
    build._process_symbol = process_symbol
    namespace = argparse.Namespace(start_date='2020-01-01', end_date='2026-08-22',
        daily_partitions=ROOT / 'data/raw/daily_partitioned', tradability=ROOT / 'data/raw/tradability',
        dataset_out=DATA / 'events.parquet', labels_out=DATA / 'labels.parquet',
        manifest_out=DATA / 'dataset_manifest.json', workers=8, symbol_limit=None,
        horizons=[5], entry_modes=['next_close'])
    print(build.build_dataset(namespace), flush=True)


canonical_process_symbol = build._process_symbol

# Defined separately so multiprocessing children resolve the original function.
def process_symbol(task):
    build.compute_left_side_rule_features = fast_left_rules
    return canonical_process_symbol(task)


if __name__ == '__main__':
    main()
