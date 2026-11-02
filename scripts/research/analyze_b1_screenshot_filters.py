"""Screen screenshot-inspired filters against frozen out-of-fold B1 rankings.

This is a candidate event study, not an executable portfolio backtest.
Weekly MACD uses only completed prior weeks; N structure is a rolling proxy.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from quant.features.variable_library import build_continuous_ohlc

OUT = ROOT / 'reports/research/b1_screenshot_filters_20261004'
PRED = ROOT / 'reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet'
COST = .002


def macd(close):
    dif = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return dif, dif - dif.ewm(span=9, adjust=False).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(PRED)
    pred = pred.loc[pred.B1].copy()
    pred['date'] = pd.to_datetime(pred.date).dt.normalize()
    symbols = set(pred.symbol)
    parts = []
    columns = ['symbol','date','open','high','low','close','pre_close','volume']
    for p in sorted((ROOT / 'data/raw/daily_partitioned').glob('year_month=*/data.parquet')):
        month = p.parent.name.split('=')[1]
        if month < '202201' or month > '202609':
            continue
        f = pd.read_parquet(p, columns=columns)
        parts.append(f.loc[f.symbol.isin(symbols)])
    daily = pd.concat(parts, ignore_index=True)
    daily['date'] = pd.to_datetime(daily.date).dt.normalize()
    if daily.duplicated(['symbol','date']).any():
        raise RuntimeError('Duplicate daily rows')
    print(f'Loaded {len(daily):,} daily bars; {len(pred):,} B1 predictions', flush=True)
    frames = []
    for n,(symbol,f) in enumerate(daily.groupby('symbol', sort=True),1):
        f = f.sort_values('date').reset_index(drop=True)
        continuous = build_continuous_ohlc(f)
        c,h,l = continuous.close,continuous.high,continuous.low
        dif,_ = macd(c)
        week = f.date.dt.to_period('W-FRI')
        wc = pd.Series(c.to_numpy(), index=week).groupby(level=0).last()
        _,wh = macd(wc)
        # Shift by calendar week, so no current week's eventual Friday is visible.
        completed = wh.copy(); completed.index = completed.index + 1
        weekly = week.map(completed).to_numpy()
        v = pd.to_numeric(f.volume)
        recent_low = l.shift(1).rolling(10).min()
        prior_low = l.shift(11).rolling(10).min()
        recent_high = h.shift(1).rolling(10).max()
        prior_high = h.shift(11).rolling(10).max()
        f['weekly_red'] = weekly > 0
        f['daily_dif_positive'] = dif > 0
        f['above_ma60'] = c > c.rolling(60, min_periods=60).mean()
        f['rising_lows'] = recent_low > prior_low
        f['rising_highs'] = recent_high > prior_high
        f['shrink_v5_v20'] = v.rolling(5).mean() < .8*v.rolling(20).mean()
        f['terminal_return'] = c.shift(-6)/c.shift(-1)-1
        f['entry_date'] = f.date.shift(-1)
        f['exit_date'] = f.date.shift(-6)
        # Next-close entry must be the next market session; detected later.
        frames.append(f[['symbol','date','entry_date','exit_date','terminal_return',
                         'weekly_red','daily_dif_positive','above_ma60',
                         'rising_lows','rising_highs','shrink_v5_v20']])
        if n%1000 == 0: print(f'Features {n}/{len(symbols)}', flush=True)
    features = pd.concat(frames, ignore_index=True)
    frame = pred.merge(features, on=['symbol','date'], how='left', validate='one_to_one')
    if frame['entry_date'].isna().any():
        raise RuntimeError('Missing candidate market bars')
    calendar = np.sort(daily.date.unique())
    pos = np.searchsorted(calendar, frame.date.to_numpy(), side='right')
    expected = calendar[np.minimum(pos,len(calendar)-1)]
    # Reject suspensions rather than silently enter at the next available stock bar.
    valid = frame.terminal_return.notna() & frame.entry_date.eq(expected)
    excluded = int((~valid).sum())
    frame = frame.loc[valid].copy()
    checks = ['weekly_red','daily_dif_positive','above_ma60','rising_lows','rising_highs','shrink_v5_v20']
    variants = {'baseline': []}
    variants.update({name: [name] for name in checks})
    variants['trend'] = checks[:3]
    variants['trend_shrink'] = checks[:3]+['shrink_v5_v20']
    variants['n_structure'] = ['rising_lows','rising_highs']
    variants['all_technical'] = checks
    rows = []
    rng = np.random.default_rng(20261004)
    for fold in ('A','B','C'):
        local = frame.loc[frame.fold.eq(fold)].copy()
        # Prevent using returns beyond the fold boundary.
        fold_end = pred.loc[pred.fold.eq(fold),'date'].max()
        local = local.loc[local.exit_date.le(fold_end)]
        dates = pd.Index(sorted(local.date.unique()))
        base_daily = None
        for name,conditions in variants.items():
            mask = pd.Series(True,index=local.index)
            for condition in conditions: mask &= local[condition].fillna(False)
            selected = local.loc[mask].sort_values(
                ['date','pred_unified_left_long_task_deep','symbol'],ascending=[True,False,True]
            ).groupby('date',sort=False).head(20).copy()
            net = selected.terminal_return-COST
            # Fixed 20 slots; unused slots are cash, so coverage affects return.
            slots = net.groupby(selected.date).sum().reindex(dates,fill_value=0)/20
            if name == 'baseline': base_daily = slots
            delta = (slots-base_daily).to_numpy()
            # Moving blocks of 20 sessions account partly for overlapping horizons.
            draws = []
            for _ in range(1000):
                starts = rng.integers(0,len(delta),size=int(np.ceil(len(delta)/20)))
                indices = np.concatenate([(s+np.arange(20))%len(delta) for s in starts])[:len(delta)]
                draws.append(delta[indices].mean())
            loss = -net.loc[net.lt(0)].sum()
            rows.append(dict(fold=fold,variant=name,trades=len(selected),
                active_days=selected.date.nunique(),total_days=len(dates),
                avg_net_return=net.mean(),win_rate=net.gt(0).mean(),
                profit_factor=net.loc[net.gt(0)].sum()/loss if loss else np.nan,
                slot_cohort_return=slots.mean(),delta_slot_return=delta.mean(),
                delta_ci_low=np.quantile(draws,.025),delta_ci_high=np.quantile(draws,.975)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT/'metrics.csv',index=False)
    frame.to_parquet(OUT/'candidate_filter_context.parquet',index=False)
    payload = dict(status='success',scope='frozen_out_of_fold_B1_candidates_only',
        predictions=str(PRED.relative_to(ROOT)),round_trip_cost_bps=20,
        entry='next trading session close',exit='5 subsequent stock sessions after entry',
        excluded_missing_or_delayed_entries=excluded,weekly_macd='previous completed calendar week; 12/26/9 EMA',
        n_structure_proxy='previous 10 bars min/max > preceding 10 bars min/max; current bar excluded',
        folds={'A':'2024 exploratory','B':'2025 original model selection','C':'2026 through Aug 14 original confirmation; not new untouched holdout'},
        limitations=['Not production stable/aggressive exit simulation or portfolio NAV',
          'Frozen candidates: J<13/25 replacement cannot be tested on J<0 universe',
          'Industry leadership/main theme lacks audited point-in-time evidence',
          'Screenshot row 8 obscured and omitted','Rolling N proxy is not swing-pivot recognition',
          'Filtering precedes Top20; unfilled slots remain cash',
          'No additional limit-down execution or slippage simulation beyond frozen original eligibility',
          'Multiple variants: confidence intervals are exploratory and not multiplicity-adjusted'])
    (OUT/'manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print(metrics.loc[metrics.fold.isin(['B','C'])].to_string(index=False),flush=True)

if __name__ == '__main__': main()
