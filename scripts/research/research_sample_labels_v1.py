"""Time-ordered sample, target and feature ablations on five-session outcomes."""
from __future__ import annotations

import argparse
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/research')]
from quant.research.right_side_unified_labels import build_right_side_unified_labels
from quant.research.local_ranknet import make_pairs, derivatives, percentile_scores
from evaluate_local_ranknet_fusion import PARAMS, matrix

SOURCE = ROOT / 'reports/research/full_rank_comparison_v2'
OUT = ROOT / 'reports/research/sample_label_research_v1'
KEYS = ['symbol', 'date']
FOLDS = {
    '2025H2': ('2025-01-01', '2025-04-01', '2025-07-01', '2026-01-01'),
    '2026': ('2025-07-01', '2025-10-01', '2026-01-01', '2026-09-04'),
}
ARMS = ['return', 'drop_005', 'drop_010', 'soft_005', 'first_event',
        'overlap_weight', 'recent_12m', 'decay_180d', 'event_features',
        'good_path', 'path_boundary', 'profit', 'quantile10',
        'rank', 'rank_margin', 'observable_filter']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def label_symbol(task):
    signals, daily, calendar = task
    return build_right_side_unified_labels(
        signals, daily, calendar, horizons=[5], entry_modes=['next_close'],
        daily_symbol_column='symbol', daily_date_column='date',
    )


def prepare():
    meta = json.loads((SOURCE / 'dataset_complete.json').read_text())
    hashes = {}
    for side in ['left', 'right']:
        path = SOURCE / f'dataset_{side}.parquet'
        hashes[str(path)] = digest(path)
        if hashes[str(path)] != meta['dataset_hashes'][side]:
            raise ValueError(f'Frozen source changed: {path}')
    paths = sorted(p for p in (ROOT / 'data/raw/daily_partitioned').glob('*/*.parquet')
                   if '202401' <= p.parent.name.split('=')[-1] <= '202609')
    columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'pre_close']
    bars = pl.concat([pl.read_parquet(p, columns=columns) for p in paths],
                     how='vertical_relaxed').with_columns(pl.col('date').cast(pl.Datetime('ns')))
    if bars.select(pl.struct(['symbol', 'date']).is_duplicated().any()).item():
        raise ValueError('Duplicate raw bars')
    calendar = pd.DatetimeIndex(bars['date'].unique().sort().to_numpy())
    pd.DataFrame({'date': calendar}).to_parquet(OUT / 'calendar.parquet', index=False)
    save('raw_hashes.json', {str(p): digest(p) for p in paths})
    for side in ['left', 'right']:
        frame = pd.read_parquet(SOURCE / f'dataset_{side}.parquet')
        frame['date'] = pd.to_datetime(frame.source_date)
        if frame.duplicated(KEYS).any():
            raise ValueError('Duplicate signal-day candidate')
        frame = frame.sort_values(['symbol', 'date']).reset_index(drop=True)
        if side == 'left':
            labels_path = ROOT / 'data/research/left_side_disclosure_gain_20261029/labels.parquet'
            labels = pd.read_parquet(labels_path)
            hashes[str(labels_path)] = digest(labels_path)
        else:
            wanted = {s: g[KEYS] for s, g in frame.groupby('symbol', sort=True)}
            tasks = ((wanted[key[0]], group.to_pandas(), calendar)
                     for key, group in bars.sort(['symbol', 'date']).partition_by('symbol', as_dict=True).items()
                     if key[0] in wanted)
            pieces = []
            with ProcessPoolExecutor(max_workers=4) as pool:
                for number, part in enumerate(pool.map(label_symbol, tasks, chunksize=8)):
                    pieces.append(part)
                    if number % 500 == 0:
                        print('RIGHT LABEL SYMBOLS', number, flush=True)
            labels = pd.concat(pieces, ignore_index=True)
        labels = labels[KEYS + ['entry_date', 'label_end_date', 'terminal_return',
                               'mfe', 'mae', 'good_path5', 'locked_limit_up', 'maturity_reason']]
        frame = frame.drop(columns=[c for c in labels.columns if c not in KEYS and c in frame])
        frame = frame.merge(labels, on=KEYS, how='left', validate='one_to_one')
        frame['outcome'] = frame.terminal_return - .002
        frame['unfilled'] = frame.locked_limit_up.fillna(False).astype(bool)
        frame.loc[frame.unfilled, 'outcome'] = 0.
        positions = calendar.searchsorted(frame.date) + 6
        frame['planned_maturity'] = pd.to_datetime([
            calendar[i] if i < len(calendar) else pd.NaT for i in positions])
        frame['maturity'] = frame.label_end_date.fillna(frame.planned_maturity)
        day_number = calendar.get_indexer(frame.date)
        frame['event_gap'] = pd.Series(day_number).groupby(frame.symbol).diff().fillna(999)
        new_event = frame.event_gap.gt(5)
        event_id = new_event.groupby(frame.symbol).cumsum()
        frame['event_age'] = frame.groupby([frame.symbol, event_id]).cumcount().astype(float)
        frame['event_recent_count'] = frame.groupby('symbol').date.transform(
            lambda dates: pd.Series(calendar.searchsorted(dates), index=dates.index).diff().le(5).rolling(5, min_periods=1).sum())
        for c in ['turnover_rate', 'volatility_20d', 'volume_relative_20d']:
            frame[f'event_rank_{c}'] = frame.groupby('date')[c].rank(pct=True)
        frame['event_vol_volume'] = frame.volatility_20d * frame.volume_relative_20d
        frame = frame.sort_values(['date', 'symbol']).reset_index(drop=True)
        frame.to_parquet(OUT / f'dataset_{side}.parquet', index=False)
        save(f'data_audit_{side}.json', {
            'rows': len(frame), 'dates': [frame.date.min(), frame.date.max()],
            'known_outcomes': int(frame.outcome.notna().sum()), 'unfilled': int(frame.unfilled.sum()),
            'unknown_reasons': frame.loc[frame.outcome.isna(), 'maturity_reason'].value_counts(dropna=False).to_dict(),
            'delayed_labels': int(frame.maturity.gt(frame.planned_maturity).sum()),
            'good_path_positive_terminal_loss': int((frame.good_path5.eq(True) & frame.outcome.lt(0)).sum()),
            'good_path_positive': int(frame.good_path5.eq(True).sum()),
        })
    save('sources.json', hashes)


def sample_weights(frame, arm, cutoff):
    keep = pd.Series(True, index=frame.index)
    weights = pd.Series(1., index=frame.index)
    if arm in ['drop_005', 'drop_010']:
        keep = frame.outcome.abs().ge(.005 if arm == 'drop_005' else .01)
    if arm == 'soft_005':
        weights = (frame.outcome.abs() / .005).clip(.1, 1.)
    if arm == 'first_event':
        keep = frame.event_age.eq(0)
    if arm == 'path_boundary':
        keep = ((frame.mfe - .05).abs().ge(.005) & (frame.mae + .03).abs().ge(.005))
    if arm == 'recent_12m':
        keep = frame.date.ge(pd.Timestamp(cutoff) - pd.DateOffset(months=12))
    if arm == 'decay_180d':
        weights *= np.exp2(-(pd.Timestamp(cutoff) - frame.date).dt.days / 180)
    if arm == 'overlap_weight':
        # Concurrency within each stock, including actual delayed label intervals.
        for _, part in frame.groupby('symbol', sort=False):
            days = pd.date_range(part.date.min(), part.maturity.max(), freq='D')
            starts = days.searchsorted(part.entry_date)
            ends = days.searchsorted(part.maturity)
            delta = np.zeros(len(days) + 1)
            np.add.at(delta, starts, 1)
            np.add.at(delta, ends + 1, -1)
            concurrency = np.cumsum(delta[:-1])
            inverse = np.divide(1., concurrency, out=np.zeros_like(concurrency), where=concurrency > 0)
            integral = np.r_[0., np.cumsum(inverse)]
            weights.loc[part.index] = (integral[ends + 1] - integral[starts]) / (ends - starts + 1)
    # Every retained date has equal aggregate weight, except explicit time decay.
    weights = weights.loc[keep]
    dates = frame.loc[keep, 'date']
    weights /= weights.groupby(dates).transform('sum')
    if arm == 'decay_180d':
        weights *= np.exp2(-(pd.Timestamp(cutoff) - dates).dt.days / 180)
    return keep, (weights / weights.mean()).to_numpy()


def ledger(frame, scores, threshold=-np.inf, eligible=None, rank_scores=None):
    """120 independently settled equal initial slots, at most 20 attempts per day.

    Reports terminal valuation, not marked-to-market NAV. An unresolved position
    locks only its own slot and gets a -10% terminal stress mark.
    """
    candidates = frame[KEYS + ['entry_date', 'maturity', 'outcome', 'unfilled']].copy()
    candidates['score'] = scores
    candidates['rank_score'] = scores if rank_scores is None else rank_scores
    mask = candidates.score.ge(threshold)
    if eligible is not None:
        mask &= eligible
    picks = candidates.loc[mask].sort_values(['date', 'rank_score', 'symbol'], ascending=[True, False, True])
    capital = np.ones(120) / 120
    available = np.full(120, np.datetime64('1900-01-01', 'ns'))
    trades = []
    for _, group in picks.groupby('date', sort=True):
        if group.entry_date.isna().all():
            continue
        entry = group.entry_date.dropna().iloc[0].to_datetime64()
        free = list(np.flatnonzero(available <= entry))
        for row in group.head(min(20, len(free))).itertuples():
            if row.unfilled:
                trades.append({'date': row.date, 'symbol': row.symbol, 'net': 0., 'filled': False, 'unknown': False, 'contribution': 0.})
                continue
            slot = free.pop(0)
            unknown = pd.isna(row.outcome)
            net = -.10 if unknown else row.outcome
            contribution = capital[slot] * net
            capital[slot] += contribution
            available[slot] = np.datetime64('2100-01-01') if unknown else row.maturity.to_datetime64()
            trades.append({'date': row.date, 'symbol': row.symbol, 'net': net, 'filled': True, 'unknown': unknown, 'contribution': contribution})
    trades = pd.DataFrame(trades, columns=['date', 'symbol', 'net', 'filled', 'unknown', 'contribution'])
    filled = trades.loc[trades.filled.eq(True)]
    return {
        'capital_return': float(capital.sum() - 1), 'attempts': len(trades), 'filled': len(filled),
        'unknown': int(trades.unknown.sum()),
        'win_rate': float(filled.net.gt(0).mean()) if len(filled) else None,
        'mean_net': float(filled.net.mean()) if len(filled) else None,
        'candidate_coverage': float(mask.mean()),
    }, trades


def slices(frame, bounds):
    fit_end, calibration_end, test_start, test_end = bounds
    fit = frame.loc[frame.date.lt(fit_end) & frame.maturity.lt(fit_end) & frame.outcome.notna()].copy()
    cal = frame.loc[frame.date.ge(fit_end) & frame.date.lt(calibration_end) & frame.maturity.lt(calibration_end) & frame.outcome.notna()].copy()
    result = [fit, cal]
    for start, end in [(calibration_end, test_start), (test_start, test_end)]:
        # End the signal window using the market calendar known before entry.
        # Actual delayed exits remain unknown and retain their capital lock.
        part = frame.loc[frame.date.ge(start) & frame.date.lt(end) & frame.planned_maturity.lt(end)].copy()
        part.loc[part.maturity.ge(end), 'outcome'] = np.nan
        result.append(part)
    return result


def train(replay=False):
    meta = json.loads((SOURCE / 'dataset_complete.json').read_text())
    records, choices, training = [], [], []
    for side in ['left', 'right']:
        frame = pd.read_parquet(OUT / f'dataset_{side}.parquet')
        for fold, bounds in FOLDS.items():
            fit, cal, valid, test = slices(frame, bounds)
            core = [c for c in meta['core'] if fit[c].nunique() > 1]
            raw_scores = {}
            for arm in ARMS:
                print('TRAIN', side, fold, arm, flush=True)
                started = time.monotonic()
                train_frame = fit
                if arm in ['good_path', 'path_boundary']:
                    train_frame = fit.loc[fit.good_path5.notna()]
                if arm in ['rank', 'rank_margin']:
                    # Same-day pairs across the full cohort. Pair membership uses no future feature.
                    train_frame = fit.sort_values(['date', 'symbol'])
                keep, weights = sample_weights(train_frame, arm, bounds[0])
                train_frame = train_frame.loc[keep]
                cols = core + ([c for c in frame if c.startswith('event_')] if arm == 'event_features' else [])
                cols = list(dict.fromkeys(cols))
                target = train_frame.outcome.clip(-.5, .5)
                objective = 'reg:squarederror'
                if arm in ['good_path', 'path_boundary', 'profit']:
                    target = train_frame.good_path5.astype(float) if arm != 'profit' else train_frame.outcome.gt(0).astype(float)
                    objective = 'binary:logistic'
                params = {**PARAMS, 'objective': objective}
                if arm == 'quantile10':
                    params.update(objective='reg:quantileerror', quantile_alpha=.1)
                data = matrix(train_frame, cols)
                data.set_label(target)
                data.set_weight(weights)
                custom = None
                if arm in ['rank', 'rank_margin']:
                    pairs = make_pairs(train_frame.date, np.zeros(len(train_frame)), train_frame.outcome)
                    if arm == 'rank_margin':
                        difference = np.abs(train_frame.outcome.to_numpy()[pairs.left] - train_frame.outcome.to_numpy()[pairs.right])
                        pairs = replace(pairs, weight=pairs.weight * np.minimum(difference / .005, 1.))
                    params.update(base_score=0., disable_default_eval_metric=1)
                    custom = lambda prediction, _: derivatives(prediction, pairs)
                model_path = OUT / f'model_{side}_{fold}_{arm}.ubj'
                if replay:
                    model = xgb.Booster()
                    model.load_model(model_path)
                    assert model.feature_names == cols
                else:
                    model = xgb.train(params, data, num_boost_round=60, obj=custom)
                    model.save_model(model_path)
                values = [model.predict(matrix(part, cols)) for part in [cal, valid, test]]
                raw_scores[arm] = values
                iso = IsotonicRegression(out_of_bounds='clip').fit(values[0], cal.outcome.clip(-.5,.5),
                    sample_weight=1 / cal.groupby('date').date.transform('size'))
                scores = [iso.predict(value) for value in values[1:]]
                save(f'calibrator_{side}_{fold}_{arm}.json', {'x': iso.X_thresholds_.tolist(), 'y': iso.y_thresholds_.tolist()})
                eligibility = [None, None]
                if arm == 'observable_filter':
                    low = fit.turnover_rate.quantile(.1)
                    high = fit.volatility_20d.quantile(.9)
                    eligibility = [p.turnover_rate.ge(low) & p.volatility_20d.le(high) & p.event_age.eq(0) for p in [valid, test]]
                grid = []
                for threshold in [-np.inf, 0., .002, .005, .01]:
                    metrics, _ = ledger(valid, scores[0], threshold, eligibility[0], values[1])
                    grid.append({'threshold': threshold, **metrics})
                eligible = [r for r in grid if r['capital_return'] > 0 and r['filled'] >= 100]
                selected = max(eligible, key=lambda r: r['capital_return'])['threshold'] if eligible else np.inf
                choices.append({'side': side, 'fold': fold, 'arm': arm, 'selected_threshold': selected, 'validation_grid': grid})
                for policy, prediction, threshold in [('fixed_top20', values[2], -np.inf), ('validation_gate', scores[1], selected)]:
                    metrics, trades = ledger(test, prediction, threshold, eligibility[1], values[2])
                    records.append({'side': side, 'fold': fold, 'arm': arm, 'policy': policy, **metrics})
                    trades.to_parquet(OUT / f'trades_{side}_{fold}_{arm}_{policy}.parquet', index=False)
                prediction = test[KEYS + ['outcome', 'maturity', 'entry_date', 'unfilled']].copy()
                prediction['raw_score'], prediction['expected_net'] = values[2], scores[1]
                prediction.to_parquet(OUT / f'predictions_{side}_{fold}_{arm}.parquet', index=False)
                training.append({'side': side, 'fold': fold, 'arm': arm, 'rows': len(train_frame),
                    'retained': len(train_frame)/len(fit), 'features': len(cols), 'fit_label_end': str(train_frame.maturity.max()),
                    'cal_label_end': str(cal.maturity.max()), 'replay': replay, 'seconds': time.monotonic()-started})
                pd.DataFrame(records).to_csv(OUT / 'results.csv', index=False)
                pd.DataFrame(training).to_csv(OUT / 'training.csv', index=False)
                save('choices.json', choices)
            # Risk contribution selected on validation outcomes only.
            risk_grid = []
            for risk_weight in [0., .25, .5, 1.]:
                score = raw_scores['return'][1] + risk_weight * raw_scores['quantile10'][1]
                metrics, _ = ledger(valid, score)
                risk_grid.append((metrics['capital_return'], risk_weight))
            risk_weight = max(risk_grid)[1]
            metrics, trades = ledger(test, raw_scores['return'][2] + risk_weight * raw_scores['quantile10'][2])
            records.append({'side': side, 'fold': fold, 'arm': 'return_risk', 'policy': 'fixed_top20', **metrics})
            trades.to_parquet(OUT / f'trades_{side}_{fold}_return_risk_fixed_top20.parquet', index=False)
            save(f'risk_selection_{side}_{fold}.json', {'validation_grid': risk_grid, 'selected': risk_weight})
            pd.DataFrame(records).to_csv(OUT / 'results.csv', index=False)
    save('complete.json', {'models': len(training), 'results': len(records), 'script_sha256': digest(__file__)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'train', 'replay'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        save('protocol.json', {
            'folds': FOLDS, 'arms': ARMS, 'params': PARAMS, 'rounds': 60,
            'label': 'T+1 close, five subsequent stock bars, 20bp round-trip cost',
            'boundaries': 'Outcome/path boundary filtering is TRAIN ONLY. All test candidates retained. Observable filter uses fit-only thresholds and signal-day inputs.',
            'selection': 'Calibration then validation then chronological test. Training/calibration labels mature strictly before next phase.',
            'scope': '129-feature research controls on existing left and pure-right cohorts; not a reproduction of the production 237-input right classifier or left 152-input fusion.',
            'limitations': ['Historical test dates already explored.', 'Right labels reconstructed from current raw vintage with canonical proxy limit detection.',
                'Five stock-bar terminal close is an execution proxy; future locked-down exits have not been reconstructed.',
                'Independent 120-slot ledger reports terminal valuation, not MTM drawdown; same-close reinvestment proxy.',
                'Left frozen labels and right current-vintage labels are compared only within side.',
                'Unknown outcomes remain in evaluation, lock only their position and receive -10% stress valuation.'],
        })
        prepare()
    else:
        train(replay=args.stage == 'replay')


if __name__ == '__main__':
    main()
