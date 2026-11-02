#!/usr/bin/env python
"""Frozen-parameter left-model ablation of disclosure timing, research only.

Actual dates become usable strictly after their disclosure date. Historical
appointment dates lack publication timestamps and are explicitly non-PIT.
No production artifact, registry or refresh configuration is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.research.left_side_unified_features import LEFT_SIDE_SIGNALS
from quant.research.right_side_long_task import LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold
from scripts.research.validate_unified_left_side_models import (
    _daily_paths, _fit_unified_model, _market_calendar, _split_validation_stages,
)

DATA = ROOT / 'data/research/left_side_disclosure_gain_20261029'
REPORT = ROOT / 'reports/research/left_side_disclosure_gain_20261029'
SINCE = 'report_trading_days_since_actual'
UNTIL = 'report_trading_days_until_appointment_approx'
STALE = 'report_trading_days_since_actual_refresh2'
AVAILABLE = 'report_actual_date_available'
ARMS = {'baseline': (), 'since_actual': (SINCE,),
        'until_appointment_approx': (UNTIL,), 'both_approx': (SINCE, UNTIL),
        'since_actual_refresh2': (STALE,), 'availability_control': (AVAILABLE,)}
SOURCE_MAP = {'SECURITY_CODE': 'symbol', 'REPORT_DATE': 'period',
              'FIRST_APPOINT_DATE': 'first', 'FIRST_CHANGE_DATE': 'change1',
              'SECOND_CHANGE_DATE': 'change2', 'THIRD_CHANGE_DATE': 'change3',
              'ACTUAL_PUBLISH_DATE': 'actual'}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_period(period: str) -> dict:
    folder = DATA / 'disclosure_raw'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{period}.json'
    if path.exists():
        rows = json.loads(path.read_text())
        return {'period': period, 'rows': len(rows), 'requests': 0,
                'sha256': sha256(path), 'cached': True}
    prior = ROOT / 'reports/disclosure_date_validation_20261029' / f'raw_eastmoney_{period}.json'
    calls = 0
    started = time.monotonic()
    if prior.exists():
        rows = json.loads(prior.read_text())
    else:
        rows = []
        params = {'reportName': 'RPT_PUBLIC_BS_APPOIN', 'columns': ','.join(SOURCE_MAP),
                  'pageSize': 500, 'pageNumber': 1,
                  'sortColumns': 'SECURITY_CODE', 'sortTypes': '1',
                  'filter': f"(REPORT_DATE='{period}')"}
        with requests.Session() as session:
            while True:
                for attempt in range(2):
                    try:
                        calls += 1
                        response = session.get('https://datacenter-web.eastmoney.com/api/data/v1/get',
                                               params=params, timeout=(8, 20))
                        response.raise_for_status()
                        payload = response.json()
                        result = payload.get('result')
                        if not result or not result.get('data'):
                            raise ValueError(f'empty disclosure response for {period}')
                        break
                    except (requests.RequestException, ValueError):
                        if attempt:
                            raise
                rows.extend(result['data'])
                if params['pageNumber'] >= int(result['pages']):
                    if len(rows) != int(result['count']):
                        raise ValueError(f'incomplete pages: {period}')
                    break
                params['pageNumber'] += 1
    frame = pd.DataFrame(rows)
    if not set(SOURCE_MAP).issubset(frame) or frame['SECURITY_CODE'].duplicated().any():
        raise ValueError(f'invalid disclosure keys/schema: {period}')
    save_json(path, rows)
    output = {'period': period, 'rows': len(rows), 'requests': calls,
              'cached': False, 'seconds': round(time.monotonic()-started, 2),
              'sha256': sha256(path)}
    print(json.dumps(output), flush=True)
    return output


def download() -> None:
    periods = pd.period_range('2018Q4', '2026Q2', freq='Q')
    dates = [p.end_time.strftime('%Y-%m-%d') for p in periods]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download_period, dates))
    save_json(REPORT / 'source_manifest.json', {
        'fetched_at': pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        'provider': 'AKShare Eastmoney upstream, named-field parser; old wrapper incompatible',
        'periods': results,
        'historical_appointment_publication_timestamps_available': False,
    })


def timing_for_symbol(dates: np.ndarray, records: pd.DataFrame,
                      calendar: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calendar counts with release visibility and two-session polling.

    Since = number of exchange sessions in (actual_date, signal_date].
    No release is visible on actual_date itself. Polling snapshots are indexed
    to the common exchange calendar, never to this stock's sparse signals.
    Approx appointments become visible after period end; this assumption is
    not historical evidence and is never presented as point-in-time data.
    """
    idx = np.searchsorted(calendar, dates, side='right') - 1
    actual = np.sort(records['actual'].dropna().unique().astype('datetime64[ns]'))
    actual = np.unique(actual)
    visible = np.searchsorted(calendar, actual, side='right')
    release_count = visible
    since = np.full(len(dates), np.nan, dtype=np.float32)
    stale = since.copy()
    # A disclosure on session D is available starting D+1. With a refresh every
    # two sessions, an odd-index release waits until the next even-index poll.
    for destination, observed in [(since, idx), (stale, idx - idx % 2)]:
        pos = np.searchsorted(visible, observed, side='right') - 1
        valid = pos >= 0
        destination[valid] = idx[valid] + 1 - release_count[pos[valid]]
    until = np.full(len(dates), np.nan, dtype=np.float32)
    latest = records[['first', 'change1', 'change2', 'change3']].ffill(axis=1).iloc[:, -1]
    for start, planned, released in zip(records['period'] + pd.Timedelta(days=1), latest, records['actual']):
        if pd.isna(start) or pd.isna(planned):
            continue
        valid = (dates >= start.to_datetime64()) & (dates <= planned.to_datetime64())
        if pd.notna(released):
            valid &= dates <= released.to_datetime64()
        count = np.searchsorted(calendar, planned.to_datetime64(), side='right') - idx[valid] - 1
        prior = until[valid]
        until[valid] = np.where(np.isnan(prior), count, np.minimum(prior, count))
    return since, until, stale


def self_check() -> None:
    cal = pd.bdate_range('2024-01-01', '2024-01-15').to_numpy()
    records = pd.DataFrame({'period': pd.to_datetime(['2023-12-31', '2024-03-31']),
                            'first': pd.to_datetime(['2024-01-10', None]),
                            'change1': pd.to_datetime(['2024-01-08', None]),
                            'change2': pd.to_datetime([None, None]),
                            'change3': pd.to_datetime([None, None]),
                            'actual': pd.to_datetime(['2024-01-05', '2024-04-30'])})
    dates = pd.to_datetime(['2024-01-04', '2024-01-05', '2024-01-08',
                            '2024-01-09', '2024-01-10']).to_numpy()
    since, until, stale = timing_for_symbol(dates, records, cal)
    np.testing.assert_allclose(since, [np.nan, np.nan, 1, 2, 3], equal_nan=True)
    np.testing.assert_allclose(stale, [np.nan, np.nan, np.nan, 2, 3], equal_nan=True)
    np.testing.assert_allclose(until, [2, 1, np.nan, np.nan, np.nan], equal_nan=True)
    # A weekend disclosure is unavailable before Monday, when age is 1.
    records.loc[0, 'actual'] = pd.Timestamp('2024-01-06')
    since, _, _ = timing_for_symbol(dates, records, cal)
    np.testing.assert_allclose(since, [np.nan, np.nan, 1, 2, 3], equal_nan=True)
    print('Timing self-check passed: no same-day/future release, holiday counts, latest revision, two-session polling.', flush=True)


def build_timing() -> None:
    self_check()
    keys = pd.read_parquet(DATA / 'events.parquet', columns=['symbol', 'date'])
    keys['date'] = pd.to_datetime(keys['date']).dt.normalize()
    paths = _daily_paths(ROOT / 'data/raw/daily_partitioned', '2018-01-01', '2026-10-01')
    calendar = _market_calendar(paths, '2018-01-01', '2026-10-01').to_numpy()
    pd.Series(calendar).to_csv(DATA / 'calendar.csv', index=False, header=['date'])
    frames = []
    for path in sorted((DATA / 'disclosure_raw').glob('*.json')):
        f = pd.DataFrame(json.loads(path.read_text())).rename(columns=SOURCE_MAP)
        f = f[list(SOURCE_MAP.values())]
        f['symbol'] = f['symbol'].astype(str).str.zfill(6)
        for c in ['period', 'first', 'change1', 'change2', 'change3', 'actual']:
            f[c] = pd.to_datetime(f[c], errors='coerce')
        frames.append(f)
    records = pd.concat(frames, ignore_index=True)
    if records.duplicated(['symbol', 'period']).any():
        raise ValueError('duplicated stock/report-period')
    keys['code'] = keys['symbol'].str.split('.').str[0].str.zfill(6)
    by_code = {s: g for s, g in records.groupby('symbol', sort=False)}
    for col in [SINCE, UNTIL, STALE]:
        keys[col] = np.nan
    empty = records.iloc[:0]
    for code, group in keys.groupby('code', sort=False):
        values = timing_for_symbol(group['date'].to_numpy(), by_code.get(code, empty), calendar)
        keys.loc[group.index, [SINCE, UNTIL, STALE]] = np.column_stack(values)
    keys = keys.drop(columns='code')
    for col in [SINCE, UNTIL, STALE]:
        keys[col] = keys[col].astype('float32')
    keys.to_parquet(DATA / 'timing_features.parquet', index=False)
    coverage = keys.assign(year=keys['date'].dt.year).groupby('year').agg(
        rows=('symbol', 'size'), since_coverage=(SINCE, lambda s: s.notna().mean()),
        appointment_coverage=(UNTIL, lambda s: s.notna().mean()),
        stale_coverage=(STALE, lambda s: s.notna().mean()),
    )
    coverage.to_csv(REPORT / 'feature_coverage.csv')
    print(coverage.to_string(), flush=True)


def load_frame() -> pd.DataFrame:
    columns = ['symbol', 'date', *LEFT_SIDE_FACTOR_COLUMNS, *LEFT_SIDE_SIGNALS]
    events = pd.read_parquet(DATA / 'events.parquet', columns=columns)
    labels = pd.read_parquet(DATA / 'labels.parquet', columns=[
        'symbol', 'date', 'mature', 'locked_limit_up', 'good_path5',
        'terminal_return', 'label_end_date'])
    labels = labels[labels.mature & ~labels.locked_limit_up & labels.good_path5.notna()]
    frame = events.merge(labels[['symbol', 'date', 'good_path5', 'terminal_return', 'label_end_date']],
                         on=['symbol', 'date'], validate='one_to_one')
    features = pd.read_parquet(DATA / 'timing_features.parquet')
    frame = frame.merge(features, on=['symbol', 'date'], validate='one_to_one')
    frame = frame.sort_values(['date', 'symbol'], kind='stable').reset_index(drop=True)
    frame['good_path5'] = frame.good_path5.astype('int8')
    frame[AVAILABLE] = frame[SINCE].notna().astype('float32')
    return frame


def evaluate(frame: pd.DataFrame, probability: np.ndarray, arm: str, fold: str) -> tuple[dict, pd.DataFrame]:
    out = frame[['symbol', 'date', 'good_path5', 'terminal_return']].copy()
    out['probability'] = probability
    out['arm'] = arm
    out['fold'] = fold
    row = {'fold': fold, 'arm': arm, 'rows': len(out),
           'ap': float(average_precision_score(out.good_path5, probability)),
           'auc': float(roc_auc_score(out.good_path5, probability)),
           'appointment_approximation': UNTIL in ARMS[arm]}
    daily = []
    for k in [10, 20]:
        ranked = out.sort_values(['date', 'probability', 'symbol'], ascending=[True, False, True], kind='stable')
        selected = ranked.groupby('date', sort=False).head(k).copy()
        selected['net'] = selected.terminal_return - .002
        net = selected.net
        gp = net[net > 0].sum(); gl = -net[net < 0].sum()
        row.update({f'top{k}_trades': len(selected), f'top{k}_net_return': float(net.mean()),
                    f'top{k}_win_rate': float((net > 0).mean()),
                    f'top{k}_profit_factor': float(gp/gl) if gl else None})
        by_date = selected.groupby('date').agg(net=('net', 'mean'), count=('net', 'size'))
        by_date['top_k'] = k
        by_date['arm'] = arm
        by_date['fold'] = fold
        daily.append(by_date.reset_index())
    return row, pd.concat(daily, ignore_index=True)


def block_interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261029)
    length = len(values)
    starts = rng.integers(0, length, size=(2000, int(np.ceil(length/5))))
    positions = (starts[..., None] + np.arange(5)) % length
    samples = values[positions.reshape(2000, -1)[:, :length]].mean(axis=1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def train(jobs: int, folds: list[str]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = {
        'target': 'good_path5', 'entry': 'next_close', 'horizon': 5,
        'round_trip_cost_bps': 20, 'base_features': list(LEFT_SIDE_FACTOR_COLUMNS),
        'arms': ARMS, 'model_spec': asdict(LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC),
        'folds': [asdict(f) for f in DEFAULT_YEAR_FOLDS if f.name in folds],
        'no_hyperparameter_or_threshold_search': True,
        'historical_appointment_features_are_not_point_in_time': True,
        'since_visibility': 'strictly next exchange session after actual_date',
        'refresh2': 'poll on even global exchange-calendar indices; carry stale latest report',
        'appointment_visibility_assumption': 'period_end+1 through last revised planned date or known actual release, non-PIT',
        'evaluation': 'all samples retained regardless of timing coverage; paired daily top10/top20',
        'uncertainty': '2000 circular block bootstraps, block length 5 exchange sessions',
        'previously_observed_test_years': True, 'production_modified': False,
        'input_hashes': {name: sha256(DATA / name) for name in [
            'events.parquet', 'labels.parquet', 'timing_features.parquet']},
    }
    contract_path = REPORT / 'experiment_contract.json'
    if contract_path.exists():
        prior = json.loads(contract_path.read_text())
        if prior['input_hashes'] != contract['input_hashes']:
            raise ValueError('Research inputs changed: use a new report directory')
    save_json(contract_path, contract)
    frame = load_frame()
    print(f'Training frame {len(frame):,} rows, {frame.date.min()} to {frame.date.max()}', flush=True)
    for fold in DEFAULT_YEAR_FOLDS:
        if fold.name not in folds:
            continue
        splits = split_by_year_fold(frame, fold)
        early, calibration, _ = _split_validation_stages(splits.validation.reset_index(drop=True), 'good_path5')
        for arm, additions in ARMS.items():
            metrics_path = REPORT / f'metrics_{fold.name}_{arm}.json'
            if metrics_path.exists():
                continue
            started = time.monotonic()
            print(f'Fitting {fold.name} {arm}: train={len(splits.train):,}, early={len(early):,}, test={len(splits.test):,}', flush=True)
            model = _fit_unified_model(splits.train.reset_index(drop=True), early, calibration,
                                       [*LEFT_SIDE_FACTOR_COLUMNS, *additions], 'good_path5', n_jobs=jobs)
            probability = model.predict_proba(splits.test)[:, 1]
            metrics, daily = evaluate(splits.test, probability, arm, fold.name)
            metrics['seconds'] = round(time.monotonic()-started, 2)
            metrics['best_iteration'] = model.base_model.best_iteration
            importance = dict(zip(model.base_model.feature_names_in_, model.base_model.classifier.feature_importances_.tolist()))
            metrics['new_feature_importance'] = {c: importance.get(c, 0.0) for c in additions}
            metrics['train_rows'] = len(splits.train)
            metrics['validation_rows'] = len(splits.validation)
            daily.to_parquet(REPORT / f'daily_{fold.name}_{arm}.parquet', index=False)
            pred = splits.test[['symbol', 'date']].copy()
            pred['probability'] = probability
            pred.to_parquet(REPORT / f'predictions_{fold.name}_{arm}.parquet', index=False)
            save_json(metrics_path, metrics)
            print(json.dumps(metrics), flush=True)
            del model, probability, pred
            gc.collect()
        del splits, early, calibration
        gc.collect()
    summarize()


def summarize() -> None:
    metrics = pd.DataFrame([json.loads(p.read_text()) for p in sorted(REPORT.glob('metrics_*.json'))])
    metrics.to_csv(REPORT / 'metrics.csv', index=False)
    daily = pd.concat([pd.read_parquet(p) for p in sorted(REPORT.glob('daily_*.parquet'))], ignore_index=True)
    daily.to_parquet(REPORT / 'daily_returns.parquet', index=False)
    deltas = []
    for fold in sorted(metrics.fold.unique()):
        sub = metrics[metrics.fold == fold].set_index('arm')
        baseline = sub.loc['baseline']
        for arm in sub.index:
            if arm == 'baseline':
                continue
            row = {'fold': fold, 'arm': arm, 'delta_ap': sub.loc[arm, 'ap']-baseline.ap}
            for k in [10, 20]:
                group = daily[(daily.fold == fold) & (daily.top_k == k)]
                paired = group.pivot(index='date', columns='arm', values='net')
                delta = (paired[arm]-paired.baseline).dropna().to_numpy()
                row[f'top{k}_delta_net_return'] = sub.loc[arm, f'top{k}_net_return']-baseline[f'top{k}_net_return']
                row[f'top{k}_daily_equal_weight_delta'] = float(delta.mean())
                row[f'top{k}_daily_delta_ci95'] = block_interval(delta)
            deltas.append(row)
    save_json(REPORT / 'increment_comparison.json', deltas)
    print(json.dumps(deltas, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download', 'timing', 'train', 'self-check', 'summarize'])
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--folds', nargs='+', default=['A', 'B', 'C'])
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.command == 'download': download()
    elif args.command == 'timing': build_timing()
    elif args.command == 'train': train(args.jobs, args.folds)
    elif args.command == 'self-check': self_check()
    else: summarize()


if __name__ == '__main__':
    main()
