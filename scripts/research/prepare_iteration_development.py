"""Build development-only model inputs and dated industry relative features."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.ml.rank_fusion import file_sha256
from model_iteration_v2_protocol import OUT, BOUNDS
from build_iteration_fresh_features import scan

KEYS = ['symbol', 'date']
INDUSTRY_FEATURES = ['industry_return_20d', 'industry_return_60d', 'industry_relative_20d',
                     'industry_relative_60d', 'industry_breadth', 'industry_member_count', 'industry_known']
EVENT_FEATURES = ['event_age', 'event_gap', 'event_recent_count', 'execution_liquidity', 'execution_vol_volume']


def industry_features(keys):
    cache = OUT / 'industry_features.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    membership_path = OUT / 'industry_source/memberships.parquet'
    membership = pd.read_parquet(membership_path)
    membership = membership[['ts_code', 'l1_code', 'in_date', 'out_date']].drop_duplicates()
    membership['available'] = pd.to_datetime(membership.in_date, format='%Y%m%d', errors='coerce').astype('datetime64[ns]')
    membership['expired'] = pd.to_datetime(membership.out_date, format='%Y%m%d', errors='coerce').astype('datetime64[ns]')
    membership = membership.dropna(subset=['available']).rename(columns={'ts_code': 'symbol'})
    paths = sorted(p for p in (ROOT / 'data/raw/daily_partitioned').glob('*/*.parquet')
                   if '202310' <= p.parent.name.split('=')[-1] <= '202611')
    raw = scan(paths, ['symbol', 'date', 'close', 'pre_close'])
    raw['date'] = pd.to_datetime(raw.date).astype('datetime64[ns]')
    raw = raw.loc[raw.date.le('2026-11-11')].sort_values(KEYS).reset_index(drop=True)
    if raw.duplicated(KEYS).any():
        raise ValueError('Duplicate bars in industry reconstruction')
    raw['ret'] = raw.close / raw.pre_close - 1
    raw['ret'] = raw.ret.where(np.isfinite(raw.ret) & raw.ret.gt(-1))
    for window in [20, 60]:
        raw[f'stock_return_{window}d'] = np.expm1(np.log1p(raw.ret).groupby(raw.symbol).transform(lambda s: s.rolling(window).sum()))
    raw = pd.merge_asof(raw.sort_values(['date', 'symbol']), membership.sort_values(['available', 'symbol']),
        left_on='date', right_on='available', by='symbol', direction='backward')
    valid = raw.available.notna() & (raw.expired.isna() | raw.date.lt(raw.expired))
    raw.loc[~valid, 'l1_code'] = None
    conflicts = []
    for symbol, group in membership.groupby('symbol', sort=False):
        records = list(group.itertuples())
        for i, first in enumerate(records):
            for second in records[i + 1:]:
                if first.l1_code == second.l1_code:
                    continue
                start = max(first.available, second.available)
                end = min(first.expired if pd.notna(first.expired) else pd.Timestamp.max,
                          second.expired if pd.notna(second.expired) else pd.Timestamp.max)
                if start < end:
                    conflicts.append((symbol, start, end))
    by_symbol = raw.groupby('symbol', sort=False).indices
    for symbol, start, end in conflicts:
        indexes = by_symbol.get(symbol, [])
        part = raw.iloc[indexes]
        raw.loc[part.index[part.date.ge(start) & part.date.lt(end)], 'l1_code'] = None
    known = raw.loc[raw.l1_code.notna()]
    grouped = known.groupby(['l1_code', 'date']).agg(industry_return_1d=('ret', 'mean'),
        industry_breadth=('ret', lambda s: float(s.gt(0).mean())), industry_member_count=('symbol', 'size')).reset_index()
    grouped = grouped.sort_values(['l1_code', 'date'])
    for window in [20, 60]:
        grouped[f'industry_return_{window}d'] = np.expm1(np.log1p(grouped.industry_return_1d).groupby(grouped.l1_code).transform(lambda s: s.rolling(window).sum()))
    raw = raw.merge(grouped, on=['l1_code', 'date'], how='left', validate='many_to_one')
    for window in [20, 60]:
        raw[f'industry_relative_{window}d'] = raw[f'stock_return_{window}d'] - raw[f'industry_return_{window}d']
    raw['industry_known'] = raw.l1_code.notna().astype(float)
    result = keys.merge(raw[KEYS + INDUSTRY_FEATURES + ['l1_code']], on=KEYS, how='left', validate='one_to_one')
    result.to_parquet(cache, index=False)
    (OUT / 'industry_audit.json').write_text(json.dumps({'membership_rows': len(membership),
        'overlapping_conflicting_intervals': len(conflicts), 'ambiguous_membership': 'Set to unknown',
        'candidate_rows': len(result), 'known_fraction': float(result.industry_known.fillna(0).mean()),
        'membership_sha256': file_sha256(membership_path),
        'scope': 'All observed members in each industry on that day; interval-based membership and trailing windows only. Retrospective provider revisions remain possible.'}, indent=2))
    return result


def main():
    meta = json.loads((ROOT / 'reports/research/full_rank_comparison_v2/dataset_complete.json').read_text())
    core = meta['core']
    extras = ['turnover_rate', 'market_breadth_above_ma20_ratio', 'market_volatility_13w',
              'selector_return_20d', 'event_age', 'event_gap', 'event_recent_count']
    labels = ['maturity', 'planned_maturity', 'outcome', 'mae', 'mfe', 'good_path5', 'unfilled']
    sources = {}
    for side in ['left', 'right']:
        source = ROOT / f'reports/research/sample_label_research_v1/dataset_{side}.parquet'
        columns = list(dict.fromkeys(KEYS + core + extras + labels))
        frame = pd.read_parquet(source, columns=columns)
        frame['date'] = pd.to_datetime(frame.date).astype('datetime64[ns]')
        sources[side] = frame
    keys = pd.concat([f[KEYS] for f in sources.values()] + [pd.read_parquet(OUT / 'fresh_features/keys.parquet')], ignore_index=True).drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    industry = industry_features(keys)
    for side, frame in sources.items():
        frame = frame.merge(industry, on=KEYS, how='left', validate='one_to_one')
        frame['execution_liquidity'] = frame.turnover_rate / frame.volatility_20d.clip(lower=.05)
        frame['execution_vol_volume'] = frame.volatility_20d * frame.volume_relative_20d
        frame.to_parquet(OUT / f'development_{side}.parquet', index=False)
    (OUT / 'development_manifest.json').write_text(json.dumps({'core': core, 'industry': INDUSTRY_FEATURES,
        'event': EVENT_FEATURES, 'bounds': BOUNDS,
        'rows': {s: len(f) for s, f in sources.items()},
        'source_hashes': {s: file_sha256(ROOT / f'reports/research/sample_label_research_v1/dataset_{s}.parquet') for s in sources},
        'script_sha256': file_sha256(Path(__file__))}, indent=2))
    print('DEVELOPMENT READY', {s: len(f) for s, f in sources.items()}, flush=True)


if __name__ == '__main__':
    main()
