"""Reconstruct full-registry features on unfiltered 2024+ research candidates.

No financial snapshot is certified PIT just because its snapshot date is old.
Every registry column receives an admission decision; no outcome selects keys.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/research')]
import evaluate_local_ranknet_fusion as prior
from quant.features.factor_registry import registry_frame
from quant.features.index_ma_state import compute_index_ma_state_features
from quant.features.project_factor_layer import calculate_legacy_market_factors
from quant.features.variable_library import build_continuous_ohlc
from quant.research.left_side_unified_features import compute_left_side_rule_features, compute_left_side_signal_flags
from quant.research.left_rule_replay import replay_left_rules
from quant.research.right_side_unified_features import compute_right_side_rule_features
from quant.research.short_side_groups import GROUP_MEMBERS
from train_chan_daily_models import add_stock_features

OUT = Path(os.environ.get('FULL_RANK_OUT', ROOT / 'reports/research/full_rank_comparison_v1'))
KEYS = ['symbol', 'date']


def stock_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Price-only materializers; never receive labels, current names or industry."""
    raw = raw.sort_values('date').reset_index(drop=True)
    if raw.duplicated('date').any():
        raise ValueError('Duplicate raw bar')
    right = compute_right_side_rule_features(raw)
    left = replay_left_rules(raw)
    flags = compute_left_side_signal_flags(raw, rule_features=left)
    # The retrospective ST name is intentionally absent from raw inputs.
    extras = calculate_legacy_market_factors(raw, factor_schema_version='current')
    continuous = raw.copy()
    continuous[['open', 'high', 'low', 'close']] = build_continuous_ohlc(raw)[['open', 'high', 'low', 'close']]
    chan_price = add_stock_features(continuous)
    c = continuous.close
    price = pd.DataFrame(index=raw.index)
    price['median_close_60'] = c.rolling(60).median()
    price['ma_120_slope_20d'] = c.rolling(120).mean().pct_change(20, fill_method=None)
    for w in [20, 60, 120]:
        price[f'close_to_ma{w}'] = c / c.rolling(w).mean()
    price['close_to_median60'] = c / price.median_close_60
    pct = raw.pct_chg.astype(float)
    price['selector_return_1d'] = pct
    for w in [3, 5, 10, 20, 60]:
        price[f'selector_return_{w}d'] = np.expm1(np.log1p(pct / 100).rolling(w).sum()) * 100
    for w in [5, 10, 20, 60]:
        price[f'selector_volatility_{w}d'] = pct.rolling(w).std()
        price[f'selector_volume_relative_{w}d'] = raw.volume / raw.volume.rolling(w).mean()
    for w in [5, 10, 20]:
        price[f'selector_positive_ratio_{w}d'] = pct.gt(0).astype(float).rolling(w).mean()
    for w in [5, 20]:
        price[f'selector_turnover_relative_{w}d'] = raw.turnover / raw.turnover.rolling(w).mean()
    price['selector_amplitude_1d'] = (raw.high - raw.low) / raw.pre_close * 100
    price['selector_close_pos'] = (raw.close - raw.low) / (raw.high - raw.low)
    for name, col in [('gap', 'open'), ('high', 'high'), ('low', 'low')]:
        price[f'selector_{name}_1d'] = (raw[col] / raw.pre_close - 1) * 100
    for name in ['SB1', 'SUPER_B1', 'LOW_PULLBACK']:
        price[f'group__{name}'] = flags[name].astype(float)
    result = pd.concat([raw[KEYS], right, left, extras, chan_price.drop(columns=KEYS), price], axis=1)
    result = result.loc[:, ~result.columns.duplicated(keep='first')]
    names = registry_frame().query("role in ['feature', 'research_feature']").name
    # Limit-up classification consumes ST/name metadata; do not manufacture historical ST.
    keep = [c for c in names if c in result and not c.startswith('limit_up_cnt_')]
    return pd.concat([raw[KEYS], result[keep].astype('float32').replace([np.inf, -np.inf], np.nan), flags.B1.astype(float).rename('_left_B1')], axis=1)


def worker(task: tuple[str, pd.DataFrame, list[pd.Timestamp]]) -> str:
    symbol, raw, dates = task
    full = stock_features(raw)
    full.loc[full.date.isin(dates)].to_parquet(OUT / 'stock_v2' / f'{symbol}.parquet', index=False)
    return symbol


def strict_join(keys: pd.DataFrame, source: pd.DataFrame, *, prior_day: bool = False) -> pd.DataFrame:
    """Exact signal-day observation, or strict earlier publication snapshot."""
    keys = keys.assign(date=pd.to_datetime(keys.date).astype('datetime64[ns]'))
    source = source.assign(date=pd.to_datetime(source.date).astype('datetime64[ns]'))
    if source.duplicated(KEYS).any():
        raise ValueError('Duplicate source keys')
    if prior_day:
        a = keys.assign(_row=np.arange(len(keys))).sort_values(['date', 'symbol'])
        b = source.rename(columns={'date': '_available'}).sort_values(['_available', 'symbol'])
        result = pd.merge_asof(a, b, left_on='date', right_on='_available', by='symbol',
                               allow_exact_matches=False, direction='backward', tolerance=pd.Timedelta(days=7))
        valid = result._available.notna()
        assert (result.loc[valid, '_available'] < result.loc[valid, 'date']).all()
        return result.sort_values('_row').drop(columns=['_row', '_available']).reset_index(drop=True)
    return keys.merge(source, on=KEYS, how='left', validate='one_to_one')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stock_v2').mkdir(exist_ok=True)
    registry = registry_frame().query("role in ['feature', 'research_feature']").copy()
    prior_protocol = json.loads((prior.OUT / 'protocol.json').read_text())
    for name, digest in prior_protocol['input_hashes'].items():
        if prior.sha256(ROOT / name) != digest:
            raise ValueError(f'Original causal cohort source changed: {name}')
    original = json.loads((ROOT / 'reports/research/right_t012_full_dataset_v4/protocol.json').read_text())['registry_candidates']
    source_paths: set[Path] = set()
    cohorts = {}
    for side, frame, core in prior.datasets():
        cohorts[side] = frame.loc[frame.source_date.ge('2024-01-01')].reset_index(drop=True)
    keys = pd.concat([f[['symbol', 'source_date']].rename(columns={'source_date': 'date'}) for f in cohorts.values()]).drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    paths = sorted(p for p in (ROOT / 'data/raw/daily_partitioned').glob('*/*.parquet') if '201801' <= p.parent.name.split('=')[-1] <= '202609')
    source_paths.update(paths)
    raw = pl.concat([pl.read_parquet(p, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'volume', 'amount', 'turnover']) for p in paths], how='vertical_relaxed').with_columns(pl.col('date').cast(pl.Datetime('ns'))).sort(['symbol', 'date'])
    print('Raw bars', raw.height, 'candidate keys', len(keys), flush=True)
    # Prefix invariance tests are recorded per output column, before training.
    checks = []
    for sym in ['000001.SZ', '600000.SH', '300750.SZ']:
        sample = raw.filter(pl.col('symbol') == sym).to_pandas()
        pd.testing.assert_frame_equal(replay_left_rules(sample), compute_left_side_rule_features(sample), check_exact=False, rtol=1e-12, atol=1e-12)
        full = stock_features(sample)
        for cutoff in ['2024-06-28', '2025-06-30']:
            part = stock_features(sample.loc[sample.date.le(cutoff)])
            expected = full.loc[full.date.le(cutoff)].reset_index(drop=True)
            for c in part.columns.difference(KEYS):
                checks.append(dict(symbol=sym, cutoff=cutoff, name=c, passed=bool(np.allclose(part[c], expected[c], equal_nan=True, rtol=1e-5, atol=1e-6))))
    pd.DataFrame(checks).to_csv(OUT / 'prefix_checks.csv', index=False)
    failed = set(pd.DataFrame(checks).query('not passed').name)
    if failed:
        print('Quarantined prefix failures', sorted(failed), flush=True)
    wanted = {s: g.date.tolist() for s, g in keys.groupby('symbol')}
    symbols = sorted(wanted)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for start in range(0, len(symbols), 64):
            batch = [s for s in symbols[start:start+64] if not (OUT / 'stock_v2' / f'{s}.parquet').exists()]
            groups = raw.filter(pl.col('symbol').is_in(batch)).partition_by('symbol', as_dict=True)
            tasks = [(s, groups[(s,)].to_pandas(), wanted[s]) for s in batch]
            list(pool.map(worker, tasks, chunksize=4))
            print('Rebuilt stocks', min(start+64, len(symbols)), '/', len(symbols), flush=True)
    stock = pd.concat([pd.read_parquet(OUT / 'stock_v2' / f'{s}.parquet') for s in symbols], ignore_index=True)
    # Existing 129 columns retain their audited calculator and warmup, isolating extra-variable gain.
    basic_owned = set(registry.query("layer == 'project_daily'").name) - set(core)
    extra_cols = [c for c in stock if c not in KEYS + core and c not in failed and c not in basic_owned]
    extra = strict_join(keys, stock[KEYS + extra_cols])
    admitted = {c: ('raw_price_rebuilt', 'signal_close', 'prefix_invariance') for c in extra_cols}
    del stock
    # Full-market cross section is computed before selecting strategy candidates.
    market = raw.filter(pl.col('date') >= pd.Timestamp('2024-01-01')).group_by('date').agg(
        pl.col('pct_chg').mean().alias('selector_market_mean_1d'),
        pl.col('pct_chg').median().alias('selector_market_median_1d'),
        pl.col('pct_chg').std().alias('selector_market_dispersion_1d'),
        (pl.col('pct_chg') > 0).mean().alias('selector_market_up_ratio_1d'),
        (pl.col('pct_chg') >= 5).mean().alias('selector_market_up5_ratio_1d'),
        (pl.col('pct_chg') <= -5).mean().alias('selector_market_down5_ratio_1d')).sort('date').to_pandas()
    for w in [5, 20]:
        for a, b in [('mean', 'mean'), ('up_ratio', 'up_ratio')]:
            market[f'selector_market_{a}_{w}d'] = market[f'selector_market_{b}_1d'].rolling(w).mean()
    del raw
    index_paths = sorted((ROOT / 'reports/research/right_t012_index_sources_v1').glob('*_index_daily.parquet'))
    source_paths.update(index_paths)
    indices = pd.concat([pd.read_parquet(p) for p in index_paths], ignore_index=True)
    im = compute_index_ma_state_features(indices)
    for cutoff in ['2024-06-28', '2025-06-30']:
        short = compute_index_ma_state_features(indices, as_of=cutoff)
        pd.testing.assert_frame_equal(short.reset_index(drop=True), im.loc[im.date.le(cutoff)].reset_index(drop=True))
    for s, label in [(market, 'full_universe_selector_market'), (im, 'index_ma_public_launch_mask')]:
        cols = [c for c in s if c != 'date' and c in set(registry.name) and c not in extra]
        extra = extra.merge(s[['date', *cols]], on='date', how='left', validate='many_to_one')
        admitted.update({c: (label, 'signal_close', 'causal_rolling') for c in cols})
    extra['selector_excess_return_1d'] = extra.selector_return_1d - extra.selector_market_mean_1d
    admitted['selector_excess_return_1d'] = ('full_universe_selector_market', 'signal_close', 'causal_rolling')
    for file in ['stock', 'market']:
        path = ROOT / f'reports/research/right_t012_research_factors_v3/{file}.parquet'
        source_paths.add(path)
        s = pd.read_parquet(path)
        cols = [c for c in s if c in set(registry.name) and c not in extra and c not in core]
        if file == 'stock':
            extra = extra.merge(s[KEYS + cols], on=KEYS, how='left', validate='one_to_one')
        else:
            extra = extra.merge(s[['date', *cols]], on='date', how='left', validate='many_to_one')
        admitted.update({c: (str(path.relative_to(ROOT)), 'signal_close', 'raw_daily_not_financial_snapshot') for c in cols})
    p = ROOT / 'reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet'
    source_paths.add(p)
    basic_cols = [c for c in registry.query("layer == 'project_daily'").name if c not in core]
    basics = pd.read_parquet(p, columns=KEYS + basic_cols)
    # These are dated vendor daily_basic observations, not latest financial statements.
    extra = extra.merge(basics, on=KEYS, how='left', validate='one_to_one')
    admitted.update({c: (str(p.relative_to(ROOT)), 'signal_close_published_after_close_before_next_close', 'dated_vendor_daily_basic_revision_not_archived') for c in basic_cols})
    db_map = {'db_turnover_rate': 'turnover_rate', 'db_turnover_rate_f': 'turnover_rate_f', 'db_volume_ratio': 'ts_volume_ratio',
              **{f'db_{c}': c for c in ['total_mv_log', 'circ_mv_log', 'float_mv_ratio', 'free_float_share_ratio', 'pe_ttm_inv', 'pb_inv', 'ps_ttm_inv']}}
    for target, source in db_map.items():
        extra[target] = extra[source]
        admitted[target] = ('canonical_daily_basic_alias', 'signal_close', 'dated_vendor_daily_basic_revision_not_archived')
    for target, source in [('db_total_mv_pct_rank', 'total_mv_log'), ('db_turnover_rate_pct_rank', 'turnover_rate'), ('db_volume_ratio_pct_rank', 'ts_volume_ratio'), ('db_pb_pct_rank', 'pb_inv')]:
        ranks = basics[KEYS].assign(**{target: basics.groupby('date')[source].rank(pct=True)})
        extra = extra.merge(ranks, on=KEYS, how='left', validate='one_to_one')
        admitted[target] = ('full_universe_daily_basic_rank', 'signal_close', 'candidate_pool_independent_rank')
    # Reconstructed pre-outcome right signals, combined with independently rebuilt left flags.
    p = ROOT / 'reports/research/right_t012_history_v1/all_daily_signals.parquet'
    source_paths.add(p)
    signals = pd.read_parquet(p)
    for group, members in GROUP_MEMBERS.items():
        name = f'group__{group}'
        if name in extra or name not in set(registry.name) or not set(members).issubset(signals):
            continue
        flags = signals[members].astype('boolean') if isinstance(members, list) else signals[list(members)].astype('boolean')
        value = flags.eq(True).any(axis=1).astype(float).mask(flags.isna().all(axis=1))
        extra = extra.merge(signals[KEYS].assign(**{name: value}), on=KEYS, how='left', validate='one_to_one')
        admitted[name] = ('unfiltered_causal_signal_history', 'signal_close', 'no_outcome_filter')
    group_cols = [c for c in extra if c.startswith('group__')]
    if len(group_cols) == 13:
        extra['matched_count'] = extra[group_cols].sum(axis=1, min_count=13) + extra['_left_B1']
        admitted['matched_count'] = ('14_groups_including_B1', 'signal_close', 'no_outcome_filter')
    # Top-list data are released after the source day; use strictly earlier source day.
    from quant.features.market_sentiment import read_top_list_features
    top_root = ROOT / 'data/raw/top_list'
    top = read_top_list_features(top_root, start=pd.Timestamp('2023-12-01')).rename(columns={'ts_code': 'symbol'})
    source_paths.update(top_root.glob('*.parquet'))
    top_cols = ['top_list_count', 'top_net_amount_ratio', 'top_net_rate']
    if len(top):
        # Do not forward-fill an event as though it occurred on today's date.
        prev_dates = pd.Series(pd.DatetimeIndex(basics.date.unique()).sort_values()).shift(1)
        date_map = dict(zip(pd.DatetimeIndex(basics.date.unique()).sort_values(), prev_dates))
        query = keys.copy(); query['date'] = query.date.map(date_map)
        aligned = strict_join(query, top[KEYS + top_cols])
        extra[top_cols] = aligned[top_cols].to_numpy()
        admitted.update({c: ('top_list_previous_session', 'strict_previous_session', 'absent_event_unknown_not_zero') for c in top_cols})
    records = []
    for row in registry.to_dict('records'):
        name = row['name']
        decision = admitted.get(name)
        if name in core:
            decision = ('existing_audited_129', 'signal_close', 'prior_causal_contract')
        reason = ''
        if decision is None:
            if name in failed:
                reason = 'prefix_invariance_failed'
            elif name.startswith('analyst_'):
                reason = 'analyst_history_source_removed_no_report_publication_vintages'
            elif name in ['chan_center_width', 'chan_score']:
                reason = 'full_history_fractal_strokes_use_future_confirmation_and_pivot_replacement'
            elif name.startswith('limit_up_cnt_'):
                reason = 'requires_historical_ST_status_not_current_name'
            elif row['layer'] in ['long_snapshot', 'long_research', 'long_external_candidate']:
                reason = 'snapshot_only_publication_revision_or_historical_industry_not_certified'
            else:
                reason = 'original_source_unavailable_no_verified_materialization'
        records.append(dict(name=name, original_776=name in original, layer=row['layer'], admitted=decision is not None,
                            source=decision[0] if decision else row['source'], timing=decision[1] if decision else '',
                            evidence=decision[2] if decision else '', exclusion_reason=reason))
    records.extend(dict(name=c, original_776=False, layer='joinquant_price_candidate', admitted=True, source='existing_audited_129', timing='signal_close', evidence='prior_causal_contract', exclusion_reason='') for c in core if c not in set(registry.name))
    audit = pd.DataFrame(records)
    audit.to_csv(OUT / 'column_audit.csv', index=False)
    allowed = audit.loc[audit.admitted, 'name'].tolist()
    extra = extra[[*KEYS, *[c for c in allowed if c not in core]]]
    for side, frame in cohorts.items():
        f = frame.merge(extra.rename(columns={'date': 'source_date'}), on=['symbol', 'source_date'], how='left', validate='one_to_one')
        f[allowed] = f[allowed].astype('float32').replace([np.inf, -np.inf], np.nan)
        f.to_parquet(OUT / f'dataset_{side}.parquet', index=False)
        pd.DataFrame({'name': allowed, 'coverage': f[allowed].notna().mean(), 'unique': f[allowed].nunique()}).to_csv(OUT / f'coverage_{side}.csv', index=False)
        print('DATASET', side, len(f), len(allowed), flush=True)
    prior.save_json(OUT / 'dataset_complete.json', dict(candidate_count=len(audit), original_776_admitted=int((audit.admitted & audit.original_776).sum()), admitted=len(allowed),
        original_776=776, new_index_columns=341, jq_columns=19, feature_start='2024-01-01', raw_warmup='2018-01-01',
        prefix_failures=sorted(failed), columns=allowed, core=core,
        limitations='No archived vendor vintages. Mechanical temporal audit and causal reconstruction, NOT absolute vendor-PIT certification. Statement/analyst/industry snapshots quarantined.',
        inputs={**prior_protocol['input_hashes'], **{str(p.relative_to(ROOT)): prior.sha256(p) for p in sorted(source_paths)}},
        dataset_hashes={s: prior.sha256(OUT / f'dataset_{s}.parquet') for s in cohorts},
        script_hash=prior.sha256(Path(__file__))))


if __name__ == '__main__':
    main()
