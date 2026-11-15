"""Materialize the new interval once per stock, without opening its outcomes."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.features import rank_runtime as calc
from quant.features.active_market_value import KEY_INDEX_SPECS
from quant.ml.rank_fusion import RankFusion, file_sha256
from quant.routine.rank_feature_runtime import _dated
from quant.routine.right_side_unified_production import _normalize_ranking_scores
from model_iteration_v2_protocol import OUT, BOUNDS

KEYS = ['symbol', 'date']


def scan(paths, columns, symbols=None, *, code='ts_code'):
    schema = dict(pl.read_parquet_schema(paths[0]))
    if 'date' in schema:
        schema['date'] = pl.Datetime('us')
    for column in columns:
        if column not in ['symbol', 'ts_code', 'date', 'trade_date']:
            schema[column] = pl.Float64
    query = pl.scan_parquet([str(p) for p in paths], hive_partitioning=False,
                            schema=schema, cast_options=pl.ScanCastOptions(datetime_cast='nanosecond-downcast', float_cast='upcast'),
                            missing_columns='insert', extra_columns='ignore')
    if symbols is not None:
        query = query.filter(pl.col(code).is_in(symbols))
    return query.select(columns).collect().to_pandas()


def stock_task(task):
    raw, dates, required = task
    full = calc.stock_price_features(raw)
    if 'selector_return_20d' in required and 'selector_return_20d' not in full:
        history = raw.sort_values('date')
        alias = history[KEYS].copy()
        alias['selector_return_20d'] = np.expm1(np.log1p(history.pct_chg.astype(float) / 100).rolling(20).sum()) * 100
        full = full.merge(alias, on=KEYS, validate='one_to_one')
    columns = [c for c in required if c in full]
    return full.loc[full.date.isin(dates), KEYS + columns].copy()


def main():
    output = OUT / 'fresh_features'
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / 'protocol.json'
    if not protocol_path.exists():
        raise RuntimeError('Freeze the experiment protocol first')
    source_hashes, frames = {}, []
    artifact = joblib.load(ROOT / 'models/production/right_side_unified_canonical_v2/ranking.joblib')
    for side in ['left', 'right']:
        folder = ROOT / f'data/features/{side}_side_unified'
        paths = sorted(p for p in folder.glob('*_features.parquet')
                       if p.name[:8].isdigit() and '20260904' <= p.name[:8] <= '20261111')
        if side == 'left':
            paths.append(ROOT / 'data/features/left_side_ranknet_v6/latest_features.parquet')
        pieces = []
        for path in paths:
            part = pd.read_parquet(path)
            part['date'] = pd.to_datetime(part.date).astype('datetime64[ns]')
            part = part.loc[part.date.ge(BOUNDS['holdout_start']) & part.date.lt(BOUNDS['holdout_end'])]
            if part.empty:
                continue
            source_hashes[str(path)] = file_sha256(path)
            pieces.append(part)
        source = pd.concat(pieces, ignore_index=True)
        if source.duplicated(KEYS).any():
            raise RuntimeError(f'Duplicate {side} archived candidates')
        source['side'] = side
        if side == 'right':
            raw_score = artifact['model'].predict_proba(source)[:, 1]
            source['base_score'] = _normalize_ranking_scores(raw_score, artifact['score_normalization']) / 100
        source.to_parquet(output / f'archive_{side}.parquet', index=False)
        frames.append(source[KEYS + ['side']])
    candidate_keys = pd.concat(frames, ignore_index=True)
    keys = candidate_keys[KEYS].drop_duplicates().sort_values(KEYS).reset_index(drop=True)
    keys.to_parquet(output / 'keys.parquet', index=False)
    model = RankFusion.load(ROOT / 'models/production/left_side_ranknet_v6/manifest.json')
    core = json.loads((ROOT / 'reports/research/full_rank_comparison_v2/dataset_complete.json').read_text())['core']
    required = sorted(set(core) | set(model.required_features) | {'volatility_20d', 'volatility_60d',
                      'return_120d', 'market_return_26w', 'market_breadth_above_ma20_ratio',
                      'selector_return_20d', 'volume_relative_20d', 'turnover_rate'})
    raw_root = ROOT / 'data/raw'
    price_paths = sorted(p for p in (raw_root / 'daily_partitioned').glob('*/*.parquet')
                         if '201401' <= p.parent.name.split('=')[-1] <= '202611')
    basic_paths = sorted(p for p in (raw_root / 'daily_basic').glob('*.parquet')
                         if p.stem.isdigit() and '20130101' <= p.stem <= '20261111')
    calendar = pd.DatetimeIndex(pd.to_datetime([p.stem for p in basic_paths]))
    index_paths = [raw_root / f'index_{spec.ts_code}.parquet' for spec in KEY_INDEX_SPECS]
    flow_paths = {kind: sorted(p for p in (raw_root / kind).glob(f'tushare_{kind}_*.parquet')
                              if '20231001' <= p.stem.rsplit('_', 1)[-1] <= '20261111')
                  for kind in ['moneyflow', 'margin_detail']}
    top_paths = sorted(p for p in (raw_root / 'top_list').glob('tushare_top_list_*.parquet')
                       if '20260501' <= p.stem.rsplit('_', 1)[-1] <= '20261111')
    all_paths = price_paths + basic_paths + index_paths + top_paths + sum(flow_paths.values(), [])
    inventory = lambda: [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in all_paths]
    initial = inventory()
    identity = hashlib.sha256(json.dumps([initial, file_sha256(Path(calc.__file__)),
        file_sha256(Path(__file__)), file_sha256(output / 'keys.parquet'), required]).encode()).hexdigest()
    cache = output / identity
    cache.mkdir(exist_ok=True)
    market_file, ranks_file = cache / 'market.parquet', cache / 'ranks.parquet'
    if not market_file.exists():
        print('FRESH GLOBAL MARKET', flush=True)
        raw = scan([p for p in price_paths if p.parent.name.split('=')[-1] >= '201801'],
                   ['symbol', 'date', 'close', 'pre_close', 'pct_chg'])
        raw['date'] = pd.to_datetime(raw.date).astype('datetime64[ns]')
        raw = raw.loc[raw.date.le('2026-11-11')]
        market, ranks = calc.market_features(raw)
        market = market.loc[market.date.ge('2026-09-04')]
        ranks = ranks.loc[ranks.date.ge('2026-09-04')]
        market.to_parquet(market_file, index=False)
        ranks.to_parquet(ranks_file, index=False)
        del raw
    market, ranks = pd.read_parquet(market_file), pd.read_parquet(ranks_file)
    indices = _dated(pd.concat([pd.read_parquet(p) for p in index_paths], ignore_index=True))
    indices = calc.index_features(indices.loc[indices.date.between('2018-01-01', '2026-11-11')])
    target_basic = _dated(scan([p for p in basic_paths if p.stem >= '20260904'], ['ts_code', 'trade_date', 'total_mv']))
    ranked_basic = target_basic.merge(ranks[KEYS], on=KEYS, validate='one_to_one')
    ranked_basic['db_total_mv_pct_rank'] = np.log(ranked_basic.total_mv.replace(0, np.nan)).astype('float32').groupby(ranked_basic.date).rank(pct=True)
    basic_columns = ['ts_code', 'trade_date', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'total_mv', 'circ_mv',
                     'total_share', 'float_share', 'free_share', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm']
    price_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 'volume', 'amount', 'turnover']
    symbols = sorted(keys.symbol.unique())
    with ProcessPoolExecutor(max_workers=4) as pool:
        for begin in range(0, len(symbols), 128):
            batch = symbols[begin:begin + 128]
            path = cache / f'batch_{begin:05d}.parquet'
            if path.exists():
                continue
            subset = keys.loc[keys.symbol.isin(batch)].copy()
            raw = scan(price_paths, price_columns, batch, code='symbol')
            raw = _dated(raw.loc[pd.to_datetime(raw.date).le('2026-11-11')]).assign(vol=lambda x: x.volume)
            dates = subset.groupby('symbol').date.apply(list).to_dict()
            tasks = [(group, dates[symbol], required) for symbol, group in raw.groupby('symbol', sort=True)]
            prices = pd.concat(pool.map(stock_task, tasks, chunksize=4), ignore_index=True)
            basic = _dated(scan(basic_paths, basic_columns, batch))
            basic_current = calc.basic_features(basic.loc[basic.date.ge('2018-11-01')])
            result = subset.merge(prices, on=KEYS, how='left', validate='one_to_one')
            def attach(other, on=KEYS):
                nonlocal result
                columns = [c for c in required if c in other and c not in result]
                result = result.merge(other[[*on, *columns]], on=on, how='left',
                                      validate='many_to_one' if on == ['date'] else 'one_to_one')
            attach(basic_current)
            attach(calc.jq_price_features(raw, basic))
            attach(calc.periodic_pb_features(subset, basic, calendar=calendar))
            attach(market, ['date'])
            attach(ranks)
            attach(indices, ['date'])
            if 'return_120d_minus_market' in required:
                result['return_120d_minus_market'] = result.return_120d.astype('float32') - result.market_return_26w
            if 'dividend_to_volatility' in required:
                dv = basic_current[KEYS + ['dv_ttm']].merge(prices[KEYS + ['volatility_60d']], on=KEYS, validate='one_to_one')
                dv['dividend_to_volatility'] = dv.dv_ttm.astype('float32') / dv.volatility_60d.astype('float32').replace(0, np.nan)
                attach(dv)
            result = result.drop(columns='db_total_mv_pct_rank', errors='ignore').merge(
                ranked_basic[KEYS + ['db_total_mv_pct_rank']], on=KEYS, how='left', validate='one_to_one')
            for kind, paths in flow_paths.items():
                fields = ['buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount', 'buy_lg_amount', 'sell_lg_amount',
                          'buy_elg_amount', 'sell_elg_amount', 'net_mf_amount'] if kind == 'moneyflow' else ['rzye', 'rqye', 'rzmre', 'rzche']
                flow = scan(paths, ['ts_code', 'trade_date', *fields], batch)
                attach(calc.align_prior(subset, calc.flow_features(flow, kind)))
            events = _dated(scan(top_paths, ['ts_code', 'trade_date', 'net_amount', 'amount'], batch))
            attach(calc.top_list_features(subset, events, calendar,
                pd.DatetimeIndex(pd.to_datetime([p.stem.rsplit('_', 1)[-1] for p in top_paths]))))
            missing = set(required) - set(result)
            if missing:
                raise RuntimeError(f'Unmaterialized fresh features: {sorted(missing)}')
            result = result[KEYS + required].copy()
            result[required] = result[required].astype('float32').replace([np.inf, -np.inf], np.nan)
            result.to_parquet(path, index=False)
            print('FRESH STOCKS', min(begin + 128, len(symbols)), '/', len(symbols), flush=True)
    features = pd.concat([pd.read_parquet(p) for p in sorted(cache.glob('batch_*.parquet'))], ignore_index=True)
    pd.testing.assert_frame_equal(features[KEYS].sort_values(KEYS).reset_index(drop=True), keys)
    features.to_parquet(output / 'features.parquet', index=False)
    for side in ['left', 'right']:
        archive = pd.read_parquet(output / f'archive_{side}.parquet')
        source = archive.drop(columns=[c for c in required if c in archive]).merge(features, on=KEYS, validate='one_to_one')
        if side == 'left':
            source['base_score'] = model.predict(source).ranking_score.to_numpy()
        source.to_parquet(output / f'inputs_{side}.parquet', index=False)
    if inventory() != initial or any(file_sha256(Path(p)) != h for p, h in source_hashes.items()):
        raise RuntimeError('Source changed during fresh feature materialization')
    (output / 'manifest.json').write_text(json.dumps({'rows': len(keys), 'side_rows': len(candidate_keys),
        'symbols': len(symbols), 'dates': int(keys.date.nunique()), 'required_features': required,
        'source_identity': identity, 'archive_hashes': source_hashes,
        'features_sha256': file_sha256(output / 'features.parquet'),
        'scope': 'Dated full production feature archives; all archived candidates retained, no outcome filter.',
        'labels_opened': False, 'script_sha256': file_sha256(Path(__file__))}, indent=2))
    print('FRESH FEATURES COMPLETE', len(keys), flush=True)


if __name__ == '__main__':
    main()
