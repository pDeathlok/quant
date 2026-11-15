"""Recover dated daily flows; quarantine retrospective research snapshots."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/research')]
from build_full_rank_dataset import OUT, KEYS, strict_join
import evaluate_local_ranknet_fusion as prior
from quant.features.long_external_factors import build_moneyflow_weekly, build_margin_weekly


def main():
    meta = json.loads((OUT/'dataset_complete.json').read_text())
    if meta.get('augmentation_complete'):
        print('Dated source augmentation already complete', flush=True); return
    frames = {s: pd.read_parquet(OUT/f'dataset_{s}.parquet') for s in ['left', 'right']}
    keys = pd.concat([f[['symbol', 'source_date']].rename(columns={'source_date':'date'}) for f in frames.values()]).drop_duplicates().reset_index(drop=True)
    additional = keys.copy(); sources={}; inputs={}; details={}
    for folder, build in [('moneyflow', build_moneyflow_weekly), ('margin_detail', build_margin_weekly)]:
        print('SOURCE', folder, flush=True)
        paths = [p for p in sorted((ROOT/'data/raw'/folder).glob('*.parquet')) if '20231001' <= p.stem.rsplit('_',1)[-1] <= '20260904']
        source = pl.concat([pl.read_parquet(p) for p in paths], how='diagonal_relaxed').to_pandas()
        # No outcome-conditioned keys, and no use of partially populated flow records.
        required = (['buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount','buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount','net_mf_amount']
                    if folder == 'moneyflow' else ['rzye','rqye','rzmre','rzche'])
        details[folder] = dict(rows=len(source), incomplete_rows=int(source[required].isna().any(axis=1).sum()))
        source = source.sort_values(['ts_code','trade_date']).reset_index(drop=True)
        key_cols = ['ts_code','trade_date']
        repeated = source.loc[source.duplicated(key_cols,keep=False)]
        conflicts = repeated.groupby(key_cols)[required].nunique(dropna=False).gt(1).any(axis=1)
        ambiguous = conflicts.index[conflicts]
        details[folder].update(duplicate_rows=len(repeated), conflicting_keys=len(ambiguous))
        source = source.drop_duplicates(key_cols,keep='first').reset_index(drop=True)
        # Identical copies are harmless; conflicting vintages stay unavailable.
        conflict_mask = pd.MultiIndex.from_frame(source[key_cols]).isin(ambiguous)
        source.loc[conflict_mask,required] = np.nan
        source_keys = source[['ts_code','trade_date']].drop_duplicates().rename(columns={'trade_date':'date'})
        source_keys.date = pd.to_datetime(source_keys.date)
        values = build(source_keys, source).rename(columns={'ts_code':'symbol'})
        # Legacy helper fills absent flow fields with zero. Mask every affected
        # observation/window so unavailable flow is never reported as neutral.
        if folder == 'moneyflow':
            bad = source[required[:-1]].isna().any(axis=1)
            for c in ['large_net_amount_ratio','small_net_amount_ratio','medium_net_amount_ratio']:
                values.loc[bad.to_numpy(),c] = np.nan
            values.loc[(bad | source.net_mf_amount.isna()).to_numpy(),'moneyflow_net_ratio'] = np.nan
            for w in [3,5]:
                invalid = bad.groupby(source.ts_code,sort=False).transform(lambda s:s.rolling(w,min_periods=1).max()).gt(0)
                values.loc[invalid.to_numpy(),f'large_net_{w}d_ratio'] = np.nan
            invalid = source[['buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount']].isna().any(axis=1)
            invalid = invalid.groupby(source.ts_code,sort=False).transform(lambda s:s.rolling(5,min_periods=1).max()).gt(0)
            values.loc[invalid.to_numpy(),'large_flow_persistence_5d'] = np.nan
        else:
            values['margin_balance_change'] = source.groupby('ts_code',sort=False).rzye.pct_change(20,fill_method=None).to_numpy()
            values['short_pressure_change_5d'] = source.groupby('ts_code',sort=False).rqye.pct_change(5,fill_method=None).to_numpy()
            invalid = source[['rzmre','rzche']].isna().any(axis=1).groupby(source.ts_code,sort=False).transform(lambda s:s.rolling(5,min_periods=1).max()).gt(0)
            values.loc[invalid.to_numpy(),'margin_buy_ratio_5d'] = np.nan
        cols = values.columns.difference(KEYS).tolist()
        aligned = strict_join(keys, values, prior_day=True)
        additional[cols] = aligned[cols].to_numpy()
        sources.update({c:(f'raw_{folder}', 'strict_prior_observation_max7days', 'dated_daily_source') for c in cols})
        inputs.update({str(p.relative_to(ROOT)):prior.sha256(p) for p in paths})
        del source, values
    # Recover simple long-model interactions from original dated daily inputs.
    factor_path = ROOT/'reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet'
    f = pd.read_parquet(factor_path, columns=KEYS+['pe_ttm','pb','ps_ttm','dv_ttm','return_120d','volatility_60d'])
    for source, target in [('pe_ttm','earnings_yield'),('pb','book_yield'),('ps_ttm','sales_yield')]:
        f[target] = 1/f[source].where(f[source] > 0)
    f['dividend_to_volatility'] = f.dv_ttm/f.volatility_60d.replace(0,np.nan)
    for c in ['pe_ttm','pb','dv_ttm','return_120d','volatility_60d']:
        f[f'{c}_cross_section_pct'] = f.groupby('date')[c].rank(pct=True)
    cols = [c for c in f if c not in KEYS+['pe_ttm','pb','ps_ttm','dv_ttm','return_120d','volatility_60d']]
    additional = additional.merge(f[KEYS+cols], on=KEYS, how='left', validate='one_to_one')
    sources.update({c:('dated_full_market_daily_basic', 'signal_close', 'full_universe_rank_not_candidate_filtered') for c in cols})
    inputs[str(factor_path.relative_to(ROOT))] = prior.sha256(factor_path)
    del f
    # Price-only market proxies are calculated over the complete observed universe.
    bars_path = ROOT/'reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet'
    bars = pd.read_parquet(bars_path, columns=KEYS+['close','pre_close'])
    bars['ret'] = (bars.close / bars.pre_close - 1) * 100
    bars.loc[bars.pre_close.le(0), 'ret'] = np.nan
    bars = bars.assign(up=bars.ret.gt(0), up_limit=bars.ret.ge(9.5), down_limit=bars.ret.le(-9.5), strong=bars.ret.ge(5))
    market = bars.groupby('date').agg(n=('ret','count'),limit_up_count_proxy=('up_limit','sum'),down=('down_limit','sum'),
        strong=('strong','sum'),market_up_ratio=('up','mean'),market_median_ret_1d=('ret','median'))
    market['limit_up_ratio_proxy'] = market.limit_up_count_proxy / market.n.replace(0,np.nan)
    market['limit_down_ratio_proxy'] = market.down / market.n.replace(0,np.nan)
    market['strong_up_ratio_proxy'] = market.strong / market.n.replace(0,np.nan)
    market['market_sentiment_5d'] = market.limit_up_ratio_proxy.rolling(5,min_periods=2).mean()
    market['market_panic_5d'] = market.limit_down_ratio_proxy.rolling(5,min_periods=2).mean()
    market = market.drop(columns=['n','down','strong']).reset_index()
    additional = additional.merge(market,on='date',how='left',validate='many_to_one')
    sources.update({c:('raw_market_price_proxies', 'signal_close', 'corporate_action_aware_pre_close_return') for c in market if c!='date'})
    inputs[str(bars_path.relative_to(ROOT))] = prior.sha256(bars_path)
    del bars
    # Published reports were backfilled for CURRENT long candidates. Even after
    # dropping snapshot-only values, historical missingness reveals that future pool.
    report_path = ROOT/'data/raw/analyst_forecasts.parquet'
    reports = pd.read_parquet(report_path, columns=['source','snapshot_only','report_date'])
    details['analyst'] = dict(total_rows=len(reports), snapshot_only_rows=int(reports.snapshot_only.eq(True).sum()),
        admitted=False, reason='collection universe follows current long candidates; no historical collection-universe manifest',
        evidence='src/quant/routine/pipeline.py::_refresh_analyst_forecast_snapshot / _latest_long_analyst_symbols')
    audit = pd.read_csv(OUT/'column_audit.csv')
    mask = audit.name.str.startswith('analyst_')
    audit.loc[mask, 'exclusion_reason'] = 'current_candidate_universe_backfill_and_snapshot_values_no_historical_collection_contract'
    for name,(source,timing,evidence) in sources.items():
        mask = audit.name.eq(name)
        audit.loc[mask,['admitted','source','timing','evidence','exclusion_reason']] = [True,source,timing,evidence,'']
    audit.to_csv(OUT/'column_audit.csv',index=False)
    allowed = audit.loc[audit.admitted,'name'].tolist()
    newcols = [c for c in allowed if c not in meta['columns']]
    for side, frame in frames.items():
        frame = frame.merge(additional[KEYS+newcols].rename(columns={'date':'source_date'}), on=['symbol','source_date'],how='left',validate='one_to_one')
        # Replay intermediates were ratios; canonical interaction definitions subtract one.
        for c in ['close_to_ma20','close_to_ma60','close_to_ma120','close_to_median60']:
            frame[c] -= 1.
        frame[allowed] = frame[allowed].astype('float32').replace([np.inf,-np.inf],np.nan)
        frame.to_parquet(OUT/f'dataset_{side}.parquet',index=False)
        pd.DataFrame({'name':allowed,'coverage':frame[allowed].notna().mean(),'unique':frame[allowed].nunique()}).to_csv(OUT/f'coverage_{side}.csv',index=False)
        meta['dataset_hashes'][side] = prior.sha256(OUT/f'dataset_{side}.parquet')
    meta.update(augmentation_complete=True,columns=allowed,admitted=len(allowed),original_776_admitted=int((audit.admitted&audit.original_776).sum()))
    meta['inputs'].update(inputs)
    prior.save_json(OUT/'source_recovery_audit.json',details)
    prior.save_json(OUT/'dataset_complete.json',meta)
    print('AUGMENTED',len(newcols),'columns; total',len(allowed),flush=True)


if __name__ == '__main__':main()
