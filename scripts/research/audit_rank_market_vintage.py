"""Separate historical source membership changes from calculator differences."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.data.market_data_store import MarketDataStore,MarketDataStoreConfig
from quant.features.rank_runtime import market_features,KEYS
from quant.routine.rank_feature_runtime import _dated


def main() -> None:
    start='2018-01-01';end='2026-06-30'
    old_keys=pd.read_parquet(ROOT/'reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet',columns=KEYS)
    old_keys=old_keys.loc[old_keys.date.between(start,end)]
    store=MarketDataStore(MarketDataStoreConfig(backend='parquet',root=ROOT/'data/raw'))
    current=_dated(store.read_market_range('daily',start_date='20180101',end_date='20260630',
        columns=('ts_code','trade_date','close','pre_close','pct_chg','volume','vol')))
    same=pd.concat([current.loc[current.date.lt('2024-01-01')],
                    current.loc[current.date.ge('2024-01-01')].merge(old_keys,on=KEYS,validate='one_to_one')],ignore_index=True)
    market,ranks=market_features(same)
    columns=['market_breadth_above_ma5_ratio','market_breadth_ma60_eligible_count',
             'market_breadth_new_high_low_ratio_60d','market_sentiment_5d',
             'return_120d_cross_section_pct','volatility_60d_cross_section_pct','db_total_mv_pct_rank']
    # Names are taken from the recorded mismatch report, not inferred aliases.
    columns=sorted({e['feature'] for date in ['2026-03-31','2026-06-30']
        for e in json.loads((ROOT/f'reports/research/rank_runtime_parity/{date}.json').read_text())['mismatches']
        if e['feature'] not in ['ts_volume_ratio_ma20','turnover_rate_f_ma20']})
    original=pd.read_parquet(ROOT/'reports/research/full_rank_comparison_v2/dataset_left.parquet',columns=KEYS+columns)
    checks=[]
    for date in ['2026-03-31','2026-06-30']:
        keys=old_keys.loc[old_keys.date.eq(date)]
        actual=keys.merge(ranks,on=KEYS,validate='one_to_one').merge(market,on='date',validate='many_to_one')
        basic=pd.read_parquet(ROOT/f'data/raw/daily_basic/{date.replace("-", "")}.parquet')
        basic=basic.loc[basic.ts_code.isin(keys.symbol)].copy()
        rank=np.log(basic.total_mv.replace(0,np.nan)).astype('float32').rank(pct=True)
        actual['db_total_mv_pct_rank']=actual.symbol.map(pd.Series(rank.to_numpy(),index=basic.ts_code))
        expected=original.loc[original.date.eq(date)].sort_values('symbol').head(6)
        comparison=expected[KEYS].merge(actual[KEYS+columns],on=KEYS,validate='one_to_one')
        errors=[]
        for col in columns:
            if not np.isclose(expected[col],comparison[col],rtol=1e-5,atol=1e-6,equal_nan=True).all():
                errors.append({'feature':col,'expected':expected[col].tolist(),'actual':comparison[col].tolist()})
        now=current.loc[current.date.eq(date),'symbol']
        from quant.features.rank_runtime import ASHARE
        now=now.loc[now.str.fullmatch(ASHARE)|now.eq('302132.SZ')]
        checks.append(dict(date=date,rows=len(expected),columns=columns,source_added_symbols=sorted(set(now)-set(keys.symbol)),
                           same_membership_passed=not errors,errors=errors))
    basic_checks=[]
    expected=pd.read_parquet(ROOT/'reports/research/full_rank_comparison_v2/dataset_left.parquet',columns=KEYS+['ts_volume_ratio_ma20','turnover_rate_f_ma20'])
    expected=expected.loc[expected.date.eq('2026-06-30')].sort_values('symbol').head(6)
    frozen=pd.read_parquet(ROOT/'reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet',columns=KEYS+['ts_volume_ratio','turnover_rate_f'])
    frozen=frozen.loc[frozen.symbol.isin(expected.symbol)&frozen.date.le('2026-06-30')].sort_values(KEYS).groupby('symbol').tail(20)
    live=pd.concat([pd.read_parquet(ROOT/f'data/raw/daily_basic/{date:%Y%m%d}.parquet',columns=['ts_code','volume_ratio','turnover_rate_f']).assign(date=date)
                    for date in sorted(frozen.date.unique())],ignore_index=True)
    live=live.rename(columns={'ts_code':'symbol','volume_ratio':'ts_volume_ratio'})
    joined=frozen.merge(live,on=KEYS,suffixes=('_frozen','_current'),validate='one_to_one')
    for col in ['ts_volume_ratio','turnover_rate_f']:
        changed=joined.loc[~np.isclose(joined[f'{col}_frozen'],joined[f'{col}_current'],equal_nan=True),KEYS+[f'{col}_frozen',f'{col}_current']]
        rebuilt=frozen.groupby('symbol')[col].mean().reindex(expected.symbol)
        np.testing.assert_allclose(rebuilt.to_numpy(),expected[f'{col}_ma20'].to_numpy(),rtol=1e-6,atol=1e-6)
        basic_checks.append(dict(feature=f'{col}_ma20',same_input_passed=True,changed_cells=changed.to_dict('records')))
    report=dict(passed=all(c['same_membership_passed'] for c in checks),checks=checks,basic_checks=basic_checks,
        note='Production retains the complete current observed universe. Historical source membership is restored only in this audit, never in live inference.')
    out=ROOT/'reports/research/rank_runtime_parity/source_vintage_audit.json'
    out.write_text(json.dumps(report,indent=2,default=str));print(json.dumps(report,indent=2,default=str))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
