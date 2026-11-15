"""Evidence tables for timing, sample membership and historical feature coverage."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.features.long_weekly_factors import IDENTIFIER_COLUMNS,LABEL_PREFIXES

OUT=ROOT/'reports/research/local_ranknet_fusion_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_parquet(ROOT/'data/features/long_entry/weekly_training_v2.parquet')
    index=pd.read_parquet(ROOT/'data/raw/index_000300.SH.parquet')
    calendar=pd.to_datetime(index.trade_date.astype(str),format='%Y%m%d').sort_values().drop_duplicates().to_numpy(dtype='datetime64[ns]')
    positions=np.searchsorted(calendar,d.entry_date.to_numpy(dtype='datetime64[ns]'))+126
    exact=pd.Series(pd.to_datetime([calendar[i] if i<len(calendar) else pd.NaT for i in positions]),index=d.index)
    proxy=d.date+pd.Timedelta(weeks=27)
    purges=[]
    for cutoff in ['2023-01-01','2024-01-01','2025-01-01','2026-01-01']:
        admitted=d.date.ge('2020-01-01')&d.date.lt(cutoff)&d.return_26w.notna()&proxy.lt(cutoff)
        purges.append(dict(cutoff=cutoff,admitted=int(admitted.sum()),not_yet_mature=int((admitted&(exact.ge(cutoff)|exact.isna())).sum())))
    pd.DataFrame(purges).to_csv(OUT/'long_purge_audit.csv',index=False)
    d.assign(actual_maturity=exact,proxy_maturity=proxy).loc[d.return_26w.notna()&exact.gt(proxy),['ts_code','date','entry_date','actual_maturity','proxy_maturity']].to_csv(OUT/'long_maturity_differences.csv',index=False)
    d.groupby(d.date.dt.year).agg(rows=('date','size'),dates=('date','nunique'),stocks=('ts_code','nunique'),known26=('return_26w',lambda x:x.notna().mean())).to_csv(OUT/'long_sample_audit.csv')
    cols=[c for c in d.select_dtypes('number') if c not in IDENTIFIER_COLUMNS and not c.startswith(LABEL_PREFIXES)]
    long_feature_count=len(cols)
    coverage=[]
    for year,g in d.groupby(d.date.dt.year):
        for c in cols:
            x=g[c].to_numpy(float)
            coverage.append(dict(year=year,factor=c,coverage=float(np.isfinite(x).mean()),infinite=int(np.isinf(x).sum()),unique=int(g[c].nunique())))
    pd.DataFrame(coverage).to_csv(OUT/'long_feature_coverage.csv',index=False)
    financial=[]
    for name in ['fina_indicator','income','cashflow','balancesheet']:
        path=ROOT/f'data/raw/{name}.parquet'
        if not path.exists():continue
        names=pq.ParquetFile(path).schema.names
        cols=[c for c in ['ts_code','end_date','ann_date','f_ann_date','report_type'] if c in names]
        f=pd.read_parquet(path,columns=cols)
        financial.append(dict(source=name,rows=len(f),symbols=f.ts_code.nunique(),latest_period=str(f.end_date.max()),has_actual_announcement='f_ann_date' in f,
                              same_symbol_period_duplicates=int(f.duplicated(['ts_code','end_date']).sum())))
    pd.DataFrame(financial).to_csv(OUT/'financial_source_audit.csv',index=False)
    labels=pd.read_parquet(ROOT/'data/research/left_side_disclosure_gain_20261029/labels.parquet')
    cal=pd.to_datetime(pd.read_csv(ROOT/'data/research/left_side_disclosure_gain_20261029/calendar.csv').date).to_numpy(dtype='datetime64[ns]')
    ix=np.searchsorted(cal,labels.date.to_numpy(dtype='datetime64[ns]'))+6
    planned=pd.Series(pd.to_datetime([cal[i] if i<len(cal) else pd.NaT for i in ix]))
    delays=(labels.label_end_date-planned).dt.days
    summary={'left_delayed_exits':int(delays.gt(0).sum()),'left_max_delay_days':int(delays.max()),
             'left_wrong_calendar_purge_crossings':{c:int((planned.lt(c)&labels.label_end_date.ge(c)).sum()) for c in ['2023-01-01','2024-01-01','2025-01-01','2026-01-01']},
             'long_is_static_current_good_stock_pool':False,
             'long_universe_evidence':'tea_master_long.prepare_data uses date-specific market-cap filter; weekly quality carries backward monthly scores; source vintage/industry snapshots remain unresolved',
             'long_price_issue':'long_dividend_quality.load_daily_monthly_features uses raw close pct_change; attach_forward_returns uses raw close ratio without corporate-action normalization',
             'long_financial_issue':'load_financial_asof uses ann_date <= signal_date without announcement time, revision vintage or first-report selection',
             'long_feature_count':long_feature_count}
    (OUT/'training_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
