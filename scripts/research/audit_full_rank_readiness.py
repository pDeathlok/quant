"""Repair whole-history readiness gates and rebuild chronological OOF scores."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'scripts/research')]
import evaluate_local_ranknet_fusion as prior
from build_full_rank_dataset import OUT


def readiness(keys, observations, rule):
    bars = observations.copy()
    bars['available'] = (bars.date.dt.to_period(rule).dt.end_time.dt.normalize()).astype('datetime64[ns]')
    bars = bars[['symbol','available']].drop_duplicates().sort_values(['symbol','available'])
    bars['observed_periods'] = bars.groupby('symbol').cumcount()+1
    keys = keys.assign(_row=np.arange(len(keys)),date=pd.to_datetime(keys.date).astype('datetime64[ns]'))
    joined = pd.merge_asof(keys.sort_values(['date','symbol']),bars.sort_values(['available','symbol']),
                           left_on='date',right_on='available',by='symbol',direction='backward')
    return joined.sort_values('_row').observed_periods.fillna(0).to_numpy()


def main():
    meta=json.loads((OUT/'dataset_complete.json').read_text())
    paths=[p for p in (ROOT/'data/raw/daily_partitioned').glob('*/*.parquet') if '201801'<=p.parent.name.split('=')[-1]<='202609']
    observations=pl.concat([pl.read_parquet(p,columns=['symbol','date']).with_columns(pl.col('date').cast(pl.Datetime('ns'))) for p in paths]).to_pandas()
    notes=[]
    for side in ['left','right']:
        frame=pd.read_parquet(OUT/f'dataset_{side}.parquet')
        keys=frame[['symbol','source_date']].rename(columns={'source_date':'date'})
        for rule,prefix in [('W-FRI','kdj_w_'),('M','kdj_m_')]:
            counts=readiness(keys,observations,rule)
            cols=[c for c in frame if c.startswith(prefix)]
            bad=counts<10
            notes.append(dict(side=side,rule=rule,masked_rows=int(bad.sum()),changed_cells=int(frame.loc[bad,cols].notna().sum().sum())))
            frame.loc[bad,cols]=np.nan
        frame.to_parquet(OUT/f'dataset_{side}.parquet',index=False)
        meta['dataset_hashes'][side]=prior.sha256(OUT/f'dataset_{side}.parquet')
        base=json.loads((prior.OUT/f'features_{side}_2026.json').read_text())['base50']
        outputs=[];splits=[]
        boundaries=list(pd.date_range('2024-04-01','2025-07-01',freq='QS'))
        for start,end in zip(boundaries[:-1],boundaries[1:]):
            train=frame.loc[frame.date.lt(start)&frame.maturity.lt(start)&frame.outcome.notna()]
            predict=frame.loc[frame.date.ge(start)&frame.date.lt(end)].copy()
            print('OOF',side,str(start.date()),len(train),len(predict),flush=True)
            model=prior.base_model(train,base)
            predict['oof_score']=model.predict(prior.matrix(predict,base))
            predict['oof_percentile']=prior.percentile_scores(predict.date,predict.oof_score)
            outputs.append(predict[prior.KEYS+['oof_score','oof_percentile']])
            model.save_model(OUT/f'oof_model_{side}_{start.date()}.ubj')
            splits.append(dict(side=side,start=str(start),end=str(end),rows=len(train),max_label_date=str(train.maturity.max())))
        pd.concat(outputs,ignore_index=True).to_parquet(OUT/f'oof_{side}.parquet',index=False)
        pd.DataFrame(splits).to_csv(OUT/f'oof_splits_{side}.csv',index=False)
    meta['readiness_repaired']=True
    prior.save_json(OUT/'readiness_audit.json',dict(reason='Whole-history len(resampled_bars)>=10 could unlock earlier IPO rows; now require10 completed observed periods as of signal date',rows=notes,
        oof='Rebuilt from2024Q2; each quarter trains solely on2024+ earlier dates with actual outcomes matured before quarter start'))
    audit=pd.read_csv(OUT/'column_audit.csv')
    mask=audit.name.str.startswith(('kdj_w_','kdj_m_'))
    audit.loc[mask,'evidence']='causal10_completed_period_mask_and_rebuilt_OOF'
    audit.to_csv(OUT/'column_audit.csv',index=False)
    prior.save_json(OUT/'dataset_complete.json',meta)


if __name__=='__main__':main()
