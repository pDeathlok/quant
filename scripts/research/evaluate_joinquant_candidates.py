"""Reproducible exploratory additions on frozen left/right/long research cohorts."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'scripts/research')]
from quant.features.joinquant_candidates import PRICE,FINANCE,VERSION,price_factors,financial_events,align_events
from quant.features.long_weekly_factors import IDENTIFIER_COLUMNS,LABEL_PREFIXES
from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from scripts.research.validate_left_side_disclosure_gain import load_frame,sha256
from scripts.research.train_right_t012_full_registry import FOLDS,split_masks,SIGNALS
OUT=ROOT/'reports/research/joinquant_candidates_v1'
RAW=ROOT/'data/raw'
KEYS=['symbol','date']


def cohorts() -> dict[str,tuple[pd.DataFrame,list[str]]]:
    left=load_frame(); left['outcome']=left.terminal_return-.002;left['maturity']=left.label_end_date
    right=pd.read_parquet(ROOT/'reports/research/right_t012_full_dataset_v4/dataset.parquet',columns=list(dict.fromkeys(KEYS+['planned_exit_date','net_close_net_after_exit_audit',*PROJECT_FACTOR_COLUMNS,*SIGNALS,'episode_age','screen_to_entry_return'])))
    right['outcome']=right.net_close_net_after_exit_audit;right['maturity']=right.planned_exit_date
    long=pd.read_parquet(ROOT/'data/features/long_entry/weekly_training_v2.parquet').rename(columns={'ts_code':'symbol'})
    # Existing 26-week utility label, not an executable portfolio return.
    long=long.loc[pd.to_datetime(long.date).ge('2020-01-01')].copy()
    long['outcome']=long.label_entry_utility_26w;long['maturity']=pd.to_datetime(long.date)+pd.Timedelta(weeks=27)
    base=[c for c in long.select_dtypes('number') if c not in IDENTIFIER_COLUMNS and not c.startswith(LABEL_PREFIXES) and c != 'outcome']
    result={'left':(left,list(LEFT_SIDE_FACTOR_COLUMNS)), 'right':(right,list(dict.fromkeys([*PROJECT_FACTOR_COLUMNS,*SIGNALS,'episode_age','screen_to_entry_return']))),'long':(long,base)}
    for d,_ in result.values():
        d['symbol']=d.symbol.astype(str);d['date']=pd.to_datetime(d.date).astype('datetime64[ns]');d['maturity']=pd.to_datetime(d.maturity).astype('datetime64[ns]')
    return result


def build(keys: pd.DataFrame) -> pd.DataFrame:
    cache=OUT/'features.parquet'
    if cache.exists() and "--rebuild" not in sys.argv:
        cached = pd.read_parquet(cache)
        expected = keys.sort_values(KEYS).reset_index(drop=True)
        actual = cached[KEYS].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False)
        contract_path = OUT / 'contract.json'
        if not contract_path.exists():
            raise ValueError('Feature cache has no provenance contract; use --rebuild')
        contract = json.loads(contract_path.read_text())
        if contract['version'] != VERSION:
            raise ValueError('Feature cache version changed; use --rebuild')
        for name, digest in contract['input_hashes'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Input changed: {name}; use --rebuild')
        return cached
    start=(keys.date.min()-pd.Timedelta(days=420)).strftime('%Y%m')
    paths=sorted(p for p in RAW.glob('daily_partitioned/year_month=*/*.parquet') if p.parent.name.split('=')[1]>=start)
    pieces=[]
    for p in paths:
        pieces.append(pl.read_parquet(p,columns=['ts_code','trade_date','open','high','low','close','pre_close','pct_chg','vol']))
    bars=pl.concat(pieces,how='vertical_relaxed').unique(['ts_code','trade_date']).rename({'ts_code':'symbol'})
    bars=bars.with_columns(pl.col('trade_date').cast(pl.String).str.strptime(pl.Date,'%Y%m%d').cast(pl.Datetime('ns')).alias('date'))
    basic_paths=[p for p in sorted((RAW/'daily_basic').glob('*.parquet')) if p.stem[:6]>=start]
    basics=pl.concat([pl.read_parquet(p,columns=['ts_code','trade_date','turnover_rate','total_mv']) for p in basic_paths],how='vertical_relaxed').rename({'ts_code':'symbol'}).unique(['symbol','trade_date'])
    bars=bars.join(basics,on=['symbol','trade_date'],how='left').sort(['symbol','date'])
    # Reconstruct continuous prices from daily pre_close; inverse-adjust shares.
    bars=bars.with_columns(((pl.col('close')/pl.col('pre_close')).cum_prod()*pl.col('pre_close').first()).over('symbol').alias('_continuous'),(pl.col('pct_chg')/100).alias('ret'))
    bars=bars.with_columns((pl.col('_continuous')/pl.col('close')).alias('_scale'))
    bars=bars.with_columns(*[(pl.col(c)*pl.col('_scale')).alias(c) for c in ['open','high','low','close']],(pl.col('vol')/pl.col('_scale')).alias('volume'))
    k=pl.from_pandas(keys).with_columns(pl.col('date').cast(pl.Datetime('ns')))
    print('price factors',bars.height,flush=True)
    market=price_factors(bars).join(k,on=KEYS,how='inner').to_pandas()
    mv=bars.select(*KEYS,'total_mv').join(k,on=KEYS,how='inner').to_pandas()
    del bars,basics,pieces
    out=keys.merge(market,on=KEYS,how='left',validate='one_to_one').merge(mv,on=KEYS,how='left',validate='one_to_one')
    for name,fields,ttm in [('cashflow',['n_cashflow_act'],True),('income',['revenue','operate_profit'],True),('fina_indicator',['grossprofit_margin','ar_turn','inv_turn'],False)]:
        print('financial',name,flush=True)
        events=financial_events(pd.read_parquet(RAW/f'{name}.parquet'),fields,ttm=ttm)
        aligned=align_events(keys,events).drop(columns='available')
        out=out.merge(aligned,on=KEYS,how='left',validate='one_to_one')
    mv=out.total_mv.where(out.total_mv.gt(0))*10000
    out[FINANCE[0]]=out.n_cashflow_act/mv
    out[FINANCE[1]]=out.operate_profit/mv
    out[FINANCE[2]]=out.operate_profit/out.revenue.where(out.revenue.gt(0))
    out[FINANCE[3]]=out.revenue_growth
    out[FINANCE[4]]=out.operate_profit_growth
    out[FINANCE[5]]=out.grossprofit_margin_delta
    out[FINANCE[6]]=360/out.ar_turn.where(out.ar_turn.gt(0))+360/out.inv_turn.where(out.inv_turn.gt(0))
    out=out[KEYS+list(PRICE+FINANCE)].replace([np.inf,-np.inf],np.nan)
    out.to_parquet(cache,index=False)
    return out


def evaluate() -> None:
    if '--allow-legacy-timing' not in sys.argv:
        raise SystemExit('Legacy experiment quarantined: right entry-day factor timing and left future-conditioned sample selection require correction. Use evaluate_local_ranknet_fusion.py / evaluate_ranknet_cash_policy.py; --allow-legacy-timing is only for reproducing the superseded historical result.')
    OUT.mkdir(parents=True,exist_ok=True)
    sets=cohorts(); keys=pd.concat([d[KEYS] for d,_ in sets.values()]).drop_duplicates()
    keys=keys.sort_values(KEYS).reset_index(drop=True)
    features=build(keys)
    contract={'version':VERSION,'factors':{'price':PRICE,'finance':FINANCE},'features_sha256':sha256(OUT/'features.parquet'),
              'scope':'exploratory fixed-model retraining; existing historical test periods reused; not production model promotion',
              'params':{'max_iter':60,'max_leaf_nodes':15,'min_samples_leaf':150,'l2_regularization':30,'learning_rate':.06,'early_stopping':False,'random_state':73},
              'financial_visibility':'strictly after announcement; first publication only, provider vintage unverified',
              'price_semantics':'local split-continuous OHLCV; skew unbiased, kurtosis excess; not vendor parity-certified',
              'long_label':'existing 26-week utility, not net portfolio return; 27-week purge',
              'costs':'left terminal_return minus20bps; right audited net outcome; no new execution simulation',
              'input_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [RAW/'cashflow.parquet',RAW/'income.parquet',RAW/'fina_indicator.parquet']}}
    (OUT/'contract.json').write_text(json.dumps(contract,indent=2,ensure_ascii=False))
    metrics=[]; single=[]; comparisons=[]
    for side,(d,baseline) in sets.items():
        d=d.merge(features,on=KEYS,how='left',validate='one_to_one')
        d[list(PRICE+FINANCE)].notna().groupby(d.date.dt.year).mean().to_csv(OUT/f'coverage_{side}.csv')
        folds=FOLDS if side=='right' else {str(y):y for y in (2024,2025,2026)}
        for fold,bounds in folds.items():
            if side=='right':
                masks=split_masks(d,bounds);train=masks['fit'];test=masks['diagnostic']
            else:
                boundary=pd.Timestamp(int(bounds),1,1)
                # Leave previous calendar year for future tuning; all arms fixed now.
                train=d.date.lt(boundary-pd.DateOffset(years=1)) & d.maturity.lt(boundary-pd.DateOffset(years=1))
                test=d.date.ge(boundary)&d.date.lt(boundary+pd.DateOffset(years=1))
            train &= d.outcome.notna();test &= d.maturity.le(pd.Timestamp('2026-11-10'))
            if train.sum()<200 or test.sum()<200:continue
            # Pre-filter only using fit coverage/variation. All arms score identical test rows.
            added=[c for c in PRICE+FINANCE if d.loc[train,c].notna().mean()>=.5 and d.loc[train,c].nunique()>5]
            tr=d.loc[train];te=d.loc[test];daily={}
            for c in added:
                direction=tr[c].corr(tr.outcome,method='spearman');direction=1 if direction>=0 else -1
                ic=te.groupby('date').apply(lambda g:g[c].corr(g.outcome,method='spearman') if g[c].notna().sum()>=10 else np.nan,include_groups=False)
                single.append(dict(side=side,fold=fold,factor=c,fit_direction=direction,test_rank_ic=ic.mean(),signed_test_ic=ic.mean()*direction,coverage=te[c].notna().mean(),dates=ic.count()))
            for arm,cols in [('baseline',baseline),('price',baseline+[c for c in added if c in PRICE]),('finance',baseline+[c for c in added if c in FINANCE]),('all',baseline+added)]:
                columns=list(dict.fromkeys(c for c in cols if pd.api.types.is_numeric_dtype(d[c]) and tr[c].nunique()>1))
                print(side,fold,arm,len(tr),len(te),len(columns),flush=True)
                model=HistGradientBoostingRegressor(**contract['params'])
                weights=1/tr.groupby('date').date.transform('size');weights=weights/weights.mean()
                with threadpool_limits(limits=4):
                    model.fit(tr[columns].astype('float32').replace([np.inf,-np.inf],np.nan),(tr.outcome if side=='long' else tr.outcome.clip(-.5,.5)),sample_weight=weights)
                    score=model.predict(te[columns].astype('float32').replace([np.inf,-np.inf],np.nan))
                scored=te[KEYS+['outcome']].assign(score=score)
                picks=scored.sort_values(['date','score','symbol'],ascending=[True,False,True]).groupby('date').head(20)
                stats=picks.groupby('date').outcome.agg(['mean','count','size'])
                daily[arm]=stats['mean'].where(stats['count'].eq(stats['size']))
                ic=scored.groupby('date').apply(lambda g:g.score.corr(g.outcome,method='spearman') if len(g)>=10 else np.nan,include_groups=False).mean()
                metrics.append(dict(side=side,fold=fold,arm=arm,rows=len(te),features=len(columns),top20_mean_outcome=daily[arm].mean(),unknown_picked_outcomes=int(picks.outcome.isna().sum()),complete_dates=int(daily[arm].notna().sum()),rank_ic=ic))
                scored.to_parquet(OUT/f'predictions_{side}_{fold}_{arm}.parquet',index=False)
                pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
            for arm in ('price','finance','all'):
                delta=(daily[arm]-daily['baseline']).dropna().to_numpy()
                # 26-week weekly label overlap needs longer blocks for long cohort.
                rng=np.random.default_rng(73);length=27 if side=='long' else 10;n=len(delta)
                samples=[]
                for _ in range(1000 if n >= 2*length else 0):
                    ids=(rng.integers(0,n,size=(int(np.ceil(n/length)),1))+np.arange(length))%n
                    samples.append(delta[ids.ravel()[:n]].mean())
                comparisons.append(dict(side=side,fold=fold,arm=arm,dates=n,delta=delta.mean() if n else np.nan,ci_low=np.quantile(samples,.025) if samples else np.nan,ci_high=np.quantile(samples,.975) if samples else np.nan,insufficient_blocks=n<2*length))
            pd.DataFrame(comparisons).to_csv(OUT/'comparisons.csv',index=False)
            pd.DataFrame(single).to_csv(OUT/'single_factors.csv',index=False)
    (OUT/'complete.json').write_text(json.dumps({'status':'complete','metrics':len(metrics),'features':len(PRICE+FINANCE),'production_promoted':False}))

if __name__=='__main__':evaluate()
