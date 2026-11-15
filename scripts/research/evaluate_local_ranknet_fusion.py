"""Purged chronological RankNet fusion with score-neighbourhood ablations.

Historical diagnostics only. This deliberately corrects the earlier experiments'
entry-day new-factor join and future-conditioned left candidate population.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/research')]
from quant.features.joinquant_candidates import PRICE, price_factors
from quant.research.local_ranknet import percentile_scores, make_pairs, derivatives, mature_before, choose_blend
from quant.research.auxiliary_return_loss import daily_picks
from evaluate_auxiliary_return_losses import intervals
from validate_left_side_disclosure_gain import sha256

OUT = ROOT/'reports/research/local_ranknet_fusion_v1'
KEYS = ['symbol', 'date']
PARAMS = dict(tree_method='hist', max_depth=3, min_child_weight=100,
              learning_rate=.06, reg_lambda=30, seed=73, nthread=4)
ROUNDS = 60
BANDS = {'day': None, 'local05': .05, 'local10': .10, 'local20': .20}
ALPHAS = (.1, .25, .5, .75, 1.)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def new_price_features(keys, cache_name='right_screen_price', start='202201'):
    keys=keys.copy()
    keys['date']=pd.to_datetime(keys.date).astype('datetime64[ns]')
    cache = OUT/f'{cache_name}.parquet'
    if cache.exists():
        value = pd.read_parquet(cache)
        pd.testing.assert_frame_equal(value[KEYS].sort_values(KEYS).reset_index(drop=True),
                                      keys.sort_values(KEYS).reset_index(drop=True), check_dtype=False)
        return value
    print('Building price factors', cache_name, flush=True)
    paths = sorted(p for p in (ROOT/'data/raw/daily_partitioned').glob('*/*.parquet')
                   if p.parent.name.split('=')[-1] >= start)
    bars = pl.concat([pl.read_parquet(p, columns=['ts_code','trade_date','open','high','low','close','pre_close','pct_chg','vol']) for p in paths], how='vertical_relaxed').unique(['ts_code','trade_date']).rename({'ts_code':'symbol'})
    basics = pl.concat([pl.read_parquet(p, columns=['ts_code','trade_date','turnover_rate'])
                        for p in sorted((ROOT/'data/raw/daily_basic').glob('*.parquet')) if p.stem >= start+'01'], how='vertical_relaxed').unique(['ts_code','trade_date']).rename({'ts_code':'symbol'})
    bars = bars.join(basics, on=['symbol','trade_date'], how='left').with_columns(
        pl.col('trade_date').cast(pl.String).str.strptime(pl.Date,'%Y%m%d').cast(pl.Datetime('ns')).alias('date')).sort(['symbol','date'])
    bars = bars.with_columns(((pl.col('close')/pl.col('pre_close')).cum_prod()*pl.col('pre_close').first()).over('symbol').alias('_continuous'), (pl.col('pct_chg')/100).alias('ret'))
    bars = bars.with_columns((pl.col('_continuous')/pl.col('close')).alias('_scale'))
    bars = bars.with_columns(*[(pl.col(c)*pl.col('_scale')).alias(c) for c in ['open','high','low','close']], (pl.col('vol')/pl.col('_scale')).alias('volume'))
    factors = price_factors(bars).join(pl.from_pandas(keys), on=KEYS, how='inner').to_pandas()
    value = keys.merge(factors, on=KEYS, how='left', validate='one_to_one')
    value.to_parquet(cache, index=False)
    return value


def datasets():
    market = [c for c in pq.ParquetFile(ROOT/'reports/research/overnight_unfiltered_market_factors_v1/by_symbol/000001.SZ.parquet').schema.names if c not in KEYS+['market_row_present']]
    data = ROOT/'data/research/left_side_disclosure_gain_20261029'
    left = pd.read_parquet(data/'events.parquet', columns=KEYS+market)
    labels = pd.read_parquet(data/'labels.parquet')
    left = left.merge(labels[KEYS+['terminal_return','locked_limit_up','maturity_reason','label_end_date']], on=KEYS, validate='one_to_one')
    left['outcome'] = left.terminal_return-.002
    # Selection happens before next-day execution. Known unfilled attempts stay
    # in top20 with zero return (cash); missing market data remain unknown.
    left.loc[left.locked_limit_up, 'outcome'] = 0.
    cal = pd.to_datetime(pd.read_csv(data/'calendar.csv').date).sort_values().to_numpy()
    pos = np.searchsorted(cal, left.date.to_numpy()) + 6
    left['planned_maturity'] = pd.to_datetime([cal[i] if i < len(cal) else pd.NaT for i in pos])
    left['maturity'] = pd.to_datetime(left.label_end_date).fillna(left.planned_maturity)
    left['source_date'] = left.date
    jq = pd.read_parquet(ROOT/'reports/research/joinquant_candidates_v1/features.parquet', columns=KEYS+list(PRICE))
    # The legacy cache keys came from a future-filtered cohort. Reconstruct all
    # absent keys from raw bars so cache availability cannot reveal future fills.
    absent = left[KEYS].merge(jq[KEYS], on=KEYS, how='left', indicator=True)
    absent = absent.loc[absent._merge.eq('left_only'),KEYS].reset_index(drop=True)
    if len(absent):
        supplement = new_price_features(absent, 'left_missing_price', '201811')
        jq = pd.concat([jq,supplement],ignore_index=True)
    left = left.merge(jq, on=KEYS, how='left', validate='one_to_one')
    root = ROOT/'reports/research/right_t012_full_dataset_v4'
    right = pd.read_parquet(root/'all_candidates.parquet')
    right = right.loc[right.date.notna() & right.planned_exit_date.notna()].copy()
    labels_r = pd.read_parquet(root/'dataset.parquet', columns=KEYS+['net_close_net_after_exit_audit'])
    right = right.merge(labels_r, on=KEYS, how='left', validate='one_to_one')
    right['outcome'] = right.net_close_net_after_exit_audit
    right.loc[right.at_upper.eq(True), 'outcome'] = 0.
    right['maturity'] = right.planned_exit_date
    right['planned_maturity'] = right.planned_exit_date
    right['source_date'] = pd.to_datetime(right.screen_date).astype('datetime64[ns]')
    assert right.source_date.lt(right.date).all()
    factors = pd.read_parquet(ROOT/'reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet', columns=KEYS+market).rename(columns={'date':'source_date'})
    right = right.merge(factors, on=['symbol','source_date'], how='left', validate='one_to_one')
    screen_keys = right[['symbol','source_date']].rename(columns={'source_date':'date'}).drop_duplicates().reset_index(drop=True)
    p = new_price_features(screen_keys).rename(columns={'date':'source_date'})
    right = right.merge(p, on=['symbol','source_date'], how='left', validate='one_to_one')
    audit = {'left_original_events': len(left), 'left_prior_kept': int(labels.mature.sum()),
             'left_exclusion_reasons': labels.maturity_reason.value_counts().to_dict(),
             'right_screen_lag_violations': int(right.source_date.ge(right.date).sum()),
             'right_all_attempts': len(right), 'right_unfilled_cash': int(right.at_upper.eq(True).sum()),
             'right_unknown_outcomes': int(right.outcome.isna().sum()),
             'excluded_features': 'All statement/valuation/ownership variables, current entry-day bars, screen_to_entry_return; only market-derived and 19 JQ price/turnover candidates.',
             'left_jq_missing_rows': int(left[list(PRICE)].isna().all(axis=1).sum())}
    audit['left_delayed_exit_rows']=int(left.maturity.gt(left.planned_maturity).sum())
    audit['left_entry_mode']='next_close; horizon=5 stock-sellable sessions; actual end date used for training purge'
    save_json(OUT/'sample_audit.json', audit)
    for side, frame in [('left',left),('right',right)]:
        for c in ['date','source_date','maturity','planned_maturity']:
            frame[c] = pd.to_datetime(frame[c]).astype('datetime64[ns]')
        frame = frame.sort_values(['date','symbol']).reset_index(drop=True)
        assert not frame.duplicated(KEYS).any()
        yield side, frame, market+list(PRICE)


def matrix(frame, cols):
    return xgb.DMatrix(frame[cols].astype('float32').replace([np.inf,-np.inf],np.nan), nthread=4)


def evaluation_slice(frame,start,end):
    # Planned calendar is known ex ante. Never discard a candidate because its
    # future suspension happens to push the actual exit past this boundary.
    out=frame.loc[frame.date.ge(start)&frame.date.lt(end)&frame.planned_maturity.lt(end)].copy()
    out.loc[out.maturity.ge(end),'outcome']=np.nan
    return out


def base_model(frame, cols):
    d = matrix(frame, cols)
    d.set_label(frame.outcome.clip(-.5,.5))
    w = (1/frame.groupby('date').date.transform('size')).to_numpy(copy=True)
    d.set_weight(w/w.mean())
    return xgb.train({**PARAMS, 'objective':'reg:squarederror'}, d, num_boost_round=ROUNDS)


def distribution_audit(tr, va, te, columns, side, fold):
    records=[]
    for c in columns:
        fit = tr[c].replace([np.inf,-np.inf],np.nan).dropna()
        edges = np.unique(fit.quantile(np.linspace(0,1,11)).to_numpy())
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        def probabilities(values):
            valid=values[np.isfinite(values)]
            hist=np.r_[np.histogram(valid,edges)[0],len(values)-len(valid)].astype(float)+.5
            return hist/hist.sum()
        p = probabilities(tr[c].to_numpy(float))
        for stage, d in [('fit',tr),('validation',va),('test',te)]:
            values=d[c].to_numpy(float); q=probabilities(values)
            finite=values[np.isfinite(values)]
            records.append(dict(side=side,fold=fold,stage=stage,factor=c,
                                coverage=float(np.isfinite(values).mean()),infinite=int(np.isinf(values).sum()),
                                psi=float(np.sum((q-p)*np.log(q/p))),
                                p01=float(np.quantile(finite,.01)) if len(finite) else None,
                                p50=float(np.median(finite)) if len(finite) else None,
                                p99=float(np.quantile(finite,.99)) if len(finite) else None))
    pd.DataFrame(records).to_csv(OUT/f'distribution_{side}_{fold}.csv', index=False)
    sample = tr[columns].iloc[::max(1,len(tr)//15000)].corr(method='spearman')
    duplicate = [(a,b,float(sample.loc[a,b])) for i,a in enumerate(columns) for b in columns[i+1:]
                 if abs(sample.loc[a,b]) > .995]
    save_json(OUT/f'redundancy_{side}_{fold}.json', duplicate)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'complete.json').unlink(missing_ok=True)
    inputs = [ROOT/'data/research/left_side_disclosure_gain_20261029'/f for f in ['events.parquet','labels.parquet','calendar.csv']]
    inputs += [ROOT/p for p in ['reports/research/right_t012_full_dataset_v4/all_candidates.parquet',
              'reports/research/right_t012_full_dataset_v4/dataset.parquet',
              'reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet',
              'reports/research/joinquant_candidates_v1/features.parquet']]
    save_json(OUT/'protocol.json', dict(params=PARAMS, rounds=ROUNDS, bands=BANDS, alphas=ALPHAS,
        base='50 factors selected by gain ONLY in initial warmup period, fixed before all OOF predictions',
        secondary='Exact pairwise logistic RankNet loss on trees; remaining/all audit-admitted factors; equal total loss per date; non-neural adaptation',
        score_distance='within-day percentile of chronological OOF base scores, not calibrated probability; exact overlapping neighbourhood',
        selection='validation-only maximize stressed net subject to win >= base50; base50 fallback; no test tuning',
        execution='top20 signal attempts; known unfilled=0; unknown remains selected and stressed -10%; right all inputs from screen day before entry day',
        limits=['Reused historical diagnostic periods, not prospective holdout','Source revisions/universe survivorship not fully certified','Daily top20 trade means, NOT portfolio NAV','Long cohort audited separately, no fusion promotion'],
        input_hashes={str(p.relative_to(ROOT)):sha256(p) for p in inputs},
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/quant/research/local_ranknet.py']}))
    results=[]; grid_rows=[]; splits=[]
    for side, frame, candidates in datasets():
        folds = ({'2024':('2023-01-01','2024-01-01','2025-01-01'),
                  '2025':('2024-01-01','2025-01-01','2026-01-01'),
                  '2026':('2025-01-01','2026-01-01','2026-11-11')} if side=='left' else
                 {'2025H2':('2025-01-01','2025-07-01','2026-01-01'),
                  '2026':('2025-07-01','2026-01-01','2026-09-04')})
        for fold, (fit_end,test_start,test_end) in folds.items():
            print('START',side,fold,flush=True)
            tr = frame.loc[mature_before(frame,fit_end)].copy()
            va = evaluation_slice(frame,fit_end,test_start)
            te = evaluation_slice(frame,test_start,test_end)
            warm_end = '2021-01-01' if side=='left' else '2024-04-01'
            warm = tr.loc[mature_before(tr,warm_end)]
            cols = [c for c in candidates if warm[c].notna().mean()>=.5 and warm[c].nunique()>1]
            selector = base_model(warm,cols)
            gain = selector.get_score(importance_type='gain')
            first = sorted(cols,key=lambda c:(-gain.get(c,0),c))[:50]
            remaining = [c for c in cols if c not in first]
            assert len(first)==50 and len(remaining)>0
            save_json(OUT/f'features_{side}_{fold}.json', {'warm_end':warm_end,'base50':first,'remaining':remaining,'excluded':[c for c in candidates if c not in cols]})
            oof = pd.Series(np.nan,index=tr.index)
            boundaries = list(pd.date_range(warm_end,fit_end,freq='6MS' if side=='left' else '3MS'))
            if boundaries[-1] != pd.Timestamp(fit_end): boundaries.append(pd.Timestamp(fit_end))
            for start,end in zip(boundaries[:-1],boundaries[1:]):
                past = tr.loc[mature_before(tr,start)]
                future = tr.loc[tr.date.ge(start)&tr.date.lt(end)]
                if future.empty: continue
                assert past.maturity.max()<future.date.min()
                model=base_model(past,first)
                oof.loc[future.index]=model.predict(matrix(future,first))
                splits.append(dict(side=side,fold=fold,train_end=str(past.date.max()),label_end=str(past.maturity.max()),prediction_start=str(future.date.min()),prediction_end=str(future.date.max()),rows=len(future)))
            sub = tr.loc[oof.notna()].copy().reset_index(drop=True)
            near = percentile_scores(sub.date,oof.dropna().to_numpy())
            sub[KEYS+['maturity']].assign(oof_score=oof.dropna().to_numpy(),oof_percentile=near).to_parquet(OUT/f'oof_{side}_{fold}.parquet',index=False)
            model = base_model(tr,first)
            model.save_model(OUT/f'model_{side}_{fold}_base50.ubj')
            vs = percentile_scores(va.date, model.predict(matrix(va,first)))
            ts = percentile_scores(te.date, model.predict(matrix(te,first)))
            vdays={'base50':daily_picks(va,vs)[1]}; tdays={'base50':daily_picks(te,ts)[1]}
            score_map={'base50':ts}; fixed={}
            # Equal-feature single model is a control for the benefit of mere expansion.
            all_model=base_model(tr,cols)
            av=percentile_scores(va.date,all_model.predict(matrix(va,cols)))
            at=percentile_scores(te.date,all_model.predict(matrix(te,cols)))
            vdays['single_all']=daily_picks(va,av)[1];tdays['single_all']=daily_picks(te,at)[1];score_map['single_all']=at
            all_model.save_model(OUT/f'model_{side}_{fold}_single_all.ubj')
            for group,band in BANDS.items():
                pairs=make_pairs(sub.date,near,sub.outcome,bandwidth=band)
                for scope,features in [('remaining',remaining),('all',cols)]:
                    name=f'{group}_{scope}'
                    print(side,fold,name,'rows',len(sub),'pairs',len(pairs.left),flush=True)
                    d=matrix(sub,features);d.set_label(sub.outcome)
                    ranker=xgb.train({**PARAMS,'base_score':0.,'disable_default_eval_metric':1},d,num_boost_round=ROUNDS,obj=lambda pred,dm:derivatives(pred,pairs))
                    ranker.save_model(OUT/f'model_{side}_{fold}_{name}.ubj')
                    rv=percentile_scores(va.date,ranker.predict(matrix(va,features),output_margin=True))
                    rt=percentile_scores(te.date,ranker.predict(matrix(te,features),output_margin=True))
                    for alpha in ALPHAS:
                        arm=f'{name}@{alpha}'
                        vdays[arm]=daily_picks(va,(1-alpha)*vs+alpha*rv)[1]
                        score_map[arm]=(1-alpha)*ts+alpha*rt
                        tdays[arm]=daily_picks(te,score_map[arm])[1]
                    fixed[name]=choose_blend({k:v for k,v in vdays.items() if k=='base50' or k.startswith(name+'@')})
            # Separate model-family choices are frozen by validation before test reporting.
            policies={'selected':choose_blend({k:v for k,v in vdays.items() if k!='single_all'}),
                      'day_only':choose_blend({k:v for k,v in vdays.items() if k=='base50' or k.startswith('day_')}),
                      'local_only':choose_blend({k:v for k,v in vdays.items() if k=='base50' or k.startswith('local')}),
                      'single_all':'single_all', **{f'fixed:{k}':v for k,v in fixed.items()}}
            save_json(OUT/f'choices_{side}_{fold}.json', policies)
            for arm,d in vdays.items():
                grid_rows.append(dict(side=side,fold=fold,arm=arm,validation_net=d.stress.mean(),validation_win=d.stress_win.mean(),test_net=tdays[arm].stress.mean(),test_win=tdays[arm].stress_win.mean()))
            for policy,arm in policies.items():
                a=tdays[arm];b=tdays['base50'];delta=(a.stress-b.stress).to_numpy();ci=intervals(delta)
                results.append(dict(side=side,fold=fold,policy=policy,arm=arm,features=len(cols),remaining=len(remaining),fit_rows=len(tr),sub_rows=len(sub),test_rows=len(te),dates=len(a),
                    baseline_net=b.stress.mean(),candidate_net=a.stress.mean(),net_delta_bps=delta.mean()*10000,
                    baseline_win=b.stress_win.mean(),candidate_win=a.stress_win.mean(),win_delta_pp=(a.stress_win-b.stress_win).mean()*100,
                    net_ci_low_bps=ci[0]*10000,net_ci_high_bps=ci[1]*10000,unknown_picks=int((a['count']-a['known']).sum())))
                a.to_csv(OUT/f'daily_{side}_{fold}_{policy.replace(":","_")}.csv')
                if policy in ('selected','day_only','local_only'):
                    te[KEYS+['source_date','outcome']].assign(score=score_map[arm],baseline_score=ts).to_parquet(OUT/f'predictions_{side}_{fold}_{policy}.parquet',index=False)
            distribution_audit(tr,va,te,cols,side,fold)
            pd.DataFrame(results).to_csv(OUT/'comparisons.csv',index=False)
            pd.DataFrame(grid_rows).to_csv(OUT/'validation_grid.csv',index=False)
            pd.DataFrame(splits).to_csv(OUT/'oof_splits.csv',index=False)
    save_json(OUT/'complete.json', {'status':'complete','folds':5,'production_promoted':False})


if __name__=='__main__':
    main()
