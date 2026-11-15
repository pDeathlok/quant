"""Extend the last frozen entry using unchanged Q4 entry and continuation models.

This is a targeted audit, not a replacement of the full historical evaluation.
All new inputs stay in workspace_live; frozen research files are never rewritten.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from build_index_futures_signal import LIVE, SOURCES, feature_panel, fit_forecasts
from backtest_index_futures import ROOT
from compare_index_futures_adaptation import groups_for, trade_episodes
from evaluate_index_futures_exit_value import (
    build_states, training_before, state_weights, TARGETS, STATE_FEATURES, PRODUCT_FEATURES, exit_callback,
)
from index_futures_state_models import pipeline
from evaluate_index_futures_panic import PanicReplay
from evaluate_index_futures_risk_paths import risk_replay


def main():
    out=LIVE/'last_trade_audit';out.mkdir(parents=True,exist_ok=True)
    original=pd.read_parquet(ROOT/'adaptation_comparison/quarterly_predictions.parquet')
    trades=pd.read_csv(ROOT/'panic_factors/trades.csv',dtype={'start':str,'entry':str,'exit':str})
    scenarios=[]
    for start in ('20220101','20250101','20260101'):
        for mode in ('fixed_qvix','rolling_qvix'):
            for fee,margin in ((.0006,.25),(.0003,.15)):
                part=trades.loc[trades.start.eq(start)&trades.startup_mode.eq(mode)&trades.fee.eq(fee)&trades.margin.eq(margin)]
                last=part.sort_values('entry').iloc[-1]
                trace=pd.read_parquet(ROOT/f'panic_factors/{start[:4]}/traces/{start}_mean5_{mode}_{fee}_{margin}.parquet')
                opened=trace.loc[trace.date.eq(last.entry)].iloc[0]
                assert opened.action=='enter' and not opened.held
                scenarios.append(dict(start=start,mode=mode,fee=fee,margin=margin,entry=last.entry,
                    entry_equity=float(opened.equity),policy=opened.policy))
    anchor=min(s['entry'] for s in scenarios)
    assert len({s['entry'] for s in scenarios})==1
    fresh=feature_panel()
    latest=fresh.groupby('symbol').trade_date.max().min()
    quarter=str(pd.Timestamp(latest).to_period('Q'))
    extra=fresh.loc[fresh.trade_date.gt(original.trade_date.max())&fresh.exit_date.notna()&fresh.exit_date.le(latest)].copy()
    with threadpool_limits(limits=1):
        forecasts,entry_audit=fit_forecasts(extra,quarter)
    extra=forecasts.loc[forecasts.direction.eq(1)&forecasts.bps.eq(6)].copy()
    extra['quarter']=pd.to_datetime(extra.entry_date).dt.to_period('Q').astype(str)
    extra['flat']=0.
    for product in ('IF','IH','IC','IM'):
        extra[f'long_{product}']=extra.symbol.eq(product).astype(float)
        extra[f'short_{product}']=-extra.symbol.eq(product).astype(float)
    panel=pd.concat([original.loc[original.entry_date.ge(anchor)],extra],ignore_index=True).sort_values(['entry_date','symbol'])
    assert not panel.duplicated(['entry_date','symbol']).any()
    bars=pd.concat(pd.read_parquet(SOURCES/f'{s}_daily.parquet') for s in ('IF','IH','IC','IM')).set_index(['trade_date','ts_code']).sort_index()
    market=json.loads((ROOT/'relative_indices/feature_sets.json').read_text())['combined']
    states=build_states(panel,bars,market)
    historical=pd.read_parquet(ROOT/'exit_value/states.parquet')
    old_predictions=pd.read_parquet(ROOT/'exit_value/forecasts.parquet')
    cutoff=original.loc[original.quarter.eq(quarter),'trade_date'].min()
    features=market+STATE_FEATURES+PRODUCT_FEATURES
    errors=[]
    with threadpool_limits(limits=1):
        for side in (1,-1):
            if not states.direction.eq(side).any():
                continue
            train=training_before(historical,cutoff,side)
            weight=state_weights(train,cutoff)
            model=pipeline(Ridge(alpha=.1*len(train)))
            model.fit(train[features],train[TARGETS],standardscaler__sample_weight=weight,ridge__sample_weight=weight)
            test=states.loc[states.direction.eq(side)]
            if len(test):
                raw=model.predict(test[features])
                for i,name in enumerate(TARGETS):
                    if name.startswith('risk'):raw[:,i]=np.maximum(0,raw[:,i])
                    states.loc[test.index,f'pred_{name}']=raw[:,i]
            check=old_predictions.loc[old_predictions.quarter.eq(quarter)&old_predictions.direction.eq(side)].tail(100)
            known=model.predict(check[features])
            for i,name in enumerate(TARGETS):
                if name.startswith('risk'):known[:,i]=np.maximum(0,known[:,i])
            errors.append(float(abs(known-check[[f'pred_{t}' for t in TARGETS]].to_numpy()).max()))
    assert max(errors)<1e-8
    # At the frozen overlap, feature construction and continuation predictions agree.
    overlap=states.merge(old_predictions,on=['entry','date','symbol'],suffixes=('','_old'))
    overlap_error=max(float(abs(overlap[f'pred_{t}']-overlap[f'pred_{t}_old']).max()) for t in TARGETS)
    assert overlap_error<1e-8
    replay=PanicReplay()
    evidence=dict(replay.evidence_lookup)
    for r in forecasts.itertuples(index=False):
        evidence[r.entry_date,r.symbol,r.direction,r.bps,'qvix']=(r.probability,r.pred_net,r.pred_risk)
    lookup={(r.entry,r.date,r.symbol):r for r in states.itertuples(index=False)}
    regimes=dict(replay.market_regime)
    regimes.update({(r.trade_date,r.symbol):int(r.state) for r in fresh.itertuples(index=False)})
    calendar=pd.read_parquet(SOURCES/'calendar.parquet').cal_date.sort_values()
    anchor_calendar=pd.read_csv(ROOT/'state_model_policies/global_calendar.csv',dtype={'date':str}).date.min()
    all_dates=calendar.loc[calendar.ge(anchor_calendar)].tolist()
    entry_calendar=set(all_dates[::5])
    opens,closes=bars.open.to_dict(),bars.close.to_dict()
    reports=[]
    for case in scenarios:
        fee,margin=case['fee'],case['margin']
        ledger,_,trace=risk_replay(groups_for(panel),opens,closes,fee,margin,capital=case['entry_equity'],
            policy=case['policy'],variant='mean5',trace=True,exit_decider=exit_callback(lookup,'mean5',fee),
            entry_calendar=entry_calendar,market_regime=regimes,evidence_model=4,evidence_lookup=evidence,
            soft_exit=4,evidence_names=replay.evidence_names)
        episodes=trade_episodes(ledger,opens,fee)
        name=f"{case['start']}_{case['mode']}_{fee}_{margin}"
        ledger.to_csv(out/f'{name}_ledger.csv',index=False)
        trace.to_csv(out/f'{name}_trace.csv',index=False)
        episodes.to_csv(out/f'{name}_trades.csv',index=False)
        records=episodes.to_dict('records')
        for record in records:
            record['entry_price']=float(opens[record['entry'],record['code']])
            record['exit_price']=float(opens[record['exit'],record['code']])
            record['exit_reason']=str(trace.loc[trace.date.eq(record['exit']),'action'].iloc[0]) if record['exit'] in trace.date.values else 'terminal_mark'
        reports.append(dict(**case,trades=records,final_positions=str(ledger.iloc[-1].positions),
            incremental_net=float(ledger.pnl.sum()),end_equity=float(ledger.iloc[-1].equity)))
    states.to_parquet(out/'continuation_forecasts.parquet',index=False)
    forecasts.to_parquet(out/'entry_forecasts.parquet',index=False)
    report=dict(asof=latest,ledger_end=panel.exit_date.max(),frozen_end=original.exit_date.max(),entry_model=entry_audit,
        exit_model_max_error=max(errors),overlap_max_error=overlap_error,cases=reports)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
