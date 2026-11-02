"""Point-in-time pilot: market mainlines, evidenced industrial leaders, B1.

Annual historical index membership is frozen for the subsequent calendar year.
This is an event study; it does not simulate production stop/expiry execution.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reports/research/b1_mainline_leaders_20261004'
START, END = pd.Timestamp('2025-05-06'), pd.Timestamp('2026-03-27')
COST = .002


def compute_mainlines(panel):
    """Causal daily signal. No full-sample parameters or future rolling bars."""
    parts=[]
    for code,f in panel.groupby('index_code',sort=True):
        f=f.sort_values('date').copy()
        f['equity']=(1+f.theme_return).cumprod()
        f['ret20']=f.equity.pct_change(20,fill_method=None)
        f['ret60']=f.equity.pct_change(60,fill_method=None)
        f['relative20']=f.ret20-f.market_ret20
        f['above_ma60']=f.equity.gt(f.equity.rolling(60).mean())
        f['activity']=f.amount_share.rolling(5).mean()/f.amount_share.shift(5).rolling(60).mean()
        parts.append(f)
    panel=pd.concat(parts,ignore_index=True)
    score=pd.Series(0.,index=panel.index)
    for col,weight in [('relative20',.45),('ret60',.25),('activity',.20),('breadth20',.10)]:
        score+=panel.groupby('date')[col].rank(method='average',pct=True)*weight
    panel['mainline_score']=score
    eligible=(panel.ret20.gt(0)&panel.relative20.gt(0)&panel.above_ma60&
              panel.breadth20.ge(.5)&panel.member_coverage.ge(.7))
    panel['raw_top3']=False
    top=panel.loc[eligible].sort_values(['date','mainline_score','index_code'],ascending=[True,False,True]).groupby('date').head(3)
    panel.loc[top.index,'raw_top3']=True
    panel=panel.sort_values(['index_code','date'])
    panel['persistence5']=panel.groupby('index_code').raw_top3.transform(lambda s:s.rolling(5,min_periods=5).sum())
    # Today's eligibility required; persistence alone cannot retain a broken trend.
    panel['mainline']=panel.raw_top3&panel.persistence5.ge(3)
    return panel.sort_values(['date','index_code']).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    members=pd.read_csv(OUT/'historical_members.csv')
    members['snapshot_date']=pd.to_datetime(members.snapshot_date)
    members['year']=members.snapshot_date.dt.year+1
    assert not members.duplicated(['year','index_code','symbol']).any()
    parts=[]
    cols=['symbol','date','pct_chg','amount','close','pre_close','volume']
    for p in sorted((ROOT/'data/raw/daily_partitioned').glob('year_month=*/data.parquet')):
        month=p.parent.name.split('=')[1]
        if '202401'<=month<='202604':parts.append(pd.read_parquet(p,columns=cols))
    daily=pd.concat(parts,ignore_index=True)
    daily['date']=pd.to_datetime(daily.date).dt.normalize()
    assert not daily.duplicated(['symbol','date']).any()
    print(f'Market bars {len(daily):,}',flush=True)
    market_amount=daily.groupby('date').amount.sum()
    daily=daily.loc[daily.symbol.isin(members.symbol)].sort_values(['symbol','date']).copy()
    # Trailing returns from corporate-action-continuous close/pre_close ratios.
    daily['gross']=daily.close/daily.pre_close
    daily['continuous']=daily.groupby('symbol').gross.cumprod()
    daily['above_ma20']=daily.continuous.gt(daily.groupby('symbol').continuous.transform(lambda s:s.rolling(20).mean()))
    daily['year']=daily.date.dt.year
    joined=daily.merge(members[['year','index_code','index_name','symbol','snapshot_date']],on=['year','symbol'],how='inner',validate='many_to_many')
    assert joined.snapshot_date.lt(joined.date).all()
    traded=joined.loc[joined.volume.gt(0)]
    panel=traded.groupby(['date','year','index_code','index_name']).agg(theme_return=('pct_chg','mean'),amount=('amount','sum'),breadth20=('above_ma20','mean'),observed_members=('symbol','nunique')).reset_index()
    counts=members.groupby(['year','index_code']).symbol.nunique().rename('expected_members')
    panel=panel.merge(counts,on=['year','index_code'],validate='many_to_one')
    panel['member_coverage']=panel.observed_members/panel.expected_members
    panel['theme_return']/=100
    panel['amount_share']=panel.amount/panel.date.map(market_amount)
    market=pd.read_parquet(ROOT/'data/raw/index_000300.SH.parquet')
    market['date']=pd.to_datetime(market.date if 'date' in market else market.trade_date.astype(str),errors='raise')
    market=market.sort_values('date').drop_duplicates('date').set_index('date')
    panel['market_ret20']=panel.date.map(market.close.pct_change(20,fill_method=None))
    signals=compute_mainlines(panel)
    # Meaningful causal check: extending the sample must not change old signals.
    cut=pd.Timestamp('2025-10-31')
    prefix=compute_mainlines(panel.loc[panel.date.le(cut)])
    check=signals.loc[signals.date.le(cut)]
    pd.testing.assert_frame_equal(prefix[['date','index_code','mainline_score','mainline']],check[['date','index_code','mainline_score','mainline']].reset_index(drop=True))
    signals.to_parquet(OUT/'daily_mainlines.parquet',index=False)
    signals.loc[signals.mainline].to_csv(OUT/'mainline_history.csv',index=False)
    pred=pd.read_parquet(ROOT/'reports/research/b1_screenshot_filters_20261004/candidate_filter_context.parquet')
    pred=pred.loc[pred.date.ge(START)&pred.exit_date.le(END)].copy()
    pred['year']=pred.date.dt.year
    registry=pd.read_csv(OUT/'leader_evidence.csv')
    for col in ['available_at','valid_until']:registry[col]=pd.to_datetime(registry[col])
    candidate_theme=pred[['symbol','date','year']].merge(members[['year','symbol','index_code']],on=['year','symbol'],how='inner')
    candidate_theme=candidate_theme.merge(signals[['date','index_code','mainline']],on=['date','index_code'],how='left',validate='many_to_one')
    pred=pred.merge(candidate_theme.groupby(['symbol','date']).mainline.any().rename('mainline'),on=['symbol','date'],how='left',validate='one_to_one')
    pred['mainline']=pred.mainline.fillna(False)
    covered=set(registry.index_code)
    coverage=candidate_theme.loc[candidate_theme.index_code.isin(covered)].groupby(['symbol','date']).size()
    coverage=coverage.rename('mapped_covered_themes').reset_index()
    pred=pred.merge(coverage,on=['symbol','date'],how='left',validate='one_to_one')
    pred['covered_universe']=pred.mapped_covered_themes.gt(0)
    le=candidate_theme.merge(registry,on=['symbol','index_code'],how='inner',validate='many_to_many')
    le=le.loc[le.date.gt(le.available_at)&le.date.le(le.valid_until)]
    flags=le.groupby(['symbol','date']).agg(leader=('symbol','size'),leader_mainline=('mainline','any')).reset_index()
    pred=pred.merge(flags,on=['symbol','date'],how='left',validate='one_to_one')
    pred['leader']=pred.leader.fillna(0).gt(0)
    pred['leader_mainline']=pred.leader_mainline.fillna(False)
    assert (~pred.leader_mainline|pred.mainline).all()
    assert (~pred.leader_mainline|pred.leader).all()
    pred.to_parquet(OUT/'candidate_context.parquet',index=False)
    variants={'A_current':pd.Series(True,index=pred.index),'A_covered':pred.covered_universe,'B_mainline':pred.mainline,'B_covered_mainline':pred.mainline&pred.covered_universe,'C_leader':pred.leader,'D_mainline_leader':pred.leader_mainline}
    pred['priority_mainline'] = pred.mainline.astype(int)
    pred['priority_both'] = np.select([pred.leader_mainline.to_numpy(dtype=bool), pred.mainline.to_numpy(dtype=bool), pred.leader.to_numpy(dtype=bool)], [3, 2, 1], default=0)
    variants['E_mainline_priority'] = pd.Series(True,index=pred.index)
    variants['F_combined_priority'] = pd.Series(True,index=pred.index)
    rows=[];selected_frames=[];rng=np.random.default_rng(20261004)
    for year in [2025,2026]:
        local=pred.loc[pred.date.dt.year.eq(year)]
        dates=pd.Index(sorted(local.date.unique()))
        base_slots=None
        for name,mask in variants.items():
            sort_columns = ['date','pred_unified_left_long_task_deep','symbol']
            ascending = [True,False,True]
            if name in ('E_mainline_priority','F_combined_priority'):
                sort_columns.insert(1, 'priority_mainline' if name == 'E_mainline_priority' else 'priority_both')
                ascending.insert(1,False)
            s=local.loc[mask.reindex(local.index)].sort_values(sort_columns,ascending=ascending).groupby('date').head(20).copy()
            s['variant']=name;s['net_return']=s.terminal_return-COST
            selected_frames.append(s)
            net=s.net_return
            slots=net.groupby(s.date).sum().reindex(dates,fill_value=0)/20
            if name=='A_current':base_slots=slots
            delta=(slots-base_slots).to_numpy()
            bootstrap=[]
            for _ in range(1000):
                starts=rng.integers(0,len(delta),size=int(np.ceil(len(delta)/20)))
                ix=np.concatenate([(v+np.arange(20))%len(delta) for v in starts])[:len(delta)]
                bootstrap.append(delta[ix].mean())
            loss=-net.loc[net.lt(0)].sum()
            rows.append(dict(year=year,variant=name,trades=len(s),unique_stocks=s.symbol.nunique(),active_days=s.date.nunique(),signal_days=len(dates),avg_net_return=net.mean(),win_rate=net.gt(0).mean() if len(s) else np.nan,profit_factor=net.loc[net.gt(0)].sum()/loss if loss else np.nan,slot_cohort_return=slots.mean(),slot_fill_rate=len(s)/(len(dates)*20),delta_slot_return=delta.mean(),delta_ci_low=np.quantile(bootstrap,.025),delta_ci_high=np.quantile(bootstrap,.975)))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False)
    selected=pd.concat(selected_frames,ignore_index=True);selected.to_csv(OUT/'selected_events.csv',index=False)
    selected.loc[selected.variant.eq('D_mainline_leader')].groupby(['symbol']).agg(trades=('net_return','size'),avg_net_return=('net_return','mean')).to_csv(OUT/'combined_stock_attribution.csv')
    signals['month']=signals.date.dt.to_period('M').astype(str)
    signals.loc[signals.date.between(START,END)].groupby(['month','index_code','index_name']).agg(mainline_days=('mainline','sum'),observed_days=('date','size'),average_relative20=('relative20','mean'),average_activity=('activity','mean')).to_csv(OUT/'monthly_mainline_review.csv')
    manifest=dict(status='success',start=str(START.date()),last_exit=str(END.date()),method='historical mainline market judgment + dated industrial leader evidence',themes=members.index_code.nunique(),leaders=registry.symbol.nunique(),round_trip_cost_bps=20,causality_test='prefix invariance passed',membership_policy='previous year-end historical membership held for next calendar year; not actual intra-year index constituent tracking',leader_policy='qualifying disclosed product ranking or oligopoly; source-specific geography; effective only after publication; 365-calendar-day maximum carry, with dated renewals for CATL/BYD/Haier; study ends 2026-03-27 before next major annual cycle',mainline='Top3 of10 themes; score .45 relative20 percentile + .25 return60 percentile + .20 activity percentile + .10 breadth percentile; positive return20 and excess20, aboveMA60, breadth>=50%, member coverage>=70%; Top3 on current day and >=3 of past5 sessions',limitations=['Limited themes and manually evidenced leader registry, not exhaustive A-share leaders','Retrospective hypothesis construction, no newly untouched holdout','Historical members obtained now from data provider; not independently archived exchange snapshots','Market mainline model is a transparent inference; catalysts not used to manually override selection','Frozen original B1 J<0 candidate universe; does not test J<13/25 expansion','Next-close entry and five-day event return, not production stop/expiry execution or portfolio NAV','Cash unfilled Top20 slots; repeated overlapping signals; no additional limit-down or liquidity execution model'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(metrics.to_string(index=False),flush=True)

if __name__=='__main__':main()
