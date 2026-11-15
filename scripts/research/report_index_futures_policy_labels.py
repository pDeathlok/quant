"""Complete, paired reporting for matched entry labels and past-only calibration."""
import json
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from evaluate_index_futures_policy_labels import (OUT,PREVIOUS,CONFIGS,ARMS,VARIANTS,PROFILES,ALL_PROFILES,
    CAPITAL,DATE_TYPES,RegimeReplay)
from index_futures_policy_labels import HORIZONS
from report_index_futures_state_models import describe
from report_index_futures_multistart import pct
from report_index_futures_reentry_cash import prefix,attribution

ARM_LABELS={'original':'原方案','hard_both':'硬入场/退出','previous_state':'上一轮状态期限',
            'prob_state':'匹配状态/仅概率校准','return_state':'匹配状态/仅收益风险校准'}
for h,label in [('h5','5日'),('h10','10日'),('h20','20日'),('state','状态期限')]:
    for family,text in [('five','原5日标签'),('matched','退出匹配标签'),('cal','匹配＋双校准')]:
        ARM_LABELS[f'{family}_{h}']=text+'/'+label
LABELS={m:('固定/' if s=='fixed' else '滚动/')+ARM_LABELS[a] for m,(s,a) in CONFIGS.items()}
KEYS=['start','variant','fee','margin','startup_mode']
FOCUS=['original','hard_both','previous_state','matched_state','cal_h5','cal_h10','cal_h20','cal_state']

@lru_cache(maxsize=128)
def ledger_at(start,variant,mode,fee,margin):
    return pd.read_parquet(OUT/start[:4]/'ledgers'/f'{start}_{variant}_{mode}_{fee}_{margin}.parquet')


def effect_pairs():
    pairs=[]
    for h in HORIZONS:
        pairs += [('label',f'five_{h}',f'matched_{h}'),('calibration',f'matched_{h}',f'cal_{h}')]
    for family in ('matched','cal'):
        for h in ('h5','h20','state'):pairs.append(('deadline',family+'_h10',family+'_'+h))
    pairs += [('probability','matched_state','prob_state'),('return','matched_state','return_state'),
              ('add_return','prob_state','cal_state'),('add_probability','return_state','cal_state'),
              ('total','original','cal_state'),('vs_hard','hard_both','cal_state'),
              ('population','previous_state','five_state'),('uniform5_total','original','cal_h5')]
    for style in ('fixed','rolling'):
        for effect,a,b in pairs:yield effect,style+'_'+a,style+'_'+b
    for arm in ARMS:yield 'rolling','fixed_'+arm,'rolling_'+arm


def model_diagnostics():
    f=pd.read_parquet(OUT/'forecasts.parquet');f['year']=f.trade_date.str[:4]
    f['alignment']=np.where(f.regime_bull.eq(0)&f.regime_bear.eq(0),'transition',
        np.where((f.regime_bull-f.regime_bear).eq(f.direction),'aligned','counter'))
    # Same paired observations for raw/calibration; readiness also separately
    # reported, so early fallback is not counted as trained-model evidence.
    records=[];reliability=[]
    keys=['year','target','bps','native_exit','direction','alignment']
    for key,g in f[f.year.ge('2022')].groupby(keys):
        mature=g[g.net.notna()&g.ready]
        for stage in ('raw','cal'):
            p,n,r=('probability','pred_net','pred_risk') if stage=='raw' else ('cal_probability','cal_net','cal_risk')
            y=mature.net.gt(0);pp=mature[p].clip(1e-6,1-1e-6)
            gate=mature[p].ge(.55)&(mature[n]-.5*mature[r]).gt(0);chosen=mature[gate]
            detail=dict(zip(keys,key),stage=stage,candidates=len(g),mature=len(mature),
                raw_ready_fraction=g.ready.mean(),cal_ready_fraction=g.calibration_ready.mean(),passed=len(chosen),
                brier=((pp-y.astype(int))**2).mean(),logloss=-(y*np.log(pp)+(1-y)*np.log(1-pp)).mean(),
                net_mae=(mature[n]-mature.net).abs().mean(),net_bias=(mature[n]-mature.net).mean(),
                risk_mae=(mature[r]-mature.risk).abs().mean(),risk_bias=(mature[r]-mature.risk).mean(),
                mean_probability=pp.mean(),actual_win=y.mean(),selected_probability=chosen[p].mean(),
                selected_win=chosen.net.gt(0).mean(),selected_net=chosen.net.mean())
            records.append(detail)
            for bin_id,z in mature.groupby(pd.cut(mature[p],np.linspace(0,1,6),include_lowest=True),observed=True):
                reliability.append(dict(zip(keys,key),stage=stage,bin=str(bin_id),samples=len(z),
                    predicted=z[p].mean(),actual=z.net.gt(0).mean(),mean_net=z.net.mean()))
    pd.DataFrame(records).to_csv(OUT/'model_diagnostics.csv',index=False)
    pd.DataFrame(reliability).to_csv(OUT/'reliability_bins.csv',index=False)
    audit=pd.read_csv(OUT/'training_audit.csv',dtype=DATE_TYPES|{'last_maturity':str,'cal_last_maturity':str})
    assert audit[audit.ready].last_maturity.lt(audit[audit.ready].cutoff).all()
    assert audit[audit.calibration_ready].cal_last_maturity.lt(audit[audit.calibration_ready].cutoff).all()
    f.to_parquet(OUT/'diagnostic_forecasts.parquet',index=False)
    return f,audit


def trade_label_audit(trades,forecasts):
    # All mechanical deadlines use matched target outcomes for the same entry,
    # independent of whether the entry model used 5d, matched or calibrated targets.
    lookup=forecasts.set_index(['entry_date','symbol','direction','bps','native_exit','target'])
    records=[]
    for key,t in trades.groupby(KEYS):
        start,variant,fee,margin,mode=key;arm=CONFIGS[mode][1]
        if arm in ('original','hard_both','previous_state'):continue
        target=arm.split('_')[-1]
        trace=pd.read_parquet(OUT/start[:4]/'traces'/f'{start}_{variant}_{mode}_{fee}_{margin}.parquet')
        entries=trace[trace.action.eq('enter')].set_index('date')
        actions=trace.set_index('date').action
        for r in t.itertuples(index=False):
            policy=entries.loc[r.entry,'policy'];native=int(policy=='opposing_exit')
            label=lookup.loc[(r.entry,r.product,r.direction,round(fee*10000),native,target)]
            action=actions.get(r.exit,'sample_end')
            equal_exit=pd.notna(label.maturity) and r.exit==label.maturity
            record=dict(zip(KEYS,key),entry=r.entry,exit=r.exit,product=r.product,side=r.side,policy=policy,
                label_exit=label.maturity,label_days=label.label_days,actual_days=r.days,actual_pnl=r.pnl,
                label_net=label.net,actual_exit_action=action,exit_matches=equal_exit,
                label_ready=label.ready,calibration_ready=label.calibration_ready)
            records.append(record)
    result=pd.DataFrame(records)
    # Convert conditional label return to actual CNY using the same entry notional.
    panel=pd.read_parquet(PREVIOUS.parent/'adaptation_comparison/quarterly_predictions.parquet')
    notionals={(r.entry_date,r.symbol):r.entry_open*r.multiplier for r in panel.itertuples(index=False)}
    result['label_pnl']=[r.label_net*notionals[r.entry,r.product] for r in result.itertuples(index=False)]
    result['cash_difference']=result.actual_pnl-result.label_pnl
    np.testing.assert_allclose(result.loc[result.exit_matches,'cash_difference'],0,rtol=0,atol=1e-6)
    mismatches=result[~result.exit_matches]
    assert mismatches.actual_exit_action.isin(['reserve_exit','strategy_switch','sample_end']).all(),mismatches.actual_exit_action.value_counts()
    result.to_csv(OUT/'trade_label_audit.csv',index=False)
    result.assign(entry_year=result.entry.str[:4]).groupby(['startup_mode','fee','margin','entry_year']).agg(
        trades=('entry','size'),match_fraction=('exit_matches','mean'),raw_ready_fraction=('label_ready','mean'),
        calibration_ready_fraction=('calibration_ready','mean')).to_csv(OUT/'label_match_by_year.csv')
    return result


def charts(data):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/quant_adaptation_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    arms=['original','hard_both','previous_state','matched_state','cal_h5','cal_h10','cal_h20','cal_state']
    labels=['Original','Hard gates','Previous state','Matched state','Calibrated 5d','Calibrated 10d','Calibrated 20d','Calibrated state']
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,year in enumerate(('2025','2026')):
        for j,style in enumerate(('fixed','rolling')):
            ax=axes[i,j]
            part=data[data.start_kind.eq('monthly')&data.start.str.startswith(year)&data.window.eq('to_latest')&data.fee.eq(.0006)&data.margin.eq(.25)]
            for arm,label in zip(arms,labels):
                g=part[part.startup_mode.eq(style+'_'+arm)].sort_values('start')
                ax.plot(g.start.str[4:6].astype(int),100*g.total_return,label=label,marker='o',ms=2)
            ax.set_title(f'{year} starts | {style} | 6bps/25% margin');ax.axhline(0,c='grey',lw=.7)
            ax.set_xlabel('Startup month');ax.set_ylabel('Net return to 2026-11-06 (%)');ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Policy-matched labels and calibration: startup sensitivity')
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/'recent_starts.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for i,start in enumerate(('20230801','20260101')):
        for j,style in enumerate(('fixed','rolling')):
            ax=axes[i,j]
            for arm,label in zip(arms,labels):
                l=ledger_at(start,'mean5',style+'_'+arm,.0006,.25)
                ax.plot(pd.to_datetime(l.exit_date),l.equity/10000,label=label,lw=1.1)
            ax.set_title(f'Start {start} | {style} | 6bps/25%');ax.set_ylabel('Equity (CNY 10k)');ax.grid(alpha=.2)
            loc=mdates.AutoDateLocator(minticks=3,maxticks=5);ax.xaxis.set_major_locator(loc);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Continuous accounts without topups')
    fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(OUT/'account_paths.png',dpi=150);plt.close(fig)


def forced_events(totals,replay):
    records=[]
    for r in totals[totals.forced.gt(0)].itertuples(index=False):
        ledger=ledger_at(r.start,r.variant,r.startup_mode,r.fee,r.margin)
        for i in ledger.index[ledger.forced_exit]:
            prior=ledger.iloc[i-1] if i else None
            held=prior.positions if prior is not None else ''
            code,side=held.split(':') if held else ('','0')
            equity=prior.equity if prior is not None else CAPITAL
            day=ledger.entry_date.iloc[i]
            notional=replay.opens[day,code]*(300 if code[:2] in ('IF','IH') else 200) if code else 0.
            records.append(dict(start=r.start,startup_mode=r.startup_mode,fee=r.fee,margin=r.margin,date=day,
                held=held,before_equity=equity,required_margin=notional*r.margin,
                open_shortfall=max(0.,notional*r.margin-equity),after_equity=ledger.equity.iloc[i],
                prior_settlement_breach=bool(prior.settlement_breach) if prior is not None else False))
    result=pd.DataFrame(records)
    result.to_csv(OUT/'forced_exit_events.csv',index=False)
    return result


def main():
    data=pd.read_csv(OUT/'summary.csv',dtype=DATE_TYPES,low_memory=False);data['start_year']=data.start.str[:4]
    totals=data[data.window.eq('to_latest')]
    assert len(totals)==5644 and totals.groupby('startup_mode').size().eq(166).all()
    assert not data.duplicated(KEYS+['window']).any()
    trades=pd.read_csv(OUT/'trades.csv',dtype=DATE_TYPES,low_memory=False)
    annual=pd.read_csv(OUT/'annual_marked.csv',dtype=DATE_TYPES,low_memory=False)
    decisions=pd.read_csv(OUT/'selection.csv',dtype=DATE_TYPES,low_memory=False)
    audit=pd.read_csv(OUT/'selection_audit.csv',dtype=DATE_TYPES,low_memory=False)
    old=pd.read_csv(PREVIOUS/'summary.csv',dtype=DATE_TYPES,low_memory=False)
    for style in ('fixed','rolling'):
        for arm,suffix in (('original','original'),('hard_both','hard_both'),('previous_state','state_horizon')):
            current=data[data.startup_mode.eq(style+'_'+arm)]
            control=old[old.startup_mode.eq(style+'_'+suffix)&old.variant.eq('mean5')]
            compare=current.merge(control,on=['start','variant','fee','margin','window'],suffixes=('','_old'),validate='one_to_one')
            assert len(compare)==len(current)==len(control)
            for col in ('total_return','max_drawdown','trades','forced'):
                np.testing.assert_allclose(compare[col],compare[col+'_old'],rtol=0,atol=1e-8)
    recon=totals.merge(trades.groupby(KEYS).pnl.sum().rename('cash_pnl'),on=KEYS,how='left')
    np.testing.assert_allclose(recon.cash_pnl.fillna(0),recon.total_return*CAPITAL,rtol=0,atol=1e-6)
    assert decisions.cutoff.lt(decisions.date).all() and audit.last_history_end.le(audit.cutoff).all()
    np.testing.assert_array_equal(decisions.equity,decisions.validation_capital)
    assert (OUT/'global_calendar.csv').read_bytes()==(PREVIOUS/'global_calendar.csv').read_bytes()
    forecasts,training=model_diagnostics()
    matches=trade_label_audit(trades,forecasts)
    monthly=data[data.start_kind.eq('monthly')]
    dist=describe(monthly,['startup_mode','variant','fee','margin','window'])
    cohorts=describe(monthly,['start_year','startup_mode','variant','fee','margin','window'])
    phases=describe(data[data.annual|data.start_kind.eq('phase')],['start_year','startup_mode','variant','fee','margin','window'])
    for name,frame in (('start_distributions',dist),('year_cohorts',cohorts),('january_phase_sensitivity',phases)):
        frame.to_csv(OUT/f'{name}.csv',index=False)
    replay=RegimeReplay();frozen=pd.read_parquet(PREVIOUS.parent/'bull_bear_regime/states_200.parquet')
    replay.market_regime={(r.trade_date,r.symbol):int(r.state) for r in frozen.dropna(subset=['state']).itertuples(index=False)}
    forced=forced_events(totals,replay)
    pairs=[];opportunities=[]
    for effect,base,new in effect_pairs():
        a=data[data.startup_mode.eq(base)];b=data[data.startup_mode.eq(new)]
        pair=b.merge(a,on=['start','variant','fee','margin','window'],suffixes=('','_base'),validate='one_to_one').copy()
        pair['effect']=effect;pair['baseline_mode']=base
        pair['return_change']=pair.total_return-pair.total_return_base
        pair['drawdown_change']=pair.max_drawdown-pair.max_drawdown_base
        cols=[*KEYS,'start_kind','start_year','window','effect','baseline_mode','return_change','drawdown_change',
              'total_return','total_return_base','max_drawdown','max_drawdown_base','flat_ratio','flat_ratio_base']
        pairs.append(pair[cols])
        for (start,variant,fee,margin),p in pair.groupby(['start','variant','fee','margin']):
            full_a=ledger_at(start,variant,base,fee,margin);full_b=ledger_at(start,variant,new,fee,margin)
            for r in p.itertuples(index=False):
                aa=prefix(full_a,r.window,start,fee,replay.opens);bb=prefix(full_b,r.window,start,fee,replay.opens)
                values=attribution(aa,bb)
                np.testing.assert_allclose(values['net_pnl_change'],r.return_change*CAPITAL,rtol=0,atol=1e-6)
                opportunities.append(dict(effect=effect,baseline_mode=base,startup_mode=new,start=start,start_year=start[:4],
                    start_kind=r.start_kind,variant=variant,fee=fee,margin=margin,window=r.window,**values))
        print('attributed',effect,base,new,flush=True)
    paired=pd.concat(pairs,ignore_index=True);paired.to_csv(OUT/'paired_effects.csv',index=False)
    op=pd.DataFrame(opportunities);op.to_csv(OUT/'opportunity_costs.csv',index=False)
    keys=['effect','baseline_mode','startup_mode','start_year','variant','fee','margin','window']
    effects=paired[paired.start_kind.eq('monthly')].groupby(keys).agg(starts=('start','size'),
        median_return_change=('return_change','median'),return_improved_fraction=('return_change',lambda s:s.gt(1e-10).mean()),
        drawdown_improved_fraction=('drawdown_change',lambda s:s.gt(1e-10).mean()),
        drawdown_worsened_fraction=('drawdown_change',lambda s:s.lt(-1e-10).mean())).reset_index()
    costs=op[op.start_kind.eq('monthly')].groupby(keys).agg(median_missed_positive=('missed_positive_pnl','median'),
        median_avoided_negative=('avoided_negative_pnl','median'),median_additional_flat_days=('additional_flat_vs_held_days','median')).reset_index()
    effects=effects.merge(costs,on=keys,validate='one_to_one');effects.to_csv(OUT/'effects_by_year.csv',index=False)
    worst=[]
    for _,g in data.groupby(['startup_mode','fee','margin','window']):
        for metric in ('total_return','max_drawdown'):worst.append(dict(g.loc[g[metric].idxmin()],criterion=metric))
    pd.DataFrame(worst).to_csv(OUT/'worst_all_starts.csv',index=False)
    state_at={(r.entry_date,r.symbol):replay.market_regime[r.trade_date,r.symbol] for g in replay.groups for r in g}
    trades['entry_regime']=[state_at[r.entry,r.product] for r in trades.itertuples(index=False)]
    trades['alignment']=np.where(trades.entry_regime.eq(0),'transition',np.where(trades.entry_regime.eq(trades.direction),'aligned','counter'))
    trades.to_csv(OUT/'trades_with_regime.csv',index=False)
    trades.groupby(KEYS+['alignment','side']).agg(trades=('pnl','size'),net_pnl=('pnl','sum'),
        win_rate=('pnl',lambda s:s.gt(0).mean()),worst_trade=('pnl','min')).to_csv(OUT/'trade_regime_by_account.csv')
    checks=dict(accounts=len(totals),windows=len(data),controls_replayed=sum(json.loads((OUT/str(y)/'checks.json').read_text())['replayed_controls'] for y in range(2022,2027)),
        training_audit_records=len(training),mature_training_and_calibration_only=True,decision_records=len(decisions),
        unique_validation_sets=audit.validation_id.nunique(),exact_current_equity=True,all_trade_cash_reconciles=True,
        opportunity_rows=len(op),opportunity_cash_reconciles=True,forced_account_count=int(totals.forced.gt(0).sum()),
        forced_events=len(forced),forced_market_dates=forced.date.nunique() if len(forced) else 0,
        label_audited_trades=len(matches),label_exit_match_fraction=matches.exit_matches.mean(),
        label_matching_cash_error=float(matches.loc[matches.exit_matches,'cash_difference'].abs().max()),
        label_divergence_causes=matches[~matches.exit_matches].actual_exit_action.value_counts().to_dict(),
        unit_tests=119,**json.loads((OUT/'engine_checks.json').read_text()))
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    report(data,dist,cohorts,annual,effects,checks)
    charts(data);print(json.dumps(checks,indent=2),flush=True)


def report(data,dist,cohorts,annual,effects,checks):
    lines=['# 入场标签匹配退出策略与概率/收益校准','',
        '结论和反例见[findings.md](findings.md)。数据截止2026-11-06；50万元最多一手、无补资。17组预登记机制×固定/滚动、5644账户。本轮新交叉固定原mean5退出；其他旧退出研究保留，不将本轮结果推广至未测试的交叉。','',
        '## 冻结定义','',
        '- 原方案、硬入场/退出、上一轮状态期限作为逐账本复现对照。MA200状态、原四种重选政策、全局入场日历、费用/资金约束保持一致。',
        '- 期限：统一5/10/20交易日与入场锁定的顺势20/过渡10/逆势5日上限。每组比较同样合格候选样本上的原固定5日标签、退出匹配标签、匹配加校准。状态期限再拆开概率和收益/风险校准作用。原退出可更早触发。',
        '- 匹配标签在原mean5连续两次负值、当前native opposing政策退出或期限首次触发时结束，扣全部开平与换月费、纳入开盘及收盘不利路径。当前政策在标签期间固定。账户资金约束、未来策略切换和样本末清算外部覆盖，trade_label_audit.csv量化剩余不匹配。',
        '- 新模型仅使用原合格候选日及选定方向，不能与上一轮全部日期/双向训练直接归因；同样样本重训5日标签控制此差异。85状态交互特征＋当前native退出标记。',
        '- 季度LR C=.1、Ridge alpha=.1*n，多空分开，过去36月成熟标签、12月半衰期。新模型至少122候选日期和两类各20行；否则回退上一轮交互预测。多头2021Q1、空头2023Q2开始就绪。',
        '- 校准只使用过去24月实际发出的季度预测及已成熟结果，按日期平衡、12月半衰期；至少60日、两个预测季度、两类各20行。多头2021Q3、空头2023Q4开始就绪；不足时保留原预测。概率为单调sigmoid，净收益/风险为向恒等映射收缩的仿射校准，参数见protocol.json。',
        '- 校准必须与基础模型训练预测分离，参见[scikit-learn官方说明](https://scikit-learn.org/stable/modules/calibration.html)。本轮使用按时间产生的历史预测，不随机划分未来数据。Brier同时涉及校准和区分能力，另列可靠性分箱、收益误差和账户表现。',
        '- 入场门槛仍是概率≥55%且净收益−0.5风险>0，允许逆势；不根据已知亏损调整门槛。滚动仅在原四政策中按过去完成窗口、实际当前权益和原切换门槛重选，不选择本轮事后最佳机制。',
        '- 仍是重复研究过的探索历史，不是全新OOT；重叠候选、起点不是独立样本。年度沿用开盘到次日开盘账本，按exit_date年份记账，不是交易所收盘结算表。','',
        '## 完整244日等长窗口','','47个月度起点，最大回撤以正损失幅度列示。','','| 成本/保证金 | 方案 | 收益中位 | P10收益 | 最差收益 | 最大回撤 | 空仓中位 | 零交易账户 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in dist[dist.window.eq('sessions_244')].itertuples():
        values=[r.median_return,r.p10_return,r.worst_return,-r.worst_drawdown,r.median_flat_ratio]
        lines.append(f'| {r.fee*10000:.0f}bps/{r.margin:.0%} | {LABELS[r.startup_mode]} | '+' | '.join(pct(v) for v in values)+f' | {r.zero_trade_accounts} |')
    lines+=['','## 逐年单年与各年启动累计至共同终点','','每格收益/最大回撤。单年各50万元；累计不重置权益，2026为部分年度。','']
    for fee,margin in ALL_PROFILES:
        for window in ('single_year','to_latest'):
            for style in ('fixed','rolling'):
                lines += [f'### {fee*10000:.0f}bps/{margin:.0%} · {window} · {style}','','| 方案 | 2022 | 2023 | 2024 | 2025 | 2026 |','|---|---:|---:|---:|---:|---:|']
                part=data[data.annual&data.fee.eq(fee)&data.margin.eq(margin)&data.window.eq(window)]
                for arm in ARMS:
                    g=part[part.startup_mode.eq(style+'_'+arm)].set_index('start_year')
                    lines.append('| '+ARM_LABELS[arm]+' | '+' | '.join(f"{pct(g.loc[y,'total_return'])}/{pct(-g.loc[y,'max_drawdown'])}" for y in map(str,range(2022,2027)))+' |')
                lines.append('')
    lines+=['## 逐年月度起点、等长窗口及空仓','','每格收益中位/最差收益/最大回撤/空仓中位。2026没有完整244日新起点。','']
    for fee,margin in PROFILES:
        for window in ('to_latest','sessions_122','sessions_244'):
            for style in ('fixed','rolling'):
                lines += [f'### {fee*10000:.0f}bps/{margin:.0%} · {window} · {style}','','| 方案 | 2022 | 2023 | 2024 | 2025 | 2026 |','|---|---:|---:|---:|---:|---:|']
                part=cohorts[cohorts.fee.eq(fee)&cohorts.margin.eq(margin)&cohorts.window.eq(window)]
                for arm in ARMS:
                    g=part[part.startup_mode.eq(style+'_'+arm)].set_index('start_year')
                    cells=[]
                    for y in map(str,range(2022,2027)):
                        cells.append('—' if y not in g.index else '/'.join(pct(v) for v in [g.loc[y,'median_return'],g.loc[y,'worst_return'],-g.loc[y,'worst_drawdown'],g.loc[y,'median_flat_ratio']]))
                    lines.append('| '+ARM_LABELS[arm]+' | '+' | '.join(cells)+' |')
                lines.append('')
    lines+=['## 2025启动持续账户的2026表现','','同一账户、不重置资金，收益分母为实际年初权益。6bps/25%。','','| 方案 | 2025收益/回撤 | 2026收益/回撤 |','|---|---:|---:|']
    part=annual[annual.start.eq('20250101')&annual.side.eq('all')&annual.fee.eq(.0006)&annual.margin.eq(.25)]
    for mode in CONFIGS:
        g=part[part.startup_mode.eq(mode)].set_index('year')
        lines.append('| '+LABELS[mode]+' | '+' | '.join(f"{pct(g.loc[y,'total_return'])}/{pct(-g.loc[y,'max_drawdown'])}" for y in ('2025','2026'))+' |')
    lines+=['','## 配对增量与机会成本','','同一账户起点配对，不用两组中位数之差代替配对中位。错失盈利和避免亏损是原账户有仓、新账户空仓的日期现金分解，不能理解为可执行的理想择时收益；不同列中位数不可直接相减。以下6bps/25%，完整其他情景见effects_by_year.csv。','']
    for window in ('to_latest','sessions_244','sessions_122'):
        lines += [f'### {window}','','| 年 | 对照→方案 | 起点 | 收益差中位 | 收益改善 | 回撤改善/恶化 | 新增空仓日 | 错失盈利/避免亏损（万元） |','|---|---|---:|---:|---:|---:|---:|---:|']
        part=effects[effects.fee.eq(.0006)&effects.margin.eq(.25)&effects.window.eq(window)]
        for r in part.sort_values(['start_year','effect','startup_mode']).itertuples():
            lines.append(f'| {r.start_year} | {LABELS[r.baseline_mode]}→{LABELS[r.startup_mode]} | {r.starts} | {pct(r.median_return_change)} | {pct(r.return_improved_fraction)} | {pct(r.drawdown_improved_fraction)}/{pct(r.drawdown_worsened_fraction)} | {r.median_additional_flat_days:.1f} | {r.median_missed_positive/10000:.2f}/{r.median_avoided_negative/10000:.2f} |')
        lines.append('')
    lines+=['## 审计','','```json',json.dumps(checks,ensure_ascii=False,indent=2),'```','',
        '附表：worst_all_starts.csv、january_phase_sensitivity.csv、annual_marked.csv、model_diagnostics.csv、reliability_bins.csv、trade_label_audit.csv、label_match_by_year.csv、trades_with_regime.csv。模型指标是候选成熟标签表现，不等于真实账户成交。完整起点逐日账本和实际权益历史验证保存在逐年目录。','',
        '![近两年起点](recent_starts.png)','','![持续账户](account_paths.png)','']
    (OUT/'report.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
