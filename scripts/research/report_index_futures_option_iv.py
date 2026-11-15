"""Evidence for self-built option IV: coverage, model increments and funded paths."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from index_futures_option_iv import OUT, BASE, QVIX, MODELS
from report_index_futures_panic import pct, table

ARMLABELS={'baseline':'原状态期限','qvix':'原QVIX','native_mf':'自建方差30','native_atm':'自建近平值30','qvix_mo':'QVIX替换MO'}
LABELS={f'{style}_{arm}':f'{label}/{name}' for style,label in [('fixed','固定'),('rolling','重选')] for arm,name in ARMLABELS.items()}


def diagnostics():
    forecasts=pd.read_parquet(OUT/'forecasts.parquet')
    panel=pd.read_parquet(BASE.parent/'adaptation_comparison/quarterly_predictions.parquet')
    forecasts=forecasts.merge(panel[['entry_date','symbol','long_60','short_60','etf_ret_1']],on=['entry_date','symbol'],validate='many_to_one')
    forecasts['candidate']=(np.maximum(forecasts.long_60,forecasts.short_60)>.45)&((forecasts.long_60-forecasts.short_60).abs()>=.1)&forecasts.etf_ret_1.notna()&forecasts.direction.eq(np.where(forecasts.long_60>forecasts.short_60,1,-1))
    forecasts['actual_net']=forecasts.net_3.where(forecasts.bps.eq(3),forecasts.net_6)
    forecasts['year']=forecasts.entry_date.str[:4]
    old=pd.read_parquet(QVIX/'forecasts.parquet');old=old[old.model.eq('qvix')]
    forecasts=forecasts.merge(old[['entry_date','symbol','direction','bps','probability','pred_net','pred_risk']].rename(columns={k:'old_'+k for k in ['probability','pred_net','pred_risk']}),on=['entry_date','symbol','direction','bps'],validate='many_to_one')
    forecasts['new_pass']=forecasts.probability.ge(.55)&(forecasts.pred_net-.5*forecasts.pred_risk).gt(0)
    forecasts['old_pass']=forecasts.old_probability.ge(.55)&(forecasts.old_pred_net-.5*forecasts.old_pred_risk).gt(0)
    rows=[]
    valid=forecasts[forecasts.candidate&forecasts.actual_net.notna()&forecasts.ready&forecasts.year.ge('2022')]
    for key,g in valid.groupby(['year','model','symbol','direction','bps']):
        for sample,h in [('all',g),('new_iv_available',g[g.fallback.eq('none')])]:
            selected=h[h.new_pass]
            rows.append(dict(zip(['year','model','symbol','direction','bps'],key),sample=sample,n=len(h),
                fallback=h.fallback.ne('none').mean(),brier=((h.probability-h.actual_net.gt(0))**2).mean(),
                old_brier=((h.old_probability-h.actual_net.gt(0))**2).mean(),
                changed_admission=int(h.new_pass.ne(h.old_pass).sum()),passed=len(selected),label_win=selected.actual_net.gt(0).mean()))
    pd.DataFrame(rows).to_csv(OUT/'model_diagnostics.csv',index=False)
    forecasts[forecasts.year.ge('2022')].groupby(['year','model','symbol']).agg(rows=('entry_date','size'),fallback_rate=('fallback',lambda s:s.ne('none').mean()),own_missing=('panic_own_missing','mean'),common_missing=('panic_common_missing','mean')).to_csv(OUT/'forecast_coverage.csv')
    ids=pd.read_parquet(OUT/'indices.parquet')
    a=ids[ids.price.eq('close')].merge(ids[ids.price.eq('settle')],on=['trade_date','product'],suffixes=('_close','_settle'))
    a['year']=a.trade_date.str[:4]
    price_rows=[]
    for (year,product),g in a.groupby(['year','product']):
        for method in ('mf','atm'):
            diff=(g[f'{method}_value_close']-g[f'{method}_value_settle']).abs().dropna()
            price_rows.append(dict(year=year,product=product,method=method,paired_days=len(diff),median_abs_vol_points=diff.median(),p95_abs_vol_points=diff.quantile(.95),maximum_abs_vol_points=diff.max()))
    pd.DataFrame(price_rows).to_csv(OUT/'close_settle_sensitivity.csv',index=False)


def main():
    diagnostics()
    d=pd.read_csv(OUT/'summary.csv',dtype={'start':str,'end':str});d['start_year']=d.start.str[:4]
    annual=pd.read_csv(OUT/'annual_marked.csv',dtype={'start':str,'year':str})
    monthly=d[d.start_kind.eq('monthly')]
    def grouped(keys):
        return monthly.groupby(keys).agg(accounts=('start','size'),median_return=('total_return','median'),q10_return=('total_return',lambda s:s.quantile(.1)),worst_return=('total_return','min'),worst_drawdown=('max_drawdown','min'),median_flat=('flat_ratio','median'),positive_fraction=('total_return',lambda s:s.gt(0).mean()),forced_accounts=('forced',lambda s:s.gt(0).sum()),median_blocked=('blocked_ratio','median')).reset_index()
    whole=grouped(['startup_mode','fee','margin','window']);whole.to_csv(OUT/'grouped_all.csv',index=False)
    years=grouped(['startup_mode','fee','margin','window','start_year']);years.to_csv(OUT/'grouped_by_year.csv',index=False)
    unique=d.drop_duplicates(['start','startup_mode','fee','margin'])
    check=json.loads((OUT/'engine_checks.json').read_text());controls=sum(json.loads((OUT/str(y)/'checks.json').read_text())['replayed_controls'] for y in range(2022,2027))
    audit=pd.read_csv(OUT/'training_audit.csv',dtype={'cutoff':str,'last_maturity':str})
    assert audit.loc[audit.ready,'last_maturity'].lt(audit.loc[audit.ready,'cutoff']).all()
    history=pd.read_csv(OUT/'history_coverage.csv')
    lines=['# Tushare期权历史与自建隐波研究（2026-11-15）','',
        '本轮先固定策略比较，再看季度重选。保留原状态期限和此前QVIX的冻结预测与账本；新增跨行权价方差30日、近平值隐波30日，以及仅把QVIX中IM自身信息换为MO方差的三组研究分支。没有按2026损失调整门槛、标签或退出。','',
        '## 数据与可用性','',
        f"下载中金所日线{int(history.loc[history.exchange.eq('CFFEX'),'rows'].sum()):,}条、上交所期权日线{int(history.loc[history.exchange.eq('SSE'),'rows'].sum()):,}条；合约列表与日线均翻页至空页，检测重复和越界。MO在2026截至11月6日203个交易日均有行情，共58,416条。原QVIX中证1000同年仅11条非空记录。",
        'SSE数据已补齐并保留，但本轮自建指标仅用IO/MO/HO。ETF期权最新静态资料包含调整后的行权价，不能直接回套过去；为避免历史行权价穿越，未构造ETF隐波。IC本轮自建分支只有共同IO因子、自身隐波缺失；QVIX对照仍保留其500ETF信息。','',
        '标准VIX使用买卖报价中间价；这里使用有成交、正持仓的日线收盘价，只能称为可审计的隐波代理。先检查合约有效期、代码行权价/方向、收盘价范围和看涨看跌平价，再计算远期、方差或近平值隐波；不按策略收益修复价格。结算价仅作敏感性诊断，未按回测收益挑选价格源。','',
        '## 固定计算方法','',
        '- 期限7–120日；同到期日至少3组有效看涨/看跌配对。取价差绝对值最小的最多5对，以各自平价远期的中位数为F；90%–10%远期离散超过F的1%则拒绝。',
        '- 跨行权价方差：以K0上下的虚值期权和K0处双边均价计算模型无关方差近似。每侧至少3个虚值行权价、覆盖F上下至少2%，最大间距不超过中位网格3倍；严重非单调报价拒绝。',
        '- 近平值：F上下5%内最近最多5个配对行权价，反解Black76虚值期权隐波，至少3个有效值，取中位数。',
        '- 两种方法分别选择夹住30日的最近有效到期日，按总方差插值；不做期限外推，不用旧值填空。缺乏7日以上近月时可能无法夹住30日，是明确的覆盖代价。',
        '- 利率为当日Shibor期限插值、简单利率转换连续复利；这是无担保利率代理，不是严格无风险利率。期权日线收盘并非同步报价，尾部仍可能截断。','']
    c=pd.read_csv(OUT/'iv_coverage.csv')
    table(lines,['年','期权','交易日','方差30可用日','近平值30可用日'],[(r.year,r.product,r.days,r.mf_days,r.atm_days) for r in c[c.year.ge(2025)&c.price.eq('close')].itertuples()])
    lines+=['## 接入与评估口径','',
        '保留原85因子，三个新分支都只加入与QVIX同形状的8项特征（共同/自身历史分位、5日变化、隐含减实际波动、缺失标记），总计93项。自建共同指标为IO；IF自身IO、IH自身HO、IM自身MO，IC无自身股指期权。QVIX替换MO分支保留其余QVIX数据。',
        '额外滞后一个交易日，再于下一开盘执行。历史分位过去252日至少60有效日；5日变化要求6个连续有效观测，所以原始隐波可用率高于完整模型特征可用率。自建共同特征不可用则回退冻结QVIX预测；替换MO分支在IM自身不可用时也回退，回退次数单独记录。',
        '多空模型季度重训，使用过去36个月已成熟的原5日净收益/风险标签，近期12月半衰期；原LR/Ridge参数、55%概率和净收益−0.5×风险门槛、mean5保护退出、状态期限20/10/5、固定全局入场日历均不变。滚动重选仍以账户实际权益回放历史窗口，使用原4政策和切换门槛。',
        f'50万元、最多一手、不补资；共{len(unique)}条账户、{len(d)}个统计窗口；{controls}条原方案/QVIX控制账本逐日一致；{check["daily_cases"]}条Python/C++路径核对，最大现金差{check["max_cash_error"]}元。','']
    coverage=pd.read_csv(OUT/'forecast_coverage.csv')
    table(lines,['2026模型','品种','预测回退比例','自身特征缺失率'],[(r.model,r.symbol,pct(r.fallback_rate),pct(r.own_missing)) for r in coverage[coverage.year.eq(2026)].itertuples()])
    for fee,margin in ((.0003,.15),(.0006,.25)):
        condition=d.fee.eq(fee)&d.margin.eq(margin)
        lines += [f'## {fee*10000:.0f}bps / {margin:.0%}保证金','']
        for style,label in [('fixed','固定政策'),('rolling','季度重选')]:
            modes=[f'{style}_{arm}' for arm in ARMLABELS]
            for window,title in [('single_year','逐年独立单年'),('to_latest','各年初启动累计至2026-11-06')]:
                lines += [f'### {label}：{title}，收益 / 最大回撤','']
                sample=d[condition&d.window.eq(window)&d.annual&d.startup_mode.isin(modes)]
                table(lines,['启动年',*ARMLABELS.values()],[(year,*[f"{pct(g.set_index('startup_mode').loc[m,'total_return'])} / {pct(-g.set_index('startup_mode').loc[m,'max_drawdown'])}" for m in modes]) for year,g in sample.groupby('start_year')])
        for window in ('sessions_122','sessions_244'):
            lines += [f'### {window}等长月度起点','']
            sample=whole[whole.fee.eq(fee)&whole.margin.eq(margin)&whole.window.eq(window)]
            table(lines,['方案','起点数','收益中位','收益10%分位','最差收益','最大回撤','空仓中位'],[(LABELS[r.startup_mode],r.accounts,pct(r.median_return),pct(r.q10_return),pct(r.worst_return),pct(-r.worst_drawdown),pct(r.median_flat)) for r in sample.itertuples()])
        lines += ['### 2026月度启动累计至今','']
        sample=years[years.fee.eq(fee)&years.margin.eq(margin)&years.window.eq('to_latest')&years.start_year.eq('2026')]
        table(lines,['方案','收益中位','最差收益','最大回撤','盈利起点比例','空仓中位'],[(LABELS[r.startup_mode],pct(r.median_return),pct(r.worst_return),pct(-r.worst_drawdown),pct(r.positive_fraction),pct(r.median_flat)) for r in sample.itertuples()])
        lines += ['### 2025启动持续账户：逐年收益与回撤','']
        sample=annual[annual.start.eq('20250101')&annual.side.eq('all')&annual.fee.eq(fee)&annual.margin.eq(margin)]
        table(lines,['方案','记账年','年度收益','年度回撤'],[(LABELS[r.startup_mode],r.year,pct(r.total_return),pct(-r.max_drawdown)) for r in sample.itertuples()])
        lines += ['### 全部月度/相邻日起点累计至今：最差账户','']
        rows=[]
        for mode,g in d[condition&d.window.eq('to_latest')].groupby('startup_mode'):
            w=g.loc[g.total_return.idxmin()];v=g.loc[g.max_drawdown.idxmin()]
            rows.append((LABELS[mode],w.start,pct(w.total_return),v.start,pct(-v.max_drawdown),int(g.forced.gt(0).sum()),pct(g.blocked_ratio.max())))
        table(lines,['方案','最差收益起点','最差收益','最大回撤起点','最大回撤','强平账户数','最大资金拒绝比例'],rows)
    lines += ['## 2026年初启动：费用与保证金分别变化','',
        '以下是预先保留的四组年度交叉情景。低费用不必对应高收益：模型标签按成本生成，且一手资金边界、筛选与后续账户权益会改变交易路径。不能把6bps/25%的好结果推广到所有高保证金账户；完整月度网格仅覆盖3bps/15%和6bps/25%，其余两组为年度交叉验证。','']
    for style,label in [('fixed','固定'),('rolling','重选')]:
        modes=[f'{style}_{arm}' for arm in ARMLABELS]
        sample=d[d.start.eq('20260101')&d.window.eq('single_year')&d.startup_mode.isin(modes)]
        lines += [f'### {label}：收益 / 最大回撤','']
        table(lines,['费用/保证金',*ARMLABELS.values()],[(f'{fee*10000:.0f}bps/{margin:.0%}',*[f"{pct(g.set_index('startup_mode').loc[m,'total_return'])} / {pct(-g.set_index('startup_mode').loc[m,'max_drawdown'])}" for m in modes]) for (fee,margin),g in sample.groupby(['fee','margin'])])
    trades=pd.read_csv(OUT/'trades.csv',dtype={'start':str,'exit':str})
    t=trades[trades.start.eq('20250101')&trades.fee.eq(.0006)&trades.margin.eq(.25)].copy();t['year']=t.exit.str[:4]
    tg=t.groupby(['startup_mode','year','side','product']).agg(trades=('pnl','size'),wins=('pnl',lambda s:s.gt(0).sum()),net_pnl=('pnl','sum')).reset_index()
    tg.to_csv(OUT/'recent_continuous_sides.csv',index=False)
    lines += ['## 多空、IM与错失收益','',
        'recent_continuous_sides.csv按方案/平仓年/多空/品种分解2025启动持续账户；trades.csv保留所有交易。model_diagnostics.csv按年份/品种/方向分别报告真实新增隐波可用时的预测变化，不把回退结果当作新指标贡献。',
        'opportunity.csv分解新增空仓期间原方案盈利日的错失、亏损日的避免、新路径平仓费用及其他路径变化，四项严格核对净收益差。该归因不能当成可单独实现的盈利。','',
        '## 边界与复现','',
        '所有历史已经多轮研究，不是新的OOT；本轮固定规则后才看策略结果，未按已知2026亏损调参。不同月度窗口重叠，不能据此宣称独立统计显著性。期权覆盖改善不等于收益改善，日线收盘也不能保证盘中保证金安全。2026仅截至11月6日，单年重新启动与持续账户不可混淆。',
        '复现：fetch_index_option_history.py → build_index_option_iv.py → index_futures_option_iv.py → evaluate_index_futures_option_iv.py → report_index_futures_option_iv.py。protocol.json冻结假设；源数据分页、哈希、到期日审计、质量拒绝和预测回退全部保留。','',
        '来源：[Tushare合约资料](https://tushare.pro/document/2?doc_id=158)、[Tushare日线](https://tushare.pro/document/2?doc_id=159)、[Shibor](https://tushare.pro/document/2?doc_id=149)、[Cboe VIX方法](https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf)、[中金所股指期权细则](https://www.cffex.com.cn/u/cms/www/202207/1820311129r5.pdf)。','']
    (OUT/'report.md').write_text('\n'.join(lines))
    paths=[BASE/'entry_dataset.parquet',BASE/'entry_forecasts.parquet',QVIX/'forecasts.parquet',OUT/'indices.parquet',OUT/'forecasts.parquet',OUT/'source_hashes.json',OUT/'protocol.json',OUT/'global_calendar.csv']
    paths += [Path('src/quant/features/option_implied_volatility.py'),*[Path('scripts/research')/name for name in ['build_index_option_iv.py','index_futures_option_iv.py','evaluate_index_futures_option_iv.py']]]
    (OUT/'run_manifest.json').write_text(json.dumps(dict(versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]),indent=2))
    print('reported',len(unique),'accounts',len(d),'windows',flush=True)


if __name__=='__main__':main()
