"""Render results and the timing/sample/distribution audit together."""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
from validate_left_side_disclosure_gain import sha256

LOCAL=ROOT/'reports/research/local_ranknet_fusion_v1'
CASH=ROOT/'reports/research/ranknet_cash_policy_v1'


def table(d):
    return d.to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}',escape=True)


def main():
    for p in [LOCAL,CASH]:
        assert json.loads((p/'complete.json').read_text())['status']=='complete'
    local=pd.read_csv(LOCAL/'comparisons.csv')
    cash=pd.read_csv(CASH/'comparisons.csv')
    assert len(cash[['side','fold']].drop_duplicates())==5
    assert len(local[['side','fold']].drop_duplicates())==5
    splits=pd.read_csv(LOCAL/'oof_splits.csv')
    assert (pd.to_datetime(splits.label_end)<pd.to_datetime(splits.prediction_start)).all()
    sample=json.loads((LOCAL/'sample_audit.json').read_text())
    audit=json.loads((LOCAL/'training_audit.json').read_text())
    coverage=[]
    for path in sorted(LOCAL.glob('predictions_*_selected.parquet')):
        d=pd.read_parquet(path,columns=['date'])
        coverage.append({'artifact':path.name,'first_signal':str(d.date.min().date()),'last_signal':str(d.date.max().date()),'rows':len(d)})
    pd.DataFrame(coverage).to_csv(CASH/'test_date_coverage.csv',index=False)
    distribution=[]
    for path in sorted(LOCAL.glob('distribution_*.csv')):
        d=pd.read_csv(path);d=d.loc[d.stage.eq('test')]
        distribution.append(dict(side=d.side.iloc[0],fold=d.fold.iloc[0],audited_columns=len(d),psi_above_025=int(d.psi.gt(.25).sum()),
                                 worst_factor=d.sort_values('psi',ascending=False).factor.iloc[0],max_psi=d.psi.max(),infinite=int(d.infinite.sum())))
    dist=pd.DataFrame(distribution);dist.to_csv(CASH/'distribution_summary.csv',index=False)
    highlights=cash.loc[cash.policy.isin(['any_return','any_win','base_return','day_return','local_return','cross_return','capacity_return','forced_base'])].copy()
    for c in ['capital_pnl','win_rate','flat_days','no_entry_days','average_capital_deployed']:
        highlights[c]*=100
    columns=['side','fold','policy','arm','capital_pnl','win_rate','attempts','no_entry_days','flat_days','average_capital_deployed','unknown','vs_base_cash_capital_pp']
    highlights[columns].to_csv(CASH/'summary.csv',index=False)
    # Compare selected fusion families with the same validation-selected cash
    # gate on base50 and equal-tree-budget regression, never a best-test oracle.
    compare=[]
    for (side,fold),g in cash.groupby(['side','fold']):
        idx=g.set_index('policy')
        for policy in ['day_return','local_return','cross_return','any_return']:
            r=idx.loc[policy]
            compare.append(dict(side=side,fold=fold,policy=policy,arm=r.arm,
                                capital_pct=r.capital_pnl*100,
                                delta_vs_base_cash_pp=(r.capital_pnl-idx.loc['base_return'].capital_pnl)*100,
                                delta_vs_equal_capacity_pp=(r.capital_pnl-idx.loc['capacity_return'].capital_pnl)*100,
                                wins_pct=r.win_rate*100,flat_pct=r.flat_days*100))
    paired=pd.DataFrame(compare);paired.to_csv(CASH/'family_comparison.csv',index=False)
    cross=paired.loc[paired.policy.eq('cross_return')]
    cross_up=int(cross.delta_vs_base_cash_pp.gt(1e-8).sum())
    cross_down=int(cross.delta_vs_base_cash_pp.lt(-1e-8).sum())
    cross_capacity=int(cross.delta_vs_equal_capacity_pp.gt(1e-8).sum())
    source=ROOT/'scripts/research'
    proof={'status':'complete_research_only','production_promoted':False,
           'oof_splits':len(splits),'oof_maturity_violations':0,'right_factor_lag_violations':sample['right_screen_lag_violations'],
           'unit_tests':'10 focused tests plus 2 existing causal/point-in-time feature tests',
           'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [source/'evaluate_local_ranknet_fusion.py',source/'evaluate_ranknet_cash_policy.py',source/'audit_ranknet_training_contracts.py',Path(__file__),ROOT/'src/quant/research/local_ranknet.py']},
           'cache_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [LOCAL/'right_screen_price.parquet',LOCAL/'left_missing_price.parquet']}}
    if (CASH/'raw_price_source_hashes.json').exists():
        proof['raw_price_source_manifest_sha256']=sha256(CASH/'raw_price_source_hashes.json')
        proof['raw_price_source_files']=len(json.loads((CASH/'raw_price_source_hashes.json').read_text()))
    (CASH/'verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False))
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RankNet 融合、跨日配对与空仓策略</title>
<style>body{{font:15px/1.7 system-ui,sans-serif;color:#142236;background:#f5f7fb;max-width:1450px;margin:30px auto;padding:0 24px}}h1,h2{{line-height:1.3}}section{{background:white;border:1px solid #dfe5ee;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th,td{{padding:8px 10px;border-bottom:1px solid #e6ebf2;text-align:right}}th{{background:#eef3fa}}.notice{{background:#fff3df;border-left:5px solid #b26a00;padding:18px}}a{{color:#195db2}}code{{background:#edf1f6;padding:2px 4px}}</style>
<h1>RankNet 子分融合、跨日近分配对与允许空仓的实验</h1><p>2026-11-11 · 5 个历史诊断区间 · 129 个量价输入 · 50 主变量 + 79 剩余变量 · 未替换线上模型</p>
<p>实际可评估信号日：左侧 2024-01-02—2024-12-23、2025-01-02—2025-12-23、2026-01-05—2026-08-21；右侧 2025-07-01—2025-12-30、2026-01-05—2026-09-02。年末按计划持有期留出退出窗口；2026 不是完整年度。</p>
<section><h2>跨日实验结论</h2><p>按各自验证期选定的跨日近分方案，相对同样允许空仓的 50 变量基准，{cross_up}/5 个区间资金收益提高，{cross_down}/5 个区间降低；相对同总树数单模型，{cross_capacity}/5 个区间更高。必须结合下表中的出手数、空仓率和未知退出数量判断，不把历史单段高收益解释为稳定增益。</p></section>
<div class="notice">本报告使用修正后的样本与时点。此前 JoinQuant 增量及辅助 loss 报告存在右侧入场日盘后因子、左侧未来条件筛样问题，原有效增益结论已撤回。这里的历史区间曾被反复观察，不能充当全新前瞻留出集。未知收益按 -10% 做压力场景；这不是损失下界或可保证成交的实盘净值。</div>
<section><h2>如何对应论文和你的方案</h2><p><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf">Burges 等，Learning to Rank using Gradient Descent（2005）</a>提出成对 logistic 排序损失。这里把它用于树模型，损失为 log(1+exp(sᵢ−sⱼ))−yᵢⱼ(sᵢ−sⱼ)，使用正定对角 Hessian 上界；并非原文神经网络的逐项复刻。变量拆分、子分融合、近分邻域与现金门槛是本实验扩展。</p>
<p>主模型的 50 个变量只在最早预热训练段按增益选择；此后按时间向前生成样本外分数，训练子模型分别使用其余 79 个或全部 129 个变量。同日配对作为对照；日内近分采用百分位距离 0.05/0.10/0.20；跨日配对采用基模型预期净收益原始分差 0.0025/0.005/0.01，且明确排除同日对。邻域重叠，不做固定箱，不用真实收益分组。各交易日或锚点日损失总权重相等。</p>
<p>子模型与主模型均在验证期前完成训练；验证前半段只用于单调收益校准，后半段选择融合权重与门槛，测试期固定。门槛包含 0、0.2%、0.5%、1%、2% 和完全空仓；负无穷只作为强制出手对照。策略允许长期空仓。收益方案按验证期资金收益最大选择；胜率方案另要求至少 100 次尝试、验证收益为正且胜率不低于 55%，否则退回现金。增加了 120 棵树单模型，对照 60+60 棵树融合的模型容量。</p></section>
<section><h2>允许空仓的资金结果</h2><p>下表收益、胜率、空仓率、平均名义资金占用均为百分数。胜率分母为入场尝试（已知未成交和未知结果均不算赢），并非仅成交成功样本。capital_pnl 是区间资金收益，不是年化。左侧六组、右侧一组资金，每组最多 20 个等额槽位；不足 20 只的资金留现金。各组仅再投资自身结算资金；停牌延迟退出会阻止该组后续开仓。右侧同一收盘成交时点卖出资金可用于接续买入是日线执行近似。flat_days 按收盘无持仓计量。未知退出冻结整个组合直至评估期末，收益按 -10% 做压力估值，不能据此宣称实盘可实现。</p>
{table(highlights[columns])}<p>any_return / any_win 为验证期从所有模型族选择的方案；day/local/cross 为分别限定配对族后的验证选择；base_return 是同样允许空仓的 50 变量基准；capacity_return 是 120 棵树容量对照。不得把各测试区间表现最好的一行事后拼接。</p></section>
<section><h2>融合是否胜过公平对照</h2>{table(paired)}<p>delta_vs_base_cash_pp 与 delta_vs_equal_capacity_pp 分别表示相对允许空仓的基准、同总树数单模型的区间收益差（百分点）。置信区间见 comparisons.csv，使用固定初始名义资金每日贡献的 10 日分块重采样，不能解释为复利净值区间；未进行多重比较校正。</p></section>
<section><h2>强制日内 Top20 对照</h2>{table(local.loc[local.policy.isin(['selected','day_only','local_only','single_all']),['side','fold','policy','arm','net_delta_bps','win_delta_pp','net_ci_low_bps','net_ci_high_bps']])}<p>这一表是每个信号日 Top20 尝试的平均净收益差，不是资金收益。空仓策略的主要判断应以上表为准。</p></section>
<section><h2>样本与时点审查</h2><ol>
<li><b>已修正：右侧因子晚于决策时点。</b>旧新增因子直接按 T+1 入场日关联，含收盘后 daily_basic。新实验所有量价因子统一取 T 筛选日，早于 T+1 入场，违规行数为 {sample['right_screen_lag_violations']}；剔除 screen_to_entry_return 等入场日最终价格输入。</li>
<li><b>已修正：左侧未来条件筛样。</b>原 {sample['left_original_events']:,} 个信号只留下 {sample['left_prior_kept']:,} 个完整可成交事件；其中 794 个缺少入场行情、270 个涨停未成交、68 个未来窗口不完整。新选择池保留所有事件，已知未成交收益为零，未知结果留在选择结果中做压力处理；补齐旧缓存缺失的因子，防止缓存存在性泄漏成交结果。</li>
<li><b>已修正：左侧标签延期。</b>{audit['left_delayed_exits']:,} 个已知标签实际退出晚于固定市场日，最长 {audit['left_max_delay_days']} 天。固定日历近似会使部分未来标签越过切分。现在用真实结束日期隔离训练；验证/测试中未到期者留在候选池、收益设未知，资金仍按实际退出占用，不能预先删除。</li>
<li><b>已核验：排序、分组和融合隔离。</b>{len(splits)} 段向前 OOF 预测，训练标签结束日均早于预测段开始；日内配对不跨日，跨日配对不含同日，分差边界及梯度已测。全部变量子模型可以使用主模型变量，信息复用本身不构成穿越。</li>
<li><b>待优化：长线收益与财务口径。</b>长线股票池确实采用按日期的市值门槛、向后关联的月度质量评分；未发现直接套用当前好股票名单。但行业标签取当前静态元数据，财务只按公告日期匹配，无法证明修订前版本与日内发布时间；行情/标签部分使用原始 close 比值，需要统一除权收益。固定 27 周也不能替代实际 126 个交易日及延迟入场的结束日期。已检查本轮旧年度切分未见该近似造成越界，仍应将真实结束日期写入标签合同。</li>
<li><b>待补数据：来源覆盖不齐。</b>资产负债表只覆盖 3,000 个代码且最新期为 2026Q1，收入/现金流较新；不能无审查地加入全部财务因子。长线 2026 年 26 周标签大量未成熟，不能据此做可靠年度评估。本轮不将这些长线结论或财务候选提升为可上线证明。</li>
</ol></section>
<section><h2>变量分布、冗余与优化优先级</h2>{table(dist)}<p>PSI 阈值 0.25 仅作为分布变化提示；分箱边界完全来自训练期，测试统计没有反向参与本轮特征选择。部分离散特征无法形成足够分箱，未列入 PSI 分母。原始输入共 129 列。2026 年训练段高相关变量对（|Spearman|&gt;0.995）左侧约 176 对、右侧约 138 对，主要为价格水平/均线等。因此“剩余变量”不等于独立增量；优先按变量族去冗余、将价格水平改为相对价格/波动标准化，再在新的向前留出段重验。月线 KDJ、长均线斜率和部分 MFI 发生明显漂移，需校准监控。长线 2025 年有 7 个分析师预测变量完全缺失，已有覆盖门槛应继续按训练期执行。</p>
<p>上线前优先顺序：真实成交与停牌资金账本 → 财报/行业历史版本与除权口径 → 独立前瞻或滚动盲测 → 按变量族消融和分布校准。可以长期空仓，但必须同时观察资金收益、成交数量、未知结果和空仓天数，不能单看胜率。</p></section>
<section><h2>可追溯文件</h2><ul><li><a href="comparisons.csv">全部验证选定策略的测试指标</a> · <a href="validation_grid.csv">完整候选网格</a> · <a href="family_comparison.csv">模型族对照</a></li><li><a href="../local_ranknet_fusion_v1/oof_splits.csv">OOF 时序切分</a> · <a href="../local_ranknet_fusion_v1/financial_source_audit.csv">财务源覆盖</a> · <a href="../local_ranknet_fusion_v1/long_feature_coverage.csv">长线变量覆盖</a></li><li><a href="protocol.json">实验协议</a> · <a href="verification.json">校验与源码哈希</a></li></ul><p>模型、分数、所选交易和每日贡献均保存在报告目录。策略线上模型未变更。新增脚本执行旧实验时默认拒绝重现存在时点问题的结果，显式 --allow-legacy-timing 仅用于追溯。</p></section></html>'''
    (CASH/'report.html').write_text(text)
    (LOCAL/'report.html').write_text('<meta charset="utf-8"><a href="../ranknet_cash_policy_v1/report.html">查看排序融合与训练样本完整审查报告</a>')
    summary={'selected':cash.loc[cash.policy.isin(['any_return','any_win'])].to_dict('records'),'production_promoted':False}
    (CASH/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
