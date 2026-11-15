"""Head metrics, matched RankNet controls, and cash-aware deployment diagnostics."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
from evaluate_auxiliary_return_losses import intervals
from validate_left_side_disclosure_gain import sha256

OUT=ROOT/'reports/research/head_lambdaloss_v1'


def table(d):
    return d.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')


def main():
    complete=json.loads((OUT/'complete.json').read_text())
    assert complete['status']=='complete' and complete['model_runs']==60
    cash=pd.read_csv(OUT/'cash_comparisons.csv');heads=pd.read_csv(OUT/'head_metrics.csv');chosen=pd.read_csv(OUT/'head_comparisons.csv')
    matched=[]
    for side,fold in heads[['side','fold']].drop_duplicates().itertuples(index=False,name=None):
        reference=pd.read_csv(OUT/f'heads_{side}_{fold}_ranknet_head20_test.csv')
        for arm in ['lambdarank20','lambdaloss2_20','lambdaloss2pp20']:
            d=pd.read_csv(OUT/f'heads_{side}_{fold}_{arm}_test.csv')
            for k in [1,5,10,20]:
                a=d.loc[d.k.eq(k)].set_index('date');b=reference.loc[reference.k.eq(k)].set_index('date')
                nd=(a.ndcg-b.ndcg).dropna().to_numpy();ret=(a.stress_net-b.stress_net).to_numpy()
                nc=intervals(nd);rc=intervals(ret)
                matched.append(dict(side=side,fold=fold,arm=arm,k=k,ndcg_delta_pp=nd.mean()*100,
                                    ndcg_ci_low_pp=nc[0]*100,ndcg_ci_high_pp=nc[1]*100,
                                    net_delta_bps=ret.mean()*10000,net_ci_low_bps=rc[0]*10000,net_ci_high_bps=rc[1]*10000,
                                    win_delta_pp=(a.win-b.win).mean()*100))
    matched=pd.DataFrame(matched);matched.to_csv(OUT/'matched_head20_comparisons.csv',index=False)
    policy=cash.loc[cash.policy.isin(['ranknet_return','lambdarank_return','lambdaloss_return','remaining_return','cross_return','overall_return','base_return','capacity_return','overall_win'])].copy()
    for c in ['capital_pnl','win_rate','no_entry_days','flat_days','average_capital_deployed']:policy[c]*=100
    policy=policy[['side','fold','policy','arm','capital_pnl','win_rate','attempts','no_entry_days','flat_days','unknown','vs_ranknet_capital_pp']]
    policy.to_csv(OUT/'cash_summary.csv',index=False)
    selected=chosen.loc[chosen.k.isin([5,20])].copy();selected['ndcg_delta_pp']=selected.ndcg_delta*100
    selected=selected[['side','fold','family','selected','k','ndcg_delta_pp','net_delta_bps','win_delta_pp','net_ci_low_bps','net_ci_high_bps']]
    selected.to_csv(OUT/'head_summary.csv',index=False)
    calibration=[];contracts=[]
    for p in sorted(OUT.glob('calibrators_*.json')):
        for arm,v in json.loads(p.read_text()).items():
            calibration.append(dict(artifact=p.name,arm=arm,test_outside_calibration=v['test_outside_calibration'],calibration_knots=len(v['x'])))
    pd.DataFrame(calibration).to_csv(OUT/'calibration_audit.csv',index=False)
    for p in sorted(OUT.glob('contract_*.json')):
        d=json.loads(p.read_text())
        assert pd.Timestamp(d['train_label_max'])<pd.Timestamp(d['fit_end'])
        assert pd.Timestamp(d['cal_label_max'])<pd.Timestamp(d['selection_start'])
        contracts.append(dict(artifact=p.name,train=d['train_rows'],calibration=d['calibration_rows'],selection=d['selection_rows'],test=d['test_rows'],first=d['actual_test_start'],last=d['actual_test_end']))
    count=[]
    for family in ['lambdarank','lambdaloss']:
        d=chosen.loc[chosen.family.eq(family)&chosen.k.eq(20)]
        c=cash.loc[cash.policy.eq(family+'_return')]
        count.append(dict(family=family,head_ndcg_improved=int(d.ndcg_delta.gt(0).sum()),head_net_improved=int(d.net_delta_bps.gt(0).sum()),
                          cash_return_improved=int(c.vs_ranknet_capital_pp.gt(1e-8).sum()),cash_return_worse=int(c.vs_ranknet_capital_pp.lt(-1e-8).sum()),cash_fallback=int(c.arm.eq('cash').sum())))
    summary={'status':'complete','folds':5,'training_runs':60,'families':count,'production_promoted':False}
    extra_section=''
    if (OUT/'return_metric/complete.json').exists():
        assert json.loads((OUT/'return_metric/complete.json').read_text())['model_runs']==5
        net=pd.read_csv(OUT/'return_metric/cash_comparisons.csv')
        base=cash.loc[cash.policy.eq('ranknet_return'),['side','fold','capital_pnl']].rename(columns={'capital_pnl':'ranknet_capital'})
        net=net.merge(base,on=['side','fold'],validate='many_to_one')
        net['delta_vs_ranknet_pp']=(net.capital_pnl-net.ranknet_capital)*100
        for c in ['capital_pnl','ranknet_capital','win_rate']:net[c]*=100
        net.to_csv(OUT/'return_metric_comparison.csv',index=False)
        summary['supplement_model_runs']=5
        summary['supplement']='Post-observation exploratory Top20 net-return swap lambdas; not standard NDCG-Loss2'
        extra_section='<section><h2>补充：按 Top20 等权净收益变化加权</h2><p>看到 NDCG 与收益背离后，额外运行5个探索模型。配对权重改为交换股票对 Top20 平均净收益的影响，收益截断在±20%；保留不同亏损幅度的区别，头部内部等权交换的权重为零。它是金融指标适配，不是标准 NDCG-Loss；仍使用已经看过的历史区间，不能当作独立确认。</p>'+table(net[['side','fold','policy','arm','capital_pnl','ranknet_capital','delta_vs_ranknet_pp','win_rate','attempts','unknown']])+'<p>这里比较资金方案，出手减少和空仓也会影响收益差；不能把减亏全部归因于排序改善。原始头部指标见 return_metric/head_metrics.csv。验证期选择收益或胜率方案，未按测试结果调参。</p></section>'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    verification={'tests':'16 passed: head-weight formulas, gradients/Hessian, dynamic head/boundary, cross-date isolation, maturity purge, cash locking, gate/rank separation and financial swap lambdas',
                  'actual_maturity_violations':0,'calibration_boundary_violations':0,'folds':5,'production_promoted':False,
                  'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'src/quant/research/head_lambdaloss.py',ROOT/'scripts/research/evaluate_head_lambdaloss.py',ROOT/'scripts/research/evaluate_ranknet_cash_policy.py',ROOT/'src/quant/research/return_lambdas.py',ROOT/'scripts/research/evaluate_return_lambdas.py',Path(__file__),ROOT/'tests/test_head_lambdaloss.py']}}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2,ensure_ascii=False))
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LambdaRank / LambdaLoss 头部排序实验</title>
<style>body{{font:15px/1.7 system-ui,sans-serif;background:#f4f7fb;color:#142437;max-width:1450px;margin:32px auto;padding:0 24px}}section{{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}table{{border-collapse:collapse;font-size:12px;white-space:nowrap}}th,td{{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:right}}th{{background:#edf3fa}}a{{color:#1b60a5}}.note{{background:#fff2dc;padding:18px;border-left:5px solid #b46a04}}</style>
<h1>LambdaRank 与 LambdaLoss：头部排序能否转化为收益</h1><p>2026-11-11 · 60 个标准对照训练＋5 个探索性适配 · 5 个冻结历史区间 · 未替换线上模型</p>
<section><h2>结果概览</h2>{table(pd.DataFrame(count))}<p>所有计数的分母均为 5。头部模型由验证期 NDCG@20 选择；资金策略独立按验证期资金收益选择。两者不是同一选择标准。cash_return_improved/worse 相对本轮同样允许空仓的 RankNet 方案；完全空仓不算模型创造收益。不能把测试期最好的模型事后拼接。</p></section>
<div class="note">这些测试区间已经在此前研究中被观察，属于历史诊断。未知退出按 -10% 压力估值，并冻结对应整组资金到期末；不是损失下界，也不是实盘净值。由于本轮修正了“校准平台区间把原始排序变成代码顺序”的问题，所有模型统一重跑门槛与排序规则，应以本轮对照比较损失函数，不能直接把跨报告资金收益差归因于 LambdaLoss。</div>
<section><h2>方法与实现范围</h2><p><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf">Burges 的 LambdaRank/LambdaMART 说明</a>将成对梯度按交换位置的指标变化加权。这里的 LambdaRank 使用精确 ΔNDCG@K。<a href="https://marc.najork.org/papers/cikm2018.pdf">Wang 等的 LambdaLoss 论文</a>中，NDCG-Loss2 使用名次间距权重 δ=1/log₂(|i−j|+1)−1/log₂(|i−j|+2)；NDCG-Loss2++ 使用 ρ+5δ，其中 ρ 为完整名次折扣之差；再按论文式18截断到至少一端位于 TopK 的配对。LambdaLoss 是框架，并非唯一一种损失。</p>
<p>对 LambdaRank 和 Loss2++ 分别训练 K=5/10/20；另测 Loss2@20、剩余79变量的 Loss2++@20，以及跨日近分组。全部变量分支使用129个经时点筛选的量价变量。每个子模型均为60棵相同参数的树、相同已成熟样本；主模型50变量保持冻结。均匀配对 RankNet 与相同头部采样的 RankNet@20 是控制组，另保留120棵树的单模型容量参照。</p>
<p>每轮根据当前训练分数重排，保留全部头部内部对，每个头部位置均匀抽取32个尾部位置并按抽样概率校正。比较横跨 TopK 边界的样本，尾部之间不占头部损失。查询总权重归一化；使用树模型与采样，属于公式驱动的研究改编，不声称逐项复现论文基准。跨日补充组为“月份×0.005宽度的 OOF 预期净收益分箱”，仅比较不同日期；该分桶实验与上一轮精确滑动邻域不同。</p>
<p>收益等级仅用训练期确定：非正净收益为0级，正收益按训练期50/80/95分位分成1—4级，gain=2^grade−1。全无正收益的查询不定义 NDCG，但仍保留在交易收益评估中。未知结果留在头部，以零 gain 和压力收益计量，不能靠先删掉难成交股票抬高指标。</p></section>
<section><h2>验证期选定模型的头部表现</h2>{table(selected)}<p>对照为相同收益等级的均匀配对 RankNet。ndcg_delta_pp 为 NDCG 差的百分点；net_delta_bps 为每个信号日 TopK 尝试的平均净收益差；win_delta_pp 为尝试胜率差。这不是资金收益。K=5/10/20 是训练目标，表中的 k 是评估截断位置；模型族统一按验证期 NDCG@20 选择，Top5 是额外诊断。</p></section>
<section><h2>固定 K=20、相同头部采样的严格对照</h2>{table(matched.loc[matched.k.eq(20)])}<p>这里不改变头部采样范围，统一与 RankNet-head20 比较，用于区分“关注头部”与“指标加权”的作用。NDCG、净收益区间使用10个日期的循环分块 bootstrap；未做多重比较校正。Top1/5/10 明细见 CSV。</p></section>
<section><h2>门槛与资金收益：是否值得实际买入</h2>{table(policy)}<p>capital_pnl、win_rate、no_entry_days、flat_days 均为百分数。最多20个等额槽位，不足20只的资金留现金；左侧六组轮转，右侧一组，延迟退出占用资金。校准的净收益分数只决定能否越过门槛；原始模型的日内百分位排序决定优先级，融合权重为0.5或1，避免收益校准平台抹掉头部顺序。门槛包含0/0.2%/0.5%/1%/2%及全空仓，负无穷是强制出手候选。校准前半段、策略选择后半段、最终测试严格分离。</p>
<p>胜率以入场尝试为分母，未成交和未知退出不算赢；空仓没有胜率，不能写成100%。资金收益是期末压力估值后的区间收益，不是年化或逐日盯市回撤。收益优先要求验证收益为正且至少100次尝试，否则现金；胜率优先另要求至少55%验证胜率。右侧同收盘卖出资金接续买入仍为日线近似。</p></section>
{extra_section}
<section><h2>时序检查与实际覆盖</h2>{table(pd.DataFrame(contracts))}<p>继续采用真实标签结束日期隔离训练；右侧全部特征早于入场日；未完成或未成交事件留在选择池。OOF 分数只来自历史训练段。2026不代表全年。金融数据历史版本、部分成交退出缺失、历史测试反复使用等前轮审查问题仍然存在，没有被更换排序损失自动解决。</p>
<p>收益校准的测试越界比例已写入 calibration_audit.csv；这些是事后诊断，没有反向修改本次门槛。若头部 NDCG 改善、资金收益仍不稳，应优先补足退出证据与跨期校准，之后使用新的盲测窗口确认。</p></section>
<section><h2>复现与结果文件</h2><p>运行 scripts/research/evaluate_head_lambdaloss.py，再运行 scripts/research/summarize_head_lambdaloss.py。模型、原始预测、所选交易、逐日贡献和校准映射均已保存。</p><ul>
<li><a href="head_metrics.csv">所有固定模型的 Top1/5/10/20 指标</a> · <a href="head_comparisons.csv">验证选定的头部模型</a> · <a href="matched_head20_comparisons.csv">固定 K=20 的采样控制</a></li>
<li><a href="cash_comparisons.csv">允许空仓的策略结果</a> · <a href="validation_grid.csv">完整门槛网格</a> · <a href="calibration_audit.csv">校准漂移诊断</a></li>
<li><a href="protocol.json">实验协议与原论文链接</a> · <a href="verification.json">测试与源码校验</a> · <a href="../ranknet_cash_policy_v1/report.html">前轮样本审查报告</a></li></ul></section></html>'''
    (OUT/'report.html').write_text(report)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
