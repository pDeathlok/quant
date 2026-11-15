"""Summarize held-out policies and compare with the previous factor experiment."""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'scripts/research')]
from quant.research.auxiliary_return_loss import daily_picks
from evaluate_auxiliary_return_losses import intervals
from validate_left_side_disclosure_gain import sha256
OUT=ROOT/'reports/research/auxiliary_return_loss_v1'


def main() -> None:
    assert json.loads((OUT/'complete.json').read_text())['status']=='complete'
    m=pd.read_csv(OUT/'metrics.csv')
    c=pd.read_csv(OUT/'comparisons.csv')
    selected=c.loc[~c.policy.str.startswith('fixed:')].copy()
    old=[]
    for row in selected.to_dict('records'):
        side,fold,policy=row['side'],str(row['fold']),row['policy']
        new=pd.read_parquet(OUT/f'predictions_{side}_{fold}_{policy}.parquet')
        previous=pd.read_parquet(ROOT/f'reports/research/joinquant_candidates_v1/predictions_{side}_{fold}_all.parquet',columns=['symbol','date','score']).rename(columns={'score':'previous_score'})
        data=new.merge(previous,on=['symbol','date'],how='left',validate='one_to_one')
        if data.previous_score.isna().any():raise ValueError('Previous experiment keys mismatch')
        _,before=daily_picks(data,data.previous_score.to_numpy())
        _,after=daily_picks(data,data.score.to_numpy())
        valid=before.net.notna()&after.net.notna()
        delta=(after.loc[valid,'net']-before.loc[valid,'net']).to_numpy()
        win=(after.loc[valid,'win']-before.loc[valid,'win']).to_numpy()
        ci=intervals(delta) if side!='long' else [np.nan,np.nan]
        old.append(dict(side=side,fold=fold,policy=policy,arm=row['arm'],paired_dates=int(valid.sum()),
                        net_delta_bps=delta.mean()*10000,win_delta_pp=win.mean()*100,
                        ci_low_bps=ci[0]*10000,ci_high_bps=ci[1]*10000,
                        stress_delta_bps=(after.stress-before.stress).mean()*10000))
    old=pd.DataFrame(old);old.to_csv(OUT/'previous_experiment_comparison.csv',index=False)
    robustness=[]
    for (side,policy),g in selected.groupby(['side','policy']):
        robustness.append(dict(side=side,policy=policy,folds=len(g),
                               positive_return_folds=int(g.net_delta_bps.gt(0).sum()),
                               positive_win_folds=int(g.win_delta_pp.gt(0).sum()),
                               both_positive_folds=int((g.net_delta_bps.gt(0)&g.win_delta_pp.gt(0)).sum()),
                               mean_fold_net_delta_bps=g.net_delta_bps.mean(),
                               mean_fold_win_delta_pp=g.win_delta_pp.mean()))
    pd.DataFrame(robustness).to_csv(OUT/'robustness.csv',index=False)
    protocol=json.loads((OUT/'protocol.json').read_text())
    paths=[ROOT/'data/research/left_side_disclosure_gain_20261029/events.parquet',
           ROOT/'data/research/left_side_disclosure_gain_20261029/labels.parquet',
           ROOT/'reports/research/right_t012_full_dataset_v4/dataset.parquet',
           ROOT/'data/features/long_entry/weekly_training_v2.parquet',
           ROOT/'src/quant/research/auxiliary_return_loss.py',
           ROOT/'scripts/research/evaluate_auxiliary_return_losses.py']
    protocol['input_and_code_hashes_at_completion']={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    notes=[
        '共49次训练，所有辅助目标真正加入同一个XGBoost原始分数的优化；同样的输入、树深、80轮训练、随机种子与选股数量。',
        'L=(1+λD×亏损严重度)×BCE(盈利标签,sigmoid(z))+λR/2×(z−标准化净收益)²+λI/2×(z−标准化相对收益)²。收益尺度只从训练期估计，回归目标截断至±3，结果按原始未截断收益评估。',
        '相对收益=个股净收益−同日训练候选池平均净收益，是训练标签，绝不作为预测输入。它不是相对旧模型的因果收益增量；旧模型增量单独用测试期成对比较。',
        '主任务盈利标签=扣费后收益>0，区别于旧模型的good_path5。联合分数不作为已校准概率使用。',
        '辅助权重仅按测试前验证期选择。兼顾方案要求验证期收益和胜率都不低于win基线；收益优先容许验证期胜率下降最多2个百分点；胜率优先要求验证期收益不下降。允许回退win基线。',
        '短线指标是每日Top20交易集合的平均单笔净收益及等日期加权胜率，不能当作组合净值或年化收益；左侧5日，右侧T012约1日，不能直接横向比较。',
        '长线改用真实26周个股收益减假设20bp成本，不再把排序效用称为收益。27周时间隔离，只评2024/25；选股池与历史版本限制继承原研究。重叠周期很长，长线不提供显著性区间。',
        '未知退出保留参与排名。验证选择将未知结果按−10%及失败处理；测试同时给双方完整日期的成对结果与−10%压力结果。',
        '短线区间按10个观测日期的循环区块bootstrap估计，未做多重检验校正；大量历史研究已复用这些诊断期，需要新数据确认，不自动上线。',
        '旧实验对照更换了算法和训练目标，不能将其全部变化归因于辅助loss；同架构win-only对照才用于隔离辅助loss效果。',
    ]
    title='辅助收益损失实验 · 2026-11-11'
    text='<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:15px system-ui;max-width:1400px;margin:35px auto;padding:20px;color:#193044}p{line-height:1.8}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:7px}th{background:#eff5f9}h2{margin-top:35px}</style><h1>'+title+'</h1>'
    text+=''.join('<p>'+html.escape(n)+'</p>' for n in notes)
    for name,frame in [('验证期选定方案的测试表现',selected),('跨折稳定性',pd.DataFrame(robustness)),('相对上一轮全量因子回归模型',old),('所有固定损失的测试对照',c),('完整指标（ES5=最差5%选中交易均值）',m)]:
        text+='<h2>'+name+'</h2>'+frame.to_html(index=False,float_format=lambda x:f'{x:.5f}')
    (OUT/'report.html').write_text(text)
    (OUT/'summary.json').write_text(json.dumps({'models':len(m),'robustness':robustness,'production_promoted':False},indent=2))


if __name__=='__main__':main()
