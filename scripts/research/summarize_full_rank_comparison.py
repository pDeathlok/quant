"""Readable, scope-explicit results; never promote the best historical test arm."""
from __future__ import annotations
import html
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/research')]
from build_full_rank_dataset import OUT
from evaluate_auxiliary_return_losses import intervals


def main():
    complete=json.loads((OUT/'complete.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['passed'], 'Independent artifact audit must pass before reporting'
    meta=json.loads((OUT/'dataset_complete.json').read_text())
    cash=pd.read_csv(OUT/'cash_comparisons.csv')
    audit=pd.read_csv(OUT/'column_audit.csv')
    admission=pd.read_csv(OUT/'fold_admission.csv')
    heads=pd.read_csv(OUT/'head_metrics.csv')
    readiness=json.loads((OUT/'readiness_audit.json').read_text())
    recovery=json.loads((OUT/'source_recovery_audit.json').read_text())
    families=['base50','core129','original_full','full_regression','full_capacity120','remaining_regression',
              'ranknet_family','lambdarank_family','lambdaloss_family','cross_family','aux_absolute_increment','overall']
    labels=dict(base50='原50列基准',core129='原129列对照',original_full='原776可用列+19聚宽',full_regression='扩展全量回归',
                full_capacity120='扩展全量120树',remaining_regression='剩余变量回归融合',ranknet_family='RankNet/近分组融合',
                lambdarank_family='LambdaRank融合',lambdaloss_family='LambdaLoss融合',cross_family='跨日近分组融合',
                aux_absolute_increment='绝对/增量收益辅助loss',overall='验证期总选择')
    rows=[]
    for side,fold in cash[['side','fold']].drop_duplicates().itertuples(index=False):
        for family in families:
            r=cash.loc[cash.side.eq(side)&cash.fold.eq(fold)&cash.policy.eq(family+'_return')].iloc[0]
            rows.append({'方向':'左侧' if side=='left' else '右侧','区间':fold,'方案':labels[family],
                '资金收益':f'{r.capital_pnl:.2%}','成交胜率':f'{r.filled_win:.2%}' if r.filled_attempts else '—',
                '尝试/成交':f'{int(r.attempts)}/{int(r.filled_attempts)}','退出未知':int(r.unknown),
                '空仓日占比':f'{r.flat_days:.1%}','对129列收益差pp':round(r.versus_core_capital_pp,2),
                '验证选定方案':r.selected})
    table=pd.DataFrame(rows);table.to_csv(OUT/'result_summary.csv',index=False)
    win_table=cash.loc[cash.policy.isin([f'{f}_win' for f in ['core129','original_full','full_regression','ranknet_family','lambdarank_family','lambdaloss_family','cross_family','overall']]),
        ['side','fold','policy','selected','capital_pnl','filled_win','filled_attempts','unknown','flat_days']].copy()
    for c in ['capital_pnl','filled_win','flat_days']:
        win_table[c]=win_table[c].map(lambda v:f'{v:.2%}' if pd.notna(v) else '—')
    scope=admission.groupby(['side','fold']).agg(候选列=('name','size'),实际输入列=('admitted','sum'))
    scope['原776实际输入']=admission.assign(used_original=admission.admitted&admission.original_776).groupby(['side','fold']).used_original.sum()
    scope=scope.reset_index()
    blocked=audit.loc[~audit.admitted].groupby('exclusion_reason').size().rename('列数').reset_index()
    drift=[]
    for p in OUT.glob('distribution_*.csv'):
        if p.name=='distribution_summary.csv':
            continue
        s=p.stem.split('_');d=pd.read_csv(p)
        for stage,g in d.groupby('stage'):
            drift.append(dict(side=s[1],fold=s[2],stage=stage,columns=len(g),psi_over025=int(g.psi.gt(.25).sum()),coverage_below_half=int(g.coverage.lt(.5).sum())))
    drift=pd.DataFrame(drift);drift.to_csv(OUT/'distribution_summary.csv',index=False)
    calibration=[]
    for path in OUT.glob('calibrators_*.json'):
        _,side,fold=path.stem.split('_')
        for arm,value in json.loads(path.read_text()).items():
            calibration.append(dict(side=side,fold=fold,arm=arm,outside_calibration=value['test_outside_calibration']))
    calibration=pd.DataFrame(calibration);calibration.to_csv(OUT/'calibration_range_summary.csv',index=False)
    comparison=[]
    for side,fold in cash[['side','fold']].drop_duplicates().itertuples(index=False):
        for arm in ['original_full','full_regression','ranknet_all','local05_all','lambdarank20_all','lambdaloss2_all','lambdaloss2pp_all','aux_absolute_increment']:
            a=pd.read_csv(OUT/f'head_{side}_{fold}_{arm}_test.csv').query('k == 20').set_index('date')
            b=pd.read_csv(OUT/f'head_{side}_{fold}_core129_test.csv').query('k == 20').set_index('date')
            difference=(a.stress_net-b.stress_net).to_numpy();ci=intervals(difference)
            comparison.append(dict(side=side,fold=fold,arm=arm,net_delta_bps=difference.mean()*10000,
                win_delta_pp=(a.win-b.win).mean()*100,ndcg_delta=(a.ndcg-b.ndcg).mean(),low_bps=ci[0]*10000,high_bps=ci[1]*10000))
    comparison=pd.DataFrame(comparison);comparison.to_csv(OUT/'top20_comparison.csv',index=False)
    positive=cash.loc[cash.policy.eq('overall_return'),['side','fold','selected','capital_pnl','filled_win','attempts','unknown']]
    evidence=cash.loc[cash.policy.isin(['core129_return','full_regression_return','ranknet_family_return','overall_return']),
        ['side','fold','policy','selected','capital_pnl','filled_win','filled_attempts','flat_days']]
    headline=table.loc[table['方案'].isin([labels[x] for x in ['core129','full_regression','ranknet_family','overall']])]
    production_config=yaml.safe_load((ROOT/'configs/strategies/left_side_unified.yaml').read_text())
    production_manifest=json.loads((ROOT/production_config['model']['artifact_manifest']).read_text())
    import evaluate_local_ranknet_fusion as prior
    actual_hash=prior.sha256(ROOT/production_config['model']['artifact'])
    assert actual_hash==production_manifest['artifact_sha256']
    left_evidence=cash.loc[cash.side.eq('left')&cash.policy.eq('ranknet_family_return'),
        ['fold','selected','capital_pnl','filled_win','versus_core_capital_pp','daily_delta_ci_low_bps','daily_delta_ci_high_bps']]
    promotion_review=dict(research_candidate='left_local10_ranknet',research_decision='advance_as_replacement_candidate',
        incumbent_release=production_manifest['release_id'],incumbent_inputs=len(production_manifest['feature_names_in']),
        incumbent_artifact=production_config['model']['artifact'],incumbent_sha256=actual_hash,
        comparison_baseline='core129_retrained_research_control_not_live_incumbent',
        absolute_profit_required_for_relative_replacement=False,
        outstanding=['time-correct paired comparison against actual incumbent methodology',
                     'live full-feature parity and deployable fit/calibration bundle'],
        evidence=left_evidence.to_dict('records'))
    (OUT/'promotion_review.json').write_text(json.dumps(promotion_review,ensure_ascii=False,indent=2))
    decision=dict(status='research_iteration_complete_no_production_promotion',
        reason='Left RankNet advances as replacement candidate; core129 is not the actual live 178-input incumbent. Paired incumbent comparison and live feature parity remain outstanding.',
        research_candidate='left_local10_ranknet',absolute_profit_required_for_relative_replacement=False,
        production_promoted=False, allowed_indefinite_cash=True,
        audit_passed=True, evidence=evidence.replace({np.nan:None}).to_dict('records'))
    (OUT/'iteration_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    right=cash.loc[cash.side.eq('right')&cash.fold.eq('2026')&cash.policy.eq('overall_return')].iloc[0]
    right_rank=cash.loc[cash.side.eq('right')&cash.fold.eq('2026')&cash.policy.eq('ranknet_family_return')].iloc[0]
    right_note=(f'右侧2026验证期总选择为{right.selected}，资金压力收益{right.capital_pnl:+.2%}、成交胜率{right.filled_win:.2%}，'
        f'{int(right.filled_attempts)}笔成交中含{int(right.unknown)}笔未知退出；延期或未知退出冻结资金组，阻止后续{int(right.delayed_capital_blocked_attempts)}次候选买入。'
        f'相对129列对照（该期现金）的日度收益差95%区块区间为{right.daily_delta_ci_low_bps:+.2f}至{right.daily_delta_ci_high_bps:+.2f}bp。'
        f'同期RankNet家族为{right_rank.capital_pnl:+.2%}、胜率{right_rank.filled_win:.2%}。\n')
    text=[
        '# 全量变量、排序损失与长期空仓对比\n',
        f'已完成 {complete["model_runs"]} 个主模型，以及10个按季度滚动的OOF基模型。未替换线上模型。所有方案使用相同候选样本、成本、资金占用和校准/选择窗口。\n',
        '## 本轮迭代结论\n',
        '本轮完成研究对比。左侧近分组RankNet两期的收益和胜率均优于129列研究对照，应推进为优先替换候选；绝对收益仍亏损不构成否决相对替换的充分理由。当前线上实际为v4、178个输入列，129列对照不是线上旧模型，因此暂未修改线上指向，需补实际线上方法的同口径时序对照与部署特征一致性。详见promotion_review.json。\n',
        '2026左侧相对129列的日度收益差95%区块区间为正，2025H2仍跨零，不能声称两期均统计显著；区间是固定初始名义资金的日度差，不是复合资金收益的置信区间。Lambda损失与辅助loss没有体现出统一优势；右侧不能用少量交易的高胜率或空仓零收益证明选股能力。\n',
        right_note,
        headline.drop(columns='验证选定方案').to_markdown(index=False)+'\n',
        '保留验证期确定门槛、无合格信号则长期空仓的机制。研究候选与现有线上模型分开存档，本轮也没有重新认证旧模型的上线资格。需要补的是正确的替换对象和可运行的生产特征契约，而不是要求新模型必须绝对盈利。具体决策及证据见iteration_decision.json。\n',
        f'审计范围为原776列 + 新注册341列指数状态 + 19列聚宽价格候选，共{len(audit)}列。原776中准入{meta["original_776_admitted"]}列，另有341指数列和19聚宽列，合计{meta["admitted"]}列进入逐折可用性筛查。全量指通过时点审计的候选全集；没有把被隔离的列塞进模型或用新增列冒充原776列。\n',
        scope.to_markdown(index=False)+'\n',
        '## 实际回测口径\n',
        '收益为固定资金槽位、计入原标签20bp往返成本后的期末资金压力收益；不是逐日盯市净值，也不是实盘已实现收益。左侧6个轮转资金组，每组最多20只；右侧1组最多20只。空位留现金，延期退出锁定资金，未知退出按−10%压力估值并冻结所在资金组；−10%并非保证损失下界。成交胜率分母包括未知退出的已假定成交尝试，未知计非胜；未成交不进入成交胜率。选择策略时使用的尝试胜率还包括未成交。\n',
        '2025H2测试为7—12月；2026测试截止9月3日，左侧实际信号截至8月21日，右侧截至9月2日。保留最后信号后的执行/结算日历；数据结束后的无新信号区间不能当作模型主动择时证据。右侧收盘卖出后同价再买是执行代理假设。\n',
        '训练从2024年开始；OOF从2024Q2开始，每季度仅用更早、且标签已实际到期的数据训练。各折只用成熟OOF样本训练所有待比较模型。验证前半期校准净收益，后半期选择融合权重和阈值，允许选择无限期现金。测试不重新调参。这些历史区间此前已被研究过，不是新的独立样本外确认。\n',
        '## 验证期按收益选定后的测试结果\n',
        table.drop(columns='验证选定方案').to_markdown(index=False)+'\n',
        '每个家族均可选择现金。不能将“少交易后的高胜率”单独视作增益；需同时看收益、成交数、资金使用和未知退出。全部权重、门槛和样本量见 result_summary.csv、cash_comparisons.csv；按胜率选定的版本也保存在 cash_comparisons.csv（policy后缀_win）。\n',
        '## 验证期优先胜率的版本\n',
        '选择条件为验证期至少100次尝试、资金收益为正且尝试胜率≥55%；无合格方案则全现金。测试结果如下，不能用测试结果再改55%门槛。\n',
        win_table.to_markdown(index=False)+'\n',
        '## 无门槛Top20排序增量\n',
        '下表对照同口径129列模型，净收益差按日等权；95%区间用10日区块、1500次重采样计算，未对多次实验校正。NDCG改善不能替代收益改善。\n',
        comparison.round(4).to_markdown(index=False)+'\n',
        'RankNet比较实际净收益大小，保留亏损之间的次序；LambdaRank/LambdaLoss使用训练期正收益分位数等级，非正收益等级0。另设ranknet_head20，在同一等级、同一头部采样下对照。头部损失均为K=20，LambdaLoss含Loss2和Loss2++（mu=5）；尾部每头部点抽样32个位置并做抽样概率修正。每轮按当前预测重排。\n',
        '辅助loss为胜负交叉熵 + 0.25×绝对净收益误差 + 0.25×相对当日均值的收益增量误差，并提高亏损样本的分类权重；尺度仅用训练期估计。该“增量”指同日均值基准，不是基础模型的预测残差。\n',
        '同日近分组使用OOF日内分位差≤0.05/0.10；跨日RankNet使用OOF预期净收益差≤25/50/100bp。跨日LambdaLoss采用月份×50bp分箱，且仅比较不同日期；这是单独的分组实验，不能把其分数单位与0.05分位差混为一谈。融合以校准收益决定是否开仓，以原始排序分位融合决定优先级，避免等值校准分数按股票代码打破平局。\n',
        '## 时点与样本审计\n',
        '- 所有特征按信号日或更早来源连接；没有使用入场日收盘产生的新指标、未来筛选后的历史特征表、screen_to_entry_return或未成熟标签。左侧按真实退出日剔除训练边界交叉样本；测试保留延迟退出和未知结果。\n',
        f'- 本次研究数据路径将周/月KDJ的整段历史长度门槛改成逐行已完成周期数，修正{sum(x["changed_cells"] for x in readiness["rows"])}个非空单元，并重建全部OOF分数。公共计算器和线上模型尚未迁移；不要直接复用旧训练缓存。\n',
        '- 新规则列做完整历史与截断历史的前缀不变性检查；数组加速的27列左侧规则逐列对照原计算器。指数状态屏蔽未公开发布的历史回溯值。所有训练列的非空率、唯一值、分位数和PSI按折保留，准入只依赖训练期。\n',
        f'- 资金流原始数据发现{recovery["moneyflow"]["conflicting_keys"]}个值冲突的重复键，未任选一个版本：对应观测及受影响滚动窗口置缺失。融资融券变化率禁止隐式向前填值。两类数据只使用严格更早、最多7天的观测。\n',
        f'- 研报源有{recovery["analyst"]["snapshot_only_rows"]:,}条snapshot_only记录；更关键的是，具体研报也会按当前长线候选池补历史。缺乏历史抓取范围契约，相关47列仍隔离。财务修订、当时行业分类及质押披露时点不可验证的快照同样不进入主实验。\n',
        '- 缠论分型使用后一天确认，整段笔结构还会替换此前端点，因此结构分和中枢宽度隔离；5列真实涨停计数依赖历史ST身份，无法用当前名称补齐。已有gt_9p5pct等固定阈值代理另行保留。\n',
        ('- 第二次准入复查另从逐日源恢复22列：周/月PE、PB历史分位，156周股息均值/标准差/稳定性，上市年限，沪深300长期收益/波动，以及龙虎榜滚动统计。周/月数据必须等周期结束后使用，龙虎榜需完整60个交易日文件覆盖；50个非空/零事件样本与原计算器独立核对一致。见periodic_recovery.json与periodic_verification.json。\n' if meta.get('periodic_recovery_columns') else ''),
        '- 当前2024+样本中左侧有484笔延迟退出，最长延迟63天；总收益标签并非每笔都对应同样的自然日资金占用。训练边界与回测已按真实日期处理，后续若优化资金效率，应单独检验单位占用日收益目标，不能把延迟仓位当作普通5日收益。右侧有4059笔未知退出，整组冻结是保守压力规则，限制了对实际策略潜力的判断。\n',
        '- 市场原始数据仍不是逐日归档的供应商版本库。上述检查解决可见的计算时点、样本筛选和标签越界问题，不构成“所有供应商历史版本绝无修订”的证明。\n',
        ('- 独立核验发现原始文件清单过度收录了测试范围之后的龙虎榜文件，其中2026-11-10文件在运行期间更新。保留原始哈希记录；确认文件中所有行均晚于研究截止日，并从当前源重算左右侧全部3列逐日龙虎榜特征，与冻结训练数据逐单元一致。其余输入哈希及训练数据哈希通过核验，详细新旧哈希见verification.json。\n' if verification.get('unused_future_manifest_changes') else ''),
        blocked.to_markdown(index=False)+'\n',
        '## 分布检查\n',drift.to_markdown(index=False)+'\n',
        'PSI使用实际训练样本的分位分箱，并单独计入缺失桶；大于0.25作为漂移提示，不使用测试期漂移值反向筛变量。输入列数不等于实际被树分裂使用的列数，后者见training_times.csv。\n',
        '测试分数超出校准期范围的比例如下。校准器只做边界截断；大量越界时，开仓收益门槛的可迁移性不足，不能把高校准分直接理解为可兑现的收益率。\n',
        calibration.groupby(['side','fold']).outside_calibration.agg(['median','max']).reset_index().round(4).to_markdown(index=False)+'\n',
        '## 后续优化依据\n',
        '同日重复的指数状态不是独立样本；900余列对应的独立市场日期仍只有数百个。当前约一季度的选择窗口面对较多方案，验证期最优方案在测试期经常失效，应先做训练期内的滚动嵌套验证和稳定性筛选，再用未参与开发的时段确认。历史高相关列与漂移数据应作为压缩维度的依据，不能按本次测试收益倒选变量。\n',
        '资金评估目前整组冻结未知退出，适合作为保守压力测试；还需在完整退出价格/成交记录下检验逐持仓资金释放与冲击成本，才能回答实盘可执行收益。公共KDJ/缠论历史计算器及旧缓存也应独立迁移审计，本实验不代表线上历史缓存已全部修复。\n',
        '复现：先依次执行build_full_rank_dataset.py、augment_full_rank_sources.py、audit_full_rank_readiness.py和evaluate_full_rank_comparison.py生成v1；再执行recover_full_rank_periodic.py，并设置FULL_RANK_OUT为full_rank_comparison_v2的绝对路径，执行evaluate_full_rank_comparison.py、audit_full_rank_results.py与summarize_full_rank_comparison.py。v2复用完全相同的8个50/129列基准与10个OOF基模型，其余80个模型重新训练。逐列台账column_audit.csv，逐折准入fold_admission.csv，样本和标签边界contract_*.json，校准calibrators_*.json，策略选择choices_*.json，明细picks_*.parquet。\n',
        '论文：[RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)、[LambdaRank/MART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)、[LambdaLoss](https://marc.najork.org/papers/cikm2018.pdf)。代码采用树提升和抽样梯度，是明确记录的工程适配。\n',
    ]
    (OUT/'README.md').write_text('\n'.join(text))
    try:
        import markdown
        body=markdown.markdown('\n'.join(text),extensions=['tables','fenced_code'])
    except ImportError:
        body='<pre>'+html.escape('\n'.join(text))+'</pre>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>全量排序实验</title><style>body{font:15px system-ui;max-width:1500px;margin:32px auto;line-height:1.7;padding:20px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px}th{background:#edf3fa}h1,h2{color:#17345c}pre{white-space:pre-wrap}</style>'+body)
    print(positive.to_string(index=False),flush=True)


if __name__=='__main__':main()
