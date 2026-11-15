"""Render completed exploratory factor evidence without promoting models."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/research/joinquant_candidates_v1'


def main() -> None:
    assert json.loads((OUT / 'complete.json').read_text())['status'] == 'complete'
    comparison = pd.read_csv(OUT / 'comparisons.csv')
    metrics = pd.read_csv(OUT / 'metrics.csv')
    single = pd.read_csv(OUT / 'single_factors.csv')
    # Do not rank long candidates on the two complete 2026 label dates.
    eligible = single.loc[single.dates.ge(30)]
    ranked = eligible.groupby(['side', 'factor']).agg(
        signed_rank_ic=('signed_test_ic', 'mean'),
        weakest_fold_ic=('signed_test_ic', 'min'),
        positive_folds=('signed_test_ic', lambda x: int(x.gt(0).sum())),
        folds=('fold', 'size'),
        direction=('fit_direction', lambda x: ','.join(map(str, sorted(x.unique())))),
    ).reset_index().sort_values(['side', 'signed_rank_ic'], ascending=[True, False])
    ranked.to_csv(OUT / 'ranking.csv', index=False)
    catalog = json.loads((ROOT / 'configs/research/joinquant_candidates.json').read_text())
    summary = {
        'tested_factors': 26, 'model_runs': len(metrics),
        'official_top_ranking_verified': False,
        'production_promoted': False,
        'conclusion': 'No stable multi-year model increment from adding all26 factors.',
        'short_top20_unit': 'mean per selected cohort trade; not portfolio annual return',
        'long_unit': '26week composite ranking utility, not money return',
        'long_2026': 'Only2complete weekly labels; excluded from ranking/conclusions.',
        'short_significance': 'All paired95%intervals include zero; no multiplicity-adjusted discovery.',
        'long_significance': 'Fewer than two27week blocks per year; confidence bounds suppressed.',
        'model_baseline': 'Matched HGB regressors with identical hyperparameters; not deployed model comparison.',
        'selection_bias': 'Historical diagnostics reused. Rank tables are exploratory, not fresh confirmation.',
        'ranking': ranked.head(0).to_dict('records'),
    }
    summary['top5_by_side'] = {s: g.head(5).to_dict('records') for s, g in ranked.groupby('side')}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    short = comparison.loc[comparison.side.ne('long')].copy()
    for c in ('delta', 'ci_low', 'ci_high'):
        short[c] *= 10000
    title = '聚宽候选因子：26项历史对照 · 2026-11-11'
    paragraphs = [
        '结论：已加入研究因子库并完成32次同参数模型对照。整批加入未形成稳定跨年份增益，暂不晋级生产。',
        '官网Top榜受地区访问限制，无法核验当前排名。这里的Top是本地历史研究结果，不代表聚宽官方Top。',
        '单因子方向仅用训练期确定；表内IC按测试年份等权平均，非收益率。多个候选、历史样本重复研究存在选择偏差。',
        '短线使用现有冻结事件池：左侧5日终值收益减20bp，右侧使用已审计退出净收益。按日Top20均值比较，不是组合净值、年化收益或新撮合回测。退出未知的选中样本保留，任一退出未知的整日均值作缺失，两组配对只用双方完整日期。',
        '基线与新增组均重新训练60轮同参数HGB回归器，按时间分段、标签到期剔除、按日期等权。不是当前生产模型本身的A/B对照。',
        '长线目标是26周收益/回撤/行业相对表现组成的排序效用，非现金收益。2026年仅2个完整周标签，不据此下结论；2024/25也不足两个27周相关区块，不报告可靠显著性。',
        '财务数据严格在公告日期之后使用，采用首次披露记录；供应商历史修订未获得原始时点版本认证。存量研究股票池本身的选择偏差保留。',
        '价格基于本地连续化OHLCV；偏度为无偏估计、峰度为超额峰度，尚未与聚宽逐值核对。三个proxy见公式清单。',
    ]
    tables = [('短线模型增量（bp；1bp=0.01个百分点）', short),
              ('本地单因子排序（剔除少于30个有效日期的折）', ranked),
              ('长线模型增量（效用分数单位）', comparison.loc[comparison.side.eq('long')]),
              ('全部模型指标', metrics), ('候选公式清单', pd.DataFrame(catalog['factors'])),
              ('暂缓项目及原因', pd.DataFrame(catalog['deferred']))]
    content = '<!doctype html><meta charset="utf-8"><title>' + title + '</title>'
    content += '<style>body{font:15px system-ui;max-width:1300px;margin:40px auto;padding:24px;color:#17202a}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ddd;padding:7px;text-align:left}th{background:#edf3f8}h2{margin-top:36px}p{line-height:1.8}</style>'
    content += '<h1>' + title + '</h1>' + ''.join('<p>' + html.escape(p) + '</p>' for p in paragraphs)
    for caption, frame in tables:
        content += '<h2>' + caption + '</h2>' + frame.to_html(index=False, float_format=lambda x: f'{x:.5f}')
    content += '<p>来源：<a href="https://www.joinquant.com/help/api/help?name=factor_values">聚宽因子文档</a>；<a href="https://www.joinquant.com/view/factorlib/list">聚宽看板（本次无法读取）</a>；<a href="https://github.com/JoinQuant/jqdatasdk">聚宽官方SDK</a>；公式核对使用此前公开留存的因子文档。</p>'
    (OUT / 'report.html').write_text(content)


if __name__ == '__main__':
    main()
