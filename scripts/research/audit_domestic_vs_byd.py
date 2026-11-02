#!/usr/bin/env python3
"""Fixed comparison audit; no fitting, tuning, or production configuration changes."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from quant.research.byd_mwu import ROOT
from quant.research.byd_wear import WearConfig
from quant.research.domestic_portfolio import START, CODES, load_panel, simulate, portfolio_metrics

OUT = ROOT / 'reports/research/domestic_vs_byd_audit_20261013'
PRIOR = ROOT / 'reports/research/domestic_portfolio_20261013'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'purpose': 'Same-account comparison audit after previous results were observed; no optimization',
        'start': START, 'funding': 1_000_000, 'frequency': 5, 'deadband': .02,
        'byd_fixed_weights': [.10, .20, 1 / 3, .40, .50, .60, 1.],
        'comparators': ['balanced', 'growth', 'gold_bond'],
        'cases': ['base', 'severe'], 'new_model_fits': 0,
        'scope': 'BYD-only fixed policies bypass previous 10% asset cap by design; shared execution unchanged',
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    panel = load_panel()
    targets = {}
    for weight in protocol['byd_fixed_weights']:
        path = np.zeros_like(panel.data['close'])
        path[:, 0] = weight * panel.data['eligible'][:, 0]
        targets[f'byd_{weight:.3f}'] = path
    with np.load(PRIOR / 'expert_targets.npz') as cache:
        for name in protocol['comparators']:
            targets[name] = cache[name]
    rows = []
    cases = {'base': WearConfig(), 'severe': WearConfig(spread=.002, impact_eta=.5)}
    for case, wear in cases.items():
        for name, target in targets.items():
            frame, orders = simulate(panel, target, wear)
            frame.to_csv(OUT / f'ledger_{case}_{name}.csv', index=False)
            orders.to_csv(OUT / f'orders_{case}_{name}.csv', index=False)
            for start in [START, '2018-01-01', '2024-01-01']:
                block = frame[frame.date.ge(start)]
                rows.append({'case': case, 'name': name, 'start': start,
                             'kind': 'same_executor_fresh_2014_account', **portfolio_metrics(block)})
            if name in protocol['comparators']:
                original = pd.read_csv(PRIOR / f'ledger_{case}_{name}.csv', parse_dates=['date', 'signal_date'])
                np.testing.assert_allclose(frame.equity, original.equity, rtol=1e-12, atol=1e-7)
            print(f'{case}: {name}', flush=True)
    # Prior strategies: align daily returns, retain original account/controller path.
    # Missing stock quotations contribute zero return on exchange-calendar dates.
    # This table is explicitly separate from fresh common-executor comparisons.
    old_files = {
        'old_byd_grid': ROOT / 'reports/research/byd_growth_20261013/ledger_base_previous_grid.csv',
        'old_byd_round1': ROOT / 'reports/research/byd_growth_20261013/ledger_base_round_1.csv',
    }
    comparison_dates = panel.dates[panel.dates >= pd.Timestamp(START)]
    for name, path in old_files.items():
        old = pd.read_csv(path, parse_dates=['date', 'signal_date']).set_index('date')
        aligned = old.reindex(comparison_dates)
        aligned['return'] = aligned['return'].fillna(0.)
        aligned['position'] = aligned['position'].ffill()
        aligned['cost'] = aligned['cost'].fillna(0.)
        aligned['orders'] = aligned['trade_shares'].fillna(0.).ne(0).astype(int)
        aligned['capacity_limited'] = aligned['capacity_limited'].astype('boolean').fillna(False).astype(bool)
        aligned['signal_date'] = panel.dates.to_series().shift(1).reindex(comparison_dates).to_numpy()
        aligned.index.name = 'date'
        aligned = aligned.reset_index()
        for start in [START, '2018-01-01', '2024-01-01']:
            block = aligned[aligned.date.ge(start)]
            rows.append({'case': 'base', 'name': name, 'start': start,
                         'kind': 'old_account_continued_dates_aligned', **portfolio_metrics(block)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'metrics.csv', index=False)
    lines = ['# 多资产与BYD同口径对照审计', '',
             '本次仅对照固定策略，不调参、不训练新模型。基础资金100万元，2014-08-01起清空持仓建立共享账户，费用、公司行为、参与率、5交易日检查与2个百分点不交易区间完全相同。只有BYD目标仓位不同：10%、20%、33.3%、40%、50%、60%、100%。单BYD对照有意不套用原多资产10%上限。', '',
             '2018/2024区间是同一2014账户的子区间收益，未重新注资。旧网格与旧第一轮在另一张表中保留原有账户与执行规则，仅对齐评价日期；缺报价日记零收益，因此不能把它们描述为完全相同执行器的实验。所有历史已经参与研究，不是盲测。', '',
             '|成本|起点|固定策略|年化净收益|波动率|最大回撤|平均持仓|订单数|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for row in result[result.kind.eq('same_executor_fresh_2014_account')].itertuples():
        lines.append(f'|{row.case}|{row.start}|{row.name}|{row.cagr:.2%}|{row.volatility:.2%}|{row.max_drawdown:.2%}|{row.average_position:.2%}|{row.orders}|')
    lines += ['', '## 原BYD策略的共同日期结果（保留原执行规则）', '',
              '|起点|原策略|年化净收益|波动率|最大回撤|平均持仓|',
              '|---|---|---:|---:|---:|---:|']
    for row in result[result.kind.eq('old_account_continued_dates_aligned')].itertuples():
        lines.append(f'|{row.start}|{row.name}|{row.cagr:.2%}|{row.volatility:.2%}|{row.max_drawdown:.2%}|{row.average_position:.2%}|')
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    manifest = {'static_baselines_match_previous_ledgers': True,
                'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                  ['scripts/research/audit_domestic_vs_byd.py', 'src/quant/research/domestic_portfolio.py']},
                'prior_sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in old_files.values()}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(result[(result.case == 'base') & (result.start == START)][['name', 'kind', 'cagr', 'volatility', 'max_drawdown']].to_string(index=False))


if __name__ == '__main__':
    main()
