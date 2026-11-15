"""Publish two retained futures strategies, research evidence and a dated preview."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/research/index_futures'
OUT = ROOT/'web/data/index-futures'
MODES = {'fixed_qvix':'固定 QVIX · 收益候选', 'rolling_qvix':'季度重选 QVIX · 稳定候选'}
PROFILES = {'stress':(.0006,.25), 'basic':(.0003,.15)}


def recent_performance(ledger):
    """Trailing results of one continuous account, including the window's start equity."""
    rows = []
    for n in (20,60,120):
        part = ledger.tail(n)
        if part.empty:
            continue
        initial = float(part.iloc[0].equity-part.iloc[0].pnl)
        path = np.r_[initial,part.equity.to_numpy()]
        rows.append(dict(sessions=len(part),start=str(part.iloc[0].entry_date),end=str(part.iloc[-1].exit_date),
            total_return=float(path[-1]/initial-1),max_drawdown=float((path/np.maximum.accumulate(path)-1).min()),
            net_pnl=float(part.pnl.sum()),flat_ratio=float(part.gross_lots.eq(0).mean())))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source = SOURCE/'panic_factors'
    all_rows = pd.read_csv(source/'summary.csv',dtype={'start':str,'end':str})
    columns = ['start','end','start_kind','annual','fee','margin','startup_mode','window','days',
               'total_return','max_drawdown','flat_ratio','trades','forced']
    rows = all_rows.loc[all_rows.startup_mode.isin(MODES)&all_rows.variant.eq('mean5'),columns].copy()
    assert not rows.duplicated(['start','fee','margin','startup_mode','window']).any()
    annual = pd.read_csv(source/'annual_marked.csv',dtype={'start':str,'year':str})
    annual = annual.loc[annual.start.eq('20250101')&annual.side.eq('all')&annual.startup_mode.isin(MODES),
                        ['startup_mode','fee','margin','year','total_return','max_drawdown']]
    trades = pd.read_csv(source/'trades.csv',dtype={'start':str,'entry':str,'exit':str})
    recent = {}
    for mode in MODES:
        recent[mode] = {}
        for profile,(fee,margin) in PROFILES.items():
            ledger = pd.read_parquet(source/'2025/ledgers'/f'20250101_mean5_{mode}_{fee}_{margin}.parquet')
            peak = np.maximum.accumulate(np.r_[500000.,ledger.equity.to_numpy()])[1:]
            curve = ledger[['exit_date','equity']].rename(columns={'exit_date':'date'}).copy()
            curve['drawdown'] = ledger.equity.to_numpy()/peak-1
            chosen = trades.loc[trades.start.eq('20250101')&trades.startup_mode.eq(mode)&trades.fee.eq(fee)&trades.margin.eq(margin)].tail(8)
            chosen = chosen[['entry','exit','code','product','side','pnl','days']].copy()
            chosen['end_mark'] = chosen.exit.eq(ledger.exit_date.max())
            recent[mode][profile] = dict(windows=recent_performance(ledger),
                equity=float(ledger.iloc[-1].equity),current_drawdown=float(curve.iloc[-1].drawdown),
                total_return=float(ledger.iloc[-1].equity/500000-1),max_drawdown=float(curve.drawdown.min()),
                curve=json.loads(curve.to_json(orient='records')),
                trades=json.loads(chosen.to_json(orient='records')))
    signal_path = SOURCE/'workspace_live/signal.json'
    signal = json.loads(signal_path.read_text()) if signal_path.exists() else None
    summary = dict(version=2,data_end='2026-11-06',research_date='2026-11-15',capital=500000,max_lots=1,
        default_mode='rolling_qvix',modes=MODES,
        descriptions={
            'fixed_qvix':'固定 base 入场政策，季度重训模型。压力情景的244日窗口收益中位数较高。',
            'rolling_qvix':'季度在原有四种政策中重选，模型也按季度更新。压力情景的244日窗口最大回撤较低。'},
        records=json.loads(rows.to_json(orient='records',double_precision=12)),
        continuous_2025=json.loads(annual.to_json(orient='records',double_precision=12)),
        recent=recent,signal=signal,source='panic_factors · mean5 · frozen research through 2026-11-06')
    temporary = OUT/'research.tmp'
    temporary.write_text(json.dumps(summary,ensure_ascii=False,separators=(',',':'),allow_nan=False))
    temporary.replace(OUT/'research.json')
    rows.to_csv(OUT/'comparison.csv',index=False)
    for source_name,target in [('state_model_policies/findings.md','strategy-research.md'),
        ('policy_label_calibration/findings.md','calibration-research.md'),('panic_factors/report.md','panic-research.md'),
        ('option_iv/report.md','option-iv-research.md')]:
        path = SOURCE/source_name
        if path.exists():
            content = re.sub(r'!\[[^\]]*\]\([^)]*\)','（图表保存在项目完整研究目录中。）',path.read_text())
            content = re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]*\)',r'\1',content)
            (OUT/target).write_text(content)
    for folder,name in [('panic_factors','panic'),('option_iv','option-iv')]:
        (OUT/f'{name}-comparison.csv').write_bytes((SOURCE/folder/'summary.csv').read_bytes())
    print(f'Published {len(MODES)} strategies, {len(rows)} windows; signal {signal["signal_date"] if signal else "unavailable"}')


if __name__=='__main__': main()
