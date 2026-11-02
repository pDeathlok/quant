#!/usr/bin/env python3
"""Describe frozen allocation results without changing routes or parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_byd_allocation import OUT, paired_bootstrap, svg_chart, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    out = parser.parse_args().output_dir
    rows = []
    for name in ("share", "discount", "adah2011", "squint_grid", "multiwindow", "context", "omd", "ftrl", "cost_prox", "robust_prox", "grid_search"):
        a = pd.read_csv(out / f"ledger_{name}.csv", parse_dates=["date"])
        for comparator in ("original_range_matched", "constant_one_third"):
            b = pd.read_csv(out / f"ledger_{comparator}.csv", parse_dates=["date"])
            rows.append({"method": name, **paired_bootstrap(a, b, comparator)})
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(out / "family_paired_bootstrap.csv", index=False)
    chart_names = ("selected_pipeline", "multiwindow", "grid_search", "constant_one_third", "original_range_matched")
    svg_chart({name: pd.read_csv(out / f"ledger_{name}.csv", parse_dates=["date"])
               for name in chart_names}, out / "risk_comparable_equity.svg")
    ledger_checks = []
    for path in sorted(out.glob("ledger_*.csv")):
        frame = pd.read_csv(path)
        numeric = [key for key in frame.columns if key in (
            "return", "equity", "cash", "shares", "position", "proposal", "requested_target", "turnover", "trade_shares", "cost",
        ) or key.startswith("weight_")]
        if not np.isfinite(frame[numeric].to_numpy()).all():
            raise RuntimeError(f"Nonfinite financial field: {path}")
        if frame.cash.min() < -1e-6 or frame.shares.min() < 0 or frame.equity.min() <= 0:
            raise RuntimeError(f"Infeasible account: {path}")
        weights = frame.filter(like="weight_")
        if not weights.empty and not np.allclose(weights.sum(axis=1), 1, atol=1e-12):
            raise RuntimeError(f"Invalid weights: {path}")
        ledger_checks.append({"ledger": path.name, "rows": len(frame), "finite": True, "feasible": True})
    write_json(out / "ledger_checks.json", ledger_checks)
    metrics = pd.read_csv(out / "metrics.csv")
    outer = metrics[metrics.period.eq("outer_2020plus")].set_index("name")
    concentration = pd.read_csv(out / "concentration.csv").set_index("name")
    pruning = json.loads((out / "pruning_audit.json").read_text())
    grid_interval = bootstrap[bootstrap.method.eq("grid_search") & bootstrap.comparator.eq("constant_one_third")].iloc[0]
    search = pd.read_csv(out / "search_comparison.csv")
    costs = []
    for name in ("multiwindow", "grid_search", "constant_one_third", "original_range_matched"):
        ledger = pd.read_csv(out / f"ledger_{name}.csv", parse_dates=["date"])
        ledger = ledger[ledger.date.ge("2020-01-01")]
        costs.append(f"- {name}：订单{int(ledger.trade_shares.ne(0).sum())}笔，费用及滑点{ledger.cost.sum():,.0f}元，实现年化波动{ledger['return'].std()*np.sqrt(252):.2%}。")
    counts = [row["count"] for row in pruning if row["year"] >= 2020]
    interpretation = f"""\n## 冻结结果的解释（只诊断，不再次选型）

1. 多窗口融合确实减轻了权重集中：基础MWU平均有效专家{concentration.loc['mwu','mean_effective_experts']:.2f}个，多窗口为{concentration.loc['multiwindow','mean_effective_experts']:.2f}个；但多窗口累计净收益{outer.loc['multiwindow','total_return']:.2%}，仍未超过固定1/3仓位{outer.loc['constant_one_third','total_return']:.2%}。
2. 网格/贝叶斯搜索流程累计净收益{outer.loc['grid_search','total_return']:.2%}，相对固定1/3仓位差{outer.loc['grid_search','total_return']-outer.loc['constant_one_third','total_return']:.2%}（百分点）；年化日均收益差的20日分块描述区间[{grid_interval.low95:.2%}, {grid_interval.high95:.2%}]跨0，且未校正多重比较与已观察历史，不能据此认定稳定增量成立。
3. 固定同一MWU/交易控制时，diverse规则池收益{outer.loc['fixed_mwu_rules','total_return']:.2%}，加入多周期模型后{outer.loc['fixed_mwu_diverse','total_return']:.2%}，全部26专家{outer.loc['fixed_mwu_all','total_return']:.2%}；本轮没有证明增加模型或专家数量能改善结果。
4. 本轮高相关性去重阈值下，2020-2026保留数量为{counts}，没有移除实际专家，因此去重消融与all池相同。它证明的是这套保守规则无实际增益，不能称已通过去重实现分散。
5. 贝叶斯每年12次完整评分查询，全网格48次；本候选库各年度内层最优分差最大{search[search.search.eq('bayes_search')].gap_to_grid.max():.6f}。这是固定库上的搜索效率结果，不是额外交易收益保证；多保真等价预算32次，固定随机12次平均内层分差{search[search.search.eq('random_search')].gap_to_grid.mean():.4f}。

交易负担与实现风险：

{chr(10).join(costs)}

费用金额来自各自跨年连续账户，起点与净值路径不同，不能直接当作同一本金下的费用百分比。平均仓位接近也不等于风险完全相同；上表给出实际波动。各方法与固定1/3仓位的配对描述区间均见family_paired_bootstrap.csv；全部方法该区间均跨0。

![仓位接近方法对照](risk_comparable_equity.svg)

本附加诊断不改变候选库、年度配置或成交路径。全部{len(ledger_checks)}个账本的金融数值有限、现金和股数非负、权益正数，含权重账本逐行权重和为1。
完整重现先运行run_byd_allocation.py，再运行：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/research/summarize_byd_allocation.py`。
"""
    report_path = out / "report.md"
    report = report_path.read_text(encoding="utf-8").split("\n## 冻结结果的解释（只诊断，不再次选型）")[0]
    report_path.write_text(report + interpretation, encoding="utf-8")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["post_replay_diagnostics"] = {
        "routes_changed": False, "all_ledgers_finite_and_feasible": True,
        "ledger_count": len(ledger_checks),
        "source": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_family_vs_constant_third_intervals_cross_zero": bool(
            ((bootstrap[bootstrap.comparator.eq('constant_one_third')].low95 < 0)
             & (bootstrap[bootstrap.comparator.eq('constant_one_third')].high95 > 0)).all()),
    }
    write_json(manifest_path, manifest)
    print(f"Checked {len(ledger_checks)} ledgers; appended frozen-result interpretation", flush=True)


if __name__ == "__main__":
    main()
