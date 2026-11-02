#!/usr/bin/env python
"""Explicit post-first-result exploration: fixed weekly market breadth gates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from run_kronos_rolling import save_json, sha256
from run_weekly_direction_research import PERIODS, backtest, period_part, rule_mask, supported

from quant.research.sequence_strategy_validation import paired_block_gain
from quant.research.trend_research import weekly_market_breadth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    save_json(
        out / "protocol.json",
        {
            "source": str(source),
            "dataset_sha256": sha256(source / "events.parquet"),
            "execution_sha256": sha256(source / "execution_bars.parquet"),
            "thresholds": [0.4, 0.5, 0.6],
            "market_gate": "prior completed week: fraction close>MA60 >= threshold and median20dreturn>0",
            "stock_rule": "exact previously frozen 2025H2-selected stock rule; no reselecting horizon or indicators",
            "selection": "2024/2025H1 support and2025H2 return; previous strict success conditions unchanged",
            "disclosure": "additional hypothesis after seeing phase1 results; all2026 outcomes exploratory, not independent confirmation",
            "code_sha256": sha256(Path(__file__)),
        },
    )
    events = pd.read_parquet(source / "events.parquet")
    prices = pd.read_parquet(source / "execution_bars.parquet")
    calendar = pd.DatetimeIndex(pd.to_datetime(json.loads((source / "calendar.json").read_text())))
    breadth = weekly_market_breadth(prices)
    breadth.to_parquet(out / "weekly_market_breadth.parquet", index=False)
    results = []
    choices = []
    search = []
    for side in ["left", "right"]:
        original = json.loads((source / f"{side}_selection_frozen.json").read_text())
        frame = events[events.side.eq(side) & events.horizon.eq(original["horizon"])].merge(
            breadth, on="date", validate="many_to_one"
        )
        stock = frame[rule_mask(frame, original["week"], original["entry"])]
        trials = []
        for threshold in [0.4, 0.5, 0.6]:
            filtered = stock[
                stock.market_breadth.ge(threshold) & stock.market_median_return20.gt(0)
            ]
            record = {"side": side, "threshold": threshold, "supported": True}
            for period in ["development_2024", "development_2025H1", "selection"]:
                block, active = period_part(filtered, calendar, period)
                ok, metrics = supported(block, selection=period == "selection")
                record[period] = metrics
                record["supported"] = record["supported"] and ok
                if period == "selection":
                    _, _, portfolio = backtest(block, prices, active)
                    record["selection_portfolio"] = portfolio
            trials.append(record)
        admitted = [
            t for t in trials if t["supported"] and t["selection_portfolio"]["total_return"] > 0
        ]
        best = max(
            admitted if admitted else trials, key=lambda t: t["selection_portfolio"]["total_return"]
        )
        choice = {
            "side": side,
            "threshold": best["threshold"],
            "admitted": bool(admitted),
            "stock_week": original["week"],
            "stock_entry": original["entry"],
            "horizon": original["horizon"],
            "selection_return": best["selection_portfolio"]["total_return"],
        }
        choices.append(choice)
        search.extend(trials)
        save_json(out / f"{side}_selection_frozen.json", choice)
        print("frozen", choice, flush=True)
        macro = stock[
            stock.market_breadth.ge(best["threshold"]) & stock.market_median_return20.gt(0)
        ]
        for period in PERIODS:
            baseline, active = period_part(frame, calendar, period)
            rule, _ = period_part(stock, calendar, period)
            gated, _ = period_part(macro, calendar, period)
            variants = {
                "baseline": baseline,
                "stock_rule_only": rule,
                "market_gate_exploratory": gated,
                "admitted_gate": gated if admitted else gated.iloc[:0],
            }
            reference = None
            for name, sample in variants.items():
                curve, trades, metrics = backtest(sample, prices, active)
                if name == "stock_rule_only":
                    reference = curve
                gain = (
                    paired_block_gain(reference, curve, repetitions=1000)
                    if name in ["market_gate_exploratory", "admitted_gate"]
                    else None
                )
                results.append(
                    {
                        "side": side,
                        "period": period,
                        "variant": name,
                        **metrics,
                        "gain_vs_stock_rule": gain,
                    }
                )
                curve.to_csv(out / f"{side}_{period}_{name}_equity.csv", index=False)
                trades.to_parquet(out / f"{side}_{period}_{name}_trades.parquet", index=False)
    save_json(out / "selection_decision.json", choices)
    save_json(out / "search.json", search)
    save_json(out / "results.json", results)
    lines = [
        "# 追加探索：周线方向＋市场广度开关",
        "",
        "这是在查看首轮结果后提出的分支，2026结果不能作为独立验证。只在2024—2025选择三个固定广度阈值，不重选周线规则/日线买点/持有期。",
        "",
        "|侧|版本|2025H2收益|2026收益|2026胜率|成交|回撤|敞口|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in results if r["period"] == "test"]:
        val = next(
            r
            for r in results
            if r["side"] == row["side"]
            and r["period"] == "selection"
            and r["variant"] == row["variant"]
        )
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['variant']}|{val['total_return']:.2%}|{row['total_return']:.2%}|{win}|{row['trades']}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.1%}|"
        )
    lines += ["", "冻结选择："] + [f"- {c}" for c in choices]
    lines += [
        "",
        "市场指标同样使用上一完整周，缺60日预热样本时不交易。开关仅控制新入场，原持仓按既定期限退出，不在关门日假定提前卖出。",
        "敞口降低能够减损，但这不等于选股alpha或高胜率。跨期准入条件未放宽；未通过时准入账户保持现金。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    save_json(
        out / "complete.json",
        {"status": "research_complete_not_production", "seconds": time.time() - started},
    )
    print((out / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
