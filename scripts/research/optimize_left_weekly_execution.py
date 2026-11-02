#!/usr/bin/env python
"""Fixed component search around the left MACD+BOLL twenty-session control."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from run_kronos_rolling import save_json, sha256
from run_weekly_direction_research import PERIODS, subset_histories, supported

from quant.research.left_weekly_execution import (
    DEVELOPMENT,
    ENTRIES,
    EXITS,
    FILTERS,
    development_rank,
    materialize,
    stock_mask,
)
from quant.research.sequence_strategy_validation import (
    capital_backtest,
    paired_block_gain,
    top_picks,
)

ROOT = Path(__file__).resolve().parents[2]


def identity(config):
    return "__".join(config)


def prepare(source, out):
    events = pd.read_parquet(source / "events.parquet")
    events = events[events.side.eq("left") & events.horizon.eq(20) & events.macd_boll].copy()
    events = events.sort_values(["symbol", "date"]).reset_index(drop=True)
    prices = pd.read_parquet(source / "execution_bars.parquet").sort_values(["symbol", "date"])
    calendar = pd.DatetimeIndex(pd.to_datetime(json.loads((source / "calendar.json").read_text())))
    paths = {name: np.full((len(events), 22), np.nan) for name in ["close", "ma5", "ma10"]}
    groups = prices.groupby("symbol", sort=False)
    for symbol, candidates in events.groupby("symbol", sort=False):
        series = groups.get_group(symbol).set_index("date").close.reindex(calendar)
        full = {"close": series, "ma5": series.rolling(5).mean(), "ma10": series.rolling(10).mean()}
        positions = calendar.get_indexer(candidates.date)
        offsets = positions[:, None] + np.arange(22)
        for name, values in full.items():
            paths[name][candidates.index] = values.to_numpy()[offsets]
    if not np.isfinite(paths["close"]).all():
        raise ValueError("frozen source event path unexpectedly missing")
    baseline, _ = materialize(events, paths, calendar, "next_close", "fixed20")
    expected = events.set_index(["symbol", "date"]).net_return.sort_index()
    actual = baseline.set_index(["symbol", "date"]).net_return.sort_index()
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    save_json(out / "coverage.json", {
        "events": len(events), "stocks": events.symbol.nunique(),
        "label_max_error": float(np.max(np.abs(actual - expected))),
        "ma_warmup": "MA5/10 from cached2024+ daily close; unknown initial windows cannot trigger exits",
        "universe_limitation": "same original preserved OOT candidates with complete20-session paths; excludes future missing/suspended paths as in original, not a point-in-time full-universe reconstruction",
    })
    return events, paths, prices, calendar


def block(frame, calendar, period):
    start, end = PERIODS[period]
    active = calendar[(calendar >= start) & (calendar < end)]
    cutoff = active[-22]
    return frame[
        frame.origin_date.ge(start) & frame.origin_date.le(cutoff) & frame.origin_exit_date.lt(end)
    ].copy(), active


def simulate(frame, prices, active, cost=0.002):
    picks = top_picks(frame, "baseline_percentile").copy()
    picks["execution_score"] = picks.baseline_percentile
    histories = subset_histories(prices, picks.symbol.unique())
    return capital_backtest(picks, histories, active, slots=20, cost=cost)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "reports/research/weekly_direction_daily_entry_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/research/left_weekly_execution_v1")
    parser.add_argument("--component-risk-at-final-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    files = [source / x for x in ["events.parquet", "execution_bars.parquet", "calendar.json"]]
    code = [Path(__file__), ROOT / "src/quant/research/left_weekly_execution.py"]
    for path in code:
        target = out / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save_json(out / "protocol.json", {
        "inputs": [{"path": str(p), "sha256": sha256(p)} for p in files],
        "code": [{"path": str(p), "sha256": sha256(p)} for p in code],
        "filters": FILTERS, "entries": ENTRIES, "exits": EXITS,
        "search": "17 single components including baseline; select one per component by worst3 development returns; at most4 combinations; then freeze before2026",
        "minimum_development_trades": 40, "maximum_development_drawdown": 0.35,
        "component_drawdown_enforced": not args.component_risk_at_final_only,
        "followup_scope": "if true, choose components with >=40trades per period but apply35%drawdown only to final candidates; four combinations unchanged; follows observed component-risk exclusion in v1",
        "entry_deadline": "confirmation in next3 sessions, next-close execution, original signal+21session maximum exit; waiting consumes holding window",
        "execution": "20slots,0.002roundtrip, actual cached eligibility; blocked exits delayed, no threshold fills",
        "admission": "unchanged original event success support criteria in all3 periods AND all3 positive portfolio returns; cash if fail",
        "limitations": "2026 already observed; component selection and follow-up combinations are multiple comparisons; source complete-path candidate restriction retained; stale model rank originally trained for5day label",
    })
    events, paths, prices, calendar = prepare(source, out)
    cache = {}
    transforms = {}
    audits = {}
    records = []

    def get_frame(config):
        filt, entry, exit_rule = config
        if config not in transforms:
            mask = stock_mask(events, filt)
            transforms[config], audits[identity(config)] = materialize(
                events.loc[mask], {k: v[mask] for k, v in paths.items()}, calendar,
                entry, exit_rule,
            )
        return transforms[config]

    def evaluate_development(config):
        name = identity(config)
        if name in cache:
            return cache[name]
        frame = get_frame(config)
        row = {"name": name, "config": config, "admitted": True}
        for period in DEVELOPMENT:
            part, active = block(frame, calendar, period)
            curve, trades, metrics = simulate(part, prices, active)
            passed, event_stats = supported(part, selection=period == "selection")
            row[period] = metrics
            row[period + "_events"] = event_stats
            row["admitted"] = row["admitted"] and passed and metrics["total_return"] > 0
            curve.to_csv(out / f"{period}_{name}_equity.csv", index=False)
            trades.to_parquet(out / f"{period}_{name}_trades.parquet", index=False)
        row["eligible"] = np.isfinite(development_rank(row)[0])
        cache[name] = row
        records.append(row)
        save_json(out / "development_running.json", records)
        print("development", name, [round(row[p]["total_return"], 4) for p in DEVELOPMENT], flush=True)
        return row

    original = ("original", "next_close", "fixed20")
    baseline = evaluate_development(original)
    components = {
        "stock": [(f, "next_close", "fixed20") for f in FILTERS],
        "entry": [("original", e, "fixed20") for e in ENTRIES],
        "exit": [("original", "next_close", e) for e in EXITS],
    }
    winners = {}
    for component, configs in components.items():
        candidates = [evaluate_development(c) for c in configs]
        def rank_component(row):
            return development_rank(row, enforce_drawdown=not args.component_risk_at_final_only)
        eligible = [r for r in candidates if np.isfinite(rank_component(r)[0])]
        winners[component] = max(eligible, key=rank_component) if eligible else baseline
    f = winners["stock"]["config"][0]
    e = winners["entry"]["config"][1]
    x = winners["exit"]["config"][2]
    save_json(out / "components_frozen.json", winners)
    for config in [(f, e, "fixed20"), (f, "next_close", x), ("original", e, x), (f, e, x)]:
        evaluate_development(config)
    eligible = [r for r in records if r["eligible"]]
    admitted = [r for r in eligible if r["admitted"]]
    selected = max(admitted or eligible or [baseline], key=development_rank)
    save_json(out / "selection_frozen.json", {
        "selected": selected, "admitted": bool(admitted),
        "component_winners": {k: v["name"] for k, v in winners.items()},
        "all_development_portfolios": records,
    })
    print("selection frozen", selected["name"], "admitted", bool(admitted), flush=True)

    configurations = list(dict.fromkeys([
        original, *[tuple(r["config"]) for r in winners.values()], tuple(selected["config"]),
    ]))
    results = []
    curves = {}
    for config in configurations:
        name = identity(config)
        part, active = block(get_frame(config), calendar, "test")
        curve, trades, metrics = simulate(part, prices, active)
        curves[name] = curve
        if config == original:
            old = pd.read_csv(source / "left_test_control_weekly_macd_boll20_equity.csv", parse_dates=["date"])
            pd.testing.assert_frame_equal(curve, old, check_exact=False, rtol=1e-12, atol=1e-12)
        gain = paired_block_gain(curves[identity(original)], curve, repetitions=1000)
        winners_pnl = trades.nlargest(5, "pnl").pnl.sum()
        total_pnl = trades.pnl.sum()
        picked = top_picks(part, "baseline_percentile")
        diagnostic = trades.merge(
            picked[["symbol", "date", "origin_date", "exit_date", "exit_trigger_date", "planned_exit_reason"]],
            left_on=["symbol", "signal_date"], right_on=["symbol", "date"],
            suffixes=("", "_planned"), how="left", validate="many_to_one",
        )
        diagnostic.to_parquet(out / f"test_{name}_trades.parquet", index=False)
        curve.to_csv(out / f"test_{name}_equity.csv", index=False)
        _, _, higher_cost = simulate(part, prices, active, cost=0.004)
        result = {
            "name": name, "config": config, **metrics, "gain_vs_control": gain,
            "cost40bps": higher_cost,
            "top5_pnl": float(winners_pnl), "total_realized_pnl": float(total_pnl),
            "top5_share_of_realized_profit": float(winners_pnl / total_pnl) if total_pnl > 0 else None,
            "mean_realized_hold_sessions": float(np.mean(
                calendar.get_indexer(trades.exit_date) - calendar.get_indexer(trades.entry_date)
            )) if len(trades) else None,
            "sell_delays": int(diagnostic.exit_date.gt(diagnostic.exit_date_planned).sum()),
        }
        results.append(result)
        print("test", name, round(metrics["total_return"], 4), flush=True)
    if not admitted:
        _, active = block(get_frame(original), calendar, "test")
        curve, trades, cash = simulate(get_frame(original).iloc[:0], prices, active)
        results.append({"name": "admitted_cash", **cash})
        curve.to_csv(out / "test_admitted_cash_equity.csv", index=False)
    save_json(out / "results.json", results)
    save_json(out / "order_transform_audits.json", audits)
    save_json(out / "postflight.json", {
        "legacy_fund_curve_parity": True,
        "timing_assertions": "every emitted event checked: confirmation<entry<exit<=originaldeadline; dynamic trigger<scheduledexit",
        "single_component_and_combination_count": len(records),
        "source_code_snapshot_hashes_match_protocol": all(sha256(p) == sha256(out / "source_snapshot" / p.relative_to(ROOT)) for p in code),
        "test_period": [str(curves[identity(original)].date.min()), str(curves[identity(original)].date.max())],
    })
    lines = [
        "# 左侧周线MACD+BOLL：选股与交易条件优化", "",
        "原周线方向、日线候选和排序保持；等待买点最长3日，消耗原20日窗口。所有动态卖点收盘确认后次日收盘成交。2026此前已观察，下面是探索复核。", "",
        f"开发期冻结选择：`{selected['name']}`；高胜率准入：{bool(admitted)}。", "",
        "|版本|2026净收益|胜率|PF|最大回撤|敞口|成交|40bps净收益|", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        win = f"{r['win_rate']:.1%}" if r["win_rate"] is not None else "—"
        pf = f"{r['profit_factor']:.2f}" if r["profit_factor"] is not None else "—"
        cost = f"{r['cost40bps']['total_return']:.2%}" if "cost40bps" in r else "—"
        lines.append(f"|{r['name']}|{r['total_return']:.2%}|{win}|{pf}|{r['max_drawdown']:.2%}|{r['mean_exposure']:.1%}|{r['trades']}|{cost}|")
    lines += ["", "开发期候选与收益：", "", "|版本|2024|2025H1|2025H2|组合样本/回撤合格|事件高胜率准入|", "|---|---:|---:|---:|---|---|"]
    for r in records:
        returns = "|".join(f"{r[p]['total_return']:.2%}" for p in DEVELOPMENT)
        lines.append(f"|{r['name']}|{returns}|{r['eligible']}|{r['admitted']}|")
    lines += ["", "局限：原候选源要求后续20日价格路径完整，可能排除未来停牌/缺失路径；本轮保留同一限制以复现对照，没有将其修复成全市场点时研究。动态止损仍会受到次日跳空和跌停影响。事件标签使用计划退出价，组合成交才反映实际权限和卖出顺延；准入事件样本重叠，Wilson仅作初筛。所有收益为各段累计非年化收益，多个候选及已观察2026存在探索偏差。没有通过准入则现金，正收益探索结果不自动上线。生产未修改。"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    save_json(out / "complete.json", {"selected": selected["name"], "admitted": bool(admitted), "completed": True})


if __name__ == "__main__":
    main()
