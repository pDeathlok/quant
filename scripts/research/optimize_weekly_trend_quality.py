#!/usr/bin/env python
"""Improve weekly uptrend definitions without changing daily candidates or execution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from run_kronos_rolling import save_json, sha256
from run_weekly_direction_research import PERIODS, backtest, period_part, supported

from quant.research.sequence_strategy_validation import event_metrics, paired_block_gain
from quant.research.weekly_trend_quality import (
    DEVELOPMENT_PERIODS,
    QUALITY_FEATURES,
    add_outcomes,
    cross_sectional_strength,
    from_daily,
    lifecycle_features,
    shortlist,
    trend_rules,
)

ROOT = Path(__file__).resolve().parents[2]


def prepare(source, out):
    manifest = json.loads((source / "input_sources.json").read_text())
    paths = [
        Path(x["path"])
        for x in manifest
        if "daily_partitioned" in x["path"]
        or x["path"].endswith("overnight_unfiltered_universe_v1/daily_bars.parquet")
    ]
    parts = []
    hashes = []
    for p in paths:
        expected = next(x["sha256"] for x in manifest if x["path"] == str(p))
        actual = sha256(p)
        if actual != expected:
            raise ValueError(f"historical source changed: {p.name}")
        if "daily_partitioned" in str(p):
            frame = pd.read_parquet(
                p,
                columns=[
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "pre_close",
                    "vol",
                ],
            ).rename(columns={"ts_code": "symbol"})
            frame["date"] = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d")
            frame = frame.drop(columns="trade_date")
        else:
            frame = pd.read_parquet(
                p, columns=["symbol", "date", "open", "high", "low", "close", "pre_close", "vol"]
            )
        parts.append(frame)
        hashes.append({"path": str(p), "sha256": actual})
    raw = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"])
    if raw.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate historical bars")
    market_weeks = pd.DatetimeIndex(
        (raw.date.dt.to_period("W-SUN") + 1).dt.start_time.unique()
    ).sort_values()
    chunks = []
    groups = raw.groupby("symbol", sort=False)
    for number, (symbol, frame) in enumerate(groups, 1):
        features = from_daily(frame)
        features = add_outcomes(features, market_weeks)
        features["symbol"] = symbol
        for column in features.select_dtypes(include="number").columns:
            features[column] = features[column].astype(np.float32)
        chunks.append(features)
        if number % 500 == 0:
            print("weekly quality features", number, "/", groups.ngroups, flush=True)
    weekly = cross_sectional_strength(pd.concat(chunks, ignore_index=True))
    weekly["trend_stage"] = weekly.trend_stage.astype("category")
    weekly.to_parquet(out / "weekly_features_labels.parquet", index=False)
    old = pd.read_parquet(
        source / "events.parquet",
        columns=[
            "symbol",
            "side",
            "date",
            "entry_date",
            "exit_date",
            "net_return",
            "horizon",
            "baseline_percentile",
            "macd_boll",
            "boll_up",
        ],
    )
    old = old[old.horizon.isin([10, 20])].copy()
    keys = old[["symbol", "date"]].drop_duplicates().sort_values("date")
    known_cols = [
        "symbol",
        "available_at",
        "date",
        *QUALITY_FEATURES,
        "trend_stage",
        "legacy_macd_boll",
        "legacy_boll_up",
    ]
    known = (
        weekly[known_cols].rename(columns={"date": "trend_source_date"}).sort_values("available_at")
    )
    joined = pd.merge_asof(
        keys, known, left_on="date", right_on="available_at", by="symbol", direction="backward"
    )
    prices = pd.read_parquet(source / "execution_bars.parquet")
    joined = joined.merge(
        prices[["symbol", "date", "close"]].rename(columns={"close": "current_close"}),
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    events = old.merge(joined, on=["symbol", "date"], how="left", validate="many_to_one")
    # The two legacy definitions must reproduce the prior experiment exactly.
    if not events.legacy_macd_boll.fillna(False).eq(events.macd_boll).all():
        raise ValueError("legacy MACD gate parity failure")
    if not events.legacy_boll_up.fillna(False).eq(events.boll_up).all():
        raise ValueError("legacy BOLL gate parity failure")
    if not (events.available_at <= events.date).all():
        raise ValueError("future weekly feature visible")
    for name, mask in trend_rules(events).items():
        events[name] = mask
    events.to_parquet(out / "events.parquet", index=False)
    save_json(out / "input_sources.json", hashes)
    save_json(
        out / "coverage.json",
        {
            "weekly_rows": len(weekly),
            "weekly_stocks": weekly.symbol.nunique(),
            "candidate_rows": len(events),
            "candidate_stocks": events.symbol.nunique(),
            "definition_count": len(trend_rules(events)),
            "legacy_gate_parity": True,
            "all_weekly_features_available": True,
        },
    )
    return events, weekly, prices


def weekly_precision(weekly):
    records = []
    masks = trend_rules(weekly)
    for period, (start, end) in PERIODS.items():
        for n in [4, 8]:
            mature = weekly.available_at.ge(start) & weekly[f"label_end{n}"].lt(end)
            for rule, mask in masks.items():
                part = weekly[mature & mask]
                records.append(
                    {
                        "period": period,
                        "rule": rule,
                        "weeks": n,
                        "stock_weeks": len(part),
                        "continued_up_rate": float(part[f"continued_up{n}"].mean())
                        if len(part)
                        else None,
                        "mean_future_return": float(part[f"future_return{n}"].mean())
                        if len(part)
                        else None,
                        "mean_adverse_return": float(part[f"future_mae{n}"].mean())
                        if len(part)
                        else None,
                    }
                )
    return records


def portfolio_rank(record):
    values = [record[p]["total_return"] for p in DEVELOPMENT_PERIODS]
    return (
        min(values),
        sum(values) / len(values),
        -max(abs(record[p]["max_drawdown"]) for p in DEVELOPMENT_PERIODS),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = {
        "source": str(source),
        "source_events_sha256": sha256(source / "events.parquet"),
        "execution_sha256": sha256(source / "execution_bars.parquet"),
        "code_hashes": {
            str(p): sha256(p)
            for p in [Path(__file__), ROOT / "src/quant/research/weekly_trend_quality.py"]
        },
        "hypothesis": "separate rebound, early transition, persistent trend, pullback, overheated and fading phases",
        "definitions": "24 fixed weekly definitions plus2 within-week daily close vetoes",
        "primary_horizon": 20,
        "robustness_horizon": 10,
        "daily_entry": "unchanged original candidates, next_close, same original score ranking",
        "shortlist": "top6 by worst-period event mean then win rate on2024/2025H1/2025H2;2 legacy controls always included",
        "selection": "maximize worst development-period portfolio return, then mean and drawdown; never read2026 for selection",
        "admission": "original60% success support criteria unchanged; fail -> cash; separately show exploratory selection",
        "weekly_availability": "previous completed natural week only; live veto uses current signal-day close, never unfinished-week MACD",
        "disclosure": "2026 and20-day control already observed; all later results exploratory robustness, not pristine confirmation",
        "trend_labels": "next4/8 market trading weeks positive return with adverse move no worse than-8%; label only, not features",
        "lifecycle_branch": args.lifecycle,
    }
    save_json(out / "protocol.json", protocol)
    if args.dataset:
        cached = args.dataset.resolve()
        events = pd.read_parquet(cached / "events.parquet")
        weekly = pd.read_parquet(cached / "weekly_features_labels.parquet")
        prices = pd.read_parquet(source / "execution_bars.parquet")
        save_json(
            out / "dataset_reference.json",
            {"path": str(cached), "events_sha256": sha256(cached / "events.parquet")},
        )
    else:
        events, weekly, prices = prepare(source, out)
    if args.lifecycle:
        weekly = pd.concat(
            [lifecycle_features(g) for _, g in weekly.groupby("symbol", sort=False)],
            ignore_index=True,
        )
        lifecycle = weekly[
            ["symbol", "available_at", "trend_age_weeks", "episode_runup", "episode_drawdown"]
        ]
        lifecycle.to_parquet(out / "lifecycle_features.parquet", index=False)
        events = events.merge(
            lifecycle, on=["symbol", "available_at"], how="left", validate="many_to_one"
        )
        for name, mask in trend_rules(events).items():
            if name.startswith("age_"):
                events[name] = mask
    calendar = pd.DatetimeIndex(pd.to_datetime(json.loads((source / "calendar.json").read_text())))
    rule_names = list(trend_rules(events))
    if args.lifecycle:
        rule_names = [
            name
            for name in rule_names
            if name.startswith("age_") or name in ["legacy_macd_boll", "legacy_boll_up"]
        ]
    records = []
    choices = []
    results = []
    case_frames = []
    shortlists = []
    # Future-week labels are diagnostic only and cannot enter selection below.
    save_json(out / "weekly_continuation_diagnostic.json", weekly_precision(weekly))
    for side in ["left", "right"]:
        side_frame = events[events.side.eq(side)]
        main_frame = side_frame[side_frame.horizon.eq(20)]
        blocks = {
            period: period_part(main_frame, calendar, period)[0] for period in DEVELOPMENT_PERIODS
        }
        stats = []
        for rule in rule_names:
            row = {"side": side, "rule": rule, "admitted": True}
            for period, block in blocks.items():
                part = block[block[rule]]
                passed, metrics = supported(part, selection=period == "selection")
                row[period] = metrics
                row["admitted"] = row["admitted"] and passed
            stats.append(row)
        names = shortlist(stats)
        shortlists.append({"side": side, "definitions": names})
        save_json(
            out / f"{side}_shortlist_frozen.json",
            {"side": side, "definitions": names, "event_stats": stats},
        )
        simulated = []
        for rule in names:
            row = {
                "side": side,
                "rule": rule,
                "admitted": next(r["admitted"] for r in stats if r["rule"] == rule),
            }
            for period, block in blocks.items():
                _, active = period_part(main_frame, calendar, period)
                curve, trades, metrics = backtest(block[block[rule]], prices, active)
                row[period] = metrics
                curve.to_csv(out / f"{side}_{period}_{rule}_equity.csv", index=False)
                trades.to_parquet(out / f"{side}_{period}_{rule}_trades.parquet", index=False)
            simulated.append(row)
        admitted = [
            r
            for r in simulated
            if r["admitted"] and min(r[p]["total_return"] for p in DEVELOPMENT_PERIODS) > 0
        ]
        selected = max(admitted if admitted else simulated, key=portfolio_rank)
        choice = {
            "side": side,
            "rule": selected["rule"],
            "admitted": bool(admitted),
            "worst_development_return": portfolio_rank(selected)[0],
            "returns": {p: selected[p]["total_return"] for p in DEVELOPMENT_PERIODS},
        }
        choices.append(choice)
        records.extend(stats)
        save_json(
            out / f"{side}_selection_frozen.json",
            {"choice": choice, "development_portfolios": simulated},
        )
        print("trend definition selection frozen", choice, flush=True)
        # Evaluate only the frozen choice and original controls, for two fixed horizons.
        names = list(dict.fromkeys([selected["rule"], "legacy_macd_boll", "legacy_boll_up"]))
        for horizon in [20, 10]:
            frame = side_frame[side_frame.horizon.eq(horizon)]
            block, active = period_part(frame, calendar, "test")
            baseline = None
            versions = {
                "unfiltered": block,
                **{name: block[block[name]] for name in names},
                "admitted_strategy": block[block[selected["rule"]]] if admitted else block.iloc[:0],
            }
            for name, part in versions.items():
                curve, trades, metrics = backtest(part, prices, active)
                if name == "unfiltered":
                    baseline = curve
                results.append(
                    {
                        "side": side,
                        "period": "test",
                        "horizon": horizon,
                        "rule": name,
                        **metrics,
                        "event_metrics": event_metrics(part),
                        "gain_vs_unfiltered": paired_block_gain(baseline, curve, repetitions=1000)
                        if name != "unfiltered"
                        else None,
                    }
                )
                curve.to_csv(out / f"{side}_test_h{horizon}_{name}_equity.csv", index=False)
                trades.to_parquet(
                    out / f"{side}_test_h{horizon}_{name}_trades.parquet", index=False
                )
                if horizon == 20 and name == selected["rule"]:
                    cases = pd.concat(
                        [part.nsmallest(30, "net_return"), part.nlargest(30, "net_return")]
                    ).copy()
                    cases["case_side"] = side
                    case_frames.append(cases)
            chosen_curve = pd.read_csv(
                out / f"{side}_test_h{horizon}_{selected['rule']}_equity.csv", parse_dates=["date"]
            )
            legacy = pd.read_csv(
                out / f"{side}_test_h{horizon}_legacy_macd_boll_equity.csv", parse_dates=["date"]
            )
            save_json(
                out / f"{side}_h{horizon}_gain_vs_old_gate.json",
                paired_block_gain(legacy, chosen_curve, repetitions=2000),
            )
        # Diagnose what the old condition actually includes; not used to select rules.
        stages = []
        for period in PERIODS:
            block, _ = period_part(main_frame, calendar, period)
            block = block[block.legacy_macd_boll]
            for stage, part in block.groupby("trend_stage", observed=True):
                stages.append(
                    {
                        "period": period,
                        "stage": str(stage),
                        "side": side,
                        **event_metrics(part),
                        "share": len(part) / len(block),
                    }
                )
        pd.DataFrame(stages).to_csv(out / f"{side}_old_gate_stage_diagnostic.csv", index=False)
    save_json(out / "rule_event_stats.json", records)
    save_json(out / "selection_decision.json", choices)
    save_json(out / "results.json", results)
    if case_frames:
        pd.concat(case_frames).to_csv(out / "selected_cases_diagnostic_only.csv", index=False)
    lines = [
        "# 上涨趋势判定优化",
        "",
        "只调整趋势定义，原日线候选、排序、入场和费用不变；20日主分析，10日稳健性。2024—2025按最差阶段表现选择，2026已观察，不是全新封存测试。",
        "",
        "|侧|20日版本|2026收益|胜率|PF|最大回撤|敞口|成交|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in results if r["horizon"] == 20]:
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        pf = f"{row['profit_factor']:.2f}" if row["profit_factor"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['rule']}|{row['total_return']:.2%}|{win}|{pf}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.1%}|{row['trades']}|"
        )
    lines += ["", "开发期固定选择："] + [f"- {c}" for c in choices]
    lines += [
        "",
        "强度/持续性/结构/相对强弱等条件是候选定义，不保证已经捕捉未来上涨；周趋势延续诊断见weekly_continuation_diagnostic.json。",
        "每日实时否决只用当前日收盘与已完成周的EMA26/13周高点比较，不使用当周完整MACD。",
        "未来4/8周标签严格隔离，不进入规则或模型选择。事件Wilson忽略重叠，仅作初筛；开发期选择与26个定义存在多重比较。",
        "准入门槛未降低，没有通过则准入账户现金。正收益探索版本也不能自动上线；持有期与收益口径参见原报告。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    save_json(
        out / "complete.json",
        {"status": "research_complete_not_production", "seconds": time.time() - started},
    )
    print((out / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
