#!/usr/bin/env python
"""Predeclared weekly-direction/daily-entry rules and monthly shallow models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_kronos_rolling import save_json, sha256
from sklearn.ensemble import HistGradientBoostingClassifier

from quant.research.kronos_rolling import refit_segments
from quant.research.sequence_strategy_validation import (
    capital_backtest,
    event_metrics,
    paired_block_gain,
    top_picks,
    wilson_lower,
)
from quant.research.trend_research import (
    DAILY_FEATURES,
    WEEKLY_FEATURES,
    WEEKLY_GATES,
    direction_features,
    forward_event_labels,
    recency_weights,
    recent_window,
)

ROOT = Path(__file__).resolve().parents[2]
PERIODS = {
    "development_2024": ("2024-01-01", "2025-01-01"),
    "development_2025H1": ("2025-01-01", "2025-07-01"),
    "selection": ("2025-07-01", "2026-01-01"),
    "test": ("2026-01-01", "2026-09-05"),
}


def prepare(out):
    source = ROOT / "reports/research"
    inputs = []
    predictions = []
    for side, folder, column in [
        ("left", "left_side_unified_v3_group4_input_parity", "pred_unified_left_long_task_deep"),
        ("right", "right_side_unified_canonical_v5_rule113", "pred_unified_long_task_deep"),
    ]:
        path = source / folder / "test_predictions.parquet"
        inputs.append({"path": str(path), "sha256": sha256(path)})
        frame = pd.read_parquet(path)
        frame = frame[frame.entry_mode.eq("next_close") & frame.horizon.eq(5)].copy()
        frame["side"] = side
        frame["baseline_score"] = frame[column]
        keep = ["symbol", "date", "side", "baseline_score"]
        if "terminal_return" in frame:
            keep += ["terminal_return"]
        predictions.append(frame[keep])
    events = pd.concat(predictions, ignore_index=True)
    events["date"] = pd.to_datetime(events.date).dt.normalize()
    events["baseline_percentile"] = events.groupby(["side", "date"]).baseline_score.rank(pct=True)
    if events.duplicated(["side", "symbol", "date"]).any():
        raise ValueError("duplicate source events")
    path = source / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    raw = pd.read_parquet(path)
    inputs.append({"path": str(path), "sha256": sha256(path)})
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    calendar = pd.DatetimeIndex(raw.date.unique()).sort_values()
    warm = []
    for p in sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet")):
        month = p.parent.name.split("=")[-1]
        if not "202201" <= month < "202401":
            continue
        part = pd.read_parquet(
            p, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol"]
        )
        part = part.rename(columns={"ts_code": "symbol"})
        part["date"] = pd.to_datetime(part.trade_date.astype(str), format="%Y%m%d")
        warm.append(part.drop(columns="trade_date"))
        inputs.append({"path": str(p), "sha256": sha256(p)})
    historical = pd.concat(
        [*warm, raw[["symbol", "date", "open", "high", "low", "close", "pre_close", "vol"]]],
        ignore_index=True,
    ).sort_values(["symbol", "date"])
    if historical.duplicated(["symbol", "date"]).any():
        raise ValueError("overlapping market sources")
    limits = []
    for p in sorted((source / "overnight_unfiltered_limits_v1").glob("*.parquet")):
        part = pd.read_parquet(p).rename(columns={"ts_code": "symbol"})
        part["date"] = pd.to_datetime(part.trade_date.astype(str), format="%Y%m%d")
        limits.append(part[["symbol", "date", "up_limit", "down_limit"]])
        inputs.append({"path": str(p), "sha256": sha256(p)})
    limits = pd.concat(limits).drop_duplicates(["symbol", "date"])
    execution = raw[["symbol", "date", "close", "vol", "valid_bar"]].merge(
        limits, on=["symbol", "date"], how="left", validate="one_to_one"
    )
    source_groups = historical.groupby("symbol", sort=False)
    event_groups = events.groupby("symbol", sort=False)
    execution_groups = execution.groupby("symbol", sort=False)
    all_events = []
    all_execution = []
    coverage = []
    symbols = sorted(set(events.symbol) & set(source_groups.groups))
    for number, symbol in enumerate(symbols, 1):
        features = direction_features(source_groups.get_group(symbol))
        if features.empty:
            continue
        known = features[
            [
                "date",
                *DAILY_FEATURES,
                *WEEKLY_FEATURES,
                *WEEKLY_GATES,
                "weekly_source_date",
                "available_at",
                "left_pullback",
                "left_stabilize",
                "right_breakout",
                "right_confirm",
            ]
        ]
        candidates = event_groups.get_group(symbol)
        for horizon in [5, 10, 20]:
            labeled = forward_event_labels(features, candidates, calendar, horizon)
            labeled = labeled.merge(known, on="date", how="left", validate="many_to_one")
            for column in [*DAILY_FEATURES, *WEEKLY_FEATURES]:
                labeled[column] = labeled[column].astype(np.float32)
            all_events.append(labeled)
        if symbol in execution_groups.groups:
            prices = features[["date", "close"]].merge(
                execution_groups.get_group(symbol),
                on="date",
                suffixes=("_adjusted", "_raw"),
                validate="one_to_one",
            )
            valid = prices.valid_bar.fillna(False).astype(bool) & prices.vol.gt(0)
            all_execution.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "date": prices.date,
                        "close": prices.close_adjusted,
                        "buy_ok": valid & prices.close_raw.lt(prices.up_limit - 1e-8),
                        "sell_ok": valid & prices.close_raw.gt(prices.down_limit + 1e-8),
                    }
                )
            )
        coverage.append(
            {
                "symbol": symbol,
                "source_candidates": len(candidates),
                "historical_bars": len(features),
            }
        )
        if number % 500 == 0:
            print("prepared", number, "/", len(symbols), flush=True)
    combined = pd.concat(all_events, ignore_index=True)
    prices = pd.concat(all_execution, ignore_index=True)
    combined.to_parquet(out / "events.parquet", index=False)
    prices.to_parquet(out / "execution_bars.parquet", index=False)
    save_json(out / "calendar.json", [str(x.date()) for x in calendar])
    save_json(out / "input_sources.json", inputs)
    right = combined[combined.side.eq("right") & combined.horizon.eq(5)]
    diff = (right.net_return + 0.002 - right.terminal_return).abs().dropna()
    save_json(
        out / "coverage.json",
        {
            "source_candidates": len(events),
            "symbols_with_events": len(symbols),
            "rows_by_side_horizon": combined.groupby(["side", "horizon"])
            .size()
            .to_dict()
            .__str__(),
            "original_right_label_comparison": len(diff),
            "label_max_abs_error": float(diff.max()),
            "universe": "all preserved historical OOT candidates; not a freshly reconstructed entire listed universe",
            "signal_dates": [str(combined.date.min()), str(combined.date.max())],
            "missing_history_symbols": sorted(set(events.symbol) - set(symbols)),
        },
    )
    pd.DataFrame(coverage).to_csv(out / "symbol_coverage.csv", index=False)
    return combined, prices, calendar


def period_part(frame, calendar, name):
    start, end = PERIODS[name]
    available = calendar[(calendar >= pd.Timestamp(start)) & (calendar < pd.Timestamp(end))]
    # All holding horizons use the same signal cutoff, enough to mature a 20-session trade.
    cutoff = available[-22]
    return frame[
        frame.date.ge(start) & frame.date.le(cutoff) & frame.exit_date.lt(end)
    ].copy(), available


def subset_histories(prices, symbols):
    return {
        str(symbol): part.drop(columns="symbol").copy()
        for symbol, part in prices[prices.symbol.isin(symbols)].groupby("symbol", sort=False)
    }


def backtest(part, prices, calendar, score="baseline_percentile"):
    eligible = part.dropna(subset=[score])
    picks = top_picks(eligible, score).copy()
    picks["execution_score"] = picks[score]
    histories = subset_histories(prices, picks.symbol.unique())
    return capital_backtest(picks, histories, calendar, slots=20, cost=0.002)


def rule_mask(frame, week, entry):
    return frame[week].fillna(False) & (frame[entry].fillna(False) if entry != "original" else True)


def supported(part, selection=False):
    metrics = event_metrics(part)
    wins = int(part.net_return.gt(0).sum())
    lower = wilson_lower(wins, len(part))
    passed = (
        len(part) >= 100
        and (metrics["win_rate"] or 0) >= (0.60 if selection else 0.50)
        and (metrics["profit_factor"] or 0) > (1.2 if selection else 1.0)
        and (metrics["average_net_return"] or 0) > 0
    )
    if selection:
        passed = passed and lower >= 0.5
    return bool(passed), {**metrics, "wilson_lower": lower}


def model_predictions(frame, calendar, out, side):
    features = [*DAILY_FEATURES, *WEEKLY_FEATURES, "baseline_percentile"]
    result = frame[frame.date.ge("2025-01-01") & frame.macd_boll].copy()
    result["model_probability"] = np.nan
    directory = out / "models" / side
    directory.mkdir(parents=True)
    audits = []
    dates = calendar[(calendar >= pd.Timestamp("2025-01-01")) & (calendar <= frame.date.max())]
    for refit, segment in refit_segments(dates, "monthly"):
        train = recent_window(frame, refit, 12)
        train = train[train.macd_boll]
        pending = result[result.date.isin(segment)]
        if len(train) < 500 or train.net_return.gt(0).nunique() < 2 or pending.empty:
            continue
        if len(train) > 80000:
            train = train.sample(80000, random_state=20261021)
        weights = recency_weights(train.date, refit, 90, equal_dates=True)
        model = HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=10,
            learning_rate=0.05,
            early_stopping=False,
            random_state=20261021,
        )
        model.fit(train[features], train.net_return.gt(0), sample_weight=weights)
        result.loc[pending.index, "model_probability"] = model.predict_proba(pending[features])[
            :, 1
        ]
        joblib.dump(model, directory / f"{refit:%Y%m%d}.joblib")
        audits.append(
            {
                "side": side,
                "refit_date": refit,
                "train_rows": len(train),
                "max_train_exit": train.exit_date.max(),
                "forecast_start": pending.date.min(),
            }
        )
    return result, audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    protocol = {
        "weekly_gates": list(WEEKLY_GATES),
        "entries": {
            "left": ["original", "left_pullback", "left_stabilize"],
            "right": ["original", "right_breakout", "right_confirm"],
        },
        "horizons": [5, 10, 20],
        "rule_count_per_side": 54,
        "periods": PERIODS,
        "selection": "2024 and2025H1 positive supported,2025H2>=60% andPF>1.2; no test tuning",
        "cash": "no qualifying rule -> all cash; separately report frozen best exploratory candidate",
        "model": "fixed MACD+BOLL gate;20-day target;monthly12month90day-decay shallow HistGradientBoosting;thresholds2025H2 only",
        "source_hashes": {
            str(p): sha256(p)
            for p in [
                Path(__file__),
                ROOT / "src/quant/research/trend_research.py",
                ROOT / "src/quant/research/sequence_strategy_validation.py",
            ]
        },
        "test": "2026 already-observed period, expanded stocks unseen by earlier256-stock studies",
        "cost": 0.002,
        "slots": 20,
        "entry": "next_close",
        "weekly_availability": "next natural week start",
    }
    save_json(out / "protocol.json", protocol)
    if args.dataset:
        data = args.dataset.resolve()
        events = pd.read_parquet(data / "events.parquet")
        prices = pd.read_parquet(data / "execution_bars.parquet")
        calendar = pd.DatetimeIndex(
            pd.to_datetime(json.loads((data / "calendar.json").read_text()))
        )
        save_json(
            out / "dataset_reference.json",
            {
                "path": str(data),
                "events_sha256": sha256(data / "events.parquet"),
                "execution_sha256": sha256(data / "execution_bars.parquet"),
            },
        )
    else:
        events, prices, calendar = prepare(out)
    audits = []
    choices = []
    rule_stats = []
    results = []
    model_audits = []
    case_frames = []
    old_symbols = set(
        json.loads(
            (ROOT / "reports/research/kronos_multiscale_expanded_v2/coverage.json").read_text()
        )["sampled_symbols"]
    )
    for side in ["left", "right"]:
        frame = events[events.side.eq(side)]
        entries = protocol["entries"][side]
        candidates = []
        for horizon in [5, 10, 20]:
            part = frame[frame.horizon.eq(horizon)]
            blocks = {
                period: period_part(part, calendar, period)[0]
                for period in ["development_2024", "development_2025H1", "selection"]
            }
            for week in WEEKLY_GATES:
                for entry in entries:
                    record = {"side": side, "week": week, "entry": entry, "horizon": horizon}
                    passed = True
                    for period, block in blocks.items():
                        sample = block[rule_mask(block, week, entry)]
                        ok, metrics = supported(sample, selection=period == "selection")
                        record[period] = metrics
                        passed = passed and ok
                    record["supported"] = passed
                    rule_stats.append(record)
                    # Selection is frozen before evaluating any2026 result.
                    sample = blocks["selection"][rule_mask(blocks["selection"], week, entry)]
                    if len(sample) >= 100:
                        _, _, metrics = backtest(
                            sample, prices, period_part(part, calendar, "selection")[1]
                        )
                        candidates.append({**record, "selection_portfolio": metrics})
        if not candidates:
            raise ValueError(f"no supported coverage for {side}")
        admissible = [
            c for c in candidates if c["supported"] and c["selection_portfolio"]["total_return"] > 0
        ]

        def score(candidate):
            return (
                candidate["selection_portfolio"]["total_return"],
                -abs(candidate["selection_portfolio"]["max_drawdown"]),
            )

        best = max(admissible if admissible else candidates, key=score)
        choices.append(
            {
                "side": side,
                "kind": "rule",
                "admitted": bool(admissible),
                "week": best["week"],
                "entry": best["entry"],
                "horizon": best["horizon"],
                "selection_return": best["selection_portfolio"]["total_return"],
            }
        )
        save_json(out / f"{side}_selection_frozen.json", choices[-1])
        print(side, "rule selection frozen", choices[-1], flush=True)
        model, model_audit = model_predictions(frame[frame.horizon.eq(20)], calendar, out, side)
        model_audits.extend(model_audit)
        model.to_parquet(out / f"{side}_model_predictions.parquet", index=False)
        model_candidates = []
        for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
            dev, _ = period_part(model, calendar, "development_2025H1")
            dev = dev[dev.model_probability.ge(threshold)]
            selection, active = period_part(model, calendar, "selection")
            selection = selection[selection.model_probability.ge(threshold)]
            dev_ok, _ = supported(dev)
            val_ok, stats = supported(selection, selection=True)
            _, _, metrics = backtest(selection, prices, active, score="model_probability")
            model_candidates.append(
                {
                    "threshold": threshold,
                    "supported": dev_ok and val_ok and metrics["total_return"] > 0,
                    "selection_portfolio": metrics,
                    "selection_events": stats,
                }
            )
        permitted = [c for c in model_candidates if c["supported"]]
        chosen = max(
            permitted if permitted else model_candidates,
            key=lambda c: c["selection_portfolio"]["total_return"],
        )
        choices.append(
            {
                "side": side,
                "kind": "model",
                "admitted": bool(permitted),
                "threshold": chosen["threshold"],
                "horizon": 20,
                "week": "macd_boll",
                "selection_return": chosen["selection_portfolio"]["total_return"],
            }
        )
        save_json(
            out / f"{side}_model_selection_frozen.json",
            {"chosen": choices[-1], "candidates": model_candidates},
        )
        # Evaluate only the preselected exploratory/admitted candidate, not all108 on test.
        for period in PERIODS:
            horizon = best["horizon"]
            base = frame[frame.horizon.eq(horizon)]
            block, active = period_part(base, calendar, period)
            rule = block[rule_mask(block, best["week"], best["entry"])]
            model_block, _ = period_part(model, calendar, period)
            model_block = model_block[model_block.model_probability.ge(chosen["threshold"])]
            versions = {
                "baseline_same_horizon": (block, "baseline_percentile"),
                "selected_rule_exploratory": (rule, "baseline_percentile"),
                "admitted_rule": (rule if admissible else rule.iloc[:0], "baseline_percentile"),
                "selected_model_exploratory": (model_block, "model_probability"),
                "admitted_model": (
                    model_block if permitted else model_block.iloc[:0],
                    "model_probability",
                ),
            }
            baseline_curve = None
            for name, (sample, column) in versions.items():
                curve, trades, metrics = backtest(sample, prices, active, score=column)
                if name == "baseline_same_horizon":
                    baseline_curve = curve
                results.append(
                    {
                        "side": side,
                        "period": period,
                        "variant": name,
                        "horizon": 20 if "model" in name else horizon,
                        **metrics,
                        "event_metrics": event_metrics(sample),
                        "gain": paired_block_gain(baseline_curve, curve, repetitions=500)
                        if name != "baseline_same_horizon" and "model" not in name
                        else None,
                    }
                )
                curve.to_csv(out / f"{side}_{period}_{name}_equity.csv", index=False)
                trades.to_parquet(out / f"{side}_{period}_{name}_trades.parquet", index=False)
                if period == "test" and name in [
                    "selected_rule_exploratory",
                    "selected_model_exploratory",
                ]:
                    cases = sample.sort_values("net_return")
                    cases = pd.concat([cases.head(30), cases.tail(30)]).copy()
                    cases["case_variant"] = name
                    case_frames.append(cases)
                    for cohort, mask in [
                        ("prior_256", sample.symbol.isin(old_symbols)),
                        ("additional_stocks", ~sample.symbol.isin(old_symbols)),
                    ]:
                        sub = sample[mask]
                        audits.append(
                            {
                                "side": side,
                                "variant": name,
                                "cohort": cohort,
                                **event_metrics(sub),
                                "symbols": sub.symbol.nunique(),
                            }
                        )
    save_json(out / "selection_decision.json", choices)
    save_json(out / "rule_search.json", rule_stats)
    save_json(out / "results.json", results)
    pd.DataFrame(model_audits).to_csv(out / "model_refit_audit.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "cohort_audit.csv", index=False)
    if case_frames:
        pd.concat(case_frames).to_csv(out / "test_cases_diagnostic_only.csv", index=False)
    lines = [
        "# 周线方向、日线买点研究",
        "",
        "所有规则和模型先在2024—2025筛选，再仅复核选中的版本。2026是已观察时间区间；扩展股票单列。",
        "",
        "|侧|版本|持有日|2025H2收益|2026收益|2026胜率|2026成交数|回撤|平均敞口|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in results if r["period"] == "test"]:
        val = next(
            r
            for r in results
            if r["side"] == row["side"]
            and r["variant"] == row["variant"]
            and r["period"] == "selection"
        )
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['variant']}|{row['horizon']}|{val['total_return']:.2%}|{row['total_return']:.2%}|{win}|{row['trades']}|{row['max_drawdown']:.2%}|{row['mean_exposure']:.1%}|"
        )
    lines += ["", "预先选择记录："] + [f"- {choice}" for choice in choices]
    lines += [
        "",
        "exploratory表示未必达到跨期准入条件，只是2025H2选择的诊断版本；不能把它当作高成功率策略上线。",
        "每侧54个固定规则组合，另有5个模型阈值，存在多重比较。Wilson按事件计算忽略了收益重叠和股票共振，只用于预筛，不作为独立统计显著性证据。",
        "周线采用上一完整自然周，周一才可见。所有持有期使用同一信号截止日，保证20日标签成熟；资金曲线涵盖整个评价期间。",
        "基准与规则使用相同持有期；模型固定20日，若规则持有期不同，不直接把两者差额当模型增益。",
        "原历史候选定义与评分模型本身没有重新生成，当前结论适用于保留候选池，未覆盖所有可能出现的股票信号。",
        "当期成交数为已平仓交易，未平仓单列；现金、缺行情、涨跌停、重复持仓及20仓位上限均按既有回测约束。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    save_json(
        out / "complete.json",
        {"status": "research_complete_not_production", "seconds": time.time() - started},
    )
    print((out / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
