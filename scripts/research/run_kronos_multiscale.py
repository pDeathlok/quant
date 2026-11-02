#!/usr/bin/env python
"""Reproducible multiscale Kronos pretraining and two-sided strategy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from quant.research.kronos_multiscale import (
    SCALES,
    KronosEOS,
    asof_windows,
    context_and_target,
    fit_subscores,
    period_bars,
    predict_subscores,
    validate_bars,
)
from quant.research.sequence_strategy_validation import (
    capital_backtest,
    choose_confidence_gate,
    choose_rule,
    event_metrics,
    paired_block_gain,
    purged_split,
    rule_text,
    top_picks,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "return1",
    "return5",
    "return20",
    "return60",
    "volume5",
    "volume20",
    "close_position",
    "range",
    "ma20_distance",
    "volatility20",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n"
    )


def load_predictions(config: dict) -> pd.DataFrame:
    frames = []
    for side, score in [
        ("left", "pred_unified_left_long_task_deep"),
        ("right", "pred_unified_long_task_deep"),
    ]:
        frame = pd.read_parquet(ROOT / config[f"{side}_predictions"])
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        frame = frame[frame.entry_mode.eq("next_close") & frame.horizon.eq(5)].copy()
        frame["side"] = side
        frame["baseline_score"] = frame[score]
        keep = ["symbol", "date", "side", "baseline_score"]
        if "terminal_return" in frame:
            keep += ["terminal_return"]
        frame = frame[keep]
        if frame.duplicated(["symbol", "date"]).any():
            raise ValueError(f"duplicate {side} historical predictions")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_histories(config: dict, events: pd.DataFrame, output: Path):
    universe_path = ROOT / "data/raw/daily_partitioned/year_month=202212/data.parquet"
    # Freeze membership before the evaluation years, including stocks later delisted.
    universe = pd.read_parquet(universe_path, columns=["ts_code"]).ts_code.unique()
    symbols = sorted(universe, key=lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    if config["symbol_limit"]:
        symbols = symbols[: config["symbol_limit"]]
    # Sample before observing returns/coverage; newly listed stocks can lack monthly history.
    selected = events[events.symbol.isin(symbols)].copy()
    raw_path = ROOT / config["raw_bars"]
    raw = pd.read_parquet(raw_path, filters=[("symbol", "in", symbols)])
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    calendar = pd.DatetimeIndex(
        pd.read_parquet(raw_path, columns=["date"]).date.unique()
    ).sort_values()
    full_sources = [
        {"path": str(raw_path), "sha256": digest(raw_path)},
        {"path": str(universe_path), "sha256": digest(universe_path)},
    ]
    historical_parts = []
    for p in sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet")):
        if p.parent.name.rsplit("=", 1)[-1] >= "202401":
            continue
        part = pd.read_parquet(
            p,
            columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol"],
            filters=[("ts_code", "in", symbols)],
        )
        part = part.rename(columns={"ts_code": "symbol"})
        part["date"] = pd.to_datetime(part.trade_date.astype(str), format="%Y%m%d")
        historical_parts.append(part.drop(columns="trade_date"))
        full_sources.append({"path": str(p), "sha256": digest(p)})
    historical = pd.concat([*historical_parts, raw.reset_index(drop=True)], ignore_index=True)
    historical = historical.sort_values(["symbol", "date"])
    if historical.duplicated(["symbol", "date"]).any():
        raise ValueError("overlapping raw history sources")
    raw = raw.set_index(["symbol", "date"])
    limits_parts = []
    for p in sorted((ROOT / config["limits_dir"]).glob("*.parquet")):
        part = pd.read_parquet(p)
        part = part[part.ts_code.isin(symbols)]
        if len(part):
            full_sources.append({"path": str(p), "sha256": digest(p)})
            part = part.rename(columns={"ts_code": "symbol"})
            part["date"] = pd.to_datetime(part.trade_date.astype(str), format="%Y%m%d")
            limits_parts.append(part[["symbol", "date", "up_limit", "down_limit"]])
    if not limits_parts:
        raise ValueError("historical limit data are required for capital backtests")
    limits = (
        pd.concat(limits_parts).drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])
    )
    # Resume checks data as well as code; an old embedding must never survive a source change.
    input_manifest = output / "raw_sources.json"
    if input_manifest.exists() and json.loads(input_manifest.read_text()) != full_sources:
        raise ValueError("raw data changed since previous execution")
    save_json(input_manifest, full_sources)
    histories, periods, records = {}, {}, []
    manifests = []
    for symbol in symbols:
        source = historical[historical.symbol.eq(symbol)].copy()
        source = source[source[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)]
        if not len(source):
            manifests.append({"symbol": symbol, "status": "missing raw history"})
            continue
        # Causal price index: corporate-action jumps are removed using the same
        # session's reference price; no future adjustment factor enters earlier bars.
        close_index = np.exp(np.log(source.close / source.pre_close).cumsum()) * 100
        multiplier = close_index / source.close
        adjusted = source[["date", "open", "high", "low", "close"]].copy()
        adjusted[["open", "high", "low", "close"]] = source[["open", "high", "low", "close"]].mul(
            multiplier, axis=0
        )
        adjusted["volume"] = source.vol.to_numpy()
        try:
            daily = validate_bars(adjusted)
        except ValueError as exc:
            manifests.append({"symbol": symbol, "status": str(exc)})
            continue
        daily["symbol"] = symbol
        known_features = daily.copy()
        for n in [1, 5, 20, 60]:
            known_features[f"return{n}"] = daily.close.pct_change(n, fill_method=None)
        for n in [5, 20]:
            known_features[f"volume{n}"] = daily.volume / daily.volume.rolling(n).mean().replace(
                0, np.nan
            )
        known_features["close_position"] = (daily.close - daily.low) / (
            daily.high - daily.low
        ).replace(0, np.nan)
        known_features["range"] = (daily.high - daily.low) / daily.close
        known_features["ma20_distance"] = daily.close / daily.close.rolling(20).mean() - 1
        known_features["volatility20"] = known_features.return1.rolling(20).std()
        known_features["close_position"] = known_features.close_position.fillna(0.5)
        known_features = known_features.set_index("date")
        history = daily[["date", "close", "volume"]].copy()
        keys = pd.MultiIndex.from_arrays([[symbol] * len(daily), daily.date])
        raw_rows = raw.reindex(keys).reset_index(drop=True)
        limit_rows = limits.reindex(keys).reset_index(drop=True)
        # Raw prices and actual point-in-time limits, never adjusted prices, gate fills.
        valid = raw_rows.valid_bar.fillna(False).astype(bool) & raw_rows.vol.gt(0)
        history["buy_ok"] = (valid & raw_rows.close.lt(limit_rows.up_limit - 1e-8)).to_numpy()
        history["sell_ok"] = (valid & raw_rows.close.gt(limit_rows.down_limit + 1e-8)).to_numpy()
        histories[symbol] = history
        periods[symbol] = {scale: period_bars(daily, scale) for scale in SCALES}
        date_positions = pd.Series(np.arange(len(daily)), index=daily.date)
        for event in selected[selected.symbol.eq(symbol)].itertuples():
            signal_pos = int(calendar.searchsorted(event.date))
            if signal_pos + 6 >= len(calendar):
                continue
            entry, exit_ = calendar[signal_pos + 1], calendar[signal_pos + 6]
            if (
                event.date not in date_positions
                or entry not in date_positions
                or exit_ not in date_positions
            ):
                continue
            window = asof_windows(periods[symbol], event.date, config["lengths"])
            if window is None:
                continue
            feature_row = known_features.loc[event.date, FEATURES].to_numpy(float)
            if not np.isfinite(feature_row).all():
                continue
            # Reject suspended/missing sessions in the label path rather than shift the horizon.
            needed = calendar[signal_pos + 1 : signal_pos + 7]
            path_rows = daily.set_index("date").reindex(needed)
            if path_rows.close.isna().any() or path_rows.volume.le(0).any():
                continue
            return_ = float(
                known_features.loc[exit_, "close"] / known_features.loc[entry, "close"] - 1
            )
            record = {
                "symbol": symbol,
                "date": event.date,
                "side": event.side,
                "baseline_score": event.baseline_score,
                "entry_date": entry,
                "exit_date": exit_,
                "net_return": return_ - config["round_trip_cost"],
                "terminal_return_rebuilt": return_,
                **dict(zip(FEATURES, feature_row)),
            }
            original = getattr(event, "terminal_return", float("nan"))
            if pd.notna(original):
                record["original_terminal_return"] = original
            records.append(record)
        manifests.append(
            {
                "symbol": symbol,
                "status": "loaded",
                "rows": len(daily),
                "start": daily.date.min(),
                "end": daily.date.max(),
            }
        )
    save_json(output / "history_sources.json", manifests)
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("no eligible multiscale events")
    label_delta = (
        (
            result.get("original_terminal_return", pd.Series(dtype=float))
            - result.terminal_return_rebuilt
        )
        .abs()
        .dropna()
    )
    save_json(
        output / "coverage.json",
        {
            "sampled_symbols": symbols,
            "available_histories": len(histories),
            "historical_prediction_rows": len(selected),
            "eligible_rows": len(result),
            "eligible_by_side_year": result.groupby(["side", result.date.dt.year])
            .size()
            .to_dict()
            .__str__(),
            "label_comparison_rows": len(label_delta),
            "label_mean_absolute_difference": float(label_delta.mean())
            if len(label_delta)
            else None,
            "label_max_absolute_difference": float(label_delta.max()) if len(label_delta) else None,
            "limitations": [
                "Hash-sampled pilot; not representative of all A shares",
                "Causal close/pre_close price index removes corporate-action jumps; not a cash-dividend total-return account",
                "Historical teacher scores are frozen OOT artifacts, not a current-model historical rescore",
                "Only events with a mature fixed-horizon label and complete multiscale history are evaluated",
            ],
        },
    )
    frozen_events = output / "events.parquet"
    if frozen_events.exists():
        prior = pd.read_parquet(frozen_events)
        pd.testing.assert_frame_equal(result, prior)
    return result, histories, periods, calendar


def pretrain(config: dict, periods: dict, output: Path, device: str) -> KronosEOS:
    torch.manual_seed(config["seed"])
    model = KronosEOS(ROOT / config["source_dir"], **config["encoder"]).to(device)
    generator = np.random.default_rng(config["seed"])
    datasets = {}
    validation = {}
    coverage = []
    cutoff = pd.Timestamp(config["pretrain_end"])
    validation_end = pd.Timestamp(config["pretrain_validation_end"])
    for scale in SCALES:
        x, y, vx, vy = [], [], [], []
        length = config["lengths"][scale]
        for symbol, series in periods.items():
            frame = series[scale]
            train_ends = [
                i
                for i in range(length - 1, len(frame) - 1)
                if frame.iloc[i + 1].available_at <= cutoff
            ]
            val_ends = [
                i
                for i in range(length - 1, len(frame) - 1)
                if cutoff < frame.iloc[i].available_at
                and frame.iloc[i + 1].available_at <= validation_end
            ]
            for ends, xx, yy in [(train_ends, x, y), (val_ends, vx, vy)]:
                sampled = generator.choice(
                    ends, min(len(ends), config["pretrain_windows_per_symbol_scale"]), replace=False
                )
                for end in sampled:
                    context, target = context_and_target(frame, int(end), length)
                    xx.append(context)
                    yy.append(target)
            coverage.append(
                {"symbol": symbol, "scale": scale, "eligible_training_windows": len(train_ends)}
            )
        if not len(x) or not len(vx):
            raise ValueError(f"no train/validation pretraining windows for {scale}")
        datasets[scale] = (np.stack(x), np.stack(y))
        validation[scale] = (np.stack(vx), np.stack(vy))
    save_json(output / "pretrain_coverage.json", coverage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    batch_size = config["pretrain_batch_size"]
    best, training_log = float("inf"), []
    for epoch in range(config["pretrain_epochs"]):
        train_losses = []
        model.train()
        for scale_id, scale in enumerate(SCALES):
            x, y = datasets[scale]
            order = generator.permutation(len(x))
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                prediction = model.next_bar(model(torch.from_numpy(x[idx]).to(device), scale_id))
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction, torch.from_numpy(y[idx]).to(device)
                )
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite pretraining loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for scale_id, scale in enumerate(SCALES):
                x, y = validation[scale]
                for start in range(0, len(x), batch_size):
                    prediction = model.next_bar(
                        model(torch.from_numpy(x[start : start + batch_size]).to(device), scale_id)
                    )
                    loss = torch.nn.functional.smooth_l1_loss(
                        prediction, torch.from_numpy(y[start : start + batch_size]).to(device)
                    )
                    validation_losses.append(float(loss))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": float(np.mean(validation_losses)),
        }
        training_log.append(record)
        print("pretrain", record, flush=True)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            torch.save(model.state_dict(), output / "encoder.pt")
        save_json(output / "pretrain_log.json", training_log)
    model.load_state_dict(torch.load(output / "encoder.pt", map_location=device, weights_only=True))
    return model.eval()


def encode(
    config: dict, model: KronosEOS, events: pd.DataFrame, periods: dict, output: Path, device: str
) -> tuple[pd.DataFrame, np.ndarray]:
    keys = (
        events[["symbol", "date"]]
        .drop_duplicates()
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    embeddings = []
    batch_size = config["embedding_batch_size"]
    for number, (symbol, part) in enumerate(keys.groupby("symbol", sort=False), 1):
        samples = [
            asof_windows(periods[symbol], row.date, config["lengths"]) for row in part.itertuples()
        ]
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                windows = [
                    torch.from_numpy(np.stack([s[scale] for s in batch])).to(device)
                    for scale in range(3)
                ]
                embeddings.append(model.concatenate(windows).cpu().numpy())
        print(f"encoded {number}/{keys.symbol.nunique()} {symbol}, events={len(part)}", flush=True)
    array = np.concatenate(embeddings).astype(np.float32)
    keys["embedding_row"] = np.arange(len(keys))
    keys.to_parquet(output / "embedding_keys.parquet", index=False)
    np.save(output / "embeddings.npy", array, allow_pickle=False)
    events = events.merge(keys, on=["symbol", "date"], how="left", validate="many_to_one")
    return events, array


def evaluate(
    config: dict,
    events: pd.DataFrame,
    embeddings: np.ndarray,
    histories: dict,
    calendar: pd.DatetimeIndex,
    output: Path,
) -> None:
    results, all_cases = [], []
    for side in ["left", "right"]:
        frame = events[events.side.eq(side)].copy()
        splits = purged_split(frame)
        if any(len(part) < 30 for part in splits.values()):
            raise ValueError(f"{side} has insufficient chronological split coverage")
        train = splits["train"]
        model = fit_subscores(
            embeddings[train.embedding_row],
            train.net_return.to_numpy(),
            epochs=config["subscore_epochs"],
            seed=config["seed"],
        )
        torch.save(model.state_dict(), output / f"{side}_subscore_head.pt")
        subscores = predict_subscores(model, embeddings[frame.embedding_row])
        frame[["sequence_return", "sequence_profit", "sequence_downside"]] = subscores
        # Stacking uses predictions from a head fit without the stacking period's labels.
        feedback = frame[frame.index.isin(splits["feedback"].index)]
        feedback_summary = (
            feedback.assign(outcome=np.where(feedback.net_return > 0, "win", "loss"))
            .groupby("outcome")[FEATURES + ["net_return"]]
            .mean()
        )
        feedback_summary.to_csv(output / f"{side}_feedback_case_summary.csv")
        fusion_features = [
            "baseline_score",
            "sequence_return",
            "sequence_profit",
            "sequence_downside",
        ]
        fusion = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        fusion.fit(feedback[fusion_features], feedback.net_return.gt(0))
        frame["fusion_score"] = fusion.predict_proba(frame[fusion_features])[:, 1]
        joblib.dump(fusion, output / f"{side}_fusion.joblib")
        # Refine after feedback cases, select only in the subsequent half-year.
        train_with_feedback = frame[
            frame.index.isin(pd.concat([splits["train"], splits["feedback"]]).index)
        ]
        selection = frame[frame.index.isin(splits["selection"].index)]
        test = frame[frame.index.isin(splits["test"].index)].copy()
        rule_models = {}
        for name, supervised, fitting in [
            ("distilled_rules", False, frame[frame.index.isin(train.index)]),
            ("feedback_rules", True, train_with_feedback),
        ]:
            tree, leaves, audit = choose_rule(
                fitting,
                selection,
                FEATURES,
                supervised=supervised,
                minimum=config["minimum_rule_samples"],
            )
            joblib.dump(
                {"tree": tree, "leaves": leaves, "features": FEATURES},
                output / f"{side}_{name}.joblib",
            )
            (output / f"{side}_{name}.txt").write_text(rule_text(tree, FEATURES, leaves))
            save_json(output / f"{side}_{name}_selection.json", audit)
            rule_models[name] = (tree, leaves)
        candidates = {
            name: top_picks(test, score, config["slots"]).assign(execution_score=lambda d: d[score])
            for name, score in [
                ("baseline", "baseline_score"),
                ("sequence_only", "sequence_profit"),
                ("fusion", "fusion_score"),
            ]
        }
        for name, (tree, leaves) in rule_models.items():
            picked = test[np.isin(tree.apply(test[FEATURES]), leaves)].copy()
            picked["execution_score"] = (
                tree.predict(picked[FEATURES]) if len(picked) else pd.Series(dtype=float)
            )
            candidates[name] = picked
        for name, score in [
            ("sequence_cash_gate", "sequence_profit"),
            ("fusion_cash_gate", "fusion_score"),
        ]:
            threshold, audit = choose_confidence_gate(
                selection, score, minimum=config["minimum_rule_samples"], top_n=config["slots"]
            )
            save_json(
                output / f"{side}_{name}_selection.json", {"threshold": threshold, "search": audit}
            )
            eligible = test[test[score] >= threshold] if threshold is not None else test.iloc[:0]
            candidates[name] = top_picks(eligible, score, config["slots"]).assign(
                execution_score=lambda d: d[score]
            )
        baseline_curve = None
        for name, picks in candidates.items():
            active_calendar = calendar[
                (calendar >= pd.Timestamp("2026-01-01")) & (calendar <= test.exit_date.max())
            ]
            curve, trades, metrics = capital_backtest(
                picks,
                histories,
                active_calendar,
                slots=config["slots"],
                cost=config["round_trip_cost"],
            )
            curve.to_csv(output / f"{side}_{name}_equity.csv", index=False)
            trades.to_parquet(output / f"{side}_{name}_trades.parquet", index=False)
            if name == "baseline":
                baseline_curve = curve
            gain = paired_block_gain(baseline_curve, curve) if name != "baseline" else None
            results.append(
                {
                    "side": side,
                    "variant": name,
                    **metrics,
                    "incremental_gain": gain,
                    "candidate_metrics": event_metrics(picks),
                    "test_available_candidates": len(test),
                    "test_selected_candidates": len(picks),
                }
            )
        # Cases are diagnostics, never fed back from the held-out year.
        cases = feedback.sort_values("net_return").groupby("side").head(20).copy()
        cases["case_role"] = "validation_feedback_loss"
        all_cases.append(cases)
        test_cases = (
            test.sort_values("net_return")
            .iloc[
                np.r_[np.arange(min(20, len(test))), np.arange(max(20, len(test) - 20), len(test))]
            ]
            .copy()
        )
        test_cases["case_role"] = "test_diagnostic_do_not_tune"
        all_cases.append(test_cases)
        frame.to_parquet(output / f"{side}_scored_events.parquet", index=False)
    pd.concat(all_cases).to_csv(output / "cases.csv", index=False)
    save_json(output / "results.json", results)
    lines = [
        "# Kronos 多周期量价验证",
        "",
        "固定抽样研究；未替换生产模型。2024训练、2025上半年反馈/融合训练、下半年规则选择、2026时间外验证。",
        "",
        "|侧|版本|成交数|胜率|组合收益|最大回撤|空仓天数占比|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        win = f"{row['win_rate']:.1%}" if row["win_rate"] is not None else "—"
        lines.append(
            f"|{row['side']}|{row['variant']}|{row['trades']}|{win}|{row['total_return']:.2%}|{row['max_drawdown']:.2%}|{row['cash_day_ratio']:.1%}|"
        )
    lines += [
        "",
        "规则不满足验证期准入条件时保持现金。无交易不等于100%胜率。",
        "输入与交易覆盖、复权快照限制见 coverage.json；仅对匹配样本报告增益，不外推全市场。",
        "事件标签为5日固定持有的净收益；组合执行在卖出受限时延期，因此组合成交收益与标签可能不同。",
    ]
    lines += ["", "## 增益检验", "", "五日区块配对bootstrap的95%区间，用于诊断，尚未校正多重比较。"]
    for row in results:
        if row["variant"] in ["sequence_only", "fusion"]:
            gain = row["incremental_gain"]
            lower, upper = gain["paired_block_95ci"]
            lines.append(
                f"- {row['side']} {row['variant']}: 增益{gain['gain']:.2%}，区间[{lower:.2%}, {upper:.2%}]，正增益证据={'有' if gain['positive_gain_supported'] else '不足'}。"
            )
    lines += [
        "",
        "首轮64只股票的部分2026结果已经查看；本轮扩样是固定协议的稳健性复核，不能宣称2026完全未被观察。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = ROOT / config["output_dir"]
    if output.exists() and not args.resume:
        raise ValueError("output exists; explicitly --resume or choose a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "config": config,
        "script_sha256": digest(Path(__file__)),
        "encoder_source_sha256": digest(ROOT / "src/quant/research/kronos_multiscale.py"),
        "validation_source_sha256": digest(
            ROOT / "src/quant/research/sequence_strategy_validation.py"
        ),
        "kronos_module_sha256": digest(ROOT / config["source_dir"] / "model/module.py"),
        "left_predictions_sha256": digest(ROOT / config["left_predictions"]),
        "right_predictions_sha256": digest(ROOT / config["right_predictions"]),
        "initialization": "random; no external pretrained weights",
        "pretrain_target": "next OHLCV via actual EOS; no target-dependent context normalization",
    }
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != fingerprint:
        raise ValueError("resume protocol/source changed")
    save_json(protocol, fingerprint)
    torch.set_num_threads(4)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print("device", device, flush=True)
    started = time.time()
    predictions = load_predictions(config)
    events, histories, periods, calendar = load_histories(config, predictions, output)
    events.to_parquet(output / "events.parquet", index=False)
    print("eligible", len(events), "symbols", events.symbol.nunique(), flush=True)
    if (output / "pretrain_complete.json").exists():
        model = KronosEOS(ROOT / config["source_dir"], **config["encoder"]).to(device)
        model.load_state_dict(
            torch.load(output / "encoder.pt", map_location=device, weights_only=True)
        )
        model.eval()
    else:
        model = pretrain(config, periods, output, device)
        save_json(
            output / "pretrain_complete.json", {"encoder_sha256": digest(output / "encoder.pt")}
        )
    if (output / "embedding_keys.parquet").exists() and (output / "embeddings.npy").exists():
        keys = pd.read_parquet(output / "embedding_keys.parquet")
        embeddings = np.load(output / "embeddings.npy", allow_pickle=False)
        events = events.merge(keys, on=["symbol", "date"], validate="many_to_one")
    else:
        events, embeddings = encode(config, model, events, periods, output, device)
    del model
    evaluate(config, events, embeddings, histories, calendar, output)
    save_json(
        output / "complete.json",
        {
            "status": "research_complete_not_production",
            "seconds": time.time() - started,
            "rows": len(events),
            "encoder_sha256": digest(output / "encoder.pt"),
            "embeddings_sha256": digest(output / "embeddings.npy"),
        },
    )
    print((output / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
