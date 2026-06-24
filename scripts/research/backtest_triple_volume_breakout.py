#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backtest 三倍量缩量盘整突破 with multiple short-term exit plans."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.features.variable_library import build_continuous_ohlc
from quant.strategies.custom.triple_volume_breakout import add_triple_volume_research_signals


DEFAULT_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/triple_volume_breakout"


@dataclass(frozen=True)
class ExitRule:
    name: str
    kind: str
    hold_days: int
    take_profit: float | None = None
    stop_loss: float | None = None
    trail_drawdown: float | None = None
    ma_exit: str | None = None


def read_daily_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "volume" not in df.columns and "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "symbol" not in df.columns:
        df["symbol"] = df["ts_code"].astype(str) if "ts_code" in df.columns else path.stem
    return df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)


def _add_research_signal_columns(df: pd.DataFrame, volume_multiple: float = 3.0) -> pd.DataFrame:
    return add_triple_volume_research_signals(df, volume_multiple=volume_multiple)


def build_candidates(
    daily_dir: Path,
    start_date: str,
    max_workers: int = 1,
    signal_mode: str = "strict",
    volume_multiple: float = 3.0,
) -> pd.DataFrame:
    files = sorted(daily_dir.glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    start_ts = pd.to_datetime(start_date)

    def process(path: Path) -> pd.DataFrame | None:
        try:
            daily = read_daily_file(path)
            if len(daily) < 90:
                return None
            signal_frame = _add_research_signal_columns(daily, volume_multiple=volume_multiple)
            signal_col = f"signal_{signal_mode}"
            if signal_col not in signal_frame.columns:
                raise ValueError(f"Unsupported signal mode: {signal_mode}")
            signals = signal_frame[
                (signal_frame["date"] >= start_ts)
                & (signal_frame[signal_col] == 1)
            ].copy()
            if signals.empty:
                return None
            keep = [
                "date",
                "symbol",
                "name",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "days_since_triple_volume",
                "triple_volume_price",
                "consolidation_range",
                "breakout_pct",
                "volume_dryness",
                "ma20_slope_5d",
                "volume_recovery",
                "candidate_score",
            ]
            present = [col for col in keep if col in signals.columns]
            return signals[present]
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
            return None

    if max_workers <= 1:
        for n, path in enumerate(files, start=1):
            result = process(path)
            if result is not None:
                frames.append(result)
            if n % 500 == 0 or n == len(files):
                print(f"  candidates: {n}/{len(files)} files")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, path) for path in files]
            for n, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result is not None:
                    frames.append(result)
                if n % 500 == 0 or n == len(files):
                    print(f"  candidates: {n}/{len(files)} files")

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_future_prices(candidates: pd.DataFrame, daily_dir: Path, max_hold_days: int) -> pd.DataFrame:
    frames = []
    for symbol in candidates["symbol"].dropna().unique():
        path = daily_dir / f"{symbol}.parquet"
        if not path.exists():
            path = daily_dir / f"{str(symbol).replace('.SH', '').replace('.SZ', '').replace('.BJ', '')}.parquet"
        if not path.exists():
            continue
        daily = build_continuous_ohlc(read_daily_file(path))
        daily["ma5"] = daily["close"].rolling(5).mean()
        daily["ma10"] = daily["close"].rolling(10).mean()
        future = daily[["symbol", "date", "close", "ma5", "ma10"]].copy()
        future["entry_open"] = daily["open"].shift(-1)
        for day in range(1, max_hold_days + 2):
            future[f"date_t{day}"] = daily["date"].shift(-day)
            for col in ["open", "high", "low", "close", "ma5", "ma10"]:
                future[f"{col}_t{day}"] = daily[col].shift(-day)
        frames.append(future)

    if not frames:
        raise RuntimeError(f"No future prices found under {daily_dir}")
    merged = candidates.merge(pd.concat(frames, ignore_index=True), on=["symbol", "date"], how="left", suffixes=("", "_px"))
    return merged.dropna(subset=["entry_open"]).copy()


def build_exit_rules() -> list[ExitRule]:
    rules: list[ExitRule] = []
    for hold in [3, 5, 8]:
        rules.append(ExitRule(f"time_T{hold + 1}_close", "time", hold))
    for hold in [5, 8]:
        for tp in [0.05, 0.08, 0.10]:
            for sl in [0.025, 0.035, 0.05]:
                rules.append(ExitRule(f"fixed_tp{tp:.0%}_sl{sl:.1%}_T{hold + 1}", "fixed", hold, tp, sl))
    for hold in [5, 8]:
        for tp in [0.06, 0.08, 0.10]:
            for trail in [0.025, 0.035, 0.05]:
                rules.append(ExitRule(f"trail_target{tp:.0%}_dd{trail:.1%}_T{hold + 1}", "trailing", hold, tp, 0.035, trail))
    for ma in ["ma5", "ma10"]:
        rules.append(ExitRule(f"breakout_or_{ma}_lost_T9", "technical", 8, 0.10, 0.035, None, ma))
    return rules


def simulate_exit(df: pd.DataFrame, rule: ExitRule) -> pd.DataFrame:
    entry = df["entry_open"].to_numpy(dtype=float)
    n = len(df)
    ret = np.full(n, np.nan)
    exit_day = np.full(n, -1, dtype=int)
    exit_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_type = np.full(n, "unknown", dtype=object)
    peak = np.zeros(n, dtype=float)
    active = np.zeros(n, dtype=bool)

    for day in range(2, rule.hold_days + 2):
        unresolved = np.isnan(ret) & ~np.isnan(entry)
        if not unresolved.any():
            break
        open_ = df[f"open_t{day}"].to_numpy(dtype=float)
        high = df[f"high_t{day}"].to_numpy(dtype=float)
        low = df[f"low_t{day}"].to_numpy(dtype=float)
        close = df[f"close_t{day}"].to_numpy(dtype=float)
        date_t = pd.to_datetime(df[f"date_t{day}"]).to_numpy(dtype="datetime64[ns]")
        valid = unresolved & ~np.isnan(close) & ~np.isnan(high) & ~np.isnan(low)
        if not valid.any():
            continue

        stop_price = entry * (1 - (rule.stop_loss or 0))
        stop_hit = valid & (rule.stop_loss is not None) & (low <= stop_price)
        if stop_hit.any():
            gap = stop_hit & (open_ <= stop_price)
            normal = stop_hit & ~gap
            ret[gap] = open_[gap] / entry[gap] - 1
            ret[normal] = stop_price[normal] / entry[normal] - 1
            exit_day[stop_hit] = day
            exit_date[stop_hit] = date_t[stop_hit]
            exit_type[stop_hit] = "stop_loss"

        still = valid & np.isnan(ret)
        if not still.any():
            continue

        if rule.kind == "fixed":
            tp_hit = still & (high >= entry * (1 + (rule.take_profit or 0)))
            ret[tp_hit] = rule.take_profit or 0
            exit_day[tp_hit] = day
            exit_date[tp_hit] = date_t[tp_hit]
            exit_type[tp_hit] = "take_profit"
        elif rule.kind == "trailing":
            peak[still] = np.maximum(peak[still], high[still])
            active |= still & (peak >= entry * (1 + (rule.take_profit or 0)))
            trail_price = peak * (1 - (rule.trail_drawdown or 0))
            trail_hit = still & active & (low <= trail_price)
            gap = trail_hit & (open_ <= trail_price)
            normal = trail_hit & ~gap
            ret[gap] = open_[gap] / entry[gap] - 1
            ret[normal] = trail_price[normal] / entry[normal] - 1
            exit_day[trail_hit] = day
            exit_date[trail_hit] = date_t[trail_hit]
            exit_type[trail_hit] = "trailing_stop"
        elif rule.kind == "technical":
            tp_hit = still & (high >= entry * (1 + (rule.take_profit or 0)))
            ret[tp_hit] = rule.take_profit or 0
            exit_day[tp_hit] = day
            exit_date[tp_hit] = date_t[tp_hit]
            exit_type[tp_hit] = "take_profit"
            still = valid & np.isnan(ret)
            ma = df[f"{rule.ma_exit}_t{day}"].to_numpy(dtype=float)
            lost = still & ((close < ma) | (close < df["triple_volume_price"].to_numpy(dtype=float)))
            ret[lost] = close[lost] / entry[lost] - 1
            exit_day[lost] = day
            exit_date[lost] = date_t[lost]
            exit_type[lost] = "technical_exit"

        expiry = valid & np.isnan(ret) & (day == rule.hold_days + 1)
        ret[expiry] = close[expiry] / entry[expiry] - 1
        exit_day[expiry] = day
        exit_date[expiry] = date_t[expiry]
        exit_type[expiry] = "expiry"

    result = df[["date", "symbol"]].copy()
    result["return_pct"] = ret * 100
    result["exit_day"] = exit_day
    result["exit_date"] = exit_date
    result["exit_type"] = exit_type
    return result.dropna(subset=["return_pct"])


def add_model_score(df: pd.DataFrame, train_end: str, label_hold_days: int = 8) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    max_high = pd.concat([out[f"high_t{day}"] for day in range(2, label_hold_days + 2)], axis=1).max(axis=1)
    min_low = pd.concat([out[f"low_t{day}"] for day in range(2, label_hold_days + 2)], axis=1).min(axis=1)
    out["label_up8_before_down3"] = ((max_high / out["entry_open"] - 1 >= 0.08) & (min_low / out["entry_open"] - 1 > -0.03)).astype(int)
    feature_cols = [
        "days_since_triple_volume",
        "consolidation_range",
        "breakout_pct",
        "volume_dryness",
        "ma20_slope_5d",
        "volume_recovery",
        "candidate_score",
    ]
    train_mask = out["date"] <= pd.to_datetime(train_end)
    train = out.loc[train_mask, feature_cols + ["label_up8_before_down3"]].replace([np.inf, -np.inf], np.nan).dropna(subset=["label_up8_before_down3"])
    meta = {"trained": False, "train_samples": int(len(train)), "positive_rate": None}
    if len(train) < 200 or train["label_up8_before_down3"].nunique() < 2:
        out["model_score"] = out["candidate_score"]
        return out, meta

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        GradientBoostingClassifier(random_state=42, n_estimators=120, max_depth=2, learning_rate=0.04),
    )
    model.fit(train[feature_cols], train["label_up8_before_down3"])
    score = model.predict_proba(out[feature_cols].replace([np.inf, -np.inf], np.nan))[:, 1]
    out["model_score"] = score
    meta.update({
        "trained": True,
        "positive_rate": float(train["label_up8_before_down3"].mean()),
        "features": feature_cols,
        "train_end": train_end,
    })
    return out, meta


def maybe_filter_candidates(df: pd.DataFrame, mode: str, max_per_day: int, threshold: int, train_end: str) -> tuple[pd.DataFrame, dict]:
    daily_count = df.groupby("date").size()
    meta = {
        "mode": mode,
        "max_per_day": max_per_day,
        "threshold": threshold,
        "avg_daily_candidates": float(daily_count.mean()) if not daily_count.empty else 0.0,
        "max_daily_candidates": int(daily_count.max()) if not daily_count.empty else 0,
        "before": int(len(df)),
    }
    if mode == "none" or (mode == "auto" and meta["avg_daily_candidates"] <= threshold and meta["max_daily_candidates"] <= threshold * 3):
        df = df.copy()
        df["selection_score"] = df["candidate_score"]
        meta["after"] = int(len(df))
        meta["filter_applied"] = False
        return df, meta

    scored, model_meta = add_model_score(df, train_end)
    score_col = "model_score" if mode in {"auto", "ml"} else "candidate_score"
    scored["selection_score"] = scored[score_col]
    rank = scored.groupby("date")["selection_score"].rank(method="first", ascending=False)
    filtered = scored[rank <= max_per_day].copy()
    meta.update(model_meta)
    meta["filter_applied"] = True
    meta["score_col"] = score_col
    meta["after"] = int(len(filtered))
    return filtered, meta


def max_drawdown(daily_returns_pct: pd.Series) -> float:
    equity = (1 + daily_returns_pct / 100).cumprod()
    dd = equity / equity.cummax() - 1
    return float(dd.min() * 100)


def summarize(trades: pd.DataFrame) -> dict:
    r = trades["return_pct"].dropna()
    if r.empty:
        return {}
    daily = trades.groupby("date")["return_pct"].mean().sort_index()
    losses = -r[r < 0].sum()
    return {
        "trades": int(len(r)),
        "days": int(len(daily)),
        "avg_return_pct": float(r.mean()),
        "median_return_pct": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "daily_avg_pct": float(daily.mean()),
        "daily_sharpe": float(np.sqrt(244) * daily.mean() / daily.std()) if daily.std() else np.nan,
        "max_drawdown_pct": max_drawdown(daily),
        "profit_factor": float(r[r > 0].sum() / losses) if losses > 0 else np.nan,
        "stop_rate": float((trades["exit_type"] == "stop_loss").mean()),
        "take_profit_rate": float((trades["exit_type"] == "take_profit").mean()),
        "technical_exit_rate": float((trades["exit_type"] == "technical_exit").mean()),
        "expiry_rate": float((trades["exit_type"] == "expiry").mean()),
    }


def evaluate(df: pd.DataFrame, exit_rules: Iterable[ExitRule]) -> pd.DataFrame:
    periods = {
        "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        "2025plus": (pd.Timestamp("2025-01-01"), pd.Timestamp.max),
        "all": (df["date"].min(), pd.Timestamp.max),
    }
    rows = []
    for rule in exit_rules:
        trades = simulate_exit(df, rule)
        for period, (start, end) in periods.items():
            metrics = summarize(trades[(trades["date"] >= start) & (trades["date"] <= end)])
            if metrics:
                rows.append({
                    "period": period,
                    "exit_rule": rule.name,
                    "exit_kind": rule.kind,
                    "hold_days": rule.hold_days,
                    "take_profit": rule.take_profit,
                    "stop_loss": rule.stop_loss,
                    "trail_drawdown": rule.trail_drawdown,
                    **metrics,
                })
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, meta: dict, output_dir: Path, timestamp: str) -> Path:
    path = output_dir / f"triple_volume_breakout_report_{timestamp}.md"
    key_cols = [
        "period",
        "exit_rule",
        "trades",
        "avg_return_pct",
        "win_rate",
        "daily_sharpe",
        "max_drawdown_pct",
        "profit_factor",
        "stop_rate",
        "take_profit_rate",
        "technical_exit_rate",
        "expiry_rate",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# 三倍量缩量盘整突破短线策略回测\n\n")
        f.write("口径：T+0 选股，T+1 开盘买入，T+2 起检查卖出；同日止损和止盈同时触发时按先止损处理。\n\n")
        f.write("## 候选与筛选\n\n")
        f.write("```json\n")
        f.write(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
        f.write("\n```\n\n")
        for period in ["2024", "2025plus", "all"]:
            part = summary[summary["period"] == period].copy()
            if part.empty:
                continue
            f.write(f"## {period} Top 15 by daily_sharpe\n\n")
            top = part.sort_values(["daily_sharpe", "avg_return_pct"], ascending=False).head(15)
            f.write(top[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
            f.write(f"## {period} Top 15 by avg_return_pct\n\n")
            top = part.sort_values(["avg_return_pct", "daily_sharpe"], ascending=False).head(15)
            f.write(top[key_cols].to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest 三倍量缩量盘整突破")
    parser.add_argument("--daily-dir", type=Path, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--filter-mode", choices=["auto", "none", "heuristic", "ml"], default="auto")
    parser.add_argument("--crowded-threshold", type=int, default=30, help="Trigger filter if average daily candidates exceed this")
    parser.add_argument("--max-per-day", type=int, default=10)
    parser.add_argument("--model-train-end", default="2024-12-31")
    parser.add_argument("--candidate-cache", type=Path, default=None)
    parser.add_argument(
        "--signal-mode",
        choices=[
            "strict",
            "pre_shrink_strict_bull",
            "soft_shrink_strict_bull",
            "avg_pre_shrink_strict_bull",
            "pre_shrink_bull_no60",
            "avg_pre_shrink_bull_no60",
            "pre_shrink_close_ma20",
            "avg_pre_shrink_close_ma20",
        ],
        default="strict",
    )
    parser.add_argument("--volume-multiple", type=float, default=3.0, help="Anchor volume multiple, e.g. 3.0 means V[-1] >= 3 * V[-2]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.candidate_cache and args.candidate_cache.exists():
        print(f"Loading candidate cache: {args.candidate_cache}")
        candidates = pd.read_parquet(args.candidate_cache)
        candidates["date"] = pd.to_datetime(candidates["date"])
    else:
        print(
            f"Building candidates from {args.daily_dir} with signal mode {args.signal_mode} "
            f"and volume multiple {args.volume_multiple}"
        )
        candidates = build_candidates(
            args.daily_dir,
            args.start_date,
            args.max_workers,
            args.signal_mode,
            args.volume_multiple,
        )
        if args.candidate_cache:
            args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
            candidates.to_parquet(args.candidate_cache, index=False)
            print(f"Candidate cache written: {args.candidate_cache}")

    print(f"Raw candidates: {len(candidates):,}")
    if candidates.empty:
        raise RuntimeError("No candidates found")

    max_hold_days = max(rule.hold_days for rule in build_exit_rules())
    candidates = add_future_prices(candidates, args.daily_dir, max_hold_days)
    print(f"Candidates with future prices: {len(candidates):,}")
    filtered, meta = maybe_filter_candidates(
        candidates,
        args.filter_mode,
        args.max_per_day,
        args.crowded_threshold,
        args.model_train_end,
    )
    meta["signal_mode"] = args.signal_mode
    meta["volume_multiple"] = args.volume_multiple
    print(f"Selected candidates: {len(filtered):,}")

    candidate_path = args.output_dir / f"triple_volume_breakout_candidates_{timestamp}.parquet"
    filtered.to_parquet(candidate_path, index=False)
    summary = evaluate(filtered, build_exit_rules())
    summary_path = args.output_dir / f"triple_volume_breakout_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    report_path = write_report(summary, meta, args.output_dir, timestamp)

    print(f"Candidates: {candidate_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
