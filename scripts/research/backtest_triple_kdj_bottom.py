#!/usr/bin/env python3
"""Causal A-share event study for daily/weekly/monthly KDJ-J below zero.

The main signal uses the current completed daily bar and only the previously
completed weekly/monthly bars.  Prices are made continuous forward in time
from raw OHLC and pre_close, so later corporate actions cannot rewrite history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (5, 20, 60)
RULES = {
    "daily_j_lt_0": ("j_d",),
    "daily_weekly_j_lt_0": ("j_d", "j_w"),
    "triple_j_lt_0": ("j_d", "j_w", "j_m"),
}


def kdj_j(frame: pd.DataFrame) -> pd.Series:
    low9 = frame["low"].rolling(9, min_periods=9).min()
    high9 = frame["high"].rolling(9, min_periods=9).max()
    rsv = ((frame["close"] - low9) / (high9 - low9).replace(0, np.nan) * 100.0)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return 3.0 * k - 2.0 * d


def continuous_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.sort_values("date").copy()
    previous = out["close"].shift(1)
    ratio = (previous / out["pre_close"]).where(out["pre_close"].gt(0))
    ratio = ratio.where(ratio.gt(0) & np.isfinite(ratio), 1.0).fillna(1.0)
    factor = ratio.cumprod()
    for column in ("open", "high", "low", "close"):
        out[column] = out[column].astype(float) * factor
    return out


def completed_period_j(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    bars = (
        daily.set_index("date")
        .resample(rule)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
    )
    bars["j"] = kdj_j(bars)
    return bars[["j"]].reset_index()


def align_completed(daily: pd.DataFrame, period: pd.DataFrame, name: str) -> pd.Series:
    # Period labels (Friday/calendar month-end) are their first safe availability
    # dates. merge_asof therefore never exposes a holiday-shortened bar early.
    aligned = pd.merge_asof(
        daily[["date"]].sort_values("date"),
        period.rename(columns={"j": name}).sort_values("date"),
        on="date",
        direction="backward",
    )
    return aligned[name]


def cooldown_onsets(mask: pd.Series, cooldown: int = 20) -> np.ndarray:
    values = mask.fillna(False).to_numpy(bool)
    chosen = np.zeros(len(values), dtype=bool)
    last = -10**9
    for i, value in enumerate(values):
        if value and (i == 0 or not values[i - 1]) and i - last > cooldown:
            chosen[i] = True
            last = i
    return chosen


def symbol_events(raw: pd.DataFrame) -> list[dict[str, object]]:
    daily = continuous_ohlc(raw)
    if len(daily) < 320:
        return []
    daily = daily.reset_index(drop=True)
    daily["j_d"] = kdj_j(daily)
    daily["j_w"] = align_completed(daily, completed_period_j(daily, "W-FRI"), "j_w")
    daily["j_m"] = align_completed(daily, completed_period_j(daily, "ME"), "j_m")
    # Require roughly one trading year before any event in addition to KDJ warm-up.
    eligible = pd.Series(np.arange(len(daily)) >= 250, index=daily.index)
    output: list[dict[str, object]] = []
    for rule_name, columns in RULES.items():
        mask = eligible.copy()
        for column in columns:
            mask &= daily[column].lt(0)
        for i in np.flatnonzero(cooldown_onsets(mask)):
            if i + max(HORIZONS) >= len(daily) or i + 1 >= len(daily):
                continue
            entry = float(daily.at[i + 1, "open"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            record: dict[str, object] = {
                "rule": rule_name,
                "symbol": str(daily.at[i, "ts_code"]),
                "signal_date": daily.at[i, "date"],
                "entry_date": daily.at[i + 1, "date"],
                "signal_close": float(daily.at[i, "close"]),
                "entry_open": entry,
                "j_d": float(daily.at[i, "j_d"]),
                "j_w": float(daily.at[i, "j_w"]) if pd.notna(daily.at[i, "j_w"]) else np.nan,
                "j_m": float(daily.at[i, "j_m"]) if pd.notna(daily.at[i, "j_m"]) else np.nan,
            }
            backward = daily.loc[max(0, i - 20) : i, "low"]
            centered = daily.loc[max(0, i - 20) : i + 20, "low"]
            future20 = daily.loc[i : i + 20, "low"]
            center_min = float(centered.min())
            future_min = float(future20.min())
            future_argmin = int(np.argmin(future20.to_numpy()))
            record["within_5pct_centered_41d_low"] = float(daily.at[i, "close"]) <= center_min * 1.05
            record["within_5pct_future_20d_low"] = float(daily.at[i, "close"]) <= future_min * 1.05
            record["bottom_in_next_5d"] = future_argmin <= 5
            record["drawdown_from_prior20_low"] = float(daily.at[i, "close"]) / float(backward.min()) - 1.0
            for horizon in HORIZONS:
                path = daily.loc[i + 1 : i + horizon]
                exit_close = float(daily.at[i + horizon, "close"])
                record[f"ret_{horizon}"] = exit_close / entry - 1.0
                record[f"mae_{horizon}"] = float(path["low"].min()) / entry - 1.0
                record[f"mfe_{horizon}"] = float(path["high"].max()) / entry - 1.0
                record[f"exit_date_{horizon}"] = daily.at[i + horizon, "date"]
            output.append(record)
    return output


def date_cluster_ci(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    clean = frame[["signal_date", column]].dropna()
    groups = clean["signal_date"].nunique()
    observations = len(clean)
    if groups < 2 or observations < 2:
        return np.nan, np.nan
    mean = clean[column].mean()
    residual = clean[column] - mean
    cluster_sums = residual.groupby(clean["signal_date"]).sum()
    # One-way cluster-robust standard error for an intercept-only regression.
    correction = groups / (groups - 1) * (observations - 1) / observations
    se = np.sqrt(correction * np.square(cluster_sums).sum() / observations**2)
    return float(mean - 1.96 * se), float(mean + 1.96 * se)


def add_benchmark_returns(events: pd.DataFrame, path: Path) -> pd.DataFrame:
    index = pd.read_parquet(path, columns=["trade_date", "open", "close"])
    index["date"] = pd.to_datetime(index["trade_date"].astype(str))
    opens = index.set_index("date")["open"]
    closes = index.set_index("date")["close"]
    out = events.copy()
    entry_index_open = out["entry_date"].map(opens)
    for horizon in HORIZONS:
        exit_index_close = out[f"exit_date_{horizon}"].map(closes)
        out[f"benchmark_ret_{horizon}"] = exit_index_close / entry_index_open - 1.0
        out[f"excess_ret_{horizon}"] = out[f"ret_{horizon}"] - out[f"benchmark_ret_{horizon}"]
    return out


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {
        "all_2011_2026": ("2011-01-01", "2026-12-31"),
        "development_2011_2019": ("2011-01-01", "2019-12-31"),
        "validation_2020_2022": ("2020-01-01", "2022-12-31"),
        "holdout_2023_2026": ("2023-01-01", "2026-12-31"),
    }
    for period, (start, end) in periods.items():
        scoped = events[events["signal_date"].between(start, end)]
        for rule_name in RULES:
            subset = scoped[scoped["rule"].eq(rule_name)]
            if subset.empty:
                continue
            base = {
                "period": period,
                "rule": rule_name,
                "events": len(subset),
                "symbols": subset["symbol"].nunique(),
                "signal_dates": subset["signal_date"].nunique(),
                "centered_low_5pct_rate": subset["within_5pct_centered_41d_low"].mean(),
                "future_low_5pct_rate": subset["within_5pct_future_20d_low"].mean(),
                "bottom_next_5d_rate": subset["bottom_in_next_5d"].mean(),
            }
            for horizon in HORIZONS:
                col = f"ret_{horizon}"
                lo, hi = date_cluster_ci(subset, col)
                base.update({
                    f"mean_ret_{horizon}": subset[col].mean(),
                    f"median_ret_{horizon}": subset[col].median(),
                    f"win_rate_{horizon}": subset[col].gt(0).mean(),
                    f"mean_mae_{horizon}": subset[f"mae_{horizon}"].mean(),
                    f"mean_mfe_{horizon}": subset[f"mfe_{horizon}"].mean(),
                    f"date_cluster_ci95_low_{horizon}": lo,
                    f"date_cluster_ci95_high_{horizon}": hi,
                })
                excess = f"excess_ret_{horizon}"
                if excess in subset:
                    ex_lo, ex_hi = date_cluster_ci(subset, excess)
                    base.update({
                        f"mean_excess_ret_{horizon}": subset[excess].mean(),
                        f"excess_win_rate_{horizon}": subset[excess].gt(0).mean(),
                        f"excess_cluster_ci95_low_{horizon}": ex_lo,
                        f"excess_cluster_ci95_high_{horizon}": ex_hi,
                    })
            rows.append(base)
    return pd.DataFrame(rows)


def load_daily(root: Path) -> pd.DataFrame:
    columns = ["ts_code", "date", "open", "high", "low", "close", "pre_close"]
    frames = [pd.read_parquet(path, columns=columns) for path in sorted(root.glob("year_month=*/data.parquet"))]
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    valid_symbol = daily["ts_code"].astype(str).str.match(r"^\d{6}\.(SH|SZ|BJ)$")
    valid_price = daily[["open", "high", "low", "close"]].gt(0).all(axis=1)
    return daily[valid_symbol & valid_price].drop_duplicates(["ts_code", "date"], keep="last")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-root", type=Path, default=Path("data/raw/daily_partitioned"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/research/triple_kdj_bottom"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/raw/index_000300.SH.parquet"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily = load_daily(args.daily_root)
    results: list[dict[str, object]] = []
    symbols = daily["ts_code"].nunique()
    for number, (_, group) in enumerate(daily.groupby("ts_code", sort=True), 1):
        results.extend(symbol_events(group))
        if number % 500 == 0:
            print(f"processed {number}/{symbols} symbols; events={len(results)}", flush=True)
    events = pd.DataFrame(results).sort_values(["signal_date", "symbol", "rule"])
    events = add_benchmark_returns(events, args.benchmark)
    summary = summarize(events)
    events.to_parquet(args.output_dir / "events.parquet", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "source": str(args.daily_root),
        "rows": int(len(daily)),
        "symbols": int(symbols),
        "first_date": str(daily["date"].min().date()),
        "last_date": str(daily["date"].max().date()),
        "price_method": "causal forward-continuous OHLC from raw OHLC/pre_close",
        "signal_availability": "daily close plus previously completed W-FRI and calendar-month bars",
        "kdj": "9-bar RSV; K,D alpha=1/3 adjust=False; J=3K-2D",
        "event_sampling": "false-to-true onset, 20 stock-trading-bar cooldown",
        "execution": "next available stock open; gross returns, no fees/slippage",
        "benchmark": str(args.benchmark),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
