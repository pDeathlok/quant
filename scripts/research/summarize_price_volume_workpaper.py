#!/usr/bin/env python3
"""Summarize one stock/index OHLCV workpaper pair for A-share research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["bars"])
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return payload, frame.sort_values("date").reset_index(drop=True)


def _return(frame: pd.DataFrame, days: int) -> float | None:
    if len(frame) <= days:
        return None
    return float(frame["close"].iloc[-1] / frame["close"].iloc[-days - 1] - 1.0)


def summarize(stock_path: Path, benchmark_path: Path) -> dict:
    stock_payload, stock = _load(stock_path)
    benchmark_payload, benchmark = _load(benchmark_path)
    merged = stock[["date", "close"]].merge(
        benchmark[["date", "close"]], on="date", suffixes=("_stock", "_benchmark")
    )

    close = stock["close"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            stock["high"] - stock["low"],
            (stock["high"] - previous_close).abs(),
            (stock["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()

    returns: dict[str, dict[str, float | None]] = {}
    for days in (5, 20, 60, 120, 250):
        stock_return = _return(stock, days)
        benchmark_return = _return(merged.rename(columns={"close_benchmark": "close"}), days)
        returns[str(days)] = {
            "stock": stock_return,
            "benchmark": benchmark_return,
            "relative": (
                stock_return - benchmark_return
                if stock_return is not None and benchmark_return is not None
                else None
            ),
        }

    moving_averages = {
        str(days): float(close.tail(days).mean()) if len(close) >= days else None
        for days in (20, 60, 120, 250)
    }
    windows: dict[str, dict[str, float | None]] = {}
    for days in (60, 250):
        values = close.tail(days)
        low = float(values.min())
        high = float(values.max())
        last = float(close.iloc[-1])
        windows[str(days)] = {
            "low": low,
            "high": high,
            "from_high": last / high - 1.0,
            "position": (last - low) / (high - low) if high > low else None,
        }

    daily_returns = close.pct_change()
    result = {
        "ticker": stock_payload["ticker"],
        "analysis_cutoff": stock_payload["analysis_cutoff"],
        "last_bar_available_at": stock_payload["last_bar_available_at"],
        "benchmark": benchmark_payload["ticker"],
        "bars": int(len(stock)),
        "last_close": float(close.iloc[-1]),
        "returns": returns,
        "moving_averages": moving_averages,
        "ma_gaps": {
            key: (float(close.iloc[-1]) / value - 1.0 if value else None)
            for key, value in moving_averages.items()
        },
        "windows": windows,
        "volume": {
            "last_to_20d_average": float(stock["volume"].iloc[-1] / stock["volume"].tail(20).mean()),
            "last_5d_to_previous_20d": float(
                stock["volume"].tail(5).mean() / stock["volume"].iloc[-25:-5].mean()
            ),
        },
        "atr14_to_close": float(true_range.tail(14).mean() / close.iloc[-1]),
        "annualized_volatility_20d": float(daily_returns.tail(20).std(ddof=1) * np.sqrt(252)),
        "rsi14": float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None,
        "macd": {
            "dif": float(dif.iloc[-1]),
            "dea": float(dea.iloc[-1]),
            "histogram": float(2 * (dif.iloc[-1] - dea.iloc[-1])),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.stock, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
