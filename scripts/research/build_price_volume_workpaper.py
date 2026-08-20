#!/usr/bin/env python3
"""Build reproducible stock and CSI 300 OHLCV workpapers from local project data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _iso_bar_available(trade_date: pd.Timestamp) -> str:
    return f"{trade_date:%Y-%m-%d}T16:00:00+08:00"


def _bars(frame: pd.DataFrame) -> list[dict[str, object]]:
    frame = frame.sort_values("trade_date")
    return [
        {
            "date": pd.Timestamp(row.trade_date).strftime("%Y-%m-%d"),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in frame.itertuples(index=False)
    ]


def _load_stock(root: Path, ticker: str, cutoff_date: str, count: int) -> pd.DataFrame:
    cutoff_month = cutoff_date.replace("-", "")[:6]
    paths = sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
    paths = [p for p in paths if p.parent.name.split("=", 1)[1] <= cutoff_month]
    parts: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
            filters=[("ts_code", "==", ticker)],
        )
        if not frame.empty:
            parts.append(frame)
    if not parts:
        raise ValueError(f"No daily data found for {ticker}")
    frame = pd.concat(parts, ignore_index=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
    frame = frame.loc[frame["trade_date"] <= pd.Timestamp(cutoff_date)].copy()
    frame = frame.drop_duplicates("trade_date", keep="last").sort_values("trade_date").tail(count)
    return frame.rename(columns={"vol": "volume"})


def _load_benchmark(root: Path, cutoff_date: str, count: int) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data/raw/index_000300.SH.parquet")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
    frame = frame.loc[frame["trade_date"] <= pd.Timestamp(cutoff_date)].copy()
    frame = frame.drop_duplicates("trade_date", keep="last").sort_values("trade_date").tail(count)
    return frame.rename(columns={"vol": "volume"})


def _payload(
    frame: pd.DataFrame,
    *,
    ticker: str,
    analysis_cutoff: str,
    source: str,
    interface: str,
    count: int,
) -> dict[str, object]:
    last = pd.Timestamp(frame["trade_date"].max())
    accessed_at = datetime.now().astimezone().replace(microsecond=0).isoformat()
    return {
        "ticker": ticker,
        "as_of_date": analysis_cutoff[:10],
        "analysis_cutoff": analysis_cutoff,
        "last_bar_available_at": _iso_bar_available(last),
        "source": source,
        "price_basis": "unadjusted",
        "volume_unit": "lots (手)",
        "bars": _bars(frame),
        "provenance": {
            "provider": "Tushare",
            "interface": interface,
            "configured_storage_backend": "project local parquet mirror",
            "mirror_parquet": True,
            "accessed_at": accessed_at,
            "publication_rule": "完整日线按交易日 16:00:00+08:00 保守视为可得",
            "requested_bar_count": count,
        },
        "data_warnings": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--analysis-cutoff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--bars", default=250, type=int)
    args = parser.parse_args()

    cutoff_date = args.analysis_cutoff[:10]
    stock = _load_stock(args.root, args.ticker, cutoff_date, args.bars)
    benchmark = _load_benchmark(args.root, cutoff_date, args.bars)
    if len(stock) < args.bars or len(benchmark) < args.bars:
        raise ValueError(f"Insufficient bars: stock={len(stock)}, benchmark={len(benchmark)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        args.output_dir / "price_volume_stock.json": _payload(
            stock,
            ticker=args.ticker,
            analysis_cutoff=args.analysis_cutoff,
            source="Tushare daily via project canonical parquet mirror",
            interface="daily",
            count=args.bars,
        ),
        args.output_dir / "price_volume_csi300.json": _payload(
            benchmark,
            ticker="000300.SH",
            analysis_cutoff=args.analysis_cutoff,
            source="Tushare index_daily via project local reference file index_000300.SH.parquet",
            interface="index_daily",
            count=args.bars,
        ),
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
