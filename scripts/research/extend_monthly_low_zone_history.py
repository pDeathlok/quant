#!/usr/bin/env python3
"""Fetch resumable 2000-2015 A-share daily history for old-cycle validation."""

from __future__ import annotations

import argparse
import calendar as month_calendar
import time
from pathlib import Path

import pandas as pd

from quant.data.tushare_fetcher import TushareDataFetcher


DAILY_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2015)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/research/monthly_low_zone_strict_extension"),
    )
    parser.add_argument("--minimum-request-interval", type=float, default=0.20)
    return parser.parse_args()


def _call_with_retry(call, *, attempts: int = 8) -> pd.DataFrame:
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            frame = call()
            return frame if frame is not None else pd.DataFrame()
        except Exception as exc:  # network/API failures need resumable retries
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
    assert last_error is not None
    raise last_error


def _fetch_uncapped_daily_range(
    fetcher: TushareDataFetcher,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    minimum_request_interval: float,
) -> pd.DataFrame:
    """Recursively split any Tushare result that reaches its 6,000-row cap."""

    frame = _call_with_retry(
        lambda: fetcher.pro.daily(
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields=DAILY_FIELDS,
        )
    )
    time.sleep(max(minimum_request_interval, 0.0))
    if len(frame) < 6000:
        return frame
    if start >= end:
        raise RuntimeError(
            f"single-date daily result hit 6000-row cap: {start.date()}"
        )
    midpoint = start + pd.Timedelta(days=(end - start).days // 2)
    left = _fetch_uncapped_daily_range(
        fetcher,
        start,
        midpoint,
        minimum_request_interval=minimum_request_interval,
    )
    right = _fetch_uncapped_daily_range(
        fetcher,
        midpoint + pd.Timedelta(days=1),
        end,
        minimum_request_interval=minimum_request_interval,
    )
    return pd.concat([left, right], ignore_index=True, sort=False)


def _fetch_year(
    fetcher: TushareDataFetcher,
    year: int,
    *,
    minimum_request_interval: float,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in range(1, 13):
        last_day = month_calendar.monthrange(year, month)[1]
        start = pd.Timestamp(year=year, month=month, day=1)
        end = pd.Timestamp(year=year, month=month, day=last_day)
        part = _fetch_uncapped_daily_range(
            fetcher,
            start,
            end,
            minimum_request_interval=minimum_request_interval,
        )
        parts.append(part)
        print(
            f"{year}-{month:02d}: rows={len(part):,}",
            flush=True,
        )
    frame = pd.concat(parts, ignore_index=True, sort=False)
    frame = (
        frame.sort_values(["trade_date", "ts_code"])
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    return frame


def _validate_year(frame: pd.DataFrame, year: int) -> None:
    required = {"ts_code", "trade_date", "open", "high", "low", "close", "amount"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"year {year} missing columns: {missing}")
    dates = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce")
    if frame.empty or dates.isna().any():
        raise ValueError(f"year {year} is empty or has invalid dates")
    if not dates.dt.year.eq(year).all():
        raise ValueError(f"year {year} contains out-of-year rows")
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError(f"year {year} contains duplicate symbol dates")


def main() -> None:
    args = _parse_args()
    if args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fetcher = TushareDataFetcher(
        cache_dir=args.output_dir / "tushare_request_cache"
    )
    for year in range(args.start_year, args.end_year + 1):
        path = args.output_dir / f"market_daily_{year}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            _validate_year(existing, year)
            print(f"{year}: cache hit rows={len(existing):,}", flush=True)
            continue
        print(f"fetching full market daily history: {year}", flush=True)
        frame = _fetch_year(
            fetcher,
            year,
            minimum_request_interval=args.minimum_request_interval,
        )
        _validate_year(frame, year)
        frame.to_parquet(path, index=False, compression="zstd")
        print(
            f"{year}: saved rows={len(frame):,} symbols={frame['ts_code'].nunique():,}",
            flush=True,
        )
    index_path = args.output_dir / "index_000001.SH_20000101_20141231.parquet"
    if not index_path.exists():
        index = _call_with_retry(
            lambda: fetcher.pro.index_daily(
                ts_code="000001.SH",
                start_date=f"{args.start_year}0101",
                end_date=f"{args.end_year}1231",
            )
        )
        if index.empty:
            raise ValueError("index history is empty")
        index = index.sort_values("trade_date").drop_duplicates(
            "trade_date", keep="last"
        )
        index.to_parquet(index_path, index=False, compression="zstd")
        print(f"index: saved rows={len(index):,}", flush=True)
    else:
        print(f"index: cache hit {index_path}", flush=True)


if __name__ == "__main__":
    main()
