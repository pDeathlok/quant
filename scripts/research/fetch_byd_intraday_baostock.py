#!/usr/bin/env python
"""Fetch BYD 5-minute bars from BaoStock for local strategy research.

BaoStock is intentionally kept as a research-only, optional dependency. The
script accepts a package directory so it can be installed under ``/tmp``
without changing the project's runtime environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-07-17")
    parser.add_argument("--frequency", default="5", choices=["5", "15", "30", "60"])
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/cache/baostock_002594_5min_20240101_20260717_qfq.parquet",
    )
    return parser.parse_args()


def load_baostock(package_dir: Path | None):
    if package_dir is not None:
        sys.path.append(str(package_dir.resolve()))
    try:
        import baostock as bs
    except ImportError as exc:  # pragma: no cover - operational guidance
        raise SystemExit(
            "BaoStock is not installed. Install it into a temporary directory and pass "
            "--package-dir, for example: python -m pip install --target /tmp/quant-baostock "
            "baostock==0.9.3"
        ) from exc
    return bs


def fetch(start: str, end: str, frequency: str, package_dir: Path | None) -> pd.DataFrame:
    bs = load_baostock(package_dir)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_code} {login.error_msg}")
    try:
        fields = "date,time,code,open,high,low,close,volume,amount,adjustflag"
        result = bs.query_history_k_data_plus(
            "sz.002594",
            fields,
            start_date=start,
            end_date=end,
            frequency=frequency,
            adjustflag="2",
        )
        if result.error_code != "0":
            raise RuntimeError(
                f"BaoStock query failed: {result.error_code} {result.error_msg}"
            )
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
    finally:
        bs.logout()

    frame = pd.DataFrame(rows, columns=fields.split(","))
    if frame.empty:
        raise RuntimeError("BaoStock returned no BYD minute bars")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["datetime"] = pd.to_datetime(
        frame["time"].str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce"
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return (
        frame.dropna(subset=["datetime", "open", "high", "low", "close"])
        .sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    frame = fetch(args.start, args.end, args.frequency, args.package_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"saved {len(frame)} bars from {frame['datetime'].min()} to "
        f"{frame['datetime'].max()} -> {args.output}"
    )


if __name__ == "__main__":
    main()
