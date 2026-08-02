"""Backfill point-in-time A-share tradability partitions from Tushare."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data.atomic_io import atomic_write_json
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.daily_basic_refresh import load_trade_dates_from_daily
from quant.routine.paths import PROJECT_ROOT
from quant.routine.reference_data_refresh import refresh_daily_tradability


DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
OUTPUT_DIR = PROJECT_ROOT / "data/raw/tradability"
AUDIT_ROOT = PROJECT_ROOT / "data/raw/source_audit"


def fetch_historical_stock_universe(fetcher: TushareDataFetcher) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    fields = "ts_code,name,market,list_status,list_date,delist_date"
    for status in ("L", "D", "P"):
        try:
            frame = fetcher.pro.stock_basic(
                exchange="",
                list_status=status,
                fields=fields,
            )
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append(f"{status}: {exc}")
    if not frames:
        raise RuntimeError("Tushare stock_basic returned no historical universe: " + "; ".join(errors))
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates("ts_code", keep="last")
        .sort_values("ts_code")
        .reset_index(drop=True)
    )


def refresh_tradability_range(
    *,
    start_date: str,
    end_date: str | None = None,
    daily_dir: Path = DAILY_DIR,
    output_dir: Path = OUTPUT_DIR,
    audit_root: Path = AUDIT_ROOT,
    fetcher: TushareDataFetcher | None = None,
    force: bool = False,
    sleep_between: float = 0.5,
    minimum_coverage_rate: float | None = None,
) -> dict[str, Any]:
    start = str(start_date).replace("-", "")
    end = str(end_date).replace("-", "") if end_date else None
    trade_dates = load_trade_dates_from_daily(daily_dir, start, end)
    if not trade_dates:
        raise RuntimeError(f"No canonical daily trade dates found from {start} to {end or 'latest'}")
    client = fetcher or TushareDataFetcher(
        cache_dir=PROJECT_ROOT / "data/cache/source_merge/tushare"
    )
    universe = fetch_historical_stock_universe(client)
    audits: list[dict[str, Any]] = []
    for index, trade_date in enumerate(trade_dates):
        output_path = output_dir / f"{trade_date}.parquet"
        if output_path.is_file() and not force:
            frame = pd.read_parquet(output_path, columns=["trade_date", "ts_code"])
            audits.append(
                {
                    "trade_date": trade_date,
                    "status": "skipped",
                    "reason": "partition_exists",
                    "rows": len(frame),
                    "path": str(output_path),
                }
            )
            continue
        try:
            result = refresh_daily_tradability(
                client,
                output_dir.parent,
                trade_date,
                stock_basic=universe,
                minimum_coverage_rate=minimum_coverage_rate,
            )
            audits.append(result)
        except Exception as exc:
            audits.append(
                {
                    "trade_date": trade_date,
                    "status": "failed",
                    "error": str(exc),
                    "path": str(output_path),
                }
            )
        if sleep_between > 0 and index + 1 < len(trade_dates):
            time.sleep(sleep_between)

    successful = sum(item["status"] == "success" for item in audits)
    skipped = sum(item["status"] == "skipped" for item in audits)
    failed = sum(item["status"] == "failed" for item in audits)
    manifest = {
        "status": "success" if failed == 0 else "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": "tushare_first_point_in_time_tradability",
        "start_date": min(trade_dates),
        "end_date": max(trade_dates),
        "trade_dates": len(trade_dates),
        "success": successful,
        "skipped": skipped,
        "failed": failed,
        "force": force,
        "audits": audits,
    }
    audit_dir = audit_root / datetime.now().strftime("%Y%m%d_%H%M%S_tradability")
    manifest_path = atomic_write_json(manifest, audit_dir / "manifest.json")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Tushare A-share tradability data.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sleep",
        type=float,
        default=float(os.getenv("ROUTINE_TRADABILITY_BACKFILL_SLEEP", "0.5")),
    )
    args = parser.parse_args()
    result = refresh_tradability_range(
        start_date=args.start,
        end_date=args.end,
        daily_dir=args.daily_dir,
        output_dir=args.output_dir,
        force=args.force,
        sleep_between=args.sleep,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
