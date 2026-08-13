from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.data.tushare_fetcher import TushareDataFetcher, validate_daily_basic_frame
from quant.routine.data_refresh import RequestLimiter, _is_retryable_error
from quant.routine.paths import PROJECT_ROOT


DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
AUDIT_ROOT = PROJECT_ROOT / "data/raw/source_audit"

# These are the raw daily_basic columns consumed by the production B1 and Chan
# feature builders. Row-count coverage alone is insufficient: Tushare can
# return a full stock cross-section while selected fields are entirely null.
# Dividend yield is naturally sparse, hence its lower (but still non-zero)
# threshold; the remaining market/share fields should cover virtually all
# listed stocks on the decision date.
DAILY_BASIC_FEATURE_COVERAGE: dict[str, float] = {
    "turnover_rate": 0.98,
    "turnover_rate_f": 0.98,
    "volume_ratio": 0.95,
    "pe": 0.70,
    "pe_ttm": 0.70,
    "pb": 0.95,
    "ps": 0.95,
    "ps_ttm": 0.95,
    "dv_ratio": 0.50,
    "dv_ttm": 0.50,
    "total_share": 0.98,
    "float_share": 0.98,
    "free_share": 0.98,
    "total_mv": 0.98,
    "circ_mv": 0.98,
}


def load_trade_dates_from_daily(daily_dir: Path, start_date: str, end_date: str | None) -> list[str]:
    return sorted(load_trade_date_symbol_counts(daily_dir, start_date, end_date))


def load_trade_date_symbol_counts(
    daily_dir: Path,
    start_date: str,
    end_date: str | None,
) -> dict[str, int]:
    counts: dict[str, set[str]] = {}
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    for path in partition_root.glob("year_month=*/data.parquet"):
        try:
            frame = pd.read_parquet(path, columns=["trade_date", "ts_code"])
        except Exception as exc:
            raise RuntimeError(f"failed to read canonical daily partition {path}: {exc}") from exc
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame[frame["trade_date"] >= start_date]
        if end_date:
            frame = frame[frame["trade_date"] <= end_date]
        for trade_date, group in frame.groupby("trade_date", sort=False):
            counts.setdefault(str(trade_date), set()).update(
                group["ts_code"].dropna().astype(str)
            )
    return {trade_date: len(symbols) for trade_date, symbols in counts.items()}


def fetch_one_trade_date(
    trade_date: str,
    output_dir: Path,
    cache_dir: Path,
    limiter: RequestLimiter,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    expected_rows: int | None = None,
    minimum_coverage_rate: float = 0.98,
    availability_retry_failures: int = 0,
    availability_retry_interval: float = 60.0,
) -> dict:
    attempts = max(1, retries + 1)
    maximum_attempts = attempts + max(0, availability_retry_failures)
    last_error = ""
    output_path = output_dir / f"{trade_date}.parquet"
    minimum_rows = max(
        1,
        math.ceil((expected_rows or 1) * minimum_coverage_rate),
    )
    if output_path.exists():
        try:
            local = validate_daily_basic_frame(
                pd.read_parquet(output_path),
                trade_date,
                minimum_rows=minimum_rows,
                required_feature_coverage=DAILY_BASIC_FEATURE_COVERAGE,
            )
        except Exception:
            # Keep the last file recoverable until a complete replacement has
            # been fetched and atomically written below.
            pass
        else:
            return {
                "trade_date": trade_date,
                "source": "local_validated",
                "status": "success",
                "rows": len(local),
                "expected_rows": expected_rows,
                "minimum_rows": minimum_rows,
                "coverage_rate": (
                    round(len(local) / expected_rows, 6) if expected_rows else None
                ),
                "feature_coverage": local.attrs.get("feature_coverage", {}),
                "path": str(output_path),
                "attempts": 0,
                "error": None,
            }
    for attempt in range(1, maximum_attempts + 1):
        try:
            limiter.wait()
            fetcher = TushareDataFetcher(cache_dir=cache_dir)
            try:
                df = validate_daily_basic_frame(
                    fetcher.get_daily_basic(trade_date),
                    trade_date,
                    minimum_rows=minimum_rows,
                    required_feature_coverage=DAILY_BASIC_FEATURE_COVERAGE,
                )
            except ValueError:
                # The fetcher's basic schema check intentionally accepts small
                # frames. Remove a cross-section cache that fails this
                # market-relative coverage gate so the retry reaches Tushare.
                (cache_dir / f"tushare_daily_basic_{trade_date}.parquet").unlink(
                    missing_ok=True
                )
                raise
            atomic_write_parquet(df, output_path, index=False)
            return {
                "trade_date": trade_date,
                "source": "tushare",
                "status": "success",
                "rows": len(df),
                "expected_rows": expected_rows,
                "minimum_rows": minimum_rows,
                "coverage_rate": (
                    round(len(df) / expected_rows, 6) if expected_rows else None
                ),
                "feature_coverage": df.attrs.get("feature_coverage", {}),
                "path": str(output_path),
                "attempts": attempt,
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            retryable = _is_retryable_error(last_error) or "daily_basic" in last_error
            availability_retry = (
                "feature coverage" in last_error
                and attempt <= availability_retry_failures
            )
            if attempt >= maximum_attempts or (not retryable and not availability_retry):
                return {
                    "trade_date": trade_date,
                    "source": "tushare",
                    "status": "failed",
                    "rows": 0,
                    "path": str(output_path),
                    "attempts": attempt,
                    "error": last_error,
                }
            if availability_retry:
                time.sleep(max(0.0, availability_retry_interval))
            else:
                time.sleep(min(retry_max_delay, retry_base_delay * (2 ** (attempt - 1))))
    return {
        "trade_date": trade_date,
        "source": "tushare",
        "status": "failed",
        "rows": 0,
        "path": str(output_path),
        "attempts": maximum_attempts,
        "error": last_error or "unknown error",
    }


def refresh_daily_basic(
    start_date: str,
    end_date: str | None = None,
    daily_dir: Path = DAILY_DIR,
    output_dir: Path = DAILY_BASIC_DIR,
    workers: int = 4,
    sleep_between: float = 0.25,
    retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 60.0,
) -> dict:
    expected_rows_by_date = load_trade_date_symbol_counts(
        daily_dir,
        start_date,
        end_date,
    )
    trade_dates = sorted(expected_rows_by_date)
    if not trade_dates:
        raise RuntimeError(f"No trade dates found in {daily_dir}")
    minimum_coverage_rate = float(
        os.getenv("ROUTINE_DAILY_BASIC_MIN_COVERAGE_RATE", "0.98")
    )
    availability_retry_failures = int(
        os.getenv("ROUTINE_DAILY_BASIC_AVAILABILITY_RETRY_FAILURES", "2")
    )
    availability_retry_interval = float(
        os.getenv("ROUTINE_DAILY_BASIC_AVAILABILITY_RETRY_INTERVAL", "60")
    )
    if not 0 < minimum_coverage_rate <= 1:
        raise ValueError(
            "ROUTINE_DAILY_BASIC_MIN_COVERAGE_RATE must be in (0, 1]"
        )

    cache_dir = PROJECT_ROOT / "data/cache/source_merge/tushare"
    limiter = RequestLimiter(sleep_between)
    audits: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                fetch_one_trade_date,
                trade_date,
                output_dir,
                cache_dir,
                limiter,
                retries,
                retry_base_delay,
                retry_max_delay,
                expected_rows_by_date.get(trade_date),
                minimum_coverage_rate,
                availability_retry_failures if trade_date == max(trade_dates) else 0,
                availability_retry_interval,
            )
            for trade_date in trade_dates
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            audits.append(future.result())
            if n % 100 == 0 or n == len(futures):
                ok = sum(1 for audit in audits if audit["status"] == "success")
                failed = sum(1 for audit in audits if audit["status"] == "failed")
                print(f"daily_basic progress: {n}/{len(futures)} done, success={ok}, failed={failed}", flush=True)

    audit_dir = AUDIT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_daily_basic")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "daily_basic_audit.csv"
    audit_df = pd.DataFrame(audits)
    atomic_write_csv(audit_df, audit_path, index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": "tushare_only_daily_basic",
        "start_date": start_date,
        "end_date": end_date,
        "daily_dir": str(daily_dir),
        "output_dir": str(output_dir),
        "audit_path": str(audit_path),
        "trade_dates": len(trade_dates),
        "latest_trade_date": max(trade_dates),
        "minimum_coverage_rate": minimum_coverage_rate,
        "required_feature_coverage": DAILY_BASIC_FEATURE_COVERAGE,
        "availability_retry_failures": availability_retry_failures,
        "availability_retry_interval_seconds": availability_retry_interval,
        "success": int((audit_df["status"] == "success").sum()),
        "failed": int((audit_df["status"] == "failed").sum()),
        "workers": workers,
        "min_request_interval_seconds": sleep_between,
        "retries": retries,
    }
    manifest_path = audit_dir / "manifest.json"
    atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Tushare daily_basic data by trade date.")
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default=None)
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DAILY_BASIC_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = refresh_daily_basic(
        start_date=args.start,
        end_date=args.end,
        daily_dir=args.daily_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        sleep_between=args.sleep,
        retries=args.retries,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
