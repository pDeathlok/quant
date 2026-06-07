from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.data_refresh import RequestLimiter, _is_retryable_error
from quant.routine.paths import PROJECT_ROOT


DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
AUDIT_ROOT = PROJECT_ROOT / "data/raw/source_audit"


def load_trade_dates_from_daily(daily_dir: Path, start_date: str, end_date: str | None) -> list[str]:
    dates: set[str] = set()
    for path in daily_dir.glob("*.parquet"):
        stem = path.stem
        if not (len(stem) == 9 and stem[6] == "." and stem[-2:] in {"SZ", "SH", "BJ"}):
            continue
        try:
            df = pd.read_parquet(path, columns=["trade_date"])
        except Exception:
            continue
        values = df["trade_date"].dropna().astype(str)
        if end_date:
            values = values[(values >= start_date) & (values <= end_date)]
        else:
            values = values[values >= start_date]
        dates.update(values.tolist())
    return sorted(dates)


def fetch_one_trade_date(
    trade_date: str,
    output_dir: Path,
    cache_dir: Path,
    limiter: RequestLimiter,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> dict:
    attempts = max(1, retries + 1)
    last_error = ""
    output_path = output_dir / f"{trade_date}.parquet"
    for attempt in range(1, attempts + 1):
        try:
            limiter.wait()
            fetcher = TushareDataFetcher(cache_dir=cache_dir)
            df = fetcher.get_daily_basic(trade_date)
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            return {
                "trade_date": trade_date,
                "source": "tushare",
                "status": "success",
                "rows": len(df),
                "path": str(output_path),
                "attempts": attempt,
                "error": None,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt >= attempts or not _is_retryable_error(last_error):
                return {
                    "trade_date": trade_date,
                    "source": "tushare",
                    "status": "failed",
                    "rows": 0,
                    "path": str(output_path),
                    "attempts": attempt,
                    "error": last_error,
                }
            time.sleep(min(retry_max_delay, retry_base_delay * (2 ** (attempt - 1))))
    return {
        "trade_date": trade_date,
        "source": "tushare",
        "status": "failed",
        "rows": 0,
        "path": str(output_path),
        "attempts": attempts,
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
    trade_dates = load_trade_dates_from_daily(daily_dir, start_date, end_date)
    if not trade_dates:
        raise RuntimeError(f"No trade dates found in {daily_dir}")

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
    audit_df.to_csv(audit_path, index=False)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": "tushare_only_daily_basic",
        "start_date": start_date,
        "end_date": end_date,
        "daily_dir": str(daily_dir),
        "output_dir": str(output_dir),
        "audit_path": str(audit_path),
        "trade_dates": len(trade_dates),
        "success": int((audit_df["status"] == "success").sum()),
        "failed": int((audit_df["status"] == "failed").sum()),
        "workers": workers,
        "min_request_interval_seconds": sleep_between,
        "retries": retries,
    }
    manifest_path = audit_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
