from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant.data.market_data_store import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import (
    DailyRefreshAudit,
    audits_to_frame,
    build_tushare_daily_audit,
    normalize_tushare_daily,
    normalize_ts_code,
)
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.paths import DAILY_DIR, PROJECT_ROOT


AUDIT_ROOT = PROJECT_ROOT / "data/raw/source_audit"


RETRYABLE_ERROR_KEYWORDS = (
    "每分钟最多访问",
    "频次",
    "frequency",
    "rate limit",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "remote end closed",
    "reset",
    "503",
    "502",
    "504",
)

NON_RETRYABLE_ERROR_KEYWORDS = (
    "未获取到",
    "missing trade_date/date",
    "missing date",
)


class RequestLimiter:
    """Process-wide request pacing with buffer for provider rate limits."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.min_interval
        if wait_for > 0:
            time.sleep(wait_for)


def load_tushare_symbols(fetcher: TushareDataFetcher, board: str = "all", limit: int | None = None) -> list[tuple[str, str]]:
    basic = fetcher.get_stock_basic()
    if board == "main":
        basic = basic[basic["ts_code"].astype(str).str.startswith(("000", "001", "002", "600", "601", "603", "605"))]
    elif board == "gem":
        basic = basic[basic["ts_code"].astype(str).str.startswith(("300", "301"))]
    rows = [(str(row.ts_code), str(row.name)) for row in basic.itertuples()]
    rows = sorted(rows)
    return rows[:limit] if limit else rows


def load_symbols_file(path: Path, fetcher: TushareDataFetcher, limit: int | None = None) -> list[tuple[str, str]]:
    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_csv(path, header=None, names=["symbol"])
    if "symbol" not in raw.columns:
        raise ValueError(f"{path} must contain a symbol column")
    symbols = raw["symbol"].dropna().astype(str).drop_duplicates().tolist()
    basic = fetcher.get_stock_basic()
    name_by_symbol = basic.set_index("ts_code")["name"].astype(str).to_dict() if "ts_code" in basic.columns else {}
    rows = [(symbol, name_by_symbol.get(symbol, symbol)) for symbol in symbols]
    return rows[:limit] if limit else rows


def _is_retryable_error(error: str) -> bool:
    message = error.lower()
    if any(keyword.lower() in message for keyword in NON_RETRYABLE_ERROR_KEYWORDS):
        return False
    return any(keyword.lower() in message for keyword in RETRYABLE_ERROR_KEYWORDS)


def _with_attempt(audit: DailyRefreshAudit, attempts: int) -> DailyRefreshAudit:
    return DailyRefreshAudit(
        symbol=audit.symbol,
        source=audit.source,
        rows=audit.rows,
        merged_rows=audit.merged_rows,
        status=audit.status,
        error=audit.error,
        attempts=attempts,
    )


def _failed_audit(symbol: str, error: str, attempts: int) -> DailyRefreshAudit:
    return DailyRefreshAudit(
        symbol=normalize_ts_code(symbol),
        source="tushare",
        rows=0,
        merged_rows=0,
        status="failed",
        error=error,
        attempts=attempts,
    )


def _refresh_one_symbol_once(
    symbol: str,
    name: str,
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    cache_dir: Path,
) -> DailyRefreshAudit:
    tushare = TushareDataFetcher(cache_dir=cache_dir / "tushare")
    ts_df = tushare.get_stock_daily(symbol, start_date, end_date, adjust=adjust)
    merged = normalize_tushare_daily(ts_df, symbol)
    merged["name"] = name
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=output_dir.parent))
    dataset = output_dir.name
    key = normalize_ts_code(symbol)
    existing = store.read_frame(dataset, key)
    if not existing.empty:
        if "date" in existing.columns:
            existing["date"] = pd.to_datetime(existing["date"])
        if "trade_date" not in existing.columns and "date" in existing.columns:
            existing["trade_date"] = existing["date"].dt.strftime("%Y%m%d")
        merged = (
            pd.concat([existing, merged], ignore_index=True, sort=False)
            .sort_values("date")
            .drop_duplicates("trade_date", keep="last")
            .reset_index(drop=True)
            )
    store.write_frame(merged, dataset, key)
    return build_tushare_daily_audit(symbol, rows=len(ts_df), merged_rows=len(merged))


def refresh_one_symbol(
    symbol: str,
    name: str,
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    cache_dir: Path,
    limiter: RequestLimiter,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    retry_jitter: float,
) -> DailyRefreshAudit:
    attempts = max(1, retries + 1)
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            limiter.wait()
            audit = _refresh_one_symbol_once(symbol, name, start_date, end_date, adjust, output_dir, cache_dir)
            return _with_attempt(audit, attempt)
        except Exception as exc:
            last_error = str(exc)
            if attempt >= attempts or not _is_retryable_error(last_error):
                return _failed_audit(symbol, last_error, attempt)
            delay = min(retry_max_delay, retry_base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, retry_jitter) if retry_jitter > 0 else 0
            print(
                f"retry scheduled: {normalize_ts_code(symbol)} attempt={attempt + 1}/{attempts} "
                f"delay={delay:.1f}s error={last_error[:160]}",
                flush=True,
            )
            time.sleep(delay)
    return _failed_audit(symbol, last_error or "unknown error", attempts)


def refresh_daily_data(
    start_date: str,
    end_date: str | None = None,
    board: str = "all",
    adjust: str = "qfq",
    output_dir: Path = DAILY_DIR,
    workers: int = 2,
    limit: int | None = None,
    sleep_between: float = 0.25,
    symbols_file: Path | None = None,
    retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 60.0,
    retry_jitter: float = 1.0,
) -> dict:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_dir = PROJECT_ROOT / "data/cache/source_merge"
    tushare = TushareDataFetcher(cache_dir=cache_dir / "tushare")
    symbols = load_symbols_file(symbols_file, tushare, limit=limit) if symbols_file else load_tushare_symbols(tushare, board=board, limit=limit)
    if not symbols:
        raise RuntimeError("Tushare stock_basic returned no symbols")

    audits: list[DailyRefreshAudit] = []
    total = len(symbols)
    limiter = RequestLimiter(sleep_between)
    if workers <= 1:
        for n, (symbol, name) in enumerate(symbols, start=1):
            audits.append(
                refresh_one_symbol(
                    symbol,
                    name,
                    start_date,
                    end_date,
                    adjust,
                    output_dir,
                    cache_dir,
                    limiter,
                    retries,
                    retry_base_delay,
                    retry_max_delay,
                    retry_jitter,
                )
            )
            if n % 100 == 0 or n == total:
                ok = sum(1 for audit in audits if audit.status != "failed")
                failed = sum(1 for audit in audits if audit.status == "failed")
                print(f"refresh progress: {n}/{total} done, success={ok}, failed={failed}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for symbol, name in symbols:
                futures[
                    executor.submit(
                        refresh_one_symbol,
                        symbol,
                        name,
                        start_date,
                        end_date,
                        adjust,
                        output_dir,
                        cache_dir,
                        limiter,
                        retries,
                        retry_base_delay,
                        retry_max_delay,
                        retry_jitter,
                    )
                ] = symbol
            for n, future in enumerate(as_completed(futures), start=1):
                audits.append(future.result())
                if n % 100 == 0 or n == len(futures):
                    ok = sum(1 for audit in audits if audit.status != "failed")
                    failed = sum(1 for audit in audits if audit.status == "failed")
                    print(f"refresh progress: {n}/{len(futures)} done, success={ok}, failed={failed}", flush=True)

    audit_dir = AUDIT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_df = audits_to_frame(audits)
    audit_path = audit_dir / "daily_source_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    failed_symbols_path = audit_dir / "failed_symbols.csv"
    failed_df = audit_df.loc[audit_df["status"] == "failed", ["symbol", "error", "attempts"]].copy()
    failed_df.to_csv(failed_symbols_path, index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_policy": "tushare_only_daily",
        "start_date": start_date,
        "end_date": end_date,
        "board": board,
        "adjust": adjust,
        "output_dir": str(output_dir),
        "storage_backend": os.getenv("MARKET_DATA_BACKEND", "mysql"),
        "market_data_sql_url_configured": bool(os.getenv("MARKET_DATA_SQL_URL")),
        "market_data_mirror_parquet": os.getenv("MARKET_DATA_MIRROR_PARQUET", "1").lower() not in {"0", "false", "no"},
        "audit_path": str(audit_path),
        "failed_symbols_path": str(failed_symbols_path),
        "symbols": len(symbols),
        "symbols_file": str(symbols_file) if symbols_file else None,
        "workers": workers,
        "min_request_interval_seconds": sleep_between,
        "retries": retries,
        "retry_base_delay_seconds": retry_base_delay,
        "retry_max_delay_seconds": retry_max_delay,
        "retry_jitter_seconds": retry_jitter,
        "success": int((audit_df["status"] != "failed").sum()),
        "failed": int((audit_df["status"] == "failed").sum()),
        "retried_symbols": int((audit_df["attempts"] > 1).sum()),
    }
    manifest_path = audit_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh daily data from Tushare only.")
    parser.add_argument("--start", default="20100101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD; default today")
    parser.add_argument("--board", default="all", choices=["all", "main", "gem"], help="Stock universe")
    parser.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", "none"], help="Adjustment type")
    parser.add_argument("--output-dir", type=Path, default=DAILY_DIR, help="Daily parquet output directory")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent workers; keep low to leave provider rate-limit buffer")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for smoke tests")
    parser.add_argument("--symbols-file", type=Path, default=None, help="Optional CSV/TXT with a symbol column to refresh only selected symbols")
    parser.add_argument("--sleep", type=float, default=0.25, help="Minimum seconds between request starts across workers")
    parser.add_argument("--retries", type=int, default=3, help="Retries per symbol for retryable network/rate-limit errors")
    parser.add_argument("--retry-base-delay", type=float, default=2.0, help="Initial retry delay seconds")
    parser.add_argument("--retry-max-delay", type=float, default=60.0, help="Maximum retry delay seconds")
    parser.add_argument("--retry-jitter", type=float, default=1.0, help="Random retry jitter seconds")
    args = parser.parse_args()

    adjust = None if args.adjust == "none" else args.adjust
    result = refresh_daily_data(
        start_date=args.start,
        end_date=args.end,
        board=args.board,
        adjust=adjust,
        output_dir=args.output_dir,
        workers=args.workers,
        limit=args.limit,
        sleep_between=args.sleep,
        symbols_file=args.symbols_file,
        retries=args.retries,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
        retry_jitter=args.retry_jitter,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
