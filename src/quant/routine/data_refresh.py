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

SUSPENDED_STATUS = "no_trade_suspended"


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


def _status_audit(symbol: str, status: str, message: str, attempts: int) -> DailyRefreshAudit:
    return DailyRefreshAudit(
        symbol=normalize_ts_code(symbol),
        source="tushare",
        rows=0,
        merged_rows=0,
        status=status,
        error=message,
        attempts=attempts,
    )


def _parse_trade_date(value: str | pd.Timestamp) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(value).replace("-", ""), format="%Y%m%d", errors="coerce")
    return None if pd.isna(parsed) else parsed


def _format_trade_date(value: pd.Timestamp) -> str:
    return value.strftime("%Y%m%d")


def _symbol_refresh_start(
    existing: pd.DataFrame,
    requested_start: str,
    adjust: str | None = None,
) -> str:
    """Resolve a safe start date for one symbol.

    Adjusted prices are anchored to the request window. Re-fetch the complete
    stored history before merging so old and new rows share one adjustment base.
    Raw prices can continue from the symbol's next missing day.
    """

    requested = _parse_trade_date(requested_start)
    if requested is None or existing.empty:
        return requested_start
    if "trade_date" in existing.columns:
        dates = pd.to_datetime(existing["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in existing.columns:
        dates = pd.to_datetime(existing["date"], errors="coerce")
    else:
        return requested_start
    latest = dates.max()
    if pd.isna(latest):
        return requested_start
    if adjust in {"qfq", "hfq"}:
        earliest = dates.min()
        return requested_start if pd.isna(earliest) else _format_trade_date(earliest)
    next_missing = latest + pd.Timedelta(days=1)
    return _format_trade_date(min(requested, next_missing))


def _latest_existing_trade_date(existing: pd.DataFrame) -> pd.Timestamp | None:
    if existing.empty:
        return None
    if "trade_date" in existing.columns:
        dates = pd.to_datetime(existing["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in existing.columns:
        dates = pd.to_datetime(existing["date"], errors="coerce")
    else:
        return None
    latest = dates.max()
    return None if pd.isna(latest) else latest


_TRADE_CAL_CACHE: dict[tuple[str, str], set[str]] = {}


def _open_trade_dates(tushare: TushareDataFetcher, start_date: str, end_date: str) -> set[str]:
    cache_key = (start_date, end_date)
    if cache_key in _TRADE_CAL_CACHE:
        return _TRADE_CAL_CACHE[cache_key]
    cal = tushare.pro.trade_cal(exchange="", start_date=start_date, end_date=end_date, is_open="1")
    if cal is None or cal.empty or "cal_date" not in cal.columns:
        _TRADE_CAL_CACHE[cache_key] = set()
        return set()
    dates = set(cal["cal_date"].dropna().astype(str))
    _TRADE_CAL_CACHE[cache_key] = dates
    return dates


def _is_fully_suspended(tushare: TushareDataFetcher, symbol: str, start_date: str, end_date: str) -> bool:
    """Return True when Tushare has no daily rows because all open dates are suspended."""

    try:
        suspend = tushare.pro.suspend_d(ts_code=normalize_ts_code(symbol), start_date=start_date, end_date=end_date)
    except Exception:
        return False
    if suspend is None or suspend.empty or "trade_date" not in suspend.columns:
        return False
    suspend_dates = set(
        suspend.loc[
            suspend.get("suspend_type", pd.Series("S", index=suspend.index)).astype(str).str.upper().eq("S"),
            "trade_date",
        ]
        .dropna()
        .astype(str)
    )
    if not suspend_dates:
        return False
    try:
        open_dates = _open_trade_dates(tushare, start_date, end_date)
    except Exception:
        open_dates = set()
    if open_dates:
        return open_dates.issubset(suspend_dates)
    return end_date in suspend_dates


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
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=output_dir.parent))
    dataset = output_dir.name
    key = normalize_ts_code(symbol)
    requested = _parse_trade_date(start_date)
    end = _parse_trade_date(end_date)
    latest_existing = store.latest_trade_date(dataset, key)
    if (
        requested is not None
        and end is not None
        and latest_existing is not None
        and latest_existing >= end
        and requested >= latest_existing
    ):
        return _status_audit(
            symbol,
            "up_to_date",
            f"本地数据已覆盖到 {end_date}，跳过重复刷新",
            attempts=0,
        )
    existing = store.read_frame(dataset, key)
    latest_existing = _latest_existing_trade_date(existing)
    effective_start = _symbol_refresh_start(existing, start_date, adjust=adjust)
    try:
        ts_df = tushare.get_stock_daily(symbol, effective_start, end_date, adjust=adjust)
    except ValueError as exc:
        if "未获取到" in str(exc) and _is_fully_suspended(tushare, symbol, effective_start, end_date):
            return _status_audit(
                symbol,
                SUSPENDED_STATUS,
                f"{effective_start}-{end_date} 全部交易日停牌，无新增日线数据",
                attempts=1,
            )
        raise
    merged = normalize_tushare_daily(ts_df, symbol)
    merged["name"] = name
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
    retry_non_retryable_errors: bool = False,
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
            if attempt >= attempts or (not retry_non_retryable_errors and not _is_retryable_error(last_error)):
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


def _refresh_symbols(
    symbols: list[tuple[str, str]],
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    cache_dir: Path,
    workers: int,
    sleep_between: float,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    retry_jitter: float,
    retry_non_retryable_errors: bool = False,
    progress_prefix: str = "refresh progress",
) -> list[DailyRefreshAudit]:
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
                    retry_non_retryable_errors=retry_non_retryable_errors,
                )
            )
            if n % 100 == 0 or n == total:
                ok = sum(1 for audit in audits if audit.status != "failed")
                failed = sum(1 for audit in audits if audit.status == "failed")
                print(f"{progress_prefix}: {n}/{total} done, success={ok}, failed={failed}", flush=True)
        return audits

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
                    retry_non_retryable_errors,
                )
            ] = symbol
        for n, future in enumerate(as_completed(futures), start=1):
            audits.append(future.result())
            if n % 100 == 0 or n == len(futures):
                ok = sum(1 for audit in audits if audit.status != "failed")
                failed = sum(1 for audit in audits if audit.status == "failed")
                print(f"{progress_prefix}: {n}/{len(futures)} done, success={ok}, failed={failed}", flush=True)
    return audits


def _retry_failed_audits(
    audits: list[DailyRefreshAudit],
    symbols_by_code: dict[str, tuple[str, str]],
    start_date: str,
    end_date: str,
    adjust: str,
    output_dir: Path,
    cache_dir: Path,
    final_retry_rounds: int,
    final_retry_workers: int,
    final_retry_sleep: float,
    retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
    retry_jitter: float,
) -> tuple[list[DailyRefreshAudit], int]:
    """Retry the remaining failed symbols in slower final passes."""

    audit_by_symbol = {audit.symbol: audit for audit in audits}
    retried_symbols: set[str] = set()
    for round_index in range(1, max(0, final_retry_rounds) + 1):
        failed_symbols = sorted(symbol for symbol, audit in audit_by_symbol.items() if audit.status == "failed")
        if not failed_symbols:
            break
        retry_symbols = [symbols_by_code[symbol] for symbol in failed_symbols if symbol in symbols_by_code]
        if not retry_symbols:
            break
        print(
            f"final failed retry round {round_index}/{final_retry_rounds}: "
            f"symbols={len(retry_symbols)} workers={final_retry_workers} sleep={final_retry_sleep}",
            flush=True,
        )
        retry_audits = _refresh_symbols(
            retry_symbols,
            start_date,
            end_date,
            adjust,
            output_dir,
            cache_dir,
            workers=final_retry_workers,
            sleep_between=final_retry_sleep,
            retries=retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retry_jitter=retry_jitter,
            retry_non_retryable_errors=True,
            progress_prefix=f"final retry progress round {round_index}",
        )
        for retry_audit in retry_audits:
            retried_symbols.add(retry_audit.symbol)
            previous = audit_by_symbol.get(retry_audit.symbol)
            cumulative_attempts = retry_audit.attempts + (previous.attempts if previous else 0)
            audit_by_symbol[retry_audit.symbol] = _with_attempt(retry_audit, cumulative_attempts)
    ordered_symbols = [audit.symbol for audit in audits]
    return [audit_by_symbol[symbol] for symbol in ordered_symbols], len(retried_symbols)


def refresh_daily_data(
    start_date: str,
    end_date: str | None = None,
    board: str = "all",
    adjust: str | None = None,
    output_dir: Path = DAILY_DIR,
    workers: int = 2,
    limit: int | None = None,
    sleep_between: float = 0.25,
    symbols_file: Path | None = None,
    retries: int = 3,
    retry_base_delay: float = 2.0,
    retry_max_delay: float = 60.0,
    retry_jitter: float = 1.0,
    final_retry_rounds: int = 2,
    final_retry_workers: int = 4,
    final_retry_sleep: float = 0.8,
) -> dict:
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    cache_dir = PROJECT_ROOT / "data/cache/source_merge"
    tushare = TushareDataFetcher(cache_dir=cache_dir / "tushare")
    symbols = load_symbols_file(symbols_file, tushare, limit=limit) if symbols_file else load_tushare_symbols(tushare, board=board, limit=limit)
    if not symbols:
        raise RuntimeError("Tushare stock_basic returned no symbols")

    audits = _refresh_symbols(
        symbols,
        start_date,
        end_date,
        adjust,
        output_dir,
        cache_dir,
        workers=workers,
        sleep_between=sleep_between,
        retries=retries,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        retry_jitter=retry_jitter,
    )
    final_retry_unique_symbols = 0
    if final_retry_rounds > 0:
        symbols_by_code = {normalize_ts_code(symbol): (symbol, name) for symbol, name in symbols}
        audits, final_retry_unique_symbols = _retry_failed_audits(
            audits,
            symbols_by_code,
            start_date,
            end_date,
            adjust,
            output_dir,
            cache_dir,
            final_retry_rounds=final_retry_rounds,
            final_retry_workers=max(1, final_retry_workers),
            final_retry_sleep=max(sleep_between, final_retry_sleep),
            retries=retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retry_jitter=retry_jitter,
        )

    audit_dir = AUDIT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_df = audits_to_frame(audits)
    audit_path = audit_dir / "daily_source_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    failed_symbols_path = audit_dir / "failed_symbols.csv"
    failed_df = audit_df.loc[audit_df["status"] == "failed", ["symbol", "error", "attempts"]].copy()
    failed_df.to_csv(failed_symbols_path, index=False)
    suspended_symbols_path = audit_dir / "suspended_symbols.csv"
    suspended_df = audit_df.loc[audit_df["status"] == SUSPENDED_STATUS, ["symbol", "error", "attempts"]].copy()
    suspended_df.to_csv(suspended_symbols_path, index=False)

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
        "suspended_symbols_path": str(suspended_symbols_path),
        "symbols": len(symbols),
        "symbols_file": str(symbols_file) if symbols_file else None,
        "workers": workers,
        "min_request_interval_seconds": sleep_between,
        "retries": retries,
        "retry_base_delay_seconds": retry_base_delay,
        "retry_max_delay_seconds": retry_max_delay,
        "retry_jitter_seconds": retry_jitter,
        "final_retry_rounds": final_retry_rounds,
        "final_retry_workers": final_retry_workers,
        "final_retry_sleep_seconds": final_retry_sleep,
        "final_retry_unique_symbols": final_retry_unique_symbols,
        "success": int((audit_df["status"] != "failed").sum()),
        "failed": int((audit_df["status"] == "failed").sum()),
        "suspended_no_data": int((audit_df["status"] == SUSPENDED_STATUS).sum()),
        "retried_symbols": int(((audit_df["attempts"] > 1) & (audit_df["status"] != "failed")).sum()),
    }
    manifest_path = audit_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh daily data from Tushare only.")
    parser.add_argument("--start", default="20100101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD; default today")
    parser.add_argument("--board", default="all", choices=["all", "main", "gem"], help="Stock universe")
    parser.add_argument(
        "--adjust",
        default="none",
        choices=["qfq", "hfq", "none"],
        help="Adjustment type; raw storage is the safe default for incremental refreshes",
    )
    parser.add_argument("--output-dir", type=Path, default=DAILY_DIR, help="Daily parquet output directory")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent workers; keep low to leave provider rate-limit buffer")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol limit for smoke tests")
    parser.add_argument("--symbols-file", type=Path, default=None, help="Optional CSV/TXT with a symbol column to refresh only selected symbols")
    parser.add_argument("--sleep", type=float, default=0.25, help="Minimum seconds between request starts across workers")
    parser.add_argument("--retries", type=int, default=3, help="Retries per symbol for retryable network/rate-limit errors")
    parser.add_argument("--retry-base-delay", type=float, default=2.0, help="Initial retry delay seconds")
    parser.add_argument("--retry-max-delay", type=float, default=60.0, help="Maximum retry delay seconds")
    parser.add_argument("--retry-jitter", type=float, default=1.0, help="Random retry jitter seconds")
    parser.add_argument("--final-retry-rounds", type=int, default=2, help="Slow final retry rounds for symbols still failed after the main pass")
    parser.add_argument("--final-retry-workers", type=int, default=4, help="Workers for final failed-symbol retry rounds")
    parser.add_argument("--final-retry-sleep", type=float, default=0.8, help="Minimum request interval for final failed-symbol retry rounds")
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
        final_retry_rounds=args.final_retry_rounds,
        final_retry_workers=args.final_retry_workers,
        final_retry_sleep=args.final_retry_sleep,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
