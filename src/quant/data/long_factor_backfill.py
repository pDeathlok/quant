"""Recoverable Tushare backfills for long-horizon factor sources.

The module deliberately separates one-time historical recovery from the daily
production refresh.  Every request is written to an idempotent partition (or a
deduplicated compact table), and every run publishes an audit manifest.  This
allows a stopped or rate-limited run to resume without discarding completed
work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from quant.data.atomic_io import atomic_write_json, atomic_write_parquet


PROJECT_START_DATE = "20130101"
TRADE_DATE_DATASETS: dict[str, dict[str, Any]] = {
    "margin_detail": {
        "method": "margin_detail",
        "directory": "margin_detail",
        "prefix": "tushare_margin_detail",
        "limit": 6000,
        "required": {"trade_date", "ts_code", "rzye", "rqye", "rzrqye"},
        "dedupe": ["trade_date", "ts_code"],
    },
    "moneyflow": {
        "method": "moneyflow",
        "directory": "moneyflow",
        "prefix": "tushare_moneyflow",
        "limit": 6000,
        "required": {"trade_date", "ts_code", "net_mf_amount"},
        "dedupe": ["trade_date", "ts_code"],
    },
    "top_list": {
        "method": "top_list",
        "directory": "top_list",
        "prefix": "tushare_top_list",
        "limit": 10000,
        "required": {"trade_date", "ts_code", "reason"},
        "dedupe": ["trade_date", "ts_code", "reason"],
    },
}


class DeferredRequest(RuntimeError):
    """The source asked for a wait longer than this foreground run permits."""


@dataclass(frozen=True)
class RequestPolicy:
    retries: int = 3
    sleep_seconds: float = 0.12
    max_retry_wait_seconds: float = 60.0


def _date_text(value: str | pd.Timestamp) -> str:
    text = str(value).strip()
    parsed = (
        pd.to_datetime(text, format="%Y%m%d", errors="raise")
        if text.isdigit() and len(text) == 8
        else pd.Timestamp(value)
    )
    return parsed.strftime("%Y%m%d")


def _redacted_error(exc: Exception) -> str:
    return str(exc)[:1000]


def _retry_delay(message: str, attempt: int, policy: RequestPolicy) -> float:
    if "1次/小时" in message:
        raise DeferredRequest(message)
    if "1次/分钟" in message or ("频率超限" in message and "/分钟" in message):
        requested = 60.0
    else:
        requested = max(0.5, policy.sleep_seconds * (2 ** max(0, attempt - 1)))
    if requested > policy.max_retry_wait_seconds:
        raise DeferredRequest(message)
    return requested


def request_frame(
    operation: Callable[[], pd.DataFrame],
    policy: RequestPolicy,
) -> pd.DataFrame:
    """Call a provider method with bounded retry and explicit deferral."""

    last_error: Exception | None = None
    for attempt in range(1, max(1, policy.retries) + 1):
        try:
            frame = operation()
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, policy.retries):
                break
            time.sleep(_retry_delay(str(exc), attempt, policy))
    assert last_error is not None
    raise last_error


def merge_deduplicated(
    path: Path,
    incoming: pd.DataFrame,
    *,
    dedupe: Iterable[str],
    sort: Iterable[str],
) -> pd.DataFrame:
    """Atomically merge one raw table; reruns preserve the same logical rows."""

    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    if combined.empty and not len(combined.columns):
        return combined
    keys = [column for column in dedupe if column in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    order = [column for column in sort if column in combined.columns]
    if order:
        combined = combined.sort_values(order, kind="mergesort").reset_index(drop=True)
    atomic_write_parquet(combined, path, index=False)
    return combined


def load_open_trade_dates(pro: Any, start_date: str, end_date: str) -> list[str]:
    calendar = pro.trade_cal(
        exchange="",
        start_date=_date_text(start_date),
        end_date=_date_text(end_date),
        is_open="1",
        fields="cal_date,is_open",
    )
    if calendar is None or calendar.empty or "cal_date" not in calendar.columns:
        raise RuntimeError("Tushare trade_cal returned no open dates")
    return sorted(calendar["cal_date"].dropna().astype(str).unique().tolist())


def _partition_valid(path: Path, spec: dict[str, Any], trade_date: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    if not set(spec["required"]) <= set(frame.columns):
        return False
    if frame.empty:
        return True
    dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    if not dates.eq(trade_date).all():
        return False
    keys = [column for column in spec["dedupe"] if column in frame.columns]
    return not keys or not frame.duplicated(keys).any()


def backfill_trade_date_partitions(
    pro: Any,
    dataset: str,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    audit_dir: Path,
    *,
    policy: RequestPolicy = RequestPolicy(),
    force: bool = False,
    max_dates: int | None = None,
    progress_every: int = 20,
) -> dict[str, Any]:
    """Backfill a cross-sectional daily source into one file per trade date."""

    if dataset not in TRADE_DATE_DATASETS:
        raise ValueError(f"unsupported trade-date dataset: {dataset}")
    spec = TRADE_DATE_DATASETS[dataset]
    output_dir = raw_dir / str(spec["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = load_open_trade_dates(pro, start_date, end_date)
    pending: list[str] = []
    skipped = 0
    for trade_date in dates:
        path = output_dir / f"{spec['prefix']}_{trade_date}.parquet"
        if not force and _partition_valid(path, spec, trade_date):
            skipped += 1
        else:
            pending.append(trade_date)
    if max_dates is not None:
        pending = pending[: max(0, max_dates)]

    audits: list[dict[str, Any]] = []
    method = getattr(pro, str(spec["method"]))
    for index, trade_date in enumerate(pending, start=1):
        path = output_dir / f"{spec['prefix']}_{trade_date}.parquet"
        status = "failed"
        rows = 0
        error: str | None = None
        try:
            frame = request_frame(
                lambda d=trade_date: method(trade_date=d),
                policy,
            )
            rows = int(len(frame))
            if rows >= int(spec["limit"]):
                raise RuntimeError(
                    f"{dataset} {trade_date} returned {rows} rows at provider limit; "
                    "partition may be truncated"
                )
            missing = set(spec["required"]) - set(frame.columns)
            if missing:
                raise ValueError(f"{dataset} {trade_date} missing columns: {sorted(missing)}")
            keys = [column for column in spec["dedupe"] if column in frame.columns]
            if keys:
                frame = frame.drop_duplicates(keys, keep="last")
            atomic_write_parquet(frame, path, index=False)
            status = "success"
        except DeferredRequest as exc:
            status = "deferred"
            error = _redacted_error(exc)
        except Exception as exc:
            error = _redacted_error(exc)
        audits.append(
            {
                "dataset": dataset,
                "trade_date": trade_date,
                "status": status,
                "rows": rows,
                "path": str(path),
                "error": error,
            }
        )
        if index % max(1, progress_every) == 0 or index == len(pending):
            ok = sum(item["status"] == "success" for item in audits)
            print(
                f"{dataset}: {index}/{len(pending)} requested success={ok} "
                f"skipped={skipped}",
                flush=True,
            )
        if status == "deferred":
            break
        if policy.sleep_seconds > 0:
            time.sleep(policy.sleep_seconds)

    audit_path = audit_dir / f"{dataset}_requests.csv"
    audit_frame = pd.DataFrame(audits)
    atomic_write_parquet(audit_frame, audit_path.with_suffix(".parquet"), index=False)
    success = sum(item["status"] == "success" for item in audits)
    failed = sum(item["status"] == "failed" for item in audits)
    deferred = sum(item["status"] == "deferred" for item in audits)
    return {
        "dataset": dataset,
        "status": "failed" if failed else ("deferred" if deferred else "success"),
        "open_dates": len(dates),
        "already_complete": skipped,
        "requested": len(audits),
        "success": success,
        "failed": failed,
        "deferred": deferred,
        "rows": int(sum(item["rows"] for item in audits if item["status"] == "success")),
        "output_dir": str(output_dir),
        "audit_path": str(audit_path.with_suffix(".parquet")),
    }


def backfill_stock_universe(
    pro: Any,
    raw_dir: Path,
    *,
    policy: RequestPolicy = RequestPolicy(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize active, delisted, paused, and approved A-share identities."""

    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    fields = (
        "ts_code,symbol,name,area,industry,market,exchange,list_status,"
        "list_date,delist_date,is_hs"
    )
    for status in ("L", "D", "P", "G"):
        try:
            frame = request_frame(
                lambda s=status: pro.stock_basic(
                    exchange="",
                    list_status=s,
                    fields=fields,
                ),
                policy,
            )
            if not frame.empty:
                if "list_status" not in frame.columns:
                    frame["list_status"] = status
                frames.append(frame)
            status_rows.append({"list_status": status, "status": "success", "rows": len(frame)})
        except Exception as exc:
            status_rows.append(
                {
                    "list_status": status,
                    "status": "failed",
                    "rows": 0,
                    "error": _redacted_error(exc),
                }
            )
    if not frames:
        raise RuntimeError("No stock universe rows were fetched")
    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates("ts_code", keep="last")
        .sort_values("ts_code")
        .reset_index(drop=True)
    )
    path = raw_dir / "stock_basic_history.parquet"
    atomic_write_parquet(combined, path, index=False)
    failed = [row for row in status_rows if row["status"] == "failed"]
    return combined, {
        "dataset": "stock_universe",
        "status": "partial" if failed else "success",
        "rows": len(combined),
        "active": int(combined["list_status"].eq("L").sum()),
        "delisted": int(combined["list_status"].eq("D").sum()),
        "status_requests": status_rows,
        "output": str(path),
    }


def _split_date_interval(start_date: str, end_date: str) -> tuple[tuple[str, str], tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    midpoint = start + (end - start) // 2
    return (
        (start.strftime("%Y%m%d"), midpoint.strftime("%Y%m%d")),
        ((midpoint + pd.Timedelta(days=1)).strftime("%Y%m%d"), end.strftime("%Y%m%d")),
    )


def fetch_complete_date_range(
    operation: Callable[[str, str], pd.DataFrame],
    start_date: str,
    end_date: str,
    *,
    provider_limit: int,
    policy: RequestPolicy,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Recursively split full-result windows that touch a provider row limit."""

    pending = [(_date_text(start_date), _date_text(end_date))]
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    while pending:
        start, end = pending.pop(0)
        frame = request_frame(lambda s=start, e=end: operation(s, e), policy)
        rows = len(frame)
        if rows >= provider_limit:
            if start == end:
                raise RuntimeError(
                    f"provider limit reached for indivisible date {start}: rows={rows}"
                )
            left, right = _split_date_interval(start, end)
            pending = [left, right, *pending]
            audits.append(
                {"start_date": start, "end_date": end, "status": "split", "rows": rows}
            )
            continue
        frames.append(frame)
        audits.append(
            {"start_date": start, "end_date": end, "status": "success", "rows": rows}
        )
        if policy.sleep_seconds > 0:
            time.sleep(policy.sleep_seconds)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, audits


def fetch_complete_pages(
    operation: Callable[[int, int], pd.DataFrame],
    *,
    page_size: int,
    policy: RequestPolicy,
    max_pages: int = 100,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fetch a provider result to exhaustion using stable limit/offset pages."""

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    seen_hashes: set[int] = set()
    for page in range(max_pages):
        offset = page * page_size
        frame = request_frame(
            lambda current_offset=offset: operation(current_offset, page_size),
            policy,
        )
        rows = len(frame)
        page_hashes = set(
            pd.util.hash_pandas_object(frame.astype(str), index=False).astype("uint64").tolist()
        )
        if page > 0 and rows and page_hashes <= seen_hashes:
            audits.append(
                {
                    "offset": offset,
                    "limit": page_size,
                    "rows": rows,
                    "status": "duplicate_tail",
                }
            )
            break
        seen_hashes.update(page_hashes)
        frames.append(frame)
        audits.append(
            {"offset": offset, "limit": page_size, "rows": rows, "status": "success"}
        )
        if rows < page_size:
            break
        if policy.sleep_seconds > 0:
            time.sleep(policy.sleep_seconds)
    else:
        raise RuntimeError(
            f"pagination did not terminate after {max_pages} pages of {page_size} rows"
        )
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, audits


def backfill_holder_trade(
    pro: Any,
    start_date: str,
    end_date: str,
    raw_dir: Path,
    audit_dir: Path,
    *,
    policy: RequestPolicy = RequestPolicy(),
    force: bool = False,
    max_years: int | None = None,
) -> dict[str, Any]:
    output_dir = raw_dir / "holder_trade_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(_date_text(start_date))
    end = pd.Timestamp(_date_text(end_date))
    years = list(range(start.year, end.year + 1))
    if max_years is not None:
        years = years[: max(0, max_years)]
    audits: list[dict[str, Any]] = []
    for index, year in enumerate(years, start=1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        path = output_dir / f"{year}.parquet"
        if path.exists() and not force:
            audits.append({"year": year, "status": "skipped", "rows": len(pd.read_parquet(path))})
            continue
        try:
            frame, window_audit = fetch_complete_date_range(
                lambda s, e: pro.stk_holdertrade(start_date=s, end_date=e),
                year_start.strftime("%Y%m%d"),
                year_end.strftime("%Y%m%d"),
                provider_limit=3000,
                policy=policy,
            )
            keys = [
                column
                for column in (
                    "ts_code",
                    "ann_date",
                    "holder_name",
                    "in_de",
                    "change_vol",
                    "avg_price",
                )
                if column in frame.columns
            ]
            if keys:
                frame = frame.drop_duplicates(keys, keep="last")
            atomic_write_parquet(frame, path, index=False)
            audits.append(
                {
                    "year": year,
                    "status": "success",
                    "rows": len(frame),
                    "requests": len(window_audit),
                }
            )
        except DeferredRequest as exc:
            audits.append(
                {"year": year, "status": "deferred", "rows": 0, "error": _redacted_error(exc)}
            )
            break
        except Exception as exc:
            audits.append(
                {"year": year, "status": "failed", "rows": 0, "error": _redacted_error(exc)}
            )
        print(f"holder_trade: {index}/{len(years)} year={year}", flush=True)

    partitions = sorted(output_dir.glob("*.parquet"))
    frames = [pd.read_parquet(path) for path in partitions]
    history = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    compact = merge_deduplicated(
        raw_dir / "holder_trade.parquet",
        history,
        dedupe=("ts_code", "ann_date", "holder_name", "in_de", "change_vol", "avg_price"),
        sort=("ann_date", "ts_code"),
    )
    audit_path = audit_dir / "holder_trade_requests.parquet"
    atomic_write_parquet(pd.DataFrame(audits), audit_path, index=False)
    failed = sum(row["status"] == "failed" for row in audits)
    deferred = sum(row["status"] == "deferred" for row in audits)
    return {
        "dataset": "holder_trade",
        "status": "failed" if failed else ("deferred" if deferred else "success"),
        "years": len(years),
        "partitions": len(partitions),
        "rows": len(compact),
        "symbols": int(compact["ts_code"].nunique()) if "ts_code" in compact else 0,
        "output": str(raw_dir / "holder_trade.parquet"),
        "audit_path": str(audit_path),
    }


def refresh_holder_trade_recent(
    pro: Any,
    end_date: str,
    raw_dir: Path,
    audit_dir: Path,
    *,
    lookback_days: int = 45,
    policy: RequestPolicy = RequestPolicy(),
) -> dict[str, Any]:
    """Refresh a rolling announcement window and merge affected year partitions."""

    end = pd.Timestamp(_date_text(end_date))
    start = end - pd.Timedelta(days=max(1, lookback_days))
    frame, request_audit = fetch_complete_date_range(
        lambda s, e: pro.stk_holdertrade(start_date=s, end_date=e),
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        provider_limit=3000,
        policy=policy,
    )
    keys = ("ts_code", "ann_date", "holder_name", "in_de", "change_vol", "avg_price")
    if not frame.empty:
        present_keys = [column for column in keys if column in frame.columns]
        if present_keys:
            frame = frame.drop_duplicates(present_keys, keep="last")
        if "ann_date" not in frame.columns:
            raise ValueError("holder_trade recent response is missing ann_date")
        years = pd.to_datetime(
            frame["ann_date"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.year
        output_dir = raw_dir / "holder_trade_history"
        output_dir.mkdir(parents=True, exist_ok=True)
        for year, year_frame in frame.groupby(years, dropna=True):
            merge_deduplicated(
                output_dir / f"{int(year)}.parquet",
                year_frame,
                dedupe=keys,
                sort=("ann_date", "ts_code"),
            )
        compact = merge_deduplicated(
            raw_dir / "holder_trade.parquet",
            frame,
            dedupe=keys,
            sort=("ann_date", "ts_code"),
        )
    else:
        compact_path = raw_dir / "holder_trade.parquet"
        compact = pd.read_parquet(compact_path) if compact_path.exists() else pd.DataFrame()
    audit_path = audit_dir / "holder_trade_recent_requests.parquet"
    atomic_write_parquet(pd.DataFrame(request_audit), audit_path, index=False)
    return {
        "dataset": "holder_trade_recent",
        "status": "success",
        "requested_start": start.strftime("%Y%m%d"),
        "requested_end": end.strftime("%Y%m%d"),
        "new_rows": len(frame),
        "total_rows": len(compact),
        "audit_path": str(audit_path),
    }


def backfill_pledge_by_symbol(
    pro: Any,
    dataset: str,
    symbols: Iterable[str],
    raw_dir: Path,
    audit_dir: Path,
    *,
    policy: RequestPolicy = RequestPolicy(),
    force: bool = False,
    symbol_offset: int = 0,
    max_symbols: int | None = None,
    progress_every: int = 50,
    compact_output: bool = True,
) -> dict[str, Any]:
    if dataset not in {"pledge_stat", "pledge_detail"}:
        raise ValueError(f"unsupported pledge dataset: {dataset}")
    output_dir = raw_dir / f"{dataset}_by_symbol"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_symbols = list(dict.fromkeys(str(symbol) for symbol in symbols))
    pending = all_symbols[max(0, symbol_offset) :]
    if max_symbols is not None:
        pending = pending[: max(0, max_symbols)]
    method = getattr(pro, dataset)
    page_size = 1000 if dataset == "pledge_stat" else 1500
    audits: list[dict[str, Any]] = []
    for index, symbol in enumerate(pending, start=1):
        path = output_dir / f"{symbol}.parquet"
        if path.exists() and not force:
            audits.append({"ts_code": symbol, "status": "skipped", "rows": len(pd.read_parquet(path))})
            continue
        try:
            frame, page_audit = fetch_complete_pages(
                lambda offset, limit, s=symbol: method(
                    ts_code=s,
                    limit=limit,
                    offset=offset,
                ),
                page_size=page_size,
                policy=policy,
            )
            keys = (
                [column for column in ("ts_code", "end_date") if column in frame.columns]
                if dataset == "pledge_stat"
                else [
                    column
                    for column in (
                        "ts_code",
                        "ann_date",
                        "holder_name",
                        "start_date",
                        "pledge_amount",
                    )
                    if column in frame.columns
                ]
            )
            if keys:
                frame = frame.drop_duplicates(keys, keep="last")
            atomic_write_parquet(frame, path, index=False)
            pagination_partial = any(
                item.get("status") == "duplicate_tail" for item in page_audit
            )
            audits.append(
                {
                    "ts_code": symbol,
                    "status": "partial" if pagination_partial else "success",
                    "rows": len(frame),
                    "requests": len(page_audit),
                    "pagination_status": (
                        "provider_duplicate_tail" if pagination_partial else "exhausted"
                    ),
                }
            )
        except DeferredRequest as exc:
            audits.append(
                {"ts_code": symbol, "status": "deferred", "rows": 0, "error": _redacted_error(exc)}
            )
            break
        except Exception as exc:
            audits.append(
                {"ts_code": symbol, "status": "failed", "rows": 0, "error": _redacted_error(exc)}
            )
        if index % max(1, progress_every) == 0 or index == len(pending):
            ok = sum(row["status"] in {"success", "skipped", "partial"} for row in audits)
            print(f"{dataset}: {index}/{len(pending)} complete={ok}", flush=True)
        if policy.sleep_seconds > 0:
            time.sleep(policy.sleep_seconds)

    partitions = sorted(output_dir.glob("*.parquet"))
    compact: pd.DataFrame | None = None
    if compact_output:
        frames = [pd.read_parquet(path) for path in partitions]
        history = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        dedupe = (
            ("ts_code", "end_date")
            if dataset == "pledge_stat"
            else ("ts_code", "ann_date", "holder_name", "start_date", "pledge_amount")
        )
        compact = merge_deduplicated(
            raw_dir / f"{dataset}.parquet",
            history,
            dedupe=dedupe,
            sort=("ts_code", "end_date" if dataset == "pledge_stat" else "ann_date"),
        )
    audit_path = audit_dir / f"{dataset}_requests.parquet"
    atomic_write_parquet(pd.DataFrame(audits), audit_path, index=False)
    failed = sum(row["status"] == "failed" for row in audits)
    deferred = sum(row["status"] == "deferred" for row in audits)
    partial = sum(row["status"] == "partial" for row in audits)
    return {
        "dataset": dataset,
        "status": (
            "failed"
            if failed
            else ("deferred" if deferred else ("partial" if partial else "success"))
        ),
        "symbol_offset": max(0, symbol_offset),
        "requested_symbols": len(pending),
        "partitions": len(partitions),
        "rows": len(compact) if compact is not None else None,
        "symbols": (
            int(compact["ts_code"].nunique())
            if compact is not None and "ts_code" in compact
            else None
        ),
        "output": str(raw_dir / f"{dataset}.parquet") if compact_output else None,
        "compacted": compact_output,
        "provider_partial_symbols": partial,
        "audit_path": str(audit_path),
    }


def make_audit_directory(audit_root: Path) -> Path:
    path = audit_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_long_factor_backfill"
    path.mkdir(parents=True, exist_ok=True)
    return path


def publish_manifest(audit_dir: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(payload, audit_dir / "manifest.json")
