"""Audit canonical daily inputs against the exchange calendar before publication."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from quant.data.atomic_io import atomic_write_json
from quant.data.daily_basic_valuation import DERIVATION_COLUMN, REFERENCE_COLUMNS, resolved_feature_mask
from quant.routine.daily_basic_refresh import (
    DAILY_BASIC_FEATURE_COVERAGE,
    LEGACY_BJ_IDENTIFIER_END,
    list_pending_daily_basic_repairs,
)
from quant.routine.paths import PROJECT_ROOT


DAILY_ROOT = PROJECT_ROOT / "data/raw/daily_partitioned"
BASIC_ROOT = PROJECT_ROOT / "data/raw/daily_basic"
REPORT_PATH = PROJECT_ROOT / "data/routine/historical_integrity_latest.json"
BACKFILL_REPORT_PATH = PROJECT_ROOT / "data/routine/historical_daily_backfill_latest.json"
STOCK_BASIC_HISTORY_PATH = PROJECT_ROOT / "data/raw/stock_basic_history.parquet"
BASIC_START_DATE = "20130104"
BSE_FIRST_TRADE_DATE = "20211115"


def audit_historical_inputs(
    open_dates: Iterable[str],
    target_date: str,
    *,
    daily_root: Path = DAILY_ROOT,
    basic_root: Path = BASIC_ROOT,
    basic_start_date: str = BASIC_START_DATE,
    stock_basic_history_path: Path = STOCK_BASIC_HISTORY_PATH,
) -> dict:
    expected = {str(date) for date in open_dates if str(date) <= target_date}
    if not expected or target_date not in expected:
        raise ValueError(f"incomplete trade calendar through {target_date}")
    failures: list[dict] = []
    warnings: list[dict] = []
    missing_traded_symbols: set[str] = set()
    checked_daily = 0
    checked_basic = 0
    valuation_verification: list[dict] = []
    bj_listing_dates: dict[str, str] | None = None
    excluded_bj_dates: set[str] = set()
    excluded_bj_symbols = 0
    excluded_bj_sample: list[dict[str, str]] = []
    calendar_by_month: dict[str, set[str]] = defaultdict(set)
    for date in expected:
        calendar_by_month[date[:6]].add(date)

    for month, month_dates in sorted(calendar_by_month.items()):
        path = daily_root / f"year_month={month}" / "data.parquet"
        if not path.is_file():
            failures.extend(
                {"trade_date": date, "dataset": "daily", "reason": "missing_partition"}
                for date in sorted(month_dates)
            )
            continue
        try:
            frame = pq.read_table(
                path, columns=["trade_date", "ts_code", "open", "high", "low", "close"]
            ).to_pandas()
        except Exception as exc:
            failures.append({
                "month": month, "dataset": "daily", "reason": "unreadable_partition",
                "detail": str(exc),
            })
            continue
        frame["trade_date"] = frame["trade_date"].astype(str)
        found = set(frame["trade_date"])
        for date in sorted(month_dates - found):
            failures.append({
                "trade_date": date, "dataset": "daily", "reason": "missing_trade_date",
            })
        for date, daily in frame.groupby("trade_date", sort=False):
            if date not in month_dates:
                continue
            checked_daily += 1
            codes = daily["ts_code"].dropna().astype(str)
            daily_codes = set(codes)
            if codes.empty or codes.duplicated().any():
                failures.append({
                    "trade_date": date, "dataset": "daily",
                    "reason": "empty_or_duplicate_symbols",
                })
            if daily[["open", "high", "low", "close"]].isna().any().any():
                failures.append({"trade_date": date, "dataset": "daily", "reason": "missing_price"})
            if date < basic_start_date:
                continue
            basic_path = basic_root / f"{date}.parquet"
            if not basic_path.is_file():
                failures.append({
                    "trade_date": date, "dataset": "daily_basic", "reason": "missing_file",
                })
                continue
            try:
                available = set(pq.ParquetFile(basic_path).schema_arrow.names)
                needed = ["ts_code", "trade_date", "total_mv", *DAILY_BASIC_FEATURE_COVERAGE,
                          *REFERENCE_COLUMNS, DERIVATION_COLUMN]
                needed = list(dict.fromkeys(needed))
                basic = pd.read_parquet(
                    basic_path, columns=[name for name in needed if name in available]
                )
            except Exception as exc:
                failures.append({
                    "trade_date": date, "dataset": "daily_basic",
                    "reason": "unreadable_file", "detail": str(exc),
                })
                continue
            checked_basic += 1
            basic_codes = (
                basic["ts_code"].dropna().astype(str)
                if "ts_code" in basic else pd.Series(dtype=str)
            )
            if "trade_date" not in basic:
                failures.append({
                    "trade_date": date, "dataset": "daily_basic",
                    "reason": "missing_trade_date_column",
                })
            elif not basic["trade_date"].astype(str).eq(date).all():
                failures.append({
                    "trade_date": date, "dataset": "daily_basic", "reason": "wrong_trade_date",
                })
            missing_codes = sorted(daily_codes - set(basic_codes))
            # daily_basic may include suspended securities; a reverse set
            # difference is not evidence of missing prices.
            if basic_codes.empty or basic_codes.duplicated().any():
                failures.append({
                    "trade_date": date, "dataset": "daily_basic",
                    "reason": "empty_or_duplicate_symbols",
                    "daily_rows": len(codes), "basic_rows": len(basic_codes),
                })
            if missing_codes:
                if date <= LEGACY_BJ_IDENTIFIER_END and bj_listing_dates is None:
                    try:
                        identities = pd.read_parquet(
                            stock_basic_history_path, columns=["ts_code", "list_date"]
                        )
                        bj_listing_dates = {
                            str(code): str(listed)
                            for code, listed in zip(
                                identities["ts_code"], identities["list_date"]
                            )
                            if str(code).endswith(".BJ")
                            and len(str(listed)) == 8
                            and str(listed).isdigit()
                        }
                    except Exception as exc:
                        bj_listing_dates = {}
                        failures.append({
                            "dataset": "stock_basic_history",
                            "reason": "unavailable_bj_listing_history",
                            "detail": str(exc),
                        })
                required_missing = []
                for code in missing_codes:
                    listed = (bj_listing_dates or {}).get(code)
                    if (
                        date <= LEGACY_BJ_IDENTIFIER_END
                        and code.endswith(".BJ")
                        and listed is not None
                        and date < max(BSE_FIRST_TRADE_DATE, listed)
                    ):
                        excluded_bj_dates.add(date)
                        excluded_bj_symbols += 1
                        if len(excluded_bj_sample) < 10:
                            excluded_bj_sample.append({
                                "trade_date": date,
                                "ts_code": code,
                                "first_required_date": max(BSE_FIRST_TRADE_DATE, listed),
                            })
                    else:
                        required_missing.append(code)
                if required_missing:
                    failures.append({
                        "trade_date": date, "dataset": "daily_basic",
                        "reason": "missing_price_symbols",
                        "symbols": required_missing[:20], "count": len(required_missing),
                    })
            if "turnover_rate" in basic and "ts_code" in basic:
                traded = set(
                    basic.loc[
                        pd.to_numeric(basic["turnover_rate"], errors="coerce").gt(0),
                        "ts_code",
                    ].dropna().astype(str)
                )
                missing_traded = sorted(traded - daily_codes)
                if missing_traded:
                    missing_traded_symbols.update(missing_traded)
                    failures.append({
                        "trade_date": date, "dataset": "daily", "reason": "missing_traded_price",
                        "count": len(missing_traded), "symbols": missing_traded,
                        "symbols_sample": missing_traded[:20],
                    })
            if "pe_ttm" in basic and set(REFERENCE_COLUMNS).issubset(basic.columns):
                observed = pd.to_numeric(basic["pe_ttm"], errors="coerce").notna()
                resolved = resolved_feature_mask(basic, "pe_ttm")
                valuation_verification.append({
                    "trade_date": date,
                    "numeric_nonnull_rows": int(observed.sum()),
                    "confirmed_loss_rows": int((resolved & ~observed).sum()),
                    "unresolved_rows": int((~resolved).sum()),
                    "unresolved_symbols": basic.loc[~resolved, "ts_code"].tolist(),
                })
            for column, threshold in DAILY_BASIC_FEATURE_COVERAGE.items():
                coverage = (
                    float(resolved_feature_mask(basic, column).mean())
                    if column in basic and len(basic) else 0.0
                )
                if coverage < threshold:
                    failures.append({
                        "trade_date": date, "dataset": "daily_basic", "reason": "feature_coverage",
                        "column": column, "actual": round(coverage, 6), "minimum": threshold,
                    })

    missing_basic_dates = {
        item.get("trade_date") for item in failures
        if item.get("dataset") == "daily_basic" and item.get("reason") == "missing_file"
    }
    for date in sorted(expected):
        if date < basic_start_date:
            continue
        if not (basic_root / f"{date}.parquet").is_file() and date not in missing_basic_dates:
            failures.append({
                "trade_date": date, "dataset": "daily_basic", "reason": "missing_file",
            })
    pending = [date for date in list_pending_daily_basic_repairs(basic_root) if date <= target_date]
    for date in pending:
        failures.append({
            "trade_date": date, "dataset": "daily_basic", "reason": "pending_source_repair",
        })
    return {
        "status": "success" if not failures else "failed",
        "target_trade_date": target_date,
        "audited_at": datetime.now().isoformat(timespec="seconds"),
        "calendar_dates": len(expected),
        "daily_dates_checked": checked_daily,
        "daily_basic_dates_checked": checked_basic,
        "daily_basic_start_date": basic_start_date,
        "valuation_verification": valuation_verification,
        "failure_count": len(failures),
        "failures": failures,
        "warning_count": len(warnings),
        "warnings": warnings,
        "source_exclusions": {
            "legacy_bj_before_listing": {
                "date_count": len(excluded_bj_dates),
                "symbol_count": excluded_bj_symbols,
                "sample": excluded_bj_sample,
            },
        },
        "missing_traded_price_symbols": sorted(missing_traded_symbols),
    }


def audit_with_tushare(target_date: str, *, report_path: Path = REPORT_PATH) -> dict:
    from quant.data.tushare_fetcher import TushareDataFetcher

    calendar = TushareDataFetcher().pro.trade_cal(
        exchange="", start_date="20100101", end_date=target_date, is_open="1"
    )
    if calendar is None or calendar.empty or "cal_date" not in calendar:
        raise RuntimeError("Tushare trade calendar unavailable for historical audit")
    result = audit_historical_inputs(calendar["cal_date"].astype(str), target_date)
    atomic_write_json(result, report_path)
    return result


def pending_backfill_change_dates(path: Path = BACKFILL_REPORT_PATH) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if payload.get("acknowledged") is True:
        return []
    return sorted(set(str(date) for date in payload.get("changed_dates", [])))


def acknowledge_backfill_change_dates(
    dates: list[str], path: Path = BACKFILL_REPORT_PATH,
) -> None:
    if not dates:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError("historical backfill is not complete")
    if sorted(set(payload.get("changed_dates", []))) != sorted(set(dates)):
        raise RuntimeError("historical backfill report changed before publication acknowledgement")
    payload["acknowledged"] = True
    atomic_write_json(payload, path)
    start, end = payload.get("start_date"), payload.get("end_date")
    if start and end:
        cache_dir = PROJECT_ROOT / "data/cache/historical_daily_backfill"
        for cached in cache_dir.glob(f"*_{start}_{end}.parquet"):
            try:
                cached.unlink()
            except OSError:
                pass
