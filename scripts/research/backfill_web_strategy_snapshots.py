"""Backfill web strategy recommendation snapshots into MySQL.

This script is intentionally service-layer based: it reuses the same payload
builders as the web app and only upserts dated recommendation snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _business_dates(start: str, end: str) -> list[str]:
    start_ts = pd.to_datetime(start, errors="raise")
    end_ts = pd.to_datetime(end, errors="raise")
    return [item.strftime("%Y-%m-%d") for item in pd.bdate_range(start_ts, end_ts)]


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _verify_mysql(dates: list[str]) -> dict[str, Any]:
    from sqlalchemy import text

    from quant.data import MarketDataStore, MarketDataStoreConfig
    from quant.webapp.services import LONG_STOCK_POOL_SNAPSHOT_TABLE, SELECTOR_SNAPSHOT_TABLE

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=PROJECT_ROOT / "data"))
    if not store.config.sql_url:
        return {"backend": "json", "selector_rows": 0, "long_rows": 0}

    with store._engine().begin() as conn:
        selector_rows = conn.execute(
            text(
                f"""
                SELECT signal_date, COUNT(*) AS rows_count, SUM(stock_count) AS stock_count
                FROM {SELECTOR_SNAPSHOT_TABLE}
                WHERE signal_date BETWEEN :start_date AND :end_date
                GROUP BY signal_date
                ORDER BY signal_date
                """
            ),
            {"start_date": dates[0], "end_date": dates[-1]},
        ).mappings().all()
        long_rows = conn.execute(
            text(
                f"""
                SELECT signal_date, variant, stock_count
                FROM {LONG_STOCK_POOL_SNAPSHOT_TABLE}
                WHERE signal_date BETWEEN :start_date AND :end_date
                ORDER BY signal_date, variant
                """
            ),
            {"start_date": dates[0], "end_date": dates[-1]},
        ).mappings().all()
    return {
        "backend": "mysql",
        "selector_rows": [dict(row) for row in selector_rows],
        "long_rows": [dict(row) for row in long_rows],
    }


def backfill(
    start: str,
    end: str,
    include_selector: bool,
    include_long: bool,
    long_variants: list[str],
) -> dict[str, Any]:
    _load_env(PROJECT_ROOT / ".env")

    from quant.webapp.services import (
        _build_long_stock_pool_cached,
        _build_tea_master_stock_pool_cached,
        _clear_selector_caches,
        _refresh_long_stock_pool_variants,
        _write_strategy_pool_snapshots,
        get_stock_selector_payload,
    )

    dates = _business_dates(start, end)
    _clear_selector_caches()
    _build_long_stock_pool_cached.cache_clear()
    _build_tea_master_stock_pool_cached.cache_clear()

    selector_results: list[dict[str, Any]] = []
    long_results: list[dict[str, Any]] = []
    for signal_date in dates:
        if include_selector:
            payload = get_stock_selector_payload(
                signal_date=signal_date,
                include_extended=True,
                use_cache=False,
                full_snapshot=True,
            )
            written = _write_strategy_pool_snapshots(payload, include_extended=True)
            selector_results.append(
                {
                    "signal_date": payload.get("signal_date"),
                    "stocks": len(payload.get("stocks") or []),
                    "strategy_pools": written,
                }
            )
            print(
                json.dumps(
                    {"step": "selector", "signal_date": signal_date, "stocks": len(payload.get("stocks") or [])},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if include_long:
            variants = _refresh_long_stock_pool_variants(long_variants, signal_date)
            long_results.extend(variants)
            print(
                json.dumps({"step": "long", "signal_date": signal_date, "variants": variants}, ensure_ascii=False),
                flush=True,
            )

    verification = _verify_mysql(dates)
    return {
        "status": "success",
        "start": dates[0] if dates else start,
        "end": dates[-1] if dates else end,
        "dates": dates,
        "selector": selector_results,
        "long": long_results,
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill web strategy snapshots into MySQL")
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default="2026-06-17")
    parser.add_argument("--skip-selector", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--long-variants", default="tea,tea_safe,v44")
    args = parser.parse_args()

    result = backfill(
        start=args.start,
        end=args.end,
        include_selector=not args.skip_selector,
        include_long=not args.skip_long,
        long_variants=[item.strip() for item in args.long_variants.split(",") if item.strip()],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
