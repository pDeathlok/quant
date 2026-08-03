#!/usr/bin/env python
"""Audit historical coverage and point-in-time usability of long-factor sources."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_json


RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_PATH = PROJECT_ROOT / "reports/long_entry_factor_inventory/source_coverage.json"
DATE_PATTERN = re.compile(r"(\d{8})(?=\.parquet$)")
PARTITION_SPECS = {
    "daily_basic": {
        "pattern": "*.parquet",
        "required": ["trade_date", "ts_code", "pe", "pb"],
        "dedupe": ["trade_date", "ts_code"],
    },
    "margin_detail": {
        "pattern": "tushare_margin_detail_*.parquet",
        "required": ["trade_date", "ts_code", "rzye", "rqye", "rzrqye"],
        "dedupe": ["trade_date", "ts_code"],
    },
    "moneyflow": {
        "pattern": "tushare_moneyflow_*.parquet",
        "required": ["trade_date", "ts_code", "net_mf_amount"],
        "dedupe": ["trade_date", "ts_code"],
    },
    "top_list": {
        "pattern": "tushare_top_list_*.parquet",
        "required": ["trade_date", "ts_code", "reason"],
        "dedupe": ["trade_date", "ts_code", "reason"],
    },
}
TABLE_SPECS = {
    "stock_basic_history": {
        "required": ["ts_code", "name", "list_status", "list_date"],
        "dedupe": ["ts_code"],
        "dates": ["list_date", "delist_date"],
    },
    "holder_trade": {
        "required": ["ts_code", "ann_date", "holder_name", "in_de", "change_vol"],
        "dedupe": ["ts_code", "ann_date", "holder_name", "in_de", "change_vol", "avg_price"],
        "dates": ["ann_date"],
    },
    "pledge_stat": {
        "required": ["ts_code", "end_date", "pledge_count", "pledge_ratio"],
        "dedupe": ["ts_code", "end_date"],
        "dates": ["end_date"],
    },
    "pledge_detail": {
        "required": ["ts_code", "ann_date", "holder_name", "pledge_amount", "start_date"],
        "dedupe": ["ts_code", "ann_date", "holder_name", "start_date", "pledge_amount"],
        "dates": ["ann_date", "start_date", "end_date", "release_date"],
    },
    "fina_indicator": {
        "required": ["ts_code", "ann_date", "end_date", "roe"],
        "dedupe": ["ts_code", "ann_date", "end_date"],
        "dates": ["ann_date", "end_date"],
    },
    "analyst_forecasts": {
        "required": ["source", "ts_code", "report_date", "forecast_year", "eps"],
        "dedupe": [],
        "dates": ["report_date"],
    },
}


def _file_date(path: Path) -> str | None:
    match = DATE_PATTERN.search(path.name)
    return match.group(1) if match else None


def audit_partition_source(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    paths = sorted((RAW_DIR / name).glob(spec["pattern"]))
    dates = [_file_date(path) for path in paths]
    dates = [date for date in dates if date]
    required = list(spec["required"])
    dedupe = list(spec["dedupe"])
    total_rows = 0
    invalid_date_rows = 0
    duplicate_rows = 0
    missing_schema_files = 0
    non_null = {column: 0 for column in required}
    for path in paths:
        metadata = pq.ParquetFile(path)
        total_rows += metadata.metadata.num_rows
        columns = set(metadata.schema.names)
        if not set(required) <= columns:
            missing_schema_files += 1
            continue
        read_columns = list(dict.fromkeys([*required, *dedupe]))
        frame = pd.read_parquet(path, columns=read_columns)
        for column in required:
            non_null[column] += int(frame[column].notna().sum())
        keys = [column for column in dedupe if column in frame.columns]
        if keys:
            duplicate_rows += int(frame.duplicated(keys).sum())
        expected_date = _file_date(path)
        if expected_date and "trade_date" in frame:
            actual = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
            invalid_date_rows += int((actual != expected_date).sum())
    return {
        "files": len(paths),
        "rows": total_rows,
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "missing_schema_files": missing_schema_files,
        "invalid_partition_date_rows": invalid_date_rows,
        "duplicate_key_rows": duplicate_rows,
        "required_non_null_rate": {
            column: round(count / total_rows, 6) if total_rows else None
            for column, count in non_null.items()
        },
    }


def audit_table(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        return {"exists": False, "rows": 0}
    frame = pd.read_parquet(path)
    required = list(spec["required"])
    keys = [column for column in spec["dedupe"] if column in frame.columns]
    result: dict[str, Any] = {
        "exists": True,
        "rows": len(frame),
        "symbols": int(frame["ts_code"].nunique()) if "ts_code" in frame else None,
        "missing_required_columns": sorted(set(required) - set(frame.columns)),
        "duplicate_key_rows": int(frame.duplicated(keys).sum()) if keys else None,
        "required_non_null_rate": {
            column: round(float(frame[column].notna().mean()), 6)
            if column in frame and len(frame)
            else None
            for column in required
        },
    }
    result["date_ranges"] = {}
    for column in spec.get("dates", []):
        if column not in frame:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if len(values):
            result["date_ranges"][column] = {
                "first": values.min().date().isoformat(),
                "last": values.max().date().isoformat(),
            }
    if name == "stock_basic_history" and "list_status" in frame:
        result["list_status_counts"] = {
            str(key): int(value) for key, value in frame["list_status"].value_counts().items()
        }
    return result


def analyst_three_year_coverage() -> dict[str, Any]:
    path = RAW_DIR / "analyst_forecasts.parquet"
    if not path.exists():
        return {"exists": False}
    frame = pd.read_parquet(
        path,
        columns=["source", "ts_code", "report_date", "forecast_year", "eps", "snapshot_only"],
    )
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
    frame["forecast_year"] = pd.to_numeric(frame["forecast_year"], errors="coerce")
    frame["report_year"] = frame["report_date"].dt.year
    latest = frame["report_date"].max()
    recent = frame[frame["report_date"] >= latest - pd.Timedelta(days=120)].copy()
    valid = recent[
        recent["eps"].notna()
        & recent["forecast_year"].between(recent["report_year"], recent["report_year"] + 2)
    ]
    counts = valid.groupby(["source", "ts_code"])["forecast_year"].nunique()
    by_source: dict[str, Any] = {}
    for source, source_frame in frame.groupby("source"):
        source_counts = counts.xs(source, level="source") if source in counts.index.levels[0] else pd.Series(dtype=float)
        by_source[str(source)] = {
            "rows": len(source_frame),
            "symbols": int(source_frame["ts_code"].nunique()),
            "first_report_date": source_frame["report_date"].min().date().isoformat(),
            "last_report_date": source_frame["report_date"].max().date().isoformat(),
            "recent_symbols_with_current_plus_two_year_eps": int((source_counts >= 3).sum()),
            "snapshot_only_rate": round(float(source_frame["snapshot_only"].fillna(False).mean()), 6),
        }
    return {
        "exists": True,
        "latest_report_date": latest.date().isoformat(),
        "by_source": by_source,
        "point_in_time_warning": (
            "Latest consensus snapshots support current pool display, but snapshot-only rows and "
            "DataYes latest consensus must not be backfilled into earlier sample dates. Historical "
            "AkShare research forecast-year labels require source-year validation before training."
        ),
    }


def main() -> None:
    partitions = {
        name: audit_partition_source(name, spec) for name, spec in PARTITION_SPECS.items()
    }
    canonical_dates = {
        _file_date(path)
        for path in (RAW_DIR / "daily_basic").glob("*.parquet")
        if _file_date(path)
    }
    for name in ("margin_detail", "moneyflow", "top_list"):
        source_dates = {
            _file_date(path)
            for path in (RAW_DIR / name).glob(PARTITION_SPECS[name]["pattern"])
            if _file_date(path)
        }
        partitions[name]["missing_vs_daily_basic"] = sorted(canonical_dates - source_dates)
        partitions[name]["extra_vs_daily_basic"] = sorted(source_dates - canonical_dates)
    payload = {
        "schema_version": "long_factor_source_coverage_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "partitions": partitions,
        "tables": {name: audit_table(name, spec) for name, spec in TABLE_SPECS.items()},
        "analyst_three_year_coverage": analyst_three_year_coverage(),
        "known_source_limits": {
            "pledge_detail_provider_duplicate_tail_symbols": 32,
            "report_rc": (
                "Current Tushare account is rate-limited to roughly one request per hour; "
                "not used as a complete historical training source."
            ),
        },
    }
    atomic_write_json(payload, OUTPUT_PATH)
    print(json.dumps({"status": "success", "output": str(OUTPUT_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
