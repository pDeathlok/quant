"""Audited, null-only Tushare repair for the current watchlist's canonical daily data."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet
import requests

from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.research.kronos_walk_forward import PROJECT_ROOT, file_sha256

FIELDS = {"open": "open", "high": "high", "low": "low", "close": "close",
          "pre_close": "pre_close", "volume": "vol", "vol": "vol", "amount": "amount"}


def fill_missing_cells(original: pd.DataFrame, response: pd.DataFrame,
                       symbols: set[str]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve all nonnull values and unrelated rows, reject unit/price inconsistencies."""
    if original.duplicated(["ts_code", "trade_date"]).any() or response.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate market-data keys")
    result = original.copy()
    lookup = response.set_index(["ts_code", "trade_date"])
    candidates = original.ts_code.isin(symbols) & original[list(FIELDS)].isna().any(axis=1)
    changes, rejected = [], []
    for index, row in original.loc[candidates].iterrows():
        key = (row.ts_code, row.trade_date)
        if key not in lookup.index:
            rejected.append({"symbol": key[0], "date": key[1], "reason": "missing_from_Tushare_response"})
            continue
        fresh = lookup.loc[key]
        reason = None
        if not np.isfinite(fresh[list(set(FIELDS.values()))].to_numpy(float)).all():
            reason = "nonfinite_Tushare_fields"
        elif fresh.vol <= 0 or fresh.amount <= 0:
            reason = "nonpositive_Tushare_volume_or_amount"
        else:
            vwap = fresh.amount * 10 / fresh.vol  # official daily units: thousand CNY / lots of 100 shares
            if not fresh.low - .02 <= vwap <= fresh.high + .02:
                reason = "Tushare_amount_volume_units_or_price_range_mismatch"
            for column in ("open", "high", "low", "close", "pre_close"):
                if pd.notna(row[column]) and not np.isclose(row[column], fresh[column], atol=.011, rtol=1e-6):
                    reason = f"existing_{column}_conflicts_with_Tushare"
            for column in ("volume", "vol"):
                if pd.notna(row[column]) and not np.isclose(row[column], fresh.vol, atol=1., rtol=1e-5):
                    reason = f"existing_{column}_units_or_value_conflicts_with_Tushare"
        if reason:
            rejected.append({"symbol": key[0], "date": key[1], "reason": reason})
            continue
        for column, source in FIELDS.items():
            if pd.isna(row[column]):
                result.at[index, column] = fresh[source]
                changes.append({"symbol": key[0], "date": key[1], "field": column, "new_value": float(fresh[source])})
    for column in original.columns:
        keep = original[column].notna()
        pd.testing.assert_series_equal(result.loc[keep, column], original.loc[keep, column], check_dtype=False)
    if len(result) != len(original):
        raise AssertionError("repair changed row count")
    return result, changes, rejected


def _token() -> str:
    value = os.getenv("TUSHARE_TOKEN", "")
    path = PROJECT_ROOT / ".env"
    if not value and path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                value = line.strip().split("=", 1)[1].strip().strip("'\"")
                break
    if not value:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    return value


def run() -> Path:
    symbols = set(json.loads((PROJECT_ROOT / "data/research/similar_patterns/watchlist.json").read_text())["symbols"])
    root = PROJECT_ROOT / "data/raw/daily_partitioned"
    dataset = arrow_dataset.dataset(root, format="parquet", partitioning="hive")
    raw = dataset.to_table(columns=["ts_code", "trade_date", *FIELDS],
                           filter=arrow_dataset.field("ts_code").isin(symbols)).to_pandas()
    missing = raw.loc[raw[list(FIELDS)].isna().any(axis=1)].copy()
    audit = PROJECT_ROOT / "data/raw/source_audit" / ("kronos_watchlist_repair_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    audit.mkdir(parents=True)
    atomic_write_parquet(missing, audit / "missing_before.parquet", index=False)
    dates = sorted(missing.trade_date.unique())
    print(f"repair scope: {len(missing)} rows, {len(dates)} dates, {missing.ts_code.nunique()} watchlist stocks", flush=True)
    if not dates:
        atomic_write_json({"status": "already_complete", "changed_cells": 0}, audit / "manifest.json")
        return audit
    token = _token()
    responses, request_log = [], []
    for number, date in enumerate(dates, 1):
        last_error = ""
        for attempt in range(3):
            try:
                reply = requests.post("https://api.tushare.pro", json={
                    "api_name": "daily", "token": token, "params": {"trade_date": date},
                    "fields": "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
                }, timeout=20)
                reply.raise_for_status()
                payload = reply.json()
                if payload.get("code") != 0:
                    raise ValueError(str(payload.get("msg", "Tushare API error")))
                data = payload.get("data") or {}
                fresh = pd.DataFrame(data.get("items", []), columns=data.get("fields", []))
                if fresh.empty or set(fresh.trade_date.astype(str)) != {date}:
                    raise ValueError("empty or wrong-date daily response")
                fresh = fresh.loc[fresh.ts_code.isin(symbols)].copy()
                if fresh.duplicated(["ts_code", "trade_date"]).any():
                    raise ValueError("duplicate response keys")
                atomic_write_parquet(fresh, audit / "responses" / f"{date}.parquet", index=False)
                responses.append(fresh)
                request_log.append({"date": date, "status": "success", "rows": len(fresh), "attempts": attempt + 1})
                print(f"fetched {number}/{len(dates)} {date}: {len(fresh)} watchlist rows", flush=True)
                break
            except Exception as exc:
                last_error = str(exc).replace(token, "<redacted>")[:300]
                time.sleep(min(2 ** attempt, 4))
        else:
            request_log.append({"date": date, "status": "failed", "error": last_error})
            print(f"fetch failed {date}: {last_error}", flush=True)
            # Stop on an authentication/connectivity failure, not 37 repeated retries.
            if not responses:
                atomic_write_json({"status": "fetch_failed", "requests": request_log}, audit / "manifest.json")
                raise RuntimeError(f"Tushare fetch failed; audit: {audit}")
        time.sleep(.15)
    source = pd.concat(responses, ignore_index=True) if responses else pd.DataFrame(columns=["ts_code", "trade_date", *set(FIELDS.values())])
    changes, rejections, partitions = [], [], []
    for month in sorted({date[:6] for date in dates}):
        path = root / f"year_month={month}" / "data.parquet"
        before_hash = file_sha256(path)
        original = parquet.ParquetFile(path).read().to_pandas()
        updated, edits, rejected = fill_missing_cells(original, source, symbols)
        rejections.extend(rejected)
        if not edits:
            continue
        if file_sha256(path) != before_hash:
            raise RuntimeError(f"partition changed concurrently: {month}")
        backup = audit / "backups" / f"year_month={month}" / "data.parquet"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        atomic_write_parquet(updated, path, index=False)
        saved = parquet.ParquetFile(path).read().to_pandas()
        pd.testing.assert_frame_equal(updated, saved)
        changes.extend(edits)
        partitions.append({"path": str(path), "backup": str(backup), "before_sha256": before_hash,
                           "after_sha256": file_sha256(path), "changed_cells": len(edits)})
        atomic_write_json({"status": "publishing", "requests": request_log, "partitions": partitions,
                           "changes": changes, "rejections": rejections}, audit / "manifest.json")
    after = arrow_dataset.dataset(root, format="parquet", partitioning="hive").to_table(
        columns=["ts_code", "trade_date", *FIELDS], filter=arrow_dataset.field("ts_code").isin(symbols)).to_pandas()
    remaining = after.loc[after[list(FIELDS)].isna().any(axis=1)]
    atomic_write_parquet(remaining, audit / "missing_after.parquet", index=False)
    pd.DataFrame(changes).to_csv(audit / "changed_cells.csv", index=False)
    atomic_write_json({"status": "complete" if remaining.empty else "partial", "requests": request_log,
                       "partitions": partitions, "changed_cells": len(changes), "remaining_rows": len(remaining),
                       "rejections": rejections, "source": "Tushare Pro daily over HTTPS",
                       "units": {"vol": "lots of 100 shares", "amount": "thousand CNY"},
                       "docs": "https://tushare.pro/document/1?doc_id=27"}, audit / "manifest.json")
    print(f"repaired {len(changes)} missing cells; {len(remaining)} rows remain incomplete; audit {audit}", flush=True)
    return audit
