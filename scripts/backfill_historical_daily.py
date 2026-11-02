#!/usr/bin/env python3
"""Restore missing historical daily bars without replacing existing prices."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.data.source_merge import normalize_tushare_market_daily
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.web_refresh_runner import DEFAULT_ENV_PATH, load_env_file


CACHE_DIR = ROOT / "data/cache/historical_daily_backfill"
REPORT_PATH = ROOT / "data/routine/historical_daily_backfill_latest.json"
PROVENANCE_PATH = ROOT / "data/routine/historical_alternate_source_bars.json"


def merge_missing_bars(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = ["ts_code", "trade_date"]
    if incoming.empty:
        return existing, 0
    if incoming.duplicated(keys).any():
        raise ValueError("incoming historical daily bars contain duplicate keys")
    if existing.empty:
        return incoming, len(incoming)
    if existing.duplicated(keys).any():
        raise ValueError("canonical historical daily bars contain duplicate keys")
    old_keys = pd.MultiIndex.from_frame(existing[keys].astype(str))
    new_keys = pd.MultiIndex.from_frame(incoming[keys].astype(str))
    added = incoming.loc[~new_keys.isin(old_keys)]
    if added.empty:
        return existing, 0
    return pd.concat([existing, added], ignore_index=True, sort=False), len(added)


def _fetch_symbol(fetcher: TushareDataFetcher, code: str, start: str, end: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{code}_{start}_{end}.parquet"
    if cache_path.is_file():
        return pd.read_parquet(cache_path)
    error = None
    for attempt in range(4):
        try:
            time.sleep(0.15)
            frame = fetcher.pro.daily(ts_code=code, start_date=start, end_date=end)
            if frame is None or frame.empty:
                return pd.DataFrame()
            if len(frame) >= 6000:
                raise RuntimeError(f"{code} hit Tushare 6000-row response limit")
            if set(frame["ts_code"].astype(str)) != {code}:
                raise ValueError(f"Tushare returned wrong symbol for {code}")
            dates = frame["trade_date"].astype(str)
            if dates.lt(start).any() or dates.gt(end).any():
                raise ValueError(f"Tushare returned out-of-range dates for {code}")
            normalized = normalize_tushare_market_daily(frame)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(normalized, cache_path, index=False)
            return normalized
        except Exception as exc:
            error = exc
            if attempt < 3:
                time.sleep(65 if "频率超限" in str(exc) else 2 ** attempt)
    raise RuntimeError(f"Tushare historical daily failed for {code}: {error}")


def _fetch_date(fetcher: TushareDataFetcher, date: str) -> pd.DataFrame:
    error = None
    for attempt in range(4):
        try:
            time.sleep(0.15)
            frame = fetcher.pro.daily(trade_date=date)
            if frame is None or frame.empty or len(frame) >= 6000:
                raise ValueError(f"Tushare returned empty or truncated daily partition for {date}")
            if set(frame["trade_date"].astype(str)) != {date}:
                raise ValueError(f"Tushare returned wrong trade date for {date}")
            return normalize_tushare_market_daily(frame)
        except Exception as exc:
            error = exc
            if attempt < 3:
                time.sleep(65 if "频率超限" in str(exc) else 2 ** attempt)
    raise RuntimeError(f"Tushare historical daily failed for {date}: {error}")


def verified_sohu_bar(code: str, date: str, rows: list[list[str]], canonical: pd.DataFrame) -> pd.DataFrame:
    """Accept an alternate-source bar only when both neighboring sessions agree."""
    parsed = {row[0].replace("-", ""): row for row in rows}
    if date not in parsed:
        raise ValueError(f"Sohu has no historical bar for {code} {date}")
    reference = canonical.loc[canonical["ts_code"].eq(code)].copy()
    reference["trade_date"] = reference["trade_date"].astype(str)
    checks = {"before": False, "after": False}
    for other_date, row in parsed.items():
        if other_date == date:
            continue
        match = reference.loc[reference["trade_date"].eq(other_date)]
        if match.empty:
            continue
        actual = match.iloc[0]
        for column, position in (("open", 1), ("close", 2), ("low", 5), ("high", 6)):
            if abs(float(actual[column]) - float(row[position])) > 0.011:
                raise ValueError(f"Sohu {code} {other_date} {column} disagrees with Tushare")
        if abs(float(actual["vol"]) - float(row[7])) > 1:
            raise ValueError(f"Sohu {code} {other_date} volume disagrees with Tushare")
        if abs(float(actual["amount"]) - float(row[8]) * 10) > 1:
            raise ValueError(f"Sohu {code} {other_date} amount disagrees with Tushare")
        checks["before" if other_date < date else "after"] = True
    if not all(checks.values()):
        raise ValueError(f"Sohu {code} {date} lacks verified neighboring sessions")
    row = parsed[date]
    previous = reference.loc[reference["trade_date"].lt(date)].sort_values("trade_date")
    following = reference.loc[reference["trade_date"].gt(date)].sort_values("trade_date")
    pre_close = float(previous.iloc[-1]["close"])
    close = float(row[2])
    if following.empty or abs(float(following.iloc[0]["pre_close"]) - close) > 0.011:
        raise ValueError(f"Sohu {code} {date} close disagrees with next Tushare pre_close")
    raw = pd.DataFrame([{
        "ts_code": code, "trade_date": date, "open": float(row[1]),
        "high": float(row[6]), "low": float(row[5]), "close": close,
        "pre_close": pre_close, "change": round(close - pre_close, 4),
        "pct_chg": round((close / pre_close - 1) * 100, 4),
        "vol": float(row[7]), "amount": float(row[8]) * 10,
    }])
    if raw[["open", "high", "low", "close", "vol", "amount"]].le(0).any().any():
        raise ValueError(f"Sohu {code} {date} contains nonpositive market values")
    if raw["high"].iloc[0] < max(raw["open"].iloc[0], close) or raw["low"].iloc[0] > min(raw["open"].iloc[0], close):
        raise ValueError(f"Sohu {code} {date} has invalid OHLC bounds")
    return normalize_tushare_market_daily(raw)


def _fetch_verified_sohu_bar(code: str, date: str, canonical: pd.DataFrame) -> pd.DataFrame:
    day = datetime.strptime(date, "%Y%m%d")
    params = {
        "code": f"cn_{code[:6]}", "start": (day - timedelta(days=7)).strftime("%Y%m%d"),
        "end": (day + timedelta(days=7)).strftime("%Y%m%d"),
        "stat": "1", "order": "D", "period": "d",
    }
    response = requests.get("https://q.stock.sohu.com/hisHq", params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("status") != 0:
        raise ValueError(f"Sohu historical response unavailable for {code} {date}")
    if payload[0].get("code") != params["code"]:
        raise ValueError(f"Sohu returned wrong symbol for {code} {date}")
    return verified_sohu_bar(code, date, payload[0].get("hq") or [], canonical)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", required=True, help="Last completed trade date, YYYYMMDD")
    parser.add_argument("--start", default="20100101")
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for a smoke test")
    parser.add_argument(
        "--symbol", action="append", default=[], help="Backfill only this symbol; repeatable"
    )
    parser.add_argument(
        "--candidate-only", action="store_true", help="Repair only gaps from the latest audit"
    )
    args = parser.parse_args()
    os.environ.update(load_env_file(DEFAULT_ENV_PATH))
    prior_changed_dates: set[str] = set()
    if REPORT_PATH.is_file():
        import json

        prior_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        if prior_report.get("acknowledged") is not True:
            prior_changed_dates = set(prior_report.get("changed_dates") or [])
    fetcher = TushareDataFetcher()
    delisted = fetcher.pro.stock_basic(
        exchange="", list_status="D", fields="ts_code,name,list_date,delist_date"
    )
    if delisted is None or delisted.empty:
        raise RuntimeError("Tushare delisted stock roster unavailable")
    roster = delisted.loc[
        delisted["list_date"].astype(str).le(args.end)
        & delisted["delist_date"].astype(str).ge(args.start)
    ]
    names = dict(zip(roster["ts_code"].astype(str), roster["name"].astype(str)))
    audit_path = ROOT / "data/routine/historical_integrity_latest.json"
    repair_dates: list[str] = []
    missing_traded_dates: dict[str, set[str]] = {}
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        candidates = set(audit.get("missing_traded_price_symbols") or [])
        for item in audit.get("failures", []):
            if item.get("reason") == "missing_traded_price":
                for code in item.get("symbols", item.get("symbols_sample", [])):
                    missing_traded_dates.setdefault(str(code), set()).add(str(item["trade_date"]))
        repair_dates = sorted({
            str(item["trade_date"])
            for item in audit.get("failures", [])
            if item.get("dataset") == "daily"
            and item.get("reason") in {"missing_trade_date", "missing_partition"}
            and item.get("trade_date")
        })
    else:
        candidates = set()
    codes = sorted(
        candidates if args.candidate_only else set(roster["ts_code"].astype(str)) | candidates
    )
    if args.symbol:
        codes = sorted(set(args.symbol))
    if args.limit is not None:
        codes = codes[:args.limit]
    print(f"historical backfill: {len(codes)} symbols", flush=True)

    by_month: dict[str, list[pd.DataFrame]] = {}
    failed: dict[str, str] = {}
    for index, code in enumerate(codes, 1):
        try:
            date_scope = missing_traded_dates.get(code) if args.candidate_only else None
            fetch_start = min(date_scope) if date_scope else args.start
            fetch_end = max(date_scope) if date_scope else args.end
            frame = _fetch_symbol(fetcher, code, fetch_start, fetch_end)
            if not frame.empty:
                if code in names:
                    frame["name"] = names[code]
                for month, part in frame.groupby(frame["trade_date"].astype(str).str[:6]):
                    by_month.setdefault(month, []).append(part)
        except Exception as exc:
            failed[code] = str(exc)
        if index % 25 == 0 or index == len(codes):
            print(f"historical fetch: {index}/{len(codes)}; failed={len(failed)}", flush=True)
    for date in repair_dates:
        try:
            frame = _fetch_date(fetcher, date)
            by_month.setdefault(date[:6], []).append(frame)
        except Exception as exc:
            failed[date] = str(exc)
    if failed:
        atomic_write_json({
            "status": "failed", "failed_symbols": failed,
            "changed_dates": sorted(prior_changed_dates),
        }, REPORT_PATH)
        return 1

    store = MarketDataStore(MarketDataStoreConfig.from_env(root=ROOT / "data/raw"))
    alternate_bars: list[dict[str, str]] = []
    for code in codes:
        for date in sorted(missing_traded_dates.get(code, ())):
            month = date[:6]
            month_source = pd.concat(by_month.get(month, []), ignore_index=True) if by_month.get(month) else pd.DataFrame()
            if not month_source.empty and bool((month_source["ts_code"].eq(code) & month_source["trade_date"].eq(date)).any()):
                continue
            canonical = store.read_market_range(
                "daily", start_date=f"{month}01",
                end_date=pd.Period(month, freq="M").end_time.strftime("%Y%m%d"),
            )
            try:
                recovered = _fetch_verified_sohu_bar(code, date, canonical)
            except Exception as exc:
                failed[f"{code}:{date}"] = str(exc)
                continue
            by_month.setdefault(month, []).append(recovered)
            alternate_bars.append({
                "ts_code": code, "trade_date": date, "source": "Sohu historical daily",
                "verified_against": "Tushare adjacent OHLC, volume, amount, and next pre_close",
                "retrieved_at": datetime.now().isoformat(timespec="seconds"),
            })
    if failed:
        atomic_write_json({
            "status": "failed", "failed_symbols": failed,
            "changed_dates": sorted(prior_changed_dates),
        }, REPORT_PATH)
        return 1
    changed_dates: set[str] = set(prior_changed_dates)
    inserted_rows = 0
    for index, month in enumerate(sorted(by_month), 1):
        source = pd.concat(by_month[month], ignore_index=True, sort=False)
        source = source.drop_duplicates(["ts_code", "trade_date"], keep="last")
        start = f"{month}01"
        end = pd.Period(month, freq="M").end_time.strftime("%Y%m%d")
        existing = store.read_market_range("daily", start_date=start, end_date=end)
        merged, inserted = merge_missing_bars(existing, source)
        mirror_path = ROOT / "data/raw/daily_partitioned" / f"year_month={month}" / "data.parquet"
        if mirror_path.is_file():
            mirror = pd.read_parquet(mirror_path, columns=["ts_code", "trade_date"])
            mirror_keys = pd.MultiIndex.from_frame(mirror.astype(str))
            canonical_keys = pd.MultiIndex.from_frame(merged[["ts_code", "trade_date"]].astype(str))
            mirror_diverged = set(mirror_keys) != set(canonical_keys)
        else:
            mirror_diverged = True
        if inserted or mirror_diverged:
            storage = store.write_market_batch(
                merged, dataset="daily", partition_column="trade_date"
            )
            inserted_rows += inserted
            changed_dates.update(storage.get("changed_partitions") or [])
            if mirror_diverged:
                changed_dates.update(source["trade_date"].astype(str).unique())
            atomic_write_json({
                "status": "running", "changed_dates": sorted(changed_dates),
                "month_completed": month,
            }, REPORT_PATH)
        print(f"historical merge: {index}/{len(by_month)} months; inserted={inserted}", flush=True)
    result = {
        "status": "success", "start_date": args.start, "end_date": args.end,
        "symbols": len(codes), "months": len(by_month), "inserted_rows": inserted_rows,
        "changed_dates": sorted(changed_dates), "failed_symbols": {},
        "alternate_source_bars": len(alternate_bars),
        "alternate_source_provenance": str(PROVENANCE_PATH) if alternate_bars else None,
    }
    if alternate_bars:
        previous = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8")) if PROVENANCE_PATH.is_file() else []
        records = {(item["ts_code"], item["trade_date"]): item for item in previous}
        records.update({(item["ts_code"], item["trade_date"]): item for item in alternate_bars})
        atomic_write_json(list(records.values()), PROVENANCE_PATH)
    atomic_write_json(result, REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
