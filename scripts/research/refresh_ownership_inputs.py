#!/usr/bin/env python3
"""Refresh auditable raw ownership and broad ETF inputs for short-side research."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Lock
import time
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.data.long_factor_backfill import RequestPolicy, request_frame
from quant.features.ownership_flow import ETF_GROUPS

DATA = ROOT / "data/research/short_side_ownership"
PAGE_SIZE = 5000


class Tushare:
    def __init__(self, token: str, requests_per_second: float):
        self.token = token
        self.interval = 1 / requests_per_second
        self.lock = Lock()
        self.next_request = 0.0
        self.requests = 0

    def call(self, api: str, **params: str | int) -> pd.DataFrame:
        def request() -> pd.DataFrame:
            with self.lock:
                time.sleep(max(0, self.next_request - time.monotonic()))
                self.next_request = time.monotonic() + self.interval
                self.requests += 1
            message = json.dumps({"api_name": api, "token": self.token, "params": params})
            req = urllib.request.Request(
                "https://api.tushare.pro", data=message.encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                result = json.load(response)
            if result.get("code") != 0:
                msg = str(result.get("msg", "")).replace(self.token, "[redacted]")
                raise RuntimeError(f"{api} {result.get('code')}: {msg}")
            data = result.get("data") or {}
            return pd.DataFrame(data.get("items", []), columns=data.get("fields", []))

        return request_frame(request, RequestPolicy(max_retry_wait_seconds=15))

    def pages(self, api: str, size: int, **params: str) -> pd.DataFrame:
        frames, hashes = [], set()
        for offset in range(0, 500_000, size):
            page = self.call(api, limit=size, offset=offset, **params)
            if not page.empty:
                digest = hashlib.sha256(page.to_json(orient="split").encode()).hexdigest()
                if digest in hashes:
                    raise ValueError(f"{api} repeated page at offset {offset}")
                if len(page) > size:
                    raise ValueError(f"{api} ignored page size {size}")
                hashes.add(digest)
                frames.append(page)
            if len(page) < size:
                return pd.concat(frames, ignore_index=True) if frames else page
        raise ValueError(f"{api} exceeded the maximum scan size")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", required=True)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--mode", choices=("backfill", "update"), default="update")
    parser.add_argument("--rate", type=float, default=2.8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".refresh.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refresh(args)


def refresh(args: argparse.Namespace) -> None:
    token = os.getenv("TUSHARE_TOKEN") or dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("Missing TUSHARE_TOKEN")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start or args.rate <= 0 or args.workers < 1:
        raise ValueError("Invalid date range or request settings")
    client = Tushare(token, args.rate)
    jobs = []
    if args.mode == "update" and not (args.root / "backfill_audit.json").exists():
        raise ValueError("Initialize the cache with a successful --mode backfill")
    previous_success = args.root / "last_success.json"
    last_success = (pd.Timestamp(json.loads(previous_success.read_text())["as_of"])
                    if previous_success.exists() else None)
    if args.mode == "update" and last_success is not None and end < last_success:
        raise ValueError("Use a separate cache root for a historical refresh")

    def enqueue(api, partition, required, *, size=None, refresh=False, merge_key=None, **params):
        jobs.append((api, partition, required, size, refresh, merge_key, params))

    enqueue("etf_basic", "universe", {"ts_code", "index_code", "index_name", "list_date"},
            refresh=True)
    date_range = lambda a, b: {"start_date": a.strftime("%Y%m%d"), "end_date": b.strftime("%Y%m%d")}
    for month in pd.period_range(start, end, freq="M"):
        a, b = max(start, month.start_time), min(end, month.end_time.normalize())
        poll_month = end.to_period("M") - 1
        if last_success is not None:
            poll_month = min(poll_month, last_success.to_period("M"))
        current = month >= poll_month
        if args.mode == "backfill" or current:
            for api, fields in (
                ("stk_holdernumber", {"ts_code", "end_date", "ann_date", "holder_num"}),
                ("stk_holdertrade", {"ts_code", "ann_date", "holder_name", "in_de", "change_ratio"}),
            ):
                enqueue(api, str(month), fields, size=2500, refresh=current, **date_range(a, b))
    top_fields = {"ts_code", "ann_date", "end_date", "holder_name",
                  "hold_amount", "hold_ratio", "hold_float_ratio"}
    if args.mode == "update":
        poll_start = end - pd.Timedelta(days=6)
        if last_success is not None:
            poll_start = min(poll_start, last_success + pd.Timedelta(days=1))
        for day in pd.date_range(max(start, poll_start), end):
            date_text = day.strftime("%Y%m%d")
            for api in ("top10_holders", "top10_floatholders"):
                enqueue(api, f"ann_{date_text}", top_fields, size=PAGE_SIZE, refresh=True, ann_date=date_text)
    for quarter in pd.period_range(start, end, freq="Q"):
        if quarter.end_time.normalize() > end:
            continue
        current = quarter >= end.to_period("Q") - 2
        if args.mode == "backfill" or current:
            for api in ("top10_holders", "top10_floatholders"):
                enqueue(api, str(quarter), top_fields,
                        size=PAGE_SIZE, refresh=current, period=quarter.end_time.strftime("%Y%m%d"))
    overlap_start = end - pd.Timedelta(days=14)
    for year in pd.period_range(start, end, freq="Y"):
        if args.mode == "update" and year.end_time.normalize() < overlap_start:
            continue
        current = year == end.to_period("Y") or year.end_time.normalize() >= overlap_start
        a, b = max(start, year.start_time), min(end, year.end_time.normalize())
        for code in sorted({code for group in ETF_GROUPS.values() for code in group}):
            for api, fields, extra in (
                ("fund_share", {"ts_code", "trade_date", "fd_share"}, {}),
                ("fund_nav", {"ts_code", "ann_date", "nav_date", "unit_nav"}, {"market": "E"}),
                ("fund_daily", {"ts_code", "trade_date", "close", "amount"}, {}),
                ("fund_adj", {"ts_code", "trade_date", "adj_factor"}, {}),
            ):
                partition = f"{code}_{year}"
                existing = args.root / api / f"{partition}.json"
                date_key = "nav_date" if api == "fund_nav" else "trade_date"
                merge = None
                first = a
                if args.mode == "update" and existing.exists():
                    previous = json.loads(existing.read_text())["requested"]
                    last = pd.Timestamp(previous["end_date"])
                    first = max(a, min(last, overlap_start))
                    # Keep multiple NAV vintages when the provider supplies them.
                    merge = [date_key, "ann_date"] if api == "fund_nav" else [date_key]
                enqueue(api, partition, fields, refresh=current, merge_key=merge,
                        ts_code=code, **date_range(first, b), **extra)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    def run(job):
        api, partition, required, size, refresh, merge_key, params = job
        path = args.root / api / f"{partition}.parquet"
        audit_path = path.with_suffix(".json")
        audit = {"api": api, "partition": partition, "requested": params}
        if path.exists() and audit_path.exists():
            previous = json.loads(audit_path.read_text())
            unchanged = previous["requested"] == params
            if not refresh and unchanged:
                return {**previous, "status": "cached"}
            if args.mode == "update" and unchanged and api.startswith("top10_"):
                elapsed = pd.Timestamp(stamp) - pd.Timestamp(previous["observed_at"])
                if elapsed < pd.Timedelta(days=7):
                    return {**previous, "status": "cached"}
        try:
            frame = client.pages(api, size, **params) if size else client.call(api, **params)
            if frame.empty and not len(frame.columns):
                frame = pd.DataFrame(columns=sorted(required))
            if required - set(frame):
                raise ValueError(f"missing columns {sorted(required - set(frame))}")
            if api in {"stk_holdernumber", "stk_holdertrade"} and not frame.empty:
                if not frame.ann_date.astype(str).between(params["start_date"], params["end_date"]).all():
                    raise ValueError("provider returned rows outside the requested announcement range")
            if api.startswith("top10_"):
                if "period" in params and (frame.empty or not frame.end_date.astype(str).eq(params["period"]).all()):
                    raise ValueError("empty or wrong-period holdings")
                if "ann_date" in params and not frame.ann_date.astype(str).eq(params["ann_date"]).all():
                    raise ValueError("holdings returned outside the requested announcement date")
            if api.startswith("fund_") and not frame.empty:
                date_key = "nav_date" if api == "fund_nav" else "trade_date"
                if not (frame.ts_code.eq(params["ts_code"]).all()
                        and frame[date_key].astype(str).between(params["start_date"], params["end_date"]).all()):
                    raise ValueError("ETF provider returned rows outside the requested code/date range")
            if path.exists():
                if merge_key:
                    old = pd.read_parquet(path)
                    if frame.empty and not old.empty:
                        raise ValueError("empty increment for previously populated ETF")
                    frame = pd.concat([old, frame], ignore_index=True).drop_duplicates(merge_key, keep="last")
                elif frame.empty and not pd.read_parquet(path).empty:
                    raise ValueError("refusing to replace a populated partition with an empty response")
            atomic_write_parquet(frame, path, index=False)
            audit.update(status="ok", rows=len(frame), observed_at=stamp)
            atomic_write_json(audit, audit_path)
        except (OSError, RuntimeError, ValueError) as exc:
            audit.update(status="failed", error=str(exc).replace(token, "[redacted]"))
        return audit

    audit = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for entry in executor.map(run, jobs):
            audit.append(entry)
            if len(audit) % 50 == 0 or entry["status"] == "failed":
                print(len(audit), "/", len(jobs), entry, flush=True)
    failures = sum(row["status"] == "failed" for row in audit)
    summary = {"status": "partial" if failures else "ok", "as_of": args.end,
               "observed_at": stamp, "requests_per_second": args.rate,
               "requests": client.requests, "mode": args.mode,
               "etf_groups": ETF_GROUPS, "partitions": audit}
    atomic_write_json(summary, args.root / "refresh_audit.json")
    if args.mode == "backfill" and not failures:
        atomic_write_json(summary, args.root / "backfill_audit.json")
    if not failures:
        atomic_write_json({"as_of": args.end, "observed_at": stamp}, previous_success)
    print(f"{len(audit)} partitions, {client.requests} requests, {failures} failures", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
