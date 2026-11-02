"""Resumable full-day limit cache for the expanded observed universe."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-extension", action="store_true")
    parser.add_argument("--event-extension-exits", action="store_true")
    args = parser.parse_args()
    if args.event_extension and args.event_extension_exits:
        parser.error("Choose entry or exit limits")
    source = PROJECT_ROOT / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
    output = PROJECT_ROOT / "reports/research/overnight_unfiltered_limits_v1"
    if args.event_extension:
        source = PROJECT_ROOT / "reports/research/overnight_event_extension_pool_v1/entry_pool.parquet"
        output = PROJECT_ROOT / "reports/research/overnight_event_extension_limits_v1"
        if sha256(source) != json.loads((source.parent / "complete.json").read_text())["dataset_sha256"]:
            raise ValueError("Extension pool changed")
    if args.event_extension_exits:
        source = PROJECT_ROOT / "reports/research/overnight_event_extension_predictions_v1/picks.parquet"
        output = PROJECT_ROOT / "reports/research/overnight_event_extension_exit_limits_v1"
        if sha256(source) != json.loads((source.parent / "complete.json").read_text())["picks_sha256"]:
            raise ValueError("Frozen picks changed")
    keys = pd.read_parquet(source, columns=["symbol", "date"])
    calendar_hash = None
    if args.event_extension_exits:
        calendar_path = PROJECT_ROOT / "reports/research/overnight_trading_calendar_v1/calendar.parquet"
        calendar_hash = sha256(calendar_path)
        calendar = pd.read_parquet(calendar_path)
        days = calendar.loc[calendar.is_open.eq(1), "date"].sort_values()
        following = dict(zip(days.iloc[:-1], days.iloc[1:], strict=True))
        keys["date"] = keys.date.map(following)
        if keys.date.isna().any():
            raise ValueError("Missing exact next trading date")
    manifest = {"source_sha256": sha256(source), "dates": sorted(keys.date.dt.strftime("%Y%m%d").unique().tolist()),
                "endpoint": "stk_limit", "policy": "Full returned day retained; fail on >=5800 row cap, no provider-error retry, no raw overwrite"}
    if calendar_hash is not None:
        manifest["calendar_sha256"] = calendar_hash
    output.mkdir(exist_ok=True)
    mp = output / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Resume universe changed")
    mp.write_text(json.dumps(manifest, indent=2))
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (PROJECT_ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Credential unavailable")
    prior_path = output / "coverage.csv"
    prior = pd.read_csv(prior_path, dtype={"date": str}).set_index("date").to_dict("index") if prior_path.exists() else {}
    records = []
    fields = ["ts_code", "trade_date", "pre_close", "up_limit", "down_limit"]
    with httpx.Client(timeout=25) as client:
        for number, (date, group) in enumerate(keys.groupby("date", sort=True), 1):
            day = date.strftime("%Y%m%d")
            path = output / f"{day}.parquet"
            if path.exists():
                if day in prior and sha256(path) != prior[day]["sha256"]:
                    raise ValueError("Resume cache hash changed")
                raw = pd.read_parquet(path)
            else:
                try:
                    response = client.post("https://api.tushare.pro", json={"api_name": "stk_limit", "token": token,
                        "params": {"trade_date": day}, "fields": ",".join(fields)})
                    response.raise_for_status()
                    body = response.json()
                except (httpx.HTTPError, ValueError):
                    raise RuntimeError(f"Transport error for {day}; credential not logged; no retry") from None
                if body.get("code") != 0 or body.get("data") is None:
                    raise RuntimeError(f"Provider error code {body.get('code')}; no quota retry")
                raw = pd.DataFrame(body["data"]["items"], columns=body["data"]["fields"])
                time.sleep(.4)
            if (not set(fields).issubset(raw.columns) or not 0 < len(raw) < 5800 or raw.ts_code.isna().any()
                    or raw.ts_code.duplicated().any() or not raw.trade_date.astype(str).eq(day).all()):
                raise ValueError(f"Invalid date/keys or possible truncated provider result on {day}")
            if not path.exists():
                raw.to_parquet(path, index=False)
            records.append({"date": day, "returned": len(raw), "wanted": len(group),
                            "missing_keys": len(set(group.symbol) - set(raw.ts_code)), "sha256": sha256(path)})
            pd.DataFrame(records).to_csv(prior_path, index=False)
            if number == 1 or number % 25 == 0 or number == len(manifest["dates"]):
                print(f"Expanded limits {number}/{len(manifest['dates'])}; {day}; rows={len(raw)} missing={records[-1]['missing_keys']}", flush=True)
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Universe changed during fetch")
    summary = {"days": len(records), "missing_keys": sum(r["missing_keys"] for r in records),
               "returned_rows": sum(r["returned"] for r in records), "script_sha256": sha256(Path(__file__)),
               "scope": "Provider date cache only; value validation, entry/exit gate and alias checks still required"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
