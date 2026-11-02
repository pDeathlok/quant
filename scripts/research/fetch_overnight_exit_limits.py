"""Fill Q4 validation exit-date limits in a separate, resumable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from overnight_right_signal_source import load_events

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    side = parser.add_mutually_exclusive_group()
    side.add_argument("--left-same-day", action="store_true")
    side.add_argument("--right-same-day", action="store_true")
    args = parser.parse_args()
    output = ROOT / "reports/research/overnight_exit_limit_history_2025q4_v1"
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    if args.left_same_day or args.right_same_day:
        output = ROOT / "reports/research/overnight_left_same_day_limits_2025h2_v1"
        source = ROOT / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
        if args.right_same_day:
            output = ROOT / "reports/research/overnight_right_same_day_limits_2025h2_v1"
            source = ROOT / "reports/research/overnight_current_right_signals_v2/all_daily_signals.parquet"
            events, _ = load_events(source)
            keys = events[["symbol", "date"]]
        else:
            keys = pd.read_parquet(source, columns=["symbol", "date"])
        calendar = pd.read_parquet(ROOT / "reports/research/overnight_same_day_market_limit_gated_v2/market_context.parquet", columns=["date"]).sort_values("date")
        calendar["exit_date"] = calendar.date.shift(-1)
        keys = keys.merge(calendar, on="date", how="left", validate="many_to_one")
        keys = keys[keys.date.ge("2025-07-01") & keys.date.lt("2026-01-01") & keys.exit_date.lt("2026-01-01")]
        keys = pd.concat([keys[["symbol", "exit_date"]], keys[["symbol", "date"]].rename(columns={"date": "exit_date"})]).drop_duplicates()
    else:
        keys = pd.read_parquet(source, columns=["symbol", "date", "exit_date"])
        keys = keys.loc[keys.date.ge("2025-10-01") & keys.date.lt("2026-01-01") & keys.exit_date.lt("2026-01-01")]
        keys = keys[["symbol", "exit_date"]].dropna().drop_duplicates()
    output.mkdir(exist_ok=True)
    manifest = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "wanted_exit_keys": len(keys),
                "scope": ("all current-day left signal entry+exit keys in2025H2; no legacy-pool intersection"
                          if args.left_same_day else "all Q4 validation candidates, not outcome-selected winners"),
                "provider": "https://tushare.pro/document/2?doc_id=183"}
    if args.right_same_day:
        manifest["scope"] = "all rebuilt current-day right signal entry+exit keys in2025H2; no legacy-pool intersection"
    mp = output / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Resume source mismatch")
    mp.write_text(json.dumps(manifest, indent=2))
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    audit = []
    with httpx.Client(timeout=25) as client:
        def request(params: dict) -> pd.DataFrame:
            try:
                response = client.post("https://api.tushare.pro", json={"api_name": "stk_limit", "token": token,
                    "params": params, "fields": "ts_code,trade_date,pre_close,up_limit,down_limit"})
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError):
                raise RuntimeError("Tushare transport failed; credentials not logged") from None
            if body.get("code") != 0 or body.get("data") is None:
                raise RuntimeError(f"Tushare provider code {body.get('code')}; stopped without retrying quota failures")
            data = body["data"]
            result = pd.DataFrame(data["items"], columns=data["fields"])
            time.sleep(.4)
            return result

        groups = list(keys.groupby("exit_date"))
        for index, (date, part) in enumerate(groups, 1):
            day, wanted = date.strftime("%Y%m%d"), set(part.symbol)
            path = output / f"{day}.parquet"
            if path.exists():
                result = pd.read_parquet(path)
            else:
                result = request({"trade_date": day})
                if len(result) >= 5800:
                    extra = [request({"trade_date": day, "ts_code": symbol}) for symbol in sorted(wanted - set(result.ts_code))]
                    result = pd.concat([result, *extra], ignore_index=True)
                if not result.trade_date.astype(str).eq(day).all() or result.ts_code.duplicated().any():
                    raise ValueError("Unexpected provider dates or duplicate limit keys")
                result = result[result.ts_code.isin(wanted)].copy()
                result.to_parquet(path, index=False)
            audit.append({"date": day, "wanted": len(wanted), "returned": len(result),
                          "missing": len(wanted - set(result.ts_code)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            pd.DataFrame(audit).to_csv(output / "coverage.csv", index=False)
            print(f"exit limits {index}/{len(groups)} {day}, missing={audit[-1]['missing']}", flush=True)
    (output / "complete.json").write_text(json.dumps({"dates": len(audit),
        "missing_keys": sum(r["missing"] for r in audit), "wanted_exit_keys": len(keys)}, indent=2))


if __name__ == "__main__":
    main()
