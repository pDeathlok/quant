"""Resumable exact-limit history for the existing overnight research universe."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    output = ROOT / "reports/research/overnight_limit_history"
    output.mkdir(exist_ok=True)
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    universe = pd.read_parquet(ROOT / "reports/research/tail_overnight/dataset.parquet",
                               columns=["symbol", "date"])
    groups = list(universe.groupby("date"))
    audit = []
    with httpx.Client(timeout=25) as client:
        def request(params: dict) -> pd.DataFrame:
            for attempt in range(3):
                try:
                    response = client.post("https://api.tushare.pro", json={
                        "api_name": "stk_limit", "token": token, "params": params,
                        "fields": "ts_code,trade_date,pre_close,up_limit,down_limit"})
                    if response.status_code != 200:
                        raise RuntimeError("non-200 response")
                    body = response.json()
                    if body.get("code") != 0 or body.get("data") is None:
                        raise RuntimeError("provider returned error")
                    data = body["data"]
                    result = pd.DataFrame(data["items"], columns=data["fields"])
                    time.sleep(.35)
                    return result
                except (httpx.HTTPError, ValueError, RuntimeError):
                    if attempt == 2:
                        raise RuntimeError("Tushare limit query failed; credentials not logged") from None
                    time.sleep(2)
            raise AssertionError("unreachable")

        for number, (date, part) in enumerate(groups, 1):
            day = date.strftime("%Y%m%d")
            path = output / f"{day}.parquet"
            wanted = set(part.symbol)
            supplemental = 0
            if path.exists():
                result = pd.read_parquet(path)
            else:
                result = request({"trade_date": day})
                if not result.empty and not result.trade_date.astype(str).eq(day).all():
                    raise ValueError("provider returned unexpected date")
                missing = wanted - set(result.ts_code)
                # The documented full-day response cap is 5800. Query missing
                # candidates individually rather than silently accepting truncation.
                if len(result) >= 5800:
                    extra = []
                    for symbol in sorted(missing):
                        extra.append(request({"trade_date": day, "ts_code": symbol}))
                        supplemental += 1
                    result = pd.concat([result, *extra], ignore_index=True)
                result = result[result.ts_code.isin(wanted)].copy()
                if result.duplicated(["ts_code", "trade_date"]).any():
                    raise ValueError("duplicate price-limit keys")
                result.to_parquet(path, index=False)
            audit.append({"date": day, "requested_stocks": len(wanted), "returned_stocks": len(result),
                          "missing": len(wanted - set(result.ts_code)), "supplemental_requests": supplemental})
            pd.DataFrame(audit).to_csv(output / "coverage.csv", index=False)
            if number % 10 == 0 or number == len(groups):
                print(f"limit history {number}/{len(groups)} through {day}; missing this date {audit[-1]['missing']}", flush=True)
    (output / "complete.json").write_text(json.dumps({"dates": len(groups),
        "missing_stock_days": sum(row["missing"] for row in audit),
        "source": "https://tushare.pro/document/2?doc_id=183",
        "scope": "original research candidate universe, exact limits, not fill proof"}, indent=2))


if __name__ == "__main__":
    main()
