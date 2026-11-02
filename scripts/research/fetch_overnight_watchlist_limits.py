"""Exact historical limits for all frozen watchlist dates, not an old candidate subset."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    snapshot = ROOT / "reports/research/kronos_watchlist_multitask_v1/watchlist_snapshot.json"
    symbols = sorted(set(json.loads(snapshot.read_text())["symbols"]))
    output = ROOT / "reports/research/overnight_watchlist_limits_2024_2026"
    output.mkdir(exist_ok=True)
    protocol = {"symbols": symbols, "start": "20240101", "end": "20260911",
                "source": "Tushare stk_limit", "watchlist_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    contract = output / "protocol.json"
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError("Limit cache contract mismatch")
    contract.write_text(json.dumps(protocol, indent=2))
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    rows = []
    with httpx.Client(timeout=25) as client:
        for number, symbol in enumerate(symbols, 1):
            path = output / f"{symbol}.parquet"
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                response = client.post("https://api.tushare.pro", json={
                    "api_name": "stk_limit", "token": token,
                    "params": {"ts_code": symbol, "start_date": protocol["start"], "end_date": protocol["end"]},
                    "fields": "ts_code,trade_date,pre_close,up_limit,down_limit"})
                if response.status_code != 200:
                    raise RuntimeError(f"Limit query HTTP failure for {symbol}")
                body = response.json()
                if body.get("code") != 0 or body.get("data") is None:
                    raise RuntimeError(f"Limit query provider code {body.get('code')} for {symbol}")
                data = body["data"]
                frame = pd.DataFrame(data["items"], columns=data["fields"])
                if (frame.empty or len(frame) >= 5800 or not frame.ts_code.eq(symbol).all()
                        or frame.duplicated(["ts_code", "trade_date"]).any()
                        or not frame.trade_date.between(protocol["start"], protocol["end"]).all()):
                    raise ValueError(f"Limit response contract failed for {symbol}")
                temporary = path.with_suffix(".pending.parquet")
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
                time.sleep(.5)
            rows.append({"symbol": symbol, "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            pd.DataFrame(rows).to_csv(output / "coverage.csv", index=False)
            if number % 10 == 0:
                print(f"watchlist limits {number}/{len(symbols)}", flush=True)
    (output / "complete.json").write_text(json.dumps({"stocks": len(rows), "rows": sum(row["rows"] for row in rows)}))


if __name__ == "__main__":
    main()
