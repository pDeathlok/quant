"""Resumable five-minute data for the frozen 90-stock overnight research pool."""

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
    output = ROOT / "reports/research/overnight_minutes_2024_2026"
    output.mkdir(exist_ok=True)
    if (output / "access_limit.json").exists():
        raise RuntimeError("Observed hourly quota prevents bulk download; review access_limit.json before proceeding")
    protocol = {"watchlist_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "symbols": symbols, "api": "stk_mins", "frequency": "5min",
                "start": "2024-01-01", "end": "2026-09-11", "calendar_chunk_days": 180,
                "purpose": "pre-close input and next-session exit-path research; no orders",
                "limitation": "current watchlist retrospective selection bias remains"}
    contract = output / "protocol.json"
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError("Minute cache protocol changed; use a separate output directory")
    if not contract.exists():
        contract.write_text(json.dumps(protocol, indent=2))
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    start, end = pd.Timestamp(protocol["start"]), pd.Timestamp(protocol["end"])
    chunks = []
    while start <= end:
        last = min(start + pd.Timedelta(days=179), end)
        chunks.append((start, last))
        start = last + pd.Timedelta(days=1)
    audit = []
    # Rechecked before use: a discovered hourly quota disables this bulk entry.
    # Wait before the first request too, avoiding an immediately preceding probe.
    last_request = time.monotonic()
    with httpx.Client(timeout=30) as client:
        for number, symbol in enumerate(symbols, 1):
            folder = output / symbol
            folder.mkdir(exist_ok=True)
            for first, last in chunks:
                path = folder / f"{first:%Y%m%d}_{last:%Y%m%d}.parquet"
                if path.exists():
                    frame = pd.read_parquet(path)
                else:
                    for attempt in range(3):
                        try:
                            while time.monotonic() - last_request < 65:
                                time.sleep(min(30, 65 - (time.monotonic() - last_request)))
                            last_request = time.monotonic()
                            response = client.post("https://api.tushare.pro", json={
                                "api_name": "stk_mins", "token": token,
                                "params": {"ts_code": symbol, "freq": "5min",
                                           "start_date": f"{first:%Y-%m-%d} 00:00:00",
                                           "end_date": f"{last:%Y-%m-%d} 23:59:59"},
                                "fields": "ts_code,trade_time,open,high,low,close,vol,amount"})
                            if response.status_code != 200:
                                raise RuntimeError("non-200 response")
                            body = response.json()
                            if body.get("code") != 0 or body.get("data") is None:
                                message = str(body.get("msg", "")).replace(token, "[REDACTED]")[:250]
                                print(f"minute provider code={body.get('code')}: {message}", flush=True)
                                raise RuntimeError("provider error; details withheld to protect credentials")
                            data = body["data"]
                            frame = pd.DataFrame(data["items"], columns=data["fields"])
                            break
                        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                            print(f"minute request attempt {attempt + 1}: {type(exc).__name__}", flush=True)
                            if attempt == 2:
                                raise RuntimeError(f"Minute query failed: {symbol} {first:%Y%m%d}") from None
                            time.sleep(2)
                    if len(frame) >= 8000:
                        raise ValueError("Possible API truncation: reduce chunk before continuing")
                    if not frame.empty:
                        frame["trade_time"] = pd.to_datetime(frame.trade_time)
                        if (not frame.ts_code.eq(symbol).all()
                                or frame.trade_time.dt.normalize().lt(first).any()
                                or frame.trade_time.dt.normalize().gt(last).any()
                                or frame.duplicated(["ts_code", "trade_time"]).any()):
                            raise ValueError("Minute response keys outside requested scope")
                        frame = frame.sort_values("trade_time")
                    temporary = path.with_suffix(".pending.parquet")
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                audit.append({"symbol": symbol, "start": str(first.date()), "end": str(last.date()),
                              "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            pd.DataFrame(audit).to_csv(output / "coverage.csv", index=False)
            print(f"minute history {number}/{len(symbols)} stocks; {len(audit)} chunks; rows {sum(r['rows'] for r in audit)}", flush=True)
    (output / "complete.json").write_text(json.dumps({"stocks": len(symbols), "chunks": len(audit),
        "rows": sum(row["rows"] for row in audit), "empty_chunks": sum(row["rows"] == 0 for row in audit)}, indent=2))


if __name__ == "__main__":
    main()
