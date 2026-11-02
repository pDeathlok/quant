"""One bounded Tushare request to investigate minute/daily close disagreements."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="store_true", help="One bounded index-minute permission probe")
    args = parser.parse_args()
    output = ROOT / ("reports/research/overnight_index_minute_probe_v1" if args.index else
                     "reports/research/overnight_minute_source_probe_v1")
    output.mkdir(exist_ok=False)
    api_name = "idx_mins" if args.index else "stk_mins"
    params = {"ts_code": "000001.SH" if args.index else "600519.SH", "freq": "5min",
              "start_date": "2025-07-01 00:00:00", "end_date": "2025-07-04 23:59:59" if args.index else "2025-12-31 23:59:59"}
    (output / "request.json").write_text(json.dumps({"api_name": api_name, "params": params,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "one index-minute permission/quality probe, no automatic retries" if args.index else
                   "one quality probe after historical hourly restriction, no automatic retries"}, indent=2))
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    with httpx.Client(timeout=30) as client:
        response = client.post("https://api.tushare.pro", json={"api_name": api_name, "token": token,
            "params": params, "fields": "ts_code,trade_time,open,high,low,close,vol,amount"})
    if response.status_code != 200:
        raise RuntimeError(f"Provider HTTP {response.status_code}")
    body = response.json()
    status = {"code": body.get("code"), "message": str(body.get("msg", "")).replace(token, "[REDACTED]")[:250]}
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2))
    if body.get("code") != 0 or body.get("data") is None:
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return
    data = body["data"]
    frame = pd.DataFrame(data["items"], columns=data["fields"])
    if frame.empty or len(frame) >= 8000:
        raise ValueError("Empty or potentially truncated response")
    frame["trade_time"] = pd.to_datetime(frame.trade_time)
    if (not frame.ts_code.eq(params["ts_code"]).all() or frame.trade_time.duplicated().any()
            or not frame.trade_time.between(params["start_date"], params["end_date"]).all()):
        raise ValueError("Response keys outside requested scope")
    frame.sort_values("trade_time").to_parquet(output / "minutes.parquet", index=False)
    print(f"source quality probe returned {len(frame)} bars", flush=True)


if __name__ == "__main__":
    main()
