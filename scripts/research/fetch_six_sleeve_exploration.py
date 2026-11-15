"""Download isolated Tushare ETF inputs for the six-direction feasibility study."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/six_sleeve_exploration_20261111/source"
CODES = ("518880.SH", "513500.SH", "159920.SZ", "511380.SH", "510880.SH",
         "512100.SH", "510300.SH", "510500.SH", "159915.SZ", "511010.SH")
END = "20261110"


def fetch_one(code: str, token: str) -> dict:
    destination = OUT / f"{code.replace('.', '_')}.parquet"
    if not destination.exists():
        tables = {}
        with requests.Session() as client:
            for api, fields in [("fund_daily", "ts_code,trade_date,open,high,low,close,vol,amount"),
                                ("fund_adj", "ts_code,trade_date,adj_factor")]:
                parts = []
                for start, end in [("20180101", "20221231"), ("20230101", END)]:
                    try:
                        response = client.post("https://api.tushare.pro", json={
                            "api_name": api, "token": token,
                            "params": {"ts_code": code, "start_date": start, "end_date": end},
                            "fields": fields}, timeout=40)
                        response.raise_for_status()
                        payload = response.json()
                    except Exception as error:
                        raise RuntimeError(f"{code} {api}: {type(error).__name__}") from None
                    if payload.get("code") != 0:
                        message = str(payload.get("msg", "")).replace(token, "[redacted]")
                        raise RuntimeError(f"{code} {api}: {message[:180]}")
                    table = pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
                    if len(table) >= 2000:
                        raise RuntimeError("Potential API truncation")
                    parts.append(table)
                tables[api] = pd.concat(parts, ignore_index=True)
        frame = tables["fund_daily"].merge(tables["fund_adj"], on=["ts_code", "trade_date"],
                                           how="left", validate="one_to_one").sort_values("trade_date")
        if frame.empty or frame.adj_factor.isna().any() or (frame.close <= 0).any():
            raise RuntimeError(f"Incomplete prices or adjustment factors for {code}")
        frame.to_parquet(destination, index=False)
    frame = pd.read_parquet(destination)
    result = {"code": code, "start": str(frame.trade_date.min()), "end": str(frame.trade_date.max()),
              "rows": len(frame), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "source": "Tushare fund_daily + fund_adj", "requested_end": END}
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    token = dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Configured Tushare token missing")
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda code: fetch_one(code, token), CODES))
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
