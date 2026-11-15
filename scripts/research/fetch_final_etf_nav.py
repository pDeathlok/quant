"""Read isolated NAV history for tradability-aware final ETF research."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/final_etf_strategy_20261111/source"
CODES = ("510300.SH", "510880.SH", "511010.SH", "511260.SH", "518880.SH",
         "159985.SZ", "159981.SZ", "159980.SZ")


def fetch_one(code: str, token: str) -> dict:
    path = OUT / f"{code.replace('.', '_')}_nav.parquet"
    if not path.exists():
        parts = []
        with requests.Session() as session:
            for year in range(2018, 2027):
                try:
                    response = session.post("https://api.tushare.pro", json={"api_name": "fund_nav", "token": token,
                        "params": {"ts_code": code, "market": "E", "start_date": f"{year}0101",
                                   "end_date": min(f"{year}1231", "20261110")},
                        "fields": "ts_code,ann_date,nav_date,unit_nav,adj_nav"}, timeout=40)
                    response.raise_for_status(); payload = response.json()
                except Exception as error:
                    raise RuntimeError(f"{code} fund_nav: {type(error).__name__}") from None
                if payload.get("code") != 0:
                    raise RuntimeError(str(payload.get("msg", "")).replace(token, "[redacted]")[:200])
                f = pd.DataFrame(payload["data"]["items"], columns=payload["data"]["fields"])
                if len(f) >= 1000:
                    raise RuntimeError("Potential response truncation")
                parts.append(f)
        f = pd.concat(parts, ignore_index=True).drop_duplicates().sort_values(["nav_date", "ann_date"])
        if f.empty or (pd.to_numeric(f.unit_nav, errors="coerce") <= 0).any():
            raise ValueError(f"Invalid NAV history: {code}")
        f.to_parquet(path, index=False)
    f = pd.read_parquet(path)
    result = {"code": code, "rows": len(f), "last": f.tail(1).to_dict("records")}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    token = dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Tushare token missing")
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda code: fetch_one(code, token), CODES))
    (OUT/"nav_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
