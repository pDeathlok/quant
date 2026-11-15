"""Download unadjusted CFFEX contract bars and their cash index counterparts."""
import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

INDICES = {"IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH", "IM": "000852.SH"}
FIRST_YEAR = {"IF": 2010, "IH": 2015, "IC": 2015, "IM": 2022}
ROOT = Path("data/research/index_futures")


def save(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def fetch(end):
    load_dotenv()
    session = requests.Session()

    def query(api, **params):
        for attempt in range(4):
            try:
                response = session.post(
                    "https://api.tushare.pro",
                    json={"api_name": api, "token": os.environ["TUSHARE_TOKEN"], "params": params},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                if payload["code"] == 0:
                    data = payload["data"]
                    return pd.DataFrame(data["items"], columns=data["fields"])
            except (ValueError, requests.RequestException):
                if attempt == 3:
                    raise RuntimeError(f"{api} request failed") from None
            time.sleep(2 ** attempt)
        raise RuntimeError(f"{api} rejected request: {payload.get('msg')}")

    ROOT.mkdir(parents=True, exist_ok=True)
    for product, index in INDICES.items():
        contracts = query("fut_basic", exchange="CFFEX", fut_type="1", fut_code=product)
        contracts = contracts[contracts.ts_code.str.fullmatch(rf"{product}\d{{4}}\.CFX")]
        save(contracts, ROOT / f"{product}_contracts.parquet")
        bars = []
        for row in contracts.itertuples():
            if str(row.list_date) > end:
                continue
            path = ROOT / f"{row.ts_code}_{min(str(row.delist_date), end)}.parquet"
            if not path.exists():
                daily = query("fut_daily", ts_code=row.ts_code, end_date=end)
                if daily.empty or len(daily) >= 2000:
                    raise RuntimeError(f"Unexpected coverage for {row.ts_code}")
                save(daily, path)
                time.sleep(0.32)
            bars.append(pd.read_parquet(path))
        daily = pd.concat(bars).sort_values(["trade_date", "ts_code"])
        if daily.duplicated(["trade_date", "ts_code"]).any():
            raise RuntimeError(f"Duplicate futures bars for {product}")
        save(daily, ROOT / f"{product}_daily.parquet")
        cash = []
        for year in range(FIRST_YEAR[product] - 1, int(end[:4]) + 1):
            cash.append(query("index_daily", ts_code=index, start_date=f"{year}0101",
                              end_date=min(f"{year}1231", end)))
            time.sleep(0.32)
        spot = pd.concat(cash).sort_values("trade_date")
        save(spot, ROOT / f"{product}_spot.parquet")
        print(product, len(contracts), len(daily), daily.trade_date.min(), daily.trade_date.max(),
              len(spot), spot.trade_date.max(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}", args.end):
        parser.error("--end must be YYYYMMDD")
    fetch(args.end)
