"""Cache Tushare historical cb_over_rate with 2,000-row pages by year."""
from __future__ import annotations

import time
import calendar
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/public_strategy_validation_20260929/material_backtests/source"


def main() -> None:
    token = dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN missing")
    pro = ts.pro_api(token)
    OUT.mkdir(parents=True, exist_ok=True)
    for year in range(2018, 2027):
        path = OUT / f"cb_over_rate_{year}.parquet"
        if path.exists():
            print(f"{year}: cached", flush=True)
            continue
        pages = []
        for quarter in range(4):
            month_start = quarter * 3 + 1
            month_end = month_start + 2
            end_day = calendar.monthrange(year, month_end)[1]
            start, end = f"{year}{month_start:02d}01", min(f"{year}{month_end:02d}{end_day:02d}", "20260616")
            if start > end:
                continue
            offset = 0
            while True:
                for attempt in range(5):
                    try:
                        page = pro.cb_daily(start_date=start, end_date=end,
                                            fields="ts_code,trade_date,cb_over_rate",
                                            offset=offset, limit=2000)
                        break
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(2 ** attempt)
                pages.append(page)
                offset += len(page)
                if len(page) < 2000:
                    break
                time.sleep(0.12)
        frame = pd.concat(pages, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
        frame.to_parquet(path, index=False)
        print(f"{year}: {len(frame)} rows, {len(pages)} pages", flush=True)


if __name__ == "__main__":
    main()
