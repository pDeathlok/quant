"""Retry only the two dates absent from old caches, without fabricating quotations."""
import json
import pandas as pd
from fetch_strategy_ten_year import OUT, query, read_token


def main():
    source = OUT / "source"
    rows = json.loads((source / "missing_observations.json").read_text())
    token = read_token()
    for row in rows:
        code, day = row["code"], row["date"]
        params = {"ts_code": code, "start_date": day, "end_date": day}
        daily = query(token, "fund_daily", params, "ts_code,trade_date,close,amount")
        adj = query(token, "fund_adj", params, "ts_code,trade_date,adj_factor")
        daily.to_parquet(source / "downloaded" / f"{code.replace('.', '_')}_{day}_fund_daily.parquet", index=False)
        adj.to_parquet(source / "downloaded" / f"{code.replace('.', '_')}_{day}_fund_adj.parquet", index=False)
        row.update(daily_rows=len(daily), adjustment_rows=len(adj))
        if len(daily) and len(adj):
            frame = daily.merge(adj, on=["ts_code", "trade_date"], validate="one_to_one")
            path = source / f"{code.replace('.', '_')}.parquet"
            old = pd.read_parquet(path)
            pd.concat([old, frame], ignore_index=True).sort_values("trade_date").to_parquet(path, index=False)
            row["status"] = "recovered_from_Tushare"
        else:
            row["status"] = "Tushare_no_daily_quotation"
        print(code, day, row["status"], flush=True)
    (source / "missing_observations.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
