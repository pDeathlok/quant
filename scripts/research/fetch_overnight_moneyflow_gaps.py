"""Read-only Tushare repair of audited moneyflow gaps into an isolated cache."""

from __future__ import annotations

import json
import os
import time

import httpx
import pandas as pd
from audit_overnight_moneyflow_coverage import AMOUNTS
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    source = PROJECT_ROOT / "reports/research/overnight_moneyflow_coverage_v1/key_availability.parquet"
    output = PROJECT_ROOT / "reports/research/overnight_moneyflow_gap_fill_v1"
    availability = pd.read_parquet(source)
    manifest = {"source_sha256": sha256(source), "purpose": "Fill audited missing days and SH/SZ missing stock rows",
                "provider": "moneyflow", "policy": "Empty provider results remain missing; no synthetic zero; no production overwrite"}
    output.mkdir(exist_ok=True)
    mp = output / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Resume source changed")
    mp.write_text(json.dumps(manifest, indent=2))
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        for line in (PROJECT_ROOT / ".env").read_text().splitlines():
            if line.strip().startswith("TUSHARE_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not token:
        raise RuntimeError("Tushare credential unavailable")
    fields = ["ts_code", "trade_date", *AMOUNTS, "net_mf_amount"]
    day_gaps = availability.loc[availability.status.eq("missing_file")]
    stock_gaps = availability.loc[availability.status.eq("missing_stock_row") & availability.exchange.isin(["SH", "SZ"])]
    tasks = [(date.strftime("%Y%m%d"), None) for date in sorted(day_gaps.flow_date.unique())]
    tasks += [(r.flow_date.strftime("%Y%m%d"), r.symbol)
              for r in stock_gaps[["flow_date", "symbol"]].drop_duplicates().sort_values(["flow_date", "symbol"]).itertuples(index=False)]
    records = []
    with httpx.Client(timeout=25) as client:
        def request(day: str, symbol: str | None) -> pd.DataFrame:
            params = {"trade_date": day}
            if symbol is not None:
                params["ts_code"] = symbol
            try:
                response = client.post("https://api.tushare.pro", json={"api_name": "moneyflow", "token": token,
                    "params": params, "fields": ",".join(fields)})
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError):
                raise RuntimeError("Tushare transport failed; credentials not logged") from None
            if body.get("code") != 0 or body.get("data") is None:
                raise RuntimeError(f"Tushare provider code {body.get('code')}; no quota-failure retry")
            raw = body["data"]
            frame = pd.DataFrame(raw["items"], columns=raw["fields"])
            time.sleep(.4)
            return frame

        for number, (day, symbol) in enumerate(tasks, 1):
            path = output / (f"day_{day}.parquet" if symbol is None else f"stock_{day}_{symbol}.parquet")
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                frame = request(day, symbol)
                if symbol is None and len(frame) >= 6000:
                    wanted = set(day_gaps.loc[day_gaps.flow_date.eq(pd.Timestamp(day)), "symbol"])
                    extra = [request(day, stock) for stock in sorted(wanted - set(frame.ts_code))]
                    frame = pd.concat([frame, *extra], ignore_index=True)
            if (not set(fields).issubset(frame.columns) or frame.ts_code.isna().any()
                    or frame.ts_code.duplicated().any() or not frame.trade_date.astype(str).eq(day).all()
                    or (symbol is not None and not frame.ts_code.eq(symbol).all())):
                raise ValueError(f"Unexpected provider keys/schema: {day}, {symbol}")
            if not path.exists():
                frame.to_parquet(path, index=False)
            records.append({"date": day, "symbol": symbol, "rows": len(frame), "path": str(path), "sha256": sha256(path)})
            pd.DataFrame(records).to_csv(output / "requests.csv", index=False)
            if number <= len(tasks) - len(stock_gaps) or number % 20 == 0 or number == len(tasks):
                print(f"Moneyflow repair {number}/{len(tasks)} requests; last rows={len(frame)}", flush=True)
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Audit source changed")
    summary = {"requests": len(records), "day_requests": sum(r["symbol"] is None for r in records),
               "single_requests": sum(r["symbol"] is not None for r in records),
               "single_returned_rows": sum(r["rows"] for r in records if r["symbol"] is not None),
               "empty_single_requests": sum(r["rows"] == 0 and r["symbol"] is not None for r in records)}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
