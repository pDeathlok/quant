"""Extend seven cached Tushare ETF series; preserve old raw prices/factor scale."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/strategy_lab_20261001_ten_year"
CODES = ("510300.SH", "510500.SH", "159915.SZ", "159920.SZ", "513500.SH", "518880.SH", "511010.SH")


def existing_path(code):
    return (ROOT / "reports/public_strategy_validation_20260929/source" / f"{code.replace('.', '_')}_2018_2026.parquet"
            if code in CODES[:3] else ROOT / "reports/strategy_lab_20260930/source" / f"{code.replace('.', '_')}.parquet")


def read_token():
    for line in (ROOT / ".env").read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key == "TUSHARE_TOKEN":
            return value.strip().strip("\"'")
    raise RuntimeError("TUSHARE_TOKEN missing")


def query(token, api, params, fields=""):
    try:
        response = requests.post("https://api.tushare.pro", json={"api_name": api, "token": token,
                                 "params": params, "fields": fields}, timeout=40)
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        raise RuntimeError(f"Tushare {api}: {type(exc).__name__}; request failed") from None
    if result.get("code") != 0:
        message = str(result.get("msg", "")).replace(token, "[redacted]")[:180]
        raise RuntimeError(f"Tushare {api}: code={result.get('code')} {message}")
    data = result["data"]
    return pd.DataFrame(data["items"], columns=data["fields"])


def main():
    token = read_token()
    source = OUT / "source"
    raw = source / "downloaded"
    raw.mkdir(parents=True, exist_ok=True)
    audit = []
    for code in CODES:
        path = source / f"{code.replace('.', '_')}.parquet"
        if path.exists():
            print(f"{code}: completed cache", flush=True)
            continue
        old = pd.read_parquet(existing_path(code)).sort_values("trade_date")
        old["trade_date"] = old.trade_date.astype(str)
        # Fetch an overlap to reconcile adjustment-factor conventions explicitly.
        overlap_end = (pd.Timestamp(old.trade_date.min()) + pd.Timedelta(days=45)).strftime("%Y%m%d")
        frames = []
        for year in range(2015, int(overlap_end[:4])+1):
            params = {"ts_code": code, "start_date": f"{year}0101", "end_date": min(f"{year}1231", overlap_end)}
            parts = {}
            for api, fields in (("fund_daily", "ts_code,trade_date,close,amount"), ("fund_adj", "ts_code,trade_date,adj_factor")):
                cache = raw / f"{code.replace('.', '_')}_{year}_{api}.parquet"
                if cache.exists():
                    parts[api] = pd.read_parquet(cache)
                else:
                    parts[api] = query(token, api, params, fields)
                    if parts[api].empty:
                        raise RuntimeError(f"{code} {year} {api}: no observations")
                    parts[api].to_parquet(cache, index=False)
                print(f"{code} {year} {api}: {len(parts[api])} rows", flush=True)
            frame = parts["fund_daily"].merge(parts["fund_adj"], on=["ts_code", "trade_date"], how="left", validate="one_to_one")
            assert frame.adj_factor.notna().all()
            frames.append(frame)
        fetched = pd.concat(frames, ignore_index=True).sort_values("trade_date")
        overlap = fetched.merge(old, on=["ts_code", "trade_date"], suffixes=("_new", "_old"))
        assert len(overlap) >= 10
        assert np.allclose(overlap.close_new, overlap.close_old, atol=1e-8, rtol=1e-8)
        ratio = overlap.adj_factor_new / overlap.adj_factor_old
        scale = float(ratio.median())
        assert np.allclose(ratio, scale, atol=1e-6, rtol=1e-6), f"{code}: adjustment-factor overlap mismatch"
        fetched["adj_factor"] /= scale
        before = fetched[fetched.trade_date < old.trade_date.min()]
        combined = pd.concat([before, old], ignore_index=True).sort_values("trade_date")
        assert not combined.duplicated(["ts_code", "trade_date"]).any()
        assert combined.trade_date.min() <= "20150115"
        assert (combined.close > 0).all() and (combined.adj_factor > 0).all()
        combined.to_parquet(path, index=False)
        audit.append({"code": code, "local_start": old.trade_date.min(), "extended_start": combined.trade_date.min(),
                      "end": combined.trade_date.max(), "new_rows": len(before), "overlap_rows": len(overlap),
                      "factor_new_to_old_scale": scale, "old_observations_preserved": True})
    # Save trading calendar to quantify missing observations, not silently call every row a day.
    calendar_path = source / "trade_calendar.parquet"
    if not calendar_path.exists():
        query(token, "trade_cal", {"exchange": "SSE", "start_date": "20150101", "end_date": "20260924"},
              "exchange,cal_date,is_open").to_parquet(calendar_path, index=False)
    # Reconstruct all coverage records even after an interrupted fetch resumes.
    for code in CODES:
        if not any(a["code"] == code for a in audit):
            old = pd.read_parquet(existing_path(code)); frame = pd.read_parquet(source/f"{code.replace('.', '_')}.parquet")
            audit.append({"code": code, "local_start": str(old.trade_date.min()), "extended_start": str(frame.trade_date.min()),
                          "end": str(frame.trade_date.max()), "new_rows": len(frame)-len(old), "old_observations_preserved": True})
    (source / "coverage.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print("Historical extension complete; credentials omitted from artifacts.", flush=True)


if __name__ == "__main__":
    main()
