"""Observed daily universe without strategy membership or current-list survivorship filtering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from overnight_market_context import ASHARE

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    output = PROJECT_ROOT / "reports/research/overnight_unfiltered_universe_v1"
    output.mkdir(exist_ok=False)
    paths = [p for p in sorted((PROJECT_ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "202401" <= p.parent.name.split("=")[1] <= "202609"]
    hashes = {str(p): sha256(p) for p in paths}
    frames, excluded = [], []
    for path in paths:
        raw = pd.read_parquet(path, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume", "amount"])
        raw["date"] = pd.to_datetime(raw.trade_date.astype(str), format="%Y%m%d", errors="raise")
        raw = raw[raw.date.between("2024-01-01", "2026-09-04")].copy()
        if raw.ts_code.isna().any():
            raise ValueError("Null security identity")
        # Verified code change; keep it even though the older shared regex omits 302xxx.
        keep = raw.ts_code.str.fullmatch(ASHARE) | raw.ts_code.eq("302132.SZ")
        excluded.append(raw.loc[~keep, ["ts_code", "date"]])
        raw = raw.loc[keep].copy()
        raw["vol"] = raw.vol.fillna(raw.volume)
        prices = raw[["open", "high", "low", "close", "pre_close", "vol"]].apply(pd.to_numeric, errors="coerce")
        valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        valid &= raw.high.ge(raw[["open", "close"]].max(axis=1) - 1e-8)
        valid &= raw.low.le(raw[["open", "close"]].min(axis=1) + 1e-8)
        raw["valid_bar"] = valid
        raw["symbol"] = raw.ts_code
        frames.append(raw[["symbol", "date", "open", "high", "low", "close", "pre_close", "vol", "amount", "valid_bar"]])
    frame = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate observed security/date")
    daily = frame.groupby("date").agg(observed=("symbol", "size"), valid=("valid_bar", "sum")).reset_index()
    if daily.valid.lt(1000).any():
        raise ValueError("Insufficient market coverage; do not silently drop partial dates")
    old_path = PROJECT_ROOT / "reports/research/tail_overnight/dataset.parquet"
    hashes[str(old_path)] = sha256(old_path)
    old = pd.read_parquet(old_path, columns=["symbol", "date"])
    compared = frame[["symbol", "date", "valid_bar"]].merge(old.assign(old_candidate=True), on=["symbol", "date"], how="outer", validate="one_to_one", indicator=True)
    coverage = compared.groupby([compared.date.dt.to_period("Q").astype(str), "_merge"], observed=True).size().reset_index(name="rows")
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during inventory")
    frame.to_parquet(output / "daily_bars.parquet", index=False)
    daily.to_csv(output / "daily_coverage.csv", index=False)
    coverage.to_csv(output / "old_pool_overlap.csv", index=False)
    pd.concat(excluded, ignore_index=True).to_parquet(output / "excluded_nonmatching_keys.parquet", index=False)
    compared.loc[compared._merge.eq("right_only"), ["symbol", "date"]].to_parquet(output / "old_only_keys.parquet", index=False)
    summary = {"rows": len(frame), "symbols": int(frame.symbol.nunique()), "days": len(daily),
               "valid_rows": int(frame.valid_bar.sum()), "invalid_rows_retained": int((~frame.valid_bar).sum()),
               "special_302132_rows": int(frame.symbol.eq("302132.SZ").sum()),
               "old_only_keys": int(compared._merge.eq("right_only").sum()),
               "dataset_sha256": sha256(output / "daily_bars.parquet"), "script_sha256": sha256(Path(__file__)),
               "scope": "All locally observed matching A-share daily keys, 2024-01-01 to2026-09-04, not proof of exhaustive exchange history",
               "policy": "No strategy membership, present-day listing filter, forward outcome or current watchlist filter; invalid bars retained and flagged",
               "next": "Audit uncovered keys and security scope; build causal training and validation features plus exact next-market-day labels"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
