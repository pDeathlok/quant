"""Actual-data prefix and stored-input parity for35 daily-basic factors."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import (
    DAILY_BASIC_PROJECT_FACTOR_COLUMNS,
    derive_daily_basic_features,
)


def different_cells(a: pd.DataFrame, b: pd.DataFrame, columns: list[str]) -> int:
    x = a[columns].astype("float32").to_numpy()
    y = b[columns].astype("float32").to_numpy()
    return int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())


def main() -> None:
    root = PROJECT_ROOT
    symbols = ["000001.SZ", "300750.SZ", "600519.SH"]
    cuts = ["20240614", "20250314", "20250903", "20260529"]
    output = root / "reports/research/overnight_basic_prefix_audit_v1"
    source = root / "reports/research/overnight_unfiltered_full_factors_v1/dataset.parquet"
    paths = [p for p in sorted((root / "data/raw/daily_basic").glob("*.parquet")) if "20231101" <= p.stem <= "20260904"]
    hashes = {str(p): sha256(p) for p in [*paths, source, root / "src/quant/features/variable_library.py"]}
    output.mkdir(exist_ok=False)
    records = []
    for path in paths:
        part = pd.read_parquet(path, filters=[("ts_code", "in", symbols)])
        if part.ts_code.duplicated().any():
            raise ValueError("Duplicate source keys")
        dates = pd.to_datetime(part.trade_date.astype(str), format="mixed").dt.strftime("%Y%m%d")
        if not dates.eq(path.stem).all():
            raise ValueError("Unexpected source date")
        part["trade_date"] = dates
        records.append(part)
    history = pd.concat(records, ignore_index=True)
    if history.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate history")
    columns = list(DAILY_BASIC_PROJECT_FACTOR_COLUMNS)
    full = derive_daily_basic_features(history)
    comparisons = []
    for symbol in symbols:
        stock = history.loc[history.ts_code.eq(symbol)]
        for cut in cuts:
            left = full.loc[full.ts_code.eq(symbol) & full.trade_date.le(cut)].reset_index(drop=True)
            right = derive_daily_basic_features(stock.loc[stock.trade_date.le(cut)]).reset_index(drop=True)
            pd.testing.assert_frame_equal(left[["ts_code", "trade_date"]], right[["ts_code", "trade_date"]])
            comparisons.append({"symbol": symbol, "cutoff": cut, "rows": len(left), "features": len(columns),
                                "different_cells": different_cells(left, right, columns)})
    stored = pd.read_parquet(source, columns=["symbol", "date", *columns], filters=[("symbol", "in", symbols)])
    replay = full.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    replay["date"] = pd.to_datetime(replay.date)
    joined = stored.merge(replay[["symbol", "date", *columns]], on=["symbol", "date"], how="left", validate="one_to_one", suffixes=("", "_new"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Stored factor key missing from source")
    saved_differences = different_cells(joined, joined[[f"{c}_new" for c in columns]].rename(columns={f"{c}_new": c for c in columns}), columns)
    pd.DataFrame(comparisons).to_csv(output / "comparisons.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Changed source: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"cases": len(comparisons), "prefix_cells": sum(r["rows"] * r["features"] for r in comparisons),
               "prefix_differences": sum(r["different_cells"] for r in comparisons), "stored_rows": len(stored),
               "stored_differences": saved_differences, "script_sha256": sha256(Path(__file__)),
               "scope": "35 formula outputs on3 fixed stocks; does not audit historical vendor revisions or publication availability"}
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    if summary["prefix_differences"] or saved_differences:
        raise ValueError("Daily-basic prefix or stored-input mismatch")
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
