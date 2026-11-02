"""Actual-data prefix invariance for the pinned overnight market-factor schema."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256
from build_overnight_current_market_factors import SCHEMA

from quant.core.paths import PROJECT_ROOT
from quant.data.source_merge import normalize_tushare_daily
from quant.features.project_factor_layer import calculate_project_market_factors
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def main() -> None:
    root = PROJECT_ROOT
    output = root / "reports/research/overnight_market_factor_prefix_audit_v1"
    symbols = ["000001.SZ", "300750.SZ", "600519.SH"]
    cutoffs = ["2024-06-14", "2025-03-14", "2025-09-03", "2026-05-29"]
    paths = [p for p in sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "201801" <= p.parent.name.split("=")[1] <= "202609"]
    formula_paths = [root / "src/quant/features/project_factor_layer.py", root / "src/quant/features/variable_library.py",
                     root / "src/quant/data/source_merge.py"]
    hashes = {str(p): sha256(p) for p in [*paths, *formula_paths]}
    output.mkdir(exist_ok=False)
    (output / "protocol.json").write_text(json.dumps({"symbols": symbols, "cutoffs": cutoffs, "schema": SCHEMA,
        "test": "Compare every past factor cell from full history against history truncated at each cutoff; no tolerance after float32 storage conversion",
        "sources": hashes}, indent=2))
    raw = ds.dataset([str(p) for p in paths], format="parquet").to_table(
        columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume", "amount", "pct_chg", "change"],
        filter=ds.field("ts_code").isin(symbols) & (ds.field("trade_date") >= "20180101") & (ds.field("trade_date") <= "20260904")).to_pandas()
    records, differences = [], []
    for symbol in symbols:
        history = normalize_tushare_daily(raw.loc[raw.ts_code.eq(symbol)].copy(), symbol).sort_values("date").reset_index(drop=True)
        full = calculate_project_market_factors(history, symbol, factor_schema_version=SCHEMA)
        columns = [c for c in PROJECT_FACTOR_COLUMNS if c in full]
        if len(columns) != 110:
            raise ValueError("Unexpected market-factor schema")
        for cutoff in cutoffs:
            prefix = history.loc[history.date.le(cutoff)].copy()
            truncated = calculate_project_market_factors(prefix, symbol, factor_schema_version=SCHEMA)
            left = full.loc[full.date.le(cutoff)].reset_index(drop=True)
            right = truncated.reset_index(drop=True)
            pd.testing.assert_frame_equal(left[["symbol", "date"]], right[["symbol", "date"]])
            a = left[columns].astype("float32").replace([np.inf, -np.inf], np.nan).to_numpy()
            b = right[columns].astype("float32").replace([np.inf, -np.inf], np.nan).to_numpy()
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            records.append({"symbol": symbol, "cutoff": cutoff, "rows": len(left), "columns": len(columns),
                            "different_cells": int((~same).sum())})
            for column_idx, column in enumerate(columns):
                bad = np.flatnonzero(~same[:, column_idx])
                for row_idx in bad[:10]:
                    differences.append({"symbol": symbol, "cutoff": cutoff, "date": str(left.date.iloc[row_idx]),
                                        "feature": column, "full": float(a[row_idx, column_idx]), "prefix": float(b[row_idx, column_idx])})
            print(json.dumps(records[-1]), flush=True)
    pd.DataFrame(records).to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(differences, columns=["symbol", "cutoff", "date", "feature", "full", "prefix"]).to_csv(output / "difference_examples.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    summary = {"cases": len(records), "compared_cells": sum(r["rows"] * r["columns"] for r in records),
               "different_cells": sum(r["different_cells"] for r in records), "script_sha256": sha256(Path(__file__)),
               "scope": "Pinned110 market factors, three fixed stocks and four truncations; not proof for every stock, vendor revisions, other77 inputs, or14:30 fills"}
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    if summary["different_cells"]:
        raise ValueError("Prefix mismatch; investigate before further training claims")
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
