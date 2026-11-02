"""Verify expanded market-factor keys and exact overlap with the original pool."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    expanded = root / "overnight_unfiltered_market_factors_v1"
    original = root / "overnight_current_market_factors_v1"
    if not (expanded / "market_complete.json").exists():
        raise ValueError("Expanded market build is not complete")
    output = root / "overnight_unfiltered_market_factor_audit_v1"
    output.mkdir(exist_ok=False)
    manifest = json.loads((expanded / "manifest.json").read_text())
    old_manifest = json.loads((original / "manifest.json").read_text())
    for field in ["schema", "history_start", "history_end", "daily_files"]:
        if manifest[field] != old_manifest[field]:
            raise ValueError(f"Original and expanded source configuration differs: {field}")
    source = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    if sha256(source) != manifest["candidate_sha256"]:
        raise ValueError("Expanded keys changed")
    keys = pd.read_parquet(source, columns=["symbol", "date"])
    coverage = pd.read_csv(expanded / "coverage.csv").set_index("symbol")
    old_coverage = pd.read_csv(original / "coverage.csv").set_index("symbol")
    if not coverage.index.is_unique or set(coverage.index) != set(keys.symbol):
        raise ValueError("Expanded symbol coverage mismatch")
    records = []
    for symbol, target in keys.groupby("symbol", sort=True):
        path = expanded / "by_symbol" / f"{symbol}.parquet"
        if sha256(path) != coverage.loc[symbol, "sha256"]:
            raise ValueError(f"Expanded shard changed: {symbol}")
        frame = pd.read_parquet(path).set_index(["symbol", "date"]).sort_index()
        expected = pd.MultiIndex.from_frame(target).sort_values()
        if not frame.index.is_unique or not frame.index.equals(expected):
            raise ValueError(f"Expanded key mismatch: {symbol}")
        cols = [c for c in PROJECT_FACTOR_COLUMNS if c in frame]
        common_count = 0
        different_rows = 0
        if symbol in old_coverage.index:
            old_path = original / "by_symbol" / f"{symbol}.parquet"
            if sha256(old_path) != old_coverage.loc[symbol, "sha256"]:
                raise ValueError(f"Original shard changed: {symbol}")
            old = pd.read_parquet(old_path).set_index(["symbol", "date"]).sort_index()
            if not old.index.is_unique or cols != [c for c in PROJECT_FACTOR_COLUMNS if c in old]:
                raise ValueError(f"Original keys/schema invalid: {symbol}")
            common = frame.index.intersection(old.index)
            common_count = len(common)
            left, right = frame.loc[common, cols].to_numpy(), old.loc[common, cols].to_numpy()
            same = (left == right) | (np.isnan(left) & np.isnan(right))
            different_rows = int((~same.all(axis=1)).sum())
        records.append({"symbol": symbol, "rows": len(frame), "features": len(cols),
                        "common_rows": common_count, "different_rows": different_rows})
    result = pd.DataFrame(records)
    result.to_csv(output / "comparison.csv", index=False)
    for path, digest in {**manifest["daily_files"], **manifest["formula_files"]}.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    summary = {"rows": int(result.rows.sum()), "symbols": len(result),
               "common_rows": int(result.common_rows.sum()),
               "different_rows": int(result.different_rows.sum()),
               "feature_counts": sorted(result.features.unique().tolist()),
               "script_sha256": sha256(Path(__file__)),
               "limitation": "Exact overlap verifies method consistency, not point-in-time vendor revisions or profitability"}
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["different_rows"] or summary["feature_counts"] != [110]:
        raise ValueError("Expanded market factors differ; investigate before training")
    (output / "complete.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
