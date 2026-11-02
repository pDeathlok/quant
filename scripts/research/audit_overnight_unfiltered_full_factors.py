"""Audit all expanded factor keys and feature overlap without a whole-frame join."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    current_root = root / "overnight_unfiltered_full_factors_v1"
    old_root = root / "overnight_current_full_factors_v1"
    output = root / "overnight_unfiltered_full_factor_audit_v1"
    paths = [current_root / "dataset.parquet", old_root / "dataset.parquet"]
    hashes = {}
    for folder, path in zip([current_root, old_root], paths, strict=True):
        complete = json.loads((folder / "complete.json").read_text())
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != complete["dataset_sha256"]:
            raise ValueError(f"Factor artifact changed: {path}")
    output.mkdir(exist_ok=False)
    manifest = json.loads((current_root / "manifest.json").read_text())
    universe_path = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    hashes[str(universe_path)] = sha256(universe_path)
    if hashes[str(universe_path)] != manifest["candidate_sha256"]:
        raise ValueError("Candidate universe changed")
    keys = ["symbol", "date"]
    columns = list(PROJECT_FACTOR_COLUMNS)
    expected = pd.MultiIndex.from_frame(pd.read_parquet(universe_path, columns=keys)).sort_values()
    old = pd.read_parquet(paths[1], columns=[*keys, *columns]).set_index(keys)
    if not old.index.is_unique:
        raise ValueError("Duplicate original keys")
    differences = np.zeros(len(columns), dtype=np.int64)
    missing_changes = np.zeros(len(columns), dtype=np.int64)
    seen = []
    common_rows = 0
    for batch in pq.ParquetFile(paths[0]).iter_batches(batch_size=65536, columns=[*keys, *columns]):
        frame = batch.to_pandas().set_index(keys)
        seen.append(frame.index)
        common = frame.index.intersection(old.index)
        common_rows += len(common)
        left = frame.loc[common, columns].to_numpy()
        right = old.loc[common, columns].to_numpy()
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        differences += (~same).sum(axis=0)
        missing_changes += (np.isnan(left) != np.isnan(right)).sum(axis=0)
    actual = seen[0].append(seen[1:])
    if not actual.is_unique or not actual.sort_values().equals(expected):
        raise ValueError("Full expanded key mismatch")
    pd.DataFrame({"feature": columns, "different_rows": differences,
                  "missingness_changes": missing_changes}).to_csv(output / "comparison.csv", index=False)
    source_hashes = {**hashes, **manifest["daily_basic_files"],
                     **manifest["market_manifest"]["daily_files"],
                     **manifest["market_manifest"]["formula_files"]}
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during audit: {path}")
    summary = {"rows": len(actual), "common_rows": common_rows, "features": len(columns),
               "different_cells": int(differences.sum()), "missingness_changes": int(missing_changes.sum()),
               "script_sha256": sha256(Path(__file__)),
               "scope": "Exact keys and old-pool feature values; not a profitability or point-in-time vendor-data proof"}
    (output / "sources.json").write_text(json.dumps(source_hashes, indent=2))
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["different_cells"] or len(columns) != 145:
        raise ValueError("Full-factor overlap differs; investigate before training")
    (output / "complete.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
