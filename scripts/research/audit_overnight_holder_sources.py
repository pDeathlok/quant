"""Reconcile annual holder events and incomplete duplicates without raw overwrite."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT

MATCH = ["ts_code", "ann_date", "holder_name", "holder_type", "change_vol", "change_ratio", "after_ratio", "avg_price"]


def main() -> None:
    root = PROJECT_ROOT / "data/raw"
    source = root / "holder_trade.parquet"
    parts = sorted((root / "holder_trade_history").glob("*.parquet"))
    if [p.stem for p in parts] != [str(y) for y in range(2013, 2027)]:
        raise ValueError("Unexpected annual partition coverage")
    output = PROJECT_ROOT / "reports/research/overnight_holder_reconciled_v1"
    output.mkdir(exist_ok=False)
    hashes = {str(p): sha256(p) for p in [source, *parts]}
    compact = pd.read_parquet(source)
    frames = []
    for path in parts:
        frame = pd.read_parquet(path)
        if not frame.ann_date.astype(str).str.startswith(path.stem).all():
            raise ValueError("Wrong year in partition")
        frame["partition"] = path.name
        frame["partition_row"] = range(len(frame))
        frames.append(frame)
    annual = pd.concat(frames, ignore_index=True)
    cols = sorted(compact.columns)
    if annual[cols].duplicated().any() or compact[cols].duplicated().any():
        raise ValueError("Unexpected exact duplicates")
    known = compact.loc[compact.in_de.notna(), cols]
    annual_values = annual[cols]
    # Row hashes only determine a stable comparison order; values are compared directly.
    def ordered(frame: pd.DataFrame) -> pd.DataFrame:
        order = pd.util.hash_pandas_object(frame, index=False).sort_values().index
        return frame.loc[order].reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(known), ordered(annual_values), check_dtype=False)
    if not annual.in_de.isin(["IN", "DE"]).all():
        raise ValueError("Unknown direction in annual records")
    incomplete = compact[compact.in_de.isna()].copy()
    incomplete["compact_row"] = incomplete.index
    matched = incomplete.merge(annual, on=MATCH, how="left", suffixes=("_compact", "_annual"), indicator=True)
    if len(matched) != len(incomplete) or not matched._merge.eq("both").all() or matched.compact_row.duplicated().any():
        raise ValueError("Incomplete records do not have unique exact annual counterparts")
    evidence = matched[["compact_row", "partition", "partition_row", "ts_code", "ann_date", "in_de_annual"]]
    evidence.to_parquet(output / "duplicate_evidence.parquet", index=False)
    annual.to_parquet(output / "events.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during reconciliation")
    summary = {"compact_rows": len(compact), "annual_rows": len(annual), "incomplete_duplicate_rows": len(incomplete),
               "match_columns": MATCH, "direction": annual.in_de.value_counts().to_dict(),
               "min_ann_date": str(annual.ann_date.min()), "max_ann_date": str(annual.ann_date.max()),
               "events_sha256": sha256(output / "events.parquet"), "script_sha256": sha256(Path(__file__)),
               "limits": "Local reconciliation only; not proof of provider completeness or point-in-time revisions. No production overwrite.",
               "timing_rule_for_features": "ann_date strictly before decision date; no same-day events without release timestamp"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
