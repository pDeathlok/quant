"""Expand the original thirteen exact-T-1 statistics without changing formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from run_tail_overnight import raw_lag_features

from quant.core.paths import PROJECT_ROOT
from quant.research.tail_overnight import attach_lagged_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-extension", action="store_true")
    args = parser.parse_args()
    root = PROJECT_ROOT
    output = root / "reports/research" / (
        "overnight_event_extension_lag_statistics_v1" if args.event_extension
        else "overnight_unfiltered_lag_statistics_v1")
    output.mkdir(exist_ok=False)
    source = root / "reports/research/overnight_unfiltered_universe_v1/daily_bars.parquet"
    old_path = root / "reports/research/tail_overnight/dataset.parquet"
    extra_paths = []
    history_end = "2026-09-04"
    if args.event_extension:
        pool = root / "reports/research/overnight_event_extension_pool_v1"
        source = pool / "entry_pool.parquet"
        boundary_path = pool / "boundary_comparison.parquet"
        proof_path = pool / "complete.json"
        proof = json.loads(proof_path.read_text())
        if sha256(source) != proof["dataset_sha256"]:
            raise ValueError("Extension pool changed")
        old_path = root / "reports/research/overnight_unfiltered_lag_statistics_v1/dataset.parquet"
        reference_proof = old_path.parent / "result.json"
        if sha256(old_path) != json.loads(reference_proof.read_text())["dataset_sha256"]:
            raise ValueError("Refreshed lag reference changed")
        extra_paths = [boundary_path, proof_path, reference_proof]
        history_end = "2026-09-10"
    paths = [p for p in sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "202309" <= p.parent.name.split("=")[1] <= "202609"]
    hashes = {str(p): sha256(p) for p in [source, old_path, *extra_paths, *paths,
        root / "scripts/research/run_tail_overnight.py", root / "src/quant/research/tail_overnight.py"]}
    keys = pd.read_parquet(source, columns=["symbol", "date"])
    if args.event_extension:
        boundary = pd.read_parquet(boundary_path)
        if not boundary["_merge"].eq("both").all() or len(boundary) != proof["boundary_rows"]:
            raise ValueError("Invalid pool boundary")
        keys = pd.concat([boundary[["symbol", "date"]], keys], ignore_index=True)
    if keys.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate candidate keys")
    # Match the original materialize default start minus 120 calendar days.
    warmup = pd.Timestamp("2024-01-01") - pd.Timedelta(days=120)
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["symbol", "date", "open", "high", "low", "close", "pre_close", "amount"])
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
        frames.append(frame.loc[frame.date.between(warmup, history_end)])
    daily = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    del frames
    if daily.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate raw keys")
    counts = daily.groupby("date").symbol.nunique()
    if counts.lt(1000).any():
        raise ValueError("Incomplete market session")
    calendar = sorted(daily.date.unique())
    observed_keys = daily[["symbol", "date"]].copy() if args.event_extension else None
    daily["amount_yuan"] = daily.amount * 1000
    print(f"lag statistics history: {len(daily)} rows", flush=True)
    statistics = raw_lag_features(daily)
    del daily
    columns = [c for c in statistics if c.startswith("daily_")]
    if len(columns) != 13:
        raise ValueError("Original statistic schema changed")
    result = attach_lagged_features(keys, statistics, calendar, columns)
    del statistics
    if not result.feature_date.lt(result.date).all():
        raise ValueError("Noncausal feature date")
    pd.testing.assert_frame_equal(result[["symbol", "date"]], keys)
    features = [f"lag1_{c}" for c in columns]
    result[features] = result[features].astype("float32").replace([np.inf, -np.inf], np.nan)
    old = pd.read_parquet(old_path, columns=["symbol", "date", "feature_date", *features])
    common = result.merge(old, on=["symbol", "date"], how="inner", validate="one_to_one", suffixes=("", "_old"))
    missing_reference_rows = 0
    if args.event_extension:
        reference_keys = boundary[["symbol", "date"]].merge(
            old[["symbol", "date"]], on=["symbol", "date"], how="left", indicator=True)
        absent = reference_keys.loc[reference_keys["_merge"].eq("left_only"), ["symbol", "date"]]
        # The original candidate pool can contain suspended/no-bar entries;
        # the refreshed reference intentionally contains valid daily bars only.
        if len(absent.merge(observed_keys, on=["symbol", "date"], how="inner")):
            raise ValueError("Observed boundary key missing from refreshed reference")
        missing_reference_rows = len(absent)
        absent.to_csv(output / "boundary_without_raw_bar.csv", index=False)
        if len(common) + missing_reference_rows != proof["boundary_rows"]:
            raise ValueError("Incomplete lag boundary comparison")
    if not common.feature_date.eq(common.feature_date_old).all():
        raise ValueError("Original feature calendar differs")
    comparisons = []
    for feature in features:
        left = common[feature].to_numpy()
        right = common[f"{feature}_old"].astype("float32").to_numpy()
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        comparisons.append({"feature": feature, "common_rows": len(common), "different_rows": int((~same).sum())})
    pd.DataFrame(comparisons).to_csv(output / "overlap_comparison.csv", index=False)
    result.to_parquet(output / "dataset.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"rows": len(result), "symbols": int(result.symbol.nunique()), "features": features,
               "warmup_start": str(warmup.date()), "history_end": history_end, "common_rows": len(common),
               "boundary_without_raw_bar": missing_reference_rows,
               "different_cells": sum(r["different_rows"] for r in comparisons),
               "dataset_sha256": sha256(output / "dataset.parquet"), "script_sha256": sha256(Path(__file__)),
               "timing": "Exact prior market date; unchanged original stock-observation rolling statistics; missing joins retained",
               "status": "Data preparation only; no new predictive performance result"}
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if summary["different_cells"]:
        raise ValueError("Overlap differs: inspect before claiming sample-only ablation")
    (output / "complete.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
