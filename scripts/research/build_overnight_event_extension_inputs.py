"""Rebuild locked 187 daily inputs and verify the shared September boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256
from overnight_market_context import market_context

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def compare(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> list[dict]:
    common = left.merge(right, on=["symbol", "date"], validate="one_to_one", suffixes=("", "_old"))
    result = []
    for column in columns:
        a = common[column].to_numpy(dtype="float32")
        b = common[f"{column}_old"].to_numpy(dtype="float32")
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        result.append({"feature": column, "common_rows": len(common), "different_cells": int((~same).sum())})
    return result


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "overnight_event_extension_inputs_v1"
    pool = root / "overnight_event_extension_pool_v1"
    factor_path = root / "overnight_event_extension_full_factors_v1/dataset.parquet"
    lag_path = root / "overnight_event_extension_lag_statistics_v1/dataset.parquet"
    old_factor_path = root / "overnight_unfiltered_full_factors_v1/dataset.parquet"
    old_aux_path = root / "overnight_unfiltered_auxiliary_v1/dataset.parquet"
    protocol_path = root / "overnight_same_day_market_limit_gated_v2/protocol.json"
    paths = [pool / "entry_pool.parquet", pool / "boundary_comparison.parquet", pool / "complete.json",
             factor_path, lag_path, old_factor_path, old_aux_path, protocol_path,
             PROJECT_ROOT / "scripts/research/overnight_market_context.py",
             PROJECT_ROOT / "src/quant/features/variable_library.py"]
    for path in [factor_path, lag_path, old_factor_path, old_aux_path]:
        proof = path.parent / "complete.json"
        if sha256(path) != json.loads(proof.read_text())["dataset_sha256"]:
            raise ValueError(f"Changed input: {path}")
        paths.append(proof)
    raw_paths = [p for p in sorted((PROJECT_ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
                 if "202310" <= p.parent.name.split("=")[1] <= "202609"]
    hashes = {str(p): sha256(p) for p in [*paths, *raw_paths]}
    output.mkdir(exist_ok=False)
    boundary = pd.read_parquet(pool / "boundary_comparison.parquet")
    entry = pd.read_parquet(pool / "entry_pool.parquet")
    proof = json.loads((pool / "complete.json").read_text())
    if sha256(pool / "entry_pool.parquet") != proof["dataset_sha256"] or not boundary["_merge"].eq("both").all():
        raise ValueError("Changed or unaudited pool")
    keys = pd.concat([boundary[["symbol", "date"]], entry[["symbol", "date"]]], ignore_index=True)
    if keys.duplicated().any():
        raise ValueError("Duplicate keys")
    raw_ds = ds.dataset([str(p) for p in raw_paths], format="parquet")
    bars = raw_ds.to_table(columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume", "amount"],
        filter=(ds.field("trade_date") >= "20231001") & (ds.field("trade_date") <= "20260910")).to_pandas()
    context = market_context(bars)
    context.to_parquet(output / "market_context.parquet", index=False)
    bars["vol"] = bars.vol.fillna(bars.volume)
    bars = bars.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    bars["date"] = pd.to_datetime(bars.date)
    bars = keys.merge(bars, on=["symbol", "date"], how="left", validate="one_to_one", indicator=True)
    missing = bars.loc[bars["_merge"].eq("left_only"), ["symbol", "date"]]
    missing.to_csv(output / "missing_daily_bars.csv", index=False)
    raw = keys.copy()
    for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        raw[f"today_raw_{column}"] = bars[column]
    raw["today_raw_return"] = bars.close / bars.pre_close - 1
    raw["today_raw_gap"] = bars.open / bars.pre_close - 1
    raw["today_raw_range"] = (bars.high - bars.low) / bars.pre_close
    raw["today_raw_close_position"] = (bars.close - bars.low) / (bars.high - bars.low).replace(0, np.nan)
    lag = pd.read_parquet(lag_path)
    pd.testing.assert_frame_equal(keys, lag[["symbol", "date"]])
    frame = raw.merge(lag, on=["symbol", "date"], validate="one_to_one")
    frame = frame.merge(context, on="date", how="left", validate="many_to_one")
    protocol = json.loads(protocol_path.read_text())
    auxiliary = [c for c in protocol["features"] if not c.startswith("today_factor_")]
    if len(auxiliary) != 42 or frame[protocol["market_context_features"]].isna().any().any():
        raise ValueError("Invalid auxiliary schema or context")
    if not frame.feature_date.lt(frame.date).all():
        raise ValueError("Noncausal lag date")
    frame[auxiliary] = frame[auxiliary].astype("float32").replace([np.inf, -np.inf], np.nan)
    factors = pd.read_parquet(factor_path)
    comparisons = compare(factors, pd.read_parquet(old_factor_path), list(PROJECT_FACTOR_COLUMNS))
    comparisons += compare(frame, pd.read_parquet(old_aux_path), auxiliary)
    pd.DataFrame(comparisons).to_csv(output / "boundary_comparison.csv", index=False)
    absent_boundary = boundary[["symbol", "date"]].merge(missing, on=["symbol", "date"])
    expected = len(boundary) - len(absent_boundary)
    if any(c["common_rows"] != expected or c["different_cells"] for c in comparisons):
        raise ValueError("Boundary input mismatch; inspect comparison, do not predict")
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(factors, on=["symbol", "date"], how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(keys, frame[["symbol", "date"]])
    features = protocol["features"]
    if len(features) != 187:
        raise ValueError("Changed locked feature schema")
    frame = frame[["symbol", "date", "feature_date", "market_row_present", "basic_row_present", *features]]
    frame.to_parquet(output / "dataset.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"rows": len(frame), "features": features, "boundary_rows": expected,
               "different_cells": 0, "missing_daily_bars": len(missing),
               "dataset_sha256": sha256(output / "dataset.parquet"), "script_sha256": sha256(Path(__file__)),
               "timing": "Daily close proxy, future deployment 14:30 approximation; no minute data",
               "status": "Inputs verified only; no extension predictions or next-day outcomes read"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}), flush=True)


if __name__ == "__main__":
    main()
