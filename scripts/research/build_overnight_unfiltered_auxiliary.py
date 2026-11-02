"""Build the unchanged 42 auxiliary inputs for an expanded-pool comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "overnight_unfiltered_auxiliary_v1"
    universe = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    lag_root = root / "overnight_unfiltered_lag_statistics_v1"
    lag_path = lag_root / "dataset.parquet"
    audit_root = root / "overnight_unfiltered_lag_amount_audit_v1"
    parent = root / "overnight_same_day_market_limit_gated_v2"
    context_path = parent / "market_context.parquet"
    raw_reference = root / "overnight_moneyflow_rolling_risk_2026_v1/same_day_raw_snapshot.parquet"
    audit = json.loads((audit_root / "complete.json").read_text())
    audit_sources = json.loads((audit_root / "sources.json").read_text())
    lag_summary = json.loads((lag_root / "result.json").read_text())
    # Strict parity failed only on audited historical amount updates. Both
    # model arms must consume these same refreshed inputs, never stale values.
    if audit["different_rows"] != audit["raw_window_replay_passed"] or audit["before_2026_differences"]:
        raise ValueError("Unresolved lag-statistic overlap difference")
    if sha256(lag_path) != lag_summary["dataset_sha256"] or sha256(lag_path) != audit_sources[str(lag_path)]:
        raise ValueError("Lag artifact changed after independent amount replay")
    if lag_summary["different_cells"] != audit["different_rows"]:
        raise ValueError("Lag differences extend beyond audited amount updates")
    paths = [universe, lag_path, context_path, raw_reference, parent / "protocol.json",
             audit_root / "complete.json", audit_root / "sources.json", lag_root / "result.json"]
    hashes = {str(p): sha256(p) for p in paths}
    output.mkdir(exist_ok=False)
    bars = pd.read_parquet(universe)
    keys = bars[["symbol", "date"]]
    raw = keys.copy()
    for column in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
        raw[f"today_raw_{column}"] = bars[column]
    raw["today_raw_return"] = bars.close / bars.pre_close - 1
    raw["today_raw_gap"] = bars.open / bars.pre_close - 1
    raw["today_raw_range"] = (bars.high - bars.low) / bars.pre_close
    raw["today_raw_close_position"] = (bars.close - bars.low) / (bars.high - bars.low).replace(0, np.nan)
    raw_cols = [c for c in raw if c.startswith("today_raw_")]
    old = pd.read_parquet(raw_reference)
    compared = raw.merge(old, on=["symbol", "date"], validate="one_to_one", suffixes=("", "_old"))
    for column in raw_cols:
        np.testing.assert_array_equal(compared[column].to_numpy(), compared[f"{column}_old"].to_numpy())
    common_rows = len(compared)
    del bars, old, compared
    lag = pd.read_parquet(lag_path)
    pd.testing.assert_frame_equal(keys, lag[["symbol", "date"]])
    frame = raw.merge(lag, on=["symbol", "date"], validate="one_to_one")
    context = pd.read_parquet(context_path)
    protocol = json.loads((parent / "protocol.json").read_text())
    market_cols = protocol["market_context_features"]
    frame = frame.merge(context[["date", *market_cols]], on="date", how="left", validate="many_to_one")
    if frame[market_cols].isna().any().any() or not frame.feature_date.lt(frame.date).all():
        raise ValueError("Missing market context or noncausal lag date")
    pd.testing.assert_frame_equal(keys, frame[["symbol", "date"]])
    features = [c for c in protocol["features"] if not c.startswith("today_factor_")]
    if len(features) != 42 or set(features) != set([*raw_cols, *lag_summary["features"], *market_cols]):
        raise ValueError("Auxiliary feature schema mismatch")
    frame[features] = frame[features].astype("float32").replace([np.inf, -np.inf], np.nan)
    frame.to_parquet(output / "dataset.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Input changed during build: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"rows": len(frame), "features": features, "raw_exact_overlap_rows": common_rows,
               "dataset_sha256": sha256(output / "dataset.parquet"), "script_sha256": sha256(Path(__file__)),
               "lag_policy": "Use refreshed amount statistics in BOTH old and expanded pool models for2026",
               "market_policy": "Reuse identical frozen broad-market context; older regex omits302132.SZ from market aggregate only",
               "timing": "Daily close proxy; deployment14:30 approximation; no minute inputs",
               "status": "42 auxiliary inputs only, not the complete187-feature model dataset"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
