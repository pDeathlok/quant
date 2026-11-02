"""Independently replay raw twenty-observation amount means for overlap differences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "overnight_unfiltered_lag_statistics_v1"
    output = root / "overnight_unfiltered_lag_amount_audit_v1"
    output.mkdir(exist_ok=False)
    feature = "lag1_daily_amount_mean20"
    current_path = source / "dataset.parquet"
    old_path = root / "tail_overnight/dataset.parquet"
    hashes = json.loads((source / "sources.json").read_text())
    hashes[str(current_path)] = sha256(current_path)
    current = pd.read_parquet(current_path, columns=["symbol", "date", "feature_date", feature])
    old = pd.read_parquet(old_path, columns=["symbol", "date", feature])
    joined = current.merge(old, on=["symbol", "date"], validate="one_to_one", suffixes=("", "_old"))
    old_value = joined[f"{feature}_old"].astype("float32")
    different = ~(joined[feature].eq(old_value) | (joined[feature].isna() & old_value.isna()))
    cases = joined.loc[different].copy()
    frames = []
    for name in hashes:
        if "/daily_partitioned/" in name:
            frame = pd.read_parquet(name, columns=["symbol", "date", "amount"])
            frames.append(frame.loc[frame.symbol.isin(cases.symbol)])
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw.date).dt.normalize()
    raw = raw.loc[raw.date.between("2023-09-03", "2026-09-04")].sort_values(["symbol", "date"])
    if raw.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate raw amount keys")
    groups = {symbol: part for symbol, part in raw.groupby("symbol")}
    replay = []
    for row in cases.itertuples(index=False):
        window = groups[row.symbol].loc[lambda f: f.date.le(row.feature_date)].tail(20)
        if len(window) != 20 or window.date.iloc[-1] != row.feature_date or window.amount.isna().any():
            raise ValueError("Difference lacks twenty exact-ended observed amounts")
        replay.append(float((window.amount.to_numpy() * 1000).mean()))
    cases["raw_window_recomputed"] = replay
    np.testing.assert_allclose(cases[feature], cases.raw_window_recomputed, rtol=1e-6, atol=.01)
    cases.to_parquet(output / "differences.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"different_rows": len(cases), "raw_window_replay_passed": len(cases),
               "old_missing_now_known": int(cases[f"{feature}_old"].isna().sum()),
               "both_known_changed": int(cases[f"{feature}_old"].notna().sum()),
               "before_2026_differences": int(cases.date.lt("2026-01-01").sum()),
               "first_difference": str(cases.date.min()), "last_difference": str(cases.date.max()),
               "policy": "Keep mismatch evidence; use refreshed inputs in BOTH pool arms for2026; never call stale-cache comparison sample-only",
               "limitation": "Raw replay verifies current calculation, not original vendor revision publication timing",
               "script_sha256": sha256(Path(__file__))}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
