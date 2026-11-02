"""Exact next-market-session OHLC and precommitted exit marks for expanded samples."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def label_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Future data goes only to labels, never entry qualification or feature columns."""
    if bars.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate observed keys")
    calendar = pd.DataFrame({"date": sorted(bars.date.unique())})
    calendar["exit_date"] = calendar.date.shift(-1)
    frame = bars[["symbol", "date", "close", "valid_bar"]].rename(columns={"close": "entry_close", "valid_bar": "entry_bar_valid"})
    frame = frame.merge(calendar, on="date", how="left", validate="many_to_one")
    following = bars[["symbol", "date", "open", "high", "low", "close", "pre_close", "valid_bar"]].rename(
        columns={c: f"next_{c}" for c in ("open", "high", "low", "close", "pre_close", "valid_bar")})
    following = following.rename(columns={"date": "exit_date"})
    frame = frame.merge(following, on=["symbol", "exit_date"], how="left", validate="many_to_one")
    missing = frame.next_valid_bar.isna()
    valid = frame.entry_bar_valid & frame.next_valid_bar.eq(True)
    corporate = frame.next_pre_close.sub(frame.entry_close).abs().gt(.011)
    frame["label_status"] = np.select(
        [frame.exit_date.isna(), missing, ~valid, corporate],
        ["end_of_observed_calendar", "missing_next_market_bar", "invalid_bar", "corporate_action_unresolved"], default="ok")
    known = frame.label_status.eq("ok")
    for price in ("open", "high", "low", "close"):
        frame[f"y_{price}"] = (frame[f"next_{price}"] / frame.entry_close - 1).where(known)
    for price in ("open", "close"):
        frame[f"net_{price}"] = frame[f"y_{price}"] - .002
    for name, threshold in (("limit05", .005), ("limit1", .01)):
        limit = np.ceil(frame.entry_close * (1 + threshold) * 100 - 1e-9) / 100
        reached = frame.next_high.ge(limit + frame.entry_close * .001)
        frame[f"{name}_penetrated"] = reached.where(known).astype("boolean")
        frame[name] = (frame.y_close.where(~reached, limit / frame.entry_close - 1) - .002).where(known)
    return frame


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    old_path = root / "overnight_same_day_market_limit_gated_v2/limit_label_source.parquet"
    hashes = {str(p): sha256(p) for p in (source, old_path)}
    expected_hash = json.loads((source.parent / "complete.json").read_text())["dataset_sha256"]
    if hashes[str(source)] != expected_hash:
        raise ValueError("Universe artifact changed")
    bars = pd.read_parquet(source)
    result = label_bars(bars)
    pd.testing.assert_frame_equal(result[["symbol", "date"]], bars[["symbol", "date"]])
    old = pd.read_parquet(old_path, columns=["symbol", "date", "exit_date", "limit05", "limit1"])
    comparison = old.merge(result[["symbol", "date", "exit_date", "label_status", "limit05", "limit1"]],
                           on=["symbol", "date"], how="left", suffixes=("_old", "_new"), validate="one_to_one")
    replay = {}
    for target in ("limit05", "limit1"):
        common = comparison[f"{target}_old"].notna() & comparison[f"{target}_new"].notna()
        np.testing.assert_allclose(comparison.loc[common, f"{target}_old"], comparison.loc[common, f"{target}_new"], rtol=1e-7, atol=1e-9)
        if not comparison.loc[common, "exit_date_old"].eq(comparison.loc[common, "exit_date_new"]).all():
            raise ValueError("Old/new next-session mapping differs")
        replay[target] = {"common_known": int(common.sum()),
                          "old_known_new_unknown": int((comparison[f"{target}_old"].notna() & comparison[f"{target}_new"].isna()).sum()),
                          "old_unknown_new_known": int((comparison[f"{target}_old"].isna() & comparison[f"{target}_new"].notna()).sum())}
    output = root / "overnight_unfiltered_labels_v1"
    output.mkdir(exist_ok=False)
    result.to_parquet(output / "labels.parquet", index=False)
    comparison.to_parquet(output / "old_label_comparison.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during label build")
    summary = {"rows": len(result), "status": result.label_status.value_counts().to_dict(), "old_replay": replay,
               "dataset_sha256": sha256(output / "labels.parquet"), "script_sha256": sha256(Path(__file__)),
               "policy": "Exact next global observed market date; 20bps cost; ceil cent sell limit and10bps penetration; close fallback; no intraday stop",
               "limitations": "Daily mark proxies only; no full expanded entry/exit limit audit; unknown outcomes must not select candidates; no model result"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
