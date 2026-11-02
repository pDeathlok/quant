"""Score frozen extension picks with unchanged labels and daily exit checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_close_exit_limits import exit_flags
from audit_overnight_right_provenance import sha256
from build_overnight_unfiltered_labels import label_bars

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    prediction_root = root / "overnight_event_extension_predictions_v1"
    pick_path = prediction_root / "picks.parquet"
    prediction_proof = prediction_root / "complete.json"
    proof = json.loads(prediction_proof.read_text())
    if sha256(pick_path) != proof["picks_sha256"]:
        raise ValueError("Frozen picks changed")
    raw_path = PROJECT_ROOT / "data/raw/daily_partitioned/year_month=202609/data.parquet"
    calendar_path = root / "overnight_trading_calendar_v1/calendar.parquet"
    limit_root = root / "overnight_event_extension_exit_limits_v1"
    coverage_path = limit_root / "coverage.csv"
    paths = [pick_path, prediction_proof, raw_path, calendar_path, coverage_path,
             limit_root / "complete.json", root / "overnight_event_extension_lock_v1/protocol.json",
             PROJECT_ROOT / "scripts/research/build_overnight_unfiltered_labels.py",
             PROJECT_ROOT / "scripts/research/audit_overnight_close_exit_limits.py"]
    hashes = {str(p): sha256(p) for p in paths}
    picks = pd.read_parquet(pick_path)
    bars = pd.read_parquet(raw_path, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume"])
    bars = bars.rename(columns={"ts_code": "symbol", "trade_date": "date"})
    bars["date"] = pd.to_datetime(bars.date)
    bars = bars.loc[bars.date.between("2026-09-01", "2026-09-11")].copy()
    bars["vol"] = bars.vol.fillna(bars.volume)
    values = bars[["open", "high", "low", "close", "pre_close", "vol"]]
    bars["valid_bar"] = np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
    bars["valid_bar"] &= bars.high.ge(bars[["open", "close"]].max(axis=1) - 1e-8)
    bars["valid_bar"] &= bars.low.le(bars[["open", "close"]].min(axis=1) + 1e-8)
    calendar = pd.read_parquet(calendar_path)
    dates = calendar.loc[calendar.is_open.eq(1) & calendar.date.between("2026-09-01", "2026-09-11"), "date"]
    if set(dates) != set(bars.date.unique()) or bars.groupby("date").symbol.nunique().lt(1000).any():
        raise ValueError("Incomplete observed market calendar")
    labeled = label_bars(bars)
    joined = picks.merge(labeled, on=["symbol", "date"], how="left", validate="one_to_one")
    if joined.label_status.isna().any():
        raise ValueError("Missing selected entry")
    parts = []
    for row in pd.read_csv(coverage_path, dtype={"date": str}).itertuples():
        path = limit_root / f"{row.date}.parquet"
        hashes[str(path)] = row.sha256
        part = pd.read_parquet(path).rename(columns={"ts_code": "symbol", "trade_date": "exit_date"})
        part["exit_date"] = pd.to_datetime(part.exit_date)
        parts.append(part[["symbol", "exit_date", "up_limit", "down_limit"]])
    joined = joined.merge(pd.concat(parts), on=["symbol", "exit_date"], how="left", validate="many_to_one")
    flags = exit_flags(joined, .01)
    joined = pd.concat([joined, flags], axis=1)
    joined["price_limit_inconsistent"] = ((joined.up_limit.gt(0) & joined.next_high.gt(joined.up_limit + .011))
        | (joined.down_limit.gt(0) & joined.next_low.lt(joined.down_limit - .011)))
    joined["exit_uncertain"] = (~joined.label_status.eq("ok") | joined.price_limit_inconsistent
        | joined.fallback_at_lower | joined.fallback_limit_unknown)
    joined["sell_limit"] = np.ceil(joined.entry_close * 1.01 * 100 - 1e-9) / 100
    joined["net20"] = joined.limit1
    joined["net40"] = joined.limit1 - .002
    joined["audited_net20"] = joined.net20.where(~joined.exit_uncertain)
    pd.testing.assert_frame_equal(picks.reset_index(drop=True), joined[picks.columns])
    output = root / "overnight_event_extension_outcomes_v1"
    output.mkdir(exist_ok=False)
    joined.to_parquet(output / "outcomes.parquet", index=False)
    summary = {"signals": len(joined), "event_hits": int(joined.limit1_penetrated.fillna(False).sum()),
               "unknown_labels": int(joined.limit1.isna().sum()), "exit_uncertain": int(joined.exit_uncertain.sum()),
               "profitable20": int(joined.net20.gt(0).sum()), "mean_net20": float(joined.net20.mean()),
               "mean_net40": float(joined.net40.mean()), "dataset_sha256": sha256(output / "outcomes.parquet"),
               "script_sha256": sha256(Path(__file__)),
               "status": "Four-date historical extension, not independent confirmation; daily proxy does not establish actual fills"}
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Audit source changed: {path}")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    (output / "REPORT.md").write_text(
        "# Frozen daily-close extension result\n\n"
        f"Signals: {len(joined)}; literal 1% limit penetration hits: {summary['event_hits']}; "
        f"unknown labels: {summary['unknown_labels']}; exit uncertainty flags: {summary['exit_uncertain']}.\n\n"
        f"Mean net return: {summary['mean_net20']:.6%} at20bps, {summary['mean_net40']:.6%} at40bps.\n\n"
        "Selection was frozen before outcome construction. No threshold, pool, model, or exit rule changed. "
        "This short historical diagnostic cannot establish80%/90% precision or14:30 fills. "
        "No detected pre-close discontinuity is only a corporate-action screen, not exhaustive corporate-action proof.\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
