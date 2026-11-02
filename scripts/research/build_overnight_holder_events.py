"""Observed holder announcement features with a strict pre-decision date boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from build_overnight_moneyflow_lag import ALIAS_DATE

from quant.core.paths import PROJECT_ROOT

FEATURES = [f"holder_observed_{direction}_{window}d" for direction in ("IN", "DE") for window in (5, 20, 60)]
FEATURES += [f"holder_days_since_{direction}" for direction in ("IN", "DE")]


def observed_features(decisions: np.ndarray, announcements: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Counts in [decision-window, decision); announcement dates must be sorted."""
    if (announcements[1:] < announcements[:-1]).any():
        raise ValueError("Announcement dates are not sorted")
    hi = np.searchsorted(announcements, decisions, side="left")
    counts = [hi - np.searchsorted(announcements, decisions - np.timedelta64(w, "D"), side="left") for w in (5, 20, 60)]
    age = np.full(len(decisions), np.nan)
    has = hi > 0
    age[has] = (decisions[has] - announcements[hi[has]-1]) / np.timedelta64(1, "D")
    return counts, age


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "overnight_holder_reconciled_v1/events.parquet"
    manifest = json.loads((source.parent / "complete.json").read_text())
    if sha256(source) != manifest["events_sha256"]:
        raise ValueError("Reconciled events changed")
    candidate_path = root / "tail_overnight/dataset.parquet"
    hashes = {str(p): sha256(p) for p in (source, candidate_path)}
    events = pd.read_parquet(source)
    events["ann"] = pd.to_datetime(events.ann_date.astype(str), format="%Y%m%d", errors="raise")
    if events.ts_code.isna().any() or not events.in_de.isin(["IN", "DE"]).all():
        raise ValueError("Invalid source symbol or direction")
    alias = events.ts_code.eq("300114.SZ") & events.ann.lt(ALIAS_DATE)
    events.loc[alias, "ts_code"] = "302132.SZ"
    by_key = {(symbol, direction): group.ann.sort_values().to_numpy(dtype="datetime64[ns]")
              for (symbol, direction), group in events.groupby(["ts_code", "in_de"])}
    keys = pd.read_parquet(candidate_path, columns=["symbol", "date"])
    records = []
    for symbol, group in keys.groupby("symbol", sort=False):
        current = group.copy()
        for direction in ("IN", "DE"):
            ann = by_key.get((symbol, direction), np.array([], dtype="datetime64[ns]"))
            counts, age = observed_features(group.date.to_numpy(dtype="datetime64[ns]"), ann)
            for window, values in zip((5, 20, 60), counts, strict=True):
                current[f"holder_observed_{direction}_{window}d"] = values.astype(float)
            current[f"holder_days_since_{direction}"] = age
        records.append(current)
    frame = pd.concat(records).sort_index()
    pd.testing.assert_frame_equal(frame[["symbol", "date"]], keys)
    frame["source_stale"] = frame.date.gt(events.ann.max())
    frame.loc[frame.source_stale, FEATURES] = np.nan
    if (frame[["holder_days_since_IN", "holder_days_since_DE"]].le(0)).any().any():
        raise ValueError("Same-day or future announcement leaked")
    output = root / "overnight_holder_events_v1"
    output.mkdir(exist_ok=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during build")
    frame.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(frame), "features": FEATURES, "source_stale_rows": int(frame.source_stale.sum()),
               "last_source_ann_date": str(events.ann.max().date()), "alias_source_rows": int(alias.sum()),
               "dataset_sha256": sha256(output / "dataset.parquet"), "script_sha256": sha256(Path(__file__)),
               "timing": "Only ann_date < decision_date; windows are calendar days, not trading days",
               "semantics": "Zero counts mean this source has no observed records; NOT verified absence of real events. No prior record age is NaN.",
               "limitations": "Vendor completeness/revisions unverified; dates after latest source event are masked conservatively; no model result"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
