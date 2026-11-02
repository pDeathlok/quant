"""Next-session calendar features only, never future prices or earnings outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def target_calendar(open_dates: pd.DatetimeIndex, decisions: pd.DatetimeIndex) -> pd.DataFrame:
    if not open_dates.is_unique or open_dates.hasnans or not open_dates.is_monotonic_increasing:
        raise ValueError("Calendar requires unique sorted nonmissing sessions")
    if not decisions.is_unique or not decisions.isin(open_dates).all():
        raise ValueError("Decision dates must be unique market sessions")
    links = pd.DataFrame({"date": open_dates})
    links["target_date"] = links.date.shift(-1)
    links["following_date"] = links.date.shift(-2)
    links = links.loc[links.date.isin(decisions)].reset_index(drop=True)
    if links[["target_date", "following_date"]].isna().any().any():
        raise ValueError("Need two known future calendar sessions, not future quotes")
    target = links.target_date
    prefix = "calendar_target_"
    links[prefix + "month"] = target.dt.month
    links[prefix + "day"] = target.dt.day
    links[prefix + "weekday"] = target.dt.weekday
    links[prefix + "week_of_month"] = (target.dt.day - 1) // 7 + 1
    links[prefix + "gap_days"] = (target - links.date).dt.days
    links[prefix + "first_month_session"] = target.dt.to_period("M").ne(links.date.dt.to_period("M"))
    links[prefix + "last_month_session"] = target.dt.to_period("M").ne(links.following_date.dt.to_period("M"))
    links[prefix + "first_quarter_session"] = target.dt.to_period("Q").ne(links.date.dt.to_period("Q"))
    links[prefix + "last_quarter_session"] = target.dt.to_period("Q").ne(links.following_date.dt.to_period("Q"))
    return links.drop(columns="following_date")


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    calendar_root = root / "overnight_trading_calendar_v1"
    calendar_path = calendar_root / "calendar.parquet"
    complete = json.loads((calendar_root / "complete.json").read_text())
    if sha256(calendar_path) != complete["dataset_sha256"]:
        raise ValueError("Calendar changed")
    universe = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    labels_path = root / "overnight_unfiltered_labels_v1/labels.parquet"
    hashes = {str(p): sha256(p) for p in [calendar_path, universe, labels_path]}
    if hashes[str(universe)] != complete["source_sha256"]:
        raise ValueError("Universe differs from calendar audit")
    output = root / "overnight_target_calendar_v1"
    output.mkdir(exist_ok=False)
    raw = pd.read_parquet(calendar_path)
    keys = pd.read_parquet(universe, columns=["symbol", "date"])
    dates = pd.DatetimeIndex(keys.date.unique()).sort_values()
    daily = target_calendar(pd.DatetimeIndex(raw.loc[raw.is_open.eq(1), "date"]), dates)
    labels = pd.read_parquet(labels_path, columns=["date", "exit_date"]).drop_duplicates()
    comparison = daily.merge(labels, on="date", how="left", validate="one_to_one")
    known = comparison.exit_date.notna()
    if not comparison.loc[known, "target_date"].equals(comparison.loc[known, "exit_date"]):
        raise ValueError("Vendor calendar target differs from existing label calendar")
    features = [c for c in daily if c.startswith("calendar_target_")]
    frame = keys.merge(daily, on="date", how="left", validate="many_to_one")
    pd.testing.assert_frame_equal(frame[["symbol", "date"]], keys)
    frame[features] = frame[features].astype("float32")
    if len(features) != 9 or not np.isfinite(frame[features].to_numpy()).all() or not frame.target_date.gt(frame.date).all():
        raise ValueError("Invalid target calendar features")
    daily.to_parquet(output / "by_date.parquet", index=False)
    frame.to_parquet(output / "dataset.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Input changed")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    summary = {"rows": len(frame), "dates": len(daily), "features": features,
               "known_label_dates_replayed": int(known.sum()), "label_end_dates_without_outcome": int((~known).sum()),
               "dataset_sha256": sha256(output / "dataset.parquet"), "script_sha256": sha256(Path(__file__)),
               "policy": "Calendar only; target_date is audit metadata, not an unrestricted timestamp feature; no year or price inputs",
               "limitation": "Calendar snapshot version timing not independently proven; month/day are not actual company earnings dates; original labels unchanged"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
