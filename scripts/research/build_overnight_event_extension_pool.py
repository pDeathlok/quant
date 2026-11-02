"""Reconstruct locked T-1 pool and require historical boundary parity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from audit_overnight_right_provenance import sha256
from run_tail_overnight import LEFT, LONG, RIGHT, historical_pools, read_daily

from quant.core.paths import PROJECT_ROOT
from quant.research.tail_overnight import POOL_NAMES, calendar_links


def main() -> None:
    root = PROJECT_ROOT
    research = root / "reports/research"
    lock_path = research / "overnight_event_extension_lock_v1/protocol.json"
    lock = json.loads(lock_path.read_text())
    for filename, digest in lock["sources"].items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"Locked source changed: {filename}")
    output = research / "overnight_event_extension_pool_v1"
    output.mkdir(exist_ok=False)
    short_paths = [root / LEFT, root / RIGHT]
    for folder in ["left_side_unified", "right_side_unified"]:
        short_paths.extend(p for p in sorted((root / "data/features" / folder).glob("*_features.parquet"))
                           if "20260831" <= p.name[:8] <= "20260909")
    raw_paths = [p for p in sorted((root / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
                 if "202309" <= p.parent.name.split("=")[1] <= "202609"]
    long_paths = [p for p in sorted((root / "data/features/long").glob("*.parquet"))
                  if "20260831" <= p.stem <= "20260909"]
    old_path = research / "tail_overnight/dataset.parquet"
    files = [lock_path, old_path, root / LONG, root / "src/quant/webapp/services.py",
             *short_paths, *raw_paths, *long_paths]
    hashes = {str(p): sha256(p) for p in files}
    filters = [("date", ">=", pd.Timestamp("2026-08-31")), ("date", "<=", pd.Timestamp("2026-09-09"))]
    short = pd.concat([pd.read_parquet(p, columns=["symbol", "date"], filters=filters)
                       for p in short_paths], ignore_index=True).drop_duplicates(["symbol", "date"])
    print(f"Short-screen keys: {len(short)}", flush=True)
    # Same full warm-up as the historical run; no next-day labels are constructed.
    daily = read_daily("2024-01-01", "2026-09-10")
    calendar = sorted(daily.date.unique())
    screen = historical_pools(short, calendar, "2026-08-31", "2026-09-09", daily)
    screen.to_parquet(output / "screen_pool.parquet", index=False)
    links = calendar_links(calendar)[["date", "exit_date"]]
    pool = screen.merge(links, on="date", how="left", validate="many_to_one")
    pool = pool.rename(columns={"date": "screen_date", "exit_date": "date"})
    pool = pool.loc[pool.date.between("2026-09-01", "2026-09-10")].copy()
    columns = ["symbol", "date", *[f"pool_{name}" for name in POOL_NAMES]]
    old = pd.read_parquet(old_path, columns=columns,
                          filters=[("date", ">=", pd.Timestamp("2026-09-01")),
                                   ("date", "<=", pd.Timestamp("2026-09-04"))])
    boundary = pool.loc[pool.date.le("2026-09-04"), columns]
    comparison = boundary.merge(old, on=["symbol", "date"], how="outer", suffixes=("_new", "_old"), indicator=True)
    comparison.to_parquet(output / "boundary_comparison.parquet", index=False)
    if not comparison._merge.eq("both").all():
        raise ValueError(f"Boundary membership mismatch: {comparison._merge.value_counts().to_dict()}")
    for name in POOL_NAMES:
        if not comparison[f"pool_{name}_new"].equals(comparison[f"pool_{name}_old"]):
            raise ValueError(f"Boundary pool flag mismatch: {name}")
    if pool.duplicated(["symbol", "date"]).any() or not pool.screen_date.lt(pool.date).all():
        raise ValueError("Invalid entry mapping")
    extension = pool.loc[pool.date.ge(lock["entry_start"])].copy()
    extension.to_parquet(output / "entry_pool.parquet", index=False)
    coverage = extension.groupby("date").size().rename("rows")
    coverage.to_csv(output / "coverage.csv")
    for filename, digest in hashes.items():
        if sha256(Path(filename)) != digest:
            raise ValueError(f"Source changed during pool construction: {filename}")
    (output / "complete.json").write_text(json.dumps({"sources": hashes, "boundary_rows": len(boundary),
        "extension_rows": len(extension), "extension_days": len(coverage),
        "dataset_sha256": sha256(output / "entry_pool.parquet"), "script_sha256": sha256(Path(__file__)),
        "status": "Pool boundary replay passed; extension predictions/outcomes not computed"}, indent=2))
    print(coverage.to_string(), flush=True)


if __name__ == "__main__":
    main()
