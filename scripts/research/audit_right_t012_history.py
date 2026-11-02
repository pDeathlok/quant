"""Check old-window parity and truncated-history causality of rebuilt signals."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from audit_overnight_right_provenance import sha256
from build_overnight_current_right_signals import compute

from quant.core.paths import PROJECT_ROOT
from quant.research.right_side_unified_signals import CANONICAL_RIGHT_SIDE_SIGNALS


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "right_t012_history_v1"
    output = root / "right_t012_history_audit_v1"
    complete = json.loads((source / "complete.json").read_text())
    path = source / "all_daily_signals.parquet"
    if sha256(path) != complete["dataset_sha256"]:
        raise ValueError("Signal build hash mismatch")
    columns = ["symbol", "date", *CANONICAL_RIGHT_SIDE_SIGNALS]
    full = pd.read_parquet(path, columns=columns)
    old_path = root / "overnight_current_right_signals_v2/all_daily_signals.parquet"
    old = pd.read_parquet(old_path, columns=columns)
    overlap = old.merge(full, on=["symbol", "date"], how="left", suffixes=("_old", "_new"), indicator=True, validate="one_to_one")
    if not overlap._merge.eq("both").all():
        raise ValueError("Old window keys missing")
    different = {}
    for signal in CANONICAL_RIGHT_SIDE_SIGNALS:
        left, right = overlap[f"{signal}_old"], overlap[f"{signal}_new"]
        mismatch = ~(left.eq(right).fillna(False) | (left.isna() & right.isna()))
        different[signal] = int(mismatch.sum())
    output.mkdir(exist_ok=False)
    (output / "overlap.json").write_text(json.dumps(dict(rows=len(old), mismatches=different), indent=2))
    if any(different.values()):
        raise ValueError(f"Old window signal drift: {different}")
    protocol = json.loads((source / "protocol.json").read_text())
    raw_paths = [p for p in protocol["sources"] if "/daily_partitioned/" in p]
    raw = ds.dataset(raw_paths, format="parquet")
    results = []
    for symbol in ["000001.SZ", "300750.SZ", "600519.SH"]:
        history = raw.to_table(filter=ds.field("ts_code").isin([symbol]), columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "volume", "amount", "pct_chg", "change"]).to_pandas()
        for cutoff in ["2024-06-14", "2025-03-14", "2026-05-29"]:
            prefix = history.loc[pd.to_datetime(history.trade_date).le(cutoff)]
            replay = compute(prefix, symbol, start="2024-01-01", end=cutoff)[columns]
            expected = full.loc[full.symbol.eq(symbol) & full.date.le(cutoff), columns]
            replay = replay.merge(expected[["symbol", "date"]], on=["symbol", "date"], validate="one_to_one")
            pd.testing.assert_frame_equal(replay.sort_values("date").reset_index(drop=True), expected.sort_values("date").reset_index(drop=True))
            results.append(dict(symbol=symbol, cutoff=cutoff, rows=len(replay), mismatches=0))
    (output / "complete.json").write_text(json.dumps(dict(overlap_rows=len(old), overlap_different_cells=0,
        prefix_checks=results, signal_sha256=complete["dataset_sha256"], old_sha256=sha256(old_path),
        limitations="Sampled prefix audit, not raw vendor revision proof", script_sha256=sha256(Path(__file__))), indent=2))
    print(f"Old overlap={len(old)} rows exact; prefix cases={len(results)} exact", flush=True)


if __name__ == "__main__":
    main()
