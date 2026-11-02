"""Freeze candidate before constructing September extension predictions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT
    research = root / "reports/research"
    candidate = research / "overnight_unfiltered_limit1_hgb_event_head_v1"
    output = research / "overnight_event_extension_lock_v1"
    output.mkdir(exist_ok=False)
    files = [candidate / "old.joblib", candidate / "protocol.json",
             research / "tail_overnight/manifest.json", root / "scripts/research/run_tail_overnight.py",
             root / "src/quant/research/tail_overnight.py"]
    hashes = {str(path): sha256(path) for path in files}
    protocol = json.loads((candidate / "protocol.json").read_text())
    if len(protocol["features"]) != 187 or protocol["target"] != "limit1":
        raise ValueError("Unexpected candidate protocol")
    lock = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "sources": hashes,
        "model": str(candidate / "old.joblib"), "features": protocol["features"],
        "rule": protocol["rule"], "exit_policy": protocol["exit_policy"],
        "classifier_event": "literal nominal sell-limit penetration, not total-return adjusted high",
        "entry_start": "2026-09-07", "entry_end": "2026-09-10", "last_exit": "2026-09-11",
        "excluded_last_entry": "2026-09-11 has no next-session bar in inspected inventory",
        "pool": "Original historical_pools union, shifted from screening day to exact next market entry day; no frozen current stock list",
        "pool_sources": "Same archived left/right short-key union and exact-date long good/price screens as original pipeline",
        "costs_bps": [20, 40], "top_k": 1,
        "entry_gate": "Same positive valid OHLCV and exact upper-limit exclusion as trained pipeline",
        "unknown_policy": "Keep selected unknown outcomes; no future-state selection filter; separately audit cash actions and exit limits",
        "no_changes": "No refit, no recalibration, no threshold tuning, no feature selection, no pool substitution",
        "preconditions": "Reproduce prior-period pool membership and original187 feature formulas before accepting extension predictions",
        "status": "Additional post-selection historical diagnostic only; four entry dates cannot prove full goal. Not guaranteed untouched by all other project research",
        "inventory_only_so_far": "Inspected September date/code counts and source filenames; no extension candidate prices or predictions evaluated in this branch",
        "script_sha256": sha256(Path(__file__)),
    }
    (output / "protocol.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2))
    print(json.dumps({"locked": str(output / "protocol.json"), "sources": len(hashes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
