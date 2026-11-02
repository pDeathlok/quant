"""Entry-only eligibility over expanded daily keys; no future labels consulted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    limits_root = root / "overnight_unfiltered_limits_v1"
    if not (limits_root / "complete.json").exists():
        raise ValueError("Full limit download is not complete")
    source = root / "overnight_unfiltered_universe_v1/daily_bars.parquet"
    frame = pd.read_parquet(source, columns=["symbol", "date", "close", "vol", "valid_bar"])
    manifest = json.loads((limits_root / "manifest.json").read_text())
    hashes = {str(source): sha256(source)}
    if hashes[str(source)] != manifest["source_sha256"]:
        raise ValueError("Changed universe")
    coverage = pd.read_csv(limits_root / "coverage.csv", dtype={"date": str}).set_index("date")
    parts = []
    for day in manifest["dates"]:
        path = limits_root / f"{day}.parquet"
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != coverage.loc[day, "sha256"]:
            raise ValueError("Limit file changed")
        p = pd.read_parquet(path, columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        if not p.trade_date.astype(str).eq(day).all() or p.ts_code.duplicated().any():
            raise ValueError("Invalid limit keys")
        p["date"] = pd.to_datetime(day)
        parts.append(p.rename(columns={"ts_code": "symbol"}).drop(columns="trade_date"))
    limits = pd.concat(parts, ignore_index=True)
    gate = frame.merge(limits, on=["symbol", "date"], how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(frame, gate[frame.columns])
    values = gate[["close", "vol", "up_limit"]].apply(pd.to_numeric, errors="coerce")
    gate["unknown_entry"] = ~(np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1) & gate.valid_bar)
    gate["at_upper"] = gate.close.ge(gate.up_limit - .0051) & ~gate.unknown_entry
    gate["eligible"] = ~gate.unknown_entry & ~gate.at_upper
    old_path = root / "overnight_moneyflow_rolling_risk_2026_v1/exact_entry_gate.parquet"
    hashes[str(old_path)] = sha256(old_path)
    old = pd.read_parquet(old_path)
    common = gate.merge(old, on=["symbol", "date"], suffixes=("_new", "_old"), validate="one_to_one")
    for name in ("unknown_entry", "at_upper", "eligible"):
        if not common[f"{name}_new"].eq(common[f"{name}_old"]).all():
            raise ValueError(f"Old/new gate disagreement: {name}; audit before use")
    output = root / "overnight_unfiltered_entry_gate_v1"
    output.mkdir(exist_ok=False)
    gate.to_parquet(output / "entry_gate.parquet", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during gating")
    summary = {"rows": len(gate), "eligible": int(gate.eligible.sum()), "unknown": int(gate.unknown_entry.sum()),
               "at_upper": int(gate.at_upper.sum()), "old_common_replayed": len(common),
               "dataset_sha256": sha256(output / "entry_gate.parquet"), "script_sha256": sha256(Path(__file__)),
               "policy": "Only current-day bar and upper limit; no outcome or exit data used in eligibility",
               "limits": "EOD proxy, not guaranteed14:30 fill; exit availability remains a separate audit"}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
