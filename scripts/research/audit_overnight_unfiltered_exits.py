"""Post-decision exit uncertainty, preserving all labels and entry decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_close_exit_limits import exit_flags
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    label_path = root / "overnight_unfiltered_labels_v1/labels.parquet"
    gate_path = root / "overnight_unfiltered_entry_gate_v1/entry_gate.parquet"
    cache = root / "overnight_unfiltered_limits_v1"
    if not (cache / "complete.json").exists():
        raise ValueError("Limit cache incomplete")
    hashes = {str(p): sha256(p) for p in (label_path, gate_path)}
    labels = pd.read_parquet(label_path)
    gate = pd.read_parquet(gate_path, columns=["symbol", "date", "eligible"])
    pd.testing.assert_frame_equal(labels[["symbol", "date"]], gate[["symbol", "date"]])
    coverage = pd.read_csv(cache / "coverage.csv", dtype={"date": str})
    parts = []
    for row in coverage.itertuples():
        path = cache / f"{row.date}.parquet"
        hashes[str(path)] = sha256(path)
        if hashes[str(path)] != row.sha256:
            raise ValueError("Limit source changed")
        part = pd.read_parquet(path, columns=["ts_code", "trade_date", "up_limit", "down_limit"])
        part = part.rename(columns={"ts_code": "symbol", "trade_date": "exit_date"})
        part["exit_date"] = pd.to_datetime(part.exit_date.astype(str), format="%Y%m%d")
        parts.append(part)
    joined = labels.merge(pd.concat(parts), on=["symbol", "exit_date"], how="left", validate="many_to_one")
    pd.testing.assert_frame_equal(joined[labels.columns], labels)
    out = joined[["symbol", "date", "exit_date", "label_status", "up_limit", "down_limit"]].copy()
    out["entry_eligible"] = gate.eligible
    known = joined.label_status.eq("ok")
    bounds = known & ((joined.up_limit.gt(0) & joined.next_high.gt(joined.up_limit + .011))
                      | (joined.down_limit.gt(0) & joined.next_low.lt(joined.down_limit - .011)))
    out["price_limit_inconsistent"] = bounds
    summaries = []
    for target, threshold in (("limit05", .005), ("limit1", .01)):
        flags = exit_flags(joined, threshold)
        for column in flags:
            out[f"{target}_{column}"] = flags[column] & known
        uncertain = ~known | bounds | (known & (flags.fallback_at_lower | flags.fallback_limit_unknown))
        out[f"{target}_exit_uncertain"] = uncertain
        out[f"{target}_net_mark"] = joined[target]
        # Keep both fields. Never silently omit these losses when scoring selected trades.
        out[f"{target}_net_after_exit_audit"] = joined[target].where(~uncertain)
        for scope, mask in (("all", pd.Series(True, index=joined.index)), ("entry_eligible", gate.eligible)):
            summaries.append({"target": target, "scope": scope, "rows": int(mask.sum()),
                "unknown_price_label": int((mask & ~known).sum()),
                "fallback_at_lower": int((mask & known & flags.fallback_at_lower).sum()),
                "fallback_limit_unknown": int((mask & known & flags.fallback_limit_unknown).sum()),
                "price_limit_inconsistent": int((mask & bounds).sum()),
                "exit_uncertain": int((mask & uncertain).sum()),
                "uncertain_known_mark_mean": float(joined.loc[mask & uncertain & known, target].mean())})
        np.testing.assert_allclose(out[f"{target}_net_mark"], labels[target], equal_nan=True, rtol=0, atol=0)
    if not out.entry_eligible.equals(gate.eligible):
        raise ValueError("Entry decisions changed")
    output = root / "overnight_unfiltered_exit_audit_v1"
    output.mkdir(exist_ok=False)
    out.to_parquet(output / "exit_audit.parquet", index=False)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during audit")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps({"summary": summaries,
        "dataset_sha256": sha256(output / "exit_audit.parquet"), "script_sha256": sha256(Path(__file__)),
        "scope": "Postdecision audit; raw marks and entry decisions unchanged; uncertainty must count in selected-trade evaluation, not ex-ante filtering",
        "limits": "No detected daily limit issue does not prove actual order fills"}, indent=2))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
