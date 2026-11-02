"""Fixed economic hurdles applied to frozen expanded-pool predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from train_overnight_direct import metrics, select

from quant.core.paths import PROJECT_ROOT


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    parent = root / "overnight_unfiltered_comparison_v1"
    json.loads((parent / "complete.json").read_text())
    output = root / "overnight_unfiltered_positive_expectation_v1"
    output.mkdir(exist_ok=False)
    paths = [parent / f"{arm}_predictions.parquet" for arm in ["old", "expanded"]]
    paths += [parent / "diagnostics.csv", parent / "protocol.json"]
    hashes = {str(p): sha256(p) for p in paths}
    protocol = {"hurdles": {"baseline": None, "cover20bps": 0., "cover40bps": .002},
                "fixed_rule": "probability>=0.7, risk<=0.1, top1; same probability/net/symbol ranking",
                "hurdle_units": "Model expected net already deducts20bps; require strictly above hurdle before ranking",
                "status": "Post-result diagnostic on reused history; not independent confirmation or threshold optimization",
                "costs_bps": [20, 40], "sources": hashes, "script_sha256": sha256(Path(__file__))}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, calibration = [], []
    for arm in ["old", "expanded"]:
        pred = pd.read_parquet(parent / f"{arm}_predictions.parquet")
        if pred.duplicated(["symbol", "date"]).any():
            raise ValueError("Duplicate forecast keys")
        for population in ["old", "expanded"]:
            candidates = pred.loc[pred.old_pool] if population == "old" else pred
            for name, hurdle in protocol["hurdles"].items():
                admitted = candidates if hurdle is None else candidates.loc[candidates.expected_net.gt(hurdle)]
                picks = select(admitted, .7, -np.inf, 1, .1).copy()
                picks.to_parquet(output / f"{arm}_{population}_{name}_picks.parquet", index=False)
                for period, lower, upper in [("2025Q4", "2025-10-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-04")]:
                    part = picks.loc[picks.date.ge(lower) & picks.date.lt(upper)].copy()
                    days = pd.DatetimeIndex(pred.loc[pred.date.ge(lower) & pred.date.lt(upper), "date"].unique()).sort_values()
                    for view in ["raw_mark", "exit_audited"]:
                        evaluated = part.copy()
                        if view == "exit_audited":
                            evaluated["net"] = evaluated.limit05_net_after_exit_audit
                        for cost in [20, 40]:
                            scored = evaluated.copy()
                            scored["net"] -= (cost - 20) / 10000
                            rows.append({"train_pool": arm, "selection_pool": population, "hurdle": name,
                                "period": period, "view": view, "cost_bps": cost,
                                **metrics(scored, days, 1), "predicted_net_mean": float(part.expected_net.mean()),
                                "loss_over2pct": int(scored.net.lt(-.02).sum()), "worst_known_net": float(scored.net.min())})
            # Descriptive bins only; never feed evaluation outcomes back into thresholds.
            for period, lower, upper in [("2025Q4", "2025-10-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-04")]:
                part = candidates.loc[candidates.date.ge(lower) & candidates.date.lt(upper)].copy()
                part["expected_bin"] = pd.cut(part.expected_net, [-np.inf, 0., .002, np.inf], labels=["nonpositive", "positive_under40bps", "covers40bps"])
                for bucket, group in part.groupby("expected_bin", observed=True):
                    calibration.append({"train_pool": arm, "selection_pool": population, "period": period,
                        "expected_bin": str(bucket), "rows": len(group), "unknown": int(group.net.isna().sum()),
                        "predicted_mean": float(group.expected_net.mean()), "known_realized_mean": float(group.net.mean()),
                        "win_unknown_as_failure": float(group.net.gt(0).mean())})
    result = pd.DataFrame(rows)
    baseline = pd.read_csv(parent / "diagnostics.csv")
    replay = result.loc[result.hurdle.eq("baseline"), baseline.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(replay, baseline, check_dtype=False, rtol=1e-10, atol=1e-12)
    result.to_csv(output / "diagnostics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "prediction_bins.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Parent changed during diagnostic: {path}")
    summary = {"baseline_rows_replayed": len(baseline), "diagnostic_rows": len(result),
               "calibration_bins": len(calibration), "status": "Diagnostic complete; no independent success claim"}
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    print(result.loc[result.train_pool.eq("expanded") & result.selection_pool.eq("expanded") & result.view.eq("exit_audited"),
        ["hurdle", "period", "cost_bps", "signals", "unknown", "win_missing_as_failure", "known_mean_net", "loss_over2pct"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
