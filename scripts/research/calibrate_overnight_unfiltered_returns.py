"""Q3-only monotone calibration of frozen net-return predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select
from train_overnight_unfiltered_comparison import date_weights

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def fit_return_calibrator(frame: pd.DataFrame) -> IsotonicRegression:
    if frame.empty or not (frame.date.ge("2025-07-01") & frame.date.lt("2025-10-01")
                           & frame.exit_date.gt(frame.date) & frame.exit_date.lt("2025-10-01")).all():
        raise ValueError("Calibration must use only fully observed2025Q3 exits")
    if not np.isfinite(frame[["expected_net", "net"]].to_numpy()).all():
        raise ValueError("Calibration scores and returns must be finite")
    return IsotonicRegression(out_of_bounds="clip").fit(frame.expected_net, frame.net,
                                                        sample_weight=date_weights(frame))


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    parent = root / "overnight_unfiltered_comparison_v1"
    json.loads((parent / "complete.json").read_text())
    parent_protocol = json.loads((parent / "protocol.json").read_text())
    paths = {
        "auxiliary": root / "overnight_unfiltered_auxiliary_v1/dataset.parquet",
        "factors": root / "overnight_unfiltered_full_factors_v1/dataset.parquet",
        "labels": root / "overnight_unfiltered_labels_v1/labels.parquet",
        "gate": root / "overnight_unfiltered_entry_gate_v1/entry_gate.parquet",
        "old": root / "tail_overnight/dataset.parquet",
    }
    hashes = {str(p): sha256(p) for p in paths.values()}
    for path, digest in hashes.items():
        if parent_protocol["sources"][path] != digest:
            raise ValueError("Training input changed")
    for arm in ["old", "expanded"]:
        for name in [f"{arm}.joblib", f"{arm}_predictions.parquet"]:
            path = parent / name
            hashes[str(path)] = sha256(path)
    output = root / "overnight_unfiltered_return_calibration_v1"
    output.mkdir(exist_ok=False)
    protocol = {"method": "date-equal weighted isotonic, fit Q3 raw20bps net; no model refit",
                "calibration": "2025-07-01<=entry<2025-10-01 and exit<2025-10-01, matching each training pool",
                "hurdles": {"cover20bps": 0., "cover40bps": .002},
                "selection": "calibrated expected net strictly above hurdle; frozen probability>=.7,risk<=.1,top1",
                "ranking": "Keep original probability/raw expected net/symbol order; calibration only gates",
                "status": "Reused historical diagnostic, not independent confirmation; no fit using Q4 or2026",
                "sources": hashes, "script_sha256": sha256(Path(__file__))}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    filters = [("date", ">=", pd.Timestamp("2025-07-01")), ("date", "<", pd.Timestamp("2025-10-01"))]
    keys = ["symbol", "date"]
    frame = pd.read_parquet(paths["auxiliary"], filters=filters)
    factors = pd.read_parquet(paths["factors"], columns=[*keys, *PROJECT_FACTOR_COLUMNS], filters=filters)
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(factors, on=keys, validate="one_to_one")
    labels = pd.read_parquet(paths["labels"], columns=[*keys, "exit_date", "limit05"], filters=filters)
    gate = pd.read_parquet(paths["gate"], columns=[*keys, "eligible"], filters=filters)
    frame = frame.merge(labels, on=keys, validate="one_to_one").merge(gate, on=keys, validate="one_to_one")
    frame = frame.loc[frame.eligible & frame.exit_date.lt("2025-10-01") & frame.limit05.notna()].copy()
    old = pd.read_parquet(paths["old"], columns=keys, filters=filters).assign(old_pool=True)
    frame = frame.merge(old, on=keys, how="left", validate="one_to_one")
    frame["old_pool"] = frame.old_pool.eq(True)
    del factors, labels, gate, old
    summaries, counts = [], []
    for arm in ["old", "expanded"]:
        model = joblib.load(parent / f"{arm}.joblib")
        if model["columns"] != parent_protocol["features"]:
            raise ValueError("Model feature schema changed")
        cal = frame.loc[frame.old_pool].copy() if arm == "old" else frame.copy()
        x = cal[model["columns"]].astype("float32").replace([np.inf, -np.inf], np.nan)
        with threadpool_limits(limits=4):
            scores = model["regressor"].predict(x)
        fit = cal[[*keys, "exit_date"]].assign(expected_net=scores, net=cal.limit05)
        calibrator = fit_return_calibrator(fit)
        fit.to_parquet(output / f"{arm}_calibration_rows.parquet", index=False)
        joblib.dump(calibrator, output / f"{arm}_return_calibrator.joblib")
        pred = pd.read_parquet(parent / f"{arm}_predictions.parquet")
        pred["calibrated_expected_net"] = calibrator.predict(pred.expected_net)
        pred.to_parquet(output / f"{arm}_predictions.parquet", index=False)
        counts.append({"arm": arm, "calibration_rows": len(fit), "knots": len(calibrator.X_thresholds_),
                       "calibrated_min": float(pred.calibrated_expected_net.min()),
                       "calibrated_max": float(pred.calibrated_expected_net.max())})
        for population in ["old", "expanded"]:
            candidates = pred.loc[pred.old_pool] if population == "old" else pred
            for name, hurdle in protocol["hurdles"].items():
                picks = select(candidates.loc[candidates.calibrated_expected_net.gt(hurdle)], .7, -np.inf, 1, .1)
                picks.to_parquet(output / f"{arm}_{population}_{name}_picks.parquet", index=False)
                for period, lower, upper in [("2025Q4", "2025-10-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-04")]:
                    part = picks.loc[picks.date.ge(lower) & picks.date.lt(upper)].copy()
                    days = pd.DatetimeIndex(pred.loc[pred.date.ge(lower) & pred.date.lt(upper), "date"].unique()).sort_values()
                    for view in ["raw_mark", "exit_audited"]:
                        val = part.copy()
                        if view == "exit_audited":
                            val["net"] = val.limit05_net_after_exit_audit
                        for cost in [20, 40]:
                            scored = val.copy()
                            scored["net"] -= (cost - 20) / 10000
                            summaries.append({"train_pool": arm, "selection_pool": population, "hurdle": name,
                                "period": period, "view": view, "cost_bps": cost, **metrics(scored, days, 1)})
        print(json.dumps(counts[-1]), flush=True)
    pd.DataFrame(summaries).to_csv(output / "diagnostics.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Input changed during calibration")
    result = {"calibration": counts, "diagnostic_rows": len(summaries), "status": "Diagnostic complete; goal not declared achieved"}
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
