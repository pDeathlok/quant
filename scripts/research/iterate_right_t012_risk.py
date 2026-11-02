"""Badcase-driven downside head; keep baseline event/return predictions frozen."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_unfiltered_comparison import date_weights
from train_right_t012 import KEYS, choose, summarize

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "right_t012_baseline_v2"
    output = root / "right_t012_risk_v1"
    parent = json.loads((source / "protocol.json").read_text())
    complete = json.loads((source / "complete.json").read_text())
    if sha256(source / "picks.parquet") != complete["picks_sha256"]:
        raise ValueError("Baseline picks changed")
    for name, digest in parent["sources"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Source changed: {name}")
    output.mkdir(exist_ok=False)
    protocol = dict(parent_protocol_sha256=sha256(source / "protocol.json"),
                    parent_predictions_sha256=sha256(source / "predictions.parquet"),
                    hypothesis="September limit1 losses average2.8793% vs winners0.8118%; model downside instead of only hit probability",
                    target="P(net20 < -2%)", maximum_risk=.1,
                    selection="Keep baseline September-selected probability threshold/rank and expected_net>0; risk<=.1 before dailyTop1",
                    scope="Only limit1; unchanged base predictions, features and model parameters",
                    status="Historical development iteration, Q4 already observed; not independent validation",
                    script_sha256=sha256(Path(__file__)))
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    pred = pd.read_parquet(source / "predictions.parquet")
    pred = pred.loc[pred.target.eq("limit1")].reset_index(drop=True)
    features = pd.read_parquet(root / "overnight_unfiltered_auxiliary_v1/dataset.parquet",
        filters=[("date", ">=", pd.Timestamp("2025-07-01")), ("date", "<", pd.Timestamp("2026-01-01"))])
    factors = pd.read_parquet(root / "overnight_unfiltered_full_factors_v1/dataset.parquet",
        columns=KEYS + list(PROJECT_FACTOR_COLUMNS),
        filters=[("date", ">=", pd.Timestamp("2025-07-01")), ("date", "<", pd.Timestamp("2026-01-01"))])
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = pred.merge(features, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(factors, on=KEYS, how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(pred[KEYS], frame[KEYS])
    x = frame[parent["features"]].astype("float32").replace([np.inf, -np.inf], np.nan)
    fit = pred.split.eq("train") & pred.net.notna()
    cal = pred.split.eq("calibration") & pred.net.notna()
    risk_y = pred.net.lt(-.02)
    baseline = joblib.load(source / "limit1.joblib")
    with threadpool_limits(limits=4):
        if not np.array_equal(baseline["calibrator"].predict(baseline["classifier"].predict_proba(x)[:, 1]), pred.probability):
            raise ValueError("Baseline feature/prediction replay differs")
        model = HistGradientBoostingClassifier(**parent["params"]).fit(x[fit], risk_y[fit], sample_weight=date_weights(frame[fit]))
        iso = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(x[cal])[:, 1], risk_y[cal], sample_weight=date_weights(frame[cal]))
        pred["risk"] = iso.predict(model.predict_proba(x)[:, 1])
    joblib.dump(dict(classifier=model, calibrator=iso, features=parent["features"]), output / "risk.joblib")
    pred.to_parquet(output / "predictions.parquet", index=False)
    grid = pd.read_csv(source / "calibration_grid.csv")
    accepted = grid.loc[grid.target.eq("limit1") & grid.dates.ge(10)]
    winner = accepted.sort_values(["daily_score", "threshold", "rank"], ascending=[False, True, True]).iloc[0]
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "overnight_unfiltered_labels_v1/labels.parquet", columns=["date"]).date.unique()).sort_values()
    rows, picks = [], []
    for split in ["calibration", "q4_diagnostic"]:
        subset = pred.loc[pred.split.eq(split) & pred.risk.le(.1)]
        picked = choose(subset, float(winner.threshold), str(winner["rank"]))
        picks.append(picked)
        start, end = ("2025-09-01", "2025-09-30") if split == "calibration" else ("2025-10-01", "2025-12-31")
        days = calendar[(calendar >= start) & (calendar < end)]
        for cost in [20, 40]:
            rows.append(dict(split=split, cost_bps=cost,
                             **summarize(picked.assign(net=picked.net - (cost - 20) / 10000), days)))
    picks_frame = pd.concat(picks, ignore_index=True)
    picks_frame.to_parquet(output / "picks.parquet", index=False)
    picks_frame.loc[picks_frame.net.isna() | picks_frame.net.le(0)].to_parquet(output / "badcases.parquet", index=False)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    if sha256(source / "predictions.parquet") != protocol["parent_predictions_sha256"]:
        raise ValueError("Parent predictions changed")
    for name, digest in parent["sources"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Source changed during fit: {name}")
    (output / "complete.json").write_text(json.dumps(dict(picks_sha256=sha256(output / "picks.parquet"),
        baseline_prediction_replay="exact", features=len(parent["features"]), status="Risk ablation complete"), indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
