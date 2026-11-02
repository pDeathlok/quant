"""Fixed sigmoid-vs-isotonic chronological diagnostic; no test-set tuning."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select
from train_overnight_unfiltered_comparison import date_weights

from quant.core.paths import PROJECT_ROOT


def logit_input(scores: pd.Series) -> np.ndarray:
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Invalid probability scores")
    clipped = scores.clip(1e-6, 1 - 1e-6).to_numpy()
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_sigmoid(scored: pd.DataFrame, boundary: str) -> dict[str, LogisticRegression]:
    fit = scored.loc[scored.date.ge("2025-07-01") & scored.date.lt(boundary)
                     & scored.exit_date.lt(boundary) & np.isfinite(scored.net)]
    if fit.empty or not fit.exit_date.gt(fit.date).all():
        raise ValueError("Invalid calibration timing")
    result = {}
    for key, column, outcome in [("probability", "raw_probability", fit.net.gt(0)),
                                  ("risk", "raw_risk", fit.net.lt(-.02))]:
        if outcome.nunique() != 2:
            raise ValueError("Insufficient calibration classes")
        model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, tol=1e-8)
        model.fit(logit_input(fit[column]), outcome, sample_weight=date_weights(fit))
        if model.n_iter_[0] >= 1000 or model.coef_[0, 0] <= 0:
            raise ValueError("Nonconverged or reversed calibration")
        result[key] = model
    return result


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "overnight_chronological_calibration_v1"
    output = root / "overnight_sigmoid_calibration_v1"
    output.mkdir(exist_ok=False)
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["change"] = "Fixed logistic on clipped logit score; C=1, date-equal weights; both probability and risk"
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    hashes = {str(source / "protocol.json"): sha256(source / "protocol.json")}
    rows, quality = [], []
    for target in ["net_open", "net_close"]:
        for calendar in [False, True]:
            suffix = "hgb_calendar" if calendar else "comparison"
            path = source / f"{target}_{suffix}_q3.parquet"
            hashes[str(path)] = sha256(path)
            scored = pd.read_parquet(path)
            for lower, upper in protocol["folds"]:
                with threadpool_limits(limits=4):
                    models = fit_sigmoid(scored, lower)
                model_path = source / f"{target}_{suffix}_{lower}_calibrators.joblib"
                hashes[str(model_path)] = sha256(model_path)
                isotonic = joblib.load(model_path)
                validation = scored.loc[scored.date.ge(lower) & scored.date.lt(upper)
                                        & scored.exit_date.lt(upper)].copy()
                for key, column in [("probability", "raw_probability"), ("risk", "raw_risk")]:
                    validation[key] = models[key].predict_proba(logit_input(validation[column]))[:, 1]
                known = validation.net.notna()
                weights = date_weights(validation.loc[known])
                for key, column, outcome in [("probability", "raw_probability", validation.net.gt(0)),
                                              ("risk", "raw_risk", validation.net.lt(-.02))]:
                    for method, predicted in [("sigmoid", validation[key].to_numpy()),
                                               ("isotonic", isotonic[key].predict(validation[column]))]:
                        error = (predicted[known] - outcome[known].to_numpy()) ** 2
                        quality.append({"target": target, "calendar": calendar, "validation_start": lower,
                                        "head": key, "method": method, "date_weighted_brier": float(np.average(error, weights=weights))})
                days = pd.DatetimeIndex(validation.date.unique()).sort_values()
                picks = select(validation, .7, -np.inf, 1, .1)
                picks.to_parquet(output / f"{target}_{suffix}_{lower}_picks.parquet", index=False)
                joblib.dump(models, output / f"{target}_{suffix}_{lower}_calibrators.joblib")
                for cost in [20, 40]:
                    adjusted = picks.copy()
                    adjusted["net"] -= (cost - 20) / 10000
                    rows.append({"target": target, "calendar": calendar, "validation_start": lower,
                                 "cost_bps": cost, "maximum_probability": float(validation.probability.max()),
                                 **metrics(adjusted, days, 1)})
            print(f"{target} calendar={calendar}: completed", flush=True)
    pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)
    pd.DataFrame(quality).to_csv(output / "probability_quality.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Changed source: {path}")
    (output / "complete.json").write_text(json.dumps({"sources": hashes, "script_sha256": sha256(Path(__file__)),
        "status": "Chronological reused-history diagnostic complete; no independent confirmation", "metric_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
