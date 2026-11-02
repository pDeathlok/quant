"""One frozen Q4-selected rule across2026; explicitly not a fresh holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_lag_only import exact_entry_gate, same_day_features
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select
from train_overnight_payoff_heads import select_net

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted-payoff", action="store_true")
    parser.add_argument("--rolling-flow-risk", action="store_true")
    args = parser.parse_args()
    if args.rolling_flow_risk and args.weighted_payoff:
        parser.error("Choose only one frozen auxiliary model")
    parent = ROOT / "reports/research/overnight_same_day_market_limit_gated_v2"
    risk_root = ROOT / "reports/research/overnight_same_day_market_risk_v1"
    output = ROOT / "reports/research/overnight_market_risk_2026_check_v1"
    auxiliary_file = "limit05_risk.joblib"
    if args.weighted_payoff:
        risk_root = ROOT / "reports/research/overnight_market_weighted_profit_v1"
        output = ROOT / "reports/research/overnight_weighted_profit_2026_check_v1"
        auxiliary_file = "limit05.joblib"
    if args.rolling_flow_risk:
        risk_root = ROOT / "reports/research/overnight_moneyflow_rolling_risk_v1"
        output = ROOT / "reports/research/overnight_moneyflow_rolling_risk_2026_v1"
    if not (risk_root / "complete.json").exists():
        raise ValueError("Auxiliary model training unfinished")
    output.mkdir(exist_ok=False)
    protocol = json.loads((parent / "protocol.json").read_text())
    paths = [parent / "limit05.joblib", risk_root / auxiliary_file,
             parent / "market_context.parquet", parent / "limit_label_source.parquet",
             ROOT / "reports/research/tail_overnight/dataset.parquet"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if args.rolling_flow_risk:
        flow_root = ROOT / "reports/research/overnight_moneyflow_rolling_v1"
        flow_path = flow_root / "dataset.parquet"
        flow_manifest = json.loads((flow_root / "complete.json").read_text())
        hashes[str(flow_path)] = hashlib.sha256(flow_path.read_bytes()).hexdigest()
        if hashes[str(flow_path)] != flow_manifest["dataset_sha256"]:
            raise ValueError("Rolling flow changed")
    lock = {"rule": {"target": "limit05", "minimum_probability": .7, "maximum_risk": .1,
                     "top_k": 1, "minimum_expected_net": None}, "period": "2026-01-01 through2026-09-04",
            "fit": "No refit or recalibration; frozen2024-2025H1 training and2025Q3 calibrators",
            "status": "cross-period diagnostic only;2026 already researched, NOT independent confirmation",
            "selection": "one rule fixed from Q4 diagnostic before2026 inference; no alternative rules evaluated",
            "entry": "current EOD exact upper-limit gate, including genuine volume alias fallback",
            "costs": [20, 40], "execution": "daily marks only;2026 exit limit/fill audit still required",
            "sources": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.weighted_payoff:
        lock["rule"] = {"target": "limit05", "minimum_probability": .8, "top_k": 3,
                        "minimum_payoff_share": None, "ranking": "weighted payoff share, probability, symbol"}
    (output / "LOCK.json").write_text(json.dumps(lock, indent=2))
    lag_cols = [c for c in protocol["features"] if c.startswith("lag1_")]
    f = pd.read_parquet(paths[-1], columns=["symbol", "date", "exit_date", "entry_policy_eligible", *lag_cols])
    labels = pd.read_parquet(parent / "limit_label_source.parquet", columns=["symbol", "date", "exit_date", "limit05"])
    pd.testing.assert_frame_equal(f[["symbol", "date", "exit_date"]], labels[["symbol", "date", "exit_date"]])
    f, cols = same_day_features(f, output, rebuilt=True)
    context = pd.read_parquet(parent / "market_context.parquet")
    market_cols = protocol["market_context_features"]
    f = f.merge(context[["date", *market_cols]], on="date", how="left", validate="many_to_one")
    cols.extend(market_cols)
    if cols != protocol["features"] or f[market_cols].isna().any().any():
        raise ValueError("Frozen feature contract mismatch")
    calendar = f[["date", "exit_date"]].drop_duplicates()
    gate = exact_entry_gate(f, output).to_numpy()
    f = f.loc[gate].reset_index(drop=True)
    net = labels.loc[gate, "limit05"].reset_index(drop=True)
    validation = f.date.ge("2025-10-01") & f.date.lt("2026-01-01") & f.exit_date.lt("2026-01-01")
    test = f.date.ge("2026-01-01") & f.exit_date.le("2026-09-04")
    x = f[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    auxiliary_cols = list(cols)
    auxiliary_x = x
    if args.rolling_flow_risk:
        flow_cols = flow_manifest["features"]
        flow = pd.read_parquet(flow_path, columns=["symbol", "date", "flow_date", *flow_cols])
        old_keys = f[["symbol", "date"]].copy()
        f = f.merge(flow, on=["symbol", "date"], how="left", validate="one_to_one")
        pd.testing.assert_frame_equal(old_keys, f[["symbol", "date"]])
        if f.flow_date.isna().any() or not f.flow_date.lt(f.date).all():
            raise ValueError("Missing or noncausal flow key")
        auxiliary_cols += flow_cols
        auxiliary_x = f[auxiliary_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    model = joblib.load(parent / "limit05.joblib")
    risk = joblib.load(risk_root / auxiliary_file)
    assert model["columns"] == cols and risk["columns"] == auxiliary_cols
    auxiliary_model = risk["models"]["classifier"] if args.weighted_payoff else risk["model"]
    auxiliary_calibrator = risk["models"]["calibrator"] if args.weighted_payoff else risk["calibrator"]
    auxiliary_column = "weighted_score" if args.weighted_payoff else "risk"
    with threadpool_limits(limits=4):
        old = pd.read_parquet(risk_root / "limit05_predictions.parquet")
        pd.testing.assert_frame_equal(old[["symbol", "date"]].reset_index(drop=True),
                                      f.loc[validation, ["symbol", "date"]].reset_index(drop=True))
        for field, predicted in [
            ("probability", model["calibrator"].predict(model["classifier"].predict_proba(x.loc[validation])[:, 1])),
            ("expected_net", model["regressor"].predict(x.loc[validation])),
            (auxiliary_column, auxiliary_calibrator.predict(auxiliary_model.predict_proba(auxiliary_x.loc[validation])[:, 1])),
        ]:
            np.testing.assert_allclose(old[field], predicted, rtol=1e-10, atol=1e-12)
        if args.weighted_payoff:
            prior_picks = select_net(old.assign(expected_net=old.weighted_score), .8, -np.inf, 3)
            prior_days = pd.DatetimeIndex(calendar.loc[calendar.date.ge("2025-10-01")
                & calendar.date.lt("2026-01-01") & calendar.exit_date.lt("2026-01-01"), "date"].unique()).sort_values()
            prior_stats = metrics(prior_picks, prior_days, 3)
            reference = pd.read_csv(risk_root / "limit05_diagnostics.csv")
            reference = reference[reference.ranker.eq("weighted_payoff_share") & reference.probability.eq(.8)
                                  & reference.top_k.eq(3) & reference.minimum_payoff_share.isna()].iloc[0]
            for key, value in prior_stats.items():
                np.testing.assert_allclose(value, reference[key], rtol=1e-10, atol=1e-12)
            prior_picks.to_parquet(output / "q4_replayed_selected.parquet", index=False)
        if args.rolling_flow_risk:
            prior_picks = select(old, .7, -np.inf, 1, .1)
            prior_days = pd.DatetimeIndex(calendar.loc[calendar.date.ge("2025-10-01")
                & calendar.date.lt("2026-01-01") & calendar.exit_date.lt("2026-01-01"), "date"].unique()).sort_values()
            reference = pd.read_csv(risk_root / "limit05_diagnostics.csv").iloc[0]
            for key, value in metrics(prior_picks, prior_days, 1).items():
                np.testing.assert_allclose(value, reference[key], rtol=1e-10, atol=1e-12)
            prior_picks.to_parquet(output / "q4_replayed_selected.parquet", index=False)
        pred = f.loc[test, ["symbol", "date", "exit_date", "entry_policy_eligible"]].copy()
        pred["net"] = net[test]
        pred["probability"] = model["calibrator"].predict(model["classifier"].predict_proba(x.loc[test])[:, 1])
        pred["expected_net"] = model["regressor"].predict(x.loc[test])
        pred[auxiliary_column] = auxiliary_calibrator.predict(auxiliary_model.predict_proba(auxiliary_x.loc[test])[:, 1])
        if args.rolling_flow_risk:
            pred["flow_window_complete"] = f.loc[test, flow_cols].notna().all(axis=1).to_numpy()
    top_k = 3 if args.weighted_payoff else 1
    picks = (select_net(pred.assign(expected_net=pred.weighted_score), .8, -np.inf, top_k)
             if args.weighted_payoff else select(pred, .7, -np.inf, top_k, .1))
    if args.weighted_payoff:
        # Keep daily-return regression units intact in the saved selected artifact.
        picks = pred.loc[picks.index].copy()
    pred.to_parquet(output / "predictions.parquet", index=False)
    picks.to_parquet(output / "selected.parquet", index=False)
    days = pd.DatetimeIndex(calendar.loc[calendar.date.ge("2026-01-01")
        & calendar.exit_date.le("2026-09-04"), "date"].unique()).sort_values()
    stress = picks.assign(net=picks.net - .002)
    summary = {"cost20": metrics(picks, days, top_k), "cost40": metrics(stress, days, top_k),
               "prior_validation_reproduced": True, "prediction_rows": len(pred),
               "limitations": lock["status"] + "; daily marks, not fills"}
    monthly = []
    for month in days.to_period("M").unique():
        day_subset = days[days.to_period("M") == month]
        selected = picks[picks.date.dt.to_period("M") == month]
        monthly.append({"month": str(month), **metrics(selected, day_subset, top_k)})
    pd.DataFrame(monthly).to_csv(output / "monthly_cost20.csv", index=False)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen source changed during cross-period check")
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
