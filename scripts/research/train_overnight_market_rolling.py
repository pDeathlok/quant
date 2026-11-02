"""Quarterly expanding-window refits with one fixed overnight selection rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select
from train_overnight_payoff_heads import select_net

from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
FOLDS = [("2025-10-01", "2026-01-01", "2026-04-01"),
         ("2026-01-01", "2026-04-01", "2026-07-01"),
         ("2026-04-01", "2026-07-01", "2026-09-05")]


def split_masks(frame: pd.DataFrame, calibration_start: str, test_start: str,
                test_end: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    train = frame.date.lt(calibration_start) & frame.exit_date.lt(calibration_start)
    cal = frame.date.ge(calibration_start) & frame.date.lt(test_start) & frame.exit_date.lt(test_start)
    # A test trade can exit in the next quarter; it is never used before its
    # label is observed. Keep the same full-year test keys as the frozen run.
    test = frame.date.ge(test_start) & frame.date.lt(test_end) & frame.exit_date.le("2026-09-04")
    if (train & cal | train & test | cal & test).any():
        raise ValueError("Overlapping rolling partitions")
    return train, cal, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted-payoff", action="store_true")
    parser.add_argument("--rolling-flow-risk", action="store_true")
    args = parser.parse_args()
    if args.rolling_flow_risk and args.weighted_payoff:
        parser.error("Choose one auxiliary experiment")
    parent = ROOT / "reports/research/overnight_same_day_market_limit_gated_v2"
    fixed = ROOT / "reports/research/overnight_market_risk_2026_check_v1"
    output = ROOT / "reports/research/overnight_market_rolling_2026_v1"
    if args.weighted_payoff:
        fixed = ROOT / "reports/research/overnight_weighted_profit_2026_check_v1"
        output = ROOT / "reports/research/overnight_weighted_rolling_2026_v1"
    if args.rolling_flow_risk:
        fixed = ROOT / "reports/research/overnight_moneyflow_rolling_risk_2026_v1"
        output = ROOT / "reports/research/overnight_moneyflow_quarterly_2026_v1"
    top_k = 3 if args.weighted_payoff else 1

    def pick_rule(pred: pd.DataFrame) -> pd.DataFrame:
        if args.weighted_payoff:
            ranked = select_net(pred.assign(expected_net=pred.weighted_score), .8, -np.inf, top_k)
            return pred.loc[ranked.index].copy()
        return select(pred, .7, -np.inf, top_k, .1)

    output.mkdir(exist_ok=False)
    protocol = json.loads((parent / "protocol.json").read_text())
    source = ROOT / "reports/research/tail_overnight/dataset.parquet"
    factors_path = ROOT / "reports/research/overnight_current_full_factors_v1/dataset.parquet"
    raw_path = fixed / "same_day_raw_snapshot.parquet"
    gate_path = fixed / "exact_entry_gate.parquet"
    label_path = parent / "limit_label_source.parquet"
    context_path = parent / "market_context.parquet"
    paths = [source, factors_path, raw_path, gate_path, label_path, context_path, fixed / "predictions.parquet"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if args.rolling_flow_risk:
        flow_root = ROOT / "reports/research/overnight_moneyflow_rolling_v1"
        flow_path = flow_root / "dataset.parquet"
        flow_manifest = json.loads((flow_root / "complete.json").read_text())
        hashes[str(flow_path)] = hashlib.sha256(flow_path.read_bytes()).hexdigest()
        if hashes[str(flow_path)] != flow_manifest["dataset_sha256"]:
            raise ValueError("Flow artifact changed")
        baseline_root = ROOT / "reports/research/overnight_market_rolling_2026_v1"
        for number in range(1, 4):
            for name in (f"fold{number}.joblib", f"fold{number}_predictions.parquet"):
                path = baseline_root / name
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = {"folds": FOLDS, "rule": {"target": "limit05", "probability": .7, "maximum_risk": .1,
            "top_k": 1, "minimum_expected_net": None}, "params": protocol["params"],
            "training": "expanding from2024; previous quarter calibration; strict target-date embargo",
            "status": "reused2026 diagnostic, NOT independent confirmation; no threshold search",
            "costs_bps": [20, 40], "execution": "EOD marks with14:30 approximation, not verified fills",
            "sources": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.weighted_payoff:
        lock["rule"] = {"target": "limit05", "probability": .8, "top_k": top_k,
                        "minimum_payoff_share": None, "ranking": "weighted payoff share, probability, symbol"}
        lock["auxiliary_target"] = "net>0 with abs(net) magnitude weights; weighted previous-quarter calibration"
    if args.rolling_flow_risk:
        lock["auxiliary_target"] = "net < -2%; quarterly refit with 24 extra exact-calendar flow features"
        lock["profit_models"] = "Reuse baseline quarterly models unchanged and replay predictions"
    (output / "LOCK.json").write_text(json.dumps(lock, indent=2))
    cols = protocol["features"]
    lag_cols = [c for c in cols if c.startswith("lag1_")]
    f = pd.read_parquet(source, columns=["symbol", "date", "exit_date", "entry_policy_eligible", *lag_cols])
    factors = pd.read_parquet(factors_path, columns=["symbol", "date", *PROJECT_FACTOR_COLUMNS])
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    f = f.merge(factors, on=["symbol", "date"], how="left", validate="one_to_one")
    del factors
    f = f.merge(pd.read_parquet(raw_path), on=["symbol", "date"], how="left", validate="one_to_one")
    f = f.merge(pd.read_parquet(context_path), on="date", how="left", validate="many_to_one")
    labels = pd.read_parquet(label_path, columns=["symbol", "date", "exit_date", "limit05"])
    pd.testing.assert_frame_equal(f[["symbol", "date", "exit_date"]], labels[["symbol", "date", "exit_date"]])
    calendar = f[["date", "exit_date"]].drop_duplicates()
    gate = pd.read_parquet(gate_path)
    pd.testing.assert_frame_equal(f[["symbol", "date"]], gate[["symbol", "date"]])
    f = f.loc[gate.eligible.to_numpy()].reset_index(drop=True)
    net = labels.loc[gate.eligible.to_numpy(), "limit05"].reset_index(drop=True)
    x = f[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    risk_cols = list(cols)
    risk_x = x
    if args.rolling_flow_risk:
        flow_cols = flow_manifest["features"]
        flow = pd.read_parquet(flow_path, columns=["symbol", "date", "flow_date", *flow_cols])
        keys_before = f[["symbol", "date"]].copy()
        f = f.merge(flow, on=["symbol", "date"], how="left", validate="one_to_one")
        pd.testing.assert_frame_equal(keys_before, f[["symbol", "date"]])
        if f.flow_date.isna().any() or not f.flow_date.lt(f.date).all():
            raise ValueError("Invalid flow key/timing")
        risk_cols += flow_cols
        risk_x = f[risk_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    predictions, summaries = [], []
    for number, (cal_start, test_start, test_end) in enumerate(FOLDS, 1):
        train, cal, test = split_masks(f, cal_start, test_start, test_end)
        train &= net.notna()
        cal &= net.notna()
        weights = 1 / f.loc[train].groupby("date").date.transform("size")
        weights /= weights.mean()
        cal_weights = 1 / f.loc[cal].groupby("date").date.transform("size")
        for mask in [train, cal]:
            if net[mask].gt(0).nunique() != 2 or net[mask].lt(-.02).nunique() != 2:
                raise ValueError("Missing profit or loss class")
        print(f"fold{number}: train={train.sum()} cal={cal.sum()} test={test.sum()}", flush=True)
        with threadpool_limits(limits=4):
            if args.rolling_flow_risk:
                baseline = joblib.load(baseline_root / f"fold{number}.joblib")
                if baseline["columns"] != cols:
                    raise ValueError("Baseline profit columns differ")
                clf, iso = baseline["classifier"], baseline["calibrator"]
            else:
                clf = HistGradientBoostingClassifier(**protocol["params"]).fit(x.loc[train], net[train].gt(0), sample_weight=weights)
                iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(x.loc[cal])[:, 1], net[cal].gt(0), sample_weight=cal_weights)
            reg = None
            if args.weighted_payoff:
                magnitude_weights = weights * net[train].abs()
                magnitude_weights /= magnitude_weights.mean()
                risk = HistGradientBoostingClassifier(**protocol["params"]).fit(
                    x.loc[train], net[train].gt(0), sample_weight=magnitude_weights)
                risk_iso = IsotonicRegression(out_of_bounds="clip").fit(
                    risk.predict_proba(x.loc[cal])[:, 1], net[cal].gt(0), sample_weight=cal_weights * net[cal].abs())
            else:
                reg = baseline["regressor"] if args.rolling_flow_risk else HistGradientBoostingRegressor(**protocol["params"]).fit(x.loc[train], net[train], sample_weight=weights)
                risk = HistGradientBoostingClassifier(**protocol["params"]).fit(risk_x.loc[train], net[train].lt(-.02), sample_weight=weights)
                risk_iso = IsotonicRegression(out_of_bounds="clip").fit(risk.predict_proba(risk_x.loc[cal])[:, 1], net[cal].lt(-.02), sample_weight=cal_weights)
            pred = f.loc[test, ["symbol", "date", "exit_date", "entry_policy_eligible"]].copy()
            pred["net"] = net[test]
            pred["probability"] = iso.predict(clf.predict_proba(x.loc[test])[:, 1])
            if reg is not None:
                pred["expected_net"] = reg.predict(x.loc[test])
            auxiliary_column = "weighted_score" if args.weighted_payoff else "risk"
            pred[auxiliary_column] = risk_iso.predict(risk.predict_proba(risk_x.loc[test])[:, 1])
            if args.rolling_flow_risk:
                prior = pd.read_parquet(baseline_root / f"fold{number}_predictions.parquet")
                replay_cols = ["symbol", "date", "exit_date", "net", "probability", "expected_net"]
                pd.testing.assert_frame_equal(pred[replay_cols].reset_index(drop=True), prior[replay_cols].reset_index(drop=True))
                pred["flow_window_complete"] = f.loc[test, flow_cols].notna().all(axis=1).to_numpy()
        pred.to_parquet(output / f"fold{number}_predictions.parquet", index=False)
        bundle = {"classifier": clf, "regressor": reg, "calibrator": iso, "columns": cols}
        if args.rolling_flow_risk:
            bundle["risk_columns"] = risk_cols
        bundle.update({("weighted_classifier" if args.weighted_payoff else "risk"): risk,
                       ("weighted_calibrator" if args.weighted_payoff else "risk_calibrator"): risk_iso})
        joblib.dump(bundle, output / f"fold{number}.joblib")
        picks = pick_rule(pred)
        _, _, day_mask = split_masks(calendar, cal_start, test_start, test_end)
        days = pd.DatetimeIndex(calendar.loc[day_mask, "date"].unique()).sort_values()
        summary = {"fold": number, "train_known": int(train.sum()), "cal_known": int(cal.sum()),
                   "cost20": metrics(picks, days, top_k), "cost40": metrics(picks.assign(net=picks.net - .002), days, top_k)}
        summaries.append(summary)
        predictions.append(pred)
        (output / f"fold{number}_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    pred = pd.concat(predictions).sort_values(["date", "symbol"]).reset_index(drop=True)
    old = pd.read_parquet(fixed / "predictions.parquet").sort_values(["date", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(pred[["symbol", "date", "exit_date", "net"]], old[["symbol", "date", "exit_date", "net"]])
    picks = pick_rule(pred)
    days = pd.DatetimeIndex(calendar.loc[calendar.date.ge("2026-01-01") & calendar.exit_date.le("2026-09-04"), "date"].unique()).sort_values()
    pred.to_parquet(output / "predictions.parquet", index=False)
    picks.to_parquet(output / "selected.parquet", index=False)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("Rolling input changed")
    result = {"folds": summaries, "same_frozen_test_keys_and_labels": True,
              "cost20": metrics(picks, days, top_k), "cost40": metrics(picks.assign(net=picks.net - .002), days, top_k)}
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
