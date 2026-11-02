"""Separate positive/negative payoff heads on frozen current-market inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def select_net(frame: pd.DataFrame, probability: float, minimum: float, top_k: int) -> pd.DataFrame:
    eligible = frame.probability.ge(probability) & frame.expected_net.ge(minimum)
    return (frame.loc[eligible].sort_values(["date", "expected_net", "probability", "symbol"],
                                           ascending=[True, False, False, True]).groupby("date").head(top_k))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted-profit", action="store_true")
    parser.add_argument("--moneyflow", action="store_true", help="Add exact T-1 flow only to weighted payoff ranker")
    parser.add_argument("--moneyflow-hit", action="store_true", help="Add flow only to hit classifier; freeze baseline payoff ranking")
    parser.add_argument("--rolling-severity", action="store_true", help="Add 24 rolling flow features to unconditional gain/loss heads")
    args = parser.parse_args()
    if args.rolling_severity:
        if args.weighted_profit or args.moneyflow_hit:
            parser.error("--rolling-severity is a regression experiment")
        args.moneyflow = True
    if args.moneyflow_hit and not (args.moneyflow and args.weighted_profit):
        parser.error("--moneyflow-hit requires --moneyflow --weighted-profit")
    if args.moneyflow and not (args.weighted_profit or args.rolling_severity):
        parser.error("--moneyflow requires --weighted-profit")
    parent = PROJECT_ROOT / "reports/research/overnight_same_day_market_limit_gated_v2"
    output = PROJECT_ROOT / "reports/research/overnight_market_payoff_heads_v1"
    if args.weighted_profit:
        output = PROJECT_ROOT / "reports/research/overnight_market_weighted_profit_v1"
    if args.moneyflow:
        output = PROJECT_ROOT / "reports/research/overnight_moneyflow_weighted_profit_v1"
    if args.moneyflow_hit:
        output = PROJECT_ROOT / "reports/research/overnight_moneyflow_hit_v1"
    if args.rolling_severity:
        output = PROJECT_ROOT / "reports/research/overnight_moneyflow_rolling_severity_v1"
    if not (parent / "complete.json").exists():
        raise ValueError("Parent incomplete")
    output.mkdir(exist_ok=False)
    protocol = json.loads((parent / "protocol.json").read_text())
    protocol.update(new_targets=["max(net,0)", "max(-net,0)"],
                    experiment="Two unconditional squared-error regressions, subtract nonnegative predictions",
                    rank_order="expected_net descending, probability descending, symbol ascending",
                    rankers=["direct_regression", "payoff_difference"],
                    probability_thresholds=[0., .8], minimum_net=[None, 0., .002], top_k=[1, 3, 5],
                    scope="Frozen legacy broad candidate pool, NOT the full rebuilt right-side signal universe",
                    confirmation="Q4 reused research validation only; no2026 inference or independent claim",
                    script_sha256=sha256(Path(__file__)))
    if args.weighted_profit:
        protocol.update(new_targets=["net>0, weighted by abs(net) times inverse date sample count"],
                        experiment="Payoff-magnitude-weighted classification; Q3 weighted isotonic calibration",
                        rankers=["direct_regression", "weighted_payoff_share"],
                        minimum_payoff_share=[None, .5],
                        score_semantics="E[max(net,0)] / E[abs(net)], NOT hit probability or predicted return")
    if args.moneyflow:
        protocol.update(rankers=["weighted_payoff_share"], probability_thresholds=[.8],
                        minimum_payoff_share=[None], top_k=[3],
                        ablation="Only payoff ranker gains six T-1 flow ratios; hit-probability gate remains frozen",
                        missing_policy="Preserve all keys and NaN; no coverage-based exclusion")
    if args.moneyflow_hit:
        protocol.update(ablation="Only hit classifier gains six T-1 flow ratios; baseline weighted payoff model frozen",
                        experiment="Date-balanced hit classification, Q3 date-balanced isotonic; frozen payoff ranking",
                        new_targets=["net>0 weighted by inverse date sample count"])
    if args.rolling_severity:
        protocol.update(rankers=["payoff_difference"], minimum_net=[None],
                        ablation="24 rolling flow features added to both unconditional gain and loss regressions; profit gate frozen")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    dataset = PROJECT_ROOT / "reports/research/tail_overnight/dataset.parquet"
    if sha256(dataset) != protocol["source_sha256"]:
        raise ValueError("Changed candidate source")
    paths = [dataset, parent / "protocol.json", parent / "same_day_raw_snapshot.parquet",
             parent / "market_context.parquet", parent / "limit_label_source.parquet",
             parent / "exact_entry_gate.parquet",
             PROJECT_ROOT / "reports/research/overnight_current_full_factors_v1/dataset.parquet"]
    hashes = {str(p): sha256(p) for p in paths}
    for record in json.loads((parent / "same_day_sources.json").read_text()):
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Parent factor source changed")
        hashes[record["path"]] = record["sha256"]
    columns = protocol["features"]
    lag = [c for c in columns if c.startswith("lag1_")]
    frame = pd.read_parquet(dataset, columns=["symbol", "date", "feature_date", "exit_date", *lag])
    factors = pd.read_parquet(paths[-1], columns=["symbol", "date", *PROJECT_FACTOR_COLUMNS])
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(factors, on=["symbol", "date"], how="left", validate="one_to_one")
    del factors
    frame = frame.merge(pd.read_parquet(parent / "same_day_raw_snapshot.parquet"),
                        on=["symbol", "date"], how="left", validate="one_to_one")
    context = pd.read_parquet(parent / "market_context.parquet")
    market_columns = protocol["market_context_features"]
    frame = frame.merge(context[["date", *market_columns]], on="date", how="left", validate="many_to_one")
    if frame[market_columns].isna().any().any() or not frame.feature_date.lt(frame.date).all():
        raise ValueError("Feature date/context error")
    base_columns = list(columns)
    if args.moneyflow:
        flow_root = PROJECT_ROOT / "reports/research/overnight_moneyflow_lag_v1"
        if args.rolling_severity:
            flow_root = PROJECT_ROOT / "reports/research/overnight_moneyflow_rolling_v1"
        flow_path = flow_root / "dataset.parquet"
        flow_manifest = json.loads((flow_root / "complete.json").read_text())
        if sha256(flow_path) != flow_manifest["dataset_sha256"]:
            raise ValueError("Moneyflow source changed")
        hashes[str(flow_path)] = flow_manifest["dataset_sha256"]
        flow_columns = flow_manifest["features"]
        flow = pd.read_parquet(flow_path, columns=["symbol", "date", "flow_date", *flow_columns])
        if not flow.flow_date.lt(flow.date).all():
            raise ValueError("Moneyflow timing error")
        previous_keys = frame[["symbol", "date"]].copy()
        pd.testing.assert_frame_equal(
            flow[["symbol", "date"]].sort_values(["date", "symbol"]).reset_index(drop=True),
            previous_keys.sort_values(["date", "symbol"]).reset_index(drop=True))
        frame = frame.merge(flow.drop(columns="flow_date"), on=["symbol", "date"], how="left", validate="one_to_one")
        pd.testing.assert_frame_equal(frame[["symbol", "date"]], previous_keys)
        if len(flow) != len(frame):
            raise ValueError("Moneyflow candidate universe changed")
        columns = [*base_columns, *flow_columns]
        protocol.update(features=columns, baseline_features=base_columns, moneyflow_sha256=hashes[str(flow_path)])
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    labels = pd.read_parquet(parent / "limit_label_source.parquet")
    pd.testing.assert_frame_equal(frame[["symbol", "date", "exit_date"]], labels[["symbol", "date", "exit_date"]])
    gate = pd.read_parquet(parent / "exact_entry_gate.parquet")
    pd.testing.assert_frame_equal(frame[["symbol", "date"]], gate[["symbol", "date"]])
    calendar = frame[["date", "exit_date"]].drop_duplicates()
    keep = gate.eligible & frame.date.lt("2026-01-01")
    frame = frame.loc[keep].reset_index(drop=True)
    labels = labels.loc[keep].reset_index(drop=True)
    x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
    fit = frame.date.lt("2025-07-01") & frame.exit_date.lt("2025-07-01")
    calibration = frame.date.ge("2025-07-01") & frame.exit_date.lt("2025-10-01")
    validation = frame.date.ge("2025-10-01") & frame.exit_date.lt("2026-01-01")
    days = pd.DatetimeIndex(calendar.loc[calendar.date.ge("2025-10-01") & calendar.date.lt("2026-01-01")
                                       & calendar.exit_date.lt("2026-01-01"), "date"].unique()).sort_values()
    summaries = []
    for target in protocol["targets"]:
        net = labels[target]
        train = fit & net.notna()
        weights = 1 / frame.loc[train].groupby("date").date.transform("size")
        weights /= weights.mean()
        base_path, model_path = parent / f"{target}_predictions.parquet", parent / f"{target}.joblib"
        hashes[str(base_path)], hashes[str(model_path)] = sha256(base_path), sha256(model_path)
        base = pd.read_parquet(base_path)
        pd.testing.assert_frame_equal(base[["symbol", "date"]].reset_index(drop=True),
                                      frame.loc[validation, ["symbol", "date"]].reset_index(drop=True))
        np.testing.assert_allclose(base.net, net[validation], equal_nan=True, rtol=1e-10, atol=1e-12)
        parent_model = joblib.load(model_path)
        if parent_model["columns"] != base_columns:
            raise ValueError("Parent feature contract changed")
        models, predictions = {}, {}
        with threadpool_limits(limits=4):
            np.testing.assert_allclose(parent_model["regressor"].predict(x.loc[validation, base_columns]), base.expected_net,
                                       rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(parent_model["calibrator"].predict(
                parent_model["classifier"].predict_proba(x.loc[validation, base_columns])[:, 1]), base.probability,
                rtol=1e-10, atol=1e-12)
            print(f"{target}: parent replay passed, fit {int(train.sum())} rows / {len(columns)} features", flush=True)
            if args.moneyflow_hit:
                baseline_root = PROJECT_ROOT / "reports/research/overnight_market_weighted_profit_v1"
                frozen_path = baseline_root / f"{target}.joblib"
                frozen_predictions_path = baseline_root / f"{target}_predictions.parquet"
                hashes[str(frozen_path)] = sha256(frozen_path)
                hashes[str(frozen_predictions_path)] = sha256(frozen_predictions_path)
                frozen = joblib.load(frozen_path)
                old = pd.read_parquet(frozen_predictions_path)
                pd.testing.assert_frame_equal(base, old[base.columns])
                if frozen["columns"] != base_columns:
                    raise ValueError("Frozen payoff columns changed")
                frozen_raw = frozen["models"]["classifier"].predict_proba(x.loc[validation, base_columns])[:, 1]
                frozen_score = frozen["models"]["calibrator"].predict(frozen_raw)
                np.testing.assert_allclose(frozen_raw, old.weighted_raw, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(frozen_score, old.weighted_score, rtol=1e-10, atol=1e-12)
                cm = calibration & net.notna()
                cal_weights = 1 / frame.loc[cm].groupby("date").date.transform("size")
                model = HistGradientBoostingClassifier(**protocol["params"])
                model.fit(x.loc[train], net[train].gt(0), sample_weight=weights)
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    model.predict_proba(x.loc[cm])[:, 1], net[cm].gt(0), sample_weight=cal_weights)
                predictions.update(weighted_raw=frozen_raw, weighted_score=frozen_score,
                                   hit_probability=calibrator.predict(model.predict_proba(x.loc[validation])[:, 1]))
                models.update(hit_classifier=model, hit_calibrator=calibrator,
                              frozen_payoff_model_path=str(frozen_path), frozen_payoff_columns=base_columns)
                print(f"{target}: frozen payoff replay passed; moneyflow hit model completed", flush=True)
            elif args.weighted_profit:
                magnitude_weights = weights * net[train].abs()
                magnitude_weights /= magnitude_weights.mean()
                cm = calibration & net.notna()
                cal_weights = net[cm].abs() / frame.loc[cm].groupby("date").date.transform("size")
                if (net[train].gt(0).nunique() != 2 or net[cm].gt(0).nunique() != 2
                        or not magnitude_weights.gt(0).any() or not cal_weights.gt(0).any()):
                    raise ValueError("Insufficient payoff classes or calibration weights")
                model = HistGradientBoostingClassifier(**protocol["params"])
                model.fit(x.loc[train], net[train].gt(0), sample_weight=magnitude_weights)
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(
                    model.predict_proba(x.loc[cm])[:, 1], net[cm].gt(0), sample_weight=cal_weights)
                predictions["weighted_raw"] = model.predict_proba(x.loc[validation])[:, 1]
                predictions["weighted_score"] = calibrator.predict(predictions["weighted_raw"])
                models.update(classifier=model, calibrator=calibrator)
                print(f"{target}: magnitude-weighted model and Q3 calibration completed", flush=True)
            else:
                if args.rolling_severity:
                    control_root = PROJECT_ROOT / "reports/research/overnight_market_payoff_heads_v1"
                    control_path = control_root / f"{target}.joblib"
                    control_pred_path = control_root / f"{target}_predictions.parquet"
                    hashes[str(control_path)], hashes[str(control_pred_path)] = sha256(control_path), sha256(control_pred_path)
                    control = joblib.load(control_path)
                    control_pred = pd.read_parquet(control_pred_path)
                    pd.testing.assert_frame_equal(base, control_pred[base.columns])
                    if control["columns"] != base_columns:
                        raise ValueError("Gain/loss control columns changed")
                    for name in ("gain", "loss"):
                        np.testing.assert_allclose(np.maximum(control["models"][name].predict(x.loc[validation, base_columns]), 0),
                            control_pred[f"predicted_{name}"], rtol=1e-10, atol=1e-12)
                    print(f"{target}: baseline gain/loss replay passed", flush=True)
                for name, response in (("gain", net.clip(lower=0)), ("loss", (-net).clip(lower=0))):
                    model = HistGradientBoostingRegressor(**protocol["params"])
                    model.fit(x.loc[train], response[train], sample_weight=weights)
                    predictions[name] = np.maximum(model.predict(x.loc[validation]), 0)
                    models[name] = model
                    print(f"{target}: {name} model completed", flush=True)
        pred = base.copy()
        if args.moneyflow:
            pred["moneyflow_available"] = frame.loc[validation, flow_columns].notna().all(axis=1).to_numpy()
        if args.weighted_profit:
            pred["weighted_raw"] = predictions["weighted_raw"]
            pred["weighted_score"] = predictions["weighted_score"]
        else:
            pred["predicted_gain"] = predictions["gain"]
            pred["predicted_loss"] = predictions["loss"]
            pred["payoff_net"] = pred.predicted_gain - pred.predicted_loss
        pd.testing.assert_frame_equal(base, pred[base.columns])
        if args.moneyflow_hit:
            pred["baseline_probability"] = pred.probability
            pred["probability"] = predictions["hit_probability"]
        pred.to_parquet(output / f"{target}_predictions.parquet", index=False)
        joblib.dump({"models": models, "columns": columns, "protocol": protocol}, output / f"{target}.joblib")
        records = []
        for ranker in protocol["rankers"]:
            candidate = pred.assign(expected_net=pred.payoff_net) if ranker == "payoff_difference" else pred
            minimums = protocol["minimum_net"]
            if ranker == "weighted_payoff_share":
                candidate = pred.assign(expected_net=pred.weighted_score)
                minimums = protocol["minimum_payoff_share"]
            for probability in protocol["probability_thresholds"]:
                for minimum in minimums:
                    for top_k in protocol["top_k"]:
                        picks = select_net(candidate, probability, -np.inf if minimum is None else minimum, top_k)
                        stats = metrics(picks, days, top_k)
                        stress = metrics(picks.assign(net=picks.net - .002), days, top_k)
                        threshold_name = "minimum_payoff_share" if ranker == "weighted_payoff_share" else "minimum_net"
                        records.append({"ranker": ranker, "probability": probability, threshold_name: minimum,
                                        "top_k": top_k, **stats, **{f"cost40_{k}": v for k, v in stress.items()}})
        table = pd.DataFrame(records)
        table.to_csv(output / f"{target}_diagnostics.csv", index=False)
        enough = table[table.signals.ge(50) & table.dates.ge(20)]
        eligible = enough[enough.unknown.eq(0) & enough.cost40_known_win.ge(.8) & enough.cost40_daily_score.gt(0)]
        summary = {"target": target, "fit_known": int(train.sum()), "validation_rows": len(pred),
                   "supported": len(enough), "eligible_before_execution_audit": len(eligible)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed during training")
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
