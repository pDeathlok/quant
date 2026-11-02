"""Add exact lagged moneyflow only to the event classifier; freeze return/risk heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from train_overnight_direct import metrics, select
from train_overnight_unfiltered_comparison import date_weights, partition_masks

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-head", action="store_true")
    args = parser.parse_args()
    root = PROJECT_ROOT / "reports/research"
    base = root / "overnight_unfiltered_limit1_hgb_event_head_v1"
    money = root / "overnight_moneyflow_rolling_v1"
    protocol = json.loads((base / "protocol.json").read_text())
    paths = {"auxiliary": root / "overnight_unfiltered_auxiliary_v1/dataset.parquet",
             "factors": root / "overnight_unfiltered_full_factors_v1/dataset.parquet",
             "labels": root / "overnight_unfiltered_labels_v1/labels.parquet",
             "gate": root / "overnight_unfiltered_entry_gate_v1/entry_gate.parquet",
             "old": root / "tail_overnight/dataset.parquet", "money": money / "dataset.parquet",
             "base_model": base / "old.joblib", "base_predictions": base / "old_predictions.parquet",
             "base_protocol": base / "protocol.json", "money_proof": money / "complete.json"}
    if args.risk_head:
        parent = root / "overnight_event_moneyflow_classifier_v1"
        paths.update(event_model=parent / "classifier.joblib", event_predictions=parent / "predictions.parquet",
                     event_complete=parent / "complete.json", event_protocol=parent / "protocol.json")
        if not paths["event_complete"].exists():
            raise ValueError("Event classifier experiment must finish first")
    hashes = {str(p): sha256(p) for p in paths.values()}
    for path, digest in protocol["sources"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Baseline source changed: {path}")
    money_proof = json.loads(paths["money_proof"].read_text())
    if hashes[str(paths["money"])] != money_proof["dataset_sha256"]:
        raise ValueError("Moneyflow artifact changed")
    for path, digest in json.loads((money / "sources.json").read_text()).items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Moneyflow source changed: {path}")
    columns = protocol["features"]
    added = money_proof["features"]
    if len(columns) != 187 or len(added) != 24 or set(columns) & set(added):
        raise ValueError("Unexpected feature schema")
    output = root / "overnight_event_moneyflow_classifier_v1"
    if args.risk_head:
        output = root / "overnight_event_moneyflow_risk_v1"
    output.mkdir(exist_ok=False)
    lock = {"sources": hashes, "features": [*columns, *added], "rule": protocol["rule"],
            "params": protocol["params"], "fit": protocol["fit"], "calibration": protocol["calibration"],
            "exit_policy": protocol["exit_policy"], "scope": "Old training and selection pool only; moneyflow coverage is the old pool",
            "change": "Only event classifier and its calibrator refit with24 lagged moneyflow inputs; return/risk predictions frozen",
            "timing": "Exact T-5 through T-1 calendar slots; no same-day moneyflow; missing windows remainNaN",
            "status": "Reused historical development experiment, not independent confirmation", "script_sha256": sha256(Path(__file__))}
    if args.risk_head:
        lock["change"] = "Freeze previous moneyflow event probability and original return regressor; only refit risk(net20<-2%) and risk calibrator with211 inputs"
    (output / "protocol.json").write_text(json.dumps(lock, indent=2))
    keys = ["symbol", "date"]
    old = pd.read_parquet(paths["old"], columns=keys)
    frame = pd.read_parquet(paths["auxiliary"]).merge(old, on=keys, validate="one_to_one")
    factors = pd.read_parquet(paths["factors"], columns=[*keys, *PROJECT_FACTOR_COLUMNS])
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(factors, on=keys, how="left", validate="one_to_one")
    del factors
    flow = pd.read_parquet(paths["money"], columns=[*keys, "flow_date", "window_start", *added])
    frame = frame.merge(flow, on=keys, how="left", validate="one_to_one")
    del flow
    if not frame.flow_date.lt(frame.date).all() or not frame.window_start.le(frame.flow_date).all():
        raise ValueError("Missing or noncausal moneyflow timing")
    labels = pd.read_parquet(paths["labels"], columns=[*keys, "exit_date", "limit1", "limit1_penetrated"])
    frame = frame.merge(labels, on=keys, how="left", validate="one_to_one")
    gate = pd.read_parquet(paths["gate"], columns=[*keys, "eligible"])
    frame = frame.merge(gate, on=keys, how="left", validate="one_to_one")
    frame = frame.loc[frame.eligible.eq(True)].reset_index(drop=True)
    train, cal, evaluation = partition_masks(frame)
    train &= frame.limit1.notna()
    cal &= frame.limit1.notna()
    if not frame.limit1.notna().equals(frame.limit1_penetrated.notna()):
        raise ValueError("Event/return knownness changed")
    x = frame[[*columns, *added]].astype("float32").replace([np.inf, -np.inf], np.nan)
    print(f"Moneyflow event classifier fit={train.sum()} calibration={cal.sum()} evaluation={evaluation.sum()}", flush=True)
    original = joblib.load(paths["base_model"])
    frozen = pd.read_parquet(paths["base_predictions"])
    frozen = frozen.loc[frozen.old_pool].reset_index(drop=True)
    evaluate = frame.loc[evaluation, keys].merge(frozen, on=keys, how="left", validate="one_to_one")
    if len(evaluate) != len(frozen) or evaluate.probability.isna().any():
        raise ValueError("Different baseline evaluation keys")
    with threadpool_limits(limits=4):
        # Exact baseline replay verifies original187 inputs before introducing moneyflow.
        base_p = original["calibrator"].predict(original["classifier"].predict_proba(x.loc[evaluation, columns])[:, 1])
        np.testing.assert_allclose(base_p, evaluate.probability, rtol=0, atol=0)
        target = frame.limit1_penetrated
        if args.risk_head:
            event_model = joblib.load(paths["event_model"])
            event_pred = pd.read_parquet(paths["event_predictions"])
            pd.testing.assert_frame_equal(evaluate[keys].reset_index(drop=True), event_pred[keys].reset_index(drop=True))
            if event_model["columns"] != [*columns, *added]:
                raise ValueError("Previous event feature schema changed")
            event_p = event_model["calibrator"].predict(event_model["classifier"].predict_proba(x.loc[evaluation])[:, 1])
            np.testing.assert_allclose(event_p, event_pred.probability, rtol=0, atol=0)
            for column in ["expected_net", "risk", "net", "event_hit", "limit1_exit_uncertain"]:
                pd.testing.assert_series_equal(evaluate[column], event_pred[column], check_names=False)
            evaluate["probability"] = event_p
            target = frame.limit1.lt(-.02)
        classifier = HistGradientBoostingClassifier(**protocol["params"])
        classifier.fit(x.loc[train], target.loc[train].astype(bool), sample_weight=date_weights(frame.loc[train]))
        raw_cal = classifier.predict_proba(x.loc[cal])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, target.loc[cal].astype(bool), sample_weight=date_weights(frame.loc[cal]))
        changed_column = "risk" if args.risk_head else "probability"
        evaluate[f"original_{changed_column}"] = evaluate[changed_column]
        evaluate[changed_column] = calibrator.predict(classifier.predict_proba(x.loc[evaluation])[:, 1])
    evaluate.to_parquet(output / "predictions.parquet", index=False)
    model_file = "risk.joblib" if args.risk_head else "classifier.joblib"
    joblib.dump({"classifier": classifier, "calibrator": calibrator, "columns": [*columns, *added]}, output / model_file)
    picks = select(evaluate, .7, -np.inf, 1, .1)
    picks.to_parquet(output / "picks.parquet", index=False)
    summaries = []
    for period, lower, upper in [("2025Q4", "2025-10-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-04")]:
        part = picks.loc[picks.date.ge(lower) & picks.date.lt(upper)]
        days = pd.DatetimeIndex(evaluate.loc[evaluate.date.ge(lower) & evaluate.date.lt(upper), "date"].unique())
        for cost in [20, 40]:
            scored = part.assign(net=part.net - (cost - 20) / 10000)
            summaries.append({"period": period, "cost": cost, "event_hits": int(part.event_hit.fillna(False).sum()),
                              "exit_uncertain": int(part.limit1_exit_uncertain.sum()), **metrics(scored, days, 1)})
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed: {path}")
    (output / "complete.json").write_text(json.dumps({"summary": summaries, "baseline_probability_replayed": len(evaluate),
        "status": "Development result only; goal not declared achieved", "picks_sha256": sha256(output / "picks.parquet")}, indent=2))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
