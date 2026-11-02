"""Fixed-rule two-by-two training/selection-pool diagnostic on common inputs."""

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
from train_overnight_direct import metrics, select

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS

RANK_INPUTS = [f"today_factor_{name}" for name in [
    "return_1d", "return_5d", "return_10d", "momentum_20d", "momentum_60d",
    "volatility_20d", "downside_volatility_20d", "turnover_rate", "ts_volume_ratio",
    "pe_ttm_inv", "pb_inv", "circ_mv",
]]


def cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Within-date percentiles over observed universe; missing stays missing."""
    if frame[["symbol", "date"]].isna().any().any() or frame.duplicated(["symbol", "date"]).any():
        raise ValueError("Invalid cross-sectional keys")
    values = frame[RANK_INPUTS].replace([np.inf, -np.inf], np.nan)
    ranked = values.groupby(frame.date).rank(method="average", pct=True, na_option="keep")
    return ranked.rename(columns={c: f"cross_rank_{c}" for c in RANK_INPUTS}).astype("float32")


def partition_masks(frame: pd.DataFrame, *, recent_fit: bool = False,
                    early_calibration: bool = False) -> tuple[pd.Series, pd.Series, pd.Series]:
    if recent_fit and early_calibration:
        raise ValueError("Incompatible training windows")
    start = "2025-01-01" if recent_fit else "2024-01-01"
    train = frame.date.ge(start) & frame.date.lt("2025-07-01") & frame.exit_date.lt("2025-07-01")
    calibration = frame.date.ge("2025-07-01") & frame.date.lt("2025-10-01") & frame.exit_date.lt("2025-10-01")
    # The last observed market date cannot yet have a next-session label.
    # Earlier missing stock outcomes stay in evaluation and selection.
    evaluation = frame.date.ge("2025-10-01") & frame.date.lt("2026-09-04")
    evaluation &= frame.date.ge("2026-01-01") | frame.exit_date.lt("2026-01-01")
    if early_calibration:
        train = frame.date.ge("2024-01-01") & frame.date.lt("2025-01-01") & frame.exit_date.lt("2025-01-01")
        calibration = frame.date.ge("2025-01-01") & frame.date.lt("2025-07-01") & frame.exit_date.lt("2025-07-01")
        q3 = frame.date.ge("2025-07-01") & frame.date.lt("2025-10-01") & frame.exit_date.lt("2025-10-01")
        evaluation |= q3
    if (train & calibration | train & evaluation | calibration & evaluation).any():
        raise ValueError("Overlapping partitions")
    return train, calibration, evaluation


def date_weights(frame: pd.DataFrame) -> pd.Series:
    weights = 1 / frame.groupby("date").date.transform("size")
    return weights / weights.mean()


def event_classification_labels(frame: pd.DataFrame, target: str) -> pd.Series:
    if target not in {"limit05", "limit1"}:
        raise ValueError("Event head requires a precommitted limit target")
    event = frame[f"{target}_penetrated"].astype("boolean")
    if not event.notna().equals(frame[target].notna()):
        raise ValueError("Event/return knownness mismatch")
    return event


def direct_exit_flags(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in {"net_open", "net_close"}:
        raise ValueError("Unsupported direct exit target")
    price = frame[f"next_{target.removeprefix('net_')}"]
    known_limit = np.isfinite(frame.down_limit) & frame.down_limit.gt(0)
    at_lower = known_limit & price.le(frame.down_limit + .0051)
    unknown = ~np.isfinite(frame[target]) | ~np.isfinite(price) | ~price.gt(0) | ~known_limit
    unknown |= at_lower | frame.price_limit_inconsistent
    return pd.DataFrame({f"{target}_exit_uncertain": unknown,
                         f"{target}_net_after_exit_audit": frame[target].mask(unknown)}, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["limit05", "limit1", "net_open", "net_close"], default="limit05")
    parser.add_argument("--backend", choices=["hgb", "xgboost"], default="hgb")
    parser.add_argument("--target-calendar", action="store_true")
    parser.add_argument("--cross-sectional-ranks", action="store_true")
    parser.add_argument("--recent-fit", action="store_true")
    parser.add_argument("--event-head", action="store_true")
    parser.add_argument("--early-calibration", action="store_true")
    args = parser.parse_args()
    if args.early_calibration and (not args.event_head or args.recent_fit or args.target_calendar or args.backend != "hgb"):
        parser.error("Early calibration requires the event head and unchanged original inputs")
    if args.cross_sectional_ranks and args.target_calendar:
        parser.error("Rank ablation must not simultaneously change calendar inputs")
    if args.recent_fit and (args.cross_sectional_ranks or args.target_calendar or args.backend != "hgb"):
        parser.error("Recent-fit ablation requires original187 inputs and HGB")
    if args.event_head and (args.target not in {"limit05", "limit1"} or args.recent_fit
                           or args.cross_sectional_ranks or (args.target_calendar and args.backend != "hgb")):
        parser.error("Event-head ablation requires a limit target; optional calendar inputs require HGB")
    target = args.target
    root = PROJECT_ROOT / "reports/research"
    factor_root = root / "overnight_unfiltered_full_factors_v1"
    factor_audit = root / "overnight_unfiltered_full_factor_audit_v1"
    # Require actual full audit completion, not a partial build/result file.
    factor_check = json.loads((factor_audit / "complete.json").read_text())
    if factor_check["different_cells"] or factor_check["features"] != 145:
        raise ValueError("Full factor audit did not pass")
    artifact_dirs = {
        "factor": (factor_root, "dataset.parquet", "dataset_sha256"),
        "auxiliary": (root / "overnight_unfiltered_auxiliary_v1", "dataset.parquet", "dataset_sha256"),
        "labels": (root / "overnight_unfiltered_labels_v1", "labels.parquet", "dataset_sha256"),
        "gate": (root / "overnight_unfiltered_entry_gate_v1", "entry_gate.parquet", "dataset_sha256"),
    }
    paths, hashes = {}, {}
    for name, (folder, filename, digest_field) in artifact_dirs.items():
        path = folder / filename
        complete = json.loads((folder / "complete.json").read_text())
        digest = sha256(path)
        if digest != complete[digest_field]:
            raise ValueError(f"Changed {name} artifact")
        paths[name] = path
        hashes[str(path)] = digest
    old_path = root / "tail_overnight/dataset.parquet"
    protocol_path = root / "overnight_same_day_market_limit_gated_v2/protocol.json"
    exit_root = root / "overnight_unfiltered_exit_audit_v1"
    exit_check = json.loads((exit_root / "complete.json").read_text())
    exit_path = exit_root / "exit_audit.parquet"
    if sha256(exit_path) != exit_check["dataset_sha256"]:
        raise ValueError("Changed exit audit artifact")
    factor_audit_sources = json.loads((factor_audit / "sources.json").read_text())
    if hashes[str(paths["factor"])] != factor_audit_sources[str(paths["factor"])]:
        raise ValueError("Full factor dataset differs from audited version")
    for path in [old_path, protocol_path, exit_path, factor_audit / "complete.json"]:
        hashes[str(path)] = sha256(path)
    protocol = json.loads(protocol_path.read_text())
    columns = protocol["features"]
    calendar_features = []
    calendar_path = root / "overnight_target_calendar_v1/dataset.parquet"
    if args.target_calendar:
        calendar_check = json.loads((calendar_path.parent / "complete.json").read_text())
        hashes[str(calendar_path)] = sha256(calendar_path)
        if hashes[str(calendar_path)] != calendar_check["dataset_sha256"]:
            raise ValueError("Changed calendar features")
        calendar_features = calendar_check["features"]
        if len(calendar_features) != 9 or not all(c.startswith("calendar_target_") for c in calendar_features):
            raise ValueError("Unexpected calendar allowlist")
        columns = [*columns, *calendar_features]
    output = root / "overnight_unfiltered_comparison_v1"
    if target != "limit05":
        output = root / f"overnight_unfiltered_{target}_comparison_v1"
    model_params = protocol["params"]
    backend_version = None
    if args.backend == "xgboost":
        import xgboost

        backend_version = xgboost.__version__
        model_params = dict(n_estimators=240, max_depth=5, learning_rate=.04, min_child_weight=50,
                            reg_lambda=30, reg_alpha=0, subsample=.8, colsample_bytree=.8,
                            tree_method="hist", device="cpu", n_jobs=4, random_state=20260912,
                            verbosity=1, validate_parameters=True)
        output = root / f"overnight_unfiltered_{target}_xgboost_v1"
    if args.target_calendar:
        output = root / f"overnight_unfiltered_{target}_{args.backend}_calendar_v1"
    rank_features = [f"cross_rank_{c}" for c in RANK_INPUTS] if args.cross_sectional_ranks else []
    if rank_features:
        columns = [*columns, *rank_features]
        output = root / f"overnight_unfiltered_{target}_{args.backend}_ranks_v1"
    if args.recent_fit:
        output = root / f"overnight_unfiltered_{target}_hgb_recent_fit_v1"
    if args.event_head:
        output = root / f"overnight_unfiltered_{target}_{args.backend}_event_head_v1"
        if args.target_calendar:
            output = root / f"overnight_unfiltered_{target}_hgb_event_calendar_v1"
    if args.early_calibration:
        output = root / f"overnight_unfiltered_{target}_hgb_early_calibration_v1"
    output.mkdir(exist_ok=False)
    lock = {"target": target, "rule": {"probability": .7, "maximum_risk": .1, "top_k": 1,
             "minimum_expected_net": None}, "params": model_params, "features": columns,
            "fit": "2024 through2025H1; exit before2025-07-01", "calibration": "2025Q3; exit before2025-10-01",
            "evaluations": ["2025Q4 with exit before2026-01-01", "2026 through2026-09-03 entry"],
            "arms": "train old/expanded crossed with select old/expanded; same refreshed187-feature data",
            "costs_bps": [20, 40], "unknown": "Keep selected unknown exits; -10% score penalty is not a fill or loss bound",
            "status": "Reused historical diagnostic only, NOT independent confirmation; no parameter or threshold sweep",
            "sources": hashes, "script_sha256": sha256(Path(__file__))}
    if args.backend == "xgboost":
        lock.update(backend="xgboost", backend_version=backend_version,
                    objectives={"classifier": "binary:logistic", "risk": "binary:logistic", "regressor": "reg:squarederror"},
                    model_comparison="Whole fixed backend/configuration ablation, not an isolated leaf-capacity experiment; no early stopping or parameter search",
                    parameter_reference="https://xgboost.readthedocs.io/en/stable/parameter.html")
    if target in {"net_open", "net_close"}:
        lock["exit_policy"] = "Exact next-session open/close mark minus20bps; no take-profit cap; lower-limit/unknown/bounds anomalies remain uncertain"
    if target == "limit1":
        lock["exit_policy"] = "Precommitted ceil(entry*1.01*100)/100 limit; require next high >= limit+entry*0.001; otherwise next close; minus20bps"
        lock["target_note"] = "Classifier predicts positive net strategy return, NOT limit-hit probability; report limit penetration separately"
    if args.target_calendar:
        lock.update(calendar_features=calendar_features, calendar_ablation="Only add nine vendor-calendar date features; no prices, earnings outcomes or unrestricted timestamps")
    if rank_features:
        lock.update(rank_inputs=RANK_INPUTS, rank_features=rank_features,
                    rank_universe="All observed valid daily rows before entry gate or old-pool membership; same-day average percentiles; missing retained",
                    rank_ablation="Only append12 contemporaneous ranks; same187 original columns, model parameters, labels and selection rule")
    if args.recent_fit:
        lock.update(fit="2025H1 only; exit before2025-07-01",
                    recent_fit_ablation="Only remove2024 training rows; calibration/evaluation unchanged; no window search")
    if args.event_head:
        lock.update(target_note="Classifier and calibrator predict limit penetration; regressor/risk still use net strategy return",
                    event_head_ablation="Only classifier labels change; unknown events excluded from fit, retained in evaluation")
        if args.backend == "xgboost":
            lock["event_head_ablation"] = "Event labels unchanged from original HGB event model; fixed backend/configuration comparison only"
        if args.target_calendar:
            lock["event_head_ablation"] = "Event labels unchanged from original event-head model; only nine calendar features added"
    if args.early_calibration:
        lock.update(fit="2024 only; exit before2025-01-01", calibration="2025H1; exit before2025-07-01",
                    evaluations=["2025Q3 with exit before2025-10-01", *lock["evaluations"]],
                    event_head_ablation="Event labels unchanged from original event-head model",
                    early_calibration_ablation="Move training earlier and expand calendar calibration from quarter to half-year; no window sweep or threshold tuning")
    (output / "protocol.json").write_text(json.dumps(lock, indent=2))
    keys = ["symbol", "date"]
    frame = pd.read_parquet(paths["auxiliary"])
    factors = pd.read_parquet(paths["factor"], columns=[*keys, *PROJECT_FACTOR_COLUMNS])
    factors = factors.rename(columns={c: f"today_factor_{c}" for c in PROJECT_FACTOR_COLUMNS})
    frame = frame.merge(factors, on=keys, how="left", validate="one_to_one")
    del factors
    if rank_features:
        ranks = cross_sectional_ranks(frame)
        frame = pd.concat([frame, ranks], axis=1)
        frame[[*keys, *rank_features]].to_parquet(output / "cross_sectional_ranks.parquet", index=False)
        del ranks
    event_columns = [f"{target}_penetrated"] if args.event_head else []
    labels = pd.read_parquet(paths["labels"], columns=[*keys, "exit_date", target, *event_columns])
    frame = frame.merge(labels, on=keys, how="left", validate="one_to_one")
    del labels
    gate = pd.read_parquet(paths["gate"], columns=[*keys, "eligible"])
    frame = frame.merge(gate, on=keys, how="left", validate="one_to_one")
    if frame.eligible.isna().any():
        raise ValueError("Missing entry gate")
    _, _, calendar_evaluation = partition_masks(frame, early_calibration=args.early_calibration)
    days = pd.DatetimeIndex(frame.loc[calendar_evaluation, "date"].unique()).sort_values()
    frame = frame.loc[frame.eligible].reset_index(drop=True)
    del gate
    old = pd.read_parquet(old_path, columns=keys).assign(old_pool=True)
    frame = frame.merge(old, on=keys, how="left", validate="one_to_one")
    frame["old_pool"] = frame.old_pool.eq(True)
    del old
    if args.target_calendar:
        calendar = pd.read_parquet(calendar_path)
        before_keys = frame[keys].copy()
        frame = frame.merge(calendar[[*keys, "target_date", *calendar_features]], on=keys, how="left", validate="one_to_one")
        pd.testing.assert_frame_equal(before_keys, frame[keys])
        known_exit = frame.exit_date.notna()
        if (frame[calendar_features].isna().any().any() or not frame.target_date.gt(frame.date).all()
                or not frame.loc[known_exit, "target_date"].equals(frame.loc[known_exit, "exit_date"])):
            raise ValueError("Calendar timing/key mismatch")
        del calendar, before_keys
    if len(columns) != 187 + len(calendar_features) + len(rank_features) or not frame.feature_date.lt(frame.date).all():
        raise ValueError("Feature schema or timing mismatch")
    x = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
    train, cal, evaluation = partition_masks(frame, recent_fit=args.recent_fit, early_calibration=args.early_calibration)
    net = frame[target]
    class_labels = event_classification_labels(frame, target) if args.event_head else net.gt(0)
    uncertain_column = f"{target}_exit_uncertain"
    audited_column = f"{target}_net_after_exit_audit"
    if target in {"limit05", "limit1"}:
        exit_audit = pd.read_parquet(exit_path, columns=[*keys, uncertain_column, audited_column])
    else:
        audit_labels = pd.read_parquet(paths["labels"], columns=[*keys, target, f"next_{target.removeprefix('net_')}"])
        limits = pd.read_parquet(exit_path, columns=[*keys, "down_limit", "price_limit_inconsistent"])
        audit_labels = audit_labels.merge(limits, on=keys, how="left", validate="one_to_one")
        if audit_labels.price_limit_inconsistent.isna().any():
            raise ValueError("Missing direct-exit limit audit keys")
        exit_audit = pd.concat([audit_labels[keys], direct_exit_flags(audit_labels, target)], axis=1)
        exit_audit.to_parquet(output / "direct_exit_audit.parquet", index=False)
        del audit_labels, limits
    summaries = []
    for arm in ["old", "expanded"]:
        membership = frame.old_pool if arm == "old" else pd.Series(True, index=frame.index)
        fm, cm = train & membership & net.notna(), cal & membership & net.notna()
        for mask in [fm, cm]:
            if class_labels[mask].nunique() != 2 or net[mask].lt(-.02).nunique() != 2:
                raise ValueError("Insufficient training/calibration classes")
        print(f"{arm}: fit={fm.sum()} calibration={cm.sum()} evaluation={evaluation.sum()}", flush=True)
        with threadpool_limits(limits=4):
            if args.backend == "xgboost":
                classifier = xgboost.XGBClassifier(objective="binary:logistic", **model_params)
                regressor = xgboost.XGBRegressor(objective="reg:squarederror", **model_params)
                risk = xgboost.XGBClassifier(objective="binary:logistic", **model_params)
            else:
                classifier = HistGradientBoostingClassifier(**model_params)
                regressor = HistGradientBoostingRegressor(**model_params)
                risk = HistGradientBoostingClassifier(**model_params)
            classifier.fit(x.loc[fm], class_labels[fm].astype(bool), sample_weight=date_weights(frame.loc[fm]))
            regressor.fit(x.loc[fm], net[fm], sample_weight=date_weights(frame.loc[fm]))
            risk.fit(x.loc[fm], net[fm].lt(-.02), sample_weight=date_weights(frame.loc[fm]))
            iso = IsotonicRegression(out_of_bounds="clip").fit(classifier.predict_proba(x.loc[cm])[:, 1], class_labels[cm].astype(bool), sample_weight=date_weights(frame.loc[cm]))
            risk_iso = IsotonicRegression(out_of_bounds="clip").fit(risk.predict_proba(x.loc[cm])[:, 1], net[cm].lt(-.02), sample_weight=date_weights(frame.loc[cm]))
            pred = frame.loc[evaluation, [*keys, "exit_date", "old_pool"]].copy()
            pred["probability"] = iso.predict(classifier.predict_proba(x.loc[evaluation])[:, 1])
            pred["expected_net"] = regressor.predict(x.loc[evaluation])
            pred["risk"] = risk_iso.predict(risk.predict_proba(x.loc[evaluation])[:, 1])
            if args.early_calibration or (args.event_head and (args.target_calendar or args.backend == "xgboost")):
                support = frame.loc[cm, keys].copy()
                support["raw_probability"] = classifier.predict_proba(x.loc[cm])[:, 1]
                support["probability"] = iso.predict(support.raw_probability)
                support["expected_net"] = regressor.predict(x.loc[cm])
                support["risk"] = risk_iso.predict(risk.predict_proba(x.loc[cm])[:, 1])
                support["event_hit"] = class_labels.loc[cm]
                support["net"] = net.loc[cm]
                support.to_parquet(output / f"{arm}_calibration_support.parquet", index=False)
        pred["net"] = net.loc[evaluation]
        if args.event_head:
            pred["event_hit"] = class_labels.loc[evaluation]
        pred = pred.merge(exit_audit, on=keys, how="left", validate="one_to_one")
        if pred[uncertain_column].isna().any():
            raise ValueError("Missing post-selection exit audit")
        pred.to_parquet(output / f"{arm}_predictions.parquet", index=False)
        joblib.dump({"classifier": classifier, "regressor": regressor, "risk": risk,
                     "calibrator": iso, "risk_calibrator": risk_iso, "columns": columns}, output / f"{arm}.joblib")
        for population in ["old", "expanded"]:
            candidates = pred.loc[pred.old_pool] if population == "old" else pred
            picks = select(candidates, .7, -np.inf, 1, .1).copy()
            picks.to_parquet(output / f"train_{arm}_select_{population}_picks.parquet", index=False)
            periods = [("2025Q4", "2025-10-01", "2026-01-01"), ("2026", "2026-01-01", "2026-09-04")]
            if args.early_calibration:
                periods.insert(0, ("2025Q3", "2025-07-01", "2025-10-01"))
            for period, lower, upper in periods:
                part = picks.loc[picks.date.ge(lower) & picks.date.lt(upper)].copy()
                period_days = days[(days >= lower) & (days < upper)]
                for view in ["raw_mark", "exit_audited"]:
                    evaluated = part.copy()
                    if view == "exit_audited":
                        evaluated["net"] = evaluated[audited_column]
                    for cost in [20, 40]:
                        scored = evaluated.copy()
                        scored["net"] -= (cost - 20) / 10000
                        summaries.append({"train_pool": arm, "selection_pool": population, "period": period,
                                          "view": view, "cost_bps": cost, **metrics(scored, period_days, 1)})
        pd.DataFrame(summaries).to_csv(output / "diagnostics.csv", index=False)
        print(f"{arm}: predictions and four-way cost/exit diagnostics saved", flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during training: {path}")
    (output / "complete.json").write_text(json.dumps({"arms": 2, "diagnostic_rows": len(summaries),
        "status": "Experiment complete, goal NOT evaluated as achieved; independent confirmation still required"}, indent=2))


if __name__ == "__main__":
    main()
