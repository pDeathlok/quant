#!/usr/bin/env python
"""Train and select overnight models under the user-authorized close proxy."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quant.research.tail_overnight_selection import (  # noqa: E402
    add_selection_features,
    choose_candidates,
    selection_metrics,
    take_profit_return,
)


def serializable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serializable))


def load_frame(source, cost_bps):
    frame = pd.read_parquet(source / "dataset.parquet")
    frame = add_selection_features(frame)
    # Outcome-independent entry gates remain the same as the first experiment.
    frame = frame[frame.entry_policy_eligible].copy()
    frame["target_open"] = frame.y_open - cost_bps / 10000
    frame["target_close"] = frame.y_close - cost_bps / 10000
    for target, threshold in [("tp1", 0.01), ("tp2", 0.02)]:
        frame[f"target_{target}"] = take_profit_return(frame, threshold, cost_bps)
    frame["target_down2"] = frame.y_low.le(-0.02).astype(float).where(frame.y_low.notna())
    return frame.reset_index(drop=True)


def feature_columns(frame):
    return [col for col in frame if col.startswith(("lag1_", "tail_", "pool_", "context_"))]


def fit_bundle(train, calibration, columns, target, args):
    col = f"target_{target}"
    known = train[train[col].notna()]
    cal = calibration[calibration[col].notna()]
    admitted = [c for c in columns if known[c].notna().mean() >= 0.05 and known[c].nunique() > 1]
    expanding = known.date.max() >= pd.Timestamp("2025-01-01")
    cache_name = (
        ("latest_risk" if target == "down2" else "latest_model")
        if expanding
        else ("risk_down2" if target == "down2" else target)
    )
    cache_path = args.output / f"{cache_name}.joblib"
    if (
        args.reuse_models
        and cache_path.exists()
        and args.cached_protocol.get("cost_bps") == args.cost_bps
        and args.cached_protocol.get("source") == str(args.source / "dataset.parquet")
    ):
        cached = joblib.load(cache_path)
        if (
            cached["target"] == target
            and cached["features"] == admitted
            and cached["model"].max_iter == args.iterations
            and cached["train_last_outcome"] == str(known.exit_date.max())
            and cached["calibration_last_outcome"] == str(cal.exit_date.max())
            and cache_path.stat().st_mtime > (args.source / "dataset.parquet").stat().st_mtime
        ):
            print(f"reuse verified model: {cache_name}", flush=True)
            return cached
    model_class = (
        HistGradientBoostingClassifier if target == "down2" else HistGradientBoostingRegressor
    )
    model = model_class(
        max_iter=args.iterations,
        max_leaf_nodes=31,
        min_samples_leaf=120,
        l2_regularization=20.0,
        learning_rate=0.06,
        early_stopping=False,
        random_state=42,
    )
    weights = 1 / known.groupby("date").date.transform("size")
    weights /= weights.mean()
    print(f"fit {target}: {len(known):,} rows, {len(admitted)} columns", flush=True)
    with threadpool_limits(limits=args.threads):
        model.fit(
            known[admitted].astype("float32").replace([np.inf, -np.inf], np.nan),
            known[col],
            sample_weight=weights,
        )
        cal = calibration[calibration[col].notna()]
        raw = (
            model.predict_proba(cal[admitted].astype("float32"))[:, 1]
            if target == "down2"
            else model.predict(cal[admitted].astype("float32"))
        )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    cal_weights = 1 / cal.groupby("date").date.transform("size")
    calibrator.fit(raw, cal[col], sample_weight=cal_weights)
    return {
        "model": model,
        "calibrator": calibrator,
        "features": admitted,
        "target": target,
        "input_mode": "close_proxy_user_approved",
        "train_last_entry": str(known.date.max()),
        "train_last_outcome": str(known.exit_date.max()),
        "calibration_last_outcome": str(cal.exit_date.max()),
    }


def predict(bundle, frame, args, *, return_raw=False):
    x = frame[bundle["features"]].astype("float32").replace([np.inf, -np.inf], np.nan)
    with threadpool_limits(limits=args.threads):
        raw = (
            bundle["model"].predict_proba(x)[:, 1]
            if bundle["target"] == "down2"
            else bundle["model"].predict(x)
        )
    calibrated = bundle["calibrator"].predict(raw)
    return (calibrated, raw) if return_raw else calibrated


def prediction_frame(frame, target, predicted_net, predicted_risk):
    cols = [
        "symbol",
        "date",
        "exit_date",
        "exit_open_executable",
        "pool_good",
        "pool_short",
        "pool_long",
        "pool_watchlist",
        "y_open",
        "y_close",
        "y_high",
        "y_low",
        "tail_return",
        "context_factor_coverage",
    ]
    out = frame[cols].copy()
    out["target_net"] = frame[f"target_{target}"]
    if isinstance(predicted_net, tuple):
        out["predicted_net"], out["raw_predicted_net"] = predicted_net
    else:
        out["predicted_net"] = predicted_net
        out["raw_predicted_net"] = predicted_net
    out["predicted_risk"] = predicted_risk
    return out


def evaluate(frame, policy):
    config = {k: policy[k] for k in ["pool", "top_k", "minimum_net", "maximum_risk"]}
    picks = choose_candidates(frame, **config)
    metrics = selection_metrics(picks, sorted(frame.date.unique()), top_k=policy["top_k"])
    return picks, metrics


def latest_candidates(predictions, policy, root, *, model_phase):
    date = predictions.date.max()
    latest = predictions[predictions.date.eq(date)].copy()
    picked, _ = evaluate(latest, policy)
    picked["selection_status"] = "passes_locked_policy"
    # A watch ranking is not a recommendation when the locked policy is empty.
    watch = (
        latest.sort_values(
            ["predicted_net", "raw_predicted_net", "symbol"], ascending=[False, False, True]
        )
        .head(20)
        .copy()
    )
    watch["selection_status"] = np.where(
        watch.symbol.isin(picked.symbol), "passes_locked_policy", "watch_only"
    )
    watch["model_phase"] = model_phase
    watch.to_csv(root / "latest_watch_ranking.csv", index=False)
    picked.to_csv(root / "latest_selected.csv", index=False)
    return {
        "data_date": str(date.date()),
        "eligible_rows": len(latest),
        "selected": len(picked),
        "selected_symbols": picked.symbol.tolist(),
        "model_phase": model_phase,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/research/tail_overnight")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/research/tail_overnight_v2")
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--reuse-models", action="store_true")
    args = parser.parse_args()
    prior_protocol = args.output / "protocol.json"
    args.cached_protocol = json.loads(prior_protocol.read_text()) if prior_protocol.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.source, args.cost_bps)
    columns = feature_columns(frame)
    train = frame[frame.exit_date.lt("2025-01-01")]
    calibration = frame[frame.date.ge("2025-01-01") & frame.exit_date.lt("2025-07-01")]
    validation = frame[frame.date.ge("2025-07-01") & frame.exit_date.lt("2026-01-01")]
    confirmation = frame[frame.date.ge("2026-01-01")]
    write_json(
        args.output / "protocol.json",
        {
            "input_mode": "close_proxy_user_approved",
            "fit": "2024",
            "calibration": "2025H1",
            "selection": "2025H2",
            "confirmation": "2026",
            "cost_bps": args.cost_bps,
            "search_criterion": "validation daily mean minus 2 standard errors; >=40 active days and >=100 picks",
            "note": "2026 was examined in earlier research; this is exploratory confirmation, not a pristine holdout",
            "features": columns,
            "source": str(args.source / "dataset.parquet"),
            "missing_outcome_search_mark": -0.10,
            "fill_assumption": "next-day target limit if high touches, otherwise close; no intraday stop",
        },
    )
    risk = fit_bundle(train, calibration, columns, "down2", args)
    joblib.dump(risk, args.output / "risk_down2.joblib")
    val_risk, test_risk = predict(risk, validation, args), predict(risk, confirmation, args)
    search_rows, test_predictions = [], {}
    for target in ["open", "close", "tp1", "tp2"]:
        bundle = fit_bundle(train, calibration, columns, target, args)
        joblib.dump(bundle, args.output / f"{target}.joblib")
        val = prediction_frame(
            validation, target, predict(bundle, validation, args, return_raw=True), val_risk
        )
        val.to_parquet(args.output / f"{target}_validation.parquet", index=False)
        test_predictions[target] = prediction_frame(
            confirmation, target, predict(bundle, confirmation, args, return_raw=True), test_risk
        )
        for pool, top_k, minimum_net, maximum_risk in itertools.product(
            ["union", "good", "short", "long"],
            [3, 5, 10],
            [-0.001, 0.0, 0.001, 0.002, 0.005],
            [1.0, 0.4, 0.25],
        ):
            policy = {
                "target": target,
                "pool": pool,
                "top_k": top_k,
                "minimum_net": minimum_net,
                "maximum_risk": maximum_risk,
            }
            _, metrics = evaluate(val, policy)
            search_rows.append({**policy, **metrics})
        print(f"completed validation search: {target}", flush=True)
    search = pd.DataFrame(search_rows)
    search.to_csv(args.output / "validation_search.csv", index=False)
    eligible = search[(search.active_days >= 40) & (search.selected_rows >= 100)]
    if eligible.empty:
        write_json(args.output / "selection.json", {"status": "no_supported_policy"})
        return
    winner = (
        eligible.sort_values(
            ["selection_score", "daily_mark_mean", "target", "pool"],
            ascending=[False, False, True, True],
        )
        .iloc[0]
        .to_dict()
    )
    winner["top_k"] = int(winner["top_k"])
    policy = {
        key: winner[key] for key in ["target", "pool", "top_k", "minimum_net", "maximum_risk"]
    }
    # Persist selection before reading any 2026 performance statistic.
    write_json(
        args.output / "selection.json",
        {
            "policy": policy,
            "validation": winner,
            "status": "validation_supported"
            if winner["selection_score"] > 0
            else "research_only_validation_not_positive",
        },
    )
    results = []
    for target, predictions in test_predictions.items():
        predictions.to_parquet(args.output / f"{target}_confirmation.parquet", index=False)
        if target == policy["target"]:
            picks, metrics = evaluate(predictions, policy)
            picks.to_parquet(args.output / "locked_policy_confirmation_picks.parquet", index=False)
            results.append({"phase": "frozen_2024_model_2026_confirmation", **metrics})
    print(f"locked policy: {policy}", flush=True)
    # Expanding fit is protocol-defined; selection policy is kept fixed.
    refit_train = frame[frame.exit_date.lt("2026-01-01")]
    refit_cal = frame[frame.date.ge("2026-01-01") & frame.exit_date.lt("2026-07-01")]
    forward = frame[frame.date.ge("2026-07-01")]
    refit_risk = fit_bundle(refit_train, refit_cal, columns, "down2", args)
    refit_model = fit_bundle(refit_train, refit_cal, columns, policy["target"], args)
    joblib.dump(refit_risk, args.output / "latest_risk.joblib")
    joblib.dump(refit_model, args.output / "latest_model.joblib")
    forecast = prediction_frame(
        forward,
        policy["target"],
        predict(refit_model, forward, args, return_raw=True),
        predict(refit_risk, forward, args),
    )
    forecast.to_parquet(args.output / "refit_forward_predictions.parquet", index=False)
    picks, metrics = evaluate(forecast, policy)
    picks.to_parquet(args.output / "refit_forward_picks.parquet", index=False)
    results.append({"phase": "expanding_fit_2026H2_confirmation", **metrics})
    pd.DataFrame(results).to_csv(args.output / "confirmation_metrics.csv", index=False)
    current = latest_candidates(
        forecast, policy, args.output, model_phase="fit_through_2025_calibrated_2026H1"
    )
    write_json(args.output / "latest.json", current)
    write_json(
        args.output / "run_summary.json",
        {
            "policy": policy,
            "validation": winner,
            "confirmation": results,
            "latest": current,
            "search_count": len(search),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": [
                "price-based research marks, not verified order fills",
                "current watchlist has no historical membership",
                "latest date is the last available local close proxy, not necessarily today",
            ],
        },
    )
    print(
        json.dumps(
            {"validation": winner, "confirmation": results, "latest": current}, default=serializable
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
