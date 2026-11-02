#!/usr/bin/env python
"""Holding-event classifier ablations, calibration and rare-signal validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from quant.research.holding_event_probability import (  # noqa: E402
    TARGETS,
    add_t0_interactions,
    choose_high_confidence,
    coherent_event_probabilities,
    event_labels,
    probability_metrics,
    threshold_metrics,
)
from quant.research.tail_overnight_selection import add_selection_features  # noqa: E402


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else str(x),
        )
    )


def fit_predict(train, calibration, forecast, columns, target, args, output_name):
    target_col = f"label_{target}"
    known = train[train[target_col].notna()]
    admitted = [c for c in columns if known[c].notna().mean() >= 0.05 and known[c].nunique() > 1]
    model = HistGradientBoostingClassifier(
        max_iter=args.iterations,
        max_leaf_nodes=31,
        min_samples_leaf=120,
        l2_regularization=20,
        learning_rate=0.06,
        early_stopping=False,
        random_state=42,
    )
    artifact_path = args.output / f"{output_name}.joblib"
    if args.reuse_fitted_models and artifact_path.exists():
        artifact = joblib.load(artifact_path)
        cal = calibration[calibration[target_col].notna()]
        if (
            artifact["features"] != admitted
            or artifact["target"] != target
            or artifact["model"].get_params() != model.get_params()
            or artifact["train_last_outcome"] != str(known.exit_date.max())
            or artifact["calibration_last_outcome"] != str(cal.exit_date.max())
        ):
            raise ValueError(f"incompatible cached model: {output_name}")
        print(f"{output_name}: reuse fitted model", flush=True)
        with threadpool_limits(limits=args.threads):
            probability = artifact["calibrator"].predict(
                artifact["model"].predict_proba(forecast[admitted].astype("float32"))[:, 1]
            )
        return probability, admitted
    weights = 1 / known.groupby("date").date.transform("size")
    weights /= weights.mean()
    print(f"{output_name}: fit {len(known):,}, features {len(admitted)}", flush=True)
    with threadpool_limits(limits=args.threads):
        model.fit(known[admitted].astype("float32"), known[target_col], sample_weight=weights)
        cal = calibration[calibration[target_col].notna()]
        raw = model.predict_proba(cal[admitted].astype("float32"))[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            raw, cal[target_col], sample_weight=1 / cal.groupby("date").date.transform("size")
        )
        probability = calibrator.predict(
            model.predict_proba(forecast[admitted].astype("float32"))[:, 1]
        )
    joblib.dump(
        {
            "model": model,
            "calibrator": calibrator,
            "features": admitted,
            "target": target,
            "reference": "T_close",
            "input_mode": "close_proxy_user_approved",
            "train_last_outcome": str(known.exit_date.max()),
            "calibration_last_outcome": str(cal.exit_date.max()),
            "requires_coherent_projection": True,
        },
        args.output / f"{output_name}.joblib",
    )
    return probability, admitted


def observed_prediction(frame, probabilities, target):
    out = (
        frame[["symbol", "date", f"label_{target}"]]
        .copy()
        .rename(columns={f"label_{target}": "outcome"})
    )
    out["probability"] = probabilities[target].to_numpy()
    return out.dropna(subset=["outcome"])


def summarize(frame, probabilities, group, phase):
    global_rows, curves, reliability = [], [], []
    for target in TARGETS:
        pred = observed_prediction(frame, probabilities, target)
        metric, bins = probability_metrics(pred.outcome, pred.probability)
        identity = {"feature_group": group, "phase": phase, "target": target}
        global_rows.append({**identity, **metric})
        curve = threshold_metrics(pred)
        for key, value in identity.items():
            curve[key] = value
        curves.append(curve)
        reliability.extend({**identity, **row} for row in bins)
    return global_rows, pd.concat(curves, ignore_index=True), reliability


def locked_metrics(frame, probabilities, policies, phase):
    rows = []
    for target, policy in policies.items():
        if policy is None:
            continue
        pred = observed_prediction(frame, probabilities, target)
        metrics = threshold_metrics(pred, [policy["threshold"]]).iloc[0].to_dict()
        rows.append({"target": target, "phase": phase, **metrics})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "reports/research/tail_overnight/dataset.parquet"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/research/holding_event_probability_v2"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--reuse-fitted-models",
        action="store_true",
        help="Re-evaluate cached models only when source data are unchanged; caller must verify this",
    )
    parser.add_argument(
        "--symbols", default="", help="Optional comma-separated symbols for latest-date output only"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.source)
    # Existing holdings remain observable even when a new close purchase is gated.
    frame = add_t0_interactions(add_selection_features(frame))
    labels = event_labels(frame)
    for target in TARGETS:
        frame[f"label_{target}"] = labels[target]
    lag = [c for c in frame if c.startswith(("lag1_", "pool_"))]
    base = lag + [c for c in frame if c.startswith(("tail_", "context_"))]
    extended = base + [c for c in frame if c.startswith("t0x_")]
    groups = {"lag1": lag, "t0_base": base, "t0_extended": extended}
    train = frame[frame.exit_date.lt("2025-01-01")]
    cal = frame[frame.date.ge("2025-01-01") & frame.exit_date.lt("2025-07-01")]
    future = frame[frame.date.ge("2025-07-01")].copy()
    val_mask = future.exit_date.lt("2026-01-01")
    test_mask = future.date.ge("2026-01-01")
    save_json(
        args.output / "protocol.json",
        {
            "targets": TARGETS,
            "feature_groups": groups,
            "fit": "2024",
            "calibration": "2025H1",
            "selection": "2025H2",
            "confirmation": "2026, previously examined",
            "no_entry_policy_filter": True,
            "objective": "precision/coverage/lift/calibration, not trading return",
            "threshold_support": "precision>=.80, block-bootstrap low90>=.70, lift>=1.2, n>=50, dates>=20",
            "bootstrap": "resample consecutive 5-session blocks; exploratory, not multiple-testing adjusted",
            "model_group_selection": "validation supported target count, then summed qualified coverage, then Brier skill",
            "reference": "T close as authorized 14:30 proxy",
        },
    )
    validation_metrics, validation_curves, all_reliability = [], [], []
    all_probabilities = {}
    group_policies = {}
    group_ranking = []
    admissions = []
    for group, columns in groups.items():
        probabilities = pd.DataFrame(index=future.index)
        for target in TARGETS:
            probabilities[target], admitted = fit_predict(
                train, cal, future, columns, target, args, f"{group}_{target}"
            )
            admissions.extend(
                {"feature_group": group, "target": target, "feature": c, "admitted": c in admitted}
                for c in columns
            )
        probabilities = coherent_event_probabilities(probabilities)
        all_probabilities[group] = probabilities
        forecast = pd.concat([future[["symbol", "date"]], probabilities], axis=1)
        forecast.to_parquet(args.output / f"{group}_probabilities.parquet", index=False)
        stats, curves, reliability = summarize(
            future.loc[val_mask], probabilities.loc[val_mask], group, "validation"
        )
        validation_metrics.extend(stats)
        validation_curves.append(curves)
        all_reliability.extend(reliability)
        policies = {t: choose_high_confidence(curves[curves.target.eq(t)]) for t in TARGETS}
        group_policies[group] = policies
        group_ranking.append(
            {
                "feature_group": group,
                "supported_targets": sum(p is not None for p in policies.values()),
                "qualified_coverage_sum": sum(
                    p["coverage"] for p in policies.values() if p is not None
                ),
                "mean_brier_skill": float(
                    np.mean([1 - row["brier"] / row["base_rate_brier"] for row in stats])
                ),
            }
        )
        print(
            f"{group}: validation supports {group_ranking[-1]['supported_targets']} targets",
            flush=True,
        )
    ranking = pd.DataFrame(group_ranking).sort_values(
        ["supported_targets", "qualified_coverage_sum", "mean_brier_skill", "feature_group"],
        ascending=[False, False, False, True],
    )
    selected_group = ranking.iloc[0].feature_group
    policies = group_policies[selected_group]
    # Selection is written before examining confirmation accuracy.
    save_json(
        args.output / "locked_policy.json",
        {
            "feature_group": selected_group,
            "policies": policies,
            "status": "validation_supported"
            if any(p is not None for p in policies.values())
            else "no_high_confidence_support",
        },
    )
    ranking.to_csv(args.output / "feature_group_selection.csv", index=False)
    pd.DataFrame(admissions).to_csv(args.output / "feature_admission.csv", index=False)
    pd.DataFrame(validation_metrics).to_csv(args.output / "validation_metrics.csv", index=False)
    pd.concat(validation_curves).to_csv(args.output / "validation_thresholds.csv", index=False)
    confirmation_metrics, confirmation_curves = [], []
    for group, probabilities in all_probabilities.items():
        stats, curves, reliability = summarize(
            future.loc[test_mask], probabilities.loc[test_mask], group, "confirmation"
        )
        confirmation_metrics.extend(stats)
        confirmation_curves.append(curves)
        all_reliability.extend(reliability)
    pd.DataFrame(confirmation_metrics).to_csv(args.output / "confirmation_metrics.csv", index=False)
    pd.concat(confirmation_curves).to_csv(args.output / "confirmation_thresholds.csv", index=False)
    locked = locked_metrics(
        future.loc[test_mask],
        all_probabilities[selected_group].loc[test_mask],
        policies,
        "frozen_2024_model_2026",
    )
    selected_stats = all_probabilities[selected_group].loc[test_mask]
    forward = future.loc[test_mask]
    phase = "frozen_2024_model_calibrated_2025H1"
    if any(p is not None for p in policies.values()):
        refit = frame[frame.exit_date.lt("2026-01-01")]
        recal = frame[frame.date.ge("2026-01-01") & frame.exit_date.lt("2026-07-01")]
        forward = frame[frame.date.ge("2026-07-01")]
        selected_stats = pd.DataFrame(index=forward.index)
        for target in TARGETS:
            selected_stats[target], _ = fit_predict(
                refit, recal, forward, groups[selected_group], target, args, f"latest_{target}"
            )
        selected_stats = coherent_event_probabilities(selected_stats)
        locked += locked_metrics(forward, selected_stats, policies, "expanding_fit_2026H2")
        phase = "fit_through_2025_calibrated_2026H1"
        pd.concat([forward[["symbol", "date"]], selected_stats], axis=1).to_parquet(
            args.output / "refit_probabilities.parquet", index=False
        )
        stats, curves, reliability = summarize(
            forward, selected_stats, selected_group, "refit_2026H2"
        )
        pd.DataFrame(stats).to_csv(args.output / "refit_metrics.csv", index=False)
        curves.to_csv(args.output / "refit_thresholds.csv", index=False)
        all_reliability.extend(reliability)
    pd.DataFrame(locked).to_csv(args.output / "locked_threshold_confirmation.csv", index=False)
    pd.DataFrame(all_reliability).to_csv(args.output / "reliability_bins.csv", index=False)
    latest_date = forward.date.max()
    latest = pd.concat([forward[["symbol", "date"]], selected_stats], axis=1)
    latest = latest[latest.date.eq(latest_date)].copy()
    phase_metrics = [
        r
        for r in locked
        if r["phase"]
        == ("expanding_fit_2026H2" if phase.startswith("fit_through") else "frozen_2024_model_2026")
    ]
    supported = []
    for target, policy in policies.items():
        conf = next((r for r in phase_metrics if r["target"] == target), None)
        okay = bool(
            conf
            and conf["signals"] >= 50
            and conf["signal_dates"] >= 20
            and conf["precision"] >= 0.8
            and conf["precision_low90"] >= 0.7
            and conf["lift"] >= 1.2
        )
        latest[f"signal_{target}"] = (
            latest[target].ge(policy["threshold"]) if policy and okay else False
        )
        if okay:
            supported.append(target)
    latest["model_phase"] = phase
    latest.to_csv(args.output / "latest_all_probabilities.csv", index=False)
    if args.symbols:
        symbols = set(args.symbols.split(","))
        latest[latest.symbol.isin(symbols)].to_csv(
            args.output / "requested_holdings_probabilities.csv", index=False
        )
        save_json(args.output / "missing_symbols.json", sorted(symbols - set(latest.symbol)))
    save_json(
        args.output / "run_summary.json",
        {
            "feature_group": selected_group,
            "group_ranking": ranking.to_dict("records"),
            "locked_threshold_confirmation": locked,
            "latest_data_date": str(latest_date.date()),
            "latest_rows": len(latest),
            "latest_model_phase": phase,
            "confirmation_supported_targets": supported,
            "latest_signals": {t: int(latest[f"signal_{t}"].sum()) for t in TARGETS},
            "new_t0_features": [c for c in extended if c.startswith("t0x_")],
        },
    )
    print(
        json.dumps(
            {
                "selected_group": selected_group,
                "confirmation_supported_targets": supported,
                "latest_date": str(latest_date),
                "latest_signals": {t: int(latest[f"signal_{t}"].sum()) for t in TARGETS},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
