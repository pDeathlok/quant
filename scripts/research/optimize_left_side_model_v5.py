#!/usr/bin/env python
"""Search and confirm a return-aligned left-side v5 ranker."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold

from scripts.research.materialize_left_side_v5_candidate_factors import (
    ACTIVE_VALUE_FEATURE_COLUMNS,
    BREADTH_FEATURE_COLUMNS,
    CANDLE_FEATURE_COLUMNS,
)
from scripts.research.validate_unified_left_side_models import (
    _fit_unified_model,
    _split_validation_stages,
)


DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
)
DEFAULT_LABELS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/labels.parquet"
)
DEFAULT_CANDIDATE_FACTORS = (
    PROJECT_ROOT / "data/research/left_side_unified_v5/candidate_factors.parquet"
)
DEFAULT_BASELINE_PREDICTIONS = (
    PROJECT_ROOT
    / "reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet"
)
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research/left_side_unified_v5"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/left_side_unified_v5"
DEFAULT_METRICS = DEFAULT_REPORT_ROOT / "experiment_metrics.csv"
DEFAULT_PREDICTIONS = DEFAULT_REPORT_ROOT / "experiment_predictions.parquet"
DEFAULT_DECISION = DEFAULT_REPORT_ROOT / "selection_decision.json"
DEFAULT_CONFIRMATION = DEFAULT_REPORT_ROOT / "confirmation_decision.json"

TARGET_NAME = "terminal_net_positive_20bps"
ROUND_TRIP_COST_BPS = 20.0
ROUND_TRIP_COST = ROUND_TRIP_COST_BPS / 10_000.0
SIGNAL_COLUMNS: tuple[str, ...] = ("B1", "SB1", "SUPER_B1", "LOW_PULLBACK")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "candle": CANDLE_FEATURE_COLUMNS,
    "breadth": BREADTH_FEATURE_COLUMNS,
    "active_value": ACTIVE_VALUE_FEATURE_COLUMNS,
    "all_new": (
        *CANDLE_FEATURE_COLUMNS,
        *BREADTH_FEATURE_COLUMNS,
        *ACTIVE_VALUE_FEATURE_COLUMNS,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    event_columns = [
        "symbol",
        "date",
        *LEFT_SIDE_FACTOR_COLUMNS,
        *SIGNAL_COLUMNS,
    ]
    events = pd.read_parquet(args.events, columns=event_columns)
    events["date"] = pd.to_datetime(events["date"], errors="raise").dt.normalize()
    candidates = pd.read_parquet(args.candidate_factors)
    candidates["date"] = pd.to_datetime(
        candidates["date"], errors="raise"
    ).dt.normalize()
    if candidates.duplicated(["symbol", "date"]).any():
        raise ValueError("candidate factor keys are duplicated")
    selected = events.merge(
        candidates,
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    del events, candidates
    gc.collect()

    labels = pd.read_parquet(
        args.labels,
        columns=[
            "symbol",
            "date",
            "label_end_date",
            "mature",
            "locked_limit_up",
            "terminal_return",
            "mfe",
            "mae",
        ],
        filters=[("entry_mode", "==", "next_close"), ("horizon", "==", 5)],
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="raise").dt.normalize()
    labels["label_end_date"] = pd.to_datetime(
        labels["label_end_date"], errors="raise"
    ).dt.normalize()
    labels = labels[
        labels["mature"]
        & ~labels["locked_limit_up"]
        & pd.to_numeric(labels["terminal_return"], errors="coerce").notna()
    ].copy()
    labels[TARGET_NAME] = pd.to_numeric(
        labels["terminal_return"], errors="coerce"
    ).gt(ROUND_TRIP_COST)
    selected = selected.merge(
        labels,
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["date", "symbol"], kind="stable")
    selected[TARGET_NAME] = selected[TARGET_NAME].astype(int)
    return selected.reset_index(drop=True)


def _profit_factor(net: pd.Series) -> float:
    gross_profit = float(net[net.gt(0.0)].sum())
    gross_loss = float(-net[net.lt(0.0)].sum())
    return gross_profit / gross_loss if gross_loss > 0.0 else float("inf")


def _top_k_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    top_k: int,
) -> dict[str, float | int]:
    ranked = frame[["date", "symbol", "terminal_return"]].copy()
    ranked["probability"] = np.asarray(probability, dtype=float)
    selected = (
        ranked.sort_values(
            ["date", "probability", "symbol"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("date", sort=False)
        .head(top_k)
    )
    net = pd.to_numeric(selected["terminal_return"], errors="coerce") - ROUND_TRIP_COST
    return {
        f"top{top_k}_trades": len(net),
        f"top{top_k}_average_net_return": float(net.mean()),
        f"top{top_k}_median_net_return": float(net.median()),
        f"top{top_k}_win_rate": float(net.gt(0.0).mean()),
        f"top{top_k}_profit_factor": _profit_factor(net),
    }


def _upper_score_discrimination(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, float | int]:
    ranked = frame[["date", TARGET_NAME]].copy()
    ranked["probability"] = np.asarray(probability, dtype=float)
    ranked["display_score"] = (
        ranked.groupby("date", sort=False)["probability"]
        .rank(method="average", pct=True)
        .mul(100.0)
    )
    lower = ranked[ranked["display_score"].between(50.0, 60.0, inclusive="left")]
    upper = ranked[ranked["display_score"].ge(90.0)]
    lower_rate = float(lower[TARGET_NAME].mean())
    upper_rate = float(upper[TARGET_NAME].mean())
    return {
        "score_50_60_samples": len(lower),
        "score_90_100_samples": len(upper),
        "score_50_60_probability": lower_rate,
        "score_90_100_probability": upper_rate,
        "upper_probability_spread": upper_rate - lower_rate,
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> dict[str, float | int]:
    target = frame[TARGET_NAME].to_numpy(dtype=int)
    values = np.asarray(probability, dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("prediction vector is invalid")
    return {
        "rows": len(frame),
        "base_rate": float(target.mean()),
        "average_precision": float(average_precision_score(target, values)),
        "roc_auc": float(roc_auc_score(target, values)),
        "brier": float(brier_score_loss(target, values)),
        **_top_k_metrics(frame, values, 10),
        **_top_k_metrics(frame, values, 20),
        **_upper_score_discrimination(frame, values),
    }


def _baseline_probability(args: argparse.Namespace, fold: str, frame: pd.DataFrame) -> np.ndarray:
    predictions = pd.read_parquet(
        args.baseline_predictions,
        filters=[("fold", "==", fold)],
    )
    predictions["date"] = pd.to_datetime(
        predictions["date"], errors="raise"
    ).dt.normalize()
    score_column = "pred_unified_left_long_task_deep"
    merged = frame[["symbol", "date"]].merge(
        predictions[["symbol", "date", score_column]],
        on=["symbol", "date"],
        how="left",
        validate="one_to_one",
    )
    probability = pd.to_numeric(merged[score_column], errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all():
        raise RuntimeError(f"production baseline predictions are incomplete for fold {fold}")
    return probability


def _run_fold(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    fold_name: str,
    experiments: Sequence[str],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    fold = {item.name: item for item in DEFAULT_YEAR_FOLDS}[fold_name]
    splits = split_by_year_fold(selected, fold)
    early, calibration, _threshold = _split_validation_stages(
        splits.validation, TARGET_NAME
    )
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    baseline_probability = _baseline_probability(args, fold_name, splits.test)
    metrics.append(
        {
            "fold": fold_name,
            "experiment": "production_v4_good_path5",
            "target": TARGET_NAME,
            "new_feature_count": 0,
            **evaluate_predictions(splits.test, baseline_probability),
        }
    )
    for experiment in experiments:
        additions = FEATURE_SETS[experiment]
        features = [*LEFT_SIDE_FACTOR_COLUMNS, *additions]
        model = _fit_unified_model(
            splits.train,
            early,
            calibration,
            features,
            TARGET_NAME,
            n_jobs=args.model_jobs,
        )
        probability = np.asarray(model.predict_proba(splits.test), dtype=float)[:, 1]
        metrics.append(
            {
                "fold": fold_name,
                "experiment": experiment,
                "target": TARGET_NAME,
                "new_feature_count": len(additions),
                **evaluate_predictions(splits.test, probability),
            }
        )
        prediction = splits.test[["symbol", "date", "terminal_return", TARGET_NAME]].copy()
        prediction.insert(0, "fold", fold_name)
        prediction["experiment"] = experiment
        prediction["probability"] = probability
        predictions.append(prediction)
        model_path = args.model_root / fold_name / f"{experiment}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        gc.collect()
    return metrics, predictions


def _selection_decision(metrics: pd.DataFrame) -> dict[str, Any]:
    b = metrics[metrics["fold"].eq("B")].set_index("experiment")
    baseline = b.loc["production_v4_good_path5"]
    candidates = b.drop(index="production_v4_good_path5").copy()
    candidates["passes_selection_gate"] = (
        candidates["top20_average_net_return"].gt(
            baseline["top20_average_net_return"]
        )
        & candidates["top20_profit_factor"].gt(baseline["top20_profit_factor"])
        & candidates["upper_probability_spread"].gt(0.02)
    )
    eligible = candidates[candidates["passes_selection_gate"]]
    if eligible.empty:
        selected_experiment = None
    else:
        selected_experiment = str(
            eligible.sort_values(
                [
                    "top20_average_net_return",
                    "top20_profit_factor",
                    "upper_probability_spread",
                ],
                ascending=False,
            ).index[0]
        )
    return {
        "status": "success",
        "schema_version": "left-side-v5-selection-decision-v1",
        "target": TARGET_NAME,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "selection_fold": "B",
        "confirmation_fold": "C",
        "baseline_experiment": "production_v4_good_path5",
        "selected_experiment": selected_experiment,
        "selected_features": list(FEATURE_SETS.get(selected_experiment or "", ())),
        "selection_gate": {
            "top20_average_net_return": "strictly_above_production_v4",
            "top20_profit_factor": "strictly_above_production_v4",
            "upper_probability_spread": ">0.02",
        },
        "candidate_metrics": candidates.reset_index().to_dict(orient="records"),
    }


def search(args: argparse.Namespace) -> dict[str, Any]:
    selected = _load_training_frame(args)
    experiments = tuple(args.experiments or FEATURE_SETS)
    unknown = sorted(set(experiments) - set(FEATURE_SETS))
    if unknown:
        raise ValueError(f"unknown feature sets: {unknown}")
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold in ("A", "B"):
        fold_metrics, fold_predictions = _run_fold(
            args, selected, fold, experiments
        )
        metrics.extend(fold_metrics)
        predictions.extend(fold_predictions)
    metric_frame = pd.DataFrame(metrics)
    atomic_write_csv(metric_frame, args.metrics_out, index=False)
    atomic_write_parquet(
        pd.concat(predictions, ignore_index=True), args.predictions_out, index=False
    )
    decision = _selection_decision(metric_frame)
    decision["events_sha256"] = _sha256(args.events)
    decision["candidate_factors_sha256"] = _sha256(args.candidate_factors)
    atomic_write_json(decision, args.decision_out, indent=2)
    return decision


def confirm(args: argparse.Namespace) -> dict[str, Any]:
    decision = json.loads(args.decision_out.read_text(encoding="utf-8"))
    experiment = decision.get("selected_experiment")
    if experiment not in FEATURE_SETS:
        raise RuntimeError("selection did not freeze a confirmable experiment")
    selected = _load_training_frame(args)
    metrics, predictions = _run_fold(args, selected, "C", (str(experiment),))
    metric_frame = pd.DataFrame(metrics)
    baseline = metric_frame.set_index("experiment").loc["production_v4_good_path5"]
    candidate = metric_frame.set_index("experiment").loc[str(experiment)]
    passed = bool(
        candidate["top20_average_net_return"] > baseline["top20_average_net_return"]
        and candidate["top20_profit_factor"] > baseline["top20_profit_factor"]
        and candidate["top20_profit_factor"] > 1.0
        and candidate["upper_probability_spread"] > 0.02
    )
    payload = {
        "status": "success",
        "schema_version": "left-side-v5-confirmation-decision-v1",
        "target": TARGET_NAME,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "selected_experiment": experiment,
        "selected_features": list(FEATURE_SETS[str(experiment)]),
        "confirmation_fold": "C",
        "baseline_metrics": baseline.to_dict(),
        "candidate_metrics": candidate.to_dict(),
        "promote_recommended": passed,
        "promotion_gate": {
            "top20_average_net_return": "strictly_above_production_v4",
            "top20_profit_factor": "strictly_above_production_v4_and_above_1",
            "upper_probability_spread": ">0.02",
        },
    }
    atomic_write_json(payload, args.confirmation_out, indent=2)
    atomic_write_parquet(
        pd.concat(predictions, ignore_index=True),
        args.predictions_out.with_name("confirmation_predictions.parquet"),
        index=False,
    )
    atomic_write_csv(
        metric_frame,
        args.metrics_out.with_name("confirmation_metrics.csv"),
        index=False,
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "confirm"):
        command = subparsers.add_parser(name)
        command.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
        command.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
        command.add_argument(
            "--candidate-factors", type=Path, default=DEFAULT_CANDIDATE_FACTORS
        )
        command.add_argument(
            "--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS
        )
        command.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
        command.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
        command.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
        command.add_argument("--decision-out", type=Path, default=DEFAULT_DECISION)
        command.add_argument("--model-jobs", type=int, default=6)
    search_parser = subparsers.choices["search"]
    search_parser.add_argument("--experiments", nargs="+", default=None)
    confirm_parser = subparsers.choices["confirm"]
    confirm_parser.add_argument(
        "--confirmation-out", type=Path, default=DEFAULT_CONFIRMATION
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = search(args) if args.command == "search" else confirm(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
