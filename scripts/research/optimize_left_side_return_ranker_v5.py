#!/usr/bin/env python
"""Search and confirm continuous return-ranking candidates for left-side v5."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.research.left_side_return_ranker import (
    fit_return_ranker,
    materialize_return_rank_targets,
)
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold

from scripts.research.materialize_left_side_v5_candidate_factors import (
    ACTIVE_VALUE_FEATURE_COLUMNS,
    CANDLE_FEATURE_COLUMNS,
)
from scripts.research.optimize_left_side_model_v5 import (
    DEFAULT_BASELINE_PREDICTIONS,
    DEFAULT_CANDIDATE_FACTORS,
    DEFAULT_EVENTS,
    DEFAULT_LABELS,
    SIGNAL_COLUMNS,
    TARGET_NAME,
    _baseline_probability,
    _load_training_frame,
    _split_validation_stages,
    evaluate_predictions,
)


DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research/left_side_return_ranker_v5"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/left_side_return_ranker_v5"
DEFAULT_METRICS = DEFAULT_REPORT_ROOT / "experiment_metrics.csv"
DEFAULT_PREDICTIONS = DEFAULT_REPORT_ROOT / "experiment_predictions.parquet"
DEFAULT_DECISION = DEFAULT_REPORT_ROOT / "selection_decision.json"
DEFAULT_CONFIRMATION = DEFAULT_REPORT_ROOT / "confirmation_decision.json"

TARGETS: tuple[str, ...] = (
    "terminal_return_daily_rank",
    "risk_adjusted_path_daily_rank",
)
INPUT_GROUPS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "candle": CANDLE_FEATURE_COLUMNS,
    "active_value": ACTIVE_VALUE_FEATURE_COLUMNS,
    "candle_active": (*CANDLE_FEATURE_COLUMNS, *ACTIVE_VALUE_FEATURE_COLUMNS),
}
EXPERIMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    f"{target}__{group}": (target, additions)
    for target in TARGETS
    for group, additions in INPUT_GROUPS.items()
}


def _run_fold(
    args: argparse.Namespace,
    selected: pd.DataFrame,
    fold_name: str,
    experiments: tuple[str, ...],
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
            "rank_target": "good_path5",
            "new_feature_count": 0,
            **evaluate_predictions(splits.test, baseline_probability),
        }
    )
    for experiment in experiments:
        target, additions = EXPERIMENTS[experiment]
        features = [*LEFT_SIDE_FACTOR_COLUMNS, *SIGNAL_COLUMNS, *additions]
        model = fit_return_ranker(
            splits.train,
            early,
            calibration,
            features=features,
            target=target,
            probability_target=TARGET_NAME,
            n_jobs=args.model_jobs,
        )
        probability = np.asarray(model.predict_proba(splits.test), dtype=float)[:, 1]
        metrics.append(
            {
                "fold": fold_name,
                "experiment": experiment,
                "rank_target": target,
                "new_feature_count": len(additions),
                **evaluate_predictions(splits.test, probability),
            }
        )
        prediction = splits.test[
            ["symbol", "date", "terminal_return", TARGET_NAME]
        ].copy()
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
    indexed = metrics.set_index(["fold", "experiment"])
    baseline_a = indexed.loc[("A", "production_v4_good_path5")]
    baseline_b = indexed.loc[("B", "production_v4_good_path5")]
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        a = indexed.loc[("A", experiment)]
        b = indexed.loc[("B", experiment)]
        passes = bool(
            b["top20_average_net_return"] > baseline_b["top20_average_net_return"]
            and b["top20_profit_factor"] > baseline_b["top20_profit_factor"]
            and a["top20_average_net_return"]
            >= baseline_a["top20_average_net_return"] - 0.001
            and a["top20_profit_factor"] > 1.10
            and b["upper_probability_spread"] > 0.02
            and a["upper_probability_spread"] > 0.0
        )
        rows.append(
            {
                "experiment": experiment,
                "passes_selection_gate": passes,
                "a_top20_average_net_return": a["top20_average_net_return"],
                "a_top20_profit_factor": a["top20_profit_factor"],
                "a_upper_probability_spread": a["upper_probability_spread"],
                "b_top20_average_net_return": b["top20_average_net_return"],
                "b_top20_profit_factor": b["top20_profit_factor"],
                "b_upper_probability_spread": b["upper_probability_spread"],
                "worst_fold_top20_return": min(
                    a["top20_average_net_return"], b["top20_average_net_return"]
                ),
            }
        )
    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates["passes_selection_gate"]]
    selected_experiment = (
        None
        if eligible.empty
        else str(
            eligible.sort_values(
                [
                    "worst_fold_top20_return",
                    "b_top20_average_net_return",
                    "b_top20_profit_factor",
                ],
                ascending=False,
            ).iloc[0]["experiment"]
        )
    )
    selected_target, selected_features = EXPERIMENTS.get(
        selected_experiment or "", (None, ())
    )
    return {
        "status": "success",
        "schema_version": "left-side-return-ranker-v5-selection-v1",
        "probability_target": TARGET_NAME,
        "selected_experiment": selected_experiment,
        "selected_rank_target": selected_target,
        "selected_features": list(selected_features),
        "selection_folds": ["A", "B"],
        "confirmation_fold": "C",
        "selection_gate": {
            "fold_b_top20_return_and_pf": "strictly_above_production_v4",
            "fold_a_top20_return": "not_more_than_10bps_below_production_v4",
            "fold_a_profit_factor": ">1.10",
            "upper_probability_spread": "A>0_and_B>0.02",
        },
        "candidates": candidates.to_dict(orient="records"),
    }


def search(args: argparse.Namespace) -> dict[str, Any]:
    selected = materialize_return_rank_targets(_load_training_frame(args))
    experiments = tuple(args.experiments or EXPERIMENTS)
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"unknown return-ranker experiments: {unknown}")
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold_name in ("A", "B"):
        fold_metrics, fold_predictions = _run_fold(
            args, selected, fold_name, experiments
        )
        metrics.extend(fold_metrics)
        predictions.extend(fold_predictions)
    metric_frame = pd.DataFrame(metrics)
    atomic_write_csv(metric_frame, args.metrics_out, index=False)
    atomic_write_parquet(
        pd.concat(predictions, ignore_index=True), args.predictions_out, index=False
    )
    decision = _selection_decision(metric_frame)
    atomic_write_json(decision, args.decision_out, indent=2)
    return decision


def confirm(args: argparse.Namespace) -> dict[str, Any]:
    decision = json.loads(args.decision_out.read_text(encoding="utf-8"))
    experiment = decision.get("selected_experiment")
    if experiment not in EXPERIMENTS:
        raise RuntimeError("selection did not freeze a return-ranker experiment")
    selected = materialize_return_rank_targets(_load_training_frame(args))
    metrics, predictions = _run_fold(args, selected, "C", (str(experiment),))
    metric_frame = pd.DataFrame(metrics).set_index("experiment")
    baseline = metric_frame.loc["production_v4_good_path5"]
    candidate = metric_frame.loc[str(experiment)]
    passed = bool(
        candidate["top10_average_net_return"]
        > baseline["top10_average_net_return"]
        and candidate["top20_average_net_return"]
        > baseline["top20_average_net_return"]
        and candidate["top20_profit_factor"] > baseline["top20_profit_factor"]
        and candidate["top20_profit_factor"] > 1.0
        and candidate["upper_probability_spread"] > 0.02
    )
    payload = {
        "status": "success",
        "schema_version": "left-side-return-ranker-v5-confirmation-v1",
        "selected_experiment": experiment,
        "selected_rank_target": decision["selected_rank_target"],
        "selected_features": decision["selected_features"],
        "confirmation_fold": "C",
        "baseline_metrics": baseline.to_dict(),
        "candidate_metrics": candidate.to_dict(),
        "promote_recommended": passed,
    }
    atomic_write_json(payload, args.confirmation_out, indent=2)
    atomic_write_csv(
        metric_frame.reset_index(),
        args.metrics_out.with_name("confirmation_metrics.csv"),
        index=False,
    )
    atomic_write_parquet(
        pd.concat(predictions, ignore_index=True),
        args.predictions_out.with_name("confirmation_predictions.parquet"),
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
    subparsers.choices["search"].add_argument(
        "--experiments", nargs="+", default=None
    )
    subparsers.choices["confirm"].add_argument(
        "--confirmation-out", type=Path, default=DEFAULT_CONFIRMATION
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = search(args) if args.command == "search" else confirm(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
