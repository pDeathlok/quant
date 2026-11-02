#!/usr/bin/env python
"""Select on A/B and confirm on C a production/return-ranker blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.research.left_side_return_ranker import materialize_return_rank_targets

from scripts.research.optimize_left_side_model_v5 import (
    DEFAULT_BASELINE_PREDICTIONS,
    TARGET_NAME,
    _baseline_probability,
    _load_training_frame,
    evaluate_predictions,
)
from scripts.research.optimize_left_side_return_ranker_v5 import (
    DEFAULT_DECISION as DEFAULT_RETURN_DECISION,
    DEFAULT_METRICS as DEFAULT_RETURN_METRICS,
    DEFAULT_MODEL_ROOT as DEFAULT_RETURN_MODEL_ROOT,
    DEFAULT_PREDICTIONS as DEFAULT_RETURN_PREDICTIONS,
    EXPERIMENTS,
    _run_fold,
)


DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/left_side_blend_v5"
DEFAULT_DECISION = DEFAULT_REPORT_ROOT / "selection_decision.json"
DEFAULT_CONFIRMATION = DEFAULT_REPORT_ROOT / "confirmation_decision.json"
DEFAULT_CONFIRMATION_METRICS = DEFAULT_REPORT_ROOT / "confirmation_metrics.csv"
DEFAULT_CONFIRMATION_PREDICTIONS = DEFAULT_REPORT_ROOT / "confirmation_predictions.parquet"
WEIGHTS: tuple[float, ...] = tuple(round(value, 2) for value in np.arange(0.50, 0.91, 0.05))


def _daily_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date", sort=False)[column].rank(method="average", pct=True)


def _blend_metrics(
    frame: pd.DataFrame,
    production_score: np.ndarray,
    candidate_score: np.ndarray,
    production_weight: float,
) -> dict[str, float | int]:
    values = frame[["date", "symbol", "terminal_return", TARGET_NAME]].copy()
    values["production_score"] = np.asarray(production_score, dtype=float)
    values["candidate_score"] = np.asarray(candidate_score, dtype=float)
    blended = (
        float(production_weight) * _daily_percentile(values, "production_score")
        + (1.0 - float(production_weight))
        * _daily_percentile(values, "candidate_score")
    )
    return evaluate_predictions(values, blended.to_numpy(dtype=float))


def _load_search_predictions(args: argparse.Namespace) -> pd.DataFrame:
    candidate = pd.read_parquet(args.return_predictions)
    candidate["date"] = pd.to_datetime(candidate["date"], errors="raise").dt.normalize()
    production = pd.read_parquet(args.baseline_predictions)[
        ["fold", "symbol", "date", "pred_unified_left_long_task_deep"]
    ]
    production["date"] = pd.to_datetime(
        production["date"], errors="raise"
    ).dt.normalize()
    return candidate.merge(
        production,
        on=["fold", "symbol", "date"],
        how="left",
        validate="many_to_one",
    )


def search(args: argparse.Namespace) -> dict[str, Any]:
    predictions = _load_search_predictions(args)
    metric_frame = pd.read_csv(args.return_metrics)
    production_metrics = metric_frame[
        metric_frame["experiment"].eq("production_v4_good_path5")
    ].set_index("fold")
    rows: list[dict[str, Any]] = []
    for experiment in sorted(predictions["experiment"].unique()):
        experiment_rows = predictions[predictions["experiment"].eq(experiment)]
        by_fold: dict[str, dict[str, float | int]] = {}
        for fold in ("A", "B"):
            current = experiment_rows[experiment_rows["fold"].eq(fold)]
            for weight in WEIGHTS:
                metrics = _blend_metrics(
                    current,
                    current["pred_unified_left_long_task_deep"].to_numpy(float),
                    current["probability"].to_numpy(float),
                    weight,
                )
                by_fold[f"{fold}:{weight:.2f}"] = metrics
        for weight in WEIGHTS:
            a = by_fold[f"A:{weight:.2f}"]
            b = by_fold[f"B:{weight:.2f}"]
            baseline_a = production_metrics.loc["A"]
            baseline_b = production_metrics.loc["B"]
            passes = bool(
                a["top10_average_net_return"]
                > baseline_a["top10_average_net_return"]
                and a["top20_average_net_return"]
                >= baseline_a["top20_average_net_return"] - 0.0005
                and a["top20_profit_factor"]
                >= baseline_a["top20_profit_factor"] - 0.02
                and b["top10_average_net_return"]
                > baseline_b["top10_average_net_return"]
                and b["top20_average_net_return"]
                > baseline_b["top20_average_net_return"]
                and b["top20_profit_factor"] > baseline_b["top20_profit_factor"]
                and a["upper_probability_spread"] > 0.0
                and b["upper_probability_spread"] > 0.02
            )
            rows.append(
                {
                    "experiment": experiment,
                    "production_weight": weight,
                    "passes_selection_gate": passes,
                    **{f"a_{key}": value for key, value in a.items()},
                    **{f"b_{key}": value for key, value in b.items()},
                }
            )
    candidates = pd.DataFrame(rows)
    eligible = candidates[candidates["passes_selection_gate"]]
    selected = (
        None
        if eligible.empty
        else eligible.sort_values(
            [
                "b_upper_probability_spread",
                "b_top20_average_net_return",
                "a_top10_average_net_return",
            ],
            ascending=False,
        ).iloc[0]
    )
    payload: dict[str, Any] = {
        "status": "success",
        "schema_version": "left-side-blend-v5-selection-v1",
        "probability_target": TARGET_NAME,
        "selected_component_experiment": (
            None if selected is None else str(selected["experiment"])
        ),
        "production_weight": (
            None if selected is None else float(selected["production_weight"])
        ),
        "component_weight": (
            None if selected is None else 1.0 - float(selected["production_weight"])
        ),
        "selection_folds": ["A", "B"],
        "confirmation_fold": "C",
        "selection_gate": {
            "fold_a_top10": "strictly_above_production_v4",
            "fold_a_top20": "not_more_than_5bps_below_production_v4",
            "fold_b_top10_top20_pf": "strictly_above_production_v4",
            "upper_probability_spread": "A>0_and_B>0.02",
        },
        "selected_metrics": None if selected is None else selected.to_dict(),
        "eligible_candidate_count": len(eligible),
    }
    atomic_write_json(payload, args.decision_out, indent=2)
    return payload


def confirm(args: argparse.Namespace) -> dict[str, Any]:
    decision = json.loads(args.decision_out.read_text(encoding="utf-8"))
    experiment = decision.get("selected_component_experiment")
    weight = decision.get("production_weight")
    if experiment not in EXPERIMENTS or weight is None:
        raise RuntimeError("blend selection did not freeze a confirmable candidate")
    selected = materialize_return_rank_targets(_load_training_frame(args))
    metrics, predictions = _run_fold(args, selected, "C", (str(experiment),))
    component = predictions[0]
    production_probability = _baseline_probability(args, "C", component)
    blend_metrics = _blend_metrics(
        component,
        production_probability,
        component["probability"].to_numpy(float),
        float(weight),
    )
    metric_frame = pd.DataFrame(metrics)
    baseline = metric_frame[
        metric_frame["experiment"].eq("production_v4_good_path5")
    ].iloc[0].to_dict()
    passed = bool(
        blend_metrics["top10_average_net_return"]
        > baseline["top10_average_net_return"]
        and blend_metrics["top20_average_net_return"]
        > baseline["top20_average_net_return"]
        and blend_metrics["top20_profit_factor"] > baseline["top20_profit_factor"]
        and blend_metrics["top20_profit_factor"] > 1.0
        and blend_metrics["upper_probability_spread"] > 0.02
    )
    payload = {
        "status": "success",
        "schema_version": "left-side-blend-v5-confirmation-v1",
        "selected_component_experiment": experiment,
        "production_weight": float(weight),
        "component_weight": 1.0 - float(weight),
        "confirmation_fold": "C",
        "baseline_metrics": baseline,
        "candidate_metrics": blend_metrics,
        "promote_recommended": passed,
    }
    atomic_write_json(payload, args.confirmation_out, indent=2)
    atomic_write_csv(
        pd.DataFrame(
            [
                {"experiment": "production_v4_good_path5", **baseline},
                {"experiment": "blend_v5", **blend_metrics},
            ]
        ),
        args.confirmation_metrics,
        index=False,
    )
    output = component.copy()
    output["production_probability"] = production_probability
    output["blended_score"] = (
        float(weight) * _daily_percentile(output, "production_probability")
        + (1.0 - float(weight)) * _daily_percentile(output, "probability")
    )
    atomic_write_parquet(output, args.confirmation_predictions, index=False)
    atomic_write_json(payload, args.confirmation_out, indent=2)
    return payload


def _shared_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--events", type=Path, default=None)
    command.add_argument("--labels", type=Path, default=None)
    command.add_argument("--candidate-factors", type=Path, default=None)
    command.add_argument(
        "--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS
    )
    command.add_argument("--return-predictions", type=Path, default=DEFAULT_RETURN_PREDICTIONS)
    command.add_argument("--return-metrics", type=Path, default=DEFAULT_RETURN_METRICS)
    command.add_argument("--return-decision", type=Path, default=DEFAULT_RETURN_DECISION)
    command.add_argument("--model-root", type=Path, default=DEFAULT_RETURN_MODEL_ROOT)
    command.add_argument("--model-jobs", type=int, default=6)
    command.add_argument("--decision-out", type=Path, default=DEFAULT_DECISION)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search")
    _shared_arguments(search_parser)
    confirm_parser = subparsers.add_parser("confirm")
    _shared_arguments(confirm_parser)
    confirm_parser.add_argument("--confirmation-out", type=Path, default=DEFAULT_CONFIRMATION)
    confirm_parser.add_argument(
        "--confirmation-metrics", type=Path, default=DEFAULT_CONFIRMATION_METRICS
    )
    confirm_parser.add_argument(
        "--confirmation-predictions", type=Path, default=DEFAULT_CONFIRMATION_PREDICTIONS
    )
    return parser


def _fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    from scripts.research.optimize_left_side_model_v5 import (
        DEFAULT_CANDIDATE_FACTORS as candidate_factors,
        DEFAULT_EVENTS as events,
        DEFAULT_LABELS as labels,
    )

    args.events = args.events or events
    args.labels = args.labels or labels
    args.candidate_factors = args.candidate_factors or candidate_factors
    return args


def main() -> None:
    args = _fill_defaults(_parser().parse_args())
    payload = search(args) if args.command == "search" else confirm(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
