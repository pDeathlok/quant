#!/usr/bin/env python
"""Validate A->B right-side playbook policies on normalized research data.

Four executable arms are fixed before fold B is evaluated: ``no_trade``, one
``static_global`` action learned on A, ``static_per_signal`` mappings learned
on A, and a shared action-conditional model trained on A.  A realized-outcome
``oracle`` is reported only as a non-executable upper bound.  Fold C is not an
accepted CLI option and is never read.

All returns are event-level counterfactual statistics with overlapping holding
windows.  The reported equal-event drawdown is a diagnostic proxy, not a
capital-aware portfolio curve.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from quant.research.right_side_playbook_dataset import (
    PLAYBOOK_DEVELOPMENT_FOLDS,
    PLAYBOOK_EVALUATION_FOLD,
    PLAYBOOK_OUTCOME_COLUMNS,
    PLAYBOOK_TRAIN_FOLD,
    audit_narrow_playbook_tables,
    file_sha256,
    join_playbook_event_outcomes,
)
from quant.research.right_side_playbook_model import (
    FIRST_LAYER_FOLD_COLUMN,
    FIRST_LAYER_PROVENANCE_COLUMN,
    FIRST_LAYER_SCORE_COLUMN,
    SharedPlaybookModel,
    admit_default_event_features,
    apply_execution_gate,
    evaluate_playbook_selections,
    fit_static_global_playbook,
    fit_static_per_signal_playbooks,
    score_static_global_playbook,
    score_static_per_signal_playbooks,
    select_no_trade_baseline,
    select_oracle_playbook,
    select_planned_playbook,
)
from quant.research.right_side_playbook_policy import PLAYBOOK_POLICY_VERSION
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/research/right_side_unified_v2_118"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/right_side_unified_v2_118"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research/right_side_unified_v2_118/playbook"
DEFAULT_EVENTS = DEFAULT_DATA_ROOT / "playbook_events.parquet"
DEFAULT_OUTCOMES = DEFAULT_DATA_ROOT / "playbook_outcomes.parquet"
DEFAULT_DATASET_MANIFEST = DEFAULT_DATA_ROOT / "playbook_dataset_manifest.json"
DEFAULT_AUDIT = DEFAULT_REPORT_ROOT / "playbook_dataset_audit.json"
DEFAULT_METRICS_JSON = DEFAULT_REPORT_ROOT / "playbook_policy_metrics_B.json"
DEFAULT_METRICS_CSV = DEFAULT_REPORT_ROOT / "playbook_policy_metrics_B.csv"
DEFAULT_ACTION_RATES = DEFAULT_REPORT_ROOT / "playbook_action_selection_rates_B.csv"
DEFAULT_SELECTIONS = DEFAULT_REPORT_ROOT / "playbook_policy_selections_B.parquet"
DEFAULT_REPORT = DEFAULT_REPORT_ROOT / "playbook_policy_validation_B.md"


def _read_events(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    events = pd.read_parquet(
        path,
        columns=list(columns) if columns is not None else None,
        filters=[("fold", "in", list(PLAYBOOK_DEVELOPMENT_FOLDS))],
    )
    events["fold"] = events["fold"].astype("string")
    events["event_id"] = events["event_id"].astype("string")
    events["symbol"] = events["symbol"].astype("string")
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
    if events["date"].isna().any():
        raise ValueError("playbook events contain invalid dates")
    if set(events["fold"].astype(str)) != set(PLAYBOOK_DEVELOPMENT_FOLDS):
        raise ValueError("playbook validation requires exactly A/B and never C")
    return events


def _read_outcome_fold(path: Path, fold: str) -> pd.DataFrame:
    if fold not in PLAYBOOK_DEVELOPMENT_FOLDS:
        raise ValueError("only A/B outcome folds may be read")
    outcomes = pd.read_parquet(
        path,
        columns=list(PLAYBOOK_OUTCOME_COLUMNS),
        filters=[("fold", "==", fold)],
    )
    outcomes["fold"] = outcomes["fold"].astype("string")
    outcomes["event_id"] = outcomes["event_id"].astype("string")
    outcomes["symbol"] = outcomes["symbol"].astype("string")
    outcomes["date"] = pd.to_datetime(outcomes["date"], errors="coerce").dt.normalize()
    if outcomes.empty or set(outcomes["fold"].astype(str)) != {fold}:
        raise ValueError(f"playbook outcome fold {fold} is empty or contaminated")
    return outcomes


def _feature_columns(args: argparse.Namespace, train_events: pd.DataFrame) -> tuple[str, ...]:
    if args.event_features:
        return tuple(dict.fromkeys(args.event_features))
    return admit_default_event_features(
        train_events,
        minimum_project_coverage=args.minimum_project_feature_coverage,
    )


def _plan_and_gate(scored: pd.DataFrame) -> pd.DataFrame:
    planned = select_planned_playbook(scored)
    return apply_execution_gate(planned, scored)


def _score_shared_model_by_action(
    model: SharedPlaybookModel,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Score B one action at a time to avoid a 9x wide factor peak."""

    score_columns = [
        "fold",
        "event_id",
        "symbol",
        "date",
        FIRST_LAYER_SCORE_COLUMN,
        FIRST_LAYER_PROVENANCE_COLUMN,
        FIRST_LAYER_FOLD_COLUMN,
        *model.event_feature_columns,
    ]
    event_payload = events[score_columns]
    scored_parts: list[pd.DataFrame] = []
    for playbook_id, action_rows in outcomes.groupby("playbook_id", sort=True):
        joined = action_rows.merge(
            event_payload,
            on=["fold", "event_id", "symbol", "date"],
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if joined["_merge"].ne("both").any():
            raise ValueError(f"B action {playbook_id} lacks event factors")
        joined = joined.drop(columns="_merge")
        scored = model.score_actions(joined)
        scored_parts.append(scored[[*PLAYBOOK_OUTCOME_COLUMNS, "predicted_utility"]])
        del joined, scored
        gc.collect()
    scored_actions = pd.concat(scored_parts, ignore_index=True, sort=False)
    if len(scored_actions) != len(outcomes):
        raise RuntimeError("shared model scoring lost action rows")
    return scored_actions


def _selection_output(selected_by_arm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    retained = [
        "fold",
        "event_id",
        "symbol",
        "date",
        "playbook_id",
        "planned_playbook_id",
        "entry_mode",
        "exit_policy_id",
        "predicted_utility",
        "eligible",
        "eligibility_reason",
        "mature",
        "maturity_reason",
        "entry_date",
        "exit_date",
        "exit_reason",
        "execution_status",
        "net_return",
        "mae",
        "round_trip_cost",
        "ambiguous_bar",
    ]
    rows: list[pd.DataFrame] = []
    for arm, selected in selected_by_arm.items():
        available = [column for column in retained if column in selected]
        frame = selected[available].copy()
        frame.insert(0, "arm", arm)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False)


def _metrics_frames(metrics: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    flat_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for row in metrics:
        flat_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "planned_playbook_selection_rate",
                    "executed_playbook_selection_rate",
                    "warning",
                }
            }
        )
        for stage, field in (
            ("planned", "planned_playbook_selection_rate"),
            ("executed", "executed_playbook_selection_rate"),
        ):
            for playbook_id, rate in row[field].items():
                action_rows.append(
                    {
                        "arm": row["arm"],
                        "stage": stage,
                        "playbook_id": playbook_id,
                        "selection_rate": rate,
                    }
                )
    return pd.DataFrame(flat_rows), pd.DataFrame(action_rows)


def _comparison_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {str(row["arm"]): row for row in metrics}
    shared = by_arm["shared_playbook_model"]
    executable_baselines = [
        by_arm["no_trade"],
        by_arm["static_global"],
        by_arm["static_per_signal"],
    ]
    best_static = max(
        executable_baselines,
        key=lambda row: float(row["average_event_net_return"]),
    )
    oracle = by_arm["oracle_upper_bound"]
    return {
        "best_static_arm": best_static["arm"],
        "shared_delta_average_event_net_return_vs_best_static": float(
            shared["average_event_net_return"]
            - best_static["average_event_net_return"]
        ),
        "shared_delta_win_rate_vs_best_static": float(
            shared["win_rate"] - best_static["win_rate"]
        )
        if np.isfinite(shared["win_rate"]) and np.isfinite(best_static["win_rate"])
        else np.nan,
        "shared_delta_drawdown_proxy_vs_best_static": float(
            shared["daily_equal_event_max_drawdown_proxy"]
            - best_static["daily_equal_event_max_drawdown_proxy"]
        ),
        "oracle_gap_average_event_net_return": float(
            oracle["average_event_net_return"]
            - shared["average_event_net_return"]
        ),
        "interpretation": (
            "positive shared return delta indicates layer-2 gain over the best A-fitted "
            "static policy; drawdown values remain event-level proxies"
        ),
    }


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    events = _read_events(args.events)
    outcomes_a = _read_outcome_fold(args.outcomes, PLAYBOOK_TRAIN_FOLD)
    outcomes_b = _read_outcome_fold(args.outcomes, PLAYBOOK_EVALUATION_FOLD)
    outcomes = pd.concat([outcomes_a, outcomes_b], ignore_index=True, sort=False)
    result = audit_narrow_playbook_tables(events, outcomes)
    result.update(
        {
            "events": str(args.events),
            "outcomes": str(args.outcomes),
            "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    )
    atomic_write_json(result, args.output)
    return result


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    events = _read_events(args.events)
    train_events = events[events["fold"].astype(str).eq(PLAYBOOK_TRAIN_FOLD)].copy()
    test_events = events[events["fold"].astype(str).eq(PLAYBOOK_EVALUATION_FOLD)].copy()
    if train_events.empty or test_events.empty:
        raise ValueError("A/B event split is empty")
    if train_events["date"].max() >= test_events["date"].min():
        raise ValueError("A/B event split is not chronological")
    features = _feature_columns(args, train_events)
    outcomes_a = _read_outcome_fold(args.outcomes, PLAYBOOK_TRAIN_FOLD)

    # Fit every executable policy before physically reading fold B outcomes.
    static_train = join_playbook_event_outcomes(
        train_events,
        outcomes_a,
        event_feature_columns=RIGHT_SIDE_SIGNALS,
    )
    global_choice = fit_static_global_playbook(static_train)
    per_signal_policy = fit_static_per_signal_playbooks(static_train)
    del static_train
    gc.collect()

    # The full shared model sees every admitted project factor, all 118 causal
    # rule factors, all 14 identities, and the frozen first-layer A OOS score.
    model = SharedPlaybookModel.fit_normalized(
        train_events,
        outcomes_a,
        event_feature_columns=features,
        fold=PLAYBOOK_EVALUATION_FOLD,
        scratch_dir=args.scratch_dir,
    )
    del outcomes_a
    gc.collect()

    # Fold B is loaded only after global/static mappings and the shared model
    # are frozen. B outcomes then score the precommitted policies uniformly.
    outcomes_b = _read_outcome_fold(args.outcomes, PLAYBOOK_EVALUATION_FOLD)
    selected_by_arm: dict[str, pd.DataFrame] = {
        "no_trade": select_no_trade_baseline(outcomes_b),
    }
    selected_by_arm["static_global"] = _plan_and_gate(
        score_static_global_playbook(outcomes_b, global_choice)
    )
    static_test = join_playbook_event_outcomes(
        test_events,
        outcomes_b,
        event_feature_columns=RIGHT_SIDE_SIGNALS,
    )
    selected_by_arm["static_per_signal"] = _plan_and_gate(
        score_static_per_signal_playbooks(static_test, per_signal_policy)
    )
    del static_test
    gc.collect()
    shared_scored = _score_shared_model_by_action(model, test_events, outcomes_b)
    selected_by_arm["shared_playbook_model"] = _plan_and_gate(shared_scored)
    selected_by_arm["oracle_upper_bound"] = select_oracle_playbook(outcomes_b)
    del shared_scored
    gc.collect()

    metrics = [
        evaluate_playbook_selections(selected, arm=arm)
        for arm, selected in selected_by_arm.items()
    ]
    comparison = _comparison_summary(metrics)
    metrics_frame, action_rates = _metrics_frames(metrics)

    args.model_root.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model_root / "shared_playbook_model_B.joblib"
    joblib.dump(model, model_path)
    model_manifest = model.manifest(
        playbook_catalog_version=PLAYBOOK_POLICY_VERSION,
        data_cutoff=str(events["date"].max().date()),
    )
    model_manifest.update(
        {
            "model": str(model_path),
            "events": {"path": str(args.events), "sha256": file_sha256(args.events)},
            "outcomes": {
                "path": str(args.outcomes),
                "sha256": file_sha256(args.outcomes),
            },
            "dataset_manifest": (
                {
                    "path": str(args.dataset_manifest),
                    "sha256": file_sha256(args.dataset_manifest),
                }
                if args.dataset_manifest.is_file()
                else None
            ),
            "fold_policy": "A_train_to_B_test_only_C_forbidden",
            "static_global_choice": asdict(global_choice),
            "static_per_signal_choices": {
                signal: asdict(choice)
                for signal, choice in per_signal_policy.choices
            },
        }
    )
    model_manifest_path = args.model_root / "shared_playbook_model_B.manifest.json"
    atomic_write_json(model_manifest, model_manifest_path)

    selections = _selection_output(selected_by_arm)
    atomic_write_parquet(
        selections,
        args.selections_out,
        index=False,
        compression="zstd",
    )
    atomic_write_csv(metrics_frame, args.metrics_csv, index=False)
    atomic_write_csv(action_rates, args.action_rates_out, index=False)
    metrics_payload = {
        "schema_version": "right-side-playbook-policy-comparison-v2",
        "evaluation_fold": PLAYBOOK_EVALUATION_FOLD,
        "training_fold": PLAYBOOK_TRAIN_FOLD,
        "fold_c": "forbidden_not_read",
        "event_feature_count": len(features),
        "event_features": list(features),
        "metrics": metrics,
        "comparison": comparison,
        "static_global_choice": asdict(global_choice),
        "static_per_signal_choices": {
            signal: asdict(choice) for signal, choice in per_signal_policy.choices
        },
        "model": str(model_path),
        "model_manifest": str(model_manifest_path),
        "selections": str(args.selections_out),
        "warning": (
            "all returns and drawdown diagnostics are event-level with overlapping "
            "holding windows; this is not a capital curve"
        ),
    }
    atomic_write_json(metrics_payload, args.metrics_json)
    return metrics_payload


def command_report(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    by_arm = {row["arm"]: row for row in payload["metrics"]}
    lines = [
        "# Right-side playbook A→B validation (research only)",
        "",
        "- Development contract: first-layer fold A OOS scores train layer 2; fold B is evaluated; fold C was not read.",
        f"- Event features: {payload['event_feature_count']} (all 118 rule factors and 14 strategy identities are mandatory).",
        "- T+1 ineligible planned actions are cancelled to `NO_TRADE`; the policy never switches to a different action after observing T+1.",
        "",
        "| Arm | Event net | Trade net | Win rate | Trade rate | Coverage | Drawdown proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in (
        "no_trade",
        "static_global",
        "static_per_signal",
        "shared_playbook_model",
        "oracle_upper_bound",
    ):
        row = by_arm[arm]
        lines.append(
            f"| {arm} | {row['average_event_net_return']:.4%} | "
            f"{row['average_trade_net_return']:.4%} | {row['win_rate']:.2%} | "
            f"{row['executed_trade_rate']:.2%} | {row['event_coverage']:.2%} | "
            f"{row['daily_equal_event_max_drawdown_proxy']:.4%} |"
        )
    comparison = payload["comparison"]
    static_global_choice = payload["static_global_choice"]
    shared = by_arm["shared_playbook_model"]
    shared_planned = sorted(
        shared["planned_playbook_selection_rate"].items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    shared_executed = sorted(
        shared["executed_playbook_selection_rate"].items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    planned_summary = ", ".join(
        f"`{playbook_id}` {rate:.2%}" for playbook_id, rate in shared_planned
    )
    executed_summary = ", ".join(
        f"`{playbook_id}` {rate:.2%}" for playbook_id, rate in shared_executed
    )
    lines.extend(
        [
            "",
            f"- Best static arm: `{comparison['best_static_arm']}`.",
            f"- Static-global A-fitted action: `{static_global_choice['playbook_id']}` "
            f"(A mean utility {static_global_choice['mean_utility']:.4%}, "
            f"rows {static_global_choice['training_rows']:,}).",
            f"- Shared model planned choices: {planned_summary}.",
            f"- Shared model executed choices after T+1 gate: {executed_summary}.",
            f"- Shared model T+1 cancellation rate: {shared['cancellation_rate']:.2%}; "
            f"B coverage: {shared['covered_events']:,}/{shared['events']:,} "
            f"({shared['event_coverage']:.2%}); immature selected events left null: "
            f"{shared['unevaluated_events']:,}.",
            "- Shared model Δ average event net return vs best static: "
            f"{comparison['shared_delta_average_event_net_return_vs_best_static']:.4%}.",
            "- Shared model remaining gap to realized oracle: "
            f"{comparison['oracle_gap_average_event_net_return']:.4%}.",
            "",
            "The oracle uses realized outcomes and is not executable. All reported returns have overlapping holding windows; the daily equal-event cumulative-sum drawdown is a diagnostic proxy, not a capital curve or production approval.",
        ]
    )
    atomic_write_text("\n".join(lines) + "\n", args.output)
    return {"report": str(args.output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    audit.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    audit.add_argument("--output", type=Path, default=DEFAULT_AUDIT)

    train = subparsers.add_parser("train")
    train.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    train.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    train.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    train.add_argument("--event-features", nargs="*")
    train.add_argument("--minimum-project-feature-coverage", type=float, default=0.50)
    train.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    train.add_argument(
        "--scratch-dir",
        type=Path,
        default=None,
        help="optional parent for the temporary disk-backed full training matrix",
    )
    train.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    train.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    train.add_argument("--action-rates-out", type=Path, default=DEFAULT_ACTION_RATES)
    train.add_argument("--selections-out", type=Path, default=DEFAULT_SELECTIONS)

    report = subparsers.add_parser("report")
    report.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_JSON)
    report.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        result = command_audit(args)
    elif args.command == "train":
        result = command_train(args)
    else:
        result = command_report(args)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
