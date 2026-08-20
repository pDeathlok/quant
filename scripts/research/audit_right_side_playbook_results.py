#!/usr/bin/env python
"""Audit completed right-side playbook artifacts without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_text
from quant.research.right_side_playbook_posthoc import (
    MonthBlockBootstrapSpec,
    attach_signal_identities,
    capital_curve_feasibility,
    compare_shared_to_static_by_signal,
    paired_monthly_stability,
    summarize_outcomes_by_fold_action,
    summarize_selections_by_arm_action,
    summarize_selections_by_signal,
    validate_artifact_chain,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS


DATA_ROOT = PROJECT_ROOT / "data/research/right_side_unified_v2_118"
REPORT_ROOT = PROJECT_ROOT / "reports/research/right_side_unified_v2_118"
MODEL_ROOT = PROJECT_ROOT / "models/research/right_side_unified_v2_118"
DEFAULT_EVENTS = DATA_ROOT / "playbook_events.parquet"
DEFAULT_OUTCOMES = DATA_ROOT / "playbook_outcomes.parquet"
DEFAULT_DATASET_MANIFEST = DATA_ROOT / "playbook_dataset_manifest.json"
DEFAULT_SELECTIONS = REPORT_ROOT / "playbook_policy_selections_B.parquet"
DEFAULT_METRICS = REPORT_ROOT / "playbook_policy_metrics_B.json"
DEFAULT_MODEL_MANIFEST = MODEL_ROOT / "playbook/shared_playbook_model_B.manifest.json"
DEFAULT_MODEL = MODEL_ROOT / "playbook/shared_playbook_model_B.joblib"
DEFAULT_FIRST_LAYER_A = (
    MODEL_ROOT
    / "next_close/h5/good_path5/A/unified_long_task_deep_rule105.manifest.json"
)
DEFAULT_FIRST_LAYER_B = (
    MODEL_ROOT
    / "next_close/h5/good_path5/B/unified_long_task_deep_rule105.manifest.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_outcomes(path: Path) -> pd.DataFrame:
    columns = [
        "fold",
        "event_id",
        "playbook_id",
        "entry_mode",
        "exit_policy_id",
        "eligible",
        "mature",
        "net_return",
        "mae",
        "round_trip_cost_bps",
    ]
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[("fold", "in", ["A", "B"])],
    )


def _read_events(path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=["fold", "event_id", *RIGHT_SIDE_SIGNALS],
        filters=[("fold", "==", "B")],
    )


def _read_selections(path: Path) -> pd.DataFrame:
    columns = [
        "arm",
        "fold",
        "event_id",
        "symbol",
        "date",
        "playbook_id",
        "planned_playbook_id",
        "entry_date",
        "exit_date",
        "net_return",
        "mae",
        "round_trip_cost",
        "eligible",
        "mature",
        "execution_status",
    ]
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[("fold", "==", "B")],
    )


def _parquet_fold_metadata(path: Path) -> dict[str, Any]:
    """Inspect fold bounds from parquet metadata without loading outcome values."""

    parquet = pq.ParquetFile(path)
    fold_index = parquet.schema_arrow.names.index("fold")
    minima: list[str] = []
    maxima: list[str] = []
    for position in range(parquet.metadata.num_row_groups):
        statistics = parquet.metadata.row_group(position).column(fold_index).statistics
        if statistics is None or not statistics.has_min_max:
            raise ValueError(f"fold metadata missing min/max statistics: {path}")
        minima.append(str(statistics.min))
        maxima.append(str(statistics.max))
    if not minima:
        raise ValueError(f"parquet contains no row groups: {path}")
    return {
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "global_min": min(minima),
        "global_max": max(maxima),
        "inspection": "parquet_metadata_only_no_label_or_return_values_read",
    }


def _format_percentage(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.4%}"


def _render_report(
    *,
    metrics_payload: dict[str, Any],
    artifact_audit: dict[str, Any],
    selection_action: pd.DataFrame,
    signal_delta: pd.DataFrame,
    monthly: pd.DataFrame,
    stability: dict[str, Any],
    signal_stability: dict[str, Any],
    capital: dict[str, Any],
    promotion_gate: dict[str, Any],
) -> str:
    del monthly
    by_arm = {row["arm"]: row for row in metrics_payload["metrics"]}
    lines = [
        "# Right-side playbook post-hoc audit (A→B, research only)",
        "",
        f"- Artifact chain: **{artifact_audit['status'].upper()}**; first layer is the selected 105-factor unified score, while layer 2 admits all 118 rule factors and 14 strategy identities.",
        "- Fold contract: A trains layer 2, B evaluates it; parquet metadata bounds are A→B/B-only and C was not read.",
        f"- Feature hashes: first-layer 105 `{artifact_audit['first_layer']['rule_feature_columns_sha256']}`, second-layer 118 `{artifact_audit['second_layer']['rule_feature_columns_sha256']}`, full event feature contract `{artifact_audit['second_layer']['model_event_feature_columns_sha256']}`.",
        "- Cost contract: every regular action includes 15 bps round-trip cost; `NO_TRADE` costs zero.",
        f"- Maturity contract: {artifact_audit['maturity_contract']['immature_regular_rows']:,} eligible action rows and {artifact_audit['maturity_contract']['immature_selected_rows']:,} selected trade rows are immature; all remain null with zero leakage.",
        "",
        "## B-fold policy comparison",
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
            f"| {arm} | {_format_percentage(row['average_event_net_return'])} | "
            f"{_format_percentage(row['average_trade_net_return'])} | "
            f"{_format_percentage(row['win_rate'])} | "
            f"{_format_percentage(row['executed_trade_rate'])} | "
            f"{_format_percentage(row['event_coverage'])} | "
            f"{_format_percentage(row['daily_equal_event_max_drawdown_proxy'])} |"
        )
    comparison = metrics_payload["comparison"]
    bootstrap = stability["bootstrap"]
    sign_flip = stability["month_cluster_sign_flip"]
    lines.extend(
        [
            "",
            f"- Shared vs best static event-net delta: {_format_percentage(comparison['shared_delta_average_event_net_return_vs_best_static'])}.",
            f"- Common-event paired delta: {_format_percentage(stability['event_weighted_delta_average_net_return'])}; month-block {bootstrap['confidence_level']:.0%} CI [{_format_percentage(bootstrap['ci_low'])}, {_format_percentage(bootstrap['ci_high'])}], P(Δ>0)={bootstrap['probability_delta_gt_zero']:.2%}.",
            f"- Monthly stability: {stability['positive_months']}/{stability['months']} positive months, {stability['negative_months']}/{stability['months']} negative; negative-month ratio {stability['negative_month_ratio']:.2%}.",
            f"- Exact month-cluster sign-flip: one-sided p={sign_flip['one_sided_p_value_shared_gt_static']:.4f}, two-sided p={sign_flip['two_sided_p_value']:.4f}. The direction-only sign test is reported separately because it ignores magnitude.",
            "- Conservative significance conclusion: **FAIL**; the block CI crosses zero and the magnitude-aware one-sided p-value is above 0.05.",
            "",
            "## Shared model action decomposition",
            "",
            "Planned and executed actions are identical in B because the upstream candidate set produced no observed T+1 cancellation cases.",
            "",
            "| Action | Selection rate | Events | Covered | Selected-action net | Win rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    shared_actions = selection_action.loc[
        selection_action["arm"].eq("shared_playbook_model")
        & selection_action["stage"].eq("executed")
    ]
    for row in shared_actions.to_dict("records"):
        lines.append(
            f"| {row['playbook_id']} | {_format_percentage(row['selection_rate'])} | "
            f"{int(row['events']):,} | {_format_percentage(row['coverage'])} | "
            f"{_format_percentage(row['average_event_net_return'])} | "
            f"{_format_percentage(row['trade_win_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Shared minus static-per-signal by strategy identity",
            "",
            "Signal memberships overlap; these rows are diagnostic slices, not additive portfolio buckets.",
            "",
            "| Signal | Events | Shared event net | Static event net | Δ event net | Shared trade rate | Static trade rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in signal_delta.to_dict("records"):
        lines.append(
            f"| {row['signal']} | {int(row['shared_events']):,} | "
            f"{_format_percentage(row['shared_average_event_net_return'])} | "
            f"{_format_percentage(row['static_average_event_net_return'])} | "
            f"{_format_percentage(row['delta_average_event_net_return'])} | "
            f"{_format_percentage(row['shared_trade_rate'])} | "
            f"{_format_percentage(row['static_trade_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"- Signal-slice direction: {signal_stability['positive_signal_slices']}/14 positive, {signal_stability['negative_signal_slices']}/14 negative. Negative slices: {', '.join(signal_stability['negative_signals'])}.",
            "- These are overlapping diagnostic slices inspected after B; they cannot be used to retrofit a B-selected hybrid policy.",
        ]
    )
    occupancy = capital["unconstrained_occupancy_envelope"]
    lines.extend(
        [
            "",
            "## Capital-curve readiness",
            "",
            f"- Assumption-bound occupancy backtest fields complete: `{capital['can_build_assumption_bound_occupancy_backtest']}` ({capital['trade_rows_with_symbol_entry_exit_net_cost']:,}/{capital['shared_mature_trades']:,} mature shared trades).",
            f"- Raw unconstrained envelope: median {occupancy['median_raw_concurrent_candidates']:.0f}, p95 {occupancy['p95_raw_concurrent_candidates']:.0f}, maximum {occupancy['maximum_raw_concurrent_candidates']:,} concurrent candidates; maximum {occupancy['maximum_raw_new_entries_per_session']:,} new entries/session. This deliberately applies no ranking Top-K or cash limit.",
            "- This result is **not** a true capital curve and is **not production-ready**. Event windows overlap and currently ignore cash, sizing, concurrency, and daily mark-to-market.",
            "- Required portfolio contract: " + ", ".join(capital["missing_portfolio_contract"]) + ".",
            "- Production blockers:",
            "",
        ]
    )
    lines.extend(f"  - {item}" for item in capital["production_blockers"])
    lines.extend(
        [
            "",
            "## Forward promotion gate",
            "",
            f"Overall: **{promotion_gate['status'].upper()}**. This is a forward contract proposed after the B audit; its thresholds must be frozen before a new untouched shadow window and must not be tuned on B.",
            "",
            "| Check | Pass | Observed / requirement |",
            "|---|---:|---|",
        ]
    )
    for check in promotion_gate["checks"]:
        lines.append(
            f"| {check['name']} | {'yes' if check['passed'] else 'no'} | {check['detail']} |"
        )
    lines.extend(
        [
            "",
            "The oracle uses realized outcomes and is non-executable. All net returns above are overlapping event-level statistics after the fixed 15 bps cost, not capital-compounded performance.",
            "",
        ]
    )
    return "\n".join(lines)


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_manifest = _load_json(args.dataset_manifest)
    model_manifest = _load_json(args.model_manifest)
    first_layer_manifests = [
        _load_json(args.first_layer_manifest_a),
        _load_json(args.first_layer_manifest_b),
    ]
    metrics_payload = _load_json(args.metrics)
    serialized_model = joblib.load(args.model)
    outcomes = _read_outcomes(args.outcomes)
    selections = _read_selections(args.selections)
    events = _read_events(args.events)
    fold_metadata = {
        "events": _parquet_fold_metadata(args.events),
        "outcomes": _parquet_fold_metadata(args.outcomes),
        "selections": _parquet_fold_metadata(args.selections),
    }

    artifact_audit = validate_artifact_chain(
        dataset_manifest=dataset_manifest,
        model_manifest=model_manifest,
        first_layer_manifests=first_layer_manifests,
        events_path=args.events,
        outcomes_path=args.outcomes,
        dataset_manifest_path=args.dataset_manifest,
        model_artifact_path=args.model,
        serialized_model_event_features=serialized_model.event_feature_columns,
        metrics_event_features=metrics_payload["event_features"],
        observed_artifact_paths={
            "model_manifest": args.model_manifest,
            "first_layer_manifest_A": args.first_layer_manifest_a,
            "first_layer_manifest_B": args.first_layer_manifest_b,
            "policy_metrics": args.metrics,
            "policy_selections": args.selections,
        },
        parquet_fold_metadata=fold_metadata,
        outcomes=outcomes,
        selections=selections,
    )
    outcome_action = summarize_outcomes_by_fold_action(outcomes)
    selection_action = summarize_selections_by_arm_action(selections)
    with_signals = attach_signal_identities(selections, events)
    selection_signal = summarize_selections_by_signal(with_signals)
    signal_delta = compare_shared_to_static_by_signal(selection_signal)
    monthly, stability = paired_monthly_stability(
        selections,
        bootstrap_spec=MonthBlockBootstrapSpec(
            iterations=args.bootstrap_iterations,
            confidence_level=args.confidence_level,
            random_seed=args.random_seed,
        ),
    )
    capital = capital_curve_feasibility(selections)
    negative_signal_rows = signal_delta.loc[
        signal_delta["delta_average_event_net_return"].lt(0)
    ]
    signal_stability = {
        "positive_signal_slices": int(
            signal_delta["delta_average_event_net_return"].gt(0).sum()
        ),
        "negative_signal_slices": int(len(negative_signal_rows)),
        "zero_signal_slices": int(
            signal_delta["delta_average_event_net_return"].eq(0).sum()
        ),
        "negative_signals": negative_signal_rows["signal"].astype(str).tolist(),
        "warning": "overlapping post-hoc diagnostic identities; not independent tests",
    }
    arm_metrics = {row["arm"]: row for row in metrics_payload["metrics"]}
    shared_metrics = arm_metrics["shared_playbook_model"]
    static_metrics = arm_metrics["static_per_signal"]
    checks = [
        {
            "name": "artifact, leakage, cost and A→B fold audit",
            "passed": artifact_audit["status"] == "pass",
            "detail": "must pass with C forbidden and exact 15 bps regular-action cost",
        },
        {
            "name": "positive paired event-return delta",
            "passed": stability["event_weighted_delta_average_net_return"] > 0.0,
            "detail": _format_percentage(
                stability["event_weighted_delta_average_net_return"]
            ),
        },
        {
            "name": "month-block 95% CI lower bound above zero",
            "passed": stability["bootstrap"]["ci_low"] > 0.0,
            "detail": f"lower={_format_percentage(stability['bootstrap']['ci_low'])}",
        },
        {
            "name": "month-cluster magnitude p≤0.05 (one-sided)",
            "passed": stability["month_cluster_sign_flip"][
                "one_sided_p_value_shared_gt_static"
            ]
            <= 0.05,
            "detail": f"p={stability['month_cluster_sign_flip']['one_sided_p_value_shared_gt_static']:.4f}",
        },
        {
            "name": "positive-month ratio ≥2/3",
            "passed": (
                stability["positive_months"] / stability["months"] >= 2.0 / 3.0
            ),
            "detail": f"{stability['positive_months']}/{stability['months']}",
        },
        {
            "name": "event drawdown proxy no worse than static-per-signal",
            "passed": shared_metrics["daily_equal_event_max_drawdown_proxy"]
            <= static_metrics["daily_equal_event_max_drawdown_proxy"],
            "detail": (
                f"shared={_format_percentage(shared_metrics['daily_equal_event_max_drawdown_proxy'])}, "
                f"static={_format_percentage(static_metrics['daily_equal_event_max_drawdown_proxy'])}"
            ),
        },
        {
            "name": "capital-aware Top-K/cash/capacity backtest",
            "passed": capital["is_true_capital_curve_now"],
            "detail": "not run",
        },
        {
            "name": "30 bps cost-stress capital backtest",
            "passed": False,
            "detail": "not run; current experiment is fixed at 15 bps",
        },
        {
            "name": "untouched future shadow window",
            "passed": False,
            "detail": "not available; B is the current development evaluation",
        },
        {
            "name": "runtime T+1 cancellation path observed",
            "passed": False,
            "detail": "0 cases after upstream exclusions; require live shadow evidence",
        },
    ]
    promotion_gate = {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "nature": "forward_only_proposed_posthoc_freeze_before_new_shadow_data",
        "checks": checks,
    }

    artifact_audit["audited_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    payload = {
        "schema_version": "right-side-playbook-posthoc-audit-v1",
        "artifact_audit": artifact_audit,
        "monthly_stability": stability,
        "signal_stability": signal_stability,
        "capital_curve_feasibility": capital,
        "forward_promotion_gate": promotion_gate,
        "warning": (
            "event-level overlapping outcomes after fixed cost; not a capital curve "
            "and not production approval"
        ),
    }
    atomic_write_csv(outcome_action, args.outcome_action_out, index=False)
    atomic_write_csv(selection_action, args.selection_action_out, index=False)
    atomic_write_csv(selection_signal, args.selection_signal_out, index=False)
    atomic_write_csv(signal_delta, args.signal_delta_out, index=False)
    atomic_write_csv(monthly, args.monthly_out, index=False)
    atomic_write_json(payload, args.summary_out)
    atomic_write_text(
        _render_report(
            metrics_payload=metrics_payload,
            artifact_audit=artifact_audit,
            selection_action=selection_action,
            signal_delta=signal_delta,
            monthly=monthly,
            stability=stability,
            signal_stability=signal_stability,
            capital=capital,
            promotion_gate=promotion_gate,
        ),
        args.report_out,
    )
    return {
        "artifact_status": artifact_audit["status"],
        "paired_events": stability["paired_events"],
        "months": stability["months"],
        "negative_month_ratio": stability["negative_month_ratio"],
        "delta": stability["event_weighted_delta_average_net_return"],
        "ci_low": stability["bootstrap"]["ci_low"],
        "ci_high": stability["bootstrap"]["ci_high"],
        "month_sign_flip_one_sided_p": stability["month_cluster_sign_flip"][
            "one_sided_p_value_shared_gt_static"
        ],
        "production_ready": capital["production_ready"],
        "promotion_gate": promotion_gate["status"],
        "summary": str(args.summary_out),
        "report": str(args.report_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--first-layer-manifest-a", type=Path, default=DEFAULT_FIRST_LAYER_A)
    parser.add_argument("--first-layer-manifest-b", type=Path, default=DEFAULT_FIRST_LAYER_B)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--outcome-action-out",
        type=Path,
        default=REPORT_ROOT / "playbook_outcome_by_fold_action.csv",
    )
    parser.add_argument(
        "--selection-action-out",
        type=Path,
        default=REPORT_ROOT / "playbook_selection_by_arm_action_B.csv",
    )
    parser.add_argument(
        "--selection-signal-out",
        type=Path,
        default=REPORT_ROOT / "playbook_selection_by_arm_signal_B.csv",
    )
    parser.add_argument(
        "--signal-delta-out",
        type=Path,
        default=REPORT_ROOT / "playbook_shared_vs_static_by_signal_B.csv",
    )
    parser.add_argument(
        "--monthly-out",
        type=Path,
        default=REPORT_ROOT / "playbook_shared_vs_static_monthly_B.csv",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=REPORT_ROOT / "playbook_posthoc_audit_B.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=REPORT_ROOT / "playbook_production_readiness_B.md",
    )
    return parser.parse_args()


def main() -> None:
    result = command_audit(parse_args())
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
