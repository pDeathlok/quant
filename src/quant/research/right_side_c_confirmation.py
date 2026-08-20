"""Confirm one frozen right-side candidate on the untouched C fold.

The policy in this module is intentionally fixed before fold C is trained.  A
pass authorizes shadow validation only; it is neither a production release nor
a portfolio-backtest conclusion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.data.atomic_io import atomic_write_csv, atomic_write_json


POLICY_VERSION = "right-side-c-confirmation-v1"


@dataclass(frozen=True)
class CConfirmationPolicy:
    minimum_confidence_level: float = 0.95
    minimum_month_blocks: int = 6
    minimum_bootstrap_valid: int = 400
    minimum_independent_coverage: float = 0.80
    minimum_delta_pr_auc: float = 0.0
    pr_auc_ci_noninferiority_margin: float = 0.005
    minimum_delta_top_lift: float = -0.05
    minimum_delta_terminal_return: float = -0.002
    minimum_signal_rows: int = 200
    minimum_signal_class_rows: int = 20
    minimum_evaluable_signals: int = 6
    signal_slice_pr_auc_margin: float = 0.020
    minimum_signal_macro_delta: float = 0.0
    minimum_signal_improvement_fraction: float = 0.50


DEFAULT_POLICY = CConfirmationPolicy()


@dataclass(frozen=True)
class CConfirmationResult:
    decision: str
    candidate: str
    scope: dict[str, Any]
    checks: pd.DataFrame
    signal_deltas: pd.DataFrame
    summary: dict[str, Any]


def c_confirmation_payload(
    result: CConfirmationResult,
    *,
    policy: CConfirmationPolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Serialize one C-only evaluation with its frozen policy contract."""

    return {
        "policy_version": POLICY_VERSION,
        "policy": asdict(policy),
        "decision": result.decision,
        "candidate": result.candidate,
        "scope": result.scope,
        "summary": result.summary,
        "checks": result.checks.to_dict("records"),
        "limitations": [
            "A pass authorizes shadow validation only; it is not production approval.",
            "Top-K terminal return is an overlapping trade-level statistic, not a capital curve.",
            "The interval reflects C-period month-block uncertainty, not retraining uncertainty.",
        ],
    }


def _one(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"expected one {description} row, got {len(frame)}")
    return frame.iloc[0]


def evaluate_c_confirmation(
    frozen_ab: dict[str, Any],
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    *,
    policy: CConfirmationPolicy = DEFAULT_POLICY,
) -> CConfirmationResult:
    """Evaluate only fold C against the candidate frozen by A/B."""

    if frozen_ab.get("decision") != "freeze_for_c_confirmation":
        raise ValueError("A/B decision did not authorize C confirmation")
    candidate = str(frozen_ab.get("selected_candidate") or "")
    if not candidate:
        raise ValueError("A/B decision has no selected_candidate")
    scope = dict(frozen_ab.get("scope") or {})
    required_scope = {"entry_mode", "horizon", "label"}
    if required_scope - set(scope):
        raise ValueError("A/B decision scope is incomplete")

    def scoped(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in required_scope:
            result = result[result[column].astype(str).eq(str(scope[column]))]
        return result[result["fold"].astype(str).eq("C")]

    paired_c = scoped(paired)
    if set(paired_c["fold"].astype(str).unique()) - {"C"}:
        raise ValueError("paired input contains non-C rows after scoping")
    pure = _one(
        paired_c[
            paired_c["candidate"].astype(str).eq(candidate)
            & paired_c["comparison_scope"].astype(str).eq("independent_model_rows")
        ],
        "C independent_model_rows paired",
    )
    full = _one(
        paired_c[
            paired_c["candidate"].astype(str).eq(candidate)
            & paired_c["comparison_scope"].astype(str).eq("all_events")
        ],
        "C all_events paired",
    )
    if str(pure.get("status")) != "ok" or str(full.get("status")) != "ok":
        raise ValueError("C paired comparison status is not ok")

    metrics_c = scoped(metrics)
    independent = _one(
        metrics_c[
            metrics_c["experiment"].astype(str).eq("independent")
            & metrics_c["signal"].astype(str).eq("ALL")
        ],
        "C aggregate independent metric",
    )
    rows = float(independent["rows"])
    fallback = float(independent.get("fallback_rows", 0.0) or 0.0)
    metric_coverage = 1.0 - fallback / max(rows, 1.0)
    paired_coverage = float(pure["paired_rows"]) / max(float(full["paired_rows"]), 1.0)
    coverage = min(metric_coverage, paired_coverage)

    signals_c = scoped(signal_metrics)
    baseline = signals_c[signals_c["experiment"].astype(str).eq("independent")][
        ["signal", "rows", "positives", "average_precision"]
    ].rename(
        columns={
            "rows": "baseline_rows",
            "positives": "baseline_positives",
            "average_precision": "baseline_average_precision",
        }
    )
    arm = signals_c[signals_c["experiment"].astype(str).eq(candidate)][
        ["signal", "rows", "positives", "average_precision"]
    ].rename(
        columns={
            "rows": "candidate_rows",
            "positives": "candidate_positives",
            "average_precision": "candidate_average_precision",
        }
    )
    signal_delta = baseline.merge(arm, on="signal", how="inner", validate="one_to_one")
    signal_delta["counts_match"] = (
        signal_delta["baseline_rows"].eq(signal_delta["candidate_rows"])
        & signal_delta["baseline_positives"].eq(signal_delta["candidate_positives"])
    )
    negatives = signal_delta["baseline_rows"] - signal_delta["baseline_positives"]
    signal_delta["evaluable"] = (
        signal_delta["counts_match"]
        & signal_delta["baseline_rows"].ge(policy.minimum_signal_rows)
        & signal_delta["baseline_positives"].ge(policy.minimum_signal_class_rows)
        & negatives.ge(policy.minimum_signal_class_rows)
        & np.isfinite(signal_delta["baseline_average_precision"])
        & np.isfinite(signal_delta["candidate_average_precision"])
    )
    signal_delta["delta_average_precision"] = (
        signal_delta["candidate_average_precision"]
        - signal_delta["baseline_average_precision"]
    )
    evaluable = signal_delta[signal_delta["evaluable"]]
    signal_min = float(evaluable["delta_average_precision"].min()) if len(evaluable) else np.nan
    signal_macro = float(evaluable["delta_average_precision"].mean()) if len(evaluable) else np.nan
    signal_improvement = float((evaluable["delta_average_precision"] > 0).mean()) if len(evaluable) else np.nan

    check_specs = [
        ("candidate_matches_frozen_ab", True, candidate),
        ("paired_status_ok", True, "ok"),
        ("confidence_level", float(pure["confidence_level"]) >= policy.minimum_confidence_level, float(pure["confidence_level"])),
        ("month_blocks", int(pure["month_blocks"]) >= policy.minimum_month_blocks, int(pure["month_blocks"])),
        ("bootstrap_valid", int(pure["delta_pr_auc_bootstrap_valid"]) >= policy.minimum_bootstrap_valid, int(pure["delta_pr_auc_bootstrap_valid"])),
        ("independent_coverage", coverage >= policy.minimum_independent_coverage, coverage),
        ("delta_pr_auc_positive", float(pure["delta_pr_auc"]) > policy.minimum_delta_pr_auc, float(pure["delta_pr_auc"])),
        ("delta_pr_auc_ci_noninferior", float(pure["delta_pr_auc_ci_low"]) >= -policy.pr_auc_ci_noninferiority_margin, float(pure["delta_pr_auc_ci_low"])),
        ("delta_top_lift_guardrail", float(full["delta_top_lift"]) >= policy.minimum_delta_top_lift, float(full["delta_top_lift"])),
        ("delta_terminal_return_guardrail", float(full["delta_daily_top_k_avg_terminal_return"]) >= policy.minimum_delta_terminal_return, float(full["delta_daily_top_k_avg_terminal_return"])),
        ("evaluable_signals", len(evaluable) >= policy.minimum_evaluable_signals, int(len(evaluable))),
        ("signal_slice_nonregression", np.isfinite(signal_min) and signal_min >= -policy.signal_slice_pr_auc_margin, signal_min),
        ("signal_macro_nonregression", np.isfinite(signal_macro) and signal_macro >= policy.minimum_signal_macro_delta, signal_macro),
        ("signal_improvement_fraction", np.isfinite(signal_improvement) and signal_improvement >= policy.minimum_signal_improvement_fraction, signal_improvement),
    ]
    checks = pd.DataFrame(
        [{"check": name, "passed": bool(passed), "value": value} for name, passed, value in check_specs]
    )
    passed = bool(checks["passed"].all())
    summary = {
        "delta_pr_auc": float(pure["delta_pr_auc"]),
        "delta_pr_auc_ci_low": float(pure["delta_pr_auc_ci_low"]),
        "delta_pr_auc_ci_high": float(pure["delta_pr_auc_ci_high"]),
        "delta_top_lift": float(full["delta_top_lift"]),
        "delta_terminal_return": float(full["delta_daily_top_k_avg_terminal_return"]),
        "independent_coverage": coverage,
        "evaluable_signals": int(len(evaluable)),
        "signal_min_delta_pr_auc": signal_min,
        "signal_macro_delta_pr_auc": signal_macro,
        "signal_improvement_fraction": signal_improvement,
    }
    return CConfirmationResult(
        decision="confirmed_for_shadow" if passed else "rejected_after_c",
        candidate=candidate,
        scope=scope,
        checks=checks,
        signal_deltas=signal_delta,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-ab", type=Path, required=True)
    parser.add_argument("--paired-c", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--signal-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_ab.read_text(encoding="utf-8"))
    result = evaluate_c_confirmation(
        frozen,
        pd.read_csv(args.paired_c),
        pd.read_csv(args.metrics),
        pd.read_csv(args.signal_metrics),
    )
    payload = c_confirmation_payload(result)
    atomic_write_json(payload, args.output)
    atomic_write_csv(result.checks, args.output.with_name(args.output.stem + "_checks.csv"), index=False)
    atomic_write_csv(result.signal_deltas, args.output.with_name(args.output.stem + "_signal_deltas.csv"), index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
