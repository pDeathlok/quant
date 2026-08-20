"""Freeze an A/B right-side model configuration before the untouched C fold.

The command consumes only saved development-fold artifacts.  It does not read
features, labels, models, or the C-fold predictions and never trains a model.

The primary comparison is paired PR-AUC on rows where an independent member
model exists.  Full-pool ranking/return, member-slice degradation, and the
independent-model coverage are explicit guardrails.  Passing this policy means
only "freeze this A/B choice and evaluate it once on C"; it is not production
approval and not a portfolio-backtest conclusion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.research.right_side_targets import (
    target_metadata,
    validate_persisted_target_metadata,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS


POLICY_VERSION = "right-side-ab-freeze-v1"
DEVELOPMENT_FOLDS: tuple[str, ...] = ("A", "B")
DEFAULT_CANDIDATES: tuple[str, ...] = (
    "unified_with_signal_id",
    "unified_balanced",
    "unified_long_task",
    "unified_long_task_balanced",
    "unified_long_task_deep",
)


@dataclass(frozen=True)
class ABFreezePolicy:
    """Pre-registered margins for advancing one frozen choice to fold C."""

    minimum_confidence_level: float = 0.95
    minimum_month_blocks: int = 6
    minimum_bootstrap_valid: int = 400
    minimum_independent_coverage: float = 0.80
    minimum_mean_pr_auc_delta: float = 0.005
    pr_auc_ci_noninferiority_margin: float = 0.005
    all_event_pr_auc_margin: float = 0.005
    all_event_top_lift_margin: float = 0.05
    all_event_return_margin: float = 0.002
    minimum_signal_rows: int = 200
    minimum_signal_class_rows: int = 20
    minimum_evaluable_signals_per_fold: int = 6
    minimum_crossfold_signals: int = 6
    signal_slice_pr_auc_margin: float = 0.020
    signal_crossfold_pr_auc_margin: float = 0.010
    minimum_signal_improvement_fraction: float = 0.50


DEFAULT_POLICY = ABFreezePolicy()


@dataclass(frozen=True)
class FreezeEvaluation:
    """Tables and top-level decision produced by :func:`evaluate_ab_freeze`."""

    decision: str
    selected_candidate: str | None
    candidate_summary: pd.DataFrame
    checks: pd.DataFrame
    signal_deltas: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _select_scope(
    frame: pd.DataFrame,
    *,
    entry_mode: str,
    horizon: int,
    label: str,
) -> pd.DataFrame:
    return frame[
        frame["entry_mode"].astype(str).eq(entry_mode)
        & pd.to_numeric(frame["horizon"], errors="coerce").eq(int(horizon))
        & frame["label"].astype(str).eq(label)
        & frame["fold"].astype(str).isin(DEVELOPMENT_FOLDS)
    ].copy()


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _fold_values(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        fold: float(frame.loc[frame["fold"].eq(fold), column].iloc[0])
        for fold in DEVELOPMENT_FOLDS
    }


def _paired_candidate_rows(
    paired: pd.DataFrame,
    candidate: str,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    selected = paired[paired["candidate"].astype(str).eq(candidate)].copy()
    pure = selected[selected["comparison_scope"].eq("independent_model_rows")].copy()
    full = selected[selected["comparison_scope"].eq("all_events")].copy()
    key = ["fold"]
    unique = not pure.duplicated(key).any() and not full.duplicated(key).any()
    complete = (
        unique
        and set(pure["fold"].astype(str)) == set(DEVELOPMENT_FOLDS)
        and set(full["fold"].astype(str)) == set(DEVELOPMENT_FOLDS)
        and len(pure) == len(DEVELOPMENT_FOLDS)
        and len(full) == len(DEVELOPMENT_FOLDS)
    )
    return pure, full, complete


def _independent_coverage(metrics: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    selected = metrics[
        metrics["experiment"].astype(str).eq("independent")
        & metrics["signal"].astype(str).eq("ALL")
    ].copy()
    complete = (
        not selected.duplicated(["fold"]).any()
        and set(selected["fold"].astype(str)) == set(DEVELOPMENT_FOLDS)
        and len(selected) == len(DEVELOPMENT_FOLDS)
    )
    selected = _numeric(selected, ("rows", "fallback_rows"))
    selected["independent_coverage"] = np.where(
        selected["rows"].gt(0),
        1.0 - selected["fallback_rows"].fillna(0.0).div(selected["rows"]),
        np.nan,
    )
    return selected, bool(complete)


def _signal_comparison(
    signal_metrics: pd.DataFrame,
    candidate: str,
    policy: ABFreezePolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["fold", "signal", "rows", "positives", "average_precision"]
    baseline = signal_metrics[
        signal_metrics["experiment"].astype(str).eq("independent")
    ][columns].copy()
    contender = signal_metrics[
        signal_metrics["experiment"].astype(str).eq(candidate)
    ][columns].copy()
    baseline = baseline.rename(
        columns={
            "rows": "baseline_rows",
            "positives": "baseline_positives",
            "average_precision": "baseline_average_precision",
        }
    )
    contender = contender.rename(
        columns={
            "rows": "candidate_rows",
            "positives": "candidate_positives",
            "average_precision": "candidate_average_precision",
        }
    )
    duplicate = baseline.duplicated(["fold", "signal"]).any() or contender.duplicated(
        ["fold", "signal"]
    ).any()
    compared = baseline.merge(
        contender,
        on=["fold", "signal"],
        how="outer",
        validate="one_to_one" if not duplicate else None,
        indicator=True,
    )
    compared = _numeric(
        compared,
        (
            "baseline_rows",
            "baseline_positives",
            "baseline_average_precision",
            "candidate_rows",
            "candidate_positives",
            "candidate_average_precision",
        ),
    )
    compared["candidate"] = candidate
    compared["counts_match"] = (
        compared["baseline_rows"].eq(compared["candidate_rows"])
        & compared["baseline_positives"].eq(compared["candidate_positives"])
    )
    negatives = compared["baseline_rows"] - compared["baseline_positives"]
    compared["evaluable"] = (
        compared["_merge"].eq("both")
        & compared["counts_match"]
        & compared["baseline_rows"].ge(policy.minimum_signal_rows)
        & compared["baseline_positives"].ge(policy.minimum_signal_class_rows)
        & negatives.ge(policy.minimum_signal_class_rows)
        & np.isfinite(compared["baseline_average_precision"])
        & np.isfinite(compared["candidate_average_precision"])
    )
    compared["delta_average_precision"] = (
        compared["candidate_average_precision"]
        - compared["baseline_average_precision"]
    )

    expected = {
        (fold, signal)
        for fold in DEVELOPMENT_FOLDS
        for signal in RIGHT_SIDE_SIGNALS
    }
    actual = set(zip(compared["fold"].astype(str), compared["signal"].astype(str)))
    output_complete = not duplicate and expected <= actual and compared["counts_match"].all()

    eligible = compared[compared["evaluable"]].copy()
    fold_summary = eligible.groupby("fold")["delta_average_precision"].agg(
        evaluable_signals="size",
        macro_delta="mean",
        worst_slice_delta="min",
    )
    crossfold = (
        eligible.groupby("signal")["delta_average_precision"]
        .agg(evaluable_folds="size", crossfold_delta="mean")
        .reset_index()
    )
    crossfold = crossfold[crossfold["evaluable_folds"].eq(len(DEVELOPMENT_FOLDS))]

    evaluable_by_fold = {
        fold: int(fold_summary.loc[fold, "evaluable_signals"])
        if fold in fold_summary.index
        else 0
        for fold in DEVELOPMENT_FOLDS
    }
    macro_by_fold = {
        fold: float(fold_summary.loc[fold, "macro_delta"])
        if fold in fold_summary.index
        else np.nan
        for fold in DEVELOPMENT_FOLDS
    }
    worst_slice = (
        float(eligible["delta_average_precision"].min())
        if not eligible.empty
        else np.nan
    )
    crossfold_count = int(len(crossfold))
    worst_crossfold = (
        float(crossfold["crossfold_delta"].min())
        if not crossfold.empty
        else np.nan
    )
    improvement_fraction = (
        float(crossfold["crossfold_delta"].ge(0.0).mean())
        if not crossfold.empty
        else np.nan
    )
    diagnostics = {
        "output_complete": bool(output_complete),
        "evaluable_by_fold": evaluable_by_fold,
        "macro_by_fold": macro_by_fold,
        "worst_slice_delta": worst_slice,
        "crossfold_signal_count": crossfold_count,
        "worst_crossfold_delta": worst_crossfold,
        "improvement_fraction": improvement_fraction,
        "unevaluable_signals": sorted(
            set(RIGHT_SIDE_SIGNALS) - set(crossfold["signal"].astype(str))
        ),
    }
    return compared.drop(columns="_merge"), diagnostics


def evaluate_ab_freeze(
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    *,
    entry_mode: str,
    horizon: int,
    label: str,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    policy: ABFreezePolicy = DEFAULT_POLICY,
) -> FreezeEvaluation:
    """Evaluate A/B only and select at most one arm to freeze for fold C."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    _require_columns(
        paired,
        {
            "comparison_scope",
            "entry_mode",
            "horizon",
            "label",
            "fold",
            "candidate",
            "status",
            "paired_rows",
            "month_blocks",
            "confidence_level",
            "delta_pr_auc",
            "delta_pr_auc_ci_low",
            "delta_top_lift",
            "delta_daily_top_k_avg_terminal_return",
            "delta_pr_auc_bootstrap_valid",
        },
        name="paired",
    )
    _require_columns(
        metrics,
        {
            "entry_mode",
            "horizon",
            "label",
            "fold",
            "experiment",
            "signal",
            "rows",
            "fallback_rows",
        },
        name="metrics",
    )
    _require_columns(
        signal_metrics,
        {
            "entry_mode",
            "horizon",
            "label",
            "fold",
            "experiment",
            "signal",
            "rows",
            "positives",
            "average_precision",
        },
        name="signal_metrics",
    )

    paired_scope = _select_scope(
        paired,
        entry_mode=entry_mode,
        horizon=horizon,
        label=label,
    )
    metrics_scope = _select_scope(
        metrics,
        entry_mode=entry_mode,
        horizon=horizon,
        label=label,
    )
    signal_scope = _select_scope(
        signal_metrics,
        entry_mode=entry_mode,
        horizon=horizon,
        label=label,
    )
    validate_persisted_target_metadata(metrics_scope, label)
    paired_scope = _numeric(
        paired_scope,
        (
            "paired_rows",
            "month_blocks",
            "confidence_level",
            "delta_pr_auc",
            "delta_pr_auc_ci_low",
            "delta_top_lift",
            "delta_daily_top_k_avg_terminal_return",
            "delta_pr_auc_bootstrap_valid",
        ),
    )
    coverage, coverage_complete = _independent_coverage(metrics_scope)
    coverage_by_fold = (
        _fold_values(coverage, "independent_coverage")
        if coverage_complete
        else {fold: np.nan for fold in DEVELOPMENT_FOLDS}
    )

    check_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    signal_parts: list[pd.DataFrame] = []

    for candidate in dict.fromkeys(str(value) for value in candidates):
        pure, full, paired_complete = _paired_candidate_rows(paired_scope, candidate)
        signal_delta, signal_info = _signal_comparison(signal_scope, candidate, policy)
        signal_parts.append(signal_delta)
        candidate_checks: list[dict[str, Any]] = []

        def add_check(name: str, passed: bool, value: Any, requirement: str) -> None:
            row = {
                "entry_mode": entry_mode,
                "horizon": int(horizon),
                "label": label,
                "candidate": candidate,
                "check": name,
                "passed": bool(passed),
                "value": value,
                "requirement": requirement,
            }
            candidate_checks.append(row)
            check_rows.append(row)

        add_check(
            "paired_ab_complete",
            paired_complete,
            {"pure_rows": len(pure), "all_event_rows": len(full)},
            "A/B each have one independent_model_rows and all_events row",
        )
        add_check(
            "independent_coverage_complete",
            coverage_complete,
            coverage_by_fold,
            "A/B each have one independent ALL metric row",
        )

        primary_delta = {fold: np.nan for fold in DEVELOPMENT_FOLDS}
        primary_ci_low = {fold: np.nan for fold in DEVELOPMENT_FOLDS}
        all_pr_delta = {fold: np.nan for fold in DEVELOPMENT_FOLDS}
        all_lift_delta = {fold: np.nan for fold in DEVELOPMENT_FOLDS}
        all_return_delta = {fold: np.nan for fold in DEVELOPMENT_FOLDS}
        paired_coverage = {fold: np.nan for fold in DEVELOPMENT_FOLDS}
        if paired_complete:
            primary_delta = _fold_values(pure, "delta_pr_auc")
            primary_ci_low = _fold_values(pure, "delta_pr_auc_ci_low")
            all_pr_delta = _fold_values(full, "delta_pr_auc")
            all_lift_delta = _fold_values(full, "delta_top_lift")
            all_return_delta = _fold_values(
                full, "delta_daily_top_k_avg_terminal_return"
            )
            full_rows = _fold_values(full, "paired_rows")
            pure_rows = _fold_values(pure, "paired_rows")
            paired_coverage = {
                fold: pure_rows[fold] / full_rows[fold]
                if full_rows[fold] > 0
                else np.nan
                for fold in DEVELOPMENT_FOLDS
            }
            ci_quality = (
                pure["status"].astype(str).eq("ok").all()
                and pure["confidence_level"].ge(policy.minimum_confidence_level).all()
                and pure["month_blocks"].ge(policy.minimum_month_blocks).all()
                and pure["delta_pr_auc_bootstrap_valid"]
                .ge(policy.minimum_bootstrap_valid)
                .all()
                and np.isfinite(pure["delta_pr_auc_ci_low"]).all()
            )
        else:
            ci_quality = False

        effective_coverage = {
            fold: min(coverage_by_fold[fold], paired_coverage[fold])
            if np.isfinite(coverage_by_fold[fold])
            and np.isfinite(paired_coverage[fold])
            else np.nan
            for fold in DEVELOPMENT_FOLDS
        }
        add_check(
            "primary_ci_quality",
            bool(ci_quality),
            {
                "minimum_confidence_level": (
                    float(pure["confidence_level"].min()) if not pure.empty else np.nan
                ),
                "minimum_month_blocks": (
                    int(pure["month_blocks"].min()) if not pure.empty else 0
                ),
                "minimum_bootstrap_valid": (
                    int(pure["delta_pr_auc_bootstrap_valid"].min())
                    if not pure.empty
                    else 0
                ),
            },
            "status=ok, CI>=95%, >=6 month blocks, >=400 valid bootstrap draws",
        )
        add_check(
            "independent_coverage",
            all(
                np.isfinite(value) and value >= policy.minimum_independent_coverage
                for value in effective_coverage.values()
            ),
            effective_coverage,
            f"effective independent-model coverage >= {policy.minimum_independent_coverage:.0%} in A and B",
        )
        add_check(
            "primary_pr_positive_each_fold",
            all(np.isfinite(value) and value > 0 for value in primary_delta.values()),
            primary_delta,
            "paired delta PR-AUC > 0 in both A and B on independent-model rows",
        )
        primary_mean = float(np.mean(list(primary_delta.values())))
        add_check(
            "primary_pr_mean_material",
            np.isfinite(primary_mean)
            and primary_mean >= policy.minimum_mean_pr_auc_delta,
            primary_mean,
            f"equal-fold mean paired delta PR-AUC >= {policy.minimum_mean_pr_auc_delta:.3f}",
        )
        add_check(
            "primary_pr_ci_noninferior_each_fold",
            all(
                np.isfinite(value)
                and value >= -policy.pr_auc_ci_noninferiority_margin
                for value in primary_ci_low.values()
            ),
            primary_ci_low,
            f"each 95% CI lower bound >= -{policy.pr_auc_ci_noninferiority_margin:.3f}",
        )
        add_check(
            "primary_pr_ci_superior_one_fold",
            any(np.isfinite(value) and value > 0 for value in primary_ci_low.values()),
            primary_ci_low,
            "at least one fold has 95% CI lower bound > 0",
        )
        add_check(
            "all_event_pr_guardrail",
            all(
                np.isfinite(value) and value >= -policy.all_event_pr_auc_margin
                for value in all_pr_delta.values()
            )
            and np.isfinite(np.mean(list(all_pr_delta.values())))
            and np.mean(list(all_pr_delta.values())) >= 0,
            all_pr_delta,
            f"each fold >= -{policy.all_event_pr_auc_margin:.3f} and A/B mean >= 0",
        )
        add_check(
            "all_event_top_lift_guardrail",
            all(
                np.isfinite(value) and value >= -policy.all_event_top_lift_margin
                for value in all_lift_delta.values()
            )
            and np.isfinite(np.mean(list(all_lift_delta.values())))
            and np.mean(list(all_lift_delta.values())) >= 0,
            all_lift_delta,
            f"each fold >= -{policy.all_event_top_lift_margin:.2f} and A/B mean >= 0",
        )
        add_check(
            "all_event_return_guardrail",
            all(
                np.isfinite(value) and value >= -policy.all_event_return_margin
                for value in all_return_delta.values()
            )
            and np.isfinite(np.mean(list(all_return_delta.values())))
            and np.mean(list(all_return_delta.values())) >= 0,
            all_return_delta,
            f"each fold >= -{policy.all_event_return_margin:.3f} and A/B mean >= 0",
        )
        add_check(
            "signal_output_complete",
            signal_info["output_complete"],
            {"expected_signal_fold_rows": len(RIGHT_SIDE_SIGNALS) * 2},
            "candidate and independent contain matching A/B rows for all 14 signals",
        )
        add_check(
            "signal_evaluable_coverage",
            all(
                count >= policy.minimum_evaluable_signals_per_fold
                for count in signal_info["evaluable_by_fold"].values()
            )
            and signal_info["crossfold_signal_count"]
            >= policy.minimum_crossfold_signals,
            {
                "by_fold": signal_info["evaluable_by_fold"],
                "crossfold": signal_info["crossfold_signal_count"],
                "unevaluable": signal_info["unevaluable_signals"],
            },
            "at least 6 evaluable signals per fold and 6 evaluable in both folds",
        )
        add_check(
            "signal_slice_nonregression",
            np.isfinite(signal_info["worst_slice_delta"])
            and signal_info["worst_slice_delta"]
            >= -policy.signal_slice_pr_auc_margin,
            signal_info["worst_slice_delta"],
            f"no evaluable signal/fold delta PR-AUC < -{policy.signal_slice_pr_auc_margin:.3f}",
        )
        add_check(
            "signal_macro_nonregression",
            all(
                np.isfinite(value) and value >= 0
                for value in signal_info["macro_by_fold"].values()
            ),
            signal_info["macro_by_fold"],
            "equal-signal macro delta PR-AUC >= 0 in both A and B",
        )
        add_check(
            "signal_crossfold_nonregression",
            np.isfinite(signal_info["worst_crossfold_delta"])
            and signal_info["worst_crossfold_delta"]
            >= -policy.signal_crossfold_pr_auc_margin,
            signal_info["worst_crossfold_delta"],
            f"no evaluable signal A/B mean delta PR-AUC < -{policy.signal_crossfold_pr_auc_margin:.3f}",
        )
        add_check(
            "signal_improvement_breadth",
            np.isfinite(signal_info["improvement_fraction"])
            and signal_info["improvement_fraction"]
            >= policy.minimum_signal_improvement_fraction,
            signal_info["improvement_fraction"],
            f">= {policy.minimum_signal_improvement_fraction:.0%} of crossfold-evaluable signals have mean delta >= 0",
        )

        passed = all(bool(row["passed"]) for row in candidate_checks)
        summaries.append(
            {
                "entry_mode": entry_mode,
                "horizon": int(horizon),
                "label": label,
                "candidate": candidate,
                "freeze_ready": passed,
                "failed_checks": int(sum(not row["passed"] for row in candidate_checks)),
                "mean_primary_delta_pr_auc": primary_mean,
                "minimum_primary_pr_auc_ci_low": (
                    float(np.min(list(primary_ci_low.values())))
                    if all(np.isfinite(list(primary_ci_low.values())))
                    else np.nan
                ),
                "mean_all_event_delta_pr_auc": float(np.mean(list(all_pr_delta.values()))),
                "mean_all_event_delta_top_lift": float(np.mean(list(all_lift_delta.values()))),
                "mean_all_event_delta_terminal_return": float(
                    np.mean(list(all_return_delta.values()))
                ),
                "minimum_independent_coverage": (
                    float(np.min(list(effective_coverage.values())))
                    if all(np.isfinite(list(effective_coverage.values())))
                    else np.nan
                ),
                "evaluable_crossfold_signals": signal_info["crossfold_signal_count"],
                "signal_improvement_fraction": signal_info["improvement_fraction"],
            }
        )

    summary = pd.DataFrame(summaries)
    ready = summary[summary["freeze_ready"]].copy()
    if ready.empty:
        decision = "do_not_advance_to_c"
        selected_candidate = None
    else:
        ready = ready.sort_values(
            [
                "minimum_primary_pr_auc_ci_low",
                "mean_primary_delta_pr_auc",
                "mean_all_event_delta_terminal_return",
                "candidate",
            ],
            ascending=[False, False, False, True],
            kind="stable",
        )
        decision = "freeze_for_c_confirmation"
        selected_candidate = str(ready.iloc[0]["candidate"])
    summary["selected_for_c"] = summary["candidate"].eq(selected_candidate)
    checks = pd.DataFrame(check_rows)
    signal_deltas = (
        pd.concat(signal_parts, ignore_index=True, sort=False)
        if signal_parts
        else pd.DataFrame()
    )
    return FreezeEvaluation(
        decision=decision,
        selected_candidate=selected_candidate,
        candidate_summary=summary,
        checks=checks,
        signal_deltas=signal_deltas,
    )


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        records.append(
            {
                key: None
                if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                else value.item()
                if isinstance(value, np.generic)
                else value
                for key, value in row.items()
            }
        )
    return records


def freeze_payload(
    evaluation: FreezeEvaluation,
    *,
    entry_mode: str,
    horizon: int,
    label: str,
    policy: ABFreezePolicy = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Return a JSON-safe audit payload for one pre-declared configuration."""

    failed = evaluation.checks[~evaluation.checks["passed"].astype(bool)]
    return {
        "policy_version": POLICY_VERSION,
        "development_folds": list(DEVELOPMENT_FOLDS),
        "scope": {
            "entry_mode": entry_mode,
            "horizon": int(horizon),
            "label": label,
        },
        "target": {
            "name": label,
            **target_metadata(label),
        },
        "decision": evaluation.decision,
        "selected_candidate": evaluation.selected_candidate,
        "policy": asdict(policy),
        "candidate_summary": _json_records(evaluation.candidate_summary),
        "failed_checks": _json_records(failed),
        "limitations": [
            "A/B pass only freezes one arm for a single untouched C-fold confirmation; it is not production approval.",
            "Per-signal guardrails are margin based and have no member-level confidence intervals.",
            "Signals below the row/class minimum are listed as unevaluable and require C/shadow monitoring.",
            "Top-K terminal returns are overlapping trade-level outcomes, not a capital curve or drawdown estimate.",
        ],
    }


def parse_args() -> argparse.Namespace:
    root = Path("reports/research/right_side_unified")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, default=root / "paired_model_comparison.csv")
    parser.add_argument("--metrics", type=Path, default=root / "model_metrics.csv")
    parser.add_argument("--signal-metrics", type=Path, default=root / "signal_metrics.csv")
    parser.add_argument("--entry-mode", choices=["next_open", "next_close"], default="next_open")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--label", default="good_path5")
    parser.add_argument("--output-root", type=Path, default=root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation = evaluate_ab_freeze(
        pd.read_csv(args.paired),
        pd.read_csv(args.metrics),
        pd.read_csv(args.signal_metrics),
        entry_mode=args.entry_mode,
        horizon=args.horizon,
        label=args.label,
    )
    stem = f"ab_freeze_{args.entry_mode}_h{args.horizon}_{args.label}"
    payload = freeze_payload(
        evaluation,
        entry_mode=args.entry_mode,
        horizon=args.horizon,
        label=args.label,
    )
    atomic_write_json(payload, args.output_root / f"{stem}.json")
    atomic_write_csv(
        evaluation.candidate_summary,
        args.output_root / f"{stem}_candidates.csv",
        index=False,
    )
    atomic_write_csv(
        evaluation.checks,
        args.output_root / f"{stem}_checks.csv",
        index=False,
    )
    atomic_write_csv(
        evaluation.signal_deltas,
        args.output_root / f"{stem}_signal_deltas.csv",
        index=False,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()


__all__ = [
    "ABFreezePolicy",
    "DEFAULT_CANDIDATES",
    "DEFAULT_POLICY",
    "DEVELOPMENT_FOLDS",
    "FreezeEvaluation",
    "POLICY_VERSION",
    "evaluate_ab_freeze",
    "freeze_payload",
]
