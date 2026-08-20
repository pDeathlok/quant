from __future__ import annotations

import pandas as pd
import pytest

from quant.research.right_side_freeze_decision import (
    POLICY_VERSION,
    evaluate_ab_freeze,
    freeze_payload,
)
from quant.research.right_side_targets import (
    TERMINAL_NET_POSITIVE_15BPS,
    target_metadata,
)
from quant.research.right_side_unified import RIGHT_SIDE_SIGNALS


SCOPE = {
    "entry_mode": "next_open",
    "horizon": 5,
    "label": "good_path5",
}


def _paired() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in (
        "unified_with_signal_id",
        "unified_balanced",
        "unified_long_task",
        "unified_long_task_balanced",
        "unified_long_task_deep",
    ):
        for fold in ("A", "B", "C"):
            passing = candidate == "unified_with_signal_id"
            primary_delta = 0.012 if fold == "A" else 0.010
            primary_low = 0.002 if fold == "A" else -0.001
            if not passing:
                primary_delta = -0.010
                primary_low = -0.020
            if fold == "C":
                primary_delta = -0.50
                primary_low = -0.60
            for comparison_scope, paired_rows in (
                ("all_events", 1000),
                ("independent_model_rows", 900),
            ):
                rows.append(
                    {
                        **SCOPE,
                        "fold": fold,
                        "candidate": candidate,
                        "comparison_scope": comparison_scope,
                        "status": "ok",
                        "paired_rows": paired_rows,
                        "month_blocks": 12,
                        "confidence_level": 0.95,
                        "delta_pr_auc": (
                            primary_delta
                            if comparison_scope == "independent_model_rows"
                            else (0.009 if passing else -0.010)
                        ),
                        "delta_pr_auc_ci_low": primary_low,
                        "delta_top_lift": 0.08 if passing else -0.10,
                        "delta_daily_top_k_avg_terminal_return": (
                            0.003 if passing else -0.004
                        ),
                        "delta_pr_auc_bootstrap_valid": 500,
                    }
                )
    return pd.DataFrame(rows)


def _metrics(*, fallback_rows: int = 100) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in ("A", "B", "C"):
        rows.extend(
            [
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "independent",
                    "signal": "ALL",
                    "rows": 1000,
                    "fallback_rows": fallback_rows,
                    "brier": 0.20,
                },
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_with_signal_id",
                    "signal": "ALL",
                    "rows": 1000,
                    "fallback_rows": 0,
                    "brier": 0.19,
                },
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_balanced",
                    "signal": "ALL",
                    "rows": 1000,
                    "fallback_rows": 0,
                    "brier": 0.21,
                },
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_long_task",
                    "signal": "ALL",
                    "rows": 1000,
                    "fallback_rows": 0,
                    "brier": 0.21,
                },
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_long_task_balanced",
                    "signal": "ALL",
                    "rows": 1000,
                    "fallback_rows": 0,
                    "brier": 0.21,
                },
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_long_task_deep",
                    "signal": "ALL",
                    "rows": 1000,
                    "fallback_rows": 0,
                    "brier": 0.21,
                },
            ]
        )
    return pd.DataFrame(rows)


def _signal_metrics(*, bad_signal: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in ("A", "B", "C"):
        for signal in RIGHT_SIDE_SIGNALS:
            baseline = 0.20
            rows.append(
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "independent",
                    "signal": signal,
                    "rows": 300,
                    "positives": 60,
                    "average_precision": baseline,
                }
            )
            delta = -0.03 if signal == bad_signal and fold == "A" else 0.01
            rows.append(
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_with_signal_id",
                    "signal": signal,
                    "rows": 300,
                    "positives": 60,
                    "average_precision": baseline + delta,
                }
            )
            rows.append(
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_balanced",
                    "signal": signal,
                    "rows": 300,
                    "positives": 60,
                    "average_precision": baseline - 0.03,
                }
            )
            rows.append(
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_long_task_balanced",
                    "signal": signal,
                    "rows": 300,
                    "positives": 60,
                    "average_precision": baseline - 0.03,
                }
            )
            rows.append(
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_long_task_deep",
                    "signal": signal,
                    "rows": 300,
                    "positives": 60,
                    "average_precision": baseline - 0.03,
                }
            )
            rows.append(
                {
                    **SCOPE,
                    "fold": fold,
                    "experiment": "unified_long_task",
                    "signal": signal,
                    "rows": 300,
                    "positives": 60,
                    "average_precision": baseline - 0.03,
                }
            )
    return pd.DataFrame(rows)


def _evaluate(
    *,
    paired: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
    signal_metrics: pd.DataFrame | None = None,
):
    return evaluate_ab_freeze(
        _paired() if paired is None else paired,
        _metrics() if metrics is None else metrics,
        _signal_metrics() if signal_metrics is None else signal_metrics,
        **SCOPE,
    )


def test_freeze_selects_only_candidate_passing_all_ab_gates_and_ignores_c() -> None:
    result = _evaluate()

    assert result.decision == "freeze_for_c_confirmation"
    assert result.selected_candidate == "unified_with_signal_id"
    summary = result.candidate_summary.set_index("candidate")
    assert bool(summary.loc["unified_with_signal_id", "freeze_ready"])
    assert not bool(summary.loc["unified_balanced", "freeze_ready"])
    assert bool(summary.loc["unified_with_signal_id", "selected_for_c"])
    assert set(result.signal_deltas["fold"]) == {"A", "B"}


def test_low_independent_coverage_blocks_freeze() -> None:
    result = _evaluate(metrics=_metrics(fallback_rows=250))

    assert result.decision == "do_not_advance_to_c"
    coverage = result.checks[
        result.checks["check"].eq("independent_coverage")
        & result.checks["candidate"].eq("unified_with_signal_id")
    ].iloc[0]
    assert not bool(coverage["passed"])


def test_material_member_degradation_blocks_freeze() -> None:
    result = _evaluate(signal_metrics=_signal_metrics(bad_signal="B2"))

    candidate_checks = result.checks[
        result.checks["candidate"].eq("unified_with_signal_id")
    ].set_index("check")
    assert not bool(candidate_checks.loc["signal_slice_nonregression", "passed"])
    assert result.decision == "do_not_advance_to_c"


def test_missing_report_comparison_scope_fails_closed() -> None:
    with pytest.raises(ValueError, match="comparison_scope"):
        _evaluate(paired=_paired().drop(columns="comparison_scope"))


def test_payload_is_explicit_that_ab_pass_is_not_production_approval() -> None:
    result = _evaluate()
    payload = freeze_payload(result, **SCOPE)

    assert payload["policy_version"] == POLICY_VERSION
    assert payload["decision"] == "freeze_for_c_confirmation"
    assert "not production approval" in payload["limitations"][0]
    assert payload["failed_checks"]


def test_freezer_validates_and_persists_cost_aligned_target_metadata() -> None:
    paired = _paired().copy()
    metrics = _metrics().copy()
    signal_metrics = _signal_metrics().copy()
    for frame in (paired, metrics, signal_metrics):
        frame["label"] = TERMINAL_NET_POSITIVE_15BPS
    for column, value in target_metadata(TERMINAL_NET_POSITIVE_15BPS).items():
        metrics[column] = value

    evaluation = evaluate_ab_freeze(
        paired,
        metrics,
        signal_metrics,
        entry_mode="next_open",
        horizon=5,
        label=TERMINAL_NET_POSITIVE_15BPS,
    )
    payload = freeze_payload(
        evaluation,
        entry_mode="next_open",
        horizon=5,
        label=TERMINAL_NET_POSITIVE_15BPS,
    )

    assert evaluation.decision == "freeze_for_c_confirmation"
    assert set(evaluation.signal_deltas["fold"]) == {"A", "B"}
    assert payload["target"]["name"] == TERMINAL_NET_POSITIVE_15BPS
    assert payload["target"]["target_threshold_return"] == pytest.approx(0.0015)
    assert payload["target"]["target_cost_bps"] == pytest.approx(15.0)

    metrics.loc[metrics.index[0], "target_cost_bps"] = 10.0
    with pytest.raises(ValueError, match="target_cost_bps"):
        evaluate_ab_freeze(
            paired,
            metrics,
            signal_metrics,
            entry_mode="next_open",
            horizon=5,
            label=TERMINAL_NET_POSITIVE_15BPS,
        )
