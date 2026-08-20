"""Paired test-set comparisons for unified right-side model research.

The helpers in this module operate only on saved out-of-sample prediction
rows.  They never refit a model or choose a threshold from the test set.  The
confidence intervals resample whole calendar months so that same-day
cross-sectional observations and nearby market conditions stay together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from quant.research.right_side_unified import binary_metrics


DEFAULT_PAIRED_CANDIDATES: tuple[str, ...] = (
    "unified_with_signal_id",
    "unified_balanced",
)
DEFAULT_PAIRED_BASELINE = "independent"
DEFAULT_PAIRED_GROUP_COLUMNS: tuple[str, ...] = (
    "entry_mode",
    "horizon",
    "label",
    "fold",
)


@dataclass(frozen=True)
class _RankInputs:
    """Arrays sorted once for repeated month-weighted rank metrics."""

    labels: np.ndarray
    block_ids: np.ndarray
    threshold_indices: np.ndarray
    constant_probability: bool


def _prepare_rank_inputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
    block_ids: np.ndarray,
) -> _RankInputs:
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    if len(sorted_probabilities) == 1:
        threshold_indices = np.array([0], dtype=np.int64)
    else:
        distinct_ends = np.flatnonzero(np.diff(sorted_probabilities) != 0)
        threshold_indices = np.r_[distinct_ends, len(sorted_probabilities) - 1]
    return _RankInputs(
        labels=labels[order].astype(np.int8, copy=False),
        block_ids=block_ids[order].astype(np.int16, copy=False),
        threshold_indices=threshold_indices.astype(np.int64, copy=False),
        constant_probability=bool(np.allclose(probabilities, probabilities[0])),
    )


def _weighted_rank_metrics(
    inputs: _RankInputs,
    block_counts: np.ndarray,
    *,
    top_fraction: float,
) -> tuple[float, float]:
    """Return PR-AUC and Top-fraction lift after whole-block replication."""

    weights = block_counts[inputs.block_ids].astype(float, copy=False)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return np.nan, np.nan
    weighted_positives = weights * inputs.labels
    total_positives = float(weighted_positives.sum())
    base_rate = total_positives / total_weight

    cumulative_weight = np.cumsum(weights)
    cumulative_positives = np.cumsum(weighted_positives)
    top_n = max(1, int(np.ceil(total_weight * top_fraction)))
    boundary = int(np.searchsorted(cumulative_weight, top_n, side="left"))
    prior_weight = float(cumulative_weight[boundary - 1]) if boundary else 0.0
    prior_positives = float(cumulative_positives[boundary - 1]) if boundary else 0.0
    boundary_take = max(0.0, min(float(weights[boundary]), top_n - prior_weight))
    selected_positives = prior_positives + boundary_take * float(inputs.labels[boundary])
    top_precision = selected_positives / float(top_n)
    if inputs.constant_probability:
        top_precision = base_rate
    top_lift = top_precision / max(base_rate, 1e-12)

    # This is the non-interpolated weighted average precision used by sklearn:
    # sum of precision at each score threshold times its recall increment.
    if total_positives <= 0 or total_positives >= total_weight:
        average_precision = np.nan
    else:
        threshold_true_positives = cumulative_positives[inputs.threshold_indices]
        threshold_totals = cumulative_weight[inputs.threshold_indices]
        precision = np.divide(
            threshold_true_positives,
            threshold_totals,
            out=np.zeros_like(threshold_true_positives),
            where=threshold_totals > 0,
        )
        positive_increments = np.diff(np.r_[0.0, threshold_true_positives])
        average_precision = float(
            np.dot(positive_increments / total_positives, precision)
        )
    return float(average_precision), float(top_lift)


def _daily_top_k_return_totals(
    dates: pd.Series,
    block_ids: np.ndarray,
    terminal_returns: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_k: int,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute selected return sums/counts for every calendar-month block."""

    ranked = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "block_id": block_ids,
            "terminal_return": terminal_returns,
            "probability": probabilities,
            "row_order": np.arange(len(dates)),
        }
    )
    selected = (
        ranked.sort_values(
            ["date", "probability", "row_order"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("date", sort=True, group_keys=False)
        .head(top_k)
    )
    totals = selected.groupby("block_id")["terminal_return"].agg(["sum", "size"])
    return (
        totals["sum"].reindex(range(block_count), fill_value=0.0).to_numpy(dtype=float),
        totals["size"].reindex(range(block_count), fill_value=0).to_numpy(dtype=float),
    )


def _percentile_interval(
    values: Sequence[float],
    confidence_level: float,
) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.nan, np.nan, 0
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high), int(len(finite))


def paired_model_comparisons(
    predictions: pd.DataFrame,
    *,
    candidate_experiments: Sequence[str] = DEFAULT_PAIRED_CANDIDATES,
    baseline_experiment: str = DEFAULT_PAIRED_BASELINE,
    group_columns: Sequence[str] = DEFAULT_PAIRED_GROUP_COLUMNS,
    prediction_prefix: str = "pred_",
    date_column: str = "date",
    terminal_return_column: str = "terminal_return",
    top_fraction: float = 0.10,
    daily_top_k: int = 10,
    bootstrap_iterations: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare unified arms with the independent arm on identical test rows.

    The observed deltas are exact full-fold test-set differences.  Confidence
    intervals use a paired calendar-month cluster bootstrap: a sampled month is
    replicated in both model arms, preserving its complete set of days and
    cross-sectional candidates.  Model fitting and threshold selection are not
    repeated, so this measures evaluation uncertainty rather than training
    uncertainty.
    """

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if daily_top_k <= 0:
        raise ValueError("daily_top_k must be positive")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if not candidate_experiments:
        raise ValueError("candidate_experiments must not be empty")

    baseline_column = f"{prediction_prefix}{baseline_experiment}"
    candidate_columns = {
        experiment: f"{prediction_prefix}{experiment}"
        for experiment in candidate_experiments
    }
    required = {
        *group_columns,
        date_column,
        terminal_return_column,
        baseline_column,
        *candidate_columns.values(),
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing paired-comparison columns: {sorted(missing)}")

    work = predictions.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    grouped = work.groupby(list(group_columns), sort=True, dropna=False)
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        scope = dict(zip(group_columns, keys))
        label_name = str(scope.get("label", ""))
        if not label_name or label_name not in group.columns:
            raise ValueError(f"prediction table missing target column for label={label_name!r}")

        for candidate, candidate_column in candidate_columns.items():
            paired = group[
                [
                    date_column,
                    label_name,
                    terminal_return_column,
                    baseline_column,
                    candidate_column,
                ]
            ].copy()
            for column in (
                label_name,
                terminal_return_column,
                baseline_column,
                candidate_column,
            ):
                paired[column] = pd.to_numeric(paired[column], errors="coerce")
            paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
            status = "ok"
            if paired.empty:
                rows.append(
                    {
                        **scope,
                        "candidate": candidate,
                        "baseline": baseline_experiment,
                        "status": "no_paired_rows",
                        "paired_rows": 0,
                        "month_blocks": 0,
                    }
                )
                continue

            labels = paired[label_name].to_numpy(dtype=int)
            baseline_probability = paired[baseline_column].to_numpy(dtype=float)
            candidate_probability = paired[candidate_column].to_numpy(dtype=float)
            terminal_returns = paired[terminal_return_column].to_numpy(dtype=float)
            block_period = paired[date_column].dt.to_period("M")
            block_ids, unique_blocks = pd.factorize(block_period, sort=True)
            block_ids = block_ids.astype(np.int16, copy=False)
            block_count = len(unique_blocks)

            baseline_binary = binary_metrics(
                labels,
                baseline_probability,
                top_fraction=top_fraction,
            )
            candidate_binary = binary_metrics(
                labels,
                candidate_probability,
                top_fraction=top_fraction,
            )
            baseline_top_k_sum, baseline_top_k_count = _daily_top_k_return_totals(
                paired[date_column],
                block_ids,
                terminal_returns,
                baseline_probability,
                top_k=daily_top_k,
                block_count=block_count,
            )
            candidate_top_k_sum, candidate_top_k_count = _daily_top_k_return_totals(
                paired[date_column],
                block_ids,
                terminal_returns,
                candidate_probability,
                top_k=daily_top_k,
                block_count=block_count,
            )
            baseline_top_k_return = float(
                baseline_top_k_sum.sum() / baseline_top_k_count.sum()
            )
            candidate_top_k_return = float(
                candidate_top_k_sum.sum() / candidate_top_k_count.sum()
            )

            observed = {
                "pr_auc": (
                    float(candidate_binary["average_precision"]),
                    float(baseline_binary["average_precision"]),
                ),
                "top_lift": (
                    float(candidate_binary["top_lift"]),
                    float(baseline_binary["top_lift"]),
                ),
                "daily_top_k_avg_terminal_return": (
                    candidate_top_k_return,
                    baseline_top_k_return,
                ),
            }
            bootstrap_deltas: dict[str, list[float]] = {
                metric: [] for metric in observed
            }
            if bootstrap_iterations and block_count >= 2:
                baseline_rank = _prepare_rank_inputs(
                    labels,
                    baseline_probability,
                    block_ids,
                )
                candidate_rank = _prepare_rank_inputs(
                    labels,
                    candidate_probability,
                    block_ids,
                )
                draws = rng.integers(
                    0,
                    block_count,
                    size=(bootstrap_iterations, block_count),
                )
                for draw in draws:
                    block_counts = np.bincount(draw, minlength=block_count)
                    baseline_ap, baseline_lift = _weighted_rank_metrics(
                        baseline_rank,
                        block_counts,
                        top_fraction=top_fraction,
                    )
                    candidate_ap, candidate_lift = _weighted_rank_metrics(
                        candidate_rank,
                        block_counts,
                        top_fraction=top_fraction,
                    )
                    baseline_count = float(np.dot(block_counts, baseline_top_k_count))
                    candidate_count = float(np.dot(block_counts, candidate_top_k_count))
                    baseline_return = (
                        float(np.dot(block_counts, baseline_top_k_sum) / baseline_count)
                        if baseline_count > 0
                        else np.nan
                    )
                    candidate_return = (
                        float(np.dot(block_counts, candidate_top_k_sum) / candidate_count)
                        if candidate_count > 0
                        else np.nan
                    )
                    bootstrap_deltas["pr_auc"].append(candidate_ap - baseline_ap)
                    bootstrap_deltas["top_lift"].append(candidate_lift - baseline_lift)
                    bootstrap_deltas["daily_top_k_avg_terminal_return"].append(
                        candidate_return - baseline_return
                    )
            elif block_count < 2:
                status = "insufficient_month_blocks_for_ci"
            else:
                status = "exact_only"

            result: dict[str, object] = {
                **scope,
                "candidate": candidate,
                "baseline": baseline_experiment,
                "status": status,
                "paired_rows": int(len(paired)),
                "month_blocks": int(block_count),
                "top_fraction": float(top_fraction),
                "daily_top_k": int(daily_top_k),
                "bootstrap_iterations": int(bootstrap_iterations),
                "confidence_level": float(confidence_level),
                "ci_method": "paired_calendar_month_cluster_bootstrap",
            }
            for metric, (candidate_value, baseline_value) in observed.items():
                low, high, valid_iterations = _percentile_interval(
                    bootstrap_deltas[metric],
                    confidence_level,
                )
                result.update(
                    {
                        f"candidate_{metric}": candidate_value,
                        f"baseline_{metric}": baseline_value,
                        f"delta_{metric}": candidate_value - baseline_value,
                        f"delta_{metric}_ci_low": low,
                        f"delta_{metric}_ci_high": high,
                        f"delta_{metric}_bootstrap_valid": valid_iterations,
                    }
                )
            rows.append(result)

    return pd.DataFrame(rows)
