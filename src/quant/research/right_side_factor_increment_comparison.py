"""Pure ranking comparison for the 118-rule-factor increment.

The comparison consumes saved out-of-sample scores on identical event rows.
It deliberately has no return/cost input: first-layer promotion is a ranking
decision, while execution economics belong to the downstream playbook model.
Uncertainty is estimated with a paired calendar-month block bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLUMNS: tuple[str, ...] = (
    "entry_mode",
    "horizon",
    "label",
    "fold",
)


@dataclass(frozen=True)
class _PreparedRanking:
    labels: np.ndarray
    block_ids: np.ndarray
    threshold_ends: np.ndarray
    constant_score: bool


def _prepare_ranking(
    labels: np.ndarray,
    scores: np.ndarray,
    block_ids: np.ndarray,
) -> _PreparedRanking:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    if len(sorted_scores) == 1:
        threshold_ends = np.array([0], dtype=np.int64)
    else:
        threshold_ends = np.r_[
            np.flatnonzero(np.diff(sorted_scores) != 0),
            len(sorted_scores) - 1,
        ].astype(np.int64, copy=False)
    return _PreparedRanking(
        labels=labels[order].astype(np.int8, copy=False),
        block_ids=block_ids[order].astype(np.int16, copy=False),
        threshold_ends=threshold_ends,
        constant_score=bool(np.allclose(scores, scores[0])),
    )


def _weighted_metrics(
    ranking: _PreparedRanking,
    block_counts: np.ndarray,
    *,
    top_fraction: float,
) -> dict[str, float]:
    weights = block_counts[ranking.block_ids].astype(float, copy=False)
    total = float(weights.sum())
    if total <= 0:
        return {
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "top10_precision": np.nan,
            "top10_lift": np.nan,
        }
    positives = weights * ranking.labels
    positive_total = float(positives.sum())
    negative_total = total - positive_total
    prevalence = positive_total / total
    cumulative_total = np.cumsum(weights)
    cumulative_positive = np.cumsum(positives)

    top_weight = max(1.0, float(np.ceil(total * top_fraction)))
    boundary = min(
        int(np.searchsorted(cumulative_total, top_weight, side="left")),
        len(weights) - 1,
    )
    prior_weight = float(cumulative_total[boundary - 1]) if boundary else 0.0
    prior_positive = float(cumulative_positive[boundary - 1]) if boundary else 0.0
    boundary_take = max(
        0.0,
        min(float(weights[boundary]), top_weight - prior_weight),
    )
    top_precision = (
        prior_positive + boundary_take * float(ranking.labels[boundary])
    ) / top_weight
    if ranking.constant_score:
        top_precision = prevalence

    group_positive = cumulative_positive[ranking.threshold_ends]
    group_total = cumulative_total[ranking.threshold_ends]
    if positive_total <= 0 or negative_total <= 0:
        pr_auc = np.nan
        roc_auc = np.nan
    else:
        precision = np.divide(
            group_positive,
            group_total,
            out=np.zeros_like(group_positive),
            where=group_total > 0,
        )
        positive_increment = np.diff(np.r_[0.0, group_positive])
        pr_auc = float(np.dot(positive_increment / positive_total, precision))
        true_positive_rate = np.r_[0.0, group_positive / positive_total, 1.0]
        false_positive = group_total - group_positive
        false_positive_rate = np.r_[0.0, false_positive / negative_total, 1.0]
        roc_auc = float(np.trapezoid(true_positive_rate, false_positive_rate))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "top10_precision": float(top_precision),
        "top10_lift": float(top_precision / max(prevalence, 1e-12)),
    }


def _daily_top_k_label_totals(
    dates: pd.Series,
    block_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    top_k: int,
    block_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranked = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "block_id": block_ids,
            "target": labels,
            "score": scores,
            "row_order": np.arange(len(labels)),
        }
    )
    selected = (
        ranked.sort_values(
            ["date", "score", "row_order"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("date", sort=True, group_keys=False)
        .head(top_k)
    )
    totals = selected.groupby("block_id")["target"].agg(["sum", "size"])
    return (
        totals["sum"].reindex(range(block_count), fill_value=0.0).to_numpy(float),
        totals["size"].reindex(range(block_count), fill_value=0.0).to_numpy(float),
    )


def _interval(values: Sequence[float], confidence_level: float) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, 0
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(low), float(high), int(len(finite))


def compare_rule_feature_versions(
    predictions: pd.DataFrame,
    *,
    candidate_column: str = "pred_unified_long_task_deep",
    baseline_column: str = "pred_unified_long_task_deep_rule105",
    label_column: str = "good_path5",
    top_fraction: float = 0.10,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
    daily_top_k: int = 10,
    confidence_level: float = 0.95,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Return paired 118-vs-105 ranking deltas by outer test fold."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    if daily_top_k <= 0:
        raise ValueError("daily_top_k must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    required = {
        *group_columns,
        "date",
        label_column,
        candidate_column,
        baseline_column,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction table missing ranking columns: {sorted(missing)}")

    work = predictions.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if "label" in group_columns:
        work = work[work["label"].astype(str).eq(label_column)].copy()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for group_key, group in work.groupby(list(group_columns), sort=True, dropna=False):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        scope = dict(zip(group_columns, keys))
        paired = group[["date", label_column, candidate_column, baseline_column]].copy()
        for column in (label_column, candidate_column, baseline_column):
            paired[column] = pd.to_numeric(paired[column], errors="coerce")
        paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
        if paired.empty:
            rows.append(
                {
                    **scope,
                    "status": "no_paired_rows",
                    "paired_rows": 0,
                    "coverage": 0.0,
                    "month_blocks": 0,
                }
            )
            continue
        labels = paired[label_column].to_numpy(dtype=int)
        candidate = paired[candidate_column].to_numpy(dtype=float)
        baseline = paired[baseline_column].to_numpy(dtype=float)
        block_ids, blocks = pd.factorize(paired["date"].dt.to_period("M"), sort=True)
        block_ids = block_ids.astype(np.int16, copy=False)
        block_count = len(blocks)
        candidate_ranking = _prepare_ranking(labels, candidate, block_ids)
        baseline_ranking = _prepare_ranking(labels, baseline, block_ids)
        observed_counts = np.ones(block_count, dtype=int)
        candidate_metrics = _weighted_metrics(
            candidate_ranking, observed_counts, top_fraction=top_fraction
        )
        baseline_metrics = _weighted_metrics(
            baseline_ranking, observed_counts, top_fraction=top_fraction
        )
        candidate_daily_sum, candidate_daily_count = _daily_top_k_label_totals(
            paired["date"], block_ids, labels, candidate,
            top_k=daily_top_k, block_count=block_count,
        )
        baseline_daily_sum, baseline_daily_count = _daily_top_k_label_totals(
            paired["date"], block_ids, labels, baseline,
            top_k=daily_top_k, block_count=block_count,
        )
        candidate_daily = float(candidate_daily_sum.sum() / candidate_daily_count.sum())
        baseline_daily = float(baseline_daily_sum.sum() / baseline_daily_count.sum())
        observed = {
            **{
                metric: (candidate_metrics[metric], baseline_metrics[metric])
                for metric in candidate_metrics
            },
            "daily_top_k_label_hit_rate": (candidate_daily, baseline_daily),
        }
        deltas: dict[str, list[float]] = {metric: [] for metric in observed}
        if bootstrap_iterations and block_count >= 2:
            for draw in rng.integers(0, block_count, size=(bootstrap_iterations, block_count)):
                counts = np.bincount(draw, minlength=block_count)
                candidate_draw = _weighted_metrics(
                    candidate_ranking, counts, top_fraction=top_fraction
                )
                baseline_draw = _weighted_metrics(
                    baseline_ranking, counts, top_fraction=top_fraction
                )
                for metric in candidate_metrics:
                    deltas[metric].append(
                        candidate_draw[metric] - baseline_draw[metric]
                    )
                candidate_count = float(np.dot(counts, candidate_daily_count))
                baseline_count = float(np.dot(counts, baseline_daily_count))
                candidate_hit = (
                    float(np.dot(counts, candidate_daily_sum) / candidate_count)
                    if candidate_count > 0 else np.nan
                )
                baseline_hit = (
                    float(np.dot(counts, baseline_daily_sum) / baseline_count)
                    if baseline_count > 0 else np.nan
                )
                deltas["daily_top_k_label_hit_rate"].append(
                    candidate_hit - baseline_hit
                )
        row: dict[str, object] = {
            **scope,
            "candidate_column": candidate_column,
            "baseline_column": baseline_column,
            "status": "ok" if deltas["pr_auc"] else "exact_only",
            "paired_rows": int(len(paired)),
            "group_rows": int(len(group)),
            "coverage": float(len(paired) / max(len(group), 1)),
            "month_blocks": int(block_count),
        }
        for metric, (candidate_value, baseline_value) in observed.items():
            low, high, valid = _interval(deltas[metric], confidence_level)
            row[f"candidate_{metric}"] = candidate_value
            row[f"baseline_{metric}"] = baseline_value
            row[f"delta_{metric}"] = candidate_value - baseline_value
            row[f"delta_{metric}_ci_low"] = low
            row[f"delta_{metric}_ci_high"] = high
            row[f"delta_{metric}_bootstrap_valid"] = valid
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["compare_rule_feature_versions"]
