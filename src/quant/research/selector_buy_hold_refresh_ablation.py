"""Walk-forward refresh cadence and history-window ablation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class RefreshPeriod:
    cutoff: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class AblationMetrics:
    rows: int
    days: int
    daily_spearman: float
    positive_daily_spearman_ratio: float
    decile_spread: float
    decile_trend: float
    top20_average_target: float


def refresh_periods(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency_months: int | None,
) -> tuple[RefreshPeriod, ...]:
    """Create non-overlapping OOT periods anchored at ``start``."""

    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError("refresh ablation end precedes start")
    if frequency_months is None:
        return (RefreshPeriod(start, end),)
    if frequency_months <= 0:
        raise ValueError("refresh frequency must be positive or None")
    periods: list[RefreshPeriod] = []
    cutoff = start
    while cutoff <= end:
        next_cutoff = cutoff + pd.DateOffset(months=frequency_months)
        periods.append(RefreshPeriod(cutoff, min(end, next_cutoff - pd.Timedelta(days=1))))
        cutoff = next_cutoff
    return tuple(periods)


def history_start(
    cutoff: pd.Timestamp,
    window_months: int | None,
    dataset_start: pd.Timestamp,
) -> pd.Timestamp:
    """Return the inclusive start of a rolling or expanding training window."""

    if window_months is None:
        return pd.Timestamp(dataset_start).normalize()
    if window_months <= 0:
        raise ValueError("history window must be positive or None")
    return max(
        pd.Timestamp(dataset_start).normalize(),
        pd.Timestamp(cutoff).normalize() - pd.DateOffset(months=window_months),
    )


def historical_scores(
    predictions: np.ndarray,
    reference: np.ndarray,
    width: float,
) -> np.ndarray:
    """Apply the production robust monotonic score transform."""

    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("historical score reference is empty")
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), 1e-6)
    z_score = (np.asarray(predictions, dtype=float) - median) / scale
    return np.clip(
        50.0 + 100.0 / np.pi * np.arctan(z_score / max(width, 0.1)),
        0.0,
        100.0,
    )


def training_target(values: pd.Series, dates: pd.Series, mode: str) -> np.ndarray:
    """Match the released buy/hold training targets."""

    numeric = pd.to_numeric(values, errors="coerce")
    if mode == "hold":
        return (
            numeric.groupby(dates).rank(method="average", pct=True).sub(0.5)
        ).to_numpy(dtype=np.float32)
    if mode != "buy":
        raise ValueError(f"unsupported selector score mode: {mode}")
    return numeric.clip(-5.0, 30.0).to_numpy(dtype=np.float32)


def date_balanced_weights(dates: pd.Series) -> np.ndarray:
    """Give every trading day equal total training weight."""

    counts = dates.groupby(dates).transform("size").astype(float)
    weights = 1.0 / counts
    return (weights / weights.mean()).to_numpy(dtype=np.float32)


def evaluate_predictions(
    frame: pd.DataFrame,
    *,
    target_column: str,
    prediction_column: str,
) -> AblationMetrics:
    """Evaluate cross-sectional ranking over stitched OOT predictions."""

    values = frame[["date", target_column, prediction_column]].copy()
    values.columns = ["date", "target", "prediction"]
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("refresh ablation evaluation frame is empty")
    values["target_rank"] = values.groupby("date")["target"].rank(
        method="average", pct=True
    )
    values["prediction_rank"] = values.groupby("date")["prediction"].rank(
        method="average", pct=True
    )
    daily = (
        values.groupby("date", sort=True)
        .apply(
            lambda part: part["target_rank"].corr(part["prediction_rank"]),
            include_groups=False,
        )
        .dropna()
    )
    values["decile"] = (
        np.ceil(values["prediction_rank"] * 10).clip(1, 10).astype(int)
    )
    deciles = values.groupby("decile", as_index=False)["target"].mean()
    low = float(deciles.loc[deciles["decile"].eq(1), "target"].iloc[0])
    high = float(deciles.loc[deciles["decile"].eq(10), "target"].iloc[0])
    trend = spearmanr(deciles["decile"], deciles["target"]).statistic
    top20 = (
        values.sort_values(["date", "prediction"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(20)
    )
    return AblationMetrics(
        rows=int(len(values)),
        days=int(values["date"].nunique()),
        daily_spearman=float(daily.mean()) if len(daily) else 0.0,
        positive_daily_spearman_ratio=float((daily > 0).mean()) if len(daily) else 0.0,
        decile_spread=high - low,
        decile_trend=float(trend) if np.isfinite(trend) else 0.0,
        top20_average_target=float(top20["target"].mean()),
    )


def metric_slices(
    predictions: pd.DataFrame,
    *,
    development_end: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    """Return combined, development, verification, and yearly metrics."""

    development_end = pd.Timestamp(development_end).normalize()
    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=predictions.index),
        "development": predictions["date"].le(development_end),
        "verification": predictions["date"].gt(development_end),
    }
    for year in sorted(predictions["date"].dt.year.unique()):
        masks[str(int(year))] = predictions["date"].dt.year.eq(year)
    output: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        part = predictions.loc[mask]
        if part.empty:
            continue
        output[name] = {
            "buy": asdict(
                evaluate_predictions(
                    part,
                    target_column="future_max_high_t5_pct",
                    prediction_column="buy_prediction",
                )
            ),
            "hold": asdict(
                evaluate_predictions(
                    part,
                    target_column="future_return_t5_pct",
                    prediction_column="hold_prediction",
                )
            ),
        }
        output[name]["mean_daily_spearman"] = float(
            np.mean(
                [
                    output[name]["buy"]["daily_spearman"],
                    output[name]["hold"]["daily_spearman"],
                ]
            )
        )
    return output


def run_walk_forward(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    development_end: pd.Timestamp,
    frequency_months: int | None,
    window_months: int | None,
    hold_buy_weight: float,
    model_factory: Callable[[], Any],
    minimum_training_rows: int,
) -> dict[str, Any]:
    """Fit at each cutoff and stitch strictly forward predictions."""

    feature_columns = list(features)
    dataset_start = pd.Timestamp(data["date"].min()).normalize()
    periods = refresh_periods(evaluation_start, evaluation_end, frequency_months)
    predictions: list[pd.DataFrame] = []
    fit_rows: list[int] = []
    period_rows: list[dict[str, Any]] = []
    for period in periods:
        start = history_start(period.cutoff, window_months, dataset_start)
        train_mask = (
            data["date"].ge(start)
            & data["date"].lt(period.cutoff)
            & data["label_end_date"].lt(period.cutoff)
            & data["training_sample"]
        )
        test_mask = data["date"].between(period.cutoff, period.end)
        train = data.loc[train_mask]
        test = data.loc[test_mask]
        if test.empty:
            continue
        if len(train) < minimum_training_rows:
            raise RuntimeError(
                f"refresh cutoff {period.cutoff.date()} has only {len(train)} training rows"
            )
        weights = date_balanced_weights(train["date"])
        fitted: dict[str, Any] = {}
        references: dict[str, np.ndarray] = {}
        for mode, target in (
            ("buy", "future_max_high_t5_pct"),
            ("hold", "future_return_t5_pct"),
        ):
            model = model_factory()
            model.fit(
                train[feature_columns],
                training_target(train[target], train["date"], mode),
                sample_weight=weights,
                verbose=False,
            )
            fitted[mode] = model
            references[mode] = np.sort(
                model.predict(train[feature_columns]).astype(float)
            )
        buy_score = historical_scores(
            fitted["buy"].predict(test[feature_columns]),
            references["buy"],
            6.0,
        )
        hold_score = historical_scores(
            fitted["hold"].predict(test[feature_columns]),
            references["hold"],
            2.0,
        )
        predictions.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "future_max_high_t5_pct": test[
                        "future_max_high_t5_pct"
                    ].to_numpy(),
                    "future_return_t5_pct": test["future_return_t5_pct"].to_numpy(),
                    "buy_prediction": buy_score,
                    "hold_prediction": (
                        hold_buy_weight * buy_score
                        + (1.0 - hold_buy_weight) * hold_score
                    ),
                }
            )
        )
        fit_rows.append(int(len(train)))
        period_rows.append(
            {
                "cutoff": period.cutoff.date().isoformat(),
                "end": period.end.date().isoformat(),
                "training_start": start.date().isoformat(),
                "training_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        )
    if not predictions:
        raise RuntimeError("refresh ablation produced no OOT predictions")
    stitched = pd.concat(predictions, ignore_index=True)
    return {
        "frequency_months": frequency_months,
        "window_months": window_months,
        "fit_count": len(period_rows) * 2,
        "average_training_rows": float(np.mean(fit_rows)),
        "periods": period_rows,
        "metrics": metric_slices(stitched, development_end=development_end),
    }


def choose_maintenance_aware_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    complexity_key: Callable[[Mapping[str, Any]], float],
    tolerance: float = 0.005,
) -> Mapping[str, Any]:
    """Choose the simplest candidate within tolerance of best development IC."""

    if not rows:
        raise ValueError("cannot choose from empty ablation rows")
    best = max(
        float(row["metrics"]["development"]["mean_daily_spearman"])
        for row in rows
    )
    eligible = [
        row
        for row in rows
        if float(row["metrics"]["development"]["mean_daily_spearman"])
        >= best - tolerance
    ]
    return min(eligible, key=complexity_key)


__all__ = [
    "AblationMetrics",
    "RefreshPeriod",
    "choose_maintenance_aware_candidate",
    "evaluate_predictions",
    "history_start",
    "refresh_periods",
    "run_walk_forward",
]
