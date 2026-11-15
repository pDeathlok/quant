"""Convex joint classification/return objectives for research ranking.

The single raw score is shared by all losses; sigmoid(score) is not a calibrated
win probability when auxiliary coefficients are nonzero.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class LossWeights:
    absolute: float = 0.0
    incremental: float = 0.0
    downside: float = 0.0

    def __post_init__(self) -> None:
        values = (self.absolute, self.incremental, self.downside)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError("Auxiliary weights must be finite and nonnegative")


def terms(net: np.ndarray, cohort_mean: np.ndarray, scale: float) -> tuple[np.ndarray, ...]:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Return scale must be positive and fit-only')
    if not np.isfinite(net).all() or not np.isfinite(cohort_mean).all():
        raise ValueError('Training outcomes must be known')
    return ((net > 0).astype(float), np.clip(net / scale, -3, 3),
            np.clip((net - cohort_mean) / scale, -3, 3),
            np.clip(-net / scale, 0, 3))


def joint_loss(score: np.ndarray, targets: tuple[np.ndarray, ...], weights: LossWeights) -> np.ndarray:
    win, absolute, increment, downside = targets
    bce = np.logaddexp(0, score) - win * score
    return ((1 + weights.downside * downside) * bce
            + .5 * weights.absolute * (score - absolute) ** 2
            + .5 * weights.incremental * (score - increment) ** 2)


def derivatives(score: np.ndarray, targets: tuple[np.ndarray, ...], weights: LossWeights,
                sample_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    win, absolute, increment, downside = targets
    p = expit(score)
    cost = 1 + weights.downside * downside
    gradient = cost * (p - win) + weights.absolute * (score - absolute) + weights.incremental * (score - increment)
    hessian = cost * p * (1 - p) + weights.absolute + weights.incremental
    return gradient * sample_weight, np.maximum(hessian, 1e-8) * sample_weight


def daily_picks(frame: pd.DataFrame, score: np.ndarray, k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank before inspecting outcomes; unknown selected exits remain unknown."""
    chosen = frame.assign(score=score).sort_values(
        ['date', 'score', 'symbol'], ascending=[True, False, True], kind='stable'
    ).groupby('date', sort=False).head(k).copy()
    chosen['win'] = chosen.outcome.gt(0).astype(float).where(chosen.outcome.notna())
    chosen['stress'] = chosen.outcome.fillna(-.10)
    chosen['stress_win'] = chosen.win.fillna(0)
    daily = chosen.groupby('date').agg(net=('outcome', 'mean'), win=('win', 'mean'),
                                      known=('outcome', 'count'), count=('outcome', 'size'),
                                      stress=('stress', 'mean'), stress_win=('stress_win', 'mean'))
    daily.loc[daily.known.ne(daily['count']), ['net', 'win']] = np.nan
    return chosen, daily
