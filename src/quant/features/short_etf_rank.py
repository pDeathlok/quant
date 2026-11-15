"""Market-conditioned residual layer for the existing five-day short rankers."""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "short-etf-ranker-v1"
MACRO_SCALES = {
    "etf_all_flow_pct_1": .015,
    "etf_all_flow_pct_5": .05,
    "etf_csi300_flow_pct_5": .05,
    "etf_csi500_flow_pct_5": .05,
    "etf_csi1000_flow_pct_5": .05,
    "etf_all_flow_accel": .015,
    "etf_all_turnover_multiple_20": .7,
    "etf_all_positive_fraction_5": .5,
}
REQUIRED = ("etf_all_flow_pct_5", "etf_all_turnover_multiple_20")
ARTIFACT = "configs/strategies/short_etf_rank.json"
MAX_SHIFT = .04


def market_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame.reindex(columns=MACRO_SCALES).to_numpy(dtype=float, na_value=np.nan, copy=True)
    turnover = list(MACRO_SCALES).index("etf_all_turnover_multiple_20")
    values[:, turnover] = np.log(np.where(values[:, turnover] > 0, values[:, turnover], np.nan))
    ready = np.isfinite(values[:, [list(MACRO_SCALES).index(key) for key in REQUIRED]]).all(axis=1)
    values[:, -1] -= .5
    return np.nan_to_num(np.clip(values / list(MACRO_SCALES.values()), -4, 4), nan=0), ready


def stock_matrix(frame: pd.DataFrame, probabilities: np.ndarray, signals: tuple[str, ...]) -> np.ndarray:
    flags = frame[list(signals)].fillna(False).astype(bool).to_numpy(dtype=float)
    score = (probabilities - .25) * 4
    return np.column_stack((np.ones(len(frame)), score, flags, flags * score[:, None]))


def apply_residual(
    stock: np.ndarray, market: np.ndarray, probability: np.ndarray,
    coefficients: np.ndarray, strength: float,
) -> np.ndarray:
    residual = np.einsum("ij,jk,ik->i", stock, coefficients, market, optimize=True)
    return np.clip(probability + np.clip(strength * residual, -MAX_SHIFT, MAX_SHIFT), 0, 1)
