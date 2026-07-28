"""Shared, strategy-exact B1 candidate gate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.factors import KDJ
from quant.features.variable_library import build_continuous_ohlc, calc_bbi


B1_GATE_METRIC_COLUMNS = [
    "b1_pct_change",
    "b1_amplitude",
    "b1_bbi",
    "b1_ma60",
    "b1_kdj_j",
]


def _aligned_shared_column(
    shared_factors: pd.DataFrame,
    column: str,
    index: pd.Index,
) -> pd.Series | None:
    if column not in shared_factors.columns:
        return None
    values = pd.to_numeric(
        shared_factors[column],
        errors="coerce",
    ).to_numpy()
    if len(values) != len(index):
        raise ValueError(
            f"shared factor length mismatch for {column}: "
            f"{len(values)} != {len(index)}"
        )
    return pd.Series(values, index=index, dtype=float)


def calculate_b1_gate(
    daily: pd.DataFrame,
    *,
    shared_factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the production B1 mask and its exact gate inputs."""

    price = build_continuous_ohlc(daily)
    close = pd.to_numeric(price["close"], errors="coerce")
    low = pd.to_numeric(price["low"], errors="coerce")
    high = pd.to_numeric(price["high"], errors="coerce")
    pct_change = close.pct_change() * 100
    amplitude = (high - low) / low.replace(0, np.nan) * 100

    shared = shared_factors if shared_factors is not None else pd.DataFrame()
    bbi = _aligned_shared_column(shared, "bbi", daily.index)
    if bbi is None:
        bbi = calc_bbi(close)
    kdj_j = _aligned_shared_column(shared, "kdj_d_j", daily.index)
    if kdj_j is None:
        kdj_j = KDJ().compute(price)["J"]
    ma60 = close.rolling(60, min_periods=20).mean()
    gate = (
        pct_change.between(-2, 2, inclusive="both")
        & (amplitude < 7)
        & (bbi > ma60)
        & (kdj_j < 0)
    )
    return pd.DataFrame(
        {
            "b1_gate": gate.fillna(False).astype(bool),
            "b1_pct_change": pct_change,
            "b1_amplitude": amplitude,
            "b1_bbi": bbi,
            "b1_ma60": ma60,
            "b1_kdj_j": kdj_j,
        },
        index=daily.index,
    )
