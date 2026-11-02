"""Explicit price-anchor normalization for the daily overnight experiment."""

import numpy as np
import pandas as pd

PRICE_COLUMNS = (
    "open", "high", "low", "pre_close", "change", "ma_5", "ma_10", "ma_20", "ma_60", "ma_120",
    "ema_5", "ema_10", "ema_20", "bb_upper", "bb_lower", "atr_14", "keltner_upper", "keltner_lower",
    "bbi", "bbi_ma60_diff", "macd_dif", "macd_dea", "macd_hist", "parabolic_sar",
    "weekly_ma55", "weekly_ma144", "weekly_ma233", "weekly_ma55_slope", "weekly_ma144_slope",
)
DROPPED = ("close", "price_log", "obv")


def relative_prices(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Change named price features only; no fitting, labels, or future rows."""
    source = [f"today_factor_{c}" for c in (*PRICE_COLUMNS, *DROPPED)]
    if set(source) - set(columns):
        raise ValueError("Required price factors missing from input contract")
    result = frame.copy()
    anchor = result.today_factor_close.where(result.today_factor_close.gt(0))
    new_columns = []
    for name in columns:
        factor = name.removeprefix("today_factor_")
        if name in source:
            if factor in DROPPED:
                continue
            new = f"today_relative_{factor}"
            result[new] = (result[name] / anchor).replace([np.inf, -np.inf], np.nan)
            new_columns.append(new)
        else:
            new_columns.append(name)
    return result, new_columns
