"""Vegas tunnel stock-picking signal.

The strategy uses the classic Vegas tunnel idea on daily bars:
EMA144/EMA169 define the trend tunnel, EMA12/EMA24 define short-term momentum.
Signals look for a right-side rebound after price has recently touched or
approached the tunnel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc


OPTIMIZED_VEGAS_TUNNEL_PARAMS = {
    "fast_span": 10,
    "momentum_span": 20,
    "tunnel_short_span": 144,
    "tunnel_long_span": 169,
    "near_tunnel_pct": 0.025,
    "pullback_window": 8,
    "max_tunnel_distance": 0.18,
}


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        raise ValueError("daily frame must contain date or trade_date")

    if "volume" not in out.columns and "vol" in out.columns:
        out = out.rename(columns={"vol": "volume"})
    if "symbol" not in out.columns and "ts_code" in out.columns:
        out["symbol"] = out["ts_code"].astype(str)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"daily frame missing required columns: {sorted(missing)}")

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _is_risk_name(name: pd.Series) -> pd.Series:
    text = name.fillna("").astype(str).str.upper()
    return text.str.contains("ST", regex=False) | text.str.contains("*", regex=False) | text.str.contains("退", regex=False)


def add_vegas_tunnel_signals(
    df: pd.DataFrame,
    *,
    fast_span: int = 12,
    momentum_span: int = 24,
    tunnel_short_span: int = 144,
    tunnel_long_span: int = 169,
    near_tunnel_pct: float = 0.035,
    pullback_window: int = 12,
    max_tunnel_distance: float = 0.18,
    volume_min_ratio: float = 1.05,
    volume_max_ratio: float = 3.0,
    min_history: int = 180,
) -> pd.DataFrame:
    """Add Vegas tunnel signal and ranking columns.

    Output highlights:
    - ``signal_vegas_tunnel``: final stock-picking signal.
    - ``vegas_tunnel_upper/lower``: EMA144/EMA169 band.
    - ``vegas_candidate_score``: ranking score for crowded signal days.
    """
    out = _normalize_daily_frame(df)
    price = build_continuous_ohlc(out)
    open_ = price["open"].astype(float)
    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)
    volume = out["volume"].astype(float)

    if fast_span <= 1 or momentum_span <= fast_span:
        raise ValueError("fast_span must be > 1 and momentum_span must be greater than fast_span")
    if tunnel_short_span <= momentum_span or tunnel_long_span <= tunnel_short_span:
        raise ValueError("tunnel spans must be greater than momentum_span and ordered short < long")

    ema12 = close.ewm(span=fast_span, adjust=False, min_periods=fast_span).mean()
    ema24 = close.ewm(span=momentum_span, adjust=False, min_periods=momentum_span).mean()
    ema144 = close.ewm(span=tunnel_short_span, adjust=False, min_periods=tunnel_short_span).mean()
    ema169 = close.ewm(span=tunnel_long_span, adjust=False, min_periods=tunnel_long_span).mean()
    tunnel_upper = pd.concat([ema144, ema169], axis=1).max(axis=1)
    tunnel_lower = pd.concat([ema144, ema169], axis=1).min(axis=1)
    tunnel_mid = (ema144 + ema169) / 2

    volume_ma20 = volume.rolling(20).mean()
    atr_proxy = ((high - low) / close.replace(0, np.nan)).rolling(20).mean()
    price_near_tunnel = low <= tunnel_upper * (1 + near_tunnel_pct)
    recent_pullback = price_near_tunnel.rolling(pullback_window).max().fillna(0).astype(bool)
    right_side_rebound = (close > ema12) & (close > open_) & (close > close.shift(1))
    trend_stack = (close > tunnel_upper) & (ema12 > ema24) & (ema24 > tunnel_upper)
    tunnel_up = (tunnel_mid > tunnel_mid.shift(5)) & (ema144 > ema144.shift(20))
    not_overheated = close / tunnel_upper.replace(0, np.nan) <= 1 + max_tunnel_distance
    volume_confirm = (volume > volume_ma20 * volume_min_ratio) & (volume < volume_ma20 * volume_max_ratio)
    history_ok = pd.Series(np.arange(len(out)), index=out.index) >= min_history

    if "name" in out.columns:
        risk_name = _is_risk_name(out["name"])
    else:
        risk_name = pd.Series(False, index=out.index)
    tradable = open_.notna() & (open_ > 0) & ~risk_name

    signal = (
        history_ok
        & trend_stack
        & tunnel_up
        & recent_pullback
        & right_side_rebound
        & volume_confirm
        & not_overheated
        & tradable
    )

    tunnel_distance = close / tunnel_upper.replace(0, np.nan) - 1
    tunnel_slope_20d = tunnel_mid / tunnel_mid.shift(20).replace(0, np.nan) - 1
    fast_spread = ema12 / ema24.replace(0, np.nan) - 1
    volume_strength = volume / volume_ma20.replace(0, np.nan) - 1
    volatility_penalty = atr_proxy.rank(pct=True)
    distance_score = (0.12 - (tunnel_distance - 0.06).abs()).rank(pct=True)

    out["ema12"] = ema12
    out["ema24"] = ema24
    out["ema144"] = ema144
    out["ema169"] = ema169
    out["vegas_tunnel_upper"] = tunnel_upper
    out["vegas_tunnel_lower"] = tunnel_lower
    out["vegas_tunnel_mid"] = tunnel_mid
    out["vegas_tunnel_slope_20d"] = tunnel_slope_20d
    out["vegas_tunnel_distance"] = tunnel_distance
    out["vegas_fast_spread"] = fast_spread
    out["vegas_volume_strength"] = volume_strength
    out["vegas_recent_pullback"] = recent_pullback.astype(int)
    out["vegas_trend_stack"] = trend_stack.astype(int)
    out["signal_vegas_tunnel"] = signal.fillna(False).astype(int)
    out["vegas_candidate_score"] = (
        0.30 * tunnel_slope_20d.rank(pct=True)
        + 0.25 * fast_spread.rank(pct=True)
        + 0.20 * distance_score
        + 0.15 * volume_strength.rank(pct=True)
        + 0.10 * (1 - volatility_penalty)
    )
    return out


def summarize_vegas_tunnel(df: pd.DataFrame) -> dict[str, float | int]:
    out = add_vegas_tunnel_signals(df)
    signal = out["signal_vegas_tunnel"] == 1
    return {
        "rows": int(len(out)),
        "signals": int(signal.sum()),
        "first_signal": out.loc[signal, "date"].min() if signal.any() else pd.NaT,
        "last_signal": out.loc[signal, "date"].max() if signal.any() else pd.NaT,
    }
