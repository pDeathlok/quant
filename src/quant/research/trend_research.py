"""Causal weekly direction, daily entry and recency-weighted model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from quant.research.kronos_multiscale import SequenceSubscores, period_bars

WEEKLY_GATES = ("macd_bull", "macd_accelerating", "boll_up", "boll_breakout", "ema_up", "macd_boll")
DAILY_FEATURES = (
    "return1",
    "return5",
    "return20",
    "return60",
    "volume20",
    "close_position",
    "range",
    "ma20_distance",
    "ma60_distance",
    "volatility20",
)
WEEKLY_FEATURES = (
    "w_macd",
    "w_hist",
    "w_hist_change",
    "w_boll_position",
    "w_boll_width",
    "w_mid_slope4",
    "w_ema10_distance",
    "w_ema26_distance",
    "w_return4",
    "w_return13",
)


def recent_window(frame: pd.DataFrame, refit: pd.Timestamp, months: int) -> pd.DataFrame:
    return frame[
        frame.date.ge(refit - pd.DateOffset(months=months))
        & frame.date.lt(refit)
        & frame.exit_date.lt(refit)
    ].copy()


def recency_weights(
    dates: pd.Series, refit: pd.Timestamp, half_life: float | None, *, equal_dates: bool = False
) -> np.ndarray:
    ages = (pd.Timestamp(refit) - pd.to_datetime(dates)).dt.total_seconds().to_numpy() / 86400
    if (ages <= 0).any() or (half_life is not None and half_life <= 0):
        raise ValueError("weights require past dates and a positive half-life")
    weights = np.ones(len(dates)) if half_life is None else np.exp2(-ages / half_life)
    if equal_dates:
        weights /= dates.map(dates.value_counts()).to_numpy()
    return (weights / weights.mean()).astype(np.float32)


def fit_weighted_subscores(
    embeddings: np.ndarray, returns: np.ndarray, weights: np.ndarray, *, epochs: int = 20
) -> SequenceSubscores:
    if len(returns) < 30 or len(returns) != len(embeddings) or len(weights) != len(returns):
        raise ValueError("insufficient or inconsistent weighted training data")
    torch.manual_seed(20261021)
    model = SequenceSubscores(embeddings.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
    y = torch.from_numpy(np.asarray(returns, dtype=np.float32))
    weight = torch.from_numpy(np.asarray(weights, dtype=np.float32))
    generator = np.random.default_rng(20261021)
    for _ in range(epochs):
        model.train()
        for batch in np.array_split(
            generator.permutation(len(x)), max(1, int(np.ceil(len(x) / 256)))
        ):
            output = model(x[batch])
            losses = (
                torch.nn.functional.smooth_l1_loss(output[:, 0], y[batch] * 100, reduction="none")
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    output[:, 1], y[batch].gt(0).float(), reduction="none"
                )
                + torch.nn.functional.smooth_l1_loss(
                    output[:, 2], (-y[batch]).clamp(min=0) * 100, reduction="none"
                )
            )
            loss = (losses * weight[batch]).sum() / weight[batch].sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def direction_features(raw: pd.DataFrame) -> pd.DataFrame:
    """One stock; return only information available at each session close."""
    frame = raw.sort_values("date").copy()
    if frame.date.duplicated().any():
        raise ValueError("duplicate historical daily bars")
    frame = frame[frame[["open", "high", "low", "close", "pre_close"]].gt(0).all(axis=1)].copy()
    index = np.exp(np.log(frame.close / frame.pre_close).cumsum()) * 100
    multiplier = index / frame.close
    daily = frame[["date", "open", "high", "low", "close"]].copy()
    daily[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].mul(
        multiplier, axis=0
    )
    daily["volume"] = frame.vol.to_numpy()
    daily = daily.reset_index(drop=True)
    close = daily.close
    for horizon in (1, 5, 20, 60):
        daily[f"return{horizon}"] = close.pct_change(horizon, fill_method=None)
    daily["volume20"] = daily.volume / daily.volume.rolling(20).mean().replace(0, np.nan)
    daily["close_position"] = (
        (close - daily.low) / (daily.high - daily.low).replace(0, np.nan)
    ).fillna(0.5)
    daily["range"] = (daily.high - daily.low) / close
    daily["ma20_distance"] = close / close.rolling(20).mean() - 1
    daily["ma60_distance"] = close / close.rolling(60).mean() - 1
    daily["volatility20"] = daily.return1.rolling(20).std()
    daily["left_pullback"] = daily.return5.lt(0) & daily.ma60_distance.gt(0) & daily.volume20.le(1)
    daily["left_stabilize"] = (
        daily.ma20_distance.between(-0.04, 0.02)
        & daily.volume20.le(1.2)
        & daily.close_position.ge(0.5)
    )
    daily["right_breakout"] = close.gt(daily.high.shift(1).rolling(20).max()) & daily.volume20.ge(
        1.2
    )
    daily["right_confirm"] = (
        daily.return1.gt(0) & daily.close_position.ge(0.6) & daily.volume20.ge(1)
    )
    weekly = period_bars(daily[["date", "open", "high", "low", "close", "volume"]], "weekly")
    wc = weekly.close
    fast = wc.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = wc.ewm(span=26, adjust=False, min_periods=26).mean()
    dif = fast - slow
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = dif - dea
    mid = wc.rolling(20).mean()
    std = wc.rolling(20).std(ddof=0)
    upper = mid + 2 * std
    ema10 = wc.ewm(span=10, adjust=False, min_periods=10).mean()
    ema26 = wc.ewm(span=26, adjust=False, min_periods=26).mean()
    weekly["w_macd"] = dif / wc
    weekly["w_hist"] = hist / wc
    weekly["w_hist_change"] = hist.diff() / wc
    weekly["w_boll_position"] = (wc - mid) / (2 * std).replace(0, np.nan)
    weekly["w_boll_width"] = 4 * std / mid
    weekly["w_mid_slope4"] = mid / mid.shift(4) - 1
    weekly["w_ema10_distance"] = wc / ema10 - 1
    weekly["w_ema26_distance"] = wc / ema26 - 1
    weekly["w_return4"] = wc.pct_change(4, fill_method=None)
    weekly["w_return13"] = wc.pct_change(13, fill_method=None)
    weekly["macd_bull"] = dif.gt(dea) & dif.gt(0)
    weekly["macd_accelerating"] = weekly.macd_bull & hist.gt(hist.shift(1))
    weekly["boll_up"] = wc.gt(mid) & mid.gt(mid.shift(4))
    weekly["boll_breakout"] = wc.gt(upper)
    weekly["ema_up"] = wc.gt(ema10) & ema10.gt(ema26) & ema10.gt(ema10.shift(4))
    weekly["macd_boll"] = weekly.macd_bull & weekly.boll_up
    weekly = weekly.rename(columns={"date": "weekly_source_date"})
    keep = ["available_at", "weekly_source_date", *WEEKLY_FEATURES, *WEEKLY_GATES]
    output = pd.merge_asof(
        daily.sort_values("date"),
        weekly[keep].sort_values("available_at"),
        left_on="date",
        right_on="available_at",
        direction="backward",
    )
    for gate in WEEKLY_GATES:
        output[gate] = output[gate].fillna(False).astype(bool)
    return output


def forward_event_labels(
    features: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
    cost: float = 0.002,
) -> pd.DataFrame:
    """Exact exchange sessions; reject missing/suspended path bars instead of shifting labels."""
    positions = calendar.get_indexer(events.date)
    valid = (positions >= 0) & (positions + horizon + 1 < len(calendar))
    source = events.loc[valid].copy()
    positions = positions[valid]
    entry_dates = calendar[positions + 1]
    exit_dates = calendar[positions + horizon + 1]
    full = features.set_index("date").reindex(calendar)
    closes = full.close.to_numpy(float)
    active = full.close.notna().to_numpy() & full.volume.fillna(0).gt(0).to_numpy()
    invalid_cum = np.r_[0, np.cumsum(~active)]
    clean = (invalid_cum[positions + horizon + 2] - invalid_cum[positions + 1]) == 0
    source = source.loc[clean].copy()
    positions = positions[clean]
    source["entry_date"] = entry_dates[clean]
    source["exit_date"] = exit_dates[clean]
    source["net_return"] = closes[positions + horizon + 1] / closes[positions + 1] - 1 - cost
    source["horizon"] = horizon
    return source


def weekly_market_breadth(prices: pd.DataFrame) -> pd.DataFrame:
    """Last completed week's market breadth; future listings/bars cannot revise past values."""
    frame = prices[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    frame["ma60"] = frame.groupby("symbol", sort=False).close.transform(
        lambda x: x.rolling(60).mean()
    )
    frame["return20"] = frame.groupby("symbol", sort=False).close.pct_change(20, fill_method=None)
    frame["eligible"] = frame.ma60.notna()
    frame["above"] = frame.close.gt(frame.ma60) & frame.eligible
    daily = (
        frame.groupby("date")
        .agg(
            eligible=("eligible", "sum"),
            above=("above", "sum"),
            median_return20=("return20", "median"),
        )
        .reset_index()
    )
    daily["breadth"] = daily.above / daily.eligible.replace(0, np.nan)
    daily["period"] = daily.date.dt.to_period("W-SUN")
    weekly = daily.groupby("period", sort=True).last()
    weekly["available_at"] = (weekly.index + 1).start_time
    weekly = weekly.rename(
        columns={
            "date": "market_source_date",
            "breadth": "market_breadth",
            "median_return20": "market_median_return20",
        }
    )
    return pd.merge_asof(
        daily[["date"]],
        weekly[
            ["available_at", "market_source_date", "market_breadth", "market_median_return20"]
        ].sort_values("available_at"),
        left_on="date",
        right_on="available_at",
        direction="backward",
    )
