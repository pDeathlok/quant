from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.strategies.convertible_bond.rotation import (
    ConvertibleBondFilterConfig,
    ConvertibleBondSelector,
)


@dataclass(frozen=True)
class ConvertibleBondTrendEnhancedConfig:
    top_n: int = 20
    max_position_weight: float = 0.05
    min_score: float = 0.0
    rebalance_threshold: float = 0.01
    min_trend_strength: float = 75.0
    min_5d_return: float = 3.0
    max_5d_return: float = 20.0
    min_1d_return: float = 0.0
    max_1d_return: float = 7.0
    min_six_sword: int = 4
    min_consecutive_six_sword: int = 1
    require_band_buy: bool = True
    max_price_position_60d: float | None = None
    max_market_median_double_low: float | None = None
    min_market_trend_20d: float | None = None
    min_market_trend_breadth: float | None = None
    filter: ConvertibleBondFilterConfig = ConvertibleBondFilterConfig(
        min_price=100.0,
        max_price=160.0,
        max_premium_rate=45.0,
        min_amount=1_000.0,
        min_remaining_size=1.0,
        min_credit_rating="AA-",
        exclude_call_risk=True,
        exclude_not_convertible=True,
    )


class ConvertibleBondTrendEnhancedSelector(ConvertibleBondSelector):
    """可转债趋势增强选债器。

    特征来自日线可复现指标，映射原文中的均线、趋势强度、短期动量、六脉神剑和波段状态。
    """

    def __init__(self, config: ConvertibleBondTrendEnhancedConfig | None = None):
        self.config = config or ConvertibleBondTrendEnhancedConfig()

    def select(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        call: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        universe = self.build_universe(daily=daily, basic=basic, call=call)
        if universe.empty:
            return universe
        filtered = self._apply_filters(universe)
        if filtered.empty:
            return filtered
        signal_mask = (
            (filtered["trend_strength"] >= self.config.min_trend_strength)
            & filtered["return_5d"].between(
                self.config.min_5d_return,
                self.config.max_5d_return,
                inclusive="both",
            )
            & filtered["return_1d"].between(
                self.config.min_1d_return,
                self.config.max_1d_return,
                inclusive="both",
            )
            & (filtered["six_sword_daily"] >= self.config.min_six_sword)
            & (
                filtered["consecutive_six_sword"]
                >= self.config.min_consecutive_six_sword
            )
        )
        if self.config.require_band_buy:
            signal_mask &= filtered["band_state"].eq("买")
        if self.config.max_price_position_60d is not None:
            signal_mask &= (
                filtered["price_position_60d"].fillna(0.5)
                <= self.config.max_price_position_60d
            )
        if self.config.max_market_median_double_low is not None:
            signal_mask &= (
                filtered["market_median_double_low"].fillna(np.inf)
                <= self.config.max_market_median_double_low
            )
        if self.config.min_market_trend_20d is not None:
            signal_mask &= (
                filtered["market_trend_20d"].fillna(0.0)
                >= self.config.min_market_trend_20d
            )
        if self.config.min_market_trend_breadth is not None:
            signal_mask &= (
                filtered["market_trend_breadth"].fillna(0.0)
                >= self.config.min_market_trend_breadth
            )
        candidates = filtered[signal_mask].copy()
        if candidates.empty:
            return candidates
        scored = self._score(candidates)
        scored = scored[scored["score"] >= self.config.min_score].copy()
        scored = scored.sort_values(
            [
                "score",
                "trend_strength",
                "consecutive_six_sword",
                "return_5d",
                "amount",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        scored["rank"] = np.arange(1, len(scored) + 1)
        return scored.head(self.config.top_n)

    def _score(self, frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        scored["trend_score"] = scored["trend_strength"].clip(0, 100) / 100.0
        scored["momentum_5d_score"] = self._rank_score(scored["return_5d"], ascending=False)
        scored["six_sword_score"] = scored["six_sword_daily"].clip(0, 6) / 6.0
        scored["continuity_score"] = self._rank_score(
            scored["consecutive_six_sword"], ascending=False
        )
        scored["liquidity_score"] = self._rank_score(scored["amount"], ascending=False)
        scored["premium_score"] = self._rank_score(scored["premium_rate"], ascending=True)
        scored["price_heat_score"] = 1.0 - scored["price_position_60d"].clip(0.0, 1.0).fillna(0.5)
        scored["score"] = (
            0.28 * scored["trend_score"]
            + 0.22 * scored["momentum_5d_score"]
            + 0.18 * scored["six_sword_score"]
            + 0.12 * scored["continuity_score"]
            + 0.10 * scored["liquidity_score"]
            + 0.06 * scored["premium_score"]
            + 0.04 * scored["price_heat_score"]
        )
        return scored

    def _standardize_daily(self, daily: pd.DataFrame) -> pd.DataFrame:
        frame = super()._standardize_daily(daily)
        defaults: dict[str, float | str] = {
            "ma_3": np.nan,
            "ma_5": np.nan,
            "ma_10": np.nan,
            "ma_15": np.nan,
            "ma_20": np.nan,
            "trend_strength": 0.0,
            "return_5d": 0.0,
            "return_1d": 0.0,
            "six_sword_daily": 0.0,
            "consecutive_six_sword": 0.0,
            "band_state": "",
            "price_position_60d": 0.5,
            "market_median_double_low": np.nan,
            "market_trend_20d": 0.0,
            "market_trend_breadth": 0.0,
        }
        for column, default in defaults.items():
            if column not in frame.columns:
                frame[column] = default
        numeric_columns = [
            "trend_strength",
            "return_5d",
            "return_1d",
            "six_sword_daily",
            "consecutive_six_sword",
            "price_position_60d",
            "market_median_double_low",
            "market_trend_20d",
            "market_trend_breadth",
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(
                float(defaults[column])
            )
        frame["band_state"] = frame["band_state"].fillna("").astype(str)
        return frame


def add_trend_enhanced_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time trend features to convertible-bond daily bars."""
    if daily.empty:
        return daily.copy()
    frame = daily.copy()
    frame["ts_code"] = frame["ts_code"].astype(str).str.upper()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "pct_chg" in frame.columns:
        frame["pct_chg"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
    else:
        frame["pct_chg"] = np.nan
    frame = frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    grouped = frame.groupby("ts_code", group_keys=False)
    for window in [3, 5, 10, 15, 20, 60]:
        frame[f"ma_{window}"] = grouped["close"].transform(
            lambda series, window=window: series.rolling(window, min_periods=window).mean()
        )

    frame["return_5d"] = grouped["close"].transform(lambda series: series.pct_change(5) * 100.0)
    computed_return_1d = grouped["close"].transform(lambda series: series.pct_change() * 100.0)
    frame["return_1d"] = frame["pct_chg"].where(frame["pct_chg"].notna(), computed_return_1d)
    frame["return_20d"] = grouped["close"].transform(lambda series: series.pct_change(20) * 100.0)
    rolling_low_60 = grouped["close"].transform(lambda series: series.rolling(60, min_periods=20).min())
    rolling_high_60 = grouped["close"].transform(
        lambda series: series.rolling(60, min_periods=20).max()
    )
    denominator = (rolling_high_60 - rolling_low_60).replace(0, np.nan)
    frame["price_position_60d"] = ((frame["close"] - rolling_low_60) / denominator).clip(0.0, 1.0)

    frame["trend_strength"] = (
        (frame["ma_3"] >= frame["ma_5"]).astype(float) * 25.0
        + (frame["ma_5"] >= frame["ma_10"]).astype(float) * 25.0
        + (frame["ma_10"] >= frame["ma_15"]).astype(float) * 25.0
        + (frame["ma_15"] >= frame["ma_20"]).astype(float) * 25.0
    )
    frame["above_ma5"] = frame["close"] > frame["ma_5"]
    frame["macd_proxy"] = frame["ma_5"] > frame["ma_20"]
    frame["pullback_not_overheated"] = frame["price_position_60d"].fillna(0.5) <= 0.88
    sword_bits = [
        frame["above_ma5"],
        frame["ma_3"] >= frame["ma_5"],
        frame["ma_5"] >= frame["ma_10"],
        frame["ma_10"] >= frame["ma_20"],
        frame["return_5d"].between(0.0, 20.0, inclusive="both"),
        frame["return_1d"].between(0.0, 7.0, inclusive="both"),
    ]
    frame["six_sword_daily"] = sum(bit.astype(int) for bit in sword_bits)
    frame["six_sword_active"] = frame["six_sword_daily"] >= 4
    frame["consecutive_six_sword"] = grouped["six_sword_active"].transform(
        _consecutive_true_count
    )
    frame["band_state"] = np.select(
        [
            frame["above_ma5"]
            & frame["macd_proxy"]
            & (
                frame["pullback_not_overheated"]
                | frame["return_5d"].between(0.0, 20.0, inclusive="both")
            )
            & (frame["return_20d"].fillna(0.0) > -10.0),
            (~frame["above_ma5"]) | (frame["trend_strength"] <= 50.0),
        ],
        ["买", "卖"],
        default="观望",
    )
    premium = pd.to_numeric(
        frame.get("premium_rate", frame.get("bond_over_rate", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    frame["double_low"] = frame["close"] + premium
    market = (
        frame.groupby("trade_date")
        .agg(
            market_median_double_low=("double_low", "median"),
            market_trend_breadth=("trend_strength", lambda series: (series >= 75.0).mean()),
        )
        .sort_index()
    )
    market_ma_20 = market["market_median_double_low"].rolling(20, min_periods=10).mean()
    market["market_trend_20d"] = market["market_median_double_low"] / market_ma_20 - 1.0
    overlap_columns = [
        column for column in market.columns if column in frame.columns
    ]
    if overlap_columns:
        frame = frame.drop(columns=overlap_columns)
    frame = frame.merge(market.reset_index(), on="trade_date", how="left")
    return frame.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _consecutive_true_count(series: pd.Series) -> pd.Series:
    values = series.fillna(False).astype(bool).to_numpy()
    counts = np.zeros(len(values), dtype=int)
    running = 0
    for index, value in enumerate(values):
        running = running + 1 if value else 0
        counts[index] = running
    return pd.Series(counts, index=series.index)
