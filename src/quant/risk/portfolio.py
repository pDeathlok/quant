"""Portfolio-level concentration, tail-risk, stress, and capacity analytics."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


class PortfolioRiskAnalyzer:
    def __init__(
        self,
        *,
        confidence_level: float = 0.95,
        max_participation_rate: float = 0.10,
        max_liquidation_days: float = 5.0,
    ) -> None:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if not 0 < max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1]")
        if max_liquidation_days <= 0:
            raise ValueError("max_liquidation_days must be positive")
        self.confidence_level = confidence_level
        self.max_participation_rate = max_participation_rate
        self.max_liquidation_days = max_liquidation_days

    @staticmethod
    def _weights(weights: pd.Series) -> pd.Series:
        values = pd.to_numeric(pd.Series(weights), errors="coerce").fillna(0.0)
        values.index = values.index.astype(str)
        values = values.groupby(level=0).sum()
        if (values < 0).any():
            raise ValueError("weights must be non-negative")
        if float(values.sum()) <= 0:
            raise ValueError("weights must contain positive exposure")
        return values

    def concentration(
        self,
        weights: pd.Series,
        *,
        industries: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        values = self._weights(weights)
        hhi = float(values.pow(2).sum())
        result: dict[str, object] = {
            "gross_exposure": float(values.sum()),
            "max_weight": float(values.max()),
            "hhi": hhi,
            "effective_positions": float(1 / hhi) if hhi > 0 else 0.0,
            "top5_weight": float(values.nlargest(5).sum()),
        }
        if industries is not None:
            labels = pd.Series(
                {symbol: str(industries.get(symbol, "UNKNOWN")) for symbol in values.index}
            )
            result["industry_exposure"] = {
                str(key): float(value)
                for key, value in values.groupby(labels).sum().sort_values(ascending=False).items()
            }
        return result

    def historical_var_cvar(self, portfolio_returns: pd.Series) -> dict[str, float | int]:
        returns = pd.to_numeric(pd.Series(portfolio_returns), errors="coerce").dropna()
        if returns.empty:
            raise ValueError("portfolio_returns are empty")
        threshold = float(returns.quantile(1 - self.confidence_level))
        tail = returns.loc[returns <= threshold]
        return {
            "confidence_level": self.confidence_level,
            "var": max(0.0, -threshold),
            "cvar": max(0.0, -float(tail.mean())),
            "observations": int(len(returns)),
        }

    def stress_test(self, weights: pd.Series, scenario_returns: pd.DataFrame) -> pd.DataFrame:
        values = self._weights(weights)
        scenarios = scenario_returns.apply(pd.to_numeric, errors="coerce")
        missing = values.index.difference(scenarios.columns.astype(str))
        if len(missing):
            raise ValueError(f"scenario_returns missing symbols: {missing.tolist()}")
        scenarios.columns = scenarios.columns.astype(str)
        portfolio_returns = scenarios.reindex(columns=values.index).mul(values, axis=1).sum(axis=1)
        return pd.DataFrame(
            {
                "portfolio_return": portfolio_returns,
                "loss": (-portfolio_returns).clip(lower=0.0),
            }
        )

    def capacity(
        self,
        *,
        positions: pd.Series,
        prices: pd.Series,
        average_daily_volume: pd.Series,
    ) -> pd.DataFrame:
        quantities = pd.to_numeric(pd.Series(positions), errors="coerce").fillna(0.0).abs()
        quantities.index = quantities.index.astype(str)
        price_values = pd.to_numeric(pd.Series(prices), errors="coerce")
        price_values.index = price_values.index.astype(str)
        adv = pd.to_numeric(pd.Series(average_daily_volume), errors="coerce")
        adv.index = adv.index.astype(str)
        missing = quantities.index.difference(price_values.dropna().index.intersection(adv.dropna().index))
        if len(missing):
            raise ValueError(f"missing price or average_daily_volume: {missing.tolist()}")
        if (price_values.reindex(quantities.index) <= 0).any() or (adv.reindex(quantities.index) <= 0).any():
            raise ValueError("prices and average_daily_volume must be positive")
        capacity_shares = adv.reindex(quantities.index) * self.max_participation_rate
        liquidation_days = quantities / capacity_shares
        frame = pd.DataFrame(
            {
                "quantity": quantities,
                "price": price_values.reindex(quantities.index),
                "position_value": quantities * price_values.reindex(quantities.index),
                "average_daily_volume": adv.reindex(quantities.index),
                "max_daily_trade_shares": capacity_shares,
                "liquidation_days": liquidation_days,
                "capacity_breach": liquidation_days > self.max_liquidation_days,
            }
        )
        frame.index.name = "symbol"
        return frame.sort_index()
