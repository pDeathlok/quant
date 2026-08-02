from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _clean_returns(values: pd.Series) -> pd.Series:
    returns = pd.Series(values, copy=True)
    returns = pd.to_numeric(returns, errors="coerce")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if returns.index.has_duplicates:
        returns = returns[~returns.index.duplicated(keep="last")]
    return returns.sort_index()


def _compound_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def _annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    if returns.empty:
        return 0.0
    growth = float((1.0 + returns).prod())
    if growth <= 0.0:
        return -1.0
    return float(growth ** (periods_per_year / len(returns)) - 1.0)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0.0:
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def _profit_factor(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    gains = float(values[values > 0.0].sum())
    losses = abs(float(values[values < 0.0].sum()))
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return gains / losses


class PerformanceAnalyzer:
    """Calculate return-series metrics while keeping trade metrics distinct."""

    def __init__(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        *,
        trades: Optional[pd.DataFrame] = None,
        costs: Optional[pd.DataFrame] = None,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.returns = _clean_returns(returns)
        self.benchmark = _clean_returns(benchmark) if benchmark is not None else None
        self.trades = trades.copy(deep=True) if trades is not None else pd.DataFrame()
        self.costs = costs.copy(deep=True) if costs is not None else pd.DataFrame()
        self.periods_per_year = periods_per_year
        self.risk_free_rate = float(risk_free_rate)

    @classmethod
    def from_artifacts(cls, artifacts: Any) -> "PerformanceAnalyzer":
        return cls(
            returns=artifacts.returns,
            benchmark=artifacts.benchmark_returns,
            trades=artifacts.trades,
            costs=artifacts.costs,
        )

    def summary(self) -> Dict[str, float]:
        returns = self.returns
        period_count = len(returns)
        total_return = _compound_return(returns)
        annualized_return = _annualized_return(returns, self.periods_per_year)
        volatility = (
            float(returns.std(ddof=1) * np.sqrt(self.periods_per_year))
            if period_count > 1
            else 0.0
        )
        risk_free_period = (
            (1.0 + self.risk_free_rate) ** (1.0 / self.periods_per_year) - 1.0
        )
        excess = returns - risk_free_period
        sharpe = (
            _safe_ratio(
                float(excess.mean() * self.periods_per_year),
                volatility,
            )
            if not excess.empty
            else 0.0
        )
        downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
        downside_deviation = (
            float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(self.periods_per_year))
            if len(downside) > 0
            else 0.0
        )
        sortino = (
            _safe_ratio(
                float(excess.mean() * self.periods_per_year),
                downside_deviation,
            )
            if not excess.empty
            else 0.0
        )
        drawdown = self.drawdown_series()
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        positive_period_rate = (
            float((returns > 0.0).mean()) if period_count > 0 else 0.0
        )
        value_at_risk = float(returns.quantile(0.05)) if period_count > 0 else 0.0
        tail = returns[returns <= value_at_risk]
        conditional_value_at_risk = float(tail.mean()) if not tail.empty else 0.0
        trade_net_pnl = self._trade_net_pnl()
        total_cost = self._total_cost()

        summary: Dict[str, float] = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": volatility,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": _safe_ratio(annualized_return, abs(max_drawdown)),
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": self._max_drawdown_duration(drawdown),
            "positive_period_rate": positive_period_rate,
            "win_rate": positive_period_rate,
            "period_count": period_count,
            "trade_count": len(self.trades),
            "period_profit_factor": _profit_factor(returns),
            "value_at_risk_95": value_at_risk,
            "conditional_value_at_risk_95": conditional_value_at_risk,
            "total_cost": total_cost,
            "trade_net_pnl": trade_net_pnl,
        }
        summary.update(self._benchmark_summary(risk_free_period))
        return summary

    def _benchmark_summary(self, risk_free_period: float) -> Dict[str, float]:
        if self.benchmark is None or self.benchmark.empty or self.returns.empty:
            return {}
        aligned = pd.concat(
            [self.returns.rename("strategy"), self.benchmark.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.empty:
            return {}

        strategy = aligned["strategy"]
        benchmark = aligned["benchmark"]
        active = strategy - benchmark
        tracking_error = (
            float(active.std(ddof=1) * np.sqrt(self.periods_per_year))
            if len(active) > 1
            else 0.0
        )
        benchmark_variance = float(benchmark.var(ddof=1)) if len(benchmark) > 1 else 0.0
        beta = (
            _safe_ratio(float(strategy.cov(benchmark)), benchmark_variance)
            if benchmark_variance > 0.0
            else 0.0
        )
        alpha_period = (strategy - risk_free_period) - beta * (
            benchmark - risk_free_period
        )
        strategy_total = _compound_return(strategy)
        benchmark_total = _compound_return(benchmark)
        return {
            "benchmark_total_return": benchmark_total,
            "excess_total_return": strategy_total - benchmark_total,
            "tracking_error": tracking_error,
            "information_ratio": _safe_ratio(
                float(active.mean() * self.periods_per_year),
                tracking_error,
            ),
            "beta": beta,
            "annualized_alpha": float(alpha_period.mean() * self.periods_per_year),
        }

    def _trade_net_pnl(self) -> float:
        if self.trades.empty or "net_pnl" not in self.trades.columns:
            return 0.0
        return float(pd.to_numeric(self.trades["net_pnl"], errors="coerce").fillna(0.0).sum())

    def _total_cost(self) -> float:
        if self.costs.empty:
            return 0.0
        if "total_cost" in self.costs.columns:
            return float(
                pd.to_numeric(self.costs["total_cost"], errors="coerce").fillna(0.0).sum()
            )
        components = [
            column
            for column in ("commission", "stamp_tax", "transfer_fee", "slippage_cost")
            if column in self.costs.columns
        ]
        if not components:
            return 0.0
        numeric = self.costs[components].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return float(numeric.sum(axis=1).sum())

    @staticmethod
    def _max_drawdown_duration(drawdown: pd.Series) -> int:
        if drawdown.empty:
            return 0
        underwater = drawdown < 0.0
        if not underwater.any():
            return 0
        groups = (~underwater).cumsum()
        return int(underwater.groupby(groups).sum().max())

    def rolling_sharpe(self, window: int = 60) -> pd.Series:
        if window <= 1:
            raise ValueError("window must be greater than one")
        risk_free_period = (
            (1.0 + self.risk_free_rate) ** (1.0 / self.periods_per_year) - 1.0
        )
        excess = self.returns - risk_free_period
        rolling_mean = excess.rolling(window).mean()
        rolling_std = self.returns.rolling(window).std()
        return rolling_mean / rolling_std * np.sqrt(self.periods_per_year)

    def drawdown_series(self) -> pd.Series:
        if self.returns.empty:
            return pd.Series(dtype=float, index=self.returns.index, name="drawdown")
        cumulative = (1.0 + self.returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        drawdown.name = "drawdown"
        return drawdown

    def monthly_returns(self) -> pd.DataFrame:
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.DataFrame()
        monthly = self.returns.resample("M").apply(lambda values: (1.0 + values).prod() - 1.0)
        return monthly.to_frame("return")
