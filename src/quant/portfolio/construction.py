"""Project-owned long-only portfolio construction algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioConstraints:
    max_weight: float = 0.10
    max_industry_weight: float = 0.30
    max_turnover: float = 1.00
    cash_buffer: float = 0.02
    min_position_weight: float = 0.00
    long_only: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if not 0 < self.max_industry_weight <= 1:
            raise ValueError("max_industry_weight must be in (0, 1]")
        if not 0 <= self.max_turnover <= 2:
            raise ValueError("max_turnover must be in [0, 2]")
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        if not 0 <= self.min_position_weight <= self.max_weight:
            raise ValueError("min_position_weight must be between 0 and max_weight")
        if not self.long_only:
            raise ValueError("only long-only A-share portfolios are supported")


@dataclass(frozen=True)
class PortfolioResult:
    target_weights: pd.Series
    cash_weight: float
    turnover: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = pd.to_numeric(
            pd.Series(self.target_weights, copy=True), errors="coerce"
        ).dropna()
        weights.index = weights.index.astype(str)
        weights = weights.groupby(level=0).sum().sort_index().astype(float)
        if (weights < -1e-12).any():
            raise ValueError("target_weights cannot contain negative values")
        if float(weights.sum()) > 1 + 1e-8:
            raise ValueError("target_weights cannot sum above 1")
        object.__setattr__(self, "target_weights", weights)
        object.__setattr__(self, "cash_weight", float(self.cash_weight))
        object.__setattr__(self, "turnover", float(self.turnover))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class PortfolioConstructor:
    def __init__(self, constraints: PortfolioConstraints | None = None) -> None:
        self.constraints = constraints or PortfolioConstraints()

    @staticmethod
    def _prepare_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(candidates, pd.DataFrame):
            raise TypeError("candidates must be a pandas DataFrame")
        symbol_column = "ts_code" if "ts_code" in candidates.columns else "symbol"
        if symbol_column not in candidates.columns:
            raise ValueError("candidates must contain ts_code or symbol")
        prepared = candidates.copy()
        prepared["symbol"] = prepared[symbol_column].astype(str).str.strip()
        if "eligible" in prepared.columns:
            prepared = prepared.loc[prepared["eligible"].fillna(False).astype(bool)]
        prepared = prepared.loc[prepared["symbol"].ne("")]
        prepared = prepared.drop_duplicates("symbol", keep="last").reset_index(drop=True)
        if "industry" not in prepared.columns:
            prepared["industry"] = "UNKNOWN"
        prepared["industry"] = prepared["industry"].fillna("UNKNOWN").astype(str)
        if prepared.empty:
            raise ValueError("candidate universe is empty")
        return prepared

    def _apply_caps(
        self,
        raw_weights: pd.Series,
        industries: Mapping[str, str] | None = None,
    ) -> pd.Series:
        weights = pd.to_numeric(raw_weights, errors="coerce").fillna(0.0).clip(lower=0.0)
        weights.index = weights.index.astype(str)
        weights = weights.groupby(level=0).sum()
        if float(weights.sum()) <= 0:
            weights[:] = 1.0
        investable = 1.0 - self.constraints.cash_buffer
        weights = weights / float(weights.sum()) * investable
        weights = weights.clip(upper=self.constraints.max_weight)

        if industries:
            industry_series = pd.Series(
                {symbol: str(industries.get(symbol, "UNKNOWN")) for symbol in weights.index}
            )
            for industry in sorted(industry_series.unique()):
                members = industry_series.index[industry_series.eq(industry)]
                total = float(weights.reindex(members).sum())
                if total > self.constraints.max_industry_weight:
                    weights.loc[members] *= self.constraints.max_industry_weight / total
        if self.constraints.min_position_weight > 0:
            weights = weights.loc[weights >= self.constraints.min_position_weight]
        return weights.sort_index()

    def _apply_turnover(
        self,
        target: pd.Series,
        current_weights: pd.Series | None,
    ) -> tuple[pd.Series, float, bool]:
        if current_weights is None:
            return target, float(target.abs().sum()), False
        current = pd.to_numeric(pd.Series(current_weights), errors="coerce").fillna(0.0)
        current.index = current.index.astype(str)
        current = current.groupby(level=0).sum().clip(lower=0.0)
        symbols = target.index.union(current.index)
        desired = target.reindex(symbols, fill_value=0.0)
        starting = current.reindex(symbols, fill_value=0.0)
        raw_turnover = float((desired - starting).abs().sum())
        if raw_turnover <= self.constraints.max_turnover or raw_turnover == 0:
            return desired.loc[desired > 1e-12].sort_index(), raw_turnover, False
        scale = self.constraints.max_turnover / raw_turnover
        adjusted = starting + (desired - starting) * scale
        return (
            adjusted.loc[adjusted > 1e-12].sort_index(),
            float((adjusted - starting).abs().sum()),
            True,
        )

    def _result(
        self,
        target: pd.Series,
        *,
        method: str,
        current_weights: pd.Series | None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> PortfolioResult:
        adjusted, turnover, turnover_limited = self._apply_turnover(
            target,
            current_weights,
        )
        payload = {
            "method": method,
            "turnover_limited": turnover_limited,
            **dict(diagnostics or {}),
        }
        return PortfolioResult(
            target_weights=adjusted,
            cash_weight=max(0.0, 1.0 - float(adjusted.sum())),
            turnover=turnover,
            diagnostics=payload,
        )

    def equal_weight(
        self,
        candidates: pd.DataFrame,
        *,
        current_weights: pd.Series | None = None,
    ) -> PortfolioResult:
        prepared = self._prepare_candidates(candidates)
        raw = pd.Series(1.0, index=prepared["symbol"])
        industries = prepared.set_index("symbol")["industry"].to_dict()
        target = self._apply_caps(raw, industries)
        return self._result(
            target,
            method="equal_weight",
            current_weights=current_weights,
        )

    def score_weight(
        self,
        candidates: pd.DataFrame,
        *,
        score_column: str = "score",
        current_weights: pd.Series | None = None,
    ) -> PortfolioResult:
        prepared = self._prepare_candidates(candidates)
        if score_column not in prepared.columns:
            raise ValueError(f"candidates missing score column: {score_column}")
        scores = pd.to_numeric(prepared[score_column], errors="coerce").fillna(0.0)
        scores.index = prepared["symbol"]
        positive = scores.clip(lower=0.0)
        positive = positive.loc[positive > 0]
        if positive.empty:
            positive = pd.Series(1.0, index=prepared["symbol"])
        industries = prepared.set_index("symbol")["industry"].to_dict()
        target = self._apply_caps(positive, industries)
        return self._result(
            target,
            method="score_weight",
            current_weights=current_weights,
        )

    def inverse_volatility(
        self,
        returns: pd.DataFrame,
        *,
        current_weights: pd.Series | None = None,
    ) -> PortfolioResult:
        clean = returns.apply(pd.to_numeric, errors="coerce")
        volatility = clean.std(ddof=1).replace(0.0, np.nan)
        inverse = (1.0 / volatility).replace([np.inf, -np.inf], np.nan).dropna()
        if inverse.empty:
            raise ValueError("returns contain no positive finite volatility")
        target = self._apply_caps(inverse)
        return self._result(
            target,
            method="inverse_volatility",
            current_weights=current_weights,
        )

    def minimum_variance(
        self,
        returns: pd.DataFrame,
        *,
        current_weights: pd.Series | None = None,
    ) -> PortfolioResult:
        clean = returns.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if clean.shape[1] < 2:
            raise ValueError("minimum_variance requires at least two assets")
        covariance = clean.cov(min_periods=2).fillna(0.0).to_numpy(dtype=float)
        covariance = covariance + np.eye(covariance.shape[0]) * 1e-10
        investable = 1.0 - self.constraints.cash_buffer
        asset_count = clean.shape[1]
        if asset_count * self.constraints.max_weight + 1e-12 < investable:
            raise ValueError("max_weight makes the minimum-variance problem infeasible")
        initial = np.repeat(investable / asset_count, asset_count)
        bounds = [(0.0, self.constraints.max_weight)] * asset_count
        solution = minimize(
            lambda weights: float(weights @ covariance @ weights),
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - investable}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not solution.success:
            raise RuntimeError(f"minimum-variance optimization failed: {solution.message}")
        target = pd.Series(solution.x, index=clean.columns, dtype=float)
        target = self._apply_caps(target)
        return self._result(
            target,
            method="minimum_variance",
            current_weights=current_weights,
            diagnostics={"solver": "SLSQP", "solver_success": True},
        )
