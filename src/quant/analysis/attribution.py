from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


class AttributionAnalyzer:
    def __init__(self, returns: pd.Series, positions: pd.DataFrame):
        self.returns = pd.Series(returns, copy=True)
        self.positions = positions.copy()

    def _contributions(self) -> pd.Series:
        if "contribution" in self.positions.columns:
            return pd.to_numeric(self.positions["contribution"], errors="coerce").fillna(0.0)
        if {"return", "weight"}.issubset(self.positions.columns):
            returns = pd.to_numeric(self.positions["return"], errors="coerce").fillna(0.0)
            weights = pd.to_numeric(self.positions["weight"], errors="coerce").fillna(0.0)
            return returns * weights
        return pd.Series(0.0, index=self.positions.index, dtype=float)

    def by_dimension(self, dimension: str) -> pd.DataFrame:
        if dimension not in self.positions.columns:
            return pd.DataFrame(columns=["attribution"])
        frame = self.positions[[dimension]].copy()
        frame["attribution"] = self._contributions()
        return (
            frame.groupby(dimension, dropna=False)["attribution"]
            .sum()
            .sort_values(ascending=False)
            .to_frame()
        )

    def by_symbol(self) -> pd.DataFrame:
        return self.by_dimension("symbol")

    def by_industry(self) -> pd.DataFrame:
        return self.by_dimension("industry")

    def by_period(self, freq: str = "D") -> pd.DataFrame:
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.DataFrame()
        period_returns = self.returns.resample(freq).apply(
            lambda values: (1 + values).prod() - 1
        )
        return period_returns.to_frame("return")

    def factor_exposure(
        self,
        factor_columns: Iterable[str] | None = None,
    ) -> Dict[str, float]:
        exposures: Dict[str, float] = {}
        if factor_columns is not None:
            weights = (
                pd.to_numeric(self.positions["weight"], errors="coerce").fillna(0.0)
                if "weight" in self.positions.columns
                else pd.Series(1.0 / max(len(self.positions), 1), index=self.positions.index)
            )
            for column in factor_columns:
                if column not in self.positions.columns:
                    continue
                values = pd.to_numeric(self.positions[column], errors="coerce")
                valid = values.notna() & weights.notna()
                exposures[column] = float((values.loc[valid] * weights.loc[valid]).sum())
            return exposures

        if "market" in self.positions.columns:
            market_ret = pd.to_numeric(self.positions["market"], errors="coerce")
            strategy_ret = pd.to_numeric(self.returns, errors="coerce")
            aligned = pd.concat([market_ret, strategy_ret], axis=1).dropna()
            if not aligned.empty:
                market_var = aligned.iloc[:, 0].var()
                if market_var > 0:
                    exposures["beta"] = float(
                        aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / market_var
                    )
        return exposures

    @staticmethod
    def brinson_attribution(
        portfolio: pd.DataFrame,
        benchmark: pd.DataFrame,
        *,
        group_column: str = "group",
        weight_column: str = "weight",
        return_column: str = "return",
    ) -> pd.DataFrame:
        required = {group_column, weight_column, return_column}
        for name, frame in (("portfolio", portfolio), ("benchmark", benchmark)):
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")
        p = portfolio[list(required)].rename(
            columns={weight_column: "portfolio_weight", return_column: "portfolio_return"}
        )
        b = benchmark[list(required)].rename(
            columns={weight_column: "benchmark_weight", return_column: "benchmark_return"}
        )
        merged = p.merge(b, on=group_column, how="outer").fillna(0.0)
        for column in (
            "portfolio_weight",
            "portfolio_return",
            "benchmark_weight",
            "benchmark_return",
        ):
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        benchmark_total = float(
            (merged["benchmark_weight"] * merged["benchmark_return"]).sum()
        )
        merged["allocation"] = (
            merged["portfolio_weight"] - merged["benchmark_weight"]
        ) * (merged["benchmark_return"] - benchmark_total)
        merged["selection"] = merged["benchmark_weight"] * (
            merged["portfolio_return"] - merged["benchmark_return"]
        )
        merged["interaction"] = (
            merged["portfolio_weight"] - merged["benchmark_weight"]
        ) * (merged["portfolio_return"] - merged["benchmark_return"])
        merged["total_effect"] = merged[
            ["allocation", "selection", "interaction"]
        ].sum(axis=1)
        return merged.sort_values(group_column).reset_index(drop=True)
