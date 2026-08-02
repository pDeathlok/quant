"""Cross-sectional factor diagnostics with point-in-time forward returns."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class FactorAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        *,
        date_column: str = "date",
        symbol_column: str = "symbol",
        minimum_cross_section: int = 5,
    ) -> None:
        if date_column not in data.columns or symbol_column not in data.columns:
            raise ValueError("factor data must contain date and symbol columns")
        if minimum_cross_section < 2:
            raise ValueError("minimum_cross_section must be at least 2")
        normalized = data.copy()
        normalized[date_column] = pd.to_datetime(normalized[date_column], errors="coerce")
        if normalized[date_column].isna().any():
            raise ValueError("factor data contain invalid dates")
        normalized[symbol_column] = normalized[symbol_column].astype(str)
        self.data = normalized.sort_values(
            [date_column, symbol_column]
        ).reset_index(drop=True)
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.minimum_cross_section = int(minimum_cross_section)

    @staticmethod
    def add_forward_returns(
        prices: pd.DataFrame,
        *,
        periods: Iterable[int] = (1, 5, 20),
        date_column: str = "date",
        symbol_column: str = "symbol",
        price_column: str = "close",
    ) -> pd.DataFrame:
        required = {date_column, symbol_column, price_column}
        missing = sorted(required - set(prices.columns))
        if missing:
            raise ValueError(f"prices missing columns: {missing}")
        output = prices.copy()
        output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
        output[price_column] = pd.to_numeric(output[price_column], errors="coerce")
        output = output.sort_values([symbol_column, date_column]).reset_index(drop=True)
        grouped = output.groupby(symbol_column, sort=False)[price_column]
        for period in periods:
            horizon = int(period)
            if horizon <= 0:
                raise ValueError("forward return periods must be positive")
            output[f"forward_return_{horizon}"] = (
                grouped.shift(-horizon) / output[price_column] - 1.0
            )
        return output

    def information_coefficient(
        self,
        factor_column: str,
        forward_return_column: str,
        *,
        method: str = "pearson",
    ) -> pd.DataFrame:
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be pearson or spearman")
        required = {factor_column, forward_return_column}
        missing = sorted(required - set(self.data.columns))
        if missing:
            raise ValueError(f"factor data missing columns: {missing}")
        rows: list[dict[str, object]] = []
        for date, group in self.data.groupby(self.date_column, sort=True):
            pair = group[[factor_column, forward_return_column]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(pair) < self.minimum_cross_section:
                continue
            ic = pair[factor_column].corr(pair[forward_return_column], method=method)
            rows.append({"date": date, "ic": float(ic), "n": len(pair), "method": method})
        return pd.DataFrame(rows, columns=["date", "ic", "n", "method"])

    def _quantile_membership(
        self,
        factor_column: str,
        quantiles: int,
    ) -> pd.DataFrame:
        if quantiles < 2:
            raise ValueError("quantiles must be at least 2")
        if factor_column not in self.data.columns:
            raise ValueError(f"factor data missing column: {factor_column}")
        frames: list[pd.DataFrame] = []
        for _, group in self.data.groupby(self.date_column, sort=True):
            current = group.copy()
            factor = pd.to_numeric(current[factor_column], errors="coerce")
            valid = factor.notna()
            current = current.loc[valid].copy()
            if len(current) < max(self.minimum_cross_section, quantiles):
                continue
            ranks = factor.loc[valid].rank(method="first")
            current["quantile"] = pd.qcut(
                ranks,
                q=quantiles,
                labels=range(1, quantiles + 1),
            ).astype(int)
            frames.append(current)
        if not frames:
            return pd.DataFrame(columns=[*self.data.columns, "quantile"])
        return pd.concat(frames, ignore_index=True, sort=False)

    def quantile_returns(
        self,
        factor_column: str,
        forward_return_column: str,
        *,
        quantiles: int = 5,
    ) -> pd.DataFrame:
        membership = self._quantile_membership(factor_column, quantiles)
        if forward_return_column not in membership.columns:
            raise ValueError(f"factor data missing column: {forward_return_column}")
        membership[forward_return_column] = pd.to_numeric(
            membership[forward_return_column], errors="coerce"
        )
        grouped = (
            membership.dropna(subset=[forward_return_column])
            .groupby([self.date_column, "quantile"], sort=True)[forward_return_column]
            .mean()
        )
        rows = [
            {"date": date, "quantile": quantile, "mean_return": float(value)}
            for (date, quantile), value in grouped.items()
        ]
        for date, values in grouped.groupby(level=0):
            by_quantile = values.droplevel(0)
            if 1 in by_quantile.index and quantiles in by_quantile.index:
                rows.append(
                    {
                        "date": date,
                        "quantile": "long_short",
                        "mean_return": float(
                            by_quantile.loc[quantiles] - by_quantile.loc[1]
                        ),
                    }
                )
        return pd.DataFrame(rows).sort_values(["date", "quantile"], key=lambda s: s.astype(str)).reset_index(drop=True)

    def quantile_turnover(
        self,
        factor_column: str,
        *,
        quantiles: int = 5,
    ) -> pd.DataFrame:
        membership = self._quantile_membership(factor_column, quantiles)
        previous: dict[int, set[str]] = {}
        rows: list[dict[str, object]] = []
        for date, date_group in membership.groupby(self.date_column, sort=True):
            for quantile in range(1, quantiles + 1):
                current = set(
                    date_group.loc[
                        date_group["quantile"].eq(quantile), self.symbol_column
                    ].astype(str)
                )
                prior = previous.get(quantile)
                turnover = (
                    np.nan
                    if prior is None or not prior
                    else 1.0 - len(current & prior) / len(prior)
                )
                rows.append(
                    {"date": date, "quantile": quantile, "turnover": turnover}
                )
                previous[quantile] = current
        return pd.DataFrame(rows, columns=["date", "quantile", "turnover"])

    def ic_decay(
        self,
        factor_column: str,
        horizons: Iterable[int],
        *,
        method: str = "spearman",
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for horizon in horizons:
            column = f"forward_return_{int(horizon)}"
            values = self.information_coefficient(
                factor_column,
                column,
                method=method,
            )["ic"]
            rows.append(
                {
                    "horizon": int(horizon),
                    "mean_ic": float(values.mean()) if not values.empty else np.nan,
                    "ic_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "periods": len(values),
                }
            )
        return pd.DataFrame(rows)
