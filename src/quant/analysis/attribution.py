import pandas as pd
import numpy as np
from typing import Dict, Optional


class AttributionAnalyzer:
    def __init__(self, returns: pd.Series, positions: pd.DataFrame):
        self.returns = returns
        self.positions = positions

    def by_symbol(self) -> pd.DataFrame:
        if "symbol" not in self.positions.columns:
            return pd.DataFrame()

        attribution = self.positions.groupby("symbol").apply(
            lambda x: (x["return"] * x["weight"]).sum()
        ).to_frame("attribution")

        return attribution.sort_values("attribution", ascending=False)

    def by_period(self, freq: str = "D") -> pd.DataFrame:
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        period_returns = self.returns.resample(freq).apply(lambda x: (1 + x).prod() - 1)
        return period_returns.to_frame("return")

    def factor_exposure(self) -> Dict[str, float]:
        exposures = {}

        if "market" in self.positions.columns:
            market_ret = self.positions["market"]
            strategy_ret = self.returns

            covariance = market_ret.cov(strategy_ret)
            market_var = market_ret.var()

            if market_var > 0:
                exposures["beta"] = covariance / market_var

        return exposures
