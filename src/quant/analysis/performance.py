import pandas as pd
import numpy as np
from typing import Dict, Optional


class PerformanceAnalyzer:
    def __init__(self, returns: pd.Series, benchmark: Optional[pd.Series] = None):
        self.returns = returns
        self.benchmark = benchmark

    def summary(self) -> Dict:
        total_return = (1 + self.returns).prod() - 1
        n_periods = len(self.returns)
        annual_factor = 252 if n_periods > 0 else 1

        annualized_return = (1 + total_return) ** (annual_factor / n_periods) - 1 if n_periods > 0 else 0
        volatility = self.returns.std() * np.sqrt(252) if len(self.returns) > 1 else 0

        sharpe = annualized_return / volatility if volatility > 0 else 0

        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        win_rate = (self.returns > 0).sum() / len(self.returns) if len(self.returns) > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trade_count": len(self.returns)
        }

    def rolling_sharpe(self, window: int = 60) -> pd.Series:
        rolling_mean = self.returns.rolling(window).mean()
        rolling_std = self.returns.rolling(window).std()
        return rolling_mean / rolling_std * np.sqrt(252)

    def drawdown_series(self) -> pd.Series:
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.cummax()
        return (cumulative - peak) / peak

    def monthly_returns(self) -> pd.DataFrame:
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        monthly = self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        return monthly.to_frame("return")
