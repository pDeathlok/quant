import akquant as aq
from akquant import BacktestResult
from typing import List, Optional, Callable, Union
import pandas as pd
from pathlib import Path

from quant.analysis.performance import PerformanceAnalyzer
from quant.backtest.artifacts import BacktestArtifacts
from quant.backtest.execution import AShareExecutionConfig
from quant.backtest.tradability import (
    AShareTradabilityPolicy,
    TradabilityDecision,
)


def _akquant_execution_kwargs(
    config: AShareExecutionConfig,
) -> dict:
    """Translate the project policy at the external-engine boundary."""

    return config.to_dict()


class BacktestEngine:
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        strategy,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.0,
        execution_config: Optional[AShareExecutionConfig] = None,
        tradability_policy: Optional[AShareTradabilityPolicy] = None,
    ):
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_parquet(data)
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.execution_config = execution_config
        self.tradability_policy = tradability_policy
        self._result: Optional[BacktestResult] = None
        self._artifacts: Optional[BacktestArtifacts] = None

    def run(
        self,
        symbols: Optional[List[str]] = None,
        benchmark: pd.Series = None,
        on_event: Callable = None,
        show_progress: bool = True,
        report_filename: Optional[str] = None
    ) -> BacktestResult:
        run_kwargs = {
            "data": self.data,
            "strategy": self.strategy,
            "initial_cash": self.initial_cash,
            "symbols": symbols,
            "commission_rate": self.commission_rate,
            "slippage": self.slippage,
            "on_event": on_event,
            "show_progress": show_progress,
        }
        if self.execution_config is not None:
            run_kwargs.update(_akquant_execution_kwargs(self.execution_config))

        result = aq.run_backtest(**run_kwargs)

        self._result = result
        artifact_metadata = {}
        if self.execution_config is not None:
            artifact_metadata["execution_policy"] = (
                self.execution_config.to_metadata()
            )
        if self.tradability_policy is not None:
            artifact_metadata["tradability"] = self.tradability_policy.metadata()
        self._artifacts = BacktestArtifacts.from_akquant(
            result,
            benchmark_returns=benchmark,
            metadata=artifact_metadata,
        )

        if report_filename:
            if benchmark is not None:
                result.report(
                    filename=report_filename,
                    show=False,
                    benchmark=benchmark
                )
            else:
                result.report(
                    filename=report_filename,
                    show=False
                )

        return result

    def run_with_report(
        self,
        symbols: Optional[List[str]] = None,
        benchmark: pd.Series = None,
        report_filename: str = "backtest_report.html"
    ) -> BacktestResult:
        return self.run(
            symbols=symbols,
            benchmark=benchmark,
            on_event=None,
            show_progress=True,
            report_filename=report_filename
        )

    @property
    def result(self) -> Optional[BacktestResult]:
        return self._result

    @property
    def artifacts(self) -> Optional[BacktestArtifacts]:
        return self._artifacts

    def get_artifacts(self) -> Optional[BacktestArtifacts]:
        return self._artifacts

    def get_metrics(self) -> dict:
        if self._artifacts is None:
            return {}
        return PerformanceAnalyzer.from_artifacts(self._artifacts).summary()

    def check_order(
        self,
        *,
        trade_date: str,
        symbol: str,
        side: str,
        price: float,
    ) -> TradabilityDecision:
        if self.tradability_policy is None:
            return TradabilityDecision(True, "policy_not_configured")
        return self.tradability_policy.check_order(
            trade_date=trade_date,
            symbol=symbol,
            side=side,
            price=price,
        )
