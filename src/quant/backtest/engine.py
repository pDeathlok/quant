import akquant as aq
from akquant import BacktestResult
from typing import List, Optional, Callable, Union
import pandas as pd
from pathlib import Path


class BacktestEngine:
    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        strategy,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.0
    ):
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_parquet(data)
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage
        self._result: Optional[BacktestResult] = None

    def run(
        self,
        symbols: Optional[List[str]] = None,
        benchmark: pd.Series = None,
        on_event: Callable = None,
        show_progress: bool = True,
        report_filename: Optional[str] = None
    ) -> BacktestResult:
        result = aq.run_backtest(
            data=self.data,
            strategy=self.strategy,
            initial_cash=self.initial_cash,
            symbols=symbols,
            commission_rate=self.commission_rate,
            slippage=self.slippage,
            on_event=on_event,
            show_progress=show_progress
        )

        self._result = result

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

    def get_metrics(self) -> dict:
        if self._result is None:
            return {}
        return self._result.summary()
