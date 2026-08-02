from __future__ import annotations

import pandas as pd
import pytest


class FakeResult:
    def __init__(self) -> None:
        dates = pd.date_range("2026-07-27", periods=2, freq="D", tz="Asia/Shanghai")
        self.initial_cash = 100_000.0
        self.equity_curve_daily = pd.Series([100_000.0, 101_000.0], index=dates)
        self.daily_returns = pd.Series([0.0, 0.01], index=dates)
        self.positions_df = pd.DataFrame()
        self.orders_df = pd.DataFrame()
        self.executions_df = pd.DataFrame()
        self.trades_df = pd.DataFrame({"net_pnl": [1_000.0]})
        self.report_calls: list[dict[str, object]] = []

    def report(self, **kwargs: object) -> None:
        self.report_calls.append(dict(kwargs))


def test_engine_preserves_raw_result_and_exposes_stable_artifacts(monkeypatch) -> None:
    from quant.backtest import BacktestEngine

    raw = FakeResult()
    monkeypatch.setattr("quant.backtest.engine.aq.run_backtest", lambda **kwargs: raw)
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-27", periods=2, freq="D"),
            "symbol": ["TEST", "TEST"],
            "close": [10.0, 10.1],
        }
    )
    benchmark = pd.Series([0.0, 0.005], index=raw.equity_curve_daily.index)
    engine = BacktestEngine(data=data, strategy=object())

    returned = engine.run(
        benchmark=benchmark,
        show_progress=False,
        report_filename="standard-report.html",
    )

    assert returned is raw
    assert engine.result is raw
    assert engine.artifacts is engine.get_artifacts()
    assert engine.artifacts is not None
    assert engine.artifacts.returns.tolist() == pytest.approx([0.0, 0.01])
    assert engine.artifacts.benchmark_returns is not None
    assert raw.report_calls == [
        {
            "filename": "standard-report.html",
            "show": False,
            "benchmark": benchmark,
        }
    ]
    metrics = engine.get_metrics()
    assert metrics["period_count"] == 2
    assert metrics["trade_count"] == 1


def test_engine_artifacts_are_none_before_run() -> None:
    from quant.backtest import BacktestEngine

    engine = BacktestEngine(
        data=pd.DataFrame({"date": [], "symbol": [], "close": []}),
        strategy=object(),
    )

    assert engine.artifacts is None
    assert engine.get_artifacts() is None
    assert engine.get_metrics() == {}
