from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_summary_separates_period_and_trade_metrics() -> None:
    from quant.analysis import PerformanceAnalyzer

    dates = pd.date_range("2026-07-27", periods=3, freq="D")
    returns = pd.Series([0.10, -0.05, 0.02], index=dates)
    trades = pd.DataFrame(
        {
            "net_pnl": [120.0, -40.0],
            "return_pct": [0.012, -0.004],
        }
    )
    costs = pd.DataFrame({"total_cost": [1.25, 0.50]})

    summary = PerformanceAnalyzer(returns, trades=trades, costs=costs).summary()

    assert summary["total_return"] == pytest.approx(1.10 * 0.95 * 1.02 - 1.0)
    assert summary["max_drawdown"] == pytest.approx(-0.05)
    assert summary["positive_period_rate"] == pytest.approx(2 / 3)
    assert summary["win_rate"] == summary["positive_period_rate"]
    assert summary["period_count"] == 3
    assert summary["trade_count"] == 2
    assert summary["total_cost"] == pytest.approx(1.75)
    assert summary["trade_net_pnl"] == pytest.approx(80.0)
    assert summary["period_profit_factor"] == pytest.approx(2.4)
    assert summary["value_at_risk_95"] == pytest.approx(returns.quantile(0.05))
    assert summary["conditional_value_at_risk_95"] == pytest.approx(-0.05)


def test_summary_aligns_benchmark_and_calculates_relative_metrics() -> None:
    from quant.analysis import PerformanceAnalyzer

    dates = pd.date_range("2026-07-27", periods=4, freq="D")
    returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates)
    benchmark = pd.Series(
        [0.005, 0.01, -0.005, 0.02, 0.50],
        index=dates.append(pd.DatetimeIndex([pd.Timestamp("2026-08-01")])),
    )

    summary = PerformanceAnalyzer(returns, benchmark=benchmark).summary()

    aligned_benchmark = benchmark.reindex(dates)
    expected_beta = returns.cov(aligned_benchmark) / aligned_benchmark.var()
    assert summary["benchmark_total_return"] == pytest.approx(
        (1.0 + aligned_benchmark).prod() - 1.0
    )
    assert summary["beta"] == pytest.approx(expected_beta)
    assert np.isfinite(summary["tracking_error"])
    assert np.isfinite(summary["information_ratio"])
    assert np.isfinite(summary["annualized_alpha"])


def test_empty_returns_produce_stable_zero_summary() -> None:
    from quant.analysis import PerformanceAnalyzer

    summary = PerformanceAnalyzer(pd.Series(dtype=float)).summary()

    assert summary["total_return"] == 0.0
    assert summary["annualized_return"] == 0.0
    assert summary["max_drawdown"] == 0.0
    assert summary["period_count"] == 0
    assert summary["trade_count"] == 0
    assert summary["value_at_risk_95"] == 0.0
    assert summary["conditional_value_at_risk_95"] == 0.0


def test_max_drawdown_duration_counts_underwater_periods() -> None:
    from quant.analysis import PerformanceAnalyzer

    dates = pd.date_range("2026-07-27", periods=5, freq="D")
    returns = pd.Series([0.10, -0.02, -0.03, 0.01, 0.10], index=dates)

    summary = PerformanceAnalyzer(returns).summary()

    assert summary["max_drawdown_duration"] == 3


def test_from_artifacts_uses_trade_and_cost_tables() -> None:
    from quant.analysis import PerformanceAnalyzer
    from quant.backtest.artifacts import BacktestArtifacts

    dates = pd.date_range("2026-07-27", periods=2, freq="D")
    artifacts = BacktestArtifacts(
        equity_curve=pd.Series([100.0, 101.0], index=dates),
        returns=pd.Series([0.0, 0.01], index=dates),
        positions=pd.DataFrame(),
        orders=pd.DataFrame(),
        executions=pd.DataFrame(),
        trades=pd.DataFrame({"net_pnl": [1.0]}),
        costs=pd.DataFrame({"total_cost": [0.25]}),
    )

    summary = PerformanceAnalyzer.from_artifacts(artifacts).summary()

    assert summary["trade_count"] == 1
    assert summary["total_cost"] == pytest.approx(0.25)
