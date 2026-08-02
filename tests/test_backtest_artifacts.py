from __future__ import annotations

from types import MappingProxyType

import pandas as pd
import pytest


class FakeAkquantResult:
    def __init__(self) -> None:
        dates = pd.date_range("2026-07-27", periods=3, freq="D", tz="Asia/Shanghai")
        self.initial_cash = 100_000.0
        self.equity_curve_daily = pd.Series(
            [100_000.0, 110_000.0, 99_000.0],
            index=dates,
            name="equity",
        )
        self.daily_returns = pd.Series([0.0, 0.1, -0.1], index=dates)
        self.positions_df = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "long_shares": [0.0, 1_000.0, 0.0],
            }
        )
        self.orders_df = pd.DataFrame(
            {
                "id": ["order-1", "order-2"],
                "symbol": ["000001.SZ", "000001.SZ"],
                "created_at": dates[:2],
                "commission": [5.0, 5.0],
            }
        )
        self.executions_df = pd.DataFrame(
            {
                "order_id": ["order-1", "order-2"],
                "symbol": ["000001.SZ", "000001.SZ"],
                "timestamp": dates[:2],
                "commission": [5.0, 5.0],
                "stamp_tax": [0.0, 10.0],
            }
        )
        self.trades_df = pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "net_pnl": [-1_000.0],
                "return_pct": [-0.01],
            }
        )


def test_from_akquant_normalizes_all_artifacts_and_costs() -> None:
    from quant.backtest.artifacts import BacktestArtifacts

    raw = FakeAkquantResult()
    benchmark = pd.Series(
        [0.0, 0.02, -0.01],
        index=raw.equity_curve_daily.index,
        name="benchmark_return",
    )

    artifacts = BacktestArtifacts.from_akquant(raw, benchmark_returns=benchmark)

    assert artifacts.equity_curve.tolist() == [100_000.0, 110_000.0, 99_000.0]
    assert artifacts.returns.tolist() == pytest.approx([0.0, 0.1, -0.1])
    assert list(artifacts.costs.columns) == [
        "timestamp",
        "order_id",
        "symbol",
        "commission",
        "stamp_tax",
        "transfer_fee",
        "slippage_cost",
        "total_cost",
    ]
    assert artifacts.costs["total_cost"].tolist() == [5.0, 15.0]
    assert artifacts.benchmark_returns is not None
    assert artifacts.benchmark_returns.tolist() == [0.0, 0.02, -0.01]
    assert artifacts.metadata == {"initial_cash": 100_000.0}
    assert isinstance(artifacts.metadata, MappingProxyType)


def test_artifacts_copy_mutable_inputs_at_boundary() -> None:
    from quant.backtest.artifacts import BacktestArtifacts

    raw = FakeAkquantResult()
    artifacts = BacktestArtifacts.from_akquant(raw)

    raw.equity_curve_daily.iloc[-1] = 1.0
    raw.positions_df.loc[0, "symbol"] = "MUTATED"
    raw.executions_df.loc[0, "commission"] = 999.0

    assert artifacts.equity_curve.iloc[-1] == 99_000.0
    assert artifacts.positions.loc[0, "symbol"] == "000001.SZ"
    assert artifacts.costs.loc[0, "commission"] == 5.0
    with pytest.raises(TypeError):
        artifacts.metadata["initial_cash"] = 1.0  # type: ignore[index]


def test_cost_ledger_falls_back_to_orders_when_executions_are_empty() -> None:
    from quant.backtest.artifacts import BacktestArtifacts

    raw = FakeAkquantResult()
    raw.executions_df = pd.DataFrame()

    artifacts = BacktestArtifacts.from_akquant(raw)

    assert artifacts.costs["order_id"].tolist() == ["order-1", "order-2"]
    assert artifacts.costs["timestamp"].tolist() == raw.orders_df["created_at"].tolist()
    assert artifacts.costs["total_cost"].sum() == 10.0


def test_artifacts_sort_and_deduplicate_time_series() -> None:
    from quant.backtest.artifacts import BacktestArtifacts

    raw = FakeAkquantResult()
    dates = raw.equity_curve_daily.index
    raw.equity_curve_daily = pd.Series(
        [110_000.0, 100_000.0, 111_000.0],
        index=[dates[1], dates[0], dates[1]],
    )

    artifacts = BacktestArtifacts.from_akquant(raw)

    assert artifacts.equity_curve.index.tolist() == [dates[0], dates[1]]
    assert artifacts.equity_curve.tolist() == [100_000.0, 111_000.0]
    assert artifacts.returns.tolist() == pytest.approx([0.0, 0.11])
