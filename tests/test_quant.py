import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "date": dates,
        "symbol": "TEST",
        "open": np.random.uniform(90, 110, 100),
        "high": np.random.uniform(100, 120, 100),
        "low": np.random.uniform(80, 100, 100),
        "close": np.random.uniform(95, 105, 100),
        "volume": np.random.uniform(1000000, 5000000, 100),
        "pct_change": np.random.uniform(-0.05, 0.05, 100)
    })
    return data


class TestMarketDataStore:
    def test_parquet_store_roundtrip(self, sample_data, tmp_path):
        from quant.data import MarketDataStore, MarketDataStoreConfig

        store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=tmp_path))
        store.write_frame(sample_data, "daily", "TEST")
        loaded = store.read_frame("daily", "TEST")

        assert len(loaded) == len(sample_data)
        assert list(loaded["symbol"].unique()) == ["TEST"]


class TestTechnicalFactors:
    def test_ma_calculation(self, sample_data):
        from quant.data.factors import MA

        ma5 = MA(window=5)
        result = ma5.compute(sample_data)

        assert len(result) == len(sample_data)
        assert not result.iloc[:4].notna().any()
        assert result.iloc[4:].notna().all()

    def test_rsi_calculation(self, sample_data):
        from quant.data.factors import RSI

        rsi = RSI(window=14)
        result = rsi.compute(sample_data)
        result = result.dropna()

        assert result.min() >= 0
        assert result.max() <= 100

    def test_project_kdj_uses_continuous_price_across_ex_right_gap(self):
        from quant.features.variable_library import calculate_project_extra_features

        rows = [
            ("2026-05-18", 34.50, 34.80, 33.71, 34.00, 34.87, -2.4950),
            ("2026-05-19", 34.28, 34.63, 33.70, 34.26, 34.00, 0.7647),
            ("2026-05-20", 33.92, 35.60, 33.73, 35.41, 34.26, 3.3567),
            ("2026-05-21", 35.33, 35.75, 34.50, 34.70, 35.41, -2.0051),
            ("2026-05-22", 34.95, 36.28, 34.45, 35.69, 34.70, 2.8530),
            ("2026-05-25", 35.76, 36.85, 35.46, 35.67, 35.69, -0.0560),
            ("2026-05-26", 35.79, 35.79, 34.62, 34.94, 35.67, -2.0465),
            ("2026-05-27", 35.26, 35.26, 33.72, 34.10, 34.94, -2.4041),
            ("2026-05-28", 33.81, 33.99, 32.92, 33.22, 34.10, -2.5806),
            ("2026-05-29", 34.02, 34.28, 32.74, 33.33, 33.22, 0.3311),
            ("2026-06-01", 33.49, 34.98, 33.21, 34.70, 33.33, 4.1104),
            ("2026-06-02", 23.14, 23.72, 23.00, 23.40, 23.15, 1.0799),
            ("2026-06-03", 23.39, 23.76, 22.08, 22.10, 23.40, -5.5556),
            ("2026-06-04", 22.30, 22.89, 21.81, 22.21, 22.10, 0.4977),
            ("2026-06-05", 22.21, 23.14, 22.01, 22.52, 22.21, 1.3958),
        ]
        data = pd.DataFrame(
            rows,
            columns=["date", "open", "high", "low", "close", "pre_close", "pct_chg"],
        )
        data["date"] = pd.to_datetime(data["date"])
        data["volume"] = np.linspace(100_000, 150_000, len(data))

        result = calculate_project_extra_features(data)

        assert result["kdj_d_j"].iloc[-1] > 0


class TestStrategies:
    def test_momentum_strategy_init(self):
        from quant.strategies.momentum.momentum import MomentumStrategy

        strategy = MomentumStrategy(
            fast_ma_period=5,
            slow_ma_period=20,
            holding_bars=5
        )

        assert strategy.name == "MomentumStrategy"
        assert strategy.fast_ma_period == 5
        assert strategy.slow_ma_period == 20


class TestBacktestEngine:
    def test_engine_creation(self, sample_data):
        from quant.backtest import BacktestEngine
        from quant.strategies.momentum.momentum import MomentumStrategy

        strategy = MomentumStrategy()
        engine = BacktestEngine(
            data=sample_data,
            strategy=strategy,
            initial_cash=1000000.0
        )

        assert engine.initial_cash == 1000000.0


class TestRiskManager:
    def test_risk_limits_creation(self):
        from quant.risk import RiskLimits, RiskManager

        limits = RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.2
        )

        manager = RiskManager(limits)

        assert manager.limits.max_position_size == 0.1
        assert manager.limits.max_drawdown == 0.2

    def test_order_rate_limit(self):
        from quant.risk import RiskManager

        manager = RiskManager()
        manager.limits.max_orders_per_minute = 3

        for _ in range(3):
            assert manager._check_order_rate() is True

        assert manager._check_order_rate() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
