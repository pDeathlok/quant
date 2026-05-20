import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


class TestDataFetcher:
    def test_fetcher_cache(self, sample_data):
        from data import DataFetcher

        fetcher = DataFetcher(cache_dir="./data/test_cache")
        assert fetcher.cache_dir.exists()

    def test_data_normalize(self, sample_data):
        from data import DataFetcher

        fetcher = DataFetcher()
        normalized = fetcher._normalize(sample_data.copy(), "TEST")

        assert "symbol" in normalized.columns
        assert normalized["symbol"].iloc[0] == "TEST"


class TestTechnicalFactors:
    def test_ma_calculation(self, sample_data):
        from data.factors import MA

        ma5 = MA(period=5)
        result = ma5.compute(sample_data)

        assert len(result) == len(sample_data)
        assert not result.iloc[:4].notna().any()
        assert result.iloc[4:].notna().all()

    def test_rsi_calculation(self, sample_data):
        from data.factors import RSI

        rsi = RSI(period=14)
        result = rsi.compute(sample_data)

        assert result.min() >= 0
        assert result.max() <= 100


class TestStrategies:
    def test_momentum_strategy_init(self):
        from strategies.examples import MomentumStrategy

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
        from backtest import BacktestEngine
        from strategies.examples import MomentumStrategy

        strategy = MomentumStrategy()
        engine = BacktestEngine(
            data=sample_data,
            strategy=strategy,
            initial_cash=1000000.0
        )

        assert engine.initial_cash == 1000000.0


class TestRiskManager:
    def test_risk_limits_creation(self):
        from risk import RiskLimits, RiskManager

        limits = RiskLimits(
            max_position_size=0.1,
            max_drawdown=0.2
        )

        manager = RiskManager(limits)

        assert manager.limits.max_position_size == 0.1
        assert manager.limits.max_drawdown == 0.2

    def test_order_rate_limit(self):
        from risk import RiskManager

        manager = RiskManager()
        manager.limits.max_orders_per_minute = 3

        for _ in range(3):
            assert manager._check_order_rate() is True

        assert manager._check_order_rate() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
