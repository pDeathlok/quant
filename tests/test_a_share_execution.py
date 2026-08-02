from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest


def test_default_a_share_execution_config_matches_documented_policy() -> None:
    from quant.backtest import AShareExecutionConfig

    config = AShareExecutionConfig()

    assert config.to_dict() == {
        "commission_rate": 0.0003,
        "stamp_tax_rate": 0.0005,
        "transfer_fee_rate": 0.00001,
        "min_commission": 5.0,
        "slippage": 0.0,
        "volume_limit_pct": 0.10,
        "lot_size": 100,
        "t_plus_one": True,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("commission_rate", -0.001, "commission_rate must be non-negative"),
        ("stamp_tax_rate", -0.001, "stamp_tax_rate must be non-negative"),
        ("transfer_fee_rate", -0.001, "transfer_fee_rate must be non-negative"),
        ("min_commission", -1.0, "min_commission must be non-negative"),
        ("slippage", -0.001, "slippage must be non-negative"),
        ("volume_limit_pct", 0.0, "volume_limit_pct must be in (0, 1]"),
        ("volume_limit_pct", 1.01, "volume_limit_pct must be in (0, 1]"),
        ("lot_size", 0, "lot_size must be positive"),
    ],
)
def test_a_share_execution_config_rejects_invalid_values(
    field: str,
    value: float,
    message: str,
) -> None:
    from quant.backtest import AShareExecutionConfig

    with pytest.raises(ValueError) as exc_info:
        AShareExecutionConfig(**{field: value})
    assert str(exc_info.value) == message


def test_engine_passes_execution_policy_to_akquant_and_artifact_metadata(
    monkeypatch,
) -> None:
    from quant.backtest import AShareExecutionConfig, BacktestEngine

    captured: dict[str, object] = {}
    raw = SimpleNamespace(initial_cash=100_000.0)

    def fake_run_backtest(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return raw

    monkeypatch.setattr("quant.backtest.engine.aq.run_backtest", fake_run_backtest)
    config = AShareExecutionConfig(
        commission_rate=0.0002,
        min_commission=3.0,
        volume_limit_pct=0.08,
    )
    engine = BacktestEngine(
        data=pd.DataFrame({"date": [], "symbol": [], "close": []}),
        strategy=object(),
        commission_rate=0.009,
        slippage=0.009,
        execution_config=config,
    )

    engine.run(show_progress=False)

    for key, value in config.to_dict().items():
        assert captured[key] == value
    assert engine.artifacts is not None
    assert engine.artifacts.metadata["execution_policy"] == config.to_metadata()


def test_engine_without_execution_policy_keeps_legacy_fee_arguments(monkeypatch) -> None:
    from quant.backtest import BacktestEngine

    captured: dict[str, object] = {}

    def fake_run_backtest(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(initial_cash=100_000.0)

    monkeypatch.setattr("quant.backtest.engine.aq.run_backtest", fake_run_backtest)
    engine = BacktestEngine(
        data=pd.DataFrame({"date": [], "symbol": [], "close": []}),
        strategy=object(),
        commission_rate=0.0007,
        slippage=0.0004,
    )

    engine.run(show_progress=False)

    assert captured["commission_rate"] == 0.0007
    assert captured["slippage"] == 0.0004
    assert "stamp_tax_rate" not in captured
    assert "t_plus_one" not in captured
    assert engine.artifacts is not None
    assert "execution_policy" not in engine.artifacts.metadata
