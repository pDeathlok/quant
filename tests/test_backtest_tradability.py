from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quant.backtest import AShareTradabilityPolicy, BacktestEngine


def _tradability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["20260730", "20260731", "20260731"],
            "ts_code": ["000001.SZ", "000001.SZ", "600000.SH"],
            "pre_close": [10.0, 10.0, 8.0],
            "up_limit": [11.0, 11.0, 8.8],
            "down_limit": [9.0, 9.0, 7.2],
            "is_suspended": [False, False, True],
            "is_st": [False, False, False],
            "st_type": ["", "", ""],
            "list_date": ["19910403", "19910403", "19991110"],
            "market": ["主板", "主板", "主板"],
        }
    )


def test_tradability_policy_rejects_suspension_and_pinned_limits() -> None:
    policy = AShareTradabilityPolicy(_tradability_frame())

    assert policy.check_order(
        trade_date="20260731", symbol="000001.SZ", side="buy", price=10.0
    ).allowed
    assert policy.check_order(
        trade_date="20260731", symbol="600000.SH", side="buy", price=8.0
    ).reason == "suspended"
    assert policy.check_order(
        trade_date="20260731", symbol="000001.SZ", side="buy", price=11.0
    ).reason == "buy_at_up_limit"
    assert policy.check_order(
        trade_date="20260731", symbol="000001.SZ", side="sell", price=9.0
    ).reason == "sell_at_down_limit"


def test_tradability_policy_fails_closed_for_missing_or_invalid_orders() -> None:
    policy = AShareTradabilityPolicy(_tradability_frame())

    assert policy.check_order(
        trade_date="20260731", symbol="000002.SZ", side="buy", price=10.0
    ).reason == "missing_tradability_data"
    assert policy.check_order(
        trade_date="20260731", symbol="000001.SZ", side="hold", price=10.0
    ).reason == "invalid_side"
    assert policy.check_order(
        trade_date="20260731", symbol="000001.SZ", side="buy", price=0.0
    ).reason == "invalid_price"


def test_backtest_engine_records_tradability_metadata_and_exposes_gate(monkeypatch) -> None:
    policy = AShareTradabilityPolicy(_tradability_frame())
    monkeypatch.setattr(
        "quant.backtest.engine.aq.run_backtest",
        lambda **kwargs: SimpleNamespace(initial_cash=100_000.0),
    )
    engine = BacktestEngine(
        data=pd.DataFrame({"date": [], "symbol": [], "close": []}),
        strategy=object(),
        tradability_policy=policy,
    )

    engine.run(show_progress=False)

    assert engine.artifacts is not None
    assert engine.artifacts.metadata["tradability"] == {
        "enforcement": "project_order_gate",
        "rows": 3,
        "symbols": 2,
        "start_date": "20260730",
        "end_date": "20260731",
    }
    assert engine.check_order(
        trade_date="20260731", symbol="600000.SH", side="sell", price=8.0
    ).reason == "suspended"
