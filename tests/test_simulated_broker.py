import json

import pandas as pd
import pytest

from quant.backtest import AShareExecutionConfig, AShareTradabilityPolicy
from quant.trading import Order, OrderManager, OrderSide, SimulatedBroker, reconcile_account


def test_simulated_broker_applies_a_share_costs_t_plus_one_and_mark_prices() -> None:
    broker = SimulatedBroker(initial_cash=100_000)
    buy = Order("000001.SZ", OrderSide.BUY, 100, price=10.0)
    broker.send_order(buy)
    fill = broker.simulate_fill(buy.order_id, 10.0, trade_date="20260105")

    assert fill.commission == pytest.approx(5.0)
    assert fill.transfer_fee == pytest.approx(0.01)
    assert fill.stamp_tax == 0.0
    assert fill.total_cost == pytest.approx(5.01)
    assert broker.get_account()["cash"] == pytest.approx(98_994.99)

    same_day_sell = Order("000001.SZ", OrderSide.SELL, 100, price=10.0)
    broker.send_order(same_day_sell)
    with pytest.raises(ValueError, match=r"T\+1"):
        broker.simulate_fill(same_day_sell.order_id, 10.0, trade_date="20260105")
    assert same_day_sell.order_id in {order.order_id for order in broker.get_pending_orders()}

    broker.update_market_prices({"000001.SZ": 12.0})
    assert broker.get_account()["positions_value"] == pytest.approx(1_200.0)

    fill = broker.simulate_fill(same_day_sell.order_id, 10.0, trade_date="20260106")
    assert fill.stamp_tax == pytest.approx(0.5)
    assert fill.total_cost == pytest.approx(5.51)
    assert broker.get_positions() == {}


def test_simulated_broker_enforces_round_lot_volume_and_tradability() -> None:
    trade_date = "20260105"
    policy = AShareTradabilityPolicy(
        pd.DataFrame(
            {
                "trade_date": [trade_date],
                "ts_code": ["000001.SZ"],
                "pre_close": [10.0],
                "up_limit": [11.0],
                "down_limit": [9.0],
                "is_suspended": [False],
                "is_st": [False],
                "st_type": [None],
                "list_date": ["19910403"],
                "market": ["主板"],
            }
        )
    )
    broker = SimulatedBroker(tradability_policy=policy)

    odd_lot = Order("000001.SZ", OrderSide.BUY, 50, price=10.0)
    broker.send_order(odd_lot)
    with pytest.raises(ValueError, match="lot size"):
        broker.simulate_fill(odd_lot.order_id, 10.0, trade_date=trade_date)

    limit_up = Order("000001.SZ", OrderSide.BUY, 100, price=11.0)
    broker.send_order(limit_up)
    with pytest.raises(ValueError, match="buy_at_up_limit"):
        broker.simulate_fill(limit_up.order_id, 11.0, trade_date=trade_date)

    too_large = Order("000001.SZ", OrderSide.BUY, 200, price=10.0)
    broker.send_order(too_large)
    with pytest.raises(ValueError, match="volume participation"):
        broker.simulate_fill(
            too_large.order_id,
            10.0,
            trade_date=trade_date,
            market_volume=1_000,
        )


def test_simulated_broker_persists_and_restores_state(tmp_path) -> None:
    state_path = tmp_path / "paper/state.json"
    broker = SimulatedBroker(initial_cash=50_000, state_path=state_path)
    order = Order("600000.SH", OrderSide.BUY, 100, price=8.0)
    broker.send_order(order)
    broker.simulate_fill(order.order_id, 8.0, trade_date="20260105")

    restored = SimulatedBroker(initial_cash=1.0, state_path=state_path)

    assert restored.get_positions() == {"600000.SH": 100}
    assert restored.get_account()["cash"] == pytest.approx(broker.get_account()["cash"])
    assert json.loads(state_path.read_text())["schema_version"] == "simulated-broker/v1"


def test_reconciliation_reports_cash_and_position_breaks() -> None:
    report = reconcile_account(
        expected_positions={"A": 100, "B": 200},
        actual_positions={"A": 90, "C": 10},
        expected_cash=1_000.0,
        actual_cash=999.0,
        cash_tolerance=0.01,
    )

    assert report.balanced is False
    assert report.cash_difference == pytest.approx(-1.0)
    assert report.position_differences == {"A": -10, "B": -200, "C": 10}


def test_order_manager_checks_tradability_before_sending() -> None:
    policy = AShareTradabilityPolicy(
        pd.DataFrame(
            {
                "trade_date": ["20260105"],
                "ts_code": ["000001.SZ"],
                "pre_close": [10.0],
                "up_limit": [11.0],
                "down_limit": [9.0],
                "is_suspended": [True],
                "is_st": [False],
                "st_type": [None],
                "list_date": ["19910403"],
                "market": ["主板"],
            }
        )
    )
    manager = OrderManager(SimulatedBroker(), tradability_policy=policy)

    order_id = manager.place_order(
        "000001.SZ",
        OrderSide.BUY,
        100,
        10.0,
        trade_date="20260105",
    )

    assert order_id is None
    assert manager.last_rejection_reason == "suspended"
    assert manager.broker.get_pending_orders() == []
