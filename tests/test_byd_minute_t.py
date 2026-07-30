import numpy as np
import pandas as pd

from quant.strategies.custom.byd_minute_t import (
    BydHolding,
    build_minute_payload,
    daily_sideways_snapshot,
    weighted_price,
)


def daily_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=100)
    close = 100 + np.sin(np.arange(100) / 8)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        }
    )


def test_weighted_price_combines_multiple_same_direction_fills() -> None:
    assert weighted_price([(500, 90.0), (1000, 93.0)]) == 92.0
    assert weighted_price([(0, None), (0, None)]) is None


def test_only_validated_positive_t_entry_is_enabled() -> None:
    payload = build_minute_payload(
        daily_frame(),
        pd.DataFrame(),
        holding=BydHolding(shares=10000, cost=110.6061, full_shares=10000),
    )

    assert payload["validation"]["execution_enabled"] is True
    assert payload["primary_action"]["action"] == "WAIT_STRICT_POSITIVE_T"
    disabled_opening_kinds = {
        "planned_reverse_t",
        "intraday_rebound_t",
        "micro_reverse_t",
        "reverse_t",
        "inventory_recovery",
    }
    disabled_alerts = [
        item for item in payload["alerts"] if item["kind"] in disabled_opening_kinds
    ]
    assert disabled_alerts
    assert all(item["execution_enabled"] is False for item in disabled_alerts)
    positive_alert = next(item for item in payload["alerts"] if item["kind"] == "positive_t")
    assert positive_alert["execution_enabled"] is True
    assert positive_alert["triggered"] is False
    assert payload["validated_positive_t"]["available"] is False


def test_existing_positive_t_lot_can_still_trigger_a_profitable_exit() -> None:
    payload = build_minute_payload(
        daily_frame(),
        pd.DataFrame(),
        holding=BydHolding(shares=10500, cost=108.0, full_shares=10000),
        bought_today_shares=500,
        bought_today_price=90.0,
    )

    exit_alert = next(item for item in payload["alerts"] if item["kind"] == "positive_t_exit")
    assert exit_alert["execution_enabled"] is True
    assert exit_alert["triggered"] is True
    assert payload["primary_action"]["kind"] == "positive_t_exit"


def test_existing_positive_t_lot_has_validated_stop_loss() -> None:
    payload = build_minute_payload(
        daily_frame(),
        pd.DataFrame(),
        holding=BydHolding(shares=10500, cost=108.0, full_shares=10000),
        open_positive_shares=500,
        open_positive_price=103.0,
    )

    stop = next(item for item in payload["alerts"] if item["kind"] == "positive_t_stop")
    assert stop["execution_enabled"] is True
    assert stop["triggered"] is True
    assert stop["price_line"] == 100.94


def test_daily_sideways_gate_excludes_strong_trend() -> None:
    sideways = daily_frame()
    sideways["close"] = 100 + np.sin(np.arange(100) / 5) * 5
    sideways["open"] = sideways["close"] - 0.1
    sideways["high"] = sideways["close"] + 1
    sideways["low"] = sideways["close"] - 1
    strong_trend = sideways.copy()
    strong_trend["close"] = np.linspace(50, 120, len(strong_trend))
    strong_trend["open"] = strong_trend["close"] - 0.1
    strong_trend["high"] = strong_trend["close"] + 1
    strong_trend["low"] = strong_trend["close"] - 1

    assert daily_sideways_snapshot(sideways)["positive_t_allowed"] is True
    assert daily_sideways_snapshot(strong_trend)["positive_t_allowed"] is False
