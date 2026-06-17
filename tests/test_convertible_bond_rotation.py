import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


def test_convertible_bond_selector_filters_call_risk_and_scores_double_low():
    from quant.strategies.convertible_bond import (
        ConvertibleBondFilterConfig,
        ConvertibleBondRotationConfig,
        ConvertibleBondSelector,
    )

    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 112.0,
                "bond_over_rate": 12.0,
                "amount": 5000.0,
                "pct_chg": 0.5,
            },
            {
                "ts_code": "123001.SZ",
                "trade_date": "20260616",
                "close": 118.0,
                "bond_over_rate": 20.0,
                "amount": 8000.0,
                "pct_chg": 2.0,
            },
            {
                "ts_code": "113001.SH",
                "trade_date": "20260616",
                "close": 128.0,
                "bond_over_rate": 30.0,
                "amount": 9000.0,
                "pct_chg": 1.0,
            },
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "bond_short_name": "低价转债",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20250101",
            },
            {
                "ts_code": "123001.SZ",
                "bond_short_name": "强赎转债",
                "remain_size": 8.0,
                "newest_rating": "AA+",
                "conv_start_date": "20250101",
            },
            {
                "ts_code": "113001.SH",
                "bond_short_name": "偏贵转债",
                "remain_size": 10.0,
                "newest_rating": "AAA",
                "conv_start_date": "20250101",
            },
        ]
    )
    call = pd.DataFrame([{"ts_code": "123001.SZ", "is_call": "公告强赎"}])
    config = ConvertibleBondRotationConfig(
        top_n=2,
        filter=ConvertibleBondFilterConfig(max_price=130.0, max_premium_rate=35.0),
    )

    selected = ConvertibleBondSelector(config).select(daily=daily, basic=basic, call=call)

    assert list(selected["ts_code"]) == ["110001.SH", "113001.SH"]
    assert selected["double_low"].iloc[0] == 124.0
    assert selected["rank"].tolist() == [1, 2]


def test_convertible_bond_rebalance_orders_emit_buy_and_sell():
    from quant.strategies.convertible_bond import ConvertibleBondSelector

    daily = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "trade_date": "20260616",
                "close": 112.0,
                "bond_over_rate": 12.0,
                "amount": 5000.0,
                "pct_chg": 0.5,
            }
        ]
    )
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "conv_start_date": "20250101",
            }
        ]
    )

    orders = ConvertibleBondSelector().rebalance_orders(
        current_weights={"113999.SH": 0.10},
        daily=daily,
        basic=basic,
    )

    assert {order.ts_code: order.action for order in orders} == {
        "110001.SH": "BUY",
        "113999.SH": "SELL",
    }
