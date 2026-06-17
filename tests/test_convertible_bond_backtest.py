import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


def test_convertible_bond_backtest_uses_prior_close_signal_and_writes_metrics():
    from quant.strategies.convertible_bond import (
        ConvertibleBondBacktestConfig,
        ConvertibleBondFilterConfig,
        ConvertibleBondRotationConfig,
        backtest_convertible_bond_rotation,
    )

    rows = []
    prices = {
        "110001.SH": [110, 111, 112, 113, 114, 115, 116],
        "123001.SZ": [120, 119, 118, 117, 116, 115, 114],
    }
    dates = ["20260102", "20260105", "20260106", "20260107", "20260108", "20260109", "20260112"]
    for ts_code, closes in prices.items():
        for trade_date, close in zip(dates, closes):
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "close": close,
                    "bond_over_rate": 10.0 if ts_code == "110001.SH" else 20.0,
                    "amount": 5000.0,
                    "pct_chg": 0.0,
                }
            )
    daily = pd.DataFrame(rows)
    basic = pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "list_date": "20200101",
                "delist_date": "",
                "conv_start_date": "20200101",
            },
            {
                "ts_code": "123001.SZ",
                "remain_size": 5.0,
                "newest_rating": "AA",
                "list_date": "20200101",
                "delist_date": "",
                "conv_start_date": "20200101",
            },
        ]
    )
    config = ConvertibleBondBacktestConfig(
        start_date="20260102",
        end_date="20260112",
        min_history_trade_dates=1,
        commission_rate=0.0,
        slippage_rate=0.0,
        selector=ConvertibleBondRotationConfig(
            top_n=1,
            max_position_weight=1.0,
            filter=ConvertibleBondFilterConfig(min_price=100.0, max_price=130.0, min_amount=0.0),
        ),
    )

    result = backtest_convertible_bond_rotation(daily=daily, basic=basic, call=pd.DataFrame(), config=config)

    assert result.summary["trade_days"] == 6
    assert result.summary["total_return"] > 0
    assert not result.targets.empty
    assert set(result.trades["ts_code"]) == {"110001.SH"}
