import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


def test_trend_enhanced_features_are_point_in_time():
    from quant.strategies.convertible_bond import add_trend_enhanced_features

    dates = pd.date_range("2026-01-01", periods=25, freq="B").strftime("%Y%m%d").tolist()
    daily = pd.DataFrame(
        {
            "ts_code": ["110001.SH"] * len(dates),
            "trade_date": dates,
            "close": list(range(100, 125)),
            "pct_chg": [0.0] + [1.0] * (len(dates) - 1),
            "amount": [5000.0] * len(dates),
            "bond_over_rate": [10.0] * len(dates),
        }
    )

    featured = add_trend_enhanced_features(daily)
    day_20 = featured[featured["trade_date"] == dates[19]].iloc[0]

    assert day_20["ma_20"] == sum(range(100, 120)) / 20
    assert day_20["trend_strength"] == 100.0
    assert day_20["six_sword_daily"] >= 4
    assert day_20["consecutive_six_sword"] > 0


def test_trend_enhanced_backtest_selects_strong_trend_bond():
    from quant.strategies.convertible_bond import (
        ConvertibleBondFilterConfig,
        ConvertibleBondTrendEnhancedBacktestConfig,
        ConvertibleBondTrendEnhancedConfig,
        backtest_convertible_bond_trend_enhanced,
    )

    dates = pd.date_range("2026-01-01", periods=40, freq="B").strftime("%Y%m%d").tolist()
    rows = []
    prices = {
        "110001.SH": [100 + index * 1.0 for index in range(len(dates))],
        "123001.SZ": [120 - index * 0.2 for index in range(len(dates))],
    }
    for ts_code, closes in prices.items():
        for index, (trade_date, close) in enumerate(zip(dates, closes)):
            previous = closes[index - 1] if index else close
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "close": close,
                    "pct_chg": 0.0 if index == 0 else (close / previous - 1.0) * 100.0,
                    "bond_over_rate": 10.0,
                    "amount": 5000.0,
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
    config = ConvertibleBondTrendEnhancedBacktestConfig(
        start_date=dates[0],
        end_date=dates[-1],
        min_history_trade_dates=20,
        commission_rate=0.0,
        slippage_rate=0.0,
        selector=ConvertibleBondTrendEnhancedConfig(
            top_n=1,
            max_position_weight=1.0,
            min_5d_return=1.0,
            max_5d_return=20.0,
            min_1d_return=0.0,
            max_1d_return=2.0,
            filter=ConvertibleBondFilterConfig(
                min_price=90.0,
                max_price=160.0,
                max_premium_rate=30.0,
                min_amount=0.0,
                min_credit_rating="AA-",
            ),
        ),
    )

    result = backtest_convertible_bond_trend_enhanced(
        daily=daily,
        basic=basic,
        call=pd.DataFrame(),
        config=config,
    )

    assert result.summary["total_return"] > 0
    assert set(result.targets["ts_code"]) == {"110001.SH"}
