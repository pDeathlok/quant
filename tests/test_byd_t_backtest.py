import pandas as pd
import pytest

from quant.research.byd_t_backtest import (
    BydTBacktestConfig,
    ChinaAStockFees,
    InventoryLedger,
    prepare_byd_intraday_bars,
    sideways_regime_mask,
)


def test_a_share_buy_does_not_increase_same_day_sellable_inventory() -> None:
    ledger = InventoryLedger(position=10000)
    ledger.start_session("2026-07-17")

    ledger.buy(500)

    assert ledger.position == 10500
    assert ledger.sellable == 10000
    ledger.sell(10000)
    with pytest.raises(ValueError, match=r"T\+1 sellability violation"):
        ledger.sell(100)

    ledger.start_session("2026-07-20")
    assert ledger.sellable == 500


def test_stamp_tax_is_charged_only_on_sell_orders() -> None:
    fees = ChinaAStockFees(
        commission_rate=0.00025,
        minimum_commission=5.0,
        stamp_tax_sell_rate=0.0005,
        transfer_fee_rate=0.00001,
        slippage_rate=0,
    )

    buy_fee = fees.order_fee(100, 500, "BUY")
    sell_fee = fees.order_fee(100, 500, "SELL")

    assert sell_fee - buy_fee == pytest.approx(25.0)


def test_intraday_features_do_not_change_when_future_bars_are_appended() -> None:
    first = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-07-17 09:35", "2026-07-17 09:40", "2026-07-17 09:45"]
            ),
            "open": [100.0, 99.8, 99.6],
            "high": [100.2, 100.0, 99.9],
            "low": [99.7, 99.5, 99.4],
            "close": [99.8, 99.6, 99.8],
            "volume": [1000, 1200, 1100],
        }
    )
    future = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-07-17 09:50"]),
            "open": [99.8],
            "high": [120.0],
            "low": [80.0],
            "close": [110.0],
            "volume": [100000],
        }
    )

    before = prepare_byd_intraday_bars(first)
    after = prepare_byd_intraday_bars(pd.concat([first, future], ignore_index=True)).iloc[:3]

    pd.testing.assert_series_equal(before["vwap"], after["vwap"], check_names=False)
    pd.testing.assert_series_equal(
        before["deviation_vwap"], after["deviation_vwap"], check_names=False
    )


def test_daily_regime_features_are_lagged_and_ignore_future_session() -> None:
    dates = pd.bdate_range("2025-01-02", periods=66)
    close = pd.Series([100 + (index % 5) * 0.2 for index in range(66)])
    bars = pd.DataFrame(
        {
            "datetime": dates + pd.Timedelta(hours=10),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
    )
    before = prepare_byd_intraday_bars(bars.iloc[:65])
    altered_future = bars.copy()
    altered_future.loc[65, ["open", "high", "low", "close"]] = [200, 220, 180, 210]
    after = prepare_byd_intraday_bars(altered_future).iloc[:65]

    regime_columns = [
        "prior_return_60",
        "prior_ma20_slope_5",
        "prior_ma20_ma60_gap",
        "prior_high_60",
        "prior_low_60",
        "prior_range_width_60",
    ]
    pd.testing.assert_frame_equal(
        before[regime_columns].reset_index(drop=True),
        after[regime_columns].reset_index(drop=True),
    )


def test_sideways_regime_rejects_strong_uptrend() -> None:
    config = BydTBacktestConfig(require_sideways_regime=True)
    bars = pd.DataFrame(
        {
            "prior_return_60": [0.04, 0.30],
            "prior_ma20_slope_5": [0.005, 0.04],
            "prior_ma20_ma60_gap": [0.01, 0.15],
            "prior_range_width_60": [0.18, 0.45],
            "prior_atr_pct": [0.025, 0.025],
        }
    )

    assert sideways_regime_mask(bars, config).tolist() == [True, False]
