from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features.long_external_factors import (
    build_holder_weekly,
    build_margin_weekly,
    build_moneyflow_weekly,
    build_pledge_weekly,
    build_top_list_weekly,
)


def test_moneyflow_and_margin_use_only_history_through_signal_date() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    signals = pd.DataFrame({"date": [dates[-1]], "ts_code": ["000001.SZ"]})
    moneyflow = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "buy_sm_amount": 10.0,
            "sell_sm_amount": 12.0,
            "buy_md_amount": 10.0,
            "sell_md_amount": 10.0,
            "buy_lg_amount": [10, 10, 10, 20, 20],
            "sell_lg_amount": 5.0,
            "buy_elg_amount": 5.0,
            "sell_elg_amount": 0.0,
            "net_mf_amount": 8.0,
        }
    )
    money = build_moneyflow_weekly(signals, moneyflow)
    assert money.loc[0, "large_flow_persistence_5d"] == 1.0
    assert money.loc[0, "large_net_5d_ratio"] > 0

    margin = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "rzye": [100, 101, 102, 103, 104],
            "rqye": [10, 11, 12, 13, 14],
            "rzmre": [20, 20, 20, 20, 20],
            "rzche": [10, 10, 10, 10, 10],
        }
    )
    result = build_margin_weekly(signals, margin)
    assert result.loc[0, "margin_balance"] == 104
    assert np.isclose(result.loc[0, "margin_buy_ratio_5d"], 1 / 3)


def test_event_and_pledge_windows_do_not_see_future_rows() -> None:
    signal_date = pd.Timestamp("2024-06-28")
    signals = pd.DataFrame(
        {"date": [signal_date], "ts_code": ["000001.SZ"], "close": [12.0]}
    )
    top = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240620", "20240701"],
            "net_amount": [10.0, 1000.0],
            "amount": [100.0, 1000.0],
            "reason": ["A", "B"],
        }
    )
    top_result = build_top_list_weekly(
        signals[["date", "ts_code"]],
        top,
        pd.bdate_range("2024-04-01", "2024-07-05"),
    )
    assert top_result.loc[0, "top_list_count_20d"] == 1
    assert np.isclose(top_result.loc[0, "top_list_net_ratio_20d"], 0.1)

    holder = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ann_date": ["20240601", "20240701"],
            "change_ratio": [1.5, 99.0],
            "change_vol": [10.0, 10.0],
            "avg_price": [10.0, 10.0],
            "in_de": ["IN", "DE"],
        }
    )
    holder_result = build_holder_weekly(signals, holder)
    assert holder_result.loc[0, "holder_net_change_ratio_30d"] == 1.5
    assert np.isclose(holder_result.loc[0, "holder_avg_price_gap"], 0.2)

    pledge = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "end_date": ["20230331", "20240329", "20240621", "20240705"],
            "pledge_ratio": [20.0, 18.0, 15.0, 99.0],
            "pledge_count": [5, 5, 4, 20],
        }
    )
    pledge_result = build_pledge_weekly(signals[["date", "ts_code"]], pledge)
    assert pledge_result.loc[0, "pledge_ratio"] == 15.0
    assert pledge_result.loc[0, "pledge_ratio_change_13w"] == -3.0
    assert pledge_result.loc[0, "pledge_ratio_change_52w"] == -5.0
