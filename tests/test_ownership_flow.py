import numpy as np
import pandas as pd

from quant.features.ownership_flow import etf_flow_features, holder_count_features, transaction_features


def test_holder_revision_keeps_latest_period_and_uses_announcement_strictly():
    raw = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 4,
        "ann_date": ["20240103", "20240108", "20240109", "20240111"],
        "end_date": ["20231228", "20231229", "20231228", "20231229"],
        "holder_num": [100, 120, 110, 115],
    })
    keys = pd.DataFrame({
        "symbol": ["000001.SZ"] * 4,
        "date": pd.to_datetime(["20240108", "20240109", "20240110", "20240112"]),
    })
    out = holder_count_features(keys, raw)
    np.testing.assert_allclose(out.own_holder_change_count, [np.nan, 20, 10, 5], equal_nan=True)
    np.testing.assert_allclose(out.own_holder_change_pct, [np.nan, .2, 120 / 110 - 1, 115 / 110 - 1], equal_nan=True)


def test_disclosed_trades_are_signed_and_visible_next_exchange_session():
    calendar = pd.bdate_range("2024-01-01", periods=10)
    keys = pd.DataFrame({"symbol": ["000001.SZ"] * 10, "date": calendar})
    raw = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 2, "ann_date": ["20240105", "20240109"],
        "in_de": ["IN", "DE"], "change_ratio": [1.5, -.5], "change_vol": [400, -100],
    })
    out = transaction_features(keys, raw, calendar).set_index("date")
    assert out.loc["2024-01-05", "own_events_count_5"] == 0
    assert out.loc["2024-01-08", "own_events_net_pct_5"] == 1.5
    assert out.loc["2024-01-10", "own_events_net_pct_5"] == 1
    assert out.loc["2024-01-10", "own_events_net_shares_5"] == 300
    assert out.loc["2024-01-10", "own_events_buy_share_5"] == .5


def test_etf_flow_masks_adjustment_and_waits_until_data_can_be_known():
    calendar = pd.bdate_range("2024-01-01", periods=8)
    code = "510300.SH"
    df = pd.DataFrame({"ts_code": code, "trade_date": calendar})
    shares = df.assign(fd_share=[100, 120, 120, 240, 240, 240, 240, 240])
    nav = df.rename(columns={"trade_date": "nav_date"}).assign(unit_nav=4.0, ann_date=calendar)
    nav.loc[1, "ann_date"] = calendar[3]
    bars = df.assign(amount=100)
    adjustments = df.assign(adj_factor=[1, 1, 1, 2, 2, 2, 2, 2])
    listings = pd.DataFrame({"ts_code": [code], "list_date": ["20200101"]})
    out, detail = etf_flow_features(shares, nav, bars, adjustments, listings, {"core": (code,)}, calendar)
    assert detail.loc[1, "flow"] == 800_000
    assert detail.loc[3, "adjustment_day"]
    assert np.isnan(detail.loc[3, "flow"])
    assert not out.loc[:3, "etf_all_flow_yuan_1"].eq(800_000).any()
    assert detail.loc[1, "ready_session"] == 4


def test_concentration_uses_matching_denominators_and_requires_ten_distinct_holders():
    from quant.features.ownership_flow import concentration_features

    keys = pd.DataFrame({
        "symbol": ["000001.SZ"] * 2, "date": pd.to_datetime(["20240425", "20240426"]),
    })
    raw = pd.DataFrame({
        "ts_code": ["000001.SZ"] * 10, "ann_date": ["20240425"] * 10,
        "end_date": ["20240331"] * 10, "holder_name": [str(i) for i in range(10)],
        "hold_amount": np.arange(10, 0, -1), "hold_ratio": np.arange(10, 0, -1) / 2,
        "hold_float_ratio": np.arange(10, 0, -1),
    })
    top = concentration_features(keys, raw, "top")
    floating = concentration_features(keys, raw, "float")
    assert np.isnan(top.loc[0, "own_top_top10_pct"])
    assert top.loc[1, "own_top_top10_pct"] == 27.5
    assert floating.loc[1, "own_float_top10_pct"] == 55
    assert floating.loc[1, "own_float_num_ge5"] == 6
    assert concentration_features(keys, raw.iloc[:9], "top").own_top_top10_pct.isna().all()


def test_etf_turnover_never_uses_signal_sessions_final_amount():
    calendar = pd.bdate_range("2024-01-01", periods=26)
    code = "510300.SH"
    raw = pd.DataFrame({"ts_code": code, "trade_date": calendar})
    nav = raw.rename(columns={"trade_date": "nav_date"}).assign(unit_nav=1.0, ann_date=calendar)
    bars = raw.assign(amount=100)
    bars.loc[23, "amount"] = 200
    listings = pd.DataFrame({"ts_code": [code], "list_date": ["20200101"]})
    features, _ = etf_flow_features(
        raw.assign(fd_share=100), nav, bars, raw.assign(adj_factor=1),
        listings, {"core": (code,)}, calendar,
    )
    assert features.loc[23, "etf_all_turnover_multiple_20"] == 1
    assert features.loc[24, "etf_all_turnover_multiple_20"] == 2
