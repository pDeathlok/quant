from __future__ import annotations

import pandas as pd
import pytest

from quant.data.tradability import build_daily_tradability


def test_build_daily_tradability_joins_limits_suspensions_and_st_status() -> None:
    stock_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "300001.SZ", "600000.SH"],
            "list_date": ["19910403", "20091030", "19991110"],
            "market": ["主板", "创业板", "主板"],
        }
    )
    limits = pd.DataFrame(
        {
            "trade_date": ["20260731", "20260731", "20260731"],
            "ts_code": ["000001.SZ", "300001.SZ", "600000.SH"],
            "pre_close": [10.0, 20.0, 8.0],
            "up_limit": [11.0, 24.0, 8.4],
            "down_limit": [9.0, 16.0, 7.6],
        }
    )
    suspensions = pd.DataFrame(
        {
            "trade_date": ["20260731"],
            "ts_code": ["600000.SH"],
            "suspend_type": ["S"],
        }
    )
    st_stocks = pd.DataFrame(
        {
            "trade_date": ["20260731"],
            "ts_code": ["600000.SH"],
            "type": ["ST"],
            "type_name": ["风险警示板"],
        }
    )

    frame, audit = build_daily_tradability(
        trade_date="20260731",
        stock_basic=stock_basic,
        limits=limits,
        suspensions=suspensions,
        st_stocks=st_stocks,
    )

    assert frame["ts_code"].tolist() == ["000001.SZ", "300001.SZ", "600000.SH"]
    suspended = frame.set_index("ts_code").loc["600000.SH"]
    assert bool(suspended["is_suspended"]) is True
    assert bool(suspended["is_st"]) is True
    assert suspended["st_type"] == "风险警示板"
    assert audit == {
        "trade_date": "20260731",
        "universe_rows": 3,
        "limit_rows": 3,
        "covered_rows": 3,
        "coverage_rate": 1.0,
        "suspended_rows": 1,
        "st_rows": 1,
    }


def test_build_daily_tradability_rejects_low_limit_coverage() -> None:
    stock_basic = pd.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ"], "list_date": ["19910403", "19910129"]}
    )
    limits = pd.DataFrame(
        {
            "trade_date": ["20260731"],
            "ts_code": ["000001.SZ"],
            "up_limit": [11.0],
            "down_limit": [9.0],
        }
    )

    with pytest.raises(ValueError, match="coverage 50.00% is below required 98.00%"):
        build_daily_tradability(
            trade_date="20260731",
            stock_basic=stock_basic,
            limits=limits,
            suspensions=pd.DataFrame(),
            st_stocks=pd.DataFrame(),
        )


def test_build_daily_tradability_deduplicates_provider_rows() -> None:
    stock_basic = pd.DataFrame({"ts_code": ["000001.SZ"]})
    limits = pd.DataFrame(
        {
            "trade_date": ["20260731", "20260731"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "up_limit": [10.9, 11.0],
            "down_limit": [9.0, 9.0],
        }
    )

    frame, audit = build_daily_tradability(
        trade_date="20260731",
        stock_basic=stock_basic,
        limits=limits,
        suspensions=pd.DataFrame(),
        st_stocks=pd.DataFrame(),
    )

    assert len(frame) == 1
    assert frame.loc[0, "up_limit"] == 11.0
    assert audit["coverage_rate"] == 1.0
