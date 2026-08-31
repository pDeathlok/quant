from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.active_market_value import (
    ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS,
    KEY_INDEX_SPECS,
    attach_active_market_value_features,
    build_active_market_value_feature_frames,
    compute_active_share_ratio,
    compute_key_index_active_market_value_features,
    compute_market_active_market_value_features,
    compute_stock_active_market_value_features,
)
from quant.features.factor_execution import build_factor_execution_plan
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
)


def _daily_basic(rows: int = 26) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-01", periods=rows)
    frames = []
    for symbol, scale in (("000001.SZ", 1.0), ("600000.SH", 2.0)):
        position = np.arange(rows, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": symbol,
                    "trade_date": dates.strftime("%Y%m%d"),
                    "turnover_rate": 5.0 + position,
                    "circ_mv": scale * (1_000.0 + position * 10.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _market_daily(rows: int = 26) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-01", periods=rows)
    frames = []
    for symbol, scale in (("000001.SZ", 1.0), ("600000.SH", 2.0)):
        position = np.arange(rows, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": symbol,
                    "trade_date": dates.strftime("%Y%m%d"),
                    "volume": scale * (100.0 + position),
                    "amount": scale * (1_000.0 + position * 10.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _index_daily(rows: int = 26) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-01", periods=rows)
    position = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "ts_code": "000300.SH",
            "trade_date": dates.strftime("%Y%m%d"),
            "close": 4_000.0 + position * 20.0,
            "vol": 1_000_000.0 + position * 10_000.0,
            "amount": 2_000_000.0 + position * 20_000.0,
        }
    )


def _index_daily_basic(rows: int = 26) -> pd.DataFrame:
    dates = pd.bdate_range("2026-07-01", periods=rows)
    position = np.arange(rows, dtype=float)
    # Tushare index_dailybasic commonly exposes CSI 300 as 399300.SZ.
    return pd.DataFrame(
        {
            "ts_code": "399300.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "float_mv": 30_000_000_000_000.0 + position * 10_000_000_000.0,
            "turnover_rate": 0.5 + position * 0.01,
        }
    )


def _index_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": ["000852.SH", "000852.SH"],
            "con_code": ["000001.SZ", "600000.SH"],
            "trade_date": ["20260701", "20260701"],
            "weight": [40.0, 60.0],
        }
    )


def test_active_share_ratio_uses_causal_turnover_recursion() -> None:
    turnover = pd.Series([10.0, 20.0, np.nan, 0.0])

    result = compute_active_share_ratio(turnover)

    expected_second = 0.20 + (12.0 / 13.0) * (1.0 - 0.20) * 0.10
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(expected_second)
    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == pytest.approx((12.0 / 13.0) * expected_second)


def test_stock_active_market_value_is_cny_and_prefix_causal() -> None:
    daily_basic = _daily_basic()

    full = compute_stock_active_market_value_features(daily_basic)
    prefix_source = daily_basic.groupby("ts_code", group_keys=False).head(20)
    prefix = compute_stock_active_market_value_features(prefix_source)

    first = full[full["ts_code"].eq("000001.SZ")].iloc[0]
    assert first["stock_active_share_ratio_13d_proxy"] == pytest.approx(0.05)
    assert first["stock_active_mv_proxy_cny"] == pytest.approx(500_000.0)
    comparable = full.merge(
        prefix,
        on=["ts_code", "trade_date"],
        suffixes=("_full", "_prefix"),
        validate="one_to_one",
    )
    for column in (
        "stock_active_share_ratio_13d_proxy",
        "stock_active_mv_proxy_cny",
        "stock_active_mv_return_1d_pct",
        "stock_active_mv_return_5d_pct",
    ):
        np.testing.assert_allclose(
            comparable[f"{column}_full"],
            comparable[f"{column}_prefix"],
            equal_nan=True,
        )


def test_market_active_value_aggregates_stocks_and_excludes_today_from_volume_baseline() -> None:
    daily_basic = _daily_basic()
    market_daily = _market_daily()

    result = compute_market_active_market_value_features(
        daily_basic,
        market_daily=market_daily,
    )

    first_stock = compute_stock_active_market_value_features(daily_basic)
    expected_first = first_stock[first_stock["trade_date"].eq("20260701")][
        "stock_active_mv_proxy_cny"
    ].sum()
    assert result.iloc[0]["market_active_mv_proxy_cny"] == pytest.approx(
        expected_first
    )
    assert result.iloc[0]["market_active_mv_ratio_proxy"] == pytest.approx(0.05)
    day_21 = result.iloc[20]
    current_volume = 3.0 * 120.0
    previous_20_mean = np.mean(3.0 * (100.0 + np.arange(20, dtype=float)))
    assert day_21["market_volume_ratio_prev20"] == pytest.approx(
        current_volume / previous_20_mean
    )


def test_key_index_features_align_tushare_alias_and_keep_missing_indices_explicit() -> None:
    result = compute_key_index_active_market_value_features(
        _index_daily(),
        _index_daily_basic(),
    )

    last = result.iloc[-1]
    assert last["index_csi300_return_1d_pct"] > 0
    assert last["index_csi300_return_5d_pct"] > last["index_csi300_return_1d_pct"]
    assert last["index_csi300_volume_ratio_prev20"] > 0
    assert last["index_csi300_active_mv_proxy_cny"] > 0
    assert 0 < last["index_csi300_active_share_ratio_13d_proxy"] <= 1
    assert np.isnan(last["index_csi1000_active_mv_proxy_cny"])
    assert {spec.slug for spec in KEY_INDEX_SPECS} >= {
        "sse_composite",
        "szse_component",
        "chinext",
        "star50",
        "csi300",
        "csi500",
        "csi1000",
        "bse50",
    }


def test_missing_index_dailybasic_uses_point_in_time_constituent_fallback() -> None:
    csi1000_daily = _index_daily().assign(ts_code="000852.SH")

    result = compute_key_index_active_market_value_features(
        pd.concat([_index_daily(), csi1000_daily], ignore_index=True),
        _index_daily_basic(),
        constituent_daily_basic=_daily_basic(),
        index_weights=_index_weights(),
    )

    last = result.iloc[-1]
    assert last["index_csi1000_active_mv_proxy_cny"] > 0
    assert last["index_csi1000_active_share_ratio_13d_proxy"] > 0
    assert last["index_csi1000_active_mv_coverage_ratio"] == pytest.approx(1.0)


def test_three_level_frames_attach_to_stock_rows_without_changing_row_count() -> None:
    daily_basic = _daily_basic()
    frames = build_active_market_value_feature_frames(
        daily_basic,
        market_daily=_market_daily(),
        index_daily=_index_daily(),
        index_daily_basic=_index_daily_basic(),
        index_weights=_index_weights(),
    )
    rows = daily_basic[["ts_code", "trade_date"]].tail(4).copy()

    attached = attach_active_market_value_features(rows, frames)

    assert len(attached) == len(rows)
    assert attached["stock_active_mv_proxy_cny"].notna().all()
    assert attached["market_active_mv_proxy_cny"].notna().all()
    assert attached["index_csi300_return_1d_pct"].notna().all()


def test_active_market_value_factors_are_registered_for_research_only() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}

    assert set(ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS) <= set(definitions)
    for name in ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS:
        assert definitions[name].role == "research_feature"
        assert definitions[name].lifecycle == "research_candidate"
        assert definitions[name].calculator_id == "active_market_value_research"
    assert not set(ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS).intersection(
        SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
    )
    plan = build_factor_execution_plan(
        [ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS[0]]
    )
    assert [calculator.calculator_id for calculator in plan] == [
        "active_market_value_research"
    ]
