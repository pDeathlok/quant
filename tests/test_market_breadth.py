from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.factor_execution import build_factor_execution_plan
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.market_breadth import (
    MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS,
    attach_market_breadth_features,
    compute_market_breadth_features,
)
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
)


def _market_daily(rows: int = 66) -> pd.DataFrame:
    dates = pd.bdate_range("2026-05-01", periods=rows)
    position = np.arange(rows, dtype=float)
    closes = {
        "000001.SZ": 100.0 + position,
        "000002.SZ": 200.0 - position,
        "600000.SH": np.full(rows, 50.0),
        "688001.SH": 80.0 + position * 2.0,
    }
    closes["000001.SZ"][-1] = closes["000001.SZ"][-2] * 1.10
    closes["000002.SZ"][-1] = closes["000002.SZ"][-2] * 0.90
    closes["688001.SH"][-1] = closes["688001.SH"][-2] * 1.06
    frames = []
    for symbol, values in closes.items():
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": symbol,
                    "trade_date": dates.strftime("%Y%m%d"),
                    "close": values,
                    "volume": np.full(rows, 1_000.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_market_breadth_counts_ratios_and_extreme_moves() -> None:
    result = compute_market_breadth_features(_market_daily())

    last = result.iloc[-1]
    assert last["market_breadth_stock_count"] == 4
    assert last["market_breadth_return_eligible_count"] == 4
    assert last["market_breadth_up_count_1d"] == 2
    assert last["market_breadth_down_count_1d"] == 1
    assert last["market_breadth_flat_count_1d"] == 1
    assert last["market_breadth_up_ratio_1d"] == pytest.approx(0.50)
    assert last["market_breadth_down_ratio_1d"] == pytest.approx(0.25)
    assert last["market_breadth_flat_ratio_1d"] == pytest.approx(0.25)
    assert last["market_breadth_advance_decline_diff_1d"] == 1
    assert last["market_breadth_advance_decline_ratio_1d"] == pytest.approx(2.0)
    assert last["market_breadth_advance_decline_spread_1d"] == pytest.approx(0.25)
    assert last["market_breadth_up5_count_1d"] == 2
    assert last["market_breadth_down5_count_1d"] == 1
    assert last["market_breadth_limit_up_count_proxy"] == 1
    assert last["market_breadth_limit_down_count_proxy"] == 1


def test_market_breadth_ma_and_new_high_low_use_eligible_denominators() -> None:
    result = compute_market_breadth_features(_market_daily())

    last = result.iloc[-1]
    assert last["market_breadth_ma20_eligible_count"] == 4
    assert last["market_breadth_above_ma20_count"] == 2
    assert last["market_breadth_above_ma20_ratio"] == pytest.approx(0.50)
    assert last["market_breadth_high_low_20d_eligible_count"] == 4
    assert last["market_breadth_new_high_20d_count"] == 2
    assert last["market_breadth_new_low_20d_count"] == 1
    assert last["market_breadth_new_high_20d_ratio"] == pytest.approx(0.50)
    assert last["market_breadth_new_low_20d_ratio"] == pytest.approx(0.25)
    assert last["market_breadth_new_high_low_diff_20d"] == 1
    assert last["market_breadth_new_high_low_spread_20d"] == pytest.approx(0.25)


def test_market_breadth_excludes_zero_volume_rows_from_return_denominator() -> None:
    daily = _market_daily()
    final_date = daily["trade_date"].max()
    daily.loc[
        daily["ts_code"].eq("688001.SH") & daily["trade_date"].eq(final_date),
        "volume",
    ] = 0.0

    last = compute_market_breadth_features(daily).iloc[-1]

    assert last["market_breadth_stock_count"] == 4
    assert last["market_breadth_return_eligible_count"] == 3
    assert last["market_breadth_up_count_1d"] == 1
    assert last["market_breadth_up_ratio_1d"] == pytest.approx(1 / 3)


def test_market_breadth_is_prefix_causal() -> None:
    daily = _market_daily()
    cutoff = sorted(daily["trade_date"].unique())[-10]

    full = compute_market_breadth_features(daily)
    prefix = compute_market_breadth_features(daily[daily["trade_date"].le(cutoff)])
    comparable = full[full["trade_date"].le(cutoff)].merge(
        prefix,
        on="trade_date",
        suffixes=("_full", "_prefix"),
        validate="one_to_one",
    )

    for column in MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            comparable[f"{column}_full"],
            comparable[f"{column}_prefix"],
            equal_nan=True,
        )


def test_market_breadth_attaches_date_features_without_changing_rows() -> None:
    features = compute_market_breadth_features(_market_daily())
    rows = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [features.iloc[-1]["trade_date"]] * 2,
        }
    )

    attached = attach_market_breadth_features(rows, features)

    assert len(attached) == 2
    assert attached["market_breadth_up_count_1d"].eq(2).all()


def test_market_breadth_factors_are_registered_for_research_only() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}

    assert set(MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS) <= set(definitions)
    for name in MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS:
        assert definitions[name].role == "research_feature"
        assert definitions[name].lifecycle == "research_candidate"
        assert definitions[name].calculator_id == "market_breadth_research"
    assert not set(MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS).intersection(
        SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
    )
    plan = build_factor_execution_plan([MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS[0]])
    assert [calculator.calculator_id for calculator in plan] == [
        "market_breadth_research"
    ]
