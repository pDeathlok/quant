from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.features.index_ma_state import (
    CORE_INDEX_SLUGS, INDEX_MA_STATE_FEATURE_COLUMNS,
    PUBLIC_START_DATES, compute_index_ma_state_features,
)
from quant.features.active_market_value import KEY_INDEX_SPECS


def frame(close, code="000300.SH", start="2019-01-01"):
    return pd.DataFrame({"ts_code": code, "trade_date": pd.bdate_range(start, periods=len(close)), "close": close})


def test_levels_crosses_streaks_and_equality_reset():
    close = np.r_[np.full(250, 100.0), 90.0, 80.0, 110.0, 120.0]
    out = compute_index_ma_state_features(frame(close))
    p = "index_csi300_ma250_"
    assert out.loc[248, p+"below_days"] != out.loc[248, p+"below_days"]
    assert out.loc[249, p+"above_days"] == 0
    assert out.loc[249, p+"below_days"] == 0
    assert out.loc[250, p+"value"] == pytest.approx(99.96)
    assert out.loc[250, p+"below_days"] == 1
    assert out.loc[250, p+"cross_down"] == 1
    assert out.loc[251, p+"below_days"] == 2
    assert out.loc[252, p+"above_days"] == 1
    assert out.loc[252, p+"cross_up"] == 1
    assert out.loc[253, p+"above_days"] == 2
    assert out.loc[253, p+"distance_pct"] == pytest.approx((120/out.loc[253,p+"value"]-1)*100)


def test_future_append_does_not_change_past_and_asof_matches_prefix():
    f = frame(np.linspace(100, 200, 340))
    prefix = compute_index_ma_state_features(f.iloc[:300])
    changed = f.copy()
    changed.loc[300:, "close"] = 1_000_000.0
    full = compute_index_ma_state_features(changed)
    pd.testing.assert_frame_equal(prefix, full.iloc[:300].reset_index(drop=True))
    limited = compute_index_ma_state_features(changed, as_of=f.trade_date.iloc[299])
    pd.testing.assert_frame_equal(prefix, limited)


def test_missing_session_resets_history_and_is_not_forward_filled():
    f = frame(np.arange(600.0)+100)
    calendar = f.trade_date
    f = f.drop(index=300)
    out = compute_index_ma_state_features(f, market_calendar=calendar)
    p = "index_csi300_ma250_"
    assert pd.isna(out.loc[300, p+"below_days"])
    assert pd.isna(out.loc[549, p+"above_days"])
    assert out.loc[550, p+"above_days"] == 1
    assert pd.isna(out.loc[550, p+"cross_up"])


def test_consensus_uses_fixed_seven_indices_and_requires_full_coverage():
    frames = [frame(np.arange(280.0)+100, spec.ts_code)
              for spec in KEY_INDEX_SPECS if spec.slug in CORE_INDEX_SLUGS]
    out = compute_index_ma_state_features(pd.concat(frames, ignore_index=True))
    assert out.iloc[-1].market_index_ma250_eligible_count == 7
    assert out.iloc[-1].market_index_ma250_above_ratio == 1
    missing = compute_index_ma_state_features(pd.concat(frames[:-1], ignore_index=True))
    assert pd.isna(missing.iloc[-1].market_index_ma250_above_ratio)


@pytest.mark.parametrize("slug,code", [("star50","000688.SH"),("bse50","899050.BJ")])
def test_backcast_is_not_visible_before_launch_and_cannot_warmup(slug, code):
    start = pd.Timestamp(PUBLIC_START_DATES[slug])
    dates = pd.bdate_range(start-pd.Timedelta(days=500), start+pd.Timedelta(days=500))
    f = pd.DataFrame({"ts_code":code,"trade_date":dates,"close":np.arange(len(dates))+100.0})
    out = compute_index_ma_state_features(f)
    p = f"index_{slug}_ma250_above"
    assert out.loc[out.date.lt(start),p].isna().all()
    available = out.loc[out.date.ge(start)]
    assert available.iloc[:249][p].isna().all()
    assert available.iloc[249][p] == 1


def test_duplicate_alias_and_date_keys_rejected():
    f = frame(np.arange(260.0)+100)
    with pytest.raises(ValueError, match="duplicate"):
        compute_index_ma_state_features(pd.concat([f, f.iloc[-1:]]))
    alias = f.assign(ts_code="399300.SZ")
    with pytest.raises(ValueError, match="alias"):
        compute_index_ma_state_features(pd.concat([f,alias]))


def test_columns_registered_as_research_and_calculator_is_resolvable():
    from quant.features.factor_registry import FACTOR_REGISTRY
    from quant.features.factor_execution import build_factor_execution_plan
    registry = {f.name:f for f in FACTOR_REGISTRY}
    assert all(registry[c].lifecycle == "research_candidate" for c in INDEX_MA_STATE_FEATURE_COLUMNS)
    assert all(registry[c].semantic_category == "trend" for c in INDEX_MA_STATE_FEATURE_COLUMNS)
    assert [x.calculator_id for x in build_factor_execution_plan([INDEX_MA_STATE_FEATURE_COLUMNS[0]])] == ["index_ma_state_research"]


@pytest.mark.parametrize("descending,expected", [(False,1.0),(True,-1.0)])
def test_trend_state_and_slope_follow_alignment(descending, expected):
    close = np.linspace(100,200,300)
    if descending:
        close = close[::-1]
    out = compute_index_ma_state_features(frame(close))
    assert out.iloc[-1].index_csi300_trend_state == expected
    assert out.iloc[-1].index_csi300_ma250_slope5_pct * expected > 0
