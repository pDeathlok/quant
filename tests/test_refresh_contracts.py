from __future__ import annotations

import pytest

from quant.application.refresh_contracts import (
    REFRESH_SCOPE_STEPS,
    REFRESH_STEP_DEFINITIONS,
    build_progress_steps,
    normalize_refresh_scope,
)


@pytest.mark.parametrize(
    ("raw_scope", "expected"),
    [
        (None, "all"),
        ("", "all"),
        ("short", "short"),
        ("cb-allotment", "cbAllotment"),
        ("allotment", "cbAllotment"),
        ("convertible_bond", "cb"),
        ("convertible-bond", "cb"),
        ("rightSideShadow", "rightSideShadow"),
    ],
)
def test_normalize_refresh_scope_supports_public_aliases(
    raw_scope: str | None,
    expected: str,
) -> None:
    assert normalize_refresh_scope(raw_scope) == expected


def test_normalize_refresh_scope_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="未知刷新范围"):
        normalize_refresh_scope("unknown")


def test_build_progress_steps_returns_fresh_mutable_records() -> None:
    first = build_progress_steps("chan")
    second = build_progress_steps("chan")

    assert [step["key"] for step in first] == ["refresh_data", "chan_model_strategy"]
    assert all(step["status"] == "pending" for step in first)
    first[0]["status"] = "success"
    assert second[0]["status"] == "pending"


@pytest.mark.parametrize("scope", sorted(REFRESH_SCOPE_STEPS))
def test_every_page_refresh_starts_from_shared_market_daily_data(scope: str) -> None:
    assert REFRESH_SCOPE_STEPS[scope][0] == "refresh_data"


def test_daily_refresh_orders_factor_layer_before_long_page_outputs() -> None:
    steps = REFRESH_SCOPE_STEPS["all"]

    assert steps.index("feature_cache") < steps.index("long_stock_pool")
    assert "统一日频因子层" in REFRESH_STEP_DEFINITIONS["feature_cache"]["label"]
    assert "长线因子截面" in REFRESH_STEP_DEFINITIONS["long_stock_pool"]["label"]
    assert "页面股票池" in REFRESH_STEP_DEFINITIONS["long_stock_pool"]["label"]
    assert "带血筹" in REFRESH_STEP_DEFINITIONS["long_stock_pool"]["label"]


def test_right_side_shadow_has_an_independent_refresh_view() -> None:
    assert REFRESH_SCOPE_STEPS["rightSideShadow"] == [
        "refresh_data",
        "signal_cache",
        "right_side_shadow_features",
        "right_side_shadow_score",
        "right_side_shadow_product",
    ]
