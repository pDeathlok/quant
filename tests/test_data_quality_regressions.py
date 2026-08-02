from __future__ import annotations

import pandas as pd

from quant.routine import convertible_bond_grid_plan as bond_plan
from quant.application.workspaces.convertible_bonds import (
    ConvertibleBondGridDependencies,
    build_convertible_bond_grid_workspace,
)
from quant.strategies.convertible_bond.grid import (
    add_low_position_features,
    add_market_state_features,
)
from quant.webapp import services


def test_similar_probability_rejects_information_collapsing_calibration() -> None:
    validation = {
        "targets": ["002594.SZ"],
        "calibrations": {
            "next_1d": {
                "x": [30.0, 45.0, 55.0, 70.0, 85.0],
                "y": [35.0, 69.23, 69.23, 69.23, 80.0],
            }
        },
        "model_selection": {
            "next_1d": {
                "selected": {
                    "source": "calibrated",
                    "bearish_max": 47.0,
                    "bullish_min": 53.0,
                    "enable_risk_gate": False,
                }
            }
        },
    }

    policy = services._similar_pattern_probability_policy(
        validation,
        "002594.SZ",
        "next_1d",
    )

    assert policy["source"] == "optimized"
    assert policy["reason"] == "calibration_information_collapse"
    assert policy["calibration_max_flat_span"] == 25.0
    assert policy["bullish_min"] == services.SIMILAR_PATTERN_CONFIG.signal_bullish_min


def test_similar_probability_does_not_apply_cross_stock_medium_term_model() -> None:
    validation = {
        "targets": ["002594.SZ"],
        "model_selection": {
            "next_1m": {
                "selected": {
                    "source": "regime_industry",
                    "bearish_max": 48.0,
                    "bullish_min": 52.0,
                }
            }
        },
    }

    policy = services._similar_pattern_probability_policy(
        validation,
        "600150.SH",
        "next_1m",
    )

    assert policy["source"] == "optimized"
    assert policy["reason"] == "symbol_out_of_validation_scope"
    assert policy["applicable"] is False


def test_selector_features_fall_back_to_same_day_market_data(monkeypatch) -> None:
    monkeypatch.setattr(services, "_selector_model_feature_rows", lambda _date: {})

    class FakeStore:
        def read_market_range(self, *args, **kwargs):
            return pd.DataFrame(
                {
                    "symbol": ["000001.SZ", "000002.SZ"],
                    "date": pd.to_datetime(["2026-07-31", "2026-07-31"]),
                    "close": [10.0, 20.0],
                }
            )

    monkeypatch.setattr(services, "MarketDataStore", lambda _config: FakeStore())
    monkeypatch.setattr(
        services,
        "_selector_watchlist_feature_row_from_daily",
        lambda symbol, signal_date, _frame: {
            "symbol": symbol,
            "date": signal_date,
            "feature_value": 1.0 if symbol == "000001.SZ" else 2.0,
        },
    )

    features = services._selector_feature_rows_for_score_rows(
        [
            {"symbol": "000001.SZ", "date": "2026-07-31"},
            {"symbol": "000002.SZ", "date": "2026-07-31"},
        ]
    )

    assert features["000001.SZ"]["feature_value"] == 1.0
    assert features["000002.SZ"]["feature_value"] == 2.0
    assert {row["_score_feature_source"] for row in features.values()} == {"live_daily"}


def test_selector_cache_upgrade_preserves_rows_without_model_features(monkeypatch) -> None:
    monkeypatch.setattr(services, "_row_display_quality_gate", lambda _row: True)
    monkeypatch.setattr(services, "_diversify_default_rows", lambda rows, _limit: rows)

    def fake_rescore(rows, _features=None):
        rows[0].update(
            {
                "selector_score": 81.0,
                "opportunity_score": 81.0,
                "holding_score": 72.0,
                "buy_score_source": "historical_return_model",
                "hold_score_source": "historical_return_model",
            }
        )
        rows[1].update(
            {
                "selector_score": 50.0,
                "opportunity_score": 50.0,
                "holding_score": 50.0,
            }
        )
        return rows

    monkeypatch.setattr(services, "_apply_historical_score_normalization", fake_rescore)
    payload = {
        "stocks": [
            {
                "symbol": "000001.SZ",
                "selector_score": 60.0,
                "opportunity_score": 60.0,
                "holding_score": 55.0,
                "matched_count": 1,
                "best_profit_factor": 2.0,
            },
            {
                "symbol": "000002.SZ",
                "selector_score": 58.0,
                "opportunity_score": 58.0,
                "holding_score": 54.0,
                "matched_count": 1,
                "best_profit_factor": 1.5,
            },
        ]
    }

    upgraded = services._display_selector_payload(payload, limit=10)
    by_symbol = {row["symbol"]: row for row in upgraded["stocks"]}

    assert by_symbol["000001.SZ"]["opportunity_score"] == 81.0
    assert by_symbol["000002.SZ"]["opportunity_score"] == 58.0


def test_missing_bond_premium_remains_missing_instead_of_becoming_zero() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["110001.SH"] * 20,
            "trade_date": pd.date_range("2026-01-01", periods=20).strftime("%Y%m%d"),
            "close": range(100, 120),
            "bond_over_rate": [float("nan")] * 20,
        }
    )

    featured = add_market_state_features(add_low_position_features(daily))

    assert featured["premium_rate"].isna().all()
    assert featured["double_low"].isna().all()
    assert featured["market_median_double_low"].isna().all()


def test_bond_plan_uses_latest_date_with_complete_premium_data() -> None:
    daily = pd.DataFrame(
        {
            "trade_date": ["20260730", "20260730", "20260731", "20260731"],
            "close": [100.0, 101.0, 102.0, 103.0],
            "bond_over_rate": [8.0, 9.0, float("nan"), float("nan")],
        }
    )

    assert bond_plan._resolve_trade_date(daily, None) == "20260730"
    assert bond_plan._premium_coverage_for_date(daily, "20260730") == 1.0
    assert bond_plan._premium_coverage_for_date(daily, "20260731") == 0.0


def test_bond_grid_cache_rejects_fabricated_zero_premiums() -> None:
    payload = {
        "candidates": [
            {"premium_rate": 0.0},
            {"premium_rate": 0.0},
            {"premium_rate": 0.0},
        ]
    }

    assert services._convertible_bond_grid_payload_is_usable(payload) is False


def test_bond_workspace_caches_quality_fallback_under_requested_date() -> None:
    writes = []
    dependencies = ConvertibleBondGridDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        read_legacy_snapshot=lambda: None,
        promote_legacy_snapshot=lambda *args, **kwargs: None,
        write_snapshot=lambda *args, **kwargs: writes.append((args, kwargs)),
        refresh_daily=lambda **kwargs: None,
        build_plan=lambda **kwargs: {
            "trade_date": "20260730",
            "data_quality": {
                "requested_trade_date": "20260731",
                "resolved_trade_date": "20260730",
                "stale": True,
            },
            "candidates": [],
        },
    )

    build_convertible_bond_grid_workspace(
        trade_date="2026-07-31",
        limit=18,
        dependencies=dependencies,
    )

    assert writes[0][0][1] == "2026-07-31"
