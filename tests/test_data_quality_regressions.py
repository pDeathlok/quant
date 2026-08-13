from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    monkeypatch.setattr(
        services,
        "_selector_live_feature_rows",
        lambda symbols, signal_date: {
            symbol: {
                "symbol": symbol,
                "date": signal_date,
                "feature_value": 1.0 if symbol == "000001.SZ" else 2.0,
                "_score_feature_source": "live_daily+daily_basic+market_cross_section",
                "_score_feature_date": signal_date,
            }
            for symbol in symbols
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
    assert {row["_score_feature_source"] for row in features.values()} == {
        "live_daily+daily_basic+market_cross_section"
    }


def test_selector_market_features_are_built_for_exact_signal_date() -> None:
    dates = pd.bdate_range("2026-07-16", periods=20)
    daily = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "pct_chg": np.tile([1.0, -1.0], len(dates)),
        }
    )

    features = services._selector_market_feature_values_from_daily(
        daily,
        dates[-1].strftime("%Y-%m-%d"),
    )

    assert set(features) == set(services.SELECTOR_MARKET_FEATURE_COLUMNS)
    assert features["selector_market_mean_1d"] == 0.0
    assert features["selector_market_median_1d"] == 0.0
    assert features["selector_market_up_ratio_1d"] == 0.5
    assert features["selector_market_mean_20d"] == 0.0
    assert features["selector_market_up_ratio_20d"] == 0.5


def test_selector_turnover_features_use_exact_date_and_trailing_history(tmp_path) -> None:
    dates = pd.bdate_range("2026-07-16", periods=20)
    for offset, trade_date in enumerate(dates, start=1):
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [trade_date.strftime("%Y%m%d")],
                "turnover_rate": [float(offset)],
            }
        ).to_parquet(tmp_path / f"{trade_date:%Y%m%d}.parquet", index=False)

    features = services._selector_turnover_feature_rows(
        ["000001.SZ"],
        dates[-1].strftime("%Y-%m-%d"),
        tmp_path,
    )["000001.SZ"]

    assert features["selector_turnover_relative_5d"] == 20.0 / 18.0
    assert features["selector_turnover_relative_20d"] == 20.0 / 10.5


def test_selector_score_date_never_falls_back_to_stale_history() -> None:
    assert services._selector_model_score_date("2026-08-12") == "2026-08-12"


def test_selector_score_labels_match_the_current_robust_reference_distribution() -> None:
    assert services._score_interpretation(85.0, 50.0)["score_percentile_label"] == (
        "历史参考约 Top 0.02%"
    )
    assert services._score_interpretation(70.0, 50.0)["score_percentile_label"] == (
        "历史参考约 Top 1.4%"
    )
    assert services._score_interpretation(60.0, 50.0)["score_percentile_label"] == (
        "历史参考约 Top 8.6%"
    )


def test_decision_regime_requires_an_exact_feature_date() -> None:
    regime = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-11")],
            "market_regime": ["risk_on"],
        }
    )

    with pytest.raises(RuntimeError, match="exact decision date"):
        services._require_exact_regime_row(
            regime,
            pd.Timestamp("2026-08-12"),
            context="test regime",
            required_columns=["market_regime"],
        )


def test_decision_regime_rejects_missing_required_values() -> None:
    regime = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-12")],
            "market_regime": ["risk_on"],
            "index_return_20d": [np.nan],
        }
    )

    with pytest.raises(RuntimeError, match="missing required values"):
        services._require_exact_regime_row(
            regime,
            pd.Timestamp("2026-08-12"),
            context="test regime",
            required_columns=["market_regime", "index_return_20d"],
        )


def test_selector_return_model_rejects_all_nan_required_feature(monkeypatch) -> None:
    class FakeImputer:
        def transform(self, frame):
            raise AssertionError("incomplete features must not reach the imputer")

    artifact = {
        "features": ["selector_return_5d", "selector_market_mean_1d"],
        "imputer": FakeImputer(),
        "model": object(),
        "score_reference": np.array([0.0, 1.0]),
    }
    monkeypatch.setattr(
        services,
        "_selector_buy_hold_models",
        lambda: {"buy": artifact},
    )
    row = {
        "symbol": "000001.SZ",
        "date": "2026-08-12",
        "matched_groups": ["B1"],
    }
    feature_rows = {
        "000001.SZ": {
            "date": "2026-08-12",
            "selector_return_5d": 1.0,
            "selector_market_mean_1d": np.nan,
            "_score_feature_source": "live_daily+daily_basic+market_cross_section",
            "_score_feature_date": "2026-08-12",
        }
    }

    services._apply_return_model_scores([row], feature_rows)

    assert "historical_buy_score" not in row
    assert row["feature_quality"]["status"] == "failed"
    assert row["feature_quality"]["date"] == "2026-08-12"
    assert row["feature_quality"]["source"] == "live_daily+daily_basic+market_cross_section"
    assert row["feature_quality"]["all_nan_features"] == [
        "selector_market_mean_1d"
    ]


def test_selector_return_model_rejects_only_the_row_with_partial_missing_features(
    monkeypatch,
) -> None:
    class FakeImputer:
        def transform(self, frame):
            assert len(frame) == 1
            assert frame.notna().all(axis=None)
            return frame.to_numpy()

    class FakeModel:
        def predict(self, frame):
            return np.array([1.0])

    artifact = {
        "features": ["selector_return_5d", "selector_market_mean_1d"],
        "imputer": FakeImputer(),
        "model": FakeModel(),
        "score_reference": np.array([0.0, 1.0]),
    }
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {"buy": artifact})
    rows = [
        {"symbol": "000001.SZ", "date": "2026-08-12", "matched_groups": []},
        {"symbol": "000002.SZ", "date": "2026-08-12", "matched_groups": []},
    ]
    feature_rows = {
        "000001.SZ": {
            "date": "2026-08-12",
            "selector_return_5d": 1.0,
            "selector_market_mean_1d": np.nan,
            "_score_feature_source": "live",
            "_score_feature_date": "2026-08-12",
        },
        "000002.SZ": {
            "date": "2026-08-12",
            "selector_return_5d": 2.0,
            "selector_market_mean_1d": 0.5,
            "_score_feature_source": "live",
            "_score_feature_date": "2026-08-12",
        },
    }

    services._apply_return_model_scores(rows, feature_rows)

    assert rows[0]["feature_quality"]["status"] == "failed"
    assert rows[0]["feature_quality"]["missing_features"] == [
        "selector_market_mean_1d"
    ]
    assert "historical_buy_score" not in rows[0]
    assert rows[1]["feature_quality"]["status"] == "complete"
    assert "historical_buy_score" in rows[1]


def test_watchlist_reports_exact_date_feature_failure_instead_of_dropping_symbol(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        services,
        "_latest_similar_pattern_target_date",
        lambda _symbols: "2026-08-12",
    )
    monkeypatch.setattr(services, "_selector_model_feature_rows", lambda _date: {})

    def fail_live_features(_symbols, _date):
        raise RuntimeError("daily input unavailable")

    monkeypatch.setattr(services, "_selector_live_feature_rows", fail_live_features)
    monkeypatch.setattr(
        services,
        "_selector_buy_hold_models",
        lambda: {"buy": {"features": ["selector_return_5d"]}},
    )

    result = services._watchlist_buy_hold_scores(("000001.SZ",))

    assert set(result) == {"000001.SZ"}
    assert result["000001.SZ"]["model_score_available"] is False
    assert result["000001.SZ"]["feature_quality"] == {
        "status": "failed",
        "error": "missing_exact_date_feature_row",
        "source": "unavailable",
        "date": "2026-08-12",
    }


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


def test_bond_grid_cache_rejects_quality_fallback_snapshot() -> None:
    payload = {
        "data_quality": {
            "status": "success",
            "premium_coverage": 1.0,
            "minimum_premium_coverage": 0.9,
            "stale": True,
        },
        "candidates": [],
    }

    assert services._convertible_bond_grid_payload_is_usable(payload) is False


def test_missing_latest_bond_premium_is_repaired_from_stock_close_and_conversion_price() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["110001.SH", "123001.SZ"],
            "trade_date": ["20260807", "20260807"],
            "close": [110.0, 125.0],
            "bond_value": [float("nan"), float("nan")],
            "bond_over_rate": [float("nan"), float("nan")],
        }
    )
    basic = pd.DataFrame(
        {
            "ts_code": ["110001.SH", "123001.SZ"],
            "stk_code": ["600001.SH", "300001.SZ"],
            "conv_price": [10.0, 20.0],
        }
    )
    stock_daily = pd.DataFrame(
        {
            "ts_code": ["600001.SH", "300001.SZ"],
            "trade_date": ["20260807", "20260807"],
            "close": [11.0, 20.0],
        }
    )

    repaired, repaired_rows = bond_plan._fill_missing_premium_fields(
        daily,
        basic,
        stock_daily,
    )

    assert repaired_rows == 2
    assert repaired["bond_value"].round(6).tolist() == [110.0, 100.0]
    assert repaired["bond_over_rate"].round(6).tolist() == [0.0, 25.0]
    assert bond_plan._premium_coverage_for_date(repaired, "20260807") == 1.0


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
