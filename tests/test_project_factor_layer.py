from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant.features.active_market_value import (
    ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS,
)
from quant.features.candlestick_context import (
    CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS,
)
from quant.features.market_breadth import (
    MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS,
)
from quant.features.canonical_factor_names import FORBIDDEN_COMPATIBILITY_ALIASES
from quant.features.factor_registry import (
    CHAN_LIVE_FACTOR_COLUMNS,
    FACTOR_REGISTRY,
    FACTOR_ALIAS_TARGETS,
    FACTOR_LEVELS,
    LONG_ANNUAL_QUALITY_ASSET_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_CASHFLOW_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_PERSISTENCE_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS,
    LONG_FACTOR_COLUMNS,
    LONG_PRODUCTION_FACTOR_COLUMNS,
    LONG_RESEARCH_FACTOR_COLUMNS,
    PRODUCTION_REGISTRY_COLUMNS,
    SELECTOR_LIVE_FACTOR_COLUMNS,
    SEMANTIC_CATEGORIES,
    validate_registry,
)
from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
    LEFT_SIDE_SIGNALS,
)
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_FACTOR_COLUMNS,
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS,
    right_side_shadow_contract_payload,
)
from quant.research.right_side_unified_features import RULE_FEATURE_COLUMNS
from quant.features.long_weekly_factors import build_long_weekly_factor_frame, long_model_candidate_columns
from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    PROJECT_FACTOR_SCHEMA_VERSION,
    admit_factors_by_sample,
    calculate_project_factor_frame,
    calculate_legacy_market_factors,
    calculate_limit_up_flags,
    calculate_project_market_factors,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.ml.feature_coverage import RequiredFeatureCoverageError
from quant.routine import pipeline
from quant.routine import b1_daily_plan


def _daily(rows: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    close = np.linspace(10.0, 16.0, rows) + np.sin(np.arange(rows) / 9.0) * 0.2
    return pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "symbol": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "pct_chg": pd.Series(close).pct_change().fillna(0) * 100,
            "volume": 1_000_000 + np.arange(rows) * 1_000,
        }
    )


def test_complete_project_factor_contract_has_all_145_canonical_columns() -> None:
    daily = _daily()
    basic = pd.DataFrame(
        {
            "ts_code": daily["ts_code"],
            "trade_date": daily["trade_date"],
            "pe_ttm": 12.0,
            "pb": 1.5,
        }
    )

    result = calculate_project_factor_frame(daily, daily_basic_features=basic)

    assert len(PROJECT_FACTOR_COLUMNS) == 145
    assert list(result.columns[4:-1]) == PROJECT_FACTOR_COLUMNS
    assert result["factor_schema_version"].eq(PROJECT_FACTOR_SCHEMA_VERSION).all()
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(result.columns)


def test_project_market_factors_do_not_change_when_future_rows_are_appended() -> None:
    daily = _daily()
    base = calculate_project_market_factors(daily)
    future_dates = pd.bdate_range(daily["date"].iloc[-1] + pd.Timedelta(days=1), periods=20)
    # Simulate a 2-for-1 split after the historical sample. Recomputing with
    # these future rows must not rewrite prior absolute price factors.
    split_pre_close = daily["close"].iloc[-1] / 2.0
    future_close = split_pre_close + np.arange(1, 21) * 0.025
    future = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "symbol": "000001.SZ",
            "trade_date": future_dates.strftime("%Y%m%d"),
            "date": future_dates,
            "open": future_close * 0.995,
            "high": future_close * 1.02,
            "low": future_close * 0.98,
            "close": future_close,
            "pre_close": np.r_[split_pre_close, future_close[:-1]],
            "pct_chg": pd.Series(np.r_[split_pre_close, future_close]).pct_change().iloc[1:].to_numpy() * 100,
            "volume": 1_400_000 + np.arange(20) * 1_000,
        }
    )
    extended = calculate_project_market_factors(pd.concat([daily, future], ignore_index=True))
    columns = [
        "alpha003",
        "kdj_d_j",
        "weekly_ma55",
        "ma_120",
        "close",
        "return_120d",
        "volatility_60d",
    ]

    pd.testing.assert_frame_equal(
        base[columns],
        extended.iloc[: len(base)][columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_sample_admission_uses_only_population_and_degeneracy() -> None:
    frame = pd.DataFrame(
        {
            "enough": np.arange(100, dtype=float),
            "sparse": [1.0] * 9 + [np.nan] * 91,
            "constant": [1.0] * 100,
        }
    )

    admitted, report = admit_factors_by_sample(
        frame,
        ["enough", "sparse", "constant", "missing"],
        minimum_non_null_rows=10,
        minimum_coverage=0.10,
    )

    assert admitted == ["enough"]
    assert dict(zip(report["factor"], report["reason"])) == {
        "enough": "admitted",
        "sparse": "insufficient_rows",
        "constant": "constant_or_degenerate",
        "missing": "missing_column",
    }


def test_registry_is_unique_and_covers_daily_and_long_contracts() -> None:
    validate_registry()
    names = {definition.name for definition in FACTOR_REGISTRY}

    assert set(PROJECT_FACTOR_COLUMNS) <= names
    assert set(LONG_FACTOR_COLUMNS) <= names
    assert "analyst_forward_y0_eps_std_180d" in names
    assert "analyst_forward_y2_price_mean_180d" in names
    assert len(names) == len(FACTOR_REGISTRY)
    assert all(definition.point_in_time for definition in FACTOR_REGISTRY)


def test_registry_governance_separates_canonical_alias_and_lifecycle() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    role_counts = pd.Series([definition.role for definition in FACTOR_REGISTRY]).value_counts()

    assert len(FACTOR_REGISTRY) == (
        597
        + len(LEFT_SIDE_RULE_FEATURE_COLUMNS)
        + len(LEFT_SIDE_SIGNALS)
        + len(ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS)
        + len(MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS)
        + len(CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS)
    )
    assert role_counts.to_dict() == {
        "feature": 583 + len(LEFT_SIDE_RULE_FEATURE_COLUMNS),
        "strategy_identity": 14 + len(LEFT_SIDE_SIGNALS),
        "research_feature": (
            len(ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS)
            + len(MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS)
            + len(CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS)
        ),
    }
    assert FACTOR_ALIAS_TARGETS == {}
    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(definitions)
    assert set(PRODUCTION_REGISTRY_COLUMNS) <= set(definitions)
    assert set(SELECTOR_LIVE_FACTOR_COLUMNS) <= set(definitions)
    assert set(CHAN_LIVE_FACTOR_COLUMNS) <= set(definitions)
    assert len(LONG_PRODUCTION_FACTOR_COLUMNS) == 82
    assert len(LONG_RESEARCH_FACTOR_COLUMNS) == 108
    assert not set(LONG_PRODUCTION_FACTOR_COLUMNS) & set(LONG_RESEARCH_FACTOR_COLUMNS)
    assert definitions["rsi_6"].lifecycle == "research_candidate"
    assert definitions["selector_return_1d"].lifecycle == "production_model"
    for name in ACTIVE_MARKET_VALUE_RESEARCH_FEATURE_COLUMNS:
        assert definitions[name].role == "research_feature"
        assert definitions[name].lifecycle == "research_candidate"
    for name in MARKET_BREADTH_RESEARCH_FEATURE_COLUMNS:
        assert definitions[name].role == "research_feature"
        assert definitions[name].lifecycle == "research_candidate"
    for name in CANDLE_CONTEXT_RESEARCH_FEATURE_COLUMNS:
        assert definitions[name].role == "research_feature"
        assert definitions[name].lifecycle == "research_candidate"
    assert definitions["short_balance"].family == "margin"
    assert definitions["small_net_amount_ratio"].family == "moneyflow"


def test_registry_uses_strategy_neutral_business_categories() -> None:
    assert all(
        definition.semantic_category in SEMANTIC_CATEGORIES
        for definition in FACTOR_REGISTRY
    )
    assert all(definition.factor_level in FACTOR_LEVELS for definition in FACTOR_REGISTRY)
    assert all(definition.calculator_id for definition in FACTOR_REGISTRY)
    assert all(definition.calculation_owner for definition in FACTOR_REGISTRY)
    assert all(definition.consumers == definition.active_consumers for definition in FACTOR_REGISTRY)
    assert not any(
        token in definition.semantic_category
        for definition in FACTOR_REGISTRY
        for token in ("right_side", "left_side", "selector", "chan", "good_stock")
    )
    assert all(
        definition.refresh_cadence == "trade_daily"
        for definition in FACTOR_REGISTRY
        if definition.lifecycle
        in {"production_model", "production_materialized", "strategy_identity"}
    )
def test_registry_includes_computable_psy_and_vr_candidates() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}

    for name in ("psy_12", "vr_6", "vr_12", "vr_24"):
        assert definitions[name].lifecycle == "research_candidate"
        assert definitions[name].calculation_entrypoint.endswith(
            "calculate_legacy_market_factors"
        )


def test_project_layer_emits_no_compatibility_aliases() -> None:
    project = calculate_project_market_factors(_daily())

    assert FORBIDDEN_COMPATIBILITY_ALIASES.isdisjoint(project.columns)
    assert {"close", "ma_20"} <= set(project.columns)
    assert {"kdj_k", "kdj_d", "kdj_j"}.isdisjoint(
        calculate_legacy_market_factors(_daily()).columns
    )


def test_limit_up_flags_respect_board_st_and_exchange_limit_prices() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-08-21", "2026-08-21", "2026-08-21", "2026-08-21"]
            ),
            "ts_code": ["600000.SH", "600001.SH", "300001.SZ", "430001.BJ"],
            "name": ["主板", "ST测试", "创业板", "北交所"],
            "pre_close": [10.0, 10.0, 10.0, 10.0],
            "close": [11.0, 10.5, 11.0, 13.0],
        }
    )

    assert calculate_limit_up_flags(frame).tolist() == [True, True, False, True]

    frame.loc[2, "close"] = 12.0
    frame["up_limit"] = [11.0, 10.5, 12.0, 13.0]
    assert calculate_limit_up_flags(frame).all()


def test_registry_covers_stable_annual_quality_model_contract() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}

    assert len(LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS) == 23
    assert len(LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS) == 11
    assert len(LONG_ANNUAL_QUALITY_FACTOR_COLUMNS) == 34
    assert set(LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS) == {
        *LONG_ANNUAL_QUALITY_CASHFLOW_FACTOR_COLUMNS,
        *LONG_ANNUAL_QUALITY_PERSISTENCE_FACTOR_COLUMNS,
        *LONG_ANNUAL_QUALITY_ASSET_FACTOR_COLUMNS,
    }
    assert set(LONG_ANNUAL_QUALITY_FACTOR_COLUMNS) <= set(LONG_FACTOR_COLUMNS)
    assert set(LONG_ANNUAL_QUALITY_FACTOR_COLUMNS) <= set(definitions)
    assert all(
        definitions[name].frequency == "weekly"
        and definitions[name].point_in_time
        and definitions[name].consumers == ("long_entry_quality_shadow",)
        for name in LONG_ANNUAL_QUALITY_FACTOR_COLUMNS
    )
    assert all(
        "announcement_pit" in definitions[name].source
        for name in LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS
    )
    assert definitions["industry_value_score"].source.endswith("current_industry_mapping")


def test_registry_covers_right_side_shadow_113_factor_contract() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    payload = right_side_shadow_contract_payload()

    assert len(RULE_FEATURE_COLUMNS) == 113
    assert len(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS) == 258
    assert len(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS) == 14
    assert payload["factor_contract_sha256"] == (
        RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256
    )
    assert set(RULE_FEATURE_COLUMNS) <= set(definitions)
    assert set(RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS) <= set(definitions)
    assert all(
        definitions[name].point_in_time
        and definitions[name].frequency == "daily"
        and {"right_side_unified_shadow", "right_side_unified"}
        <= set(definitions[name].consumers)
        for name in RULE_FEATURE_COLUMNS
    )
    assert all(
        definitions[name].role == "strategy_identity"
        for name in RIGHT_SIDE_SHADOW_IDENTITY_COLUMNS
    )


def test_registry_covers_left_side_unified_contract() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}

    assert set(LEFT_SIDE_RULE_FEATURE_COLUMNS) <= set(definitions)
    assert set(LEFT_SIDE_SIGNALS) <= set(definitions)
    assert all(
        definitions[name].family == "left_side_rule"
        and definitions[name].consumers == ("left_side_unified",)
        for name in LEFT_SIDE_RULE_FEATURE_COLUMNS
    )
    assert all(
        definitions[name].role == "strategy_identity"
        for name in LEFT_SIDE_SIGNALS
    )


def test_weekly_dual_pr_and_history_are_point_in_time() -> None:
    dates = pd.date_range("2020-01-03", periods=130, freq="W-FRI")
    weekly = pd.DataFrame(
        {
            "date": dates,
            "trade_date": dates.strftime("%Y%m%d"),
            "ts_code": "000001.SZ",
            "name": "测试",
            "industry": "软件",
            "close": np.linspace(10, 15, len(dates)),
            "ma_20": np.linspace(9.5, 14.5, len(dates)),
            "ma_60": np.linspace(9, 14, len(dates)),
            "ma_120": np.linspace(8, 13, len(dates)),
            "median_close_60": np.linspace(9, 14, len(dates)),
            "return_120d": 0.10,
            "volatility_60d": 0.20,
            "downside_volatility_60d": 0.15,
            "pe_ttm": 15.0,
            "pb": 2.0,
            "ps_ttm": 3.0,
            "dv_ttm": 2.0,
            "roe": 15.0,
            "good_stock_score": 70.0,
            "or_yoy": 10.0,
            "debt_to_assets": 30.0,
        }
    )
    result = build_long_weekly_factor_frame(weekly, history_windows=((2, 5),))

    assert np.isclose(result["pr_pe"].iloc[-1], 1.0)
    assert np.isclose(result["pr_pb"].iloc[-1], 100 * 2 / 15**2)
    assert result["pe_hist_percentile_5y"].iloc[103:].notna().all()
    assert "label_value_rank_52w" not in long_model_candidate_columns(result)


def test_pipeline_publishes_factor_registry_snapshot(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)

    result = pipeline.refresh_factor_registry_snapshot()

    assert result["status"] == "success"
    payload = json.loads((tmp_path / "data/features/factor_registry/latest.json").read_text())
    assert payload["factor_count"] == len(FACTOR_REGISTRY)
    assert payload["canonical_factor_count"] == (
        583 + len(LEFT_SIDE_RULE_FEATURE_COLUMNS)
    )
    assert payload["compatibility_alias_count"] == 0
    assert payload["strategy_identity_count"] == 14 + len(LEFT_SIDE_SIGNALS)
    assert payload["point_in_time_factor_count"] == payload["factor_count"]


def test_legacy_factor_mode_is_explicitly_stamped() -> None:
    result = calculate_project_market_factors(
        _daily(),
        factor_schema_version=LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    )

    assert result["factor_schema_version"].eq(
        LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
    ).all()


def test_legacy_model_is_blocked_from_current_factor_rows(monkeypatch, tmp_path) -> None:
    class LegacyModel:
        feature_names_in_ = ["alpha003"]

    monkeypatch.setattr(b1_daily_plan.joblib, "load", lambda path: LegacyModel())

    with pytest.raises(RuntimeError, match="factor schema is incompatible"):
        b1_daily_plan.predict_models(
            pd.DataFrame(
                {
                    "alpha003": [0.1],
                    "factor_schema_version": [PROJECT_FACTOR_SCHEMA_VERSION],
                }
            ),
            model_dir=tmp_path,
            model_names=("up5",),
        )


def test_legacy_model_accepts_only_legacy_factor_rows(monkeypatch, tmp_path) -> None:
    class LegacyModel:
        feature_names_in_ = ["alpha003"]

        def predict_proba(self, frame):
            return np.array([[0.25, 0.75]])

    monkeypatch.setattr(b1_daily_plan.joblib, "load", lambda path: LegacyModel())

    result = b1_daily_plan.predict_models(
        pd.DataFrame(
            {
                "alpha003": [0.1],
                "factor_schema_version": [
                    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
                ],
            }
        ),
        model_dir=tmp_path,
        model_names=("up5",),
    )

    assert result["pred_up5"].iloc[0] == pytest.approx(0.75)
    assert result.attrs["feature_coverage"]["status"] == "valid"


def test_b1_prediction_rejects_an_all_null_required_feature(monkeypatch, tmp_path) -> None:
    class LegacyModel:
        feature_names_in_ = ["alpha003"]

        def predict_proba(self, frame):
            raise AssertionError("invalid feature batches must fail before prediction")

    monkeypatch.setattr(b1_daily_plan.joblib, "load", lambda path: LegacyModel())

    with pytest.raises(RequiredFeatureCoverageError) as exc_info:
        b1_daily_plan.predict_models(
            pd.DataFrame(
                {
                    "alpha003": [np.nan],
                    "factor_schema_version": [
                        LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
                    ],
                }
            ),
            model_dir=tmp_path,
            model_names=("up5",),
        )

    assert exc_info.value.report["missing_columns"] == []
    assert exc_info.value.report["all_null_features"] == ["alpha003"]
