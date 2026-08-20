from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant.features.factor_registry import (
    FACTOR_REGISTRY,
    LONG_ANNUAL_QUALITY_ASSET_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_CASHFLOW_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_PERSISTENCE_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_RAW_FACTOR_COLUMNS,
    LONG_ANNUAL_QUALITY_SCORE_FACTOR_COLUMNS,
    LONG_FACTOR_COLUMNS,
    validate_registry,
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


def test_complete_project_factor_contract_has_all_147_columns() -> None:
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

    assert len(PROJECT_FACTOR_COLUMNS) == 147
    assert list(result.columns[4:-1]) == PROJECT_FACTOR_COLUMNS
    assert result["factor_schema_version"].eq(PROJECT_FACTOR_SCHEMA_VERSION).all()


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
        "price_level",
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


def test_registry_covers_right_side_shadow_118_factor_contract() -> None:
    definitions = {definition.name: definition for definition in FACTOR_REGISTRY}
    payload = right_side_shadow_contract_payload()

    assert len(RULE_FEATURE_COLUMNS) == 118
    assert len(RIGHT_SIDE_SHADOW_FACTOR_COLUMNS) == 265
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
