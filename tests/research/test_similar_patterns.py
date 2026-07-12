from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from quant.research.similar_patterns import (
    SimilarPatternConfig,
    build_stock_vector_cache,
    build_pattern_vector,
    build_t1_scenario_plan,
    build_sell_model_plan,
    candidate_end_indices,
    load_stock_vector_cache,
    select_best_positions_from_contiguous_matches,
    normalize_daily_frame,
    summarize_forecast,
    summarize_status_probs,
)


def test_normalize_daily_frame_accepts_tushare_schema() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20240102", "20240103"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.4],
            "low": [9.9, 10.0],
            "close": [10.2, 10.1],
            "vol": [1000, 1200],
            "pct_chg": [1.0, -0.98],
        }
    )

    daily = normalize_daily_frame(raw, "000001.SZ")

    assert list(daily["symbol"].unique()) == ["000001.SZ"]
    assert daily["date"].is_monotonic_increasing
    assert "volume" in daily.columns
    assert "pct_change" in daily.columns


def test_build_pattern_vector_has_stable_length() -> None:
    dates = pd.bdate_range("2022-01-03", periods=340)
    close = np.linspace(10, 18, len(dates)) + np.sin(np.arange(len(dates)) / 5)
    daily = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001.SZ",
            "name": "测试",
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1000, 2000, len(dates)),
            "amount": np.nan,
            "pct_change": pd.Series(close).pct_change().fillna(0) * 100,
        }
    )
    config = SimilarPatternConfig()

    vector = build_pattern_vector(daily, len(daily) - 1, config)

    assert vector is not None
    assert (
        vector.shape[0]
        == config.lookback_days * 4 + config.weekly_lookback + config.monthly_lookback + 12
    )
    assert np.isfinite(vector).all()


def test_summarizers_return_probabilities_and_quantiles() -> None:
    cases = pd.DataFrame(
        {
            "similarity": [0.5, 0.3, 0.2],
            "fwd_1d": [0.01, -0.02, 0.03],
            "fwd_20d": [0.08, -0.01, 0.02],
            "fwd_60d": [0.12, -0.15, 0.03],
            "max_drawdown_60d": [-0.05, -0.25, -0.08],
        }
    )

    forecast = summarize_forecast(cases)
    statuses = summarize_status_probs(cases)

    assert list(forecast["horizon"]) == ["next_1d", "next_1m", "next_3m"]
    assert forecast["up_probability"].between(0, 100).all()
    assert round(sum(statuses.values()), 1) == 100.0


def test_candidate_end_indices_scans_from_start_date_when_configured() -> None:
    dates = pd.bdate_range("2017-01-02", periods=650)
    daily = pd.DataFrame({"date": dates})
    config = SimilarPatternConfig(
        candidate_start_date="2018-01-01",
        candidate_step_days=1,
        max_candidates_per_symbol=5,
    )

    indices = list(candidate_end_indices(daily, config))

    assert indices
    assert daily.loc[indices[0], "date"] >= pd.Timestamp("2018-01-01")
    assert indices[1] - indices[0] == 1


def test_select_best_positions_keeps_one_per_contiguous_run() -> None:
    candidate_indices = [10, 11, 12, 20, 21, 40]
    similarities = np.array([0.051, 0.060, 0.058, 0.070, 0.065, 0.040])

    selected = select_best_positions_from_contiguous_matches(
        candidate_indices,
        similarities,
        threshold=0.055,
        candidate_step_days=1,
    )

    assert selected == [1, 3]


def test_stock_vector_cache_roundtrip(tmp_path: Path) -> None:
    dates = pd.bdate_range("2022-01-03", periods=380)
    close = np.linspace(10, 20, len(dates)) + np.sin(np.arange(len(dates)) / 7)
    daily = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001.SZ",
            "name": "测试",
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1000, 2500, len(dates)),
            "amount": np.nan,
            "pct_change": pd.Series(close).pct_change().fillna(0) * 100,
        }
    )
    daily_path = tmp_path / "000001.SZ.parquet"
    daily.to_parquet(daily_path, index=False)
    config = SimilarPatternConfig(max_candidates_per_symbol=3, candidate_step_days=5)

    record = build_stock_vector_cache(daily_path, {"name": "测试", "industry": "测试行业"}, config, tmp_path)
    cached = load_stock_vector_cache(Path(record["cache_path"]))

    assert record["status"] == "built"
    assert cached["vectors"].shape[0] > 0
    assert cached["vectors"].shape[1] == config.lookback_days * 4 + config.weekly_lookback + config.monthly_lookback + 12
    assert len(cached["dates"]) == cached["vectors"].shape[0]
    assert "fwd_1d_volume_ratio" in cached
    assert "max_runup_3d" in cached
    assert "max_drawdown_3d" in cached


def test_stock_vector_cache_rebuilds_when_daily_source_changes(tmp_path: Path) -> None:
    dates = pd.bdate_range("2022-01-03", periods=380)
    close = np.linspace(10, 20, len(dates))
    daily = pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001.SZ",
            "name": "测试",
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1000, 2500, len(dates)),
        }
    )
    daily_path = tmp_path / "000001.SZ.parquet"
    daily.to_parquet(daily_path, index=False)
    config = SimilarPatternConfig(max_candidates_per_symbol=3, candidate_step_days=5)

    first = build_stock_vector_cache(daily_path, {}, config, tmp_path)
    second = build_stock_vector_cache(daily_path, {}, config, tmp_path)
    daily.loc[len(daily)] = {
        "date": dates[-1] + pd.offsets.BDay(),
        "symbol": "000001.SZ",
        "name": "测试",
        "open": 20.0,
        "high": 20.5,
        "low": 19.8,
        "close": 20.3,
        "volume": 2600,
    }
    daily.to_parquet(daily_path, index=False)
    third = build_stock_vector_cache(daily_path, {}, config, tmp_path)

    assert first["status"] == "built"
    assert second["status"] == "cache_hit"
    assert third["status"] == "built"


def test_trade_plan_builders_return_recommendations() -> None:
    cases = pd.DataFrame(
        {
            "fwd_1d": np.linspace(-0.04, 0.04, 120),
            "fwd_1d_volume_ratio": np.linspace(0.6, 2.2, 120),
            "max_runup_3d": np.r_[np.full(60, 0.04), np.full(60, 0.01)],
            "max_drawdown_3d": np.r_[np.full(60, -0.01), np.full(60, -0.04)],
            "fwd_20d": np.linspace(-0.1, 0.1, 120),
            "similarity": np.linspace(0.055, 0.065, 120),
            "distance": np.linspace(16, 14, 120),
        }
    )

    scenario = build_t1_scenario_plan(cases, 0.03, 0.03)
    model_plan, summary = build_sell_model_plan(cases, 0.03, 0.03)

    assert not scenario.empty
    assert summary["status"] == "trained"
    assert not model_plan.empty
