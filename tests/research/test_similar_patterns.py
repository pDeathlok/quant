from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from quant.research.similar_patterns import (
    SimilarPatternConfig,
    analyze_targets_by_threshold,
    apply_probability_calibration,
    build_stock_vector_cache,
    build_pattern_vector,
    build_t1_scenario_plan,
    build_sell_model_plan,
    candidate_end_indices,
    classify_forecast_signal,
    fit_probability_calibration,
    latest_snapshot,
    load_stock_vector_cache,
    optimize_similar_cases,
    select_best_positions_from_contiguous_matches,
    normalize_daily_frame,
    summarize_forecast,
    summarize_status_probs,
)
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.factors.technical import KDJ


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


def test_normalize_daily_frame_removes_ex_right_price_jump() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 3,
            "trade_date": ["20250101", "20250102", "20250103"],
            "open": [99.0, 101.0, 33.5],
            "high": [101.0, 102.0, 34.0],
            "low": [98.0, 100.0, 33.0],
            "close": [100.0, 101.0, 33.67],
            "vol": [1000, 1100, 3300],
            "pct_chg": [0.0, 1.0, 0.0],
        }
    )

    daily = normalize_daily_frame(raw, "000001.SZ")

    assert np.isclose(daily["close"].pct_change().iloc[1], 0.01)
    assert np.isclose(daily["close"].pct_change().iloc[2], 0.0)
    assert np.isclose(daily.iloc[-1]["close"], 33.67)


def test_latest_snapshot_includes_daily_kdj_j() -> None:
    close = pd.Series([10.0, 10.4, 10.2, 10.8, 11.1, 10.9, 11.4, 11.7, 11.5, 12.0, 12.3])
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=len(close)),
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1_000, 2_000, len(close)),
        }
    )

    snapshot = latest_snapshot(daily, len(daily) - 1)

    expected = KDJ().compute(daily)["J"].iloc[-1]
    assert snapshot["kdj_daily_j"] == round(float(expected), 2)


def test_threshold_analysis_loads_target_from_partitioned_daily_store(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=380)
    close = np.linspace(10, 18, len(dates)) + np.sin(np.arange(len(dates)) / 8)
    frame = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "name": "测试",
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "vol": np.linspace(1000, 2500, len(dates)),
            "pct_chg": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
        }
    )
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=tmp_path))
    store.write_market_batch(frame)
    daily_dir = tmp_path / "daily"

    results = analyze_targets_by_threshold(
        daily_dir,
        pd.DataFrame({"ts_code": ["000001.SZ"], "name": ["测试"], "industry": ["银行"]}),
        SimilarPatternConfig(similarity_threshold=0.055),
        target_symbols=["000001.SZ"],
        max_symbols=0,
    )

    assert not (daily_dir / "000001.SZ.parquet").exists()
    assert results["000001.SZ"].target.target_date == dates[-1]


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
    assert np.isfinite(cached["fwd_1d"][-1])
    assert np.isnan(cached["fwd_20d"][-1])
    assert np.isnan(cached["fwd_60d"][-1])


def test_stock_vector_cache_supports_partitioned_daily_source(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=380)
    close = np.linspace(10, 20, len(dates))
    frame = pd.DataFrame(
        {
            "ts_code": "000001.SZ",
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "name": "测试",
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "vol": np.linspace(1000, 2500, len(dates)),
            "pct_chg": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
        }
    )
    store = MarketDataStore(MarketDataStoreConfig(backend="parquet", root=tmp_path))
    store.write_market_batch(frame)
    synthetic_path = tmp_path / "daily/000001.SZ.parquet"
    cache_dir = tmp_path / "cache"
    config = SimilarPatternConfig(max_candidates_per_symbol=3, candidate_step_days=5)

    first = build_stock_vector_cache(synthetic_path, {}, config, cache_dir)
    second = build_stock_vector_cache(synthetic_path, {}, config, cache_dir)
    next_date = dates[-1] + pd.offsets.BDay()
    store.write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [next_date.strftime("%Y%m%d")],
                "date": [next_date],
                "name": ["测试"],
                "open": [20.0],
                "high": [20.5],
                "low": [19.8],
                "close": [20.3],
                "vol": [2600.0],
                "pct_chg": [1.5],
            }
        )
    )
    third = build_stock_vector_cache(synthetic_path, {}, config, cache_dir)

    assert not synthetic_path.exists()
    assert first["status"] == "built"
    assert second["status"] == "cache_hit"
    assert third["status"] == "built"
    assert load_stock_vector_cache(Path(third["cache_path"]))["source_fingerprint"].startswith(
        "partitioned:"
    )


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


def test_optimize_similar_cases_deduplicates_events_and_caps_each_date() -> None:
    cases = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "industry": ["汽车整车", "汽车整车", "软件服务", "医药商业", "汽车整车"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02", "2024-02-02"]),
            "similarity": [0.080, 0.070, 0.065, 0.060, 0.075],
            "market_regime": ["risk_on", "risk_on", "risk_off", "neutral", "risk_on"],
            "fwd_1d": [0.02, -0.01, 0.01, -0.02, 0.03],
            "fwd_20d": [0.05, 0.01, -0.03, -0.04, 0.08],
            "fwd_60d": [0.10, 0.03, -0.08, -0.06, 0.12],
        }
    )
    config = SimilarPatternConfig(max_effective_cases=10, max_events_per_date=2)

    optimized, summary = optimize_similar_cases(
        cases,
        config,
        target_date=pd.Timestamp("2025-01-02"),
        target_industry="汽车整车",
        target_market_regime="risk_on",
    )

    assert set(optimized["symbol"]) == {"A", "C", "E"}
    assert optimized.groupby("date").size().max() == 2
    assert optimized.loc[optimized["symbol"] == "A", "forecast_weight"].iloc[0] > optimized.loc[
        optimized["symbol"] == "C", "forecast_weight"
    ].iloc[0]
    assert summary["raw_cases"] == 5
    assert summary["deduplicated_cases"] == 3
    assert 0 < summary["effective_sample_size"] <= 3


def test_classify_forecast_signal_uses_observe_zone_and_breakdown_gate() -> None:
    config = SimilarPatternConfig(signal_bearish_max=45.0, signal_bullish_min=55.0)
    healthy = {"dist_ma20": 1.0, "dist_ma60": 2.0, "drawdown_60d": -4.0, "vol_ratio20": 1.0}
    breakdown = {"dist_ma20": -4.0, "dist_ma60": -7.0, "drawdown_60d": -14.0, "vol_ratio20": 1.6}

    assert classify_forecast_signal(52.0, healthy, "neutral", config)["signal"] == "observe"
    assert classify_forecast_signal(42.0, healthy, "neutral", config)["signal"] == "bearish"
    gated = classify_forecast_signal(57.0, breakdown, "risk_off", config)
    assert gated["signal"] == "observe"
    assert gated["risk_gate"] == "blocked"
    assert gated["reasons"]


def test_probability_calibration_is_monotonic_and_serializable() -> None:
    probabilities = [35, 40, 45, 50, 55, 60, 65, 70] * 4
    outcomes = [False, False, False, False, True, True, True, True] * 4

    calibration = fit_probability_calibration(probabilities, outcomes, min_samples=20)
    calibrated = [apply_probability_calibration(value, calibration) for value in [40, 50, 60, 70]]

    assert calibration["status"] == "fitted"
    assert calibrated == sorted(calibrated)
    assert 0 <= calibrated[0] <= calibrated[-1] <= 100
    assert isinstance(calibration["x"], list)
    assert isinstance(calibration["y"], list)


def test_summarize_forecast_prefers_optimized_weight_column() -> None:
    cases = pd.DataFrame(
        {
            "similarity": [0.9, 0.1],
            "forecast_weight": [0.1, 0.9],
            "fwd_1d": [0.10, -0.10],
            "fwd_20d": [0.10, -0.10],
            "fwd_60d": [0.10, -0.10],
        }
    )

    forecast = summarize_forecast(cases).set_index("horizon")

    assert forecast.loc["next_1d", "up_probability"] == 10.0
