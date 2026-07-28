from __future__ import annotations

import numpy as np
import pandas as pd

from quant.features import daily_factor_layer as layer


def _daily(rows: int = 260, symbol: str = "000001.SZ") -> pd.DataFrame:
    dates = pd.bdate_range("2025-06-01", periods=rows)
    trend = np.linspace(10.0, 16.0, rows)
    wave = np.sin(np.arange(rows) / 7) * 0.4
    close = trend + wave
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "symbol": symbol,
            "trade_date": dates.strftime("%Y%m%d"),
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "pct_chg": pd.Series(close).pct_change().fillna(0) * 100,
            "vol": 1_000_000 + (np.arange(rows) % 17) * 25_000,
            "volume": 1_000_000 + (np.arange(rows) % 17) * 25_000,
        }
    )


def test_factor_cache_is_versioned_idempotent_and_reused(monkeypatch, tmp_path) -> None:
    daily = _daily()
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(tmp_path / "factors"))

    first = layer.attach_daily_base_factors(daily, "000001.SZ")
    cache_paths = sorted((tmp_path / "factors" / layer.FACTOR_LAYER_VERSION / "000001.SZ").glob("*.parquet"))
    assert cache_paths
    assert {"kdj_d_j", "bbi", "z_vol_ratio_5", "factor_version"} - set(first.columns) == {"factor_version"}

    monkeypatch.setattr(
        layer,
        "calculate_daily_base_factors",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache should be reused")),
    )
    second = layer.attach_daily_base_factors(daily, "000001.SZ")

    pd.testing.assert_series_equal(first["kdj_d_j"], second["kdj_d_j"], check_names=False)
    stored = pd.concat([pd.read_parquet(path) for path in cache_paths], ignore_index=True)
    assert not stored.duplicated(["symbol", "date"]).any()
    assert stored["factor_version"].eq(layer.FACTOR_LAYER_VERSION).all()


def test_in_memory_factor_attach_does_not_create_symbol_cache(monkeypatch, tmp_path) -> None:
    daily = _daily()
    factor_root = tmp_path / "factors"
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(factor_root))

    result = layer.attach_daily_base_factors(
        daily,
        "000001.SZ",
        persist_missing=False,
    )

    assert "kdj_d_j" in result.columns
    assert not factor_root.exists()


def test_z_finite_window_factors_match_existing_formulas() -> None:
    daily = layer._prepare_daily(_daily(), "000001.SZ")
    actual = layer.calculate_z_base_factors(daily)
    price = layer.build_continuous_ohlc(daily)
    close = price["close"]
    volume = daily["volume"]

    expected_ratio_5 = volume / volume.shift(1).rolling(5, min_periods=1).mean()
    expected_bbi = (
        close.rolling(3, min_periods=1).mean()
        + close.rolling(6, min_periods=2).mean()
        + close.rolling(12, min_periods=4).mean()
        + close.rolling(24, min_periods=8).mean()
    ) / 4

    pd.testing.assert_series_equal(actual["z_vol_ratio_5"], expected_ratio_5, check_names=False)
    pd.testing.assert_series_equal(actual["z_bbi"], expected_bbi, check_names=False)


def test_incremental_refresh_replaces_same_dates_without_duplicates(tmp_path) -> None:
    daily_path = tmp_path / "000001.SZ.parquet"
    factor_root = tmp_path / "factor-cache"
    daily = _daily()
    daily.to_parquet(daily_path, index=False)
    start = daily["date"].iloc[-5]

    layer.refresh_symbol_factor_cache(daily_path, factor_root, start)
    layer.refresh_symbol_factor_cache(daily_path, factor_root, start)

    stored = layer.load_daily_base_factors("000001.SZ", factor_root, start_date=start)
    assert len(stored) == 5
    assert not stored.duplicated(["symbol", "date"]).any()


def _assert_signal_factors_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    columns = layer.SIGNAL_FACTOR_COLUMNS
    pd.testing.assert_frame_equal(
        actual[columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_online_ewm_state_matches_pandas_across_missing_values() -> None:
    values = pd.Series([np.nan, 10.0, 11.0, np.nan, 9.0, 12.0])
    actual, state = layer._ewm_adjust_false(values, alpha=1 / 3)
    expected = values.ewm(alpha=1 / 3, adjust=False).mean()
    pd.testing.assert_series_equal(actual, expected)

    next_value, _ = layer._ewm_adjust_false_step(state, 13.0, alpha=1 / 3)
    expected_next = pd.concat([values, pd.Series([13.0])]).ewm(
        alpha=1 / 3,
        adjust=False,
    ).mean().iloc[-1]
    assert next_value == expected_next


def test_signal_factor_subset_matches_legacy_formulas() -> None:
    daily = _daily()
    legacy = layer.calculate_daily_base_factors(daily, "000001.SZ")
    optimized = layer.calculate_daily_signal_factors(daily, "000001.SZ")

    _assert_signal_factors_equal(optimized, legacy)


def test_signal_factor_state_appends_one_day_without_recomputing(
    monkeypatch,
    tmp_path,
) -> None:
    daily = _daily()
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(tmp_path / "factors"))

    first = layer.attach_daily_signal_factors(
        daily.iloc[:-1],
        "000001.SZ",
    )
    assert first.attrs["signal_factor_cache_mode"] == "bootstrap"

    actual = layer.attach_daily_signal_factors(daily, "000001.SZ")
    expected = layer.attach_daily_signal_factors(
        daily,
        "000001.SZ",
        persist_missing=False,
    )

    assert actual.attrs["signal_factor_cache_mode"] == "incremental"
    _assert_signal_factors_equal(actual, expected)


def test_signal_factor_state_rescales_across_corporate_action(
    monkeypatch,
    tmp_path,
) -> None:
    daily = _daily()
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(tmp_path / "factors"))
    layer.attach_daily_signal_factors(daily.iloc[:-1], "000001.SZ")

    adjusted = daily.copy()
    last = adjusted.index[-1]
    for column in ["open", "high", "low", "close"]:
        adjusted.loc[last, column] *= 0.5
    adjusted.loc[last, "pre_close"] = adjusted.loc[last - 1, "close"] * 0.5

    actual = layer.attach_daily_signal_factors(adjusted, "000001.SZ")
    expected = layer.attach_daily_signal_factors(
        adjusted,
        "000001.SZ",
        persist_missing=False,
    )

    assert actual.attrs["signal_factor_cache_mode"] == "incremental"
    _assert_signal_factors_equal(actual, expected)


def test_signal_factor_state_rebuilds_after_historical_revision(
    monkeypatch,
    tmp_path,
) -> None:
    daily = _daily()
    monkeypatch.setenv("DAILY_FACTOR_ROOT", str(tmp_path / "factors"))
    layer.attach_daily_signal_factors(daily.iloc[:-1], "000001.SZ")

    revised = daily.copy()
    revised.loc[revised.index[-30], "volume"] *= 1.2
    revised.loc[revised.index[-30], "vol"] *= 1.2
    actual = layer.attach_daily_signal_factors(revised, "000001.SZ")
    expected = layer.attach_daily_signal_factors(
        revised,
        "000001.SZ",
        persist_missing=False,
    )

    assert actual.attrs["signal_factor_cache_mode"] == "invalidated_rebuild"
    _assert_signal_factors_equal(actual, expected)
