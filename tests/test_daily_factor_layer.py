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
