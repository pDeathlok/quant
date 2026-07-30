from __future__ import annotations

from datetime import datetime

import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.features.market_sentiment import normalize_ts_code
from quant.webapp import services


def test_watch_symbol_normalization_reads_bse_symbol_from_canonical_store(
    monkeypatch,
    tmp_path,
) -> None:
    raw_root = tmp_path / "raw"
    daily_dir = raw_root / "daily"
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MARKET_DATA_ROOT", str(raw_root))
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.setattr(services, "DAILY_DIR", daily_dir)
    MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=raw_root)
    ).write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["920826.BJ"],
                "trade_date": ["20260729"],
                "close": [20.0],
            }
        )
    )

    assert services._normalize_watch_symbol("920826") == "920826.BJ"
    assert normalize_ts_code("920826") == "920826.BJ"


def test_missing_similar_pattern_source_date_is_not_treated_as_current() -> None:
    assert (
        services._similar_pattern_source_date_is_current(
            None,
            datetime(2026, 7, 24, 15, 1),
        )
        is False
    )


def test_latest_similar_pattern_date_uses_configured_market_store(
    monkeypatch,
    tmp_path,
) -> None:
    raw_root = tmp_path / "raw"
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MARKET_DATA_ROOT", str(raw_root))
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.setattr(services, "DAILY_DIR", raw_root / "daily")
    MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=raw_root)
    ).write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "920826.BJ"],
                "trade_date": ["20260728", "20260729"],
                "close": [10.0, 20.0],
            }
        )
    )

    latest = services._latest_similar_pattern_target_date(
        ["000001.SZ", "920826.BJ"]
    )

    assert latest == "2026-07-29"
