from pathlib import Path

import pandas as pd

import main
from quant.data import MarketDataStore, MarketDataStoreConfig


def test_load_backtest_data_filters_dates_and_normalizes_symbol(tmp_path: Path) -> None:
    path = tmp_path / "daily.parquet"
    pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1000, 1100, 1200],
        }
    ).to_parquet(path, index=False)

    loaded = main._load_backtest_data(path, "000001", "20240103", "20240104")

    assert loaded["trade_date"].tolist() == ["20240103", "20240104"]
    assert loaded["symbol"].unique().tolist() == ["000001.SZ"]


def test_default_backtest_loader_prefers_canonical_partitioned_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MARKET_DATA_ROOT", str(raw_root))
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=raw_root)
    ).write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["920826.BJ", "920826.BJ"],
                "trade_date": ["20260728", "20260729"],
                "open": [10.0, 10.2],
                "high": [10.3, 10.5],
                "low": [9.9, 10.1],
                "close": [10.2, 10.4],
                "volume": [1000, 1200],
            }
        )
    )

    loaded, source = main._load_symbol_backtest_data(
        "920826",
        "20260728",
        "20260729",
    )

    assert loaded["trade_date"].tolist() == ["20260728", "20260729"]
    assert loaded["symbol"].unique().tolist() == ["920826.BJ"]
    assert source == "canonical:daily/920826.BJ"
