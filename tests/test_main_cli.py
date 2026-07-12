from pathlib import Path

import pandas as pd

import main


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
