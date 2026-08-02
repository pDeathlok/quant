from pathlib import Path
from types import SimpleNamespace

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


def test_main_uses_default_a_share_execution_policy(monkeypatch, capsys) -> None:
    from quant.backtest import AShareExecutionConfig

    captured: dict[str, object] = {}
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-07-27", periods=2, freq="D"),
            "symbol": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.1],
        }
    )

    class FakeEngine:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def run(self, **kwargs: object) -> None:
            captured["run"] = kwargs

    monkeypatch.setattr(
        main,
        "_load_symbol_backtest_data",
        lambda *args: (data, "test-source"),
    )
    monkeypatch.setattr(main, "BacktestEngine", FakeEngine)
    monkeypatch.setattr(
        main,
        "STRATEGIES",
        {"momentum": lambda: SimpleNamespace(name="MomentumStrategy")},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "backtest",
            "--strategy",
            "momentum",
            "--output",
            "report.html",
        ],
    )

    main.main()

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert init_kwargs["execution_config"] == AShareExecutionConfig()
    assert captured["run"] == {"report_filename": "report.html"}
    assert "回测报告已生成" in capsys.readouterr().out
