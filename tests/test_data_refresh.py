import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from quant.data.source_merge import DailyRefreshAudit, audits_to_frame
from quant.routine import data_refresh


def test_retryable_error_classification():
    assert data_refresh._is_retryable_error("每分钟最多访问该接口800次")
    assert data_refresh._is_retryable_error("Connection reset by peer")
    assert not data_refresh._is_retryable_error("未获取到 000004.SZ 的数据")


def test_adjusted_incremental_refresh_restarts_from_first_stored_date():
    existing = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103", "20240104"],
            "close": [10.0, 10.2, 10.1],
        }
    )

    assert data_refresh._symbol_refresh_start(existing, "20240104", adjust="qfq") == "20240102"
    assert data_refresh._symbol_refresh_start(existing, "20240104", adjust=None) == "20240104"


def test_completed_market_end_date_uses_previous_day_before_data_ready(monkeypatch):
    monkeypatch.delenv("ROUTINE_MARKET_DATA_READY_TIME", raising=False)

    before_close, adjusted = data_refresh._completed_market_end_date(
        "20260730",
        now=datetime(2026, 7, 30, 9, 38),
    )
    after_ready, after_ready_adjusted = data_refresh._completed_market_end_date(
        "20260730",
        now=datetime(2026, 7, 30, 16, 1),
    )
    historical, historical_adjusted = data_refresh._completed_market_end_date(
        "20260729",
        now=datetime(2026, 7, 30, 9, 38),
    )

    assert (before_close, adjusted) == ("20260729", True)
    assert (after_ready, after_ready_adjusted) == ("20260730", False)
    assert (historical, historical_adjusted) == ("20260729", False)


def test_refresh_one_symbol_retries_then_succeeds(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_refresh_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("每分钟最多访问该接口800次")
        return DailyRefreshAudit(
            symbol="000001.SZ",
            source="tushare",
            rows=1,
            merged_rows=1,
            status="tushare_daily",
        )

    monkeypatch.setattr(data_refresh, "_refresh_one_symbol_once", fake_refresh_once)
    monkeypatch.setattr(data_refresh.time, "sleep", lambda seconds: None)

    audit = data_refresh.refresh_one_symbol(
        symbol="000001.SZ",
        name="平安银行",
        start_date="20260605",
        end_date="20260606",
        adjust=None,
        output_dir=tmp_path,
        cache_dir=tmp_path,
        limiter=data_refresh.RequestLimiter(0),
        retries=2,
        retry_base_delay=0,
        retry_max_delay=0,
        retry_jitter=0,
    )

    assert audit.status == "tushare_daily"
    assert audit.attempts == 2
    assert calls["count"] == 2


def test_audit_frame_includes_attempts():
    audit = DailyRefreshAudit(
        symbol="000001.SZ",
        source="tushare",
        rows=1,
        merged_rows=1,
        status="tushare_daily",
        attempts=3,
    )

    frame = audits_to_frame([audit])

    assert list(frame["attempts"]) == [3]
    assert isinstance(frame, pd.DataFrame)


def test_refresh_daily_data_batches_raw_market_by_trade_date(monkeypatch, tmp_path):
    class DummyPro:
        def __init__(self):
            self.daily_calls = []

        def trade_cal(self, **kwargs):
            return pd.DataFrame({"cal_date": ["20260605", "20260606"]})

        def daily(self, **kwargs):
            self.daily_calls.append(kwargs)
            trade_date = kwargs["trade_date"]
            rows = [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.8,
                    "close": 10.2,
                    "vol": 100.0,
                },
                {
                    "ts_code": "600519.SH",
                    "trade_date": trade_date,
                    "open": 1500.0,
                    "high": 1510.0,
                    "low": 1490.0,
                    "close": 1505.0,
                    "vol": 20.0,
                },
            ]
            return pd.DataFrame(rows)

    dummy_pro = DummyPro()

    class DummyTushareFetcher:
        def __init__(self, *args, **kwargs):
            self.pro = dummy_pro

    def fake_load_symbols(fetcher, board="all", limit=None):
        return [("000001.SZ", "平安银行"), ("600519.SH", "贵州茅台")]

    output_dir = tmp_path / "daily"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20260604"],
            "date": [pd.Timestamp("2026-06-04")],
            "close": [10.0],
        }
    ).to_parquet(output_dir / "000001.SZ.parquet", index=False)
    initial_store = data_refresh.MarketDataStore(
        data_refresh.MarketDataStoreConfig(backend="parquet", root=tmp_path)
    )
    initial_store.write_market_batch(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20260604"],
                "date": [pd.Timestamp("2026-06-04")],
                "close": [10.0],
            }
        )
    )

    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setattr(data_refresh, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(data_refresh, "TushareDataFetcher", DummyTushareFetcher)
    monkeypatch.setattr(data_refresh, "load_tushare_symbols", fake_load_symbols)

    manifest = data_refresh.refresh_daily_data(
        start_date="20260605",
        end_date="20260606",
        output_dir=output_dir,
        workers=1,
        sleep_between=0,
        retries=0,
        final_retry_rounds=0,
    )

    assert dummy_pro.daily_calls == [
        {"trade_date": "20260605"},
        {"trade_date": "20260606"},
    ]
    assert manifest["refresh_mode"] == "batch_by_trade_date"
    assert manifest["market_daily_requests"] == 2
    assert manifest["failed"] == 0
    store = data_refresh.MarketDataStore(
        data_refresh.MarketDataStoreConfig(backend="parquet", root=tmp_path)
    )
    ping_an = store.read_frame("daily", "000001.SZ")
    mao_tai = store.read_frame("daily", "600519.SH")
    assert ping_an["trade_date"].tolist() == ["20260604", "20260605", "20260606"]
    assert mao_tai["trade_date"].tolist() == ["20260605", "20260606"]
    assert pd.read_parquet(output_dir / "000001.SZ.parquet")["trade_date"].tolist() == ["20260604"]
    assert manifest["batch_storage"] == {
        "rows": 4,
        "sql_rows": 0,
        "parquet_partitions": 1,
        "table": "market_daily",
        "coverage": {
                "minimum_rate": 0.995,
            "expected_symbols": 2,
            "trade_dates": {
                "20260605": {"symbols": 2, "missing_symbols": 0, "coverage_rate": 1.0},
                "20260606": {"symbols": 2, "missing_symbols": 0, "coverage_rate": 1.0},
            },
        },
    }
    assert sorted((tmp_path / "daily_partitioned").glob("year_month=*/data.parquet"))


def test_market_daily_batch_rejects_possible_row_limit_truncation(monkeypatch):
    class DummyPro:
        def daily(self, **kwargs):
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "trade_date": [kwargs["trade_date"], kwargs["trade_date"]],
                }
            )

    class DummyFetcher:
        pro = DummyPro()

    monkeypatch.setattr(data_refresh, "TUSHARE_DAILY_ROW_LIMIT", 2)

    with pytest.raises(RuntimeError, match="may have truncated"):
        data_refresh._fetch_market_daily_with_retries(
            DummyFetcher(),
            "20260605",
            data_refresh.RequestLimiter(0),
            retries=0,
            retry_base_delay=0,
            retry_max_delay=0,
            retry_jitter=0,
        )


def test_market_daily_batch_rejects_incomplete_symbol_coverage() -> None:
    market = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": ["20260605", "20260605", "20260605"],
        }
    )
    expected = {f"00000{index}.SZ" for index in range(1, 6)}

    with pytest.raises(data_refresh.MarketBatchCoverageError, match="3/5"):
        data_refresh._validate_market_batch_coverage(
            market,
            expected,
            ["20260605"],
            minimum_rate=0.8,
        )


def test_refresh_daily_data_writes_failed_symbols(monkeypatch, tmp_path):
    class DummyTushareFetcher:
        def __init__(self, *args, **kwargs):
            pass

    def fake_load_symbols(fetcher, board="all", limit=None):
        return [("000004.SZ", "国华网安")]

    def fake_refresh_one_symbol(*args, **kwargs):
        return DailyRefreshAudit(
            symbol="000004.SZ",
            source="tushare",
            rows=0,
            merged_rows=0,
            status="failed",
            error="未获取到 000004.SZ 的数据",
            attempts=1,
        )

    monkeypatch.setattr(data_refresh, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(data_refresh, "TushareDataFetcher", DummyTushareFetcher)
    monkeypatch.setattr(data_refresh, "load_tushare_symbols", fake_load_symbols)
    monkeypatch.setattr(data_refresh, "refresh_one_symbol", fake_refresh_one_symbol)

    manifest = data_refresh.refresh_daily_data(
        start_date="20260605",
        end_date="20260606",
        adjust="qfq",
        output_dir=tmp_path / "daily",
        workers=1,
        sleep_between=0,
        retries=3,
    )

    failed = pd.read_csv(manifest["failed_symbols_path"])
    assert manifest["failed"] == 1
    assert manifest["retries"] == 3
    assert manifest["retried_symbols"] == 0
    assert list(failed["symbol"]) == ["000004.SZ"]
