import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from quant.data.source_merge import (
    DailyRefreshAudit,
    audits_to_frame,
    normalize_tushare_daily,
)
from quant.routine import data_refresh


@pytest.mark.parametrize("date", ["2026-08-20", "20260820", "2026/08/20"])
def test_tushare_daily_date_normalization_preserves_supported_strings(
    date: str,
) -> None:
    daily = pd.DataFrame(
        {
            "date": pd.Series([date], dtype="string"),
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [100.0],
        }
    )

    normalized = normalize_tushare_daily(daily, "000001.SZ")

    assert normalized.loc[0, "trade_date"] == "20260820"
    assert normalized.loc[0, "date"] == pd.Timestamp("2026-08-20")


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


def test_market_end_date_is_not_capped_by_an_intraday_clock_time():
    before_close = data_refresh._bounded_market_end_date(
        "20260813",
        now=datetime(2026, 8, 13, 13, 0),
    )
    after_close = data_refresh._bounded_market_end_date(
        "20260813",
        now=datetime(2026, 8, 13, 16, 0),
    )
    historical = data_refresh._bounded_market_end_date(
        "20260812",
        now=datetime(2026, 8, 13, 13, 0),
    )
    future = data_refresh._bounded_market_end_date(
        "20260814",
        now=datetime(2026, 8, 13, 16, 0),
    )

    assert before_close == ("20260813", False)
    assert after_close == ("20260813", False)
    assert historical == ("20260812", False)
    assert future == ("20260813", True)


def test_refresh_daily_data_probes_current_open_session_before_close(monkeypatch):
    real_datetime = datetime

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = real_datetime(
                2026,
                8,
                13,
                13,
                0,
                tzinfo=data_refresh.MARKET_TIMEZONE,
            )
            return current if tz is None else current.astimezone(tz)

    class DummyTushareFetcher:
        def __init__(self, *args, **kwargs):
            pass

    class ProbeReached(RuntimeError):
        pass

    calendar_ranges: list[tuple[str, str]] = []
    probed_dates: list[str] = []

    def fake_open_trade_dates(fetcher, start_date, end_date):
        calendar_ranges.append((start_date, end_date))
        return {"20260812", "20260813"}

    def stop_at_availability_probe(fetcher, trade_date, symbols, **kwargs):
        probed_dates.append(trade_date)
        raise ProbeReached

    monkeypatch.setattr(data_refresh, "datetime", FrozenDateTime)
    monkeypatch.setattr(data_refresh, "TushareDataFetcher", DummyTushareFetcher)
    monkeypatch.setattr(
        data_refresh,
        "load_tushare_symbols",
        lambda *args, **kwargs: [("000001.SZ", "平安银行")],
    )
    monkeypatch.setattr(data_refresh, "_open_trade_dates", fake_open_trade_dates)
    monkeypatch.setattr(
        data_refresh,
        "_wait_for_market_daily_availability",
        stop_at_availability_probe,
    )

    with pytest.raises(ProbeReached):
        data_refresh.refresh_daily_data(start_date="20260812", sleep_between=0)

    assert calendar_ranges == [("20260812", "20260813")]
    assert probed_dates == ["20260813"]


def test_market_daily_availability_allows_twelve_failures_before_final_probe():
    class DummyPro:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            if self.calls <= 12:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": kwargs["trade_date"],
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "vol": 100.0,
                    }
                ]
            )

    class DummyFetcher:
        pro = DummyPro()

    sleep_calls: list[float] = []
    frame, probe = data_refresh._wait_for_market_daily_availability(
        DummyFetcher(),
        "20260803",
        [("000001.SZ", "平安银行")],
        sleep_between=0,
        retry_failures=12,
        retry_interval_seconds=300,
        sleep_fn=sleep_calls.append,
    )

    assert not frame.empty
    assert probe["attempts"] == 13
    assert probe["failed_attempts"] == 12
    assert sleep_calls == [300] * 12


def test_market_daily_availability_fails_only_after_thirteenth_failed_probe():
    class DummyPro:
        def __init__(self):
            self.calls = 0

        def daily(self, **kwargs):
            self.calls += 1
            return pd.DataFrame()

    class DummyFetcher:
        pro = DummyPro()

    sleep_calls: list[float] = []
    with pytest.raises(data_refresh.MarketDataNotReadyError, match="after 13 probes"):
        data_refresh._wait_for_market_daily_availability(
            DummyFetcher(),
            "20260803",
            [("000001.SZ", "平安银行")],
            sleep_between=0,
            retry_failures=12,
            retry_interval_seconds=300,
            sleep_fn=sleep_calls.append,
        )

    assert DummyFetcher.pro.calls == 13
    assert sleep_calls == [300] * 12


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
        {"trade_date": "20260606"},
        {"trade_date": "20260605"},
    ]
    assert manifest["refresh_mode"] == "batch_by_trade_date"
    assert manifest["end_date_capped_to_today"] is False
    assert manifest["availability_policy"] == "probe_latest_open_trade_date_until_available"
    assert "market_data_ready_time" not in manifest
    assert manifest["availability_probe"]["trade_date"] == "20260606"
    assert manifest["availability_probe"]["attempts"] == 1
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


def test_refresh_daily_data_does_not_generate_previous_date_when_latest_is_unavailable(
    monkeypatch,
    tmp_path,
):
    class DummyPro:
        def __init__(self):
            self.daily_calls: list[dict[str, str]] = []

        def trade_cal(self, **kwargs):
            return pd.DataFrame({"cal_date": ["20260731", "20260803"]})

        def daily(self, **kwargs):
            self.daily_calls.append(kwargs)
            return pd.DataFrame()

    dummy_pro = DummyPro()

    class DummyTushareFetcher:
        def __init__(self, *args, **kwargs):
            self.pro = dummy_pro

    monkeypatch.setattr(data_refresh, "TushareDataFetcher", DummyTushareFetcher)
    monkeypatch.setattr(
        data_refresh,
        "load_tushare_symbols",
        lambda *args, **kwargs: [("000001.SZ", "平安银行")],
    )
    data_refresh._TRADE_CAL_CACHE.clear()

    with pytest.raises(data_refresh.MarketDataNotReadyError, match="20260803"):
        data_refresh.refresh_daily_data(
            start_date="20260731",
            end_date="20260803",
            output_dir=tmp_path / "daily",
            workers=1,
            sleep_between=0,
            retries=0,
            final_retry_rounds=0,
            availability_retry_failures=0,
            availability_retry_interval=0,
        )

    assert dummy_pro.daily_calls == [{"trade_date": "20260803"}]
    assert not (tmp_path / "daily_partitioned").exists()


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
    class DummyPro:
        def trade_cal(self, **kwargs):
            return pd.DataFrame({"cal_date": ["20260606"]})

        def daily(self, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000004.SZ",
                        "trade_date": kwargs["trade_date"],
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "vol": 100.0,
                    }
                ]
            )

    class DummyTushareFetcher:
        def __init__(self, *args, **kwargs):
            self.pro = DummyPro()

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
    data_refresh._TRADE_CAL_CACHE.clear()

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
