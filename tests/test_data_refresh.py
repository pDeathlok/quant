import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from quant.data.source_merge import DailyRefreshAudit, audits_to_frame
from quant.routine import data_refresh


def test_retryable_error_classification():
    assert data_refresh._is_retryable_error("每分钟最多访问该接口800次")
    assert data_refresh._is_retryable_error("Connection reset by peer")
    assert not data_refresh._is_retryable_error("未获取到 000004.SZ 的数据")


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
