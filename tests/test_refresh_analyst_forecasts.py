import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/research/refresh_analyst_forecasts.py"
SPEC = importlib.util.spec_from_file_location("refresh_analyst_forecasts_test_module", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _configure_paths(monkeypatch, tmp_path) -> Path:
    output = tmp_path / "analyst_forecasts.parquet"
    monkeypatch.setattr(MODULE, "OUTPUT_PATH", output)
    monkeypatch.setattr(MODULE, "OUTPUT_LOCK_PATH", tmp_path / "analyst_forecasts.lock")
    monkeypatch.setattr(MODULE, "AUDIT_ROOT", tmp_path / "source_audit")
    return output


def test_atomic_merge_preserves_last_good_file_on_validation_failure(monkeypatch, tmp_path) -> None:
    output = _configure_paths(monkeypatch, tmp_path)
    good = pd.DataFrame(
        {
            "source": ["akshare_em_research"],
            "ts_code": ["000001.SZ"],
            "report_date": [pd.Timestamp("2026-07-21")],
            "org_name": ["机构甲"],
            "author_name": [None],
            "forecast_year": [2026],
        }
    )
    MODULE.append_and_dedupe(good)
    before = output.read_bytes()

    with pytest.raises(ValueError, match="missing required columns"):
        MODULE.append_and_dedupe(pd.DataFrame({"source": ["broken"]}))

    assert output.read_bytes() == before


def test_research_circuit_breaker_defers_remaining_symbols(monkeypatch, tmp_path) -> None:
    _configure_paths(monkeypatch, tmp_path)
    calls = []

    def always_fail(*, symbol):
        calls.append(symbol)
        raise ConnectionError("upstream unavailable")

    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(stock_research_report_em=always_fail))
    monkeypatch.setattr(MODULE.time, "sleep", lambda *_: None)
    monkeypatch.setattr(MODULE.random, "uniform", lambda *_: 0.0)

    result = MODULE.refresh_akshare_em_research(
        None,
        0.1,
        1,
        symbols=["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        refresh_existing=True,
        circuit_breaker_failures=2,
    )

    assert result["failed"] == 2
    assert result["deferred"] == 2
    assert result["failed_symbols"] == ["000001.SZ", "000002.SZ"]
    assert result["deferred_symbols"] == ["000003.SZ", "000004.SZ"]
    assert result["circuit_open"] is True
    assert len(calls) == 4  # two attempts for each of the first two symbols
    assert Path(result["audit_path"]).is_file()


def test_output_lock_records_owner_and_replaces_stale_empty_lock(monkeypatch, tmp_path) -> None:
    _configure_paths(monkeypatch, tmp_path)
    lock_path = MODULE.OUTPUT_LOCK_PATH
    lock_path.write_text("", encoding="utf-8")
    old_time = pd.Timestamp("2026-07-21 09:30").timestamp()
    os.utime(lock_path, (old_time, old_time))
    monkeypatch.setenv("ANALYST_FORECAST_LOCK_STALE_SECONDS", "60")

    with MODULE.output_lock():
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["stale_lock_replaced"]["reason"] == "stale lock file without a running owner"

    finished = json.loads(lock_path.read_text(encoding="utf-8"))
    assert finished["pid"] == os.getpid()
    assert "finished_at" in finished


def test_snapshot_retries_and_commits_only_valid_data(monkeypatch, tmp_path) -> None:
    output = _configure_paths(monkeypatch, tmp_path)
    calls = 0

    def flaky_snapshot(*, symbol):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("temporary failure")
        return pd.DataFrame(
            {
                "代码": ["000001"],
                "名称": ["平安银行"],
                "研报数": [12],
                "机构投资评级(近六个月)-买入": [3],
                "机构投资评级(近六个月)-增持": [2],
                "2026预测每股收益": [1.23],
            }
        )

    monkeypatch.setitem(sys.modules, "akshare", types.SimpleNamespace(stock_profit_forecast_em=flaky_snapshot))
    monkeypatch.setattr(MODULE.time, "sleep", lambda *_: None)
    monkeypatch.setattr(MODULE.random, "uniform", lambda *_: 0.0)

    result = MODULE.refresh_akshare_em_snapshot(sleep_seconds=0.1, retries=2)

    assert result["attempts"] == 2
    assert result["new_rows"] == 1
    assert output.is_file()
    saved = pd.read_parquet(output)
    assert saved.loc[0, "source"] == "akshare_em_snapshot"
