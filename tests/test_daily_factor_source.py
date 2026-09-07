from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant.data.market_data_store import MarketDataUnavailableError
from quant.features import daily_factor_layer as layer


@pytest.fixture
def local_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    daily_path = tmp_path / "raw" / "daily" / "000001.SZ.parquet"
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [1.0]}).to_parquet(
        daily_path, index=False
    )
    monkeypatch.setenv("MARKET_DATA_ROOT", str(daily_path.parent.parent))
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    return daily_path


def _canonical_store(monkeypatch: pytest.MonkeyPatch, backend: str, state: str) -> MagicMock:
    monkeypatch.setenv("MARKET_DATA_BACKEND", backend)
    store = MagicMock()
    store.config.backend = backend
    for method in (store.read_market_range, store.read_frame):
        if state == "outage":
            method.side_effect = MarketDataUnavailableError("Canonical SQL read unavailable")
        elif state == "empty":
            method.return_value = pd.DataFrame(columns=["ts_code", "trade_date", "close"])
        else:
            method.return_value = pd.DataFrame({"trade_date": ["20260904"], "close": [42.0]})
    monkeypatch.setattr(layer, "MarketDataStore", MagicMock(return_value=store))
    return store


@pytest.mark.parametrize("backend", ["mysql", "sql"])
@pytest.mark.parametrize("state", ["outage", "empty", "missing_symbol_column"])
def test_factor_refresh_never_uses_files_when_canonical_sql_is_unusable(
    local_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, state: str
) -> None:
    store = _canonical_store(monkeypatch, backend, state)
    worker = MagicMock(return_value={"date_max": "2026-09-04", "rows": 1})
    monkeypatch.setattr(layer, "refresh_symbol_factor_cache", worker)
    publish = MagicMock()
    monkeypatch.setattr(layer, "atomic_write_json", publish)

    with pytest.raises((RuntimeError, ValueError)) as caught:
        layer.refresh_daily_factor_layer(
            local_source.parent, factor_root=tmp_path / "factors", workers=1, executor_type="threads"
        )

    if state == "outage":
        assert caught.value is store.read_market_range.side_effect
    else:
        assert not isinstance(caught.value, MarketDataUnavailableError)
        assert "canonical SQL" in str(caught.value)
    worker.assert_not_called()
    publish.assert_not_called()
    assert not (tmp_path / "factors").exists()


@pytest.mark.parametrize("state", ["outage", "empty"])
def test_symbol_factor_worker_does_not_bypass_canonical_sql(
    local_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    store = _canonical_store(monkeypatch, "sql", state)
    file_read = MagicMock(side_effect=AssertionError("must not read unverified source parquet"))
    monkeypatch.setattr(pd, "read_parquet", file_read)

    with pytest.raises((RuntimeError, ValueError)) as caught:
        layer.refresh_symbol_factor_cache(local_source, tmp_path / "factors", "2026-09-04")

    if state == "outage":
        assert caught.value is store.read_frame.side_effect
    else:
        assert not isinstance(caught.value, MarketDataUnavailableError)
    store.read_frame.assert_called_once_with("daily", "000001.SZ")
    file_read.assert_not_called()
    assert not (tmp_path / "factors").exists()


def test_empty_supplied_factor_source_is_not_a_cache_hit(
    local_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = MagicMock(side_effect=AssertionError("empty source must fail before factor work"))
    monkeypatch.setattr(layer, "_prepare_daily", prepare)

    with pytest.raises(ValueError, match="No daily source rows"):
        layer.refresh_symbol_factor_cache(
            local_source, tmp_path / "factors", "2026-09-04", source_frame=pd.DataFrame()
        )

    prepare.assert_not_called()


def test_explicit_file_backend_can_use_legacy_factor_sources(
    local_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _canonical_store(monkeypatch, "file", "empty")
    worker = MagicMock(return_value={"date_max": "2026-09-04", "rows": 1})
    monkeypatch.setattr(layer, "refresh_symbol_factor_cache", worker)

    result = layer.refresh_daily_factor_layer(
        local_source.parent, factor_root=tmp_path / "factors", workers=1, executor_type="threads"
    )

    assert result["symbols"] == 1
    assert worker.call_args.args[0] == local_source
    assert worker.call_args.args[3] is None


def test_sql_factor_refresh_passes_canonical_frame_without_reading_same_date_mirror(
    local_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _canonical_store(monkeypatch, "sql", "empty")
    canonical = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260904"], "close": [42.0]})
    store.read_market_range.return_value = canonical
    worker = MagicMock(return_value={"date_max": "2026-09-04", "rows": 1})
    monkeypatch.setattr(layer, "refresh_symbol_factor_cache", worker)
    file_read = MagicMock(side_effect=AssertionError("source mirror must not be read"))
    monkeypatch.setattr(pd, "read_parquet", file_read)

    result = layer.refresh_daily_factor_layer(
        local_source.parent, factor_root=tmp_path / "factors", workers=1, executor_type="threads"
    )

    assert result["symbols"] == 1
    pd.testing.assert_frame_equal(worker.call_args.args[3], canonical)
    file_read.assert_not_called()
