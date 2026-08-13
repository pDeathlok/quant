from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.data.tushare_fetcher import TushareDataFetcher, validate_daily_basic_frame
from quant.routine import daily_basic_refresh


def _complete_daily_basic(trade_date: str = "20260722") -> pd.DataFrame:
    rows = 4
    return pd.DataFrame(
        {
            "ts_code": [f"00000{index}.SZ" for index in range(1, rows + 1)],
            "trade_date": [trade_date] * rows,
            "turnover_rate": [1.0] * rows,
            "turnover_rate_f": [1.2] * rows,
            "volume_ratio": [0.9] * rows,
            "pe": [10.0] * rows,
            "pe_ttm": [11.0] * rows,
            "pb": [1.1] * rows,
            "ps": [1.2] * rows,
            "ps_ttm": [1.3] * rows,
            "dv_ratio": [0.5, 0.4, None, None],
            "dv_ttm": [0.6, 0.5, None, None],
            "total_share": [100.0] * rows,
            "float_share": [80.0] * rows,
            "free_share": [70.0] * rows,
            "total_mv": [1000.0] * rows,
            "circ_mv": [800.0] * rows,
        }
    )


def test_validate_daily_basic_rejects_empty_and_wrong_date() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_daily_basic_frame(pd.DataFrame(), "20260722")

    wrong_date = pd.DataFrame(
        {"ts_code": ["000001.SZ"], "trade_date": ["20260721"]}
    )
    with pytest.raises(ValueError, match="trade_date mismatch"):
        validate_daily_basic_frame(wrong_date, "20260722")


def test_get_daily_basic_does_not_keep_empty_cache(monkeypatch, tmp_path: Path) -> None:
    cache_path = tmp_path / "tushare_daily_basic_20260722.parquet"
    pd.DataFrame().to_parquet(cache_path, index=False)

    class EmptyPro:
        def daily_basic(self, **kwargs):
            return pd.DataFrame(columns=["ts_code", "trade_date"])

    fetcher = TushareDataFetcher.__new__(TushareDataFetcher)
    fetcher.cache_dir = tmp_path
    fetcher._memory_cache = {}
    fetcher.pro = EmptyPro()

    with pytest.raises(ValueError, match="returned 0 rows"):
        fetcher.get_daily_basic("20260722")

    assert not cache_path.exists()


def test_validate_daily_basic_rejects_model_feature_columns_with_no_current_values() -> None:
    partial = _complete_daily_basic()
    partial["volume_ratio"] = None

    with pytest.raises(ValueError, match="feature coverage"):
        validate_daily_basic_frame(
            partial,
            "20260722",
            required_feature_coverage=daily_basic_refresh.DAILY_BASIC_FEATURE_COVERAGE,
        )


def test_validate_daily_basic_reports_required_feature_coverage() -> None:
    frame = validate_daily_basic_frame(
        _complete_daily_basic(),
        "20260722",
        required_feature_coverage=daily_basic_refresh.DAILY_BASIC_FEATURE_COVERAGE,
    )

    assert frame.attrs["feature_coverage"]["volume_ratio"] == 1.0
    assert frame.attrs["feature_coverage"]["dv_ratio"] == 0.5


def test_fetch_one_trade_date_preserves_last_good_file_on_empty_response(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    output_path = output_dir / "20260722.parquet"
    old = pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": ["20260722"]})
    old.to_parquet(output_path, index=False)

    class EmptyFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return pd.DataFrame(columns=["ts_code", "trade_date"])

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", EmptyFetcher)

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=1,
        retry_base_delay=0,
        retry_max_delay=0,
    )

    assert result["status"] == "failed"
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), old)


def test_latest_daily_basic_waits_for_complete_model_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    responses = [_complete_daily_basic(), _complete_daily_basic()]
    responses[0]["free_share"] = None

    class DelayedCompleteFetcher:
        def __init__(self, **kwargs):
            pass

        def get_daily_basic(self, trade_date: str) -> pd.DataFrame:
            return responses.pop(0)

    monkeypatch.setattr(
        daily_basic_refresh,
        "TushareDataFetcher",
        DelayedCompleteFetcher,
    )
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
        availability_retry_failures=1,
        availability_retry_interval=0,
    )

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert result["feature_coverage"]["free_share"] == 1.0


def test_complete_local_daily_basic_is_validated_without_refetching(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "daily_basic"
    output_dir.mkdir()
    _complete_daily_basic().to_parquet(output_dir / "20260722.parquet", index=False)

    class RefetchMustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("complete local daily_basic should not be refetched")

    monkeypatch.setattr(daily_basic_refresh, "TushareDataFetcher", RefetchMustNotRun)

    result = daily_basic_refresh.fetch_one_trade_date(
        "20260722",
        output_dir,
        tmp_path / "cache",
        daily_basic_refresh.RequestLimiter(0),
        retries=0,
        retry_base_delay=0,
        retry_max_delay=0,
        expected_rows=4,
        minimum_coverage_rate=1.0,
    )

    assert result["status"] == "success"
    assert result["source"] == "local_validated"
    assert result["attempts"] == 0
