from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.data.tushare_fetcher import TushareDataFetcher, validate_daily_basic_frame
from quant.routine import daily_basic_refresh


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
