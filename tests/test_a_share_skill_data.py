from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.research.a_share_skill_data import (
    SkillMarketDataError,
    build_index_price_volume_payload,
    build_stock_price_volume_payload,
    normalize_a_share_ticker,
)


class FakeMarketStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.config = SimpleNamespace(backend="parquet", mirror_parquet=True)

    def read_market_range(self, *args, **kwargs) -> pd.DataFrame:
        return self.frame.copy()


def make_daily_frame(rows: int = 130, ticker: str = "600519.SH") -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=rows)
    values = []
    for index, trade_date in enumerate(dates):
        close = 100 + index * 0.2
        values.append(
            {
                "ts_code": ticker,
                "trade_date": trade_date.strftime("%Y%m%d"),
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "vol": 1000 + index,
                "volume": 999999,
            }
        )
    return pd.DataFrame(values)


def test_stock_adapter_reuses_canonical_fields_and_tushare_lot_unit() -> None:
    frame = make_daily_frame()
    last_date = pd.to_datetime(frame.iloc[-1]["trade_date"]).date().isoformat()

    payload = build_stock_price_volume_payload(
        "600519",
        f"{last_date}T16:00:00+08:00",
        store=FakeMarketStore(frame),
        bar_count=120,
    )

    assert payload["ticker"] == "600519.SH"
    assert payload["price_basis"] == "unadjusted"
    assert payload["volume_unit"] == "lots (手)"
    assert payload["bars"][-1]["volume"] == float(frame.iloc[-1]["vol"])
    assert payload["last_bar_available_at"] == f"{last_date}T16:00:00+08:00"
    assert len(payload["bars"]) == 120


def test_stock_adapter_excludes_same_day_bar_before_conservative_publish_time() -> None:
    frame = make_daily_frame(rows=70)
    final_date = pd.to_datetime(frame.iloc[-1]["trade_date"]).date()
    previous_date = pd.to_datetime(frame.iloc[-2]["trade_date"]).date().isoformat()

    payload = build_stock_price_volume_payload(
        "600519.SH",
        f"{final_date.isoformat()}T15:59:59+08:00",
        store=FakeMarketStore(frame),
        bar_count=70,
    )

    assert payload["bars"][-1]["date"] == previous_date
    assert len(payload["bars"]) == 69


def test_stock_adapter_rejects_insufficient_complete_history() -> None:
    frame = make_daily_frame(rows=59)
    final_date = pd.to_datetime(frame.iloc[-1]["trade_date"]).date().isoformat()

    with pytest.raises(SkillMarketDataError, match="at least 60"):
        build_stock_price_volume_payload(
            "600519.SH",
            f"{final_date}T18:00:00+08:00",
            store=FakeMarketStore(frame),
        )


def test_nine_prefix_bare_code_normalizes_to_beijing_exchange() -> None:
    assert normalize_a_share_ticker("920000") == "920000.BJ"
    fetcher = TushareDataFetcher.__new__(TushareDataFetcher)
    assert fetcher._normalize_symbol("920000") == "920000.BJ"


def test_index_adapter_reads_existing_reference_file(tmp_path: Path) -> None:
    frame = make_daily_frame(ticker="000300.SH").drop(columns=["volume"])
    path = tmp_path / "index_000300.SH.parquet"
    frame.to_parquet(path, index=False)
    final_date = pd.to_datetime(frame.iloc[-1]["trade_date"]).date().isoformat()

    payload = build_index_price_volume_payload(
        "000300.SH",
        f"{final_date}T18:00:00+08:00",
        path=path,
        bar_count=120,
    )

    assert payload["ticker"] == "000300.SH"
    assert payload["provenance"]["interface"] == "index_daily"
    assert len(payload["bars"]) == 120
