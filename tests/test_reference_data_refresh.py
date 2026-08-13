from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant.data.tushare_fetcher import TushareDataFetcher
from quant.routine.reference_data_refresh import (
    refresh_financial_periods,
    refresh_reference_data,
)


class FakePro:
    def trade_cal(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"cal_date": [kwargs["start_date"]], "is_open": ["1"]})

    def index_daily(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000300.SH", "000300.SH"],
                "trade_date": ["20260720", "20260721"],
                "close": [4500.0, 4510.0],
            }
        )

    def fina_indicator_vip(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {"ts_code": ["000001.SZ"], "ann_date": ["20260721"], "end_date": [kwargs["period"]], "roe": [12.0]}
        )

    def income_vip(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260721"],
                "end_date": [kwargs["period"]],
                "report_type": ["1"],
                "revenue": [100.0],
            }
        )

    def cashflow_vip(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20260721"],
                "end_date": [kwargs["period"]],
                "report_type": ["1"],
                "n_cashflow_act": [80.0],
            }
        )

    def stk_limit(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [kwargs["trade_date"]],
                "ts_code": ["000001.SZ"],
                "pre_close": [10.0],
                "up_limit": [11.0],
                "down_limit": [9.0],
            }
        )

    def suspend_d(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "suspend_type", "suspend_timing"]
        )

    def stock_st(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "name", "type", "type_name"]
        )

    def margin_detail(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [kwargs["trade_date"]],
                "ts_code": ["000001.SZ"],
                "rzye": [10.0],
                "rqye": [1.0],
                "rzrqye": [11.0],
            }
        )

    def moneyflow(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [kwargs["trade_date"]],
                "ts_code": ["000001.SZ"],
                "net_mf_amount": [2.0],
            }
        )

    def top_list(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": [kwargs["trade_date"]],
                "ts_code": ["000001.SZ"],
                "reason": ["test"],
            }
        )

    def stk_holdertrade(self, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": [kwargs["end_date"]],
                "holder_name": ["holder"],
                "in_de": ["IN"],
                "change_vol": [1.0],
                "avg_price": [10.0],
            }
        )


class FakeFetcher:
    def __init__(self) -> None:
        self.pro = FakePro()
        self.force_refresh = False

    def get_stock_basic(self, *, force_refresh: bool = False) -> pd.DataFrame:
        self.force_refresh = force_refresh
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],
                "name": ["平安银行"],
                "industry": ["银行"],
            }
        )


def test_reference_refresh_is_idempotent_and_updates_every_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    audit_root = raw_dir / "source_audit"
    fetcher = FakeFetcher()

    first = refresh_reference_data(
        "20260721",
        fetcher=fetcher,
        raw_dir=raw_dir,
        audit_root=audit_root,
        financial_periods=1,
    )
    second = refresh_reference_data(
        "20260721",
        fetcher=fetcher,
        raw_dir=raw_dir,
        audit_root=audit_root,
        financial_periods=1,
    )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert fetcher.force_refresh is True
    assert first["steps"]["stock_basic"]["polled_through"] == "20260721"
    assert first["steps"]["financials"]["polled_through"] == "20260721"
    assert (
        first["steps"]["long_factor_sources"]["datasets"]["top_list"][
            "polled_through"
        ]
        == "20260721"
    )
    assert len(pd.read_parquet(raw_dir / "stock_basic.parquet")) == 1
    assert len(pd.read_parquet(raw_dir / "index_000300.SH.parquet")) == 2
    assert len(pd.read_parquet(raw_dir / "fina_indicator.parquet")) == 1
    assert len(pd.read_parquet(raw_dir / "income.parquet")) == 1
    assert len(pd.read_parquet(raw_dir / "cashflow.parquet")) == 1
    tradability = pd.read_parquet(raw_dir / "tradability" / "20260721.parquet")
    assert tradability["ts_code"].tolist() == ["000001.SZ"]
    assert first["steps"]["tradability"]["coverage_rate"] == 1.0
    assert second["steps"]["tradability"]["status"] == "success"
    assert second["steps"]["long_factor_sources"]["status"] == "success"
    assert len(list((raw_dir / "moneyflow").glob("*.parquet"))) == 1
    assert len(pd.read_parquet(raw_dir / "holder_trade.parquet")) == 1
    assert not pd.read_parquet(raw_dir / "fina_indicator.parquet").duplicated(["ts_code", "ann_date", "end_date"]).any()
    assert second["steps"]["financials"]["status"] == "skipped"
    assert "already completed today" in second["steps"]["financials"]["reason"]
    assert second["steps"]["financials"]["polled_through"] == "20260721"


def test_financial_empty_poll_advances_but_exception_does_not(
    tmp_path: Path,
) -> None:
    class EmptyPro:
        def fina_indicator_vip(self, **kwargs):
            return pd.DataFrame()

        def income_vip(self, **kwargs):
            return pd.DataFrame()

        def cashflow_vip(self, **kwargs):
            return pd.DataFrame()

    empty = refresh_financial_periods(
        type("Fetcher", (), {"pro": EmptyPro()})(),
        tmp_path / "empty",
        "20260721",
        period_count=1,
        sleep_seconds=0,
    )

    assert empty["status"] == "success"
    assert empty["polled_through"] == "20260721"

    class PartialPro(EmptyPro):
        def income_vip(self, **kwargs):
            raise RuntimeError("provider unavailable")

    partial = refresh_financial_periods(
        type("Fetcher", (), {"pro": PartialPro()})(),
        tmp_path / "partial",
        "20260721",
        period_count=1,
        sleep_seconds=0,
    )

    assert partial["status"] == "partial"
    assert partial["polled_through"] is None


def test_reference_refresh_can_skip_financials(tmp_path: Path) -> None:
    result = refresh_reference_data(
        "20260721",
        include_financials=False,
        fetcher=FakeFetcher(),
        raw_dir=tmp_path / "raw",
        audit_root=tmp_path / "audit",
    )

    assert result["status"] == "success"
    assert result["steps"]["financials"]["status"] == "skipped"


def test_reference_refresh_reports_partial_when_index_is_stale(tmp_path: Path) -> None:
    fetcher = FakeFetcher()
    fetcher.pro.index_daily = lambda **kwargs: pd.DataFrame(
        {
            "ts_code": ["000300.SH"],
            "trade_date": ["20260720"],
            "close": [4500.0],
        }
    )

    result = refresh_reference_data(
        "20260721",
        include_financials=False,
        include_long_factor_sources=False,
        fetcher=fetcher,
        raw_dir=tmp_path / "raw",
        audit_root=tmp_path / "audit",
    )

    assert result["status"] == "partial"
    assert result["steps"]["index_000300"]["status"] == "partial"
    assert result["steps"]["index_000300"]["latest_trade_date"] == "20260720"


def test_stock_basic_cache_obeys_ttl_and_force_refresh(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "tushare_stock_basic_all.parquet"
    pd.DataFrame({"ts_code": ["OLD.SZ"], "symbol": ["OLD"]}).to_parquet(cache_path, index=False)

    class StockPro:
        calls = 0

        def stock_basic(self, **kwargs) -> pd.DataFrame:
            self.calls += 1
            return pd.DataFrame({"ts_code": ["NEW.SZ"], "symbol": ["NEW"]})

    fetcher = TushareDataFetcher.__new__(TushareDataFetcher)
    fetcher.cache_dir = cache_dir
    fetcher._memory_cache = {}
    fetcher.pro = StockPro()

    cached = fetcher.get_stock_basic(max_age_hours=24)
    refreshed = fetcher.get_stock_basic(force_refresh=True, max_age_hours=24)

    assert cached["ts_code"].tolist() == ["OLD.SZ"]
    assert refreshed["ts_code"].tolist() == ["NEW.SZ"]
    assert fetcher.pro.calls == 1
