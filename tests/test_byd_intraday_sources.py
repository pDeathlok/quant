from __future__ import annotations

import pandas as pd
import pytest

from quant.routine.byd_intraday_sources import TushareBydSource, refresh_byd_with_preferred_sources


def test_provider_order_and_failure_receipts() -> None:
    factories = tuple((name, object()) for name in ("tushare", "akshare", "baostock"))
    called = []

    def refresh(*, source_factory, **kwargs):
        called.append(source_factory)
        assert kwargs["expected_trade_date"] == "2026-09-15"
        if source_factory is factories[0][1]:
            raise RuntimeError("stk_mins code=40203: daily quota exhausted")
        if source_factory is factories[1][1]:
            raise ValueError("insufficient historical coverage")
        return {"status": "success", "provider": "baostock"}

    result = refresh_byd_with_preferred_sources(
        refresh, source_factories=factories, expected_trade_date="2026-09-15",
    )
    assert called == [factory for _, factory in factories]
    assert result["provider"] == "baostock"
    assert [row["status"] for row in result["provider_attempts"]] == ["failed", "failed", "success"]


def test_successful_tushare_never_calls_fallbacks() -> None:
    called = []
    def refresh(*, source_factory):
        called.append(source_factory)
        return {"status": "success", "provider": "tushare"}
    factories = (("tushare", object()), ("akshare", object()), ("baostock", object()))
    result = refresh_byd_with_preferred_sources(refresh, source_factories=factories)
    assert called == [factories[0][1]]
    assert result["provider"] == "tushare"


def test_all_provider_failures_remain_failures() -> None:
    def refresh(**kwargs):
        raise ValueError("incomplete session")
    with pytest.raises(RuntimeError, match="preferred providers"):
        refresh_byd_with_preferred_sources(refresh, source_factories=(("tushare", object()),))


def test_tushare_minutes_use_daily_qfq_anchor_and_keep_share_units(monkeypatch) -> None:
    source = TushareBydSource(token="test", session=object())
    source.anchor = 4.0
    source.factors = pd.Series([2.0], index=pd.DatetimeIndex(["2026-09-14"]))
    raw = pd.DataFrame({
        "ts_code": ["002594.SZ"], "trade_time": ["2026-09-14 09:35:00"],
        "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0],
        "vol": [1000.0], "amount": [100500.0],
    })
    monkeypatch.setattr(source, "_query", lambda *args, **kwargs: raw)
    result = source.minutes("2026-09-14", "2026-09-14")
    assert result.loc[0, "open"] == 50.0
    assert result.loc[0, "close"] == 50.5
    assert result.loc[0, "volume"] == 1000
    assert result.loc[0, "amount"] == 100500


def test_adjustment_factor_holes_are_not_filled() -> None:
    source = TushareBydSource(token="test", session=object())
    source.anchor = 4.0
    with pytest.raises(ValueError, match="factor missing"):
        source._adjust(pd.DataFrame({"date": pd.to_datetime(["2026-09-15"])}))


def test_tushare_quota_errors_are_redacted() -> None:
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"code": 40203, "msg": "quota private-token"}
    class Session:
        def post(self, *args, **kwargs):
            assert kwargs["timeout"] == (10, 30)
            return Response()
    source = TushareBydSource(token="private-token", session=Session())
    with pytest.raises(RuntimeError, match="40203") as error:
        source._query("stk_mins")
    assert "private-token" not in str(error.value)


def test_minute_permission_probe_precedes_reference_download_and_closes_on_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    closed = []
    source = TushareBydSource(
        token="test", session=SimpleNamespace(close=lambda: closed.append(True)), probe_date="2026-09-15",
    )
    def query(api, **params):
        assert api == "stk_mins"
        assert params["end_date"] == "2026-09-15 15:30:00"
        raise RuntimeError("quota")
    monkeypatch.setattr(source, "_query", query)
    with pytest.raises(RuntimeError, match="quota"):
        with source:
            pytest.fail("Unavailable provider must not download reference history")
    assert closed == [True]


def test_akshare_volume_is_converted_from_lots_and_prices_are_adjusted(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from quant.routine.byd_intraday_sources import AkShareBydSource

    def minutes(**kwargs):
        assert kwargs["adjust"] == ""
        return pd.DataFrame({
            "时间": ["2026-09-15 09:35:00"], "开盘": [100], "收盘": [101],
            "最高": [102], "最低": [99], "成交量": [10], "成交额": [100500],
        })
    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(stock_zh_a_hist_min_em=minutes))
    source = AkShareBydSource(token="test", session=object())
    source.anchor = 4.0
    source.factors = pd.Series([2.0], index=pd.DatetimeIndex(["2026-09-15"]))
    result = source.minutes("2026-09-15", "2026-09-15")
    assert result.loc[0, "volume"] == 1000
    assert result.loc[0, "close"] == 50.5
