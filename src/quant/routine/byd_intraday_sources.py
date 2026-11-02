"""Ordered providers for the existing BYD historical-training refresh contract.

Minutes are never silently mixed across providers. Each candidate must pass the
refresh operation's complete-session and daily-price checks before publication.
Tushare/AkShare raw minutes use the same Tushare adjustment anchor as daily bars.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import requests


class TushareBydSource:
    provider = "tushare"

    def __init__(self, *, token: str | None = None, session: Any = None, probe_date: str | None = None) -> None:
        self.token = token or os.environ.get("TUSHARE_TOKEN")
        if not self.token:
            raise RuntimeError("Tushare credential unavailable")
        self.session = session or requests.Session()
        self.anchor: float | None = None
        self.factors = pd.Series(dtype=float)
        self.probe_date = probe_date

    def __enter__(self) -> TushareBydSource:
        try:
            if self.probe_date:
                self._probe(self.probe_date)
        except Exception:
            self.session.close()
            raise
        return self

    def _probe(self, day: str) -> None:
        # Reject unavailable minute permissions before fetching years of references.
        frame = self._query(
            "stk_mins", ts_code="002594.SZ", freq="5min",
            start_date=f"{day} 09:00:00", end_date=f"{day} 15:30:00",
        )
        if frame.empty:
            raise ValueError("Tushare current BYD minute history unavailable")

    def __exit__(self, *args: object) -> None:
        self.session.close()

    def _query(self, api: str, **params: Any) -> pd.DataFrame:
        response = self.session.post(
            "https://api.tushare.pro",
            json={"api_name": api, "token": self.token, "params": params, "fields": ""},
            timeout=(10, 30),
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            message = str(body.get("msg", "")).replace(self.token, "[REDACTED]")
            raise RuntimeError(f"Tushare {api} code={body.get('code')}: {message}")
        payload = body.get("data") or {}
        frame = pd.DataFrame(payload.get("items") or [], columns=payload.get("fields") or [])
        if len(frame) >= (8000 if api == "stk_mins" else 5000):
            raise ValueError(f"Tushare {api} response may be truncated")
        return frame

    def _history(self, api: str, start: str, end: str) -> pd.DataFrame:
        first, last = pd.Timestamp(start), pd.Timestamp(end)
        pieces = []
        while first <= last:
            stop = min(first + pd.DateOffset(years=4) - pd.Timedelta(days=1), last)
            params = {"start_date": first.strftime("%Y%m%d"), "end_date": stop.strftime("%Y%m%d")}
            if api == "trade_cal":
                params["exchange"] = "SZSE"
            else:
                params["ts_code"] = "002594.SZ"
            pieces.append(self._query(api, **params))
            first = stop + pd.Timedelta(days=1)
        return pd.concat(pieces, ignore_index=True)

    def calendar(self, start: str, end: str) -> pd.DataFrame:
        return self._history("trade_cal", start, end).rename(
            columns={"cal_date": "calendar_date", "is_open": "is_trading_day"},
        )

    def daily(self, start: str, end: str) -> pd.DataFrame:
        daily = self._history("daily", start, end)
        factors = self._history("adj_factor", start, end)
        for frame in (daily, factors):
            if frame.empty or not frame["ts_code"].eq("002594.SZ").all():
                raise ValueError("Tushare BYD reference missing or wrong symbol")
            frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="raise")
            if frame["date"].duplicated().any():
                raise ValueError("Tushare duplicate BYD reference dates")
        factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="raise")
        if not np.isfinite(factors["adj_factor"]).all() or factors["adj_factor"].le(0).any():
            raise ValueError("Tushare invalid adjustment factors")
        self.factors = factors.set_index("date")["adj_factor"].sort_index()
        self.anchor = float(self.factors.iloc[-1])
        daily = self._adjust(daily)
        daily["code"], daily["adjustflag"], daily["tradestatus"] = "sz.002594", "2", "1"
        # Missing exchange sessions require explicit suspension evidence.
        calendar = self.calendar(start, end)
        expected = pd.to_datetime(calendar.loc[calendar.is_trading_day.astype(str).eq("1"), "calendar_date"])
        missing = set(expected) - set(daily["date"])
        if missing:
            suspensions = self._history("suspend_d", start, end)
            suspended = set(pd.to_datetime(
                suspensions.loc[
                    suspensions["ts_code"].eq("002594.SZ")
                    & suspensions["suspend_type"].eq("S")
                    & suspensions["suspend_timing"].fillna("").eq(""), "trade_date",
                ], format="%Y%m%d", errors="raise",
            ))
            if not missing <= suspended:
                raise ValueError("Tushare missing BYD daily bars without full-day suspension evidence")
            inactive = pd.DataFrame({"date": sorted(missing), "code": "sz.002594", "adjustflag": "2", "tradestatus": "0"})
            daily = pd.concat([daily, inactive], ignore_index=True)
        return daily.sort_values("date").reset_index(drop=True)

    def _adjust(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.anchor is None:
            raise ValueError("Daily adjustment reference must be loaded before minutes")
        result = frame.copy()
        ratio = result["date"].map(self.factors) / self.anchor
        if ratio.isna().any():
            raise ValueError("Tushare adjustment factor missing for observed bars")
        for column in ("open", "high", "low", "close"):
            result[column] = pd.to_numeric(result[column], errors="raise") * ratio
        return result

    def minutes(self, start: str, end: str) -> pd.DataFrame:
        frame = self._query(
            "stk_mins", ts_code="002594.SZ", freq="5min",
            start_date=f"{start} 09:00:00", end_date=f"{end} 15:30:00",
        )
        if frame.empty or not frame["ts_code"].eq("002594.SZ").all():
            raise ValueError("Tushare BYD minutes missing or wrong symbol")
        frame = frame.rename(columns={"vol": "volume"})
        frame["datetime"] = pd.to_datetime(frame["trade_time"], errors="raise")
        return self._normalize_minutes(frame, start, end)

    def _normalize_minutes(self, frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        frame = frame.copy()
        frame["date"] = frame["datetime"].dt.normalize()
        if not frame["date"].between(pd.Timestamp(start), pd.Timestamp(end)).all():
            raise ValueError("Minute provider returned dates outside requested interval")
        frame = self._adjust(frame)
        frame["code"], frame["adjustflag"] = "sz.002594", "2"
        return frame


class AkShareBydSource(TushareBydSource):
    provider = "akshare"

    def _probe(self, day: str) -> None:
        import akshare as ak

        frame = ak.stock_zh_a_hist_min_em(
            symbol="002594", start_date=f"{day} 09:00:00", end_date=f"{day} 15:30:00",
            period="5", adjust="",
        )
        if frame.empty:
            raise ValueError("AkShare current BYD minute history unavailable")

    def minutes(self, start: str, end: str) -> pd.DataFrame:
        import akshare as ak

        frame = ak.stock_zh_a_hist_min_em(
            symbol="002594", start_date=f"{start} 09:00:00", end_date=f"{end} 15:30:00",
            period="5", adjust="",
        ).rename(columns={
            "时间": "datetime", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })
        if frame.empty:
            raise ValueError("AkShare does not cover the requested BYD minute history")
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="raise") * 100
        return self._normalize_minutes(frame, start, end)


def refresh_byd_with_preferred_sources(
    refresh_operation: Callable[..., dict[str, Any]], *,
    source_factories: tuple[tuple[str, Callable[..., Any]], ...] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    from quant.routine.byd_intraday_refresh import BaoStockBydSource

    factories = source_factories or (
        ("tushare", partial(TushareBydSource, probe_date=kwargs.get("expected_trade_date"))),
        ("akshare", partial(AkShareBydSource, probe_date=kwargs.get("expected_trade_date"))),
        ("baostock", BaoStockBydSource),
    )
    attempts: list[dict[str, str]] = []
    for name, factory in factories:
        print(f"[byd-training] provider={name} status=checking", flush=True)
        try:
            result = refresh_operation(source_factory=factory, **kwargs)
        except (RuntimeError, ValueError, OSError, KeyError, ImportError) as exc:
            attempts.append({"provider": name, "status": "failed", "error": str(exc)[:600]})
            print(f"[byd-training] provider={name} status=failed error={str(exc)[:600]}", flush=True)
            continue
        attempts.append({"provider": name, "status": str(result["status"])})
        print(f"[byd-training] provider={name} status={result['status']}", flush=True)
        return {**result, "provider_attempts": attempts}
    raise RuntimeError(f"BYD minute history unavailable from preferred providers: {attempts}")
