"""Explicit, offline-injectable BYD refresh operation; never called by page reads.

Wire ``refresh_byd_intraday_when_due`` before building the BYD daily workspace.
Pass the completed pipeline trade date and its Shanghai-local run clock. Keep
the data node TTL at 60 days; due is elapsed >= 60, including the exact boundary.
Use data.byd_intraday.read_byd_intraday_training_evidence in the runtime adapter.

Each due run fetches a small full-history qfq DAILY reference and a complete
calendar. Existing minutes may be retained only if every session's opening
anchor and price envelope still agree with that reference. A rebase triggers a full minute
rebuild, never a guessed multiplier. The reference is rechecked before commit.
"""

from __future__ import annotations

import fcntl
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quant.data.atomic_io import atomic_write_parquet
from quant.data.byd_intraday import (
    DEFAULT_CACHE_DIR,
    MAX_STALENESS_DAYS,
    PRICE_COLUMNS,
    TRAINING_FILENAME,
    local_timestamp,
    read_byd_training_frame,
    validate_training_bars,
)
from quant.infrastructure.publication import publication_write_path


class BydIntradaySource(Protocol):
    """Provider-normalized frames; ranges are inclusive Shanghai-local dates.

    calendar: calendar_date/is_trading_day, one row per calendar date, including
    closed days. daily: date/code/adjustflag/tradestatus/OHLC, one row per OPEN
    market date, including explicitly verified suspensions (tradestatus=0).
    minutes: datetime/date/code/adjustflag/OHLC/volume/amount, 48 bar-END times
    per active session. code=sz.002594 and adjustflag=2 are normalized contract
    labels, not a requirement to use BaoStock. Optional time is the legacy
    17-digit BaoStock time; other adapters should omit it. Prices must already
    use one consistent qfq anchor across ALL requests. Never label raw as qfq.
    """

    def calendar(self, start: str, end: str) -> pd.DataFrame: ...
    def daily(self, start: str, end: str) -> pd.DataFrame: ...
    def minutes(self, start: str, end: str) -> pd.DataFrame: ...


def _date(value: object) -> pd.Timestamp:
    parsed = local_timestamp(value)
    if parsed != parsed.normalize():
        raise ValueError("BYD expected a complete date, not a time")
    return parsed


def _daily_reference(
    source: BydIntradaySource, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    first, last = start.date().isoformat(), end.date().isoformat()
    calendar = source.calendar(first, last).copy()
    required = {"calendar_date", "is_trading_day"}
    if not required.issubset(calendar.columns):
        raise ValueError("BYD missing calendar fields")
    calendar["calendar_date"] = pd.to_datetime(calendar["calendar_date"], errors="raise")
    dates = pd.DatetimeIndex(calendar["calendar_date"])
    if dates.has_duplicates or not dates.sort_values().equals(pd.date_range(start, end)):
        raise ValueError("BYD incomplete calendar date coverage")
    trading = calendar["is_trading_day"].astype(str)
    if not trading.isin(["0", "1"]).all():
        raise ValueError("BYD invalid trading-day flags")
    expected = pd.DatetimeIndex(calendar.loc[trading.eq("1"), "calendar_date"]).sort_values()
    daily = source.daily(first, last).copy()
    if not {"date", "code", "adjustflag", "tradestatus", *PRICE_COLUMNS}.issubset(daily.columns):
        raise ValueError("BYD missing daily reference fields")
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    observed = pd.DatetimeIndex(daily["date"])
    if observed.has_duplicates or not observed.sort_values().equals(expected):
        raise ValueError("BYD incomplete daily date coverage")
    if not daily["code"].eq("sz.002594").all() or not daily["adjustflag"].astype(str).eq("2").all():
        raise ValueError("BYD invalid daily symbol/adjustment")
    status = daily["tradestatus"].astype(str)
    if not status.isin(["0", "1"]).all():
        raise ValueError("BYD invalid suspension flags")
    daily = daily.loc[status.eq("1"), ["date", *PRICE_COLUMNS]].set_index("date").sort_index()
    daily = daily.apply(pd.to_numeric, errors="raise")
    if daily.empty or not np.isfinite(daily.to_numpy(dtype=float)).all() or (daily <= 0).any().any():
        raise ValueError("BYD invalid active daily prices")
    if (daily["high"] < daily[PRICE_COLUMNS].max(axis=1)).any() or (
        daily["low"] > daily[PRICE_COLUMNS].min(axis=1)
    ).any():
        raise ValueError("BYD invalid daily reference OHLC ordering")
    return daily


def _agrees(bars: pd.DataFrame, daily: pd.DataFrame) -> bool:
    """Prove session coverage and adjustment scale, not cross-frequency equality."""
    aggregated = bars.groupby("date").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    if not aggregated.index.equals(daily.index):
        return False
    anchor_matches = np.allclose(aggregated["open"], daily["open"], rtol=1e-6, atol=1e-4)
    high_tolerance = daily["high"].abs() * 1e-6 + 1e-4
    low_tolerance = daily["low"].abs() * 1e-6 + 1e-4
    return bool(
        anchor_matches
        and (aggregated["high"] <= daily["high"] + high_tolerance).all()
        and (aggregated["low"] >= daily["low"] - low_tolerance).all()
    )


def _cross_frequency_audit(bars: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    aggregated = bars.groupby("date").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    differences = (aggregated[PRICE_COLUMNS] - daily[PRICE_COLUMNS]).abs()
    return {
        "status": "coherent",
        "policy": "complete_sessions_daily_open_anchor_and_price_envelope",
        "checked_sessions": len(daily),
        "cross_frequency_differing_sessions": int((differences > 1e-4).any(axis=1).sum()),
        "max_absolute_price_difference": differences.max().to_dict(),
        "raw_minute_prices_modified": False,
    }


def _fetch_minutes(source: BydIntradaySource, daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        raise ValueError("BYD no active sessions in minute request")
    pieces = []
    # Bounded requests avoid a single multi-year provider result stream.
    for _, chunk in daily.groupby(daily.index.to_period("M")):
        first, last = chunk.index.min(), chunk.index.max()
        bars = validate_training_bars(source.minutes(first.date().isoformat(), last.date().isoformat()))
        if not _agrees(bars, chunk):
            aggregated = bars.groupby("date").agg(
                open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"),
            )
            difference = (aggregated[PRICE_COLUMNS] - chunk[PRICE_COLUMNS]).abs()
            raise ValueError(
                "BYD minute coverage/prices disagree with complete qfq daily reference: "
                f"interval={first.date()}..{last.date()} "
                f"observed_days={len(aggregated)} expected_days={len(chunk)} "
                f"missing_dates={chunk.index.difference(aggregated.index).strftime('%Y-%m-%d').tolist()} "
                f"unexpected_dates={aggregated.index.difference(chunk.index).strftime('%Y-%m-%d').tolist()} "
                f"max_price_difference={difference.max().to_dict()}"
            )
        pieces.append(bars)
    return validate_training_bars(pd.concat(pieces, ignore_index=True))


def _result(frame: pd.DataFrame, now: pd.Timestamp, *, status: str, reason: str) -> dict[str, Any]:
    checkpoint = frame["date"].max()
    return {
        "status": status, "reason": reason,
        "source": frame.attrs["intraday_training_source"],
        "fingerprint": frame.attrs["intraday_training_fingerprint"],
        "provider": frame.attrs.get("provider", "unknown"),
        "watermark": checkpoint.date().isoformat(),
        "checked_at": checkpoint.isoformat(),
        "intraday_training_max_date": checkpoint.date().isoformat(),
        "intraday_training_role": "historical_model_training_and_validation_samples",
        "intraday_is_current_feature": False,
        "age_days": (now - checkpoint).total_seconds() / 86400,
        "max_staleness_days": MAX_STALENESS_DAYS,
        "rows": len(frame),
        "cross_frequency_audit": frame.attrs.get("cross_frequency_audit"),
    }


def refresh_byd_intraday_when_due(
    *, expected_trade_date: str,
    now: datetime | pd.Timestamp | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    history_start: str = "2020-01-01",
    source_factory: Callable[[], AbstractContextManager[BydIntradaySource]] | None = None,
) -> dict[str, Any]:
    """Refresh at >=60 elapsed days; source_factory returns a source context manager.

    A healthy skip does not open a provider session. Missing data is bootstrapped
    from history_start; invalid existing data fails closed for explicit repair.
    Failures propagate to the composite operation and leave the old file intact.
    ``expected_trade_date`` must be a completed market date (holidays accepted).
    The default lazy wrapper chooses Tushare first, then AkShare/BaoStock only
    when necessary. Inject a factory to test offline or select a provider.
    Orchestrators should bound wall time; provider I/O is synchronous.
    """
    if source_factory is None:
        from quant.routine.byd_intraday_sources import refresh_byd_with_preferred_sources

        return refresh_byd_with_preferred_sources(
            refresh_byd_intraday_when_due,
            expected_trade_date=expected_trade_date,
            now=now,
            cache_dir=cache_dir,
            history_start=history_start,
        )
    current = local_timestamp(now if now is not None else pd.Timestamp.now(tz="Asia/Shanghai"))
    end = _date(expected_trade_date)
    if end > current.normalize() or (end == current.normalize() and current.hour < 15):
        raise ValueError("BYD requested decision date has not closed")
    cache_dir = Path(cache_dir).absolute()
    target = publication_write_path(cache_dir / TRAINING_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("BYD training refresh already running") from exc
        try:
            old = read_byd_training_frame(cache_dir)
        except FileNotFoundError:
            old = None
        if old is not None:
            maximum = old["date"].max()
            if maximum > end:
                raise ValueError("BYD training history extends beyond decision date")
            if (current - maximum).total_seconds() / 86400 < MAX_STALENESS_DAYS:
                return _result(old, current, status="skipped", reason="within_60_day_sla")
            start = old["date"].min()
        else:
            start = _date(history_start)
        if start > end:
            raise ValueError("BYD history_start is after decision date")
        with source_factory() as source:
            reference = _daily_reference(source, start, end)
            provider_changed = old is not None and old.attrs.get("provider") != getattr(source, "provider", "injected")
            rebuild = old is None or provider_changed or not _agrees(old, reference.loc[:old["date"].max()])
            reason = "bootstrap" if old is None else "provider_changed" if provider_changed else "qfq_history_changed" if rebuild else "incremental"
            if rebuild:
                combined = _fetch_minutes(source, reference)
            else:
                # Re-fetch the last five sessions, including their entire days.
                overlap_start = old["date"].drop_duplicates().iloc[-min(5, old["date"].nunique())]
                tail = _fetch_minutes(source, reference.loc[overlap_start:])
                # Both inputs' raw timestamps were already validated. A provider
                # switch may omit the legacy time column; datetime is canonical.
                combined = validate_training_bars(pd.concat(
                    [
                        old.loc[old["date"] < overlap_start].drop(columns="time", errors="ignore"),
                        tail.drop(columns="time", errors="ignore"),
                    ], ignore_index=True,
                ))
            if not _agrees(combined, reference):
                raise ValueError("BYD merged history failed full date/qfq validation")
            after = _daily_reference(source, start, end)
            if not reference.equals(after):
                raise RuntimeError("BYD qfq reference changed during refresh; retry without publishing")
            checkpoint = combined["date"].max()
            if (current - checkpoint).total_seconds() / 86400 >= MAX_STALENESS_DAYS:
                raise RuntimeError("BYD provider history remains at or beyond 60-day refresh boundary")
            if old is not None and checkpoint <= old["date"].max():
                raise RuntimeError("BYD refresh made no date progress")
            combined.attrs = {
                "byd_training_contract_version": 1,
                "qfq_validation": "all_session_daily_open_and_price_envelope",
                "cross_frequency_audit": _cross_frequency_audit(combined, reference),
                "provider": getattr(source, "provider", "injected"),
            }
        # A failing provider context exit must not publish before a fallback.
        atomic_write_parquet(combined, target, index=False)
        return _result(read_byd_training_frame(cache_dir), current, status="success", reason=reason)


class BaoStockBydSource:
    """Lazy BaoStock adapter. Check errors both before and after row iteration."""

    provider = "baostock"

    def __enter__(self) -> BaoStockBydSource:
        try:
            import baostock
        except ImportError as exc:
            raise RuntimeError("BYD refresh requires the optional baostock package") from exc
        self.bs = baostock
        response = self.bs.login()
        self._check(response)
        return self

    def __exit__(self, *args: object) -> None:
        self.bs.logout()

    @staticmethod
    def _check(result: Any) -> None:
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {result.error_code} {result.error_msg}")

    def _frame(self, result: Any) -> pd.DataFrame:
        self._check(result)
        rows = []
        while result.next():
            self._check(result)
            rows.append(result.get_row_data())
        self._check(result)
        return pd.DataFrame(rows, columns=result.fields)

    def calendar(self, start: str, end: str) -> pd.DataFrame:
        return self._frame(self.bs.query_trade_dates(start_date=start, end_date=end))

    def daily(self, start: str, end: str) -> pd.DataFrame:
        return self._frame(self.bs.query_history_k_data_plus(
            "sz.002594", "date,code,open,high,low,close,adjustflag,tradestatus",
            start_date=start, end_date=end, frequency="d", adjustflag="2",
        ))

    def minutes(self, start: str, end: str) -> pd.DataFrame:
        frame = self._frame(self.bs.query_history_k_data_plus(
            "sz.002594", "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start, end_date=end, frequency="5", adjustflag="2",
        ))
        frame["datetime"] = pd.to_datetime(
            frame["time"].astype(str).str[:14], format="%Y%m%d%H%M%S", errors="raise",
        )
        return frame
