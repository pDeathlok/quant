"""Leakage-resistant labels for unified right-side A-share models.

Signals are known after the signal-date close.  Entry is attempted on the next
*market* trading date, supplied explicitly by the caller, at either that day's
open or close.  Outcome observation starts on the first sellable session after
the entry session, which enforces A-share T+1.  Prices used for execution gates
remain raw; returns use the project's causal continuous price scale.

The module is intentionally data-frame-only.  It performs no file or network
I/O and does not infer a market calendar from a single stock's observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc


EntryMode = Literal["next_open", "next_close"]

DEFAULT_HORIZONS: tuple[int, ...] = (3, 5, 10)
DEFAULT_ENTRY_MODES: tuple[EntryMode, ...] = ("next_open", "next_close")


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _normalize_dates(values: pd.Series, *, name: str) -> pd.Series:
    """Parse project YYYYMMDD values and ordinary datetime-like values."""

    text = values.astype("string").str.strip()
    compact = text.str.replace("-", "", regex=False)
    eight_digit = compact.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(text, errors="coerce")
    if eight_digit.any():
        parsed.loc[eight_digit] = pd.to_datetime(
            compact.loc[eight_digit],
            format="%Y%m%d",
            errors="coerce",
        )
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        samples = values.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains invalid dates: {samples}")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _normalize_market_calendar(market_calendar: Sequence[object]) -> pd.DatetimeIndex:
    raw = pd.Series(list(market_calendar), dtype="object")
    if raw.empty:
        raise ValueError("market_calendar must contain at least one trading date")
    dates = _normalize_dates(raw, name="market_calendar")
    if dates.isna().any():
        raise ValueError("market_calendar cannot contain missing dates")
    if dates.duplicated().any():
        duplicates = dates.loc[dates.duplicated()].dt.strftime("%Y-%m-%d").head(3).tolist()
        raise ValueError(f"market_calendar contains duplicate dates: {duplicates}")
    return pd.DatetimeIndex(dates.sort_values().to_numpy())


def _normalize_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def detect_next_locked_limit_up(
    next_day: pd.DataFrame,
    *,
    open_column: str = "open",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
    pre_close_column: str = "pre_close",
    up_limit_column: str = "up_limit",
    exact_price_tolerance: float = 0.0051,
    one_price_relative_tolerance: float = 1e-10,
    proxy_return_threshold: float = 0.048,
) -> pd.DataFrame:
    """Detect an all-day locked limit-up and disclose the evidence source.

    Exact point-in-time ``up_limit`` wins whenever it is finite and positive.
    Only rows without an exact limit price may fall back to the conservative
    proxy: raw OHLC is one price and that price is at least 4.8% above raw
    ``pre_close``.  A present exact limit that does not match must never be
    overridden by the proxy.

    Returns a frame aligned to ``next_day.index`` with:

    - ``locked_limit_up``: ordinary ``bool``;
    - ``locked_limit_source``: ``exact_up_limit``, ``ohlc_4p8_proxy``, or
      ``unavailable``.
    """

    _require_columns(
        next_day,
        {open_column, high_column, low_column, close_column},
        name="next_day",
    )
    if exact_price_tolerance < 0:
        raise ValueError("exact_price_tolerance must be non-negative")
    if one_price_relative_tolerance < 0:
        raise ValueError("one_price_relative_tolerance must be non-negative")
    if proxy_return_threshold <= 0:
        raise ValueError("proxy_return_threshold must be positive")

    prices = pd.DataFrame(
        {
            "open": pd.to_numeric(next_day[open_column], errors="coerce"),
            "high": pd.to_numeric(next_day[high_column], errors="coerce"),
            "low": pd.to_numeric(next_day[low_column], errors="coerce"),
            "close": pd.to_numeric(next_day[close_column], errors="coerce"),
        },
        index=next_day.index,
    )
    pre_close = (
        pd.to_numeric(next_day[pre_close_column], errors="coerce")
        if pre_close_column in next_day.columns
        else pd.Series(np.nan, index=next_day.index, dtype=float)
    )
    up_limit = (
        pd.to_numeric(next_day[up_limit_column], errors="coerce")
        if up_limit_column in next_day.columns
        else pd.Series(np.nan, index=next_day.index, dtype=float)
    )

    finite_prices = pd.Series(
        np.isfinite(prices.to_numpy(dtype=float)).all(axis=1),
        index=next_day.index,
    )
    exact_available = pd.Series(
        np.isfinite(up_limit.to_numpy(dtype=float)) & up_limit.gt(0).to_numpy(),
        index=next_day.index,
    )
    exact_matches = pd.DataFrame(
        {
            column: np.isclose(
                prices[column].to_numpy(dtype=float),
                up_limit.to_numpy(dtype=float),
                rtol=1e-9,
                atol=exact_price_tolerance,
                equal_nan=False,
            )
            for column in prices.columns
        },
        index=next_day.index,
    ).all(axis=1)
    exact_locked = exact_available & finite_prices & exact_matches

    scale = prices.abs().max(axis=1).clip(lower=1.0)
    one_price = (
        prices.max(axis=1) - prices.min(axis=1)
    ).abs().le(scale * one_price_relative_tolerance)
    valid_proxy = (
        ~exact_available
        & finite_prices
        & pd.Series(np.isfinite(pre_close.to_numpy(dtype=float)), index=next_day.index)
        & pre_close.gt(0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        proxy_return = prices["open"] / pre_close - 1.0
    proxy_locked = valid_proxy & one_price & proxy_return.ge(proxy_return_threshold)

    source = pd.Series("unavailable", index=next_day.index, dtype="string")
    source.loc[exact_available] = "exact_up_limit"
    source.loc[valid_proxy] = "ohlc_4p8_proxy"
    return pd.DataFrame(
        {
            "locked_limit_up": (exact_locked | proxy_locked).fillna(False).astype(bool),
            "locked_limit_source": source,
        },
        index=next_day.index,
    )


def map_signals_to_next_market_date(
    signals: pd.DataFrame,
    market_calendar: Sequence[object],
    *,
    signal_date_column: str = "date",
) -> pd.DataFrame:
    """Attach the next global market date without consulting stock-local bars."""

    _require_columns(signals, {signal_date_column}, name="signals")
    calendar = _normalize_market_calendar(market_calendar)
    out = signals.copy()
    signal_dates = _normalize_dates(out[signal_date_column], name=signal_date_column)
    missing = signal_dates.isna()
    not_in_calendar = ~missing & ~signal_dates.isin(calendar)
    if not_in_calendar.any():
        samples = signal_dates.loc[not_in_calendar].dt.strftime("%Y-%m-%d").head(3).tolist()
        raise ValueError(f"signal dates are absent from market_calendar: {samples}")

    positions = calendar.searchsorted(signal_dates.fillna(calendar[-1]), side="right")
    entry_values = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    has_next = ~missing.to_numpy() & (positions < len(calendar))
    entry_values[has_next] = calendar.to_numpy(dtype="datetime64[ns]")[positions[has_next]]
    out["entry_date"] = pd.to_datetime(entry_values)
    out["has_next_market_date"] = has_next
    return out


def _prepare_daily(
    daily: pd.DataFrame,
    *,
    symbol_column: str,
    date_column: str,
) -> pd.DataFrame:
    _require_columns(
        daily,
        {symbol_column, date_column, "open", "high", "low", "close", "pre_close"},
        name="daily",
    )
    out = pd.DataFrame(index=daily.index)
    out["_label_symbol"] = daily[symbol_column].astype("string").str.strip()
    out["_label_date"] = _normalize_dates(daily[date_column], name=date_column)
    for column in ("open", "high", "low", "close", "pre_close"):
        out[f"_raw_{column}"] = pd.to_numeric(daily[column], errors="coerce")
    if out["_label_symbol"].isna().any() or out["_label_symbol"].eq("").any():
        raise ValueError("daily contains missing symbols")
    if out["_label_date"].isna().any():
        raise ValueError("daily contains missing dates")
    if out.duplicated(["_label_symbol", "_label_date"]).any():
        raise ValueError("daily contains duplicate symbol/date rows")
    out = out.sort_values(["_label_symbol", "_label_date"]).reset_index(drop=True)

    for column in ("open", "high", "low", "close"):
        out[f"_adjusted_{column}"] = np.nan
    for _, positions in out.groupby("_label_symbol", sort=False).groups.items():
        index = list(positions)
        symbol_daily = pd.DataFrame(
            {
                "date": out.loc[index, "_label_date"],
                "open": out.loc[index, "_raw_open"],
                "high": out.loc[index, "_raw_high"],
                "low": out.loc[index, "_raw_low"],
                "close": out.loc[index, "_raw_close"],
                "pre_close": out.loc[index, "_raw_pre_close"],
            },
            index=index,
        )
        adjusted = build_continuous_ohlc(symbol_daily)
        for column in ("open", "high", "low", "close"):
            out.loc[index, f"_adjusted_{column}"] = pd.to_numeric(
                adjusted[column], errors="coerce"
            ).to_numpy()
    return out


def _prepare_tradability(
    tradability: pd.DataFrame | None,
    *,
    symbol_column: str,
    date_column: str,
) -> pd.DataFrame:
    columns = ["_label_symbol", "_label_date", "_up_limit", "_is_suspended"]
    if tradability is None or tradability.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        tradability,
        {symbol_column, date_column, "up_limit"},
        name="tradability",
    )
    out = pd.DataFrame(index=tradability.index)
    out["_label_symbol"] = tradability[symbol_column].astype("string").str.strip()
    out["_label_date"] = _normalize_dates(tradability[date_column], name=date_column)
    out["_up_limit"] = pd.to_numeric(tradability["up_limit"], errors="coerce")
    out["_is_suspended"] = (
        _normalize_boolean(tradability["is_suspended"])
        if "is_suspended" in tradability.columns
        else False
    )
    if out.duplicated(["_label_symbol", "_label_date"]).any():
        raise ValueError("tradability contains duplicate symbol/date rows")
    return out.loc[:, columns].reset_index(drop=True)


def _forward_window(series: pd.Series, horizon: int, operation: str) -> pd.Series:
    shifted_reversed = series.shift(-1).iloc[::-1]
    rolling = shifted_reversed.rolling(horizon, min_periods=horizon)
    if operation == "max":
        result = rolling.max()
    elif operation == "min":
        result = rolling.min()
    elif operation == "sum":
        result = rolling.sum()
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"unsupported forward operation: {operation}")
    return result.iloc[::-1]


def _build_future_paths(
    daily: pd.DataFrame,
    wanted_entries: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    output_columns = [
        "_label_symbol",
        "entry_date",
        "horizon",
        "_path_label_end_date",
        "_path_complete",
        "_future_max_high",
        "_future_min_low",
        "_future_terminal_close",
    ]
    if wanted_entries.empty:
        return pd.DataFrame(columns=output_columns)

    wanted_by_symbol = {
        str(symbol): set(pd.to_datetime(group["entry_date"]).dropna())
        for symbol, group in wanted_entries.groupby("_label_symbol", sort=False)
    }
    rows: list[pd.DataFrame] = []
    for symbol, group in daily.groupby("_label_symbol", sort=False):
        wanted_dates = wanted_by_symbol.get(str(symbol))
        if not wanted_dates:
            continue
        ordered = group.sort_values("_label_date").reset_index(drop=True)
        finite_bar = pd.Series(
            np.isfinite(
                ordered[
                    [
                        "_adjusted_open",
                        "_adjusted_high",
                        "_adjusted_low",
                        "_adjusted_close",
                    ]
                ].to_numpy(dtype=float)
            ).all(axis=1),
            index=ordered.index,
            dtype=float,
        )
        wanted = ordered["_label_date"].isin(wanted_dates)
        if not wanted.any():
            continue
        for horizon in horizons:
            part = pd.DataFrame(
                {
                    "_label_symbol": ordered["_label_symbol"],
                    "entry_date": ordered["_label_date"],
                    "horizon": horizon,
                    "_path_label_end_date": ordered["_label_date"].shift(-horizon),
                    "_future_max_high": _forward_window(
                        ordered["_adjusted_high"], horizon, "max"
                    ),
                    "_future_min_low": _forward_window(
                        ordered["_adjusted_low"], horizon, "min"
                    ),
                    "_future_terminal_close": ordered["_adjusted_close"].shift(-horizon),
                    "_future_valid_count": _forward_window(finite_bar, horizon, "sum"),
                }
            )
            part["_path_complete"] = (
                part["_future_valid_count"].eq(float(horizon))
                & part["_path_label_end_date"].notna()
                & part["_future_max_high"].notna()
                & part["_future_min_low"].notna()
                & part["_future_terminal_close"].notna()
            )
            rows.append(part.loc[wanted, output_columns])
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.concat(rows, ignore_index=True)


def _empty_result(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.iloc[0:0].copy()
    out["entry_mode"] = pd.Series(dtype="string")
    out["horizon"] = pd.Series(dtype="int64")
    out["sellable_horizon"] = pd.Series(dtype="int64")
    out["entry_date"] = pd.Series(dtype="datetime64[ns]")
    out["label_end_date"] = pd.Series(dtype="datetime64[ns]")
    out["has_next_market_date"] = pd.Series(dtype=bool)
    out["entry_executable"] = pd.Series(dtype=bool)
    out["locked_limit_up"] = pd.Series(dtype=bool)
    out["locked_limit_source"] = pd.Series(dtype="string")
    out["mature"] = pd.Series(dtype=bool)
    for column in ("entry_price", "entry_raw_price", "mfe", "mae", "terminal", "terminal_return"):
        out[column] = pd.Series(dtype=float)
    for column in ("hit_up3", "hit_up5", "hit_up8", "hit_down3", "good_path5"):
        out[column] = pd.Series(dtype="boolean")
    out["maturity_reason"] = pd.Series(dtype="string")
    return out


def build_right_side_unified_labels(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    market_calendar: Sequence[object],
    tradability: pd.DataFrame | None = None,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    entry_modes: Sequence[EntryMode] = DEFAULT_ENTRY_MODES,
    signal_symbol_column: str = "symbol",
    signal_date_column: str = "date",
    daily_symbol_column: str = "ts_code",
    daily_date_column: str = "trade_date",
    tradability_symbol_column: str = "ts_code",
    tradability_date_column: str = "trade_date",
) -> pd.DataFrame:
    """Build long-form T+1 labels for unified right-side model research.

    The result has one row per ``signal x entry_mode x horizon``.  ``horizon``
    counts sellable sessions beginning after the entry session.  Returns are
    decimal fractions.  ``good_path5`` means MFE reached +5% while MAE stayed
    strictly above -3% throughout the complete window.

    Non-executable entries and incomplete tail windows have ``mature=False``;
    every outcome label, including nullable hit flags, is then missing.
    """

    _require_columns(
        signals,
        {signal_symbol_column, signal_date_column},
        name="signals",
    )
    if signals.empty:
        return _empty_result(signals)

    normalized_horizons = tuple(int(value) for value in horizons)
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError("horizons cannot contain duplicates")
    normalized_modes = tuple(entry_modes)
    if not normalized_modes or any(
        mode not in {"next_open", "next_close"} for mode in normalized_modes
    ):
        raise ValueError("entry_modes must contain next_open and/or next_close")
    if len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("entry_modes cannot contain duplicates")

    calendar = _normalize_market_calendar(market_calendar)
    prepared_daily = _prepare_daily(
        daily,
        symbol_column=daily_symbol_column,
        date_column=daily_date_column,
    )
    dates_outside_calendar = ~prepared_daily["_label_date"].isin(calendar)
    if dates_outside_calendar.any():
        samples = (
            prepared_daily.loc[dates_outside_calendar, "_label_date"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        raise ValueError(f"daily dates are absent from market_calendar: {samples}")
    prepared_tradability = _prepare_tradability(
        tradability,
        symbol_column=tradability_symbol_column,
        date_column=tradability_date_column,
    )

    base = signals.copy()
    base["_signal_order"] = np.arange(len(base))
    base["_label_symbol"] = base[signal_symbol_column].astype("string").str.strip()
    base["_label_signal_date"] = _normalize_dates(
        base[signal_date_column], name=signal_date_column
    )
    if base["_label_symbol"].isna().any() or base["_label_symbol"].eq("").any():
        raise ValueError("signals contains missing symbols")
    if base["_label_signal_date"].isna().any():
        raise ValueError("signals contains missing dates")
    if (~base["_label_signal_date"].isin(calendar)).any():
        samples = (
            base.loc[~base["_label_signal_date"].isin(calendar), "_label_signal_date"]
            .dt.strftime("%Y-%m-%d")
            .head(3)
            .tolist()
        )
        raise ValueError(f"signal dates are absent from market_calendar: {samples}")

    positions = calendar.searchsorted(base["_label_signal_date"], side="right")
    entry_dates = np.full(len(base), np.datetime64("NaT"), dtype="datetime64[ns]")
    has_next_market_date = positions < len(calendar)
    entry_dates[has_next_market_date] = calendar.to_numpy(dtype="datetime64[ns]")[
        positions[has_next_market_date]
    ]
    base["entry_date"] = pd.to_datetime(entry_dates)
    base["has_next_market_date"] = has_next_market_date

    entry_daily = prepared_daily.rename(columns={"_label_date": "entry_date"}).copy()
    entry_daily["_has_entry_bar"] = True
    base = base.merge(
        entry_daily,
        on=["_label_symbol", "entry_date"],
        how="left",
        validate="many_to_one",
    )
    entry_tradability = prepared_tradability.rename(
        columns={"_label_date": "entry_date"}
    )
    base = base.merge(
        entry_tradability,
        on=["_label_symbol", "entry_date"],
        how="left",
        validate="many_to_one",
    )
    base["_has_entry_bar"] = base["_has_entry_bar"].fillna(False).astype(bool)
    base["_is_suspended"] = base["_is_suspended"].fillna(False).astype(bool)

    detected = detect_next_locked_limit_up(
        pd.DataFrame(
            {
                "open": base["_raw_open"],
                "high": base["_raw_high"],
                "low": base["_raw_low"],
                "close": base["_raw_close"],
                "pre_close": base["_raw_pre_close"],
                "up_limit": base["_up_limit"],
            },
            index=base.index,
        )
    )
    base["locked_limit_up"] = detected["locked_limit_up"]
    base["locked_limit_source"] = detected["locked_limit_source"]

    # A tradability row can mark a suspended date even when a stale daily row is
    # present.  Such rows must not count as future sellable sessions.
    path_daily = prepared_daily.merge(
        prepared_tradability[["_label_symbol", "_label_date", "_is_suspended"]],
        on=["_label_symbol", "_label_date"],
        how="left",
        validate="one_to_one",
    )
    path_daily["_is_suspended"] = path_daily["_is_suspended"].fillna(False).astype(bool)
    path_daily = path_daily.loc[~path_daily["_is_suspended"]].copy()
    wanted_entries = base.loc[
        base["entry_date"].notna(), ["_label_symbol", "entry_date"]
    ].drop_duplicates()
    future_paths = _build_future_paths(
        path_daily,
        wanted_entries,
        normalized_horizons,
    )

    expanded_parts: list[pd.DataFrame] = []
    for mode_order, mode in enumerate(normalized_modes):
        for horizon_order, horizon in enumerate(normalized_horizons):
            part = base.copy()
            part["entry_mode"] = mode
            part["horizon"] = horizon
            part["sellable_horizon"] = horizon
            part["_mode_order"] = mode_order
            part["_horizon_order"] = horizon_order
            expanded_parts.append(part)
    out = pd.concat(expanded_parts, ignore_index=True, sort=False)
    out = out.merge(
        future_paths,
        on=["_label_symbol", "entry_date", "horizon"],
        how="left",
        validate="many_to_one",
    )
    out["_path_complete"] = out["_path_complete"].fillna(False).astype(bool)

    is_open = out["entry_mode"].eq("next_open")
    out["entry_raw_price"] = np.where(
        is_open, out["_raw_open"], out["_raw_close"]
    )
    out["entry_price"] = np.where(
        is_open, out["_adjusted_open"], out["_adjusted_close"]
    )
    finite_entry = (
        pd.Series(np.isfinite(out["entry_price"].to_numpy(dtype=float)), index=out.index)
        & out["entry_price"].gt(0)
        & pd.Series(
            np.isfinite(out["entry_raw_price"].to_numpy(dtype=float)), index=out.index
        )
        & out["entry_raw_price"].gt(0)
    )
    out["entry_executable"] = (
        out["has_next_market_date"]
        & out["_has_entry_bar"]
        & ~out["_is_suspended"]
        & ~out["locked_limit_up"]
        & finite_entry
    )
    out["mature"] = out["entry_executable"] & out["_path_complete"]

    out["maturity_reason"] = pd.Series("mature", index=out.index, dtype="string")
    out.loc[~out["_path_complete"], "maturity_reason"] = "incomplete_future_window"
    out.loc[~finite_entry, "maturity_reason"] = "invalid_entry_price"
    out.loc[out["locked_limit_up"], "maturity_reason"] = "locked_limit_up"
    out.loc[~out["_has_entry_bar"], "maturity_reason"] = "missing_entry_bar"
    out.loc[out["_is_suspended"], "maturity_reason"] = "suspended_entry"
    out.loc[~out["has_next_market_date"], "maturity_reason"] = "no_next_market_date"

    valid = out["mature"]
    out["label_end_date"] = pd.to_datetime(out["_path_label_end_date"]).where(valid)
    out["mfe"] = np.nan
    out["mae"] = np.nan
    out["terminal"] = np.nan
    if valid.any():
        out.loc[valid, "mfe"] = (
            out.loc[valid, "_future_max_high"] / out.loc[valid, "entry_price"] - 1.0
        )
        out.loc[valid, "mae"] = (
            out.loc[valid, "_future_min_low"] / out.loc[valid, "entry_price"] - 1.0
        )
        # An open-entry position is exposed to the entry session's low even
        # though A-share T+1 prevents selling that day. MFE still begins on T+2
        # because a T+1 intraday high cannot be realized by an open purchase.
        open_valid = valid & is_open
        out.loc[open_valid, "mae"] = np.minimum(
            out.loc[open_valid, "mae"].to_numpy(dtype=float),
            (
                out.loc[open_valid, "_adjusted_low"]
                / out.loc[open_valid, "entry_price"]
                - 1.0
            ).to_numpy(dtype=float),
        )
        out.loc[valid, "terminal"] = (
            out.loc[valid, "_future_terminal_close"] / out.loc[valid, "entry_price"]
            - 1.0
        )
    out["terminal_return"] = out["terminal"]

    for column, threshold in (("hit_up3", 0.03), ("hit_up5", 0.05), ("hit_up8", 0.08)):
        out[column] = pd.Series(pd.NA, index=out.index, dtype="boolean")
        if valid.any():
            out.loc[valid, column] = out.loc[valid, "mfe"].ge(threshold).to_numpy()
    out["hit_down3"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if valid.any():
        out.loc[valid, "hit_down3"] = out.loc[valid, "mae"].le(-0.03).to_numpy()
    out["good_path5"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if valid.any():
        out.loc[valid, "good_path5"] = (
            out.loc[valid, "mfe"].ge(0.05) & out.loc[valid, "mae"].gt(-0.03)
        ).to_numpy()

    internal_columns = [
        column
        for column in out.columns
        if column.startswith("_") and column not in signals.columns
    ]
    out = (
        out.sort_values(["_signal_order", "_mode_order", "_horizon_order"])
        .drop(columns=internal_columns)
        .reset_index(drop=True)
    )
    return out


__all__ = [
    "DEFAULT_ENTRY_MODES",
    "DEFAULT_HORIZONS",
    "EntryMode",
    "build_right_side_unified_labels",
    "detect_next_locked_limit_up",
    "map_signals_to_next_market_date",
]
