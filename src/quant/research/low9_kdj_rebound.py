"""Causal event-study primitives for low-9 plus negative KDJ-J signals.

The signal is the simplified market-software definition of a completed
downside nine setup: nine consecutive bars whose close is below the close four
bars earlier.  Indicators use a forward-only continuous price scale so a later
corporate action never rewrites an earlier signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_J_THRESHOLDS = (0.0, -10.0, -20.0, -30.0)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


@dataclass
class PendingLow9Event:
    symbol: str
    signal_date: pd.Timestamp
    signal_bar_number: int
    previous_signal_gap_bars: float
    j_value: float
    signal_close: float
    signal_raw_close: float
    signal_amount: float
    signal_name: str | None
    signal_market_close: float
    horizons: tuple[int, ...]
    entry_date: pd.Timestamp | None = None
    entry_open: float | None = None
    entry_raw_open: float | None = None
    entry_market_open: float | None = None
    entry_one_price: bool = False
    entry_amount: float = np.nan
    bars_elapsed: int = 0
    max_high: float = -np.inf
    min_low: float = np.inf
    resolved: list[dict[str, object]] = field(default_factory=list)

    def advance(
        self,
        *,
        trade_date: pd.Timestamp,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        raw_open: float,
        raw_high: float,
        raw_low: float,
        amount: float,
        market_open: float | None,
        market_close: float | None,
    ) -> bool:
        """Consume one future stock bar and return whether the event is done."""

        self.bars_elapsed += 1
        if self.bars_elapsed == 1:
            self.entry_date = trade_date
            self.entry_open = open_price
            self.entry_raw_open = raw_open
            self.entry_market_open = market_open
            self.entry_amount = amount
            raw_range = max(raw_high, raw_open) - min(raw_low, raw_open)
            self.entry_one_price = bool(
                np.isfinite(raw_range)
                and abs(raw_range) <= max(abs(raw_open), 1.0) * 1e-10
            )

        self.max_high = max(self.max_high, high_price)
        self.min_low = min(self.min_low, low_price)

        if self.bars_elapsed in self.horizons:
            close_return = close_price / self.signal_close - 1.0
            executable_return = (
                close_price / self.entry_open - 1.0
                if self.entry_open not in {None, 0.0}
                else np.nan
            )
            market_close_return = (
                market_close / self.signal_market_close - 1.0
                if market_close is not None
                and np.isfinite(self.signal_market_close)
                and self.signal_market_close != 0
                else np.nan
            )
            market_executable_return = (
                market_close / self.entry_market_open - 1.0
                if market_close is not None
                and self.entry_market_open is not None
                and self.entry_market_open != 0
                else np.nan
            )
            self.resolved.append(
                {
                    "symbol": self.symbol,
                    "name": self.signal_name,
                    "signal_date": self.signal_date,
                    "entry_date": self.entry_date,
                    "exit_date": trade_date,
                    "horizon": self.bars_elapsed,
                    "j_value": self.j_value,
                    "signal_close": self.signal_close,
                    "signal_raw_close": self.signal_raw_close,
                    "entry_open": self.entry_open,
                    "entry_raw_open": self.entry_raw_open,
                    "exit_close": close_price,
                    "signal_amount": self.signal_amount,
                    "entry_amount": self.entry_amount,
                    "entry_one_price": self.entry_one_price,
                    "previous_signal_gap_bars": self.previous_signal_gap_bars,
                    "close_return": close_return,
                    "executable_return": executable_return,
                    "market_close_return": market_close_return,
                    "market_executable_return": market_executable_return,
                    "abnormal_close_return": close_return - market_close_return,
                    "abnormal_executable_return": (
                        executable_return - market_executable_return
                    ),
                    "mfe": (
                        self.max_high / self.entry_open - 1.0
                        if self.entry_open not in {None, 0.0}
                        else np.nan
                    ),
                    "mae": (
                        self.min_low / self.entry_open - 1.0
                        if self.entry_open not in {None, 0.0}
                        else np.nan
                    ),
                }
            )
        return self.bars_elapsed >= max(self.horizons)


@dataclass
class SymbolSignalState:
    """Small causal state needed to process one symbol month by month."""

    symbol: str
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    min_history_bars: int = 60
    continuity_factor: float = 1.0
    previous_raw_close: float | None = None
    adjusted_closes: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    adjusted_highs: deque[float] = field(default_factory=lambda: deque(maxlen=9))
    adjusted_lows: deque[float] = field(default_factory=lambda: deque(maxlen=9))
    k_value: float | None = None
    d_value: float | None = None
    low_setup_count: int = 0
    bar_number: int = 0
    last_signal_bar_number: int | None = None
    pending: list[PendingLow9Event] = field(default_factory=list)

    def _continuous_prices(
        self,
        *,
        raw_open: float,
        raw_high: float,
        raw_low: float,
        raw_close: float,
        pre_close: float | None,
    ) -> tuple[float, float, float, float]:
        if (
            self.previous_raw_close is not None
            and pre_close is not None
            and pre_close > 0
        ):
            ratio = self.previous_raw_close / pre_close
            if np.isfinite(ratio) and ratio > 0:
                self.continuity_factor *= ratio
        self.previous_raw_close = raw_close
        factor = self.continuity_factor
        return (
            raw_open * factor,
            raw_high * factor,
            raw_low * factor,
            raw_close * factor,
        )

    def process_bar(
        self,
        row: Mapping[str, object],
        market_bar: Mapping[str, float] | None,
    ) -> list[dict[str, object]]:
        trade_date = pd.Timestamp(row["date"])
        raw_open = _finite(row.get("open"))
        raw_high = _finite(row.get("high"))
        raw_low = _finite(row.get("low"))
        raw_close = _finite(row.get("close"))
        if None in {raw_open, raw_high, raw_low, raw_close}:
            return []
        assert raw_open is not None
        assert raw_high is not None
        assert raw_low is not None
        assert raw_close is not None
        if raw_low <= 0 or raw_high < raw_low:
            return []

        pre_close = _finite(row.get("pre_close"))
        open_price, high_price, low_price, close_price = self._continuous_prices(
            raw_open=raw_open,
            raw_high=raw_high,
            raw_low=raw_low,
            raw_close=raw_close,
            pre_close=pre_close,
        )
        amount = _finite(row.get("amount"))
        amount_value = amount if amount is not None else np.nan
        market_open = _finite(market_bar.get("open")) if market_bar else None
        market_close = _finite(market_bar.get("close")) if market_bar else None

        resolved: list[dict[str, object]] = []
        remaining: list[PendingLow9Event] = []
        for event in self.pending:
            done = event.advance(
                trade_date=trade_date,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                raw_open=raw_open,
                raw_high=raw_high,
                raw_low=raw_low,
                amount=amount_value,
                market_open=market_open,
                market_close=market_close,
            )
            resolved.extend(event.resolved)
            event.resolved.clear()
            if not done:
                remaining.append(event)
        self.pending = remaining

        low_condition = (
            len(self.adjusted_closes) >= 4
            and close_price < self.adjusted_closes[-4]
        )
        self.low_setup_count = self.low_setup_count + 1 if low_condition else 0
        self.adjusted_closes.append(close_price)
        self.adjusted_highs.append(high_price)
        self.adjusted_lows.append(low_price)
        self.bar_number += 1

        if len(self.adjusted_highs) == 9:
            lowest = min(self.adjusted_lows)
            highest = max(self.adjusted_highs)
            if highest > lowest:
                rsv = (close_price - lowest) / (highest - lowest) * 100.0
                self.k_value = (
                    rsv
                    if self.k_value is None
                    else (rsv / 3.0 + self.k_value * 2.0 / 3.0)
                )
                self.d_value = (
                    self.k_value
                    if self.d_value is None
                    else (self.k_value / 3.0 + self.d_value * 2.0 / 3.0)
                )

        if (
            self.low_setup_count == 9
            and self.bar_number >= self.min_history_bars
            and self.k_value is not None
            and self.d_value is not None
        ):
            j_value = 3.0 * self.k_value - 2.0 * self.d_value
            previous_gap = (
                float(self.bar_number - self.last_signal_bar_number)
                if self.last_signal_bar_number is not None
                else np.nan
            )
            signal_market_close = market_close if market_close is not None else np.nan
            self.pending.append(
                PendingLow9Event(
                    symbol=self.symbol,
                    signal_date=trade_date,
                    signal_bar_number=self.bar_number,
                    previous_signal_gap_bars=previous_gap,
                    j_value=j_value,
                    signal_close=close_price,
                    signal_raw_close=raw_close,
                    signal_amount=amount_value,
                    signal_name=_optional_text(row.get("name")),
                    signal_market_close=signal_market_close,
                    horizons=self.horizons,
                )
            )
            self.last_signal_bar_number = self.bar_number
        return resolved


def newey_west_mean_test(values: Iterable[float], lag: int) -> dict[str, float]:
    """Test a time-series mean using a Bartlett-kernel HAC standard error."""

    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "p": np.nan}
    mean = float(x.mean())
    centered = x - mean
    max_lag = min(max(int(lag), 0), n - 1)
    long_run_variance = float(np.dot(centered, centered) / n)
    for current_lag in range(1, max_lag + 1):
        weight = 1.0 - current_lag / (max_lag + 1.0)
        covariance = float(
            np.dot(centered[current_lag:], centered[:-current_lag]) / n
        )
        long_run_variance += 2.0 * weight * covariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    se = sqrt(variance_of_mean)
    t_stat = mean / se if se > 0 else np.nan
    p_value = (
        float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        if np.isfinite(t_stat)
        else np.nan
    )
    return {"mean": mean, "se": se, "t": t_stat, "p": p_value}


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg false-discovery-rate adjusted p-values."""

    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    ranked = values[order]
    m = len(ranked)
    raw_adjusted = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted


def summarize_event_subset(
    frame: pd.DataFrame,
    *,
    horizon: int,
    round_trip_cost_bps: float = 20.0,
) -> dict[str, float | int]:
    """Summarize one threshold/horizon subset using signal-date clusters."""

    if frame.empty:
        return {
            "events": 0,
            "symbols": 0,
            "signal_dates": 0,
            "mean_net_return": np.nan,
            "median_net_return": np.nan,
            "win_rate": np.nan,
            "mean_abnormal_net_return": np.nan,
            "cluster_mean_abnormal_net_return": np.nan,
            "cluster_ci95_low": np.nan,
            "cluster_ci95_high": np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
            "mean_mfe": np.nan,
            "median_mfe": np.nan,
            "mean_mae": np.nan,
            "prob_mfe_3pct": np.nan,
            "prob_mfe_5pct": np.nan,
        }
    cost = float(round_trip_cost_bps) / 10_000.0
    work = frame.copy()
    work["net_return"] = work["executable_return"] - cost
    work["abnormal_net_return"] = work["abnormal_executable_return"] - cost
    date_cohorts = (
        work.groupby("signal_date", sort=True)["abnormal_net_return"]
        .mean()
        .dropna()
    )
    test = newey_west_mean_test(date_cohorts.to_numpy(), lag=horizon)
    return {
        "events": int(len(work)),
        "symbols": int(work["symbol"].nunique()),
        "signal_dates": int(work["signal_date"].nunique()),
        "mean_net_return": float(work["net_return"].mean()),
        "median_net_return": float(work["net_return"].median()),
        "win_rate": float((work["net_return"] > 0).mean()),
        "mean_abnormal_net_return": float(work["abnormal_net_return"].mean()),
        "cluster_mean_abnormal_net_return": float(test["mean"]),
        "cluster_ci95_low": float(test["mean"] - 1.96 * test["se"]),
        "cluster_ci95_high": float(test["mean"] + 1.96 * test["se"]),
        "hac_t": float(test["t"]),
        "hac_p": float(test["p"]),
        "mean_mfe": float(work["mfe"].mean()),
        "median_mfe": float(work["mfe"].median()),
        "mean_mae": float(work["mae"].mean()),
        "prob_mfe_3pct": float((work["mfe"] >= 0.03).mean()),
        "prob_mfe_5pct": float((work["mfe"] >= 0.05).mean()),
    }


def paired_incremental_j_test(
    low9_frame: pd.DataFrame,
    *,
    threshold: float,
    horizon: int,
) -> dict[str, float | int]:
    """Compare J-selected and J-rejected low-9 cohorts on common dates."""

    selected = low9_frame[low9_frame["j_value"] <= threshold]
    rejected = low9_frame[low9_frame["j_value"] > threshold]
    selected_dates = selected.groupby("signal_date")["abnormal_executable_return"].mean()
    rejected_dates = rejected.groupby("signal_date")["abnormal_executable_return"].mean()
    paired = pd.concat(
        [selected_dates.rename("selected"), rejected_dates.rename("rejected")],
        axis=1,
        join="inner",
    ).dropna()
    difference = paired["selected"] - paired["rejected"]
    test = newey_west_mean_test(difference.to_numpy(), lag=horizon)
    return {
        "paired_dates": int(len(paired)),
        "incremental_mean": float(test["mean"]),
        "incremental_hac_t": float(test["t"]),
        "incremental_hac_p": float(test["p"]),
        "incremental_ci95_low": float(test["mean"] - 1.96 * test["se"]),
        "incremental_ci95_high": float(test["mean"] + 1.96 * test["se"]),
    }
