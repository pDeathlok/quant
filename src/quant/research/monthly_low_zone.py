"""Causal monthly low-zone signals built from low-9 and multi-timeframe KDJ."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from quant.data.factors import KDJ


SIGNAL_RULES = (
    "monthly_low9",
    "monthly_j_le_minus10",
    "monthly_j_le_minus20",
    "monthly_low9_j_negative",
    "monthly_weekly_j_le_minus10",
    "monthly_low9_weekly_j_le_minus10",
    "monthly_j_le_minus10_weekly_reclaim",
)

EVENT_PERIODS = {
    "development_2013_2016": (pd.Timestamp("2013-01-01"), pd.Timestamp("2016-12-31")),
    "validation_2017_2020": (pd.Timestamp("2017-01-01"), pd.Timestamp("2020-12-31")),
    "seen_diagnostic_2021_2024": (
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
}


@dataclass(frozen=True)
class MonthlyLowZoneConfig:
    """Frozen research gates for completed monthly low-zone events."""

    minimum_history_months: int = 36
    minimum_drawdown_from_prior_peak: float = 0.50
    minimum_median_daily_amount_thousand: float = 30_000.0
    maximum_signal_staleness_sessions: int = 5
    maximum_entry_delay_sessions: int = 5
    signal_cooldown_months: int = 12
    monthly_j_threshold: float = -10.0
    monthly_extreme_j_threshold: float = -20.0
    weekly_j_threshold: float = -10.0
    maximum_missing_market_sessions: int = 60
    round_trip_cost_bps: float = 20.0
    horizons: tuple[int, int, int] = (126, 252, 504)

    def __post_init__(self) -> None:
        if self.minimum_history_months < 1:
            raise ValueError("minimum_history_months must be positive")
        if not 0.0 < self.minimum_drawdown_from_prior_peak < 1.0:
            raise ValueError("minimum_drawdown_from_prior_peak must be in (0, 1)")
        if self.minimum_median_daily_amount_thousand < 0.0:
            raise ValueError("minimum_median_daily_amount_thousand must be non-negative")
        if self.maximum_signal_staleness_sessions < 0:
            raise ValueError("maximum_signal_staleness_sessions must be non-negative")
        if self.maximum_entry_delay_sessions < 1:
            raise ValueError("maximum_entry_delay_sessions must be positive")
        if self.signal_cooldown_months < 1:
            raise ValueError("signal_cooldown_months must be positive")
        if self.maximum_missing_market_sessions < 1:
            raise ValueError("maximum_missing_market_sessions must be positive")
        if self.round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps must be non-negative")
        if len(self.horizons) != 3 or any(value < 1 for value in self.horizons):
            raise ValueError("horizons must contain three positive session counts")
        if tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("horizons must be increasing")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalize_calendar(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce")).dropna()
    normalized = pd.DatetimeIndex(sorted(normalized.normalize().unique()))
    if normalized.empty:
        raise ValueError("market_calendar must not be empty")
    return normalized


def _add_group_kdj(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("ts_code", observed=True, sort=False):
        group = group.copy()
        price = group[["adjusted_high", "adjusted_low", "adjusted_close"]].rename(
            columns={
                "adjusted_high": "high",
                "adjusted_low": "low",
                "adjusted_close": "close",
            }
        )
        values = KDJ().compute(price.reset_index(drop=True))
        group[f"{prefix}_k"] = values["K"].to_numpy()
        group[f"{prefix}_d"] = values["D"].to_numpy()
        group[f"{prefix}_j"] = values["J"].to_numpy()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else frame.copy()


def _consecutive_monthly_low9(group: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(group["adjusted_close"], errors="coerce")
    month_ordinal = pd.PeriodIndex(group["month_period"], freq="M").asi8
    lag_close = close.shift(4)
    lag_ordinal = pd.Series(month_ordinal, index=group.index).shift(4)
    prior_ordinal = pd.Series(month_ordinal, index=group.index).shift(1)
    condition = close.lt(lag_close) & (month_ordinal - lag_ordinal == 4)
    consecutive_calendar = (month_ordinal - prior_ordinal == 1) | prior_ordinal.isna()
    counts: list[int] = []
    current = 0
    for flag, adjacent in zip(condition.fillna(False), consecutive_calendar.fillna(False)):
        current = current + 1 if bool(flag) and bool(adjacent) else 0
        counts.append(current)
    return pd.Series(counts, index=group.index, dtype=int)


def build_monthly_weekly_features(
    daily: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return completed monthly features and completed W-FRI KDJ bars."""

    required = {
        "ts_code",
        "date",
        "amount",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    }
    _require_columns(daily, required, "daily")
    calendar = _normalize_calendar(market_calendar)
    panel = daily.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["ts_code", "date"])
    panel = panel.loc[panel["date"].between(calendar[0], calendar[-1], inclusive="both")]
    panel = panel.sort_values(["ts_code", "date"]).drop_duplicates(
        ["ts_code", "date"], keep="last"
    )
    panel["month_period"] = panel["date"].dt.to_period("M").astype(str)
    calendar_frame = pd.DataFrame({"date": calendar})
    calendar_frame["month_period"] = calendar_frame["date"].dt.to_period("M").astype(str)
    month_ends = calendar_frame.groupby("month_period", observed=True)["date"].max()
    month_ends = month_ends.rename("signal_date")

    monthly = (
        panel.groupby(["ts_code", "month_period"], as_index=False, observed=True)
        .agg(
            adjusted_open=("adjusted_open", "first"),
            adjusted_high=("adjusted_high", "max"),
            adjusted_low=("adjusted_low", "min"),
            adjusted_close=("adjusted_close", "last"),
            median_daily_amount=("amount", "median"),
            last_trade_date=("date", "max"),
            trading_days=("date", "count"),
        )
        .merge(month_ends, on="month_period", how="inner", validate="many_to_one")
        .sort_values(["ts_code", "signal_date"])
        .reset_index(drop=True)
    )
    signal_positions = calendar.searchsorted(
        pd.DatetimeIndex(monthly["signal_date"]), side="left"
    )
    last_positions = calendar.searchsorted(
        pd.DatetimeIndex(monthly["last_trade_date"]), side="right"
    ) - 1
    monthly["signal_staleness_sessions"] = signal_positions - last_positions
    monthly["history_months"] = monthly.groupby(
        "ts_code", observed=True, sort=False
    ).cumcount() + 1
    monthly["prior_peak"] = monthly.groupby(
        "ts_code", observed=True, sort=False
    )["adjusted_high"].transform(lambda values: values.cummax().shift(1))
    monthly["drawdown_from_prior_peak"] = (
        monthly["adjusted_close"] / monthly["prior_peak"].replace(0.0, np.nan) - 1.0
    )
    monthly = _add_group_kdj(monthly, prefix="monthly")
    low9_parts: list[pd.Series] = []
    for _, group in monthly.groupby("ts_code", observed=True, sort=False):
        low9_parts.append(_consecutive_monthly_low9(group))
    monthly["monthly_low9_count"] = (
        pd.concat(low9_parts).sort_index() if low9_parts else pd.Series(dtype=int)
    )
    monthly["monthly_low9"] = monthly["monthly_low9_count"].eq(9)

    panel["weekly_available_date"] = (
        panel["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    )
    weekly = (
        panel.groupby(["ts_code", "weekly_available_date"], as_index=False, observed=True)
        .agg(
            adjusted_open=("adjusted_open", "first"),
            adjusted_high=("adjusted_high", "max"),
            adjusted_low=("adjusted_low", "min"),
            adjusted_close=("adjusted_close", "last"),
            weekly_last_trade_date=("date", "max"),
        )
        .sort_values(["ts_code", "weekly_available_date"])
        .reset_index(drop=True)
    )
    weekly = weekly.loc[weekly["weekly_available_date"].le(calendar[-1])].copy()
    weekly = _add_group_kdj(weekly, prefix="weekly")
    weekly["weekly_prev_j"] = weekly.groupby(
        "ts_code", observed=True, sort=False
    )["weekly_j"].shift(1)
    return monthly.reset_index(drop=True), weekly.reset_index(drop=True)


def _attach_weekly_state(
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    out = monthly.copy().reset_index(drop=True)
    out["_monthly_order"] = np.arange(len(out), dtype=np.int64)
    for column in (
        "weekly_available_date",
        "weekly_j",
        "weekly_prev_j",
        "weekly_reclaim_in_signal_month",
    ):
        if column not in out:
            out[column] = np.nan
    if weekly.empty:
        return out.drop(columns="_monthly_order")
    required = {"ts_code", "weekly_available_date", "weekly_j", "weekly_prev_j"}
    _require_columns(weekly, required, "weekly")
    right = weekly.copy()
    right["weekly_available_date"] = pd.to_datetime(
        right["weekly_available_date"], errors="coerce"
    )
    right["weekly_month_period"] = right["weekly_available_date"].dt.to_period("M").astype(str)
    right["weekly_reclaim"] = right["weekly_prev_j"].le(threshold) & right[
        "weekly_j"
    ].gt(threshold)
    right["weekly_reclaim_cum_in_month"] = right.groupby(
        ["ts_code", "weekly_month_period"], observed=True, sort=False
    )["weekly_reclaim"].cumsum()
    right_by_symbol = {
        str(code): group.sort_values("weekly_available_date")
        for code, group in right.groupby("ts_code", observed=True, sort=False)
    }
    parts: list[pd.DataFrame] = []
    for code, group in out.groupby("ts_code", observed=True, sort=False):
        available = right_by_symbol.get(str(code))
        if available is None or available.empty:
            parts.append(group)
            continue
        drop_existing = [
            "weekly_available_date",
            "weekly_j",
            "weekly_prev_j",
            "weekly_reclaim_in_signal_month",
        ]
        left = group.drop(columns=drop_existing, errors="ignore")
        aligned = pd.merge_asof(
            left.sort_values("signal_date"),
            available[
                [
                    "weekly_available_date",
                    "weekly_j",
                    "weekly_prev_j",
                    "weekly_month_period",
                    "weekly_reclaim_cum_in_month",
                ]
            ],
            left_on="signal_date",
            right_on="weekly_available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned["weekly_reclaim_in_signal_month"] = (
            aligned["weekly_month_period"].eq(aligned["month_period"])
            & aligned["weekly_reclaim_cum_in_month"].gt(0)
        )
        parts.append(aligned)
    return (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values("_monthly_order")
        .drop(columns=["_monthly_order", "weekly_month_period", "weekly_reclaim_cum_in_month"])
        .reset_index(drop=True)
    )


def _cooldown_onsets(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    cooldown_months: int,
    onset_only: bool,
) -> pd.Series:
    chosen = pd.Series(False, index=frame.index, dtype=bool)
    values = mask.fillna(False).astype(bool)
    for _, group in frame.groupby("ts_code", observed=True, sort=False):
        group_values = values.loc[group.index]
        candidates = group_values & (~group_values.shift(1, fill_value=False) if onset_only else True)
        ordinals = pd.PeriodIndex(group["month_period"], freq="M").asi8
        last_ordinal = -10**9
        for index, ordinal, candidate in zip(group.index, ordinals, candidates):
            if bool(candidate) and int(ordinal) - last_ordinal >= cooldown_months:
                chosen.loc[index] = True
                last_ordinal = int(ordinal)
    return chosen


def generate_monthly_low_zone_signals(
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    config: MonthlyLowZoneConfig,
) -> pd.DataFrame:
    """Return the seven pre-registered month-end signal families."""

    required = {
        "ts_code",
        "month_period",
        "signal_date",
        "median_daily_amount",
        "history_months",
        "prior_peak",
        "drawdown_from_prior_peak",
        "signal_staleness_sessions",
        "monthly_j",
        "monthly_low9",
        "monthly_low9_count",
    }
    _require_columns(monthly, required, "monthly")
    out = monthly.copy().sort_values(["ts_code", "signal_date"]).reset_index(drop=True)
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out = _attach_weekly_state(out, weekly, config.weekly_j_threshold)
    eligible = (
        out["history_months"].ge(config.minimum_history_months)
        & out["drawdown_from_prior_peak"].le(
            -config.minimum_drawdown_from_prior_peak
        )
        & out["median_daily_amount"].ge(
            config.minimum_median_daily_amount_thousand
        )
        & out["signal_staleness_sessions"].le(
            config.maximum_signal_staleness_sessions
        )
        & out["monthly_j"].notna()
    ).fillna(False)
    monthly_low9 = eligible & out["monthly_low9"].fillna(False)
    monthly_j_low = eligible & out["monthly_j"].le(config.monthly_j_threshold)
    monthly_j_extreme = eligible & out["monthly_j"].le(
        config.monthly_extreme_j_threshold
    )
    weekly_j_low = out["weekly_j"].le(config.weekly_j_threshold)
    rule_masks: dict[str, tuple[pd.Series, bool]] = {
        "monthly_low9": (monthly_low9, False),
        "monthly_j_le_minus10": (monthly_j_low, True),
        "monthly_j_le_minus20": (monthly_j_extreme, True),
        "monthly_low9_j_negative": (
            monthly_low9 & out["monthly_j"].lt(0.0),
            False,
        ),
        "monthly_weekly_j_le_minus10": (monthly_j_low & weekly_j_low, True),
        "monthly_low9_weekly_j_le_minus10": (
            monthly_low9 & weekly_j_low,
            False,
        ),
        "monthly_j_le_minus10_weekly_reclaim": (
            monthly_j_low & out["weekly_reclaim_in_signal_month"].fillna(False),
            False,
        ),
    }
    pieces: list[pd.DataFrame] = []
    for rule in SIGNAL_RULES:
        mask, onset_only = rule_masks[rule]
        selected = _cooldown_onsets(
            out,
            mask,
            cooldown_months=config.signal_cooldown_months,
            onset_only=onset_only,
        )
        piece = out.loc[selected].copy()
        piece["rule"] = rule
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=[*out.columns, "rule", "signal_id"])
    signals = pd.concat(pieces, ignore_index=True, sort=False)
    signals = signals.sort_values(["signal_date", "ts_code", "rule"]).reset_index(drop=True)
    signals["signal_id"] = np.arange(1, len(signals) + 1, dtype=np.int64)
    return signals


def _benchmark_frame(benchmark: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "close"}
    _require_columns(benchmark, required, "benchmark")
    out = benchmark.copy()
    if "trade_date" in out:
        date_source = out["trade_date"]
    elif "date" in out:
        date_source = out["date"]
    else:
        raise ValueError("benchmark missing columns: ['trade_date or date']")
    parsed = pd.to_datetime(date_source, errors="coerce")
    compact = pd.to_datetime(
        date_source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    out["date"] = compact.fillna(parsed).dt.normalize()
    return out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def _event_base(signal: pd.Series, horizon: int) -> dict[str, Any]:
    keep = (
        "signal_id",
        "ts_code",
        "rule",
        "signal_date",
        "month_period",
        "monthly_j",
        "weekly_j",
        "monthly_low9",
        "monthly_low9_count",
        "drawdown_from_prior_peak",
        "median_daily_amount",
    )
    return {**{column: signal.get(column) for column in keep}, "horizon": horizon}


def evaluate_monthly_low_zone_events(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    benchmark: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: MonthlyLowZoneConfig,
) -> pd.DataFrame:
    """Resolve next-open entries and 126/252/504-session outcomes."""

    required_daily = {
        "ts_code",
        "date",
        "open",
        "high",
        "low",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    }
    required_signals = {"ts_code", "signal_date", "rule"}
    _require_columns(daily, required_daily, "daily")
    _require_columns(signals, required_signals, "signals")
    calendar = _normalize_calendar(market_calendar)
    market = _benchmark_frame(benchmark).set_index("date")
    market_open = pd.to_numeric(market["open"], errors="coerce").to_dict()
    market_close = pd.to_numeric(market["close"], errors="coerce").to_dict()
    calendar_positions = {pd.Timestamp(date): index for index, date in enumerate(calendar)}
    panel = daily.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel = panel.dropna(subset=["ts_code", "date"]).sort_values(["ts_code", "date"])
    panel = panel.drop_duplicates(["ts_code", "date"], keep="last")
    daily_by_symbol = {
        str(code): group.reset_index(drop=True)
        for code, group in panel.groupby("ts_code", observed=True, sort=False)
    }
    rows: list[dict[str, Any]] = []
    signals_frame = signals.copy().reset_index(drop=True)
    signals_frame["signal_date"] = pd.to_datetime(
        signals_frame["signal_date"], errors="coerce"
    ).dt.normalize()
    if "signal_id" not in signals_frame:
        signals_frame["signal_id"] = np.arange(1, len(signals_frame) + 1)
    for _, signal in signals_frame.iterrows():
        symbol = str(signal["ts_code"])
        signal_date = pd.Timestamp(signal["signal_date"])
        prices = daily_by_symbol.get(symbol)
        entry_status = "accepted"
        entry_row: pd.Series | None = None
        entry_position: int | None = None
        if prices is None or prices.empty:
            entry_status = "missing_future_bar"
        else:
            price_dates = pd.DatetimeIndex(prices["date"])
            entry_index = int(price_dates.searchsorted(signal_date, side="right"))
            if entry_index >= len(prices):
                entry_status = "missing_future_bar"
            else:
                candidate = prices.iloc[entry_index]
                entry_date = pd.Timestamp(candidate["date"])
                signal_market_position = int(calendar.searchsorted(signal_date, side="left"))
                entry_position = calendar_positions.get(entry_date)
                if entry_position is None:
                    entry_status = "entry_not_in_market_calendar"
                elif entry_position - signal_market_position > config.maximum_entry_delay_sessions:
                    entry_status = "entry_delay_exceeded"
                else:
                    raw_open = float(candidate["open"])
                    raw_high = float(candidate["high"])
                    raw_low = float(candidate["low"])
                    tolerance = max(abs(raw_open), 1.0) * 1e-10
                    if max(raw_high, raw_open) - min(raw_low, raw_open) <= tolerance:
                        entry_status = "one_price_entry"
                    elif not np.isfinite(float(candidate["adjusted_open"])) or float(
                        candidate["adjusted_open"]
                    ) <= 0.0:
                        entry_status = "invalid_entry_open"
                    else:
                        entry_row = candidate
        for horizon in config.horizons:
            record = _event_base(signal, horizon)
            record.update(
                {
                    "entry_status": entry_status,
                    "entry_date": pd.NaT,
                    "entry_open": np.nan,
                    "entry_delay_sessions": np.nan,
                    "target_date": pd.NaT,
                    "exit_date": pd.NaT,
                    "exit_reason": "entry_rejected" if entry_status != "accepted" else "",
                    "outcome_completed": False,
                    "gross_return": np.nan,
                    "net_return": np.nan,
                    "benchmark_return": np.nan,
                    "excess_net_return": np.nan,
                    "mae": np.nan,
                    "mfe": np.nan,
                }
            )
            if entry_status != "accepted" or entry_row is None or entry_position is None:
                rows.append(record)
                continue
            entry_date = pd.Timestamp(entry_row["date"])
            entry_open = float(entry_row["adjusted_open"])
            signal_market_position = int(calendar.searchsorted(signal_date, side="left"))
            record["entry_date"] = entry_date
            record["entry_open"] = entry_open
            record["entry_delay_sessions"] = entry_position - signal_market_position
            target_position = entry_position + horizon - 1
            if target_position >= len(calendar):
                record["exit_reason"] = "unresolved_at_cutoff"
                rows.append(record)
                continue
            target_date = pd.Timestamp(calendar[target_position])
            record["target_date"] = target_date
            price_dates = pd.DatetimeIndex(prices["date"])
            target_price_index = int(price_dates.searchsorted(target_date, side="left"))
            last_index = target_price_index - 1
            exact_target_bar = (
                target_price_index < len(prices)
                and pd.Timestamp(prices.iloc[target_price_index]["date"]) == target_date
            )
            if last_index < 0 and not exact_target_bar:
                record["exit_reason"] = "missing_bar_writeoff"
                record["outcome_completed"] = True
                record["net_return"] = -1.0
                record["gross_return"] = -1.0
                record["mae"] = -1.0
                record["mfe"] = 0.0
                rows.append(record)
                continue
            entry_market_open = market_open.get(entry_date, np.nan)
            target_market_close = market_close.get(target_date, np.nan)
            benchmark_return = (
                float(target_market_close / entry_market_open - 1.0)
                if np.isfinite(entry_market_open)
                and np.isfinite(target_market_close)
                and entry_market_open > 0.0
                else np.nan
            )
            if exact_target_bar:
                exit_row = prices.iloc[target_price_index]
                exit_date = target_date
                exit_value = float(exit_row["adjusted_close"])
                exit_reason = "time_exit"
                path_end = exit_date
            else:
                last_row = prices.iloc[last_index]
                last_date = pd.Timestamp(last_row["date"])
                last_market_position = calendar_positions.get(last_date)
                missing_at_target = (
                    target_position - last_market_position
                    if last_market_position is not None
                    else config.maximum_missing_market_sessions
                )
                recovery_row = (
                    prices.iloc[target_price_index]
                    if target_price_index < len(prices)
                    else None
                )
                recovery_position = (
                    calendar_positions.get(pd.Timestamp(recovery_row["date"]))
                    if recovery_row is not None
                    else None
                )
                cutoff_position = len(calendar) - 1
                missing_until_resolution = (
                    recovery_position - last_market_position
                    if recovery_position is not None and last_market_position is not None
                    else cutoff_position - last_market_position
                    if last_market_position is not None
                    else config.maximum_missing_market_sessions
                )
                if (
                    missing_at_target < config.maximum_missing_market_sessions
                    and missing_until_resolution < config.maximum_missing_market_sessions
                    and recovery_row is None
                ):
                    record["exit_reason"] = "unresolved_suspension_at_cutoff"
                    rows.append(record)
                    continue
                if (
                    missing_at_target < config.maximum_missing_market_sessions
                    and missing_until_resolution < config.maximum_missing_market_sessions
                    and recovery_row is not None
                ):
                    exit_row = recovery_row
                    exit_date = pd.Timestamp(exit_row["date"])
                    exit_value = float(exit_row["adjusted_open"])
                    exit_reason = "next_open_after_target_suspension"
                    path_end = exit_date
                    delayed_market_open = market_open.get(exit_date, np.nan)
                    benchmark_return = (
                        float(delayed_market_open / entry_market_open - 1.0)
                        if np.isfinite(entry_market_open)
                        and np.isfinite(delayed_market_open)
                        and entry_market_open > 0.0
                        else np.nan
                    )
                else:
                    exit_row = None
                    exit_date = target_date
                    exit_value = np.nan
                    exit_reason = "missing_bar_writeoff"
                    path_end = target_date
            path = prices.loc[
                prices["date"].between(entry_date, path_end, inclusive="both")
            ]
            if exit_reason == "missing_bar_writeoff":
                gross_return = -1.0
                net_return = -1.0
                mae = -1.0
                mfe = (
                    float(path["adjusted_high"].max() / entry_open - 1.0)
                    if not path.empty
                    else 0.0
                )
            else:
                gross_return = exit_value / entry_open - 1.0
                net_return = gross_return - config.round_trip_cost_bps / 10_000.0
                mae = float(path["adjusted_low"].min() / entry_open - 1.0)
                mfe = float(path["adjusted_high"].max() / entry_open - 1.0)
            record.update(
                {
                    "exit_date": exit_date,
                    "exit_reason": exit_reason,
                    "outcome_completed": True,
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "excess_net_return": (
                        net_return - benchmark_return
                        if np.isfinite(benchmark_return)
                        else np.nan
                    ),
                    "mae": mae,
                    "mfe": mfe,
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    spread = z * sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (center - spread) / denominator


def _date_cluster_summary(frame: pd.DataFrame, column: str) -> tuple[float, float, float]:
    cohorts = frame.groupby("signal_date", observed=True)[column].mean().dropna()
    if cohorts.empty:
        return np.nan, np.nan, np.nan
    mean = float(cohorts.mean())
    if len(cohorts) < 2:
        return mean, np.nan, np.nan
    se = float(cohorts.std(ddof=1) / sqrt(len(cohorts)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarize_monthly_low_zone_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize rule, period and horizon with clustered path metrics."""

    required = {
        "rule",
        "signal_date",
        "horizon",
        "entry_status",
        "outcome_completed",
        "net_return",
        "excess_net_return",
        "exit_reason",
        "mae",
        "mfe",
    }
    _require_columns(events, required, "events")
    frame = events.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for period, (start, end) in EVENT_PERIODS.items():
        scoped = frame.loc[frame["signal_date"].between(start, end, inclusive="both")]
        for rule in SIGNAL_RULES:
            rule_frame = scoped.loc[scoped["rule"].eq(rule)]
            for horizon in sorted(frame["horizon"].dropna().unique()):
                all_rows = rule_frame.loc[rule_frame["horizon"].eq(horizon)]
                accepted = all_rows.loc[all_rows["entry_status"].eq("accepted")]
                completed = accepted.loc[accepted["outcome_completed"].fillna(False)]
                returns = pd.to_numeric(completed["net_return"], errors="coerce").dropna()
                excess = pd.to_numeric(
                    completed["excess_net_return"], errors="coerce"
                ).dropna()
                gains = float(returns.loc[returns > 0.0].sum())
                losses = float(-returns.loc[returns <= 0.0].sum())
                wins = int(returns.gt(0.0).sum())
                cluster_mean, cluster_low, cluster_high = _date_cluster_summary(
                    completed, "net_return"
                )
                rows.append(
                    {
                        "period": period,
                        "rule": rule,
                        "horizon": int(horizon),
                        "signals": int(len(all_rows)),
                        "accepted_entries": int(len(accepted)),
                        "completed_events": int(len(completed)),
                        "symbols": int(completed["ts_code"].nunique())
                        if "ts_code" in completed
                        else 0,
                        "signal_dates": int(completed["signal_date"].nunique()),
                        "unresolved_events": int(
                            accepted["exit_reason"].eq("unresolved_at_cutoff").sum()
                        ),
                        "writeoff_events": int(
                            completed["exit_reason"].eq("missing_bar_writeoff").sum()
                        ),
                        "mean_net_return": float(returns.mean()) if len(returns) else np.nan,
                        "median_net_return": float(returns.median())
                        if len(returns)
                        else np.nan,
                        "win_rate": wins / len(returns) if len(returns) else np.nan,
                        "win_rate_wilson_lower_95": _wilson_lower(wins, len(returns)),
                        "profit_factor": gains / losses if losses > 0.0 else np.nan,
                        "mean_excess_net_return": float(excess.mean())
                        if len(excess)
                        else np.nan,
                        "excess_win_rate": float(excess.gt(0.0).mean())
                        if len(excess)
                        else np.nan,
                        "mean_mae": float(pd.to_numeric(completed["mae"], errors="coerce").mean())
                        if len(completed)
                        else np.nan,
                        "mean_mfe": float(pd.to_numeric(completed["mfe"], errors="coerce").mean())
                        if len(completed)
                        else np.nan,
                        "writeoff_rate": float(
                            completed["exit_reason"].eq("missing_bar_writeoff").mean()
                        )
                        if len(completed)
                        else np.nan,
                        "date_equal_mean_net_return": cluster_mean,
                        "date_cluster_ci95_low": cluster_low,
                        "date_cluster_ci95_high": cluster_high,
                    }
                )
    return pd.DataFrame(rows)
