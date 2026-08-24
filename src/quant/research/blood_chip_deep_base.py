"""Point-in-time research for deep-drawdown, exhausted-selling price bases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd

from quant.research import blood_chip as core
from quant.research.blood_chip import BloodChipBacktestResult


SIGNAL_COLUMNS = [
    "ts_code",
    "base_event_id",
    "signal_date",
    "entry_date",
    "entry_open",
    "signal_close",
    "signal_score",
    "prior_peak",
    "prior_peak_date",
    "peak_age_sessions",
    "drawdown_from_peak",
    "deep_drawdown_sessions",
    "base_low",
    "base_high",
    "base_mid",
    "base_range",
    "base_return",
    "base_position",
    "sessions_since_new_low",
    "recent_down_amount_share",
    "prior_down_amount_share",
    "down_amount_share_ratio",
    "volatility_20d",
    "volatility_prior_60d",
    "volatility_contraction_ratio",
    "prior_amount_median_20d",
]


@dataclass(frozen=True)
class DeepBaseSignalConfig:
    """Frozen causal gates for a deep drawdown followed by a quiet base."""

    minimum_history_days: int = 500
    minimum_prior_amount_thousand: float = 30_000.0
    minimum_drawdown_from_peak: float = 0.50
    minimum_peak_age_sessions: int = 120
    minimum_deep_drawdown_sessions: int = 40
    base_window_sessions: int = 60
    maximum_base_range: float = 0.25
    minimum_base_return: float = -0.12
    maximum_base_return: float = 0.12
    minimum_base_position: float = 0.15
    maximum_base_position: float = 0.65
    minimum_sessions_since_new_low: int = 10
    maximum_recent_down_amount_share: float = 0.35
    maximum_down_amount_share_ratio: float = 0.80
    maximum_volatility_contraction_ratio: float = 0.85
    signal_cooldown_sessions: int = 120

    def __post_init__(self) -> None:
        if self.minimum_history_days < self.base_window_sessions:
            raise ValueError("minimum_history_days must cover the base window")
        if not 0.0 < self.minimum_drawdown_from_peak < 1.0:
            raise ValueError("minimum_drawdown_from_peak must be in (0, 1)")
        if self.minimum_peak_age_sessions < 1:
            raise ValueError("minimum_peak_age_sessions must be positive")
        if self.minimum_deep_drawdown_sessions < 1:
            raise ValueError("minimum_deep_drawdown_sessions must be positive")
        if self.base_window_sessions != 60:
            raise ValueError("the frozen research contract requires a 60-session base")
        if self.minimum_base_return > self.maximum_base_return:
            raise ValueError("minimum_base_return must not exceed maximum_base_return")
        if self.minimum_base_position > self.maximum_base_position:
            raise ValueError("minimum_base_position must not exceed maximum_base_position")
        for field_name in (
            "maximum_base_range",
            "maximum_recent_down_amount_share",
            "maximum_down_amount_share_ratio",
            "maximum_volatility_contraction_ratio",
        ):
            if float(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeepBaseExecutionConfig:
    """Portfolio, staged construction, exit and cost policy."""

    initial_cash: float = 1_000_000.0
    maximum_positions: int = 10
    target_position_fraction: float = 0.10
    tranche_fractions: tuple[float, float, float] = (0.20, 0.30, 0.50)
    stage_policy: Literal["lower_retest", "retest_reclaim"] = "lower_retest"
    second_stage_minimum_sessions: int = 10
    third_stage_minimum_sessions: int = 20
    maximum_scale_in_sessions: int = 120
    hard_stop_below_base: float = 0.10
    structural_break_below_base: float = 0.03
    structural_break_sessions: int = 2
    structural_exit_enabled: bool = True
    maximum_missing_market_sessions: int = 60
    maximum_holding_sessions: int = 500
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    minimum_commission: float = 5.0
    slippage: float = 0.0005
    minimum_entry_gap: float = -0.07
    maximum_entry_gap: float = 0.07
    locked_limit_threshold: float = 0.048
    lot_size: int = 100

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.maximum_positions < 1:
            raise ValueError("maximum_positions must be positive")
        if not 0.0 < self.target_position_fraction <= 1.0:
            raise ValueError("target_position_fraction must be in (0, 1]")
        if len(self.tranche_fractions) != 3 or any(
            value <= 0 for value in self.tranche_fractions
        ):
            raise ValueError("tranche_fractions must contain three positive values")
        if not np.isclose(sum(self.tranche_fractions), 1.0):
            raise ValueError("tranche_fractions must sum to one")
        if self.second_stage_minimum_sessions < 1:
            raise ValueError("second_stage_minimum_sessions must be positive")
        if self.third_stage_minimum_sessions < self.second_stage_minimum_sessions:
            raise ValueError("third stage must not precede second stage")
        if self.maximum_scale_in_sessions < self.third_stage_minimum_sessions:
            raise ValueError("maximum_scale_in_sessions is too short")
        if not 0.0 < self.hard_stop_below_base < 1.0:
            raise ValueError("hard_stop_below_base must be in (0, 1)")
        if not 0.0 < self.structural_break_below_base < 1.0:
            raise ValueError("structural_break_below_base must be in (0, 1)")
        if self.structural_break_sessions < 1:
            raise ValueError("structural_break_sessions must be positive")
        if self.maximum_missing_market_sessions < 1:
            raise ValueError("maximum_missing_market_sessions must be positive")
        if self.maximum_holding_sessions < 1:
            raise ValueError("maximum_holding_sessions must be positive")
        if self.maximum_entry_gap < self.minimum_entry_gap:
            raise ValueError("maximum_entry_gap must not be below minimum_entry_gap")
        if self.lot_size < 1:
            raise ValueError("lot_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _DeepBasePosition:
    symbol: str
    event_id: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    signal_close: float
    base_low: float
    base_high: float
    base_mid: float
    hard_stop: float
    target_budget: float
    adjusted_units: float
    raw_shares: int
    entry_value: float
    invested_value: float
    buy_fees: float
    sessions_held: int
    last_close: float
    missing_market_sessions: int
    lower_retest_seen: bool
    broken_close_count: int
    structural_exit_ready: bool
    ready_stage: int | None
    tranches_filled: int
    tranche_dates: list[pd.Timestamp]
    signal_payload: dict[str, Any]


def _rolling(
    values: pd.Series,
    groups: pd.Series,
    window: int,
    operation: str,
    *,
    minimum_periods: int,
) -> pd.Series:
    return core._rolling(
        values,
        groups,
        window,
        operation,
        minimum_periods=minimum_periods,
    )


def _consecutive_true(flags: pd.Series, groups: pd.Series) -> pd.Series:
    clean = flags.fillna(False).astype(bool)
    reset = (~clean).astype(int).groupby(groups, observed=True, sort=False).cumsum()
    return clean.astype(int).groupby(
        [groups, reset],
        observed=True,
        sort=False,
    ).cumsum()


def build_deep_base_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build causal expanding-peak, selling-exhaustion and base features."""

    out = core._add_causal_continuous_prices(core._prepare_daily(daily))
    groups = out["ts_code"]
    positions = out.groupby(groups, observed=True, sort=False).cumcount()
    out["history_days"] = positions + 1

    previous_peak = out.groupby(groups, observed=True, sort=False)[
        "adjusted_high"
    ].cummax().groupby(groups, observed=True, sort=False).shift(1)
    out["prior_peak"] = previous_peak
    new_peak = previous_peak.isna() | out["adjusted_high"].gt(previous_peak)
    current_peak_date = out["date"].where(new_peak).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    current_peak_position = positions.where(new_peak).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    out["prior_peak_date"] = current_peak_date.groupby(
        groups,
        observed=True,
        sort=False,
    ).shift(1)
    prior_peak_position = current_peak_position.groupby(
        groups,
        observed=True,
        sort=False,
    ).shift(1)
    out["peak_age_sessions"] = positions - prior_peak_position
    out["drawdown_from_peak"] = (
        out["adjusted_close"] / out["prior_peak"].replace(0, np.nan) - 1.0
    )

    out["return_1d"] = out.groupby(groups, observed=True, sort=False)[
        "adjusted_close"
    ].pct_change(fill_method=None)
    out["prior_amount_median_20d"] = _rolling(
        out.groupby(groups, observed=True, sort=False)["amount"].shift(1),
        groups,
        20,
        "median",
        minimum_periods=15,
    )

    out["base_high"] = _rolling(
        out["adjusted_high"], groups, 60, "max", minimum_periods=60
    )
    out["base_low"] = _rolling(
        out["adjusted_low"], groups, 60, "min", minimum_periods=60
    )
    out["base_mid"] = (out["base_high"] + out["base_low"]) / 2.0
    out["base_range"] = out["base_high"] / out["base_low"].replace(0, np.nan) - 1.0
    prior_59_close = out.groupby(groups, observed=True, sort=False)[
        "adjusted_close"
    ].shift(59)
    out["base_return"] = (
        out["adjusted_close"] / prior_59_close.replace(0, np.nan) - 1.0
    )
    base_width = (out["base_high"] - out["base_low"]).replace(0, np.nan)
    out["base_position"] = (
        out["adjusted_close"] - out["base_low"]
    ) / base_width

    touches_rolling_low = out["adjusted_low"].le(
        out["base_low"] * (1.0 + 1e-10)
    ) & out["base_low"].notna()
    last_low_position = positions.where(touches_rolling_low).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    out["sessions_since_new_low"] = positions - last_low_position

    negative = out["return_1d"].lt(0.0)
    down_amount = out["amount"].where(negative, 0.0)
    recent_down_amount = _rolling(
        down_amount, groups, 20, "sum", minimum_periods=15
    )
    recent_total_amount = _rolling(
        out["amount"], groups, 20, "sum", minimum_periods=15
    )
    prior_down_amount = _rolling(
        down_amount.groupby(groups, observed=True, sort=False).shift(20),
        groups,
        60,
        "sum",
        minimum_periods=40,
    )
    prior_total_amount = _rolling(
        out.groupby(groups, observed=True, sort=False)["amount"].shift(20),
        groups,
        60,
        "sum",
        minimum_periods=40,
    )
    out["recent_down_amount_share"] = recent_down_amount / recent_total_amount.replace(
        0, np.nan
    )
    out["prior_down_amount_share"] = prior_down_amount / prior_total_amount.replace(
        0, np.nan
    )
    out["down_amount_share_ratio"] = out["recent_down_amount_share"] / out[
        "prior_down_amount_share"
    ].replace(0, np.nan)

    out["volatility_20d"] = _rolling(
        out["return_1d"], groups, 20, "std", minimum_periods=15
    ) * sqrt(252.0)
    out["volatility_prior_60d"] = _rolling(
        out.groupby(groups, observed=True, sort=False)["return_1d"].shift(20),
        groups,
        60,
        "std",
        minimum_periods=40,
    ) * sqrt(252.0)
    out["volatility_contraction_ratio"] = out["volatility_20d"] / out[
        "volatility_prior_60d"
    ].replace(0, np.nan)
    out["return_20d"] = out.groupby(groups, observed=True, sort=False)[
        "adjusted_close"
    ].pct_change(20, fill_method=None)
    return out


def _cooldown_signal_flags(
    eligible: pd.Series,
    groups: pd.Series,
    cooldown: int,
) -> pd.Series:
    flags = pd.Series(False, index=eligible.index, dtype=bool)
    eligible_frame = pd.DataFrame(
        {
            "group": groups,
            "eligible": eligible.fillna(False),
            "position": pd.Series(np.arange(len(eligible)), index=eligible.index),
        }
    )
    for _, frame in eligible_frame.loc[eligible_frame["eligible"]].groupby(
        "group",
        observed=True,
        sort=False,
    ):
        last_position = -cooldown
        for index, position in frame["position"].items():
            current = int(position)
            if current - last_position >= cooldown:
                flags.loc[index] = True
                last_position = current
    return flags


def generate_deep_base_signals(
    features: pd.DataFrame,
    config: DeepBaseSignalConfig,
) -> pd.DataFrame:
    """Generate the first eligible signal after each per-symbol cooldown."""

    required = {
        "ts_code",
        "date",
        "adjusted_open",
        "adjusted_close",
        "history_days",
        "prior_peak",
        "prior_peak_date",
        "peak_age_sessions",
        "drawdown_from_peak",
        "prior_amount_median_20d",
        "base_low",
        "base_high",
        "base_mid",
        "base_range",
        "base_return",
        "base_position",
        "sessions_since_new_low",
        "recent_down_amount_share",
        "prior_down_amount_share",
        "down_amount_share_ratio",
        "volatility_20d",
        "volatility_prior_60d",
        "volatility_contraction_ratio",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    out = features.copy().sort_values(["ts_code", "date"]).reset_index(drop=True)
    groups = out["ts_code"]
    deep = out["drawdown_from_peak"].le(-config.minimum_drawdown_from_peak)
    out["deep_drawdown_sessions"] = _consecutive_true(deep, groups)
    selling_contracted = (
        out["recent_down_amount_share"].le(
            config.maximum_recent_down_amount_share
        )
        | out["down_amount_share_ratio"].le(
            config.maximum_down_amount_share_ratio
        )
    )
    eligible = (
        out["history_days"].ge(config.minimum_history_days)
        & out["prior_amount_median_20d"].ge(
            config.minimum_prior_amount_thousand
        )
        & deep
        & out["peak_age_sessions"].ge(config.minimum_peak_age_sessions)
        & out["deep_drawdown_sessions"].ge(
            config.minimum_deep_drawdown_sessions
        )
        & out["base_range"].le(config.maximum_base_range)
        & out["base_return"].between(
            config.minimum_base_return,
            config.maximum_base_return,
            inclusive="both",
        )
        & out["base_position"].between(
            config.minimum_base_position,
            config.maximum_base_position,
            inclusive="both",
        )
        & out["sessions_since_new_low"].ge(
            config.minimum_sessions_since_new_low
        )
        & selling_contracted
        & out["volatility_contraction_ratio"].le(
            config.maximum_volatility_contraction_ratio
        )
    ).fillna(False)
    signal_flag = _cooldown_signal_flags(
        eligible,
        groups,
        config.signal_cooldown_sessions,
    )
    out["signal_date"] = out["date"]
    out["signal_close"] = out["adjusted_close"]
    out["entry_date"] = out.groupby(groups, observed=True, sort=False)["date"].shift(-1)
    out["entry_open"] = out.groupby(groups, observed=True, sort=False)[
        "adjusted_open"
    ].shift(-1)
    duration_component = (
        out["deep_drawdown_sessions"] / 250.0
    ).clip(0.0, 1.0)
    range_component = (
        1.0 - out["base_range"] / config.maximum_base_range
    ).clip(0.0, 1.0)
    selling_component = (1.0 - out["recent_down_amount_share"]).clip(0.0, 1.0)
    volatility_component = (
        1.0 - out["volatility_contraction_ratio"]
    ).clip(0.0, 1.0)
    out["signal_score"] = (
        0.35 * duration_component
        + 0.25 * range_component
        + 0.20 * selling_component
        + 0.20 * volatility_component
    )
    out["base_event_id"] = signal_flag.astype(int).groupby(
        groups,
        observed=True,
        sort=False,
    ).cumsum()
    signals = out.loc[signal_flag, SIGNAL_COLUMNS].copy()
    signals = signals.dropna(subset=["entry_date", "entry_open"])
    return signals.reset_index(drop=True)


def _prepare_execution_panel(daily: pd.DataFrame) -> pd.DataFrame:
    panel = daily.copy()
    adjusted = {"adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"}
    if not adjusted.issubset(panel.columns):
        panel = core._add_causal_continuous_prices(core._prepare_daily(panel))
    else:
        if "date" not in panel:
            panel["date"] = pd.to_datetime(
                panel["trade_date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
        else:
            panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel.loc[panel["date"].notna()].copy()
    if "adjustment_factor" not in panel:
        panel["adjustment_factor"] = 1.0
    panel = panel.sort_values(["ts_code", "date"]).drop_duplicates(
        ["ts_code", "date"], keep="last"
    )
    if "return_20d" not in panel:
        panel["return_20d"] = panel.groupby(
            "ts_code", observed=True, sort=False
        )["adjusted_close"].pct_change(20, fill_method=None)
    return panel.sort_values(["date", "ts_code"]).reset_index(drop=True)


def _buy_fee(value: float, config: DeepBaseExecutionConfig) -> float:
    return max(value * config.commission_rate, config.minimum_commission) + (
        value * config.transfer_fee_rate
    )


def _sell_fee(value: float, config: DeepBaseExecutionConfig) -> float:
    return (
        max(value * config.commission_rate, config.minimum_commission)
        + value * config.transfer_fee_rate
        + value * config.stamp_tax_rate
    )


def _trigger_next_stage(
    position: _DeepBasePosition,
    row: pd.Series,
    config: DeepBaseExecutionConfig,
) -> int | None:
    stage = position.tranches_filled
    if stage >= 3 or position.sessions_held > config.maximum_scale_in_sessions:
        return None
    close = float(row["adjusted_close"])
    width = position.base_high - position.base_low
    base_position = (close - position.base_low) / width if width > 0 else np.nan
    if stage == 1:
        if config.stage_policy == "retest_reclaim":
            return_20d = pd.to_numeric(row.get("return_20d"), errors="coerce")
            confirmed = (
                position.sessions_held >= config.second_stage_minimum_sessions
                and position.lower_retest_seen
                and close >= position.base_low * 0.98
                and np.isfinite(base_position)
                and 0.50 <= base_position <= 0.75
                and pd.notna(return_20d)
                and float(return_20d) > 0.0
            )
            return 1 if confirmed else None
        confirmed = (
            position.sessions_held >= config.second_stage_minimum_sessions
            and close >= position.base_low * 0.98
            and np.isfinite(base_position)
            and base_position <= 0.45
        )
        return 1 if confirmed else None
    return_20d = pd.to_numeric(row.get("return_20d"), errors="coerce")
    minimum_third_position = 0.70 if config.stage_policy == "retest_reclaim" else 0.45
    maximum_third_position = 1.00 if config.stage_policy == "retest_reclaim" else 0.90
    confirmed = (
        position.sessions_held >= config.third_stage_minimum_sessions
        and close >= position.base_low * 0.98
        and np.isfinite(base_position)
        and minimum_third_position <= base_position <= maximum_third_position
        and pd.notna(return_20d)
        and float(return_20d) > 0.0
    )
    return 2 if confirmed else None


def run_deep_base_backtest(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    config: DeepBaseExecutionConfig,
    entry_start: str,
    entry_end: str,
) -> BloodChipBacktestResult:
    """Run a causal, in-range staged portfolio with structural base exits."""

    if signals.empty:
        return core._empty_backtest_result(config.initial_cash)
    panel = _prepare_execution_panel(daily)
    signal_frame = signals.copy()
    for column in ("signal_date", "entry_date"):
        signal_frame[column] = pd.to_datetime(signal_frame[column], errors="coerce")
    start = core._normalize_date(entry_start)
    end = core._normalize_date(entry_end)
    signal_frame = signal_frame.loc[
        signal_frame["entry_date"].between(start, end, inclusive="both")
    ].copy()
    if signal_frame.empty:
        return core._empty_backtest_result(config.initial_cash)
    signal_frame["signal_score"] = pd.to_numeric(
        signal_frame.get("signal_score", 0.0), errors="coerce"
    ).fillna(0.0)
    signals_by_date = {
        pd.Timestamp(date): group.sort_values(
            ["signal_score", "ts_code"], ascending=[False, True]
        )
        for date, group in signal_frame.groupby("entry_date", sort=True)
    }
    first_entry = pd.Timestamp(signal_frame["entry_date"].min())
    last_entry = pd.Timestamp(signal_frame["entry_date"].max())
    panel = panel.loc[panel["date"].ge(first_entry)].copy()

    cash = float(config.initial_cash)
    positions: dict[str, _DeepBasePosition] = {}
    last_event_used: dict[str, int] = {}
    trade_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    previous_equity = float(config.initial_cash)
    last_date: pd.Timestamp | None = None

    def reject(signal: pd.Series, reason: str) -> None:
        rejected_rows.append(
            {
                "ts_code": str(signal["ts_code"]),
                "signal_date": pd.Timestamp(signal["signal_date"]),
                "entry_date": pd.Timestamp(signal["entry_date"]),
                "base_event_id": int(signal["base_event_id"]),
                "reason": reason,
            }
        )

    def buy_tranche(
        position: _DeepBasePosition,
        row: pd.Series,
        date: pd.Timestamp,
        stage: int,
        *,
        reference_close: float,
    ) -> bool:
        nonlocal cash
        adjusted_open = float(row["adjusted_open"])
        adjusted_high = float(row["adjusted_high"])
        adjusted_low = float(row["adjusted_low"])
        gap = adjusted_open / reference_close - 1.0
        if not config.minimum_entry_gap <= gap <= config.maximum_entry_gap:
            return False
        if core._is_locked_limit(
            adjusted_open,
            adjusted_high,
            adjusted_low,
            reference_close,
            config.locked_limit_threshold,
            direction="up",
        ):
            return False
        if adjusted_open < position.base_low * 0.98:
            return False
        if stage == 1 and adjusted_open > position.base_mid * 1.03:
            return False
        if stage == 2 and adjusted_open > position.base_high * 1.03:
            return False
        if stage == 0 and adjusted_open > position.base_high * 1.03:
            return False
        factor = float(row.get("adjustment_factor", 1.0) or 1.0)
        raw_fill = float(row["open"]) * (1.0 + config.slippage)
        tranche_budget = position.target_budget * config.tranche_fractions[stage]
        affordable = min(cash, tranche_budget)
        lots = int(affordable // (raw_fill * config.lot_size))
        raw_shares = lots * config.lot_size
        if raw_shares <= 0:
            return False
        entry_value = raw_shares * raw_fill
        fee = _buy_fee(entry_value, config)
        if entry_value + fee > cash:
            lots = int(
                (cash - config.minimum_commission)
                // (raw_fill * config.lot_size)
            )
            raw_shares = max(lots, 0) * config.lot_size
            entry_value = raw_shares * raw_fill
            fee = _buy_fee(entry_value, config) if raw_shares else 0.0
        if raw_shares <= 0 or entry_value + fee > cash:
            return False
        cash -= entry_value + fee
        position.adjusted_units += raw_shares / factor
        position.raw_shares += raw_shares
        position.entry_value += entry_value
        position.invested_value += entry_value + fee
        position.buy_fees += fee
        position.tranches_filled += 1
        position.tranche_dates.append(date)
        position.ready_stage = None
        return True

    def close_position(
        position: _DeepBasePosition,
        date: pd.Timestamp,
        exit_adjusted: float,
        reason: str,
    ) -> None:
        nonlocal cash
        exit_fill = exit_adjusted * (1.0 - config.slippage)
        exit_value = position.adjusted_units * exit_fill
        sell_fee = (
            0.0 if reason == "missing_bar_writeoff" else _sell_fee(exit_value, config)
        )
        cash += exit_value - sell_fee
        net_return = (
            exit_value - sell_fee - position.invested_value
        ) / position.invested_value
        average_entry_fill = position.entry_value / position.adjusted_units
        trade_rows.append(
            {
                **position.signal_payload,
                "ts_code": position.symbol,
                "base_event_id": position.event_id,
                "signal_date": position.signal_date,
                "entry_date": position.entry_date,
                "exit_date": date,
                "entry_fill": average_entry_fill,
                "exit_fill": exit_fill,
                "entry_value": position.entry_value,
                "invested_value": position.invested_value,
                "exit_value": exit_value,
                "fees": position.buy_fees + sell_fee,
                "raw_shares": position.raw_shares,
                "holding_sessions": position.sessions_held,
                "exit_reason": reason,
                "gross_return": exit_value / position.entry_value - 1.0,
                "net_return": net_return,
                "reentry_number": 0,
                "tranches_filled": position.tranches_filled,
                "tranche_dates": "|".join(
                    value.date().isoformat() for value in position.tranche_dates
                ),
                "planned_fractions": "|".join(
                    f"{value:.4f}" for value in config.tranche_fractions
                ),
                "deployed_fraction": min(
                    position.entry_value / position.target_budget,
                    1.0,
                ),
                "hard_stop_price": position.hard_stop,
                "broken_close_count": position.broken_close_count,
            }
        )
        positions.pop(position.symbol, None)

    for date, day_frame in panel.groupby("date", observed=True, sort=True):
        date = pd.Timestamp(date)
        last_date = date
        bars = day_frame.set_index("ts_code", drop=False)

        for symbol, position in list(positions.items()):
            if symbol not in bars.index:
                position.sessions_held += 1
                position.missing_market_sessions += 1
                if (
                    position.missing_market_sessions
                    >= config.maximum_missing_market_sessions
                ):
                    close_position(position, date, 0.0, "missing_bar_writeoff")
                continue
            row = bars.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            position.sessions_held += 1
            position.missing_market_sessions = 0
            adjusted_open = float(row["adjusted_open"])
            adjusted_high = float(row["adjusted_high"])
            adjusted_low = float(row["adjusted_low"])
            locked_down = core._is_locked_limit(
                adjusted_open,
                adjusted_high,
                adjusted_low,
                position.last_close,
                config.locked_limit_threshold,
                direction="down",
            )
            if (
                config.structural_exit_enabled
                and position.structural_exit_ready
                and not locked_down
            ):
                close_position(position, date, adjusted_open, "structural_break")
                continue
            if adjusted_low <= position.hard_stop and not locked_down:
                exit_price = (
                    adjusted_open
                    if adjusted_open <= position.hard_stop
                    else position.hard_stop
                )
                close_position(position, date, exit_price, "hard_stop")
                continue
            if (
                position.sessions_held >= config.maximum_holding_sessions
                and not locked_down
            ):
                close_position(position, date, adjusted_open, "time_exit")
                continue
            if position.ready_stage is not None:
                ready_stage = position.ready_stage
                position.ready_stage = None
                buy_tranche(
                    position,
                    row,
                    date,
                    ready_stage,
                    reference_close=position.last_close,
                )
            position.last_close = float(row["adjusted_close"])

        candidates = signals_by_date.get(date)
        if candidates is not None:
            for _, signal in candidates.iterrows():
                symbol = str(signal["ts_code"])
                event_id = int(signal["base_event_id"])
                if symbol in positions:
                    reject(signal, "already_holding")
                    continue
                if len(positions) >= config.maximum_positions:
                    reject(signal, "portfolio_full")
                    continue
                if event_id <= last_event_used.get(symbol, 0):
                    reject(signal, "event_reused")
                    continue
                if symbol not in bars.index:
                    reject(signal, "missing_entry_bar")
                    continue
                row = bars.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                marked = 0.0
                for held_symbol, held in positions.items():
                    if held_symbol in bars.index:
                        held_row = bars.loc[held_symbol]
                        if isinstance(held_row, pd.DataFrame):
                            held_row = held_row.iloc[-1]
                        mark = float(held_row["adjusted_open"])
                    else:
                        mark = held.last_close
                    marked += held.adjusted_units * mark
                opening_equity = cash + marked
                target_budget = opening_equity * config.target_position_fraction
                payload = {
                    key: signal[key]
                    for key in SIGNAL_COLUMNS
                    if key in signal.index
                }
                position = _DeepBasePosition(
                    symbol=symbol,
                    event_id=event_id,
                    signal_date=pd.Timestamp(signal["signal_date"]),
                    entry_date=date,
                    signal_close=float(signal["signal_close"]),
                    base_low=float(signal["base_low"]),
                    base_high=float(signal["base_high"]),
                    base_mid=float(signal["base_mid"]),
                    hard_stop=float(signal["base_low"])
                    * (1.0 - config.hard_stop_below_base),
                    target_budget=target_budget,
                    adjusted_units=0.0,
                    raw_shares=0,
                    entry_value=0.0,
                    invested_value=0.0,
                    buy_fees=0.0,
                    sessions_held=0,
                    last_close=float(row["adjusted_close"]),
                    missing_market_sessions=0,
                    lower_retest_seen=False,
                    broken_close_count=0,
                    structural_exit_ready=False,
                    ready_stage=None,
                    tranches_filled=0,
                    tranche_dates=[],
                    signal_payload=payload,
                )
                if not buy_tranche(
                    position,
                    row,
                    date,
                    0,
                    reference_close=float(signal["signal_close"]),
                ):
                    reject(signal, "entry_rule")
                    continue
                positions[symbol] = position
                last_event_used[symbol] = event_id

        for symbol, position in positions.items():
            if symbol not in bars.index:
                continue
            row = bars.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            position.last_close = float(row["adjusted_close"])
            width = position.base_high - position.base_low
            current_base_position = (
                (position.last_close - position.base_low) / width
                if width > 0
                else np.nan
            )
            if (
                np.isfinite(current_base_position)
                and current_base_position <= 0.45
                and position.last_close >= position.base_low * 0.98
            ):
                position.lower_retest_seen = True
            structural_level = position.base_low * (
                1.0 - config.structural_break_below_base
            )
            if position.last_close < structural_level:
                position.broken_close_count += 1
            else:
                position.broken_close_count = 0
            position.structural_exit_ready = (
                position.broken_close_count >= config.structural_break_sessions
            )
            if position.ready_stage is None:
                position.ready_stage = _trigger_next_stage(position, row, config)

        marked_value = sum(
            position.adjusted_units * position.last_close
            for position in positions.values()
        )
        equity = cash + marked_value
        equity_rows.append(
            {
                "date": date,
                "cash": cash,
                "positions": len(positions),
                "equity": equity,
                "daily_return": equity / previous_equity - 1.0,
            }
        )
        previous_equity = equity
        if date > last_entry and not positions:
            break

    if positions and last_date is not None:
        for position in list(positions.values()):
            close_position(position, last_date, position.last_close, "end_of_data")
        if equity_rows:
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["positions"] = 0
            equity_rows[-1]["equity"] = cash
            previous = (
                config.initial_cash
                if len(equity_rows) == 1
                else float(equity_rows[-2]["equity"])
            )
            equity_rows[-1]["daily_return"] = cash / previous - 1.0

    return BloodChipBacktestResult(
        equity_curve=pd.DataFrame(equity_rows),
        trades=pd.DataFrame(trade_rows),
        rejected_entries=pd.DataFrame(rejected_rows),
    )


def summarize_deep_base_result(
    result: BloodChipBacktestResult,
    benchmark: pd.DataFrame,
) -> dict[str, float | int]:
    """Extend the shared portfolio summary with capital and stage metrics."""

    summary = core.summarize_blood_chip_result(result, benchmark)
    trades = result.trades
    if trades.empty:
        return {
            **summary,
            "capital_weighted_trade_return": np.nan,
            "capital_profit_factor": np.nan,
            "average_deployed_fraction": np.nan,
            "second_stage_rate": np.nan,
            "third_stage_rate": np.nan,
            "hard_stop_rate": np.nan,
            "structural_break_rate": np.nan,
        }
    pnl = (
        pd.to_numeric(trades["exit_value"], errors="coerce")
        - pd.to_numeric(trades["fees"], errors="coerce")
        - pd.to_numeric(trades["entry_value"], errors="coerce")
    )
    invested = pd.to_numeric(trades["invested_value"], errors="coerce")
    losses = float(-pnl.loc[pnl <= 0].sum())
    filled = pd.to_numeric(trades["tranches_filled"], errors="coerce")
    return {
        **summary,
        "capital_weighted_trade_return": float(pnl.sum() / invested.sum()),
        "capital_profit_factor": (
            float(pnl.loc[pnl > 0].sum() / losses) if losses > 0 else np.nan
        ),
        "average_deployed_fraction": float(
            pd.to_numeric(trades["deployed_fraction"], errors="coerce").mean()
        ),
        "second_stage_rate": float(filled.ge(2).mean()),
        "third_stage_rate": float(filled.ge(3).mean()),
        "hard_stop_rate": float(trades["exit_reason"].eq("hard_stop").mean()),
        "structural_break_rate": float(
            trades["exit_reason"].eq("structural_break").mean()
        ),
    }
