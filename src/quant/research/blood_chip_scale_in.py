"""Causal portfolio backtest for staged blood-chip position building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quant.research import blood_chip as core
from quant.research.blood_chip import (
    BloodChipBacktestConfig,
    BloodChipBacktestResult,
    CASE_FEATURES,
)


TriggerMode = Literal["none", "confirmed", "price_only", "survival"]


@dataclass(frozen=True)
class BloodChipScaleInPolicy:
    key: str
    fractions: tuple[float, ...]
    trigger_mode: TriggerMode
    raise_stop_after_add: bool = False

    def __post_init__(self) -> None:
        if not self.fractions or any(value <= 0 for value in self.fractions):
            raise ValueError("fractions must be positive")
        if not np.isclose(sum(self.fractions), 1.0):
            raise ValueError("fractions must sum to one")
        expected = 1 if self.trigger_mode == "none" else 3
        if len(self.fractions) != expected:
            raise ValueError(f"{self.trigger_mode} policy requires {expected} tranches")


DEFAULT_SCALE_IN_POLICIES = {
    "one_shot": BloodChipScaleInPolicy("one_shot", (1.0,), "none"),
    "equal_confirmed": BloodChipScaleInPolicy(
        "equal_confirmed",
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        "confirmed",
    ),
    "increasing_confirmed": BloodChipScaleInPolicy(
        "increasing_confirmed",
        (0.20, 0.30, 0.50),
        "confirmed",
    ),
    "increasing_price_only": BloodChipScaleInPolicy(
        "increasing_price_only",
        (0.20, 0.30, 0.50),
        "price_only",
    ),
    "increasing_survival": BloodChipScaleInPolicy(
        "increasing_survival",
        (0.20, 0.30, 0.50),
        "survival",
    ),
    "increasing_survival_risk_capped": BloodChipScaleInPolicy(
        "increasing_survival_risk_capped",
        (0.20, 0.30, 0.50),
        "survival",
        raise_stop_after_add=True,
    ),
}


@dataclass
class _ScalePosition:
    symbol: str
    event_id: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    signal_close: float
    event_low: float
    stop_price: float
    target_budget: float
    adjusted_units: float
    raw_shares: int
    entry_value: float
    invested_value: float
    buy_fees: float
    sessions_held: int
    last_close: float
    last_residual_return_3d: float | None
    reentry_number: int
    tranches_filled: int
    tranche_dates: list[pd.Timestamp]
    tranche_fractions: list[float]
    ready_stage: int | None
    signal_payload: dict[str, object]


def _empty_result(initial_cash: float) -> BloodChipBacktestResult:
    return core._empty_backtest_result(initial_cash)


def _prepare_panel(daily: pd.DataFrame) -> pd.DataFrame:
    panel = daily.copy()
    if "date" not in panel:
        panel["date"] = pd.to_datetime(
            panel["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        panel["date"] = pd.to_datetime(panel["date"])
    adjusted = {"adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"}
    if not adjusted <= set(panel.columns):
        panel = core._add_causal_continuous_prices(core._prepare_daily(panel))
    if "adjustment_factor" not in panel:
        panel["adjustment_factor"] = 1.0
    if "residual_return_3d" not in panel:
        panel["residual_return_3d"] = np.nan
    return panel.sort_values(["date", "ts_code"]).drop_duplicates(
        ["date", "ts_code"], keep="last"
    )


def _trigger_next_stage(
    position: _ScalePosition,
    row: pd.Series,
    policy: BloodChipScaleInPolicy,
) -> int | None:
    stage = position.tranches_filled
    if stage >= len(policy.fractions) or policy.trigger_mode == "none":
        return None
    close = float(row["adjusted_close"])
    residual_3d = pd.to_numeric(row.get("residual_return_3d"), errors="coerce")
    if policy.trigger_mode == "price_only":
        threshold = 0.96 if stage == 1 else 0.92
        return stage if close <= position.signal_close * threshold else None
    if policy.trigger_mode == "survival":
        if stage == 1:
            survived = (
                position.sessions_held >= 5
                and close >= position.signal_close * 0.95
                and pd.notna(residual_3d)
                and float(residual_3d) >= 0.0
            )
            return stage if survived else None
        strengthened = (
            position.sessions_held >= 10
            and close >= position.signal_close
            and pd.notna(residual_3d)
            and float(residual_3d) > 0.0
        )
        return stage if strengthened else None
    if stage == 1:
        lower = position.event_low * 1.03
        upper = position.signal_close * 0.97
        stable = pd.notna(residual_3d) and float(residual_3d) >= -0.01
        return stage if lower <= close <= upper and stable else None
    repaired = (
        close >= position.signal_close
        and pd.notna(residual_3d)
        and float(residual_3d) > 0.0
    )
    return stage if repaired else None


def run_blood_chip_scale_in_backtest(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    config: BloodChipBacktestConfig,
    policy: BloodChipScaleInPolicy,
    entry_start: str,
    entry_end: str,
    *,
    maximum_open_plans: int | None = None,
    target_position_fraction: float | None = None,
) -> BloodChipBacktestResult:
    """Run a staged next-open portfolio with causal close-to-open add triggers."""

    open_plan_limit = maximum_open_plans or config.maximum_positions
    position_fraction = target_position_fraction or (1.0 / config.maximum_positions)
    if open_plan_limit < 1:
        raise ValueError("maximum_open_plans must be positive")
    if not 0.0 < position_fraction <= 1.0:
        raise ValueError("target_position_fraction must be in (0, 1]")
    if signals.empty:
        return _empty_result(config.initial_cash)
    panel = _prepare_panel(daily)
    signal_frame = signals.copy()
    for column in ("signal_date", "entry_date", "shock_date"):
        signal_frame[column] = pd.to_datetime(signal_frame[column])
    start = core._normalize_date(entry_start)
    end = core._normalize_date(entry_end)
    signal_frame = signal_frame.loc[
        signal_frame["entry_date"].between(start, end, inclusive="both")
    ].copy()
    if signal_frame.empty:
        return _empty_result(config.initial_cash)
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
    positions: dict[str, _ScalePosition] = {}
    last_event_used: dict[str, int] = {}
    trade_count_by_symbol: dict[str, int] = {}
    trade_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    previous_equity = float(config.initial_cash)
    last_date: pd.Timestamp | None = None

    def reject(signal: pd.Series, reason: str) -> None:
        rejected_rows.append(
            {
                "ts_code": str(signal["ts_code"]),
                "signal_date": pd.Timestamp(signal["signal_date"]),
                "entry_date": pd.Timestamp(signal["entry_date"]),
                "shock_event_id": int(signal["shock_event_id"]),
                "reason": reason,
            }
        )

    def buy_tranche(
        position: _ScalePosition,
        row: pd.Series,
        date: pd.Timestamp,
        stage: int,
        reference_close: float | None = None,
    ) -> bool:
        nonlocal cash
        previous_close = (
            position.last_close if reference_close is None else reference_close
        )
        adjusted_open = float(row["adjusted_open"])
        adjusted_high = float(row["adjusted_high"])
        adjusted_low = float(row["adjusted_low"])
        gap = adjusted_open / previous_close - 1.0
        if not config.minimum_entry_gap <= gap <= config.maximum_entry_gap:
            return False
        if core._is_locked_limit(
            adjusted_open,
            adjusted_high,
            adjusted_low,
            previous_close,
            config.locked_limit_threshold,
            direction="up",
        ):
            return False
        factor = float(row.get("adjustment_factor", 1.0) or 1.0)
        raw_fill = float(row["open"]) * (1.0 + config.slippage)
        tranche_budget = position.target_budget * policy.fractions[stage]
        affordable = min(cash, tranche_budget)
        lots = int(affordable // (raw_fill * config.lot_size))
        raw_shares = lots * config.lot_size
        if raw_shares <= 0:
            return False
        entry_value = raw_shares * raw_fill
        fee = core._buy_fee(entry_value, config)
        if entry_value + fee > cash:
            lots = int(
                (cash - config.minimum_commission)
                // (raw_fill * config.lot_size)
            )
            raw_shares = max(lots, 0) * config.lot_size
            entry_value = raw_shares * raw_fill
            fee = core._buy_fee(entry_value, config) if raw_shares else 0.0
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
        position.tranche_fractions.append(policy.fractions[stage])
        position.ready_stage = None
        if stage > 0 and policy.raise_stop_after_add:
            weighted_adjusted_cost = position.entry_value / position.adjusted_units
            position.stop_price = max(
                position.stop_price,
                weighted_adjusted_cost * (1.0 - config.stop_loss),
            )
        return True

    def close_position(
        position: _ScalePosition,
        date: pd.Timestamp,
        exit_adjusted: float,
        reason: str,
    ) -> None:
        nonlocal cash
        exit_fill = exit_adjusted * (1.0 - config.slippage)
        exit_value = position.adjusted_units * exit_fill
        sell_fee = core._sell_fee(exit_value, config)
        cash += exit_value - sell_fee
        net_return = (
            exit_value - sell_fee - position.invested_value
        ) / position.invested_value
        gross_return = exit_value / position.entry_value - 1.0
        average_entry_fill = position.entry_value / position.adjusted_units
        trade_rows.append(
            {
                **position.signal_payload,
                "ts_code": position.symbol,
                "shock_event_id": position.event_id,
                "signal_date": position.signal_date,
                "entry_date": position.entry_date,
                "exit_date": date,
                "entry_fill": average_entry_fill,
                "exit_fill": exit_fill,
                "signal_close": position.signal_close,
                "stop_price": position.stop_price,
                "current_residual_return_3d": position.last_residual_return_3d,
                "next_stage_ready": position.ready_stage is not None,
                "entry_value": position.entry_value,
                "invested_value": position.invested_value,
                "exit_value": exit_value,
                "fees": position.buy_fees + sell_fee,
                "raw_shares": position.raw_shares,
                "holding_sessions": position.sessions_held,
                "exit_reason": reason,
                "gross_return": gross_return,
                "net_return": net_return,
                "reentry_number": position.reentry_number,
                "tranches_filled": position.tranches_filled,
                "tranche_dates": "|".join(
                    value.date().isoformat() for value in position.tranche_dates
                ),
                "planned_fractions": "|".join(
                    f"{value:.4f}" for value in policy.fractions
                ),
                "deployed_fraction": min(
                    position.entry_value / position.target_budget, 1.0
                ),
                "scale_in_policy": policy.key,
            }
        )
        positions.pop(position.symbol, None)

    for date, day_frame in panel.groupby("date", observed=True, sort=True):
        date = pd.Timestamp(date)
        last_date = date
        bars = day_frame.set_index("ts_code", drop=False)

        for symbol, position in list(positions.items()):
            if symbol not in bars.index:
                continue
            row = bars.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            position.sessions_held += 1
            residual_3d = pd.to_numeric(
                row.get("residual_return_3d"), errors="coerce"
            )
            position.last_residual_return_3d = (
                float(residual_3d) if pd.notna(residual_3d) else None
            )
            adjusted_open = float(row["adjusted_open"])
            adjusted_high = float(row["adjusted_high"])
            adjusted_low = float(row["adjusted_low"])
            previous_close = position.last_close
            locked_down = core._is_locked_limit(
                adjusted_open,
                adjusted_high,
                adjusted_low,
                previous_close,
                config.locked_limit_threshold,
                direction="down",
            )
            stop_hit = adjusted_low <= position.stop_price
            time_hit = position.sessions_held >= config.maximum_holding_days
            if not locked_down and stop_hit:
                exit_price = (
                    adjusted_open
                    if adjusted_open <= position.stop_price
                    else position.stop_price
                )
                close_position(position, date, exit_price, "stop_loss")
                continue
            if not locked_down and time_hit:
                close_position(position, date, adjusted_open, "time_exit")
                continue
            if position.ready_stage is not None:
                ready_stage = position.ready_stage
                position.ready_stage = None
                buy_tranche(position, row, date, ready_stage)
            position.last_close = float(row["adjusted_close"])

        candidates = signals_by_date.get(date)
        if candidates is not None:
            for _, signal in candidates.iterrows():
                symbol = str(signal["ts_code"])
                event_id = int(signal["shock_event_id"])
                if symbol in positions:
                    reject(signal, "already_holding")
                    continue
                if len(positions) >= open_plan_limit:
                    reject(signal, "portfolio_full")
                    continue
                if symbol not in bars.index:
                    reject(signal, "missing_entry_bar")
                    continue
                if config.require_new_event_for_reentry and event_id <= last_event_used.get(
                    symbol, 0
                ):
                    reject(signal, "event_reused")
                    continue
                row = bars.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                signal_close = float(signal["signal_close"])
                adjusted_open = float(row["adjusted_open"])
                gap = adjusted_open / signal_close - 1.0
                if not config.minimum_entry_gap <= gap <= config.maximum_entry_gap:
                    reject(signal, "entry_gap")
                    continue
                if core._is_locked_limit(
                    adjusted_open,
                    float(row["adjusted_high"]),
                    float(row["adjusted_low"]),
                    signal_close,
                    config.locked_limit_threshold,
                    direction="up",
                ):
                    reject(signal, "locked_limit_up")
                    continue
                marked = sum(
                    held.adjusted_units
                    * (
                        float(bars.loc[held_symbol]["adjusted_open"])
                        if held_symbol in bars.index
                        and isinstance(bars.loc[held_symbol], pd.Series)
                        else held.last_close
                    )
                    for held_symbol, held in positions.items()
                )
                opening_equity = cash + marked
                target_budget = opening_equity * position_fraction
                rebound = float(signal.get("rebound_from_event_low", np.nan))
                event_low = (
                    signal_close / (1.0 + rebound)
                    if np.isfinite(rebound) and rebound > -1.0
                    else signal_close * 0.90
                )
                payload = {
                    key: signal[key]
                    for key in CASE_FEATURES
                    if key in signal.index
                }
                payload["signal_score"] = float(signal.get("signal_score", np.nan))
                position = _ScalePosition(
                    symbol=symbol,
                    event_id=event_id,
                    signal_date=pd.Timestamp(signal["signal_date"]),
                    entry_date=date,
                    signal_close=signal_close,
                    event_low=event_low,
                    stop_price=adjusted_open
                    * (1.0 + config.slippage)
                    * (1.0 - config.stop_loss),
                    target_budget=target_budget,
                    adjusted_units=0.0,
                    raw_shares=0,
                    entry_value=0.0,
                    invested_value=0.0,
                    buy_fees=0.0,
                    sessions_held=0,
                    last_close=float(row["adjusted_close"]),
                    last_residual_return_3d=(
                        float(row["residual_return_3d"])
                        if pd.notna(row.get("residual_return_3d"))
                        else None
                    ),
                    reentry_number=trade_count_by_symbol.get(symbol, 0),
                    tranches_filled=0,
                    tranche_dates=[],
                    tranche_fractions=[],
                    ready_stage=None,
                    signal_payload=payload,
                )
                if not buy_tranche(
                    position,
                    row,
                    date,
                    0,
                    reference_close=signal_close,
                ):
                    reject(signal, "insufficient_cash")
                    continue
                positions[symbol] = position
                last_event_used[symbol] = event_id
                trade_count_by_symbol[symbol] = position.reentry_number + 1

        marked_value = 0.0
        for symbol, position in positions.items():
            if symbol in bars.index:
                row = bars.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                position.last_close = float(row["adjusted_close"])
                if position.ready_stage is None:
                    position.ready_stage = _trigger_next_stage(position, row, policy)
            marked_value += position.adjusted_units * position.last_close
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
