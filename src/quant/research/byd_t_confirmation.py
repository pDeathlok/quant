"""Causal BYD T research with confirmation, settled inventory and risk sizing."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from quant.research.byd_t_backtest import (
    BydTBacktestConfig,
    ChinaAStockFees,
    InventoryLedger,
    sideways_regime_mask,
    wilson_lower_bound,
)


@dataclass(frozen=True)
class ConfirmationTConfig:
    direction: Literal["positive", "reverse"] = "positive"
    entry_mode: Literal["limit", "confirmed"] = "confirmed"
    entry_atr: float = .4
    target_atr: float = .6
    stop_atr: float = .3
    latest_entry_minute: int = 11 * 60
    regime: Literal["balanced", "bounded"] = "balanced"
    earliest_entry_minute: int = 10 * 60
    exit_minute: int = 14 * 60 + 50
    risk_budget_yuan: float = 200.
    cash_budget_yuan: float = 50000.
    maximum_shares: int = 500
    settled_shares: int = 10000
    chase_atr: float = .15
    confirmation_delay_bars: int = 1

    def __post_init__(self) -> None:
        if (self.direction not in {"positive", "reverse"}
                or self.entry_mode not in {"limit", "confirmed"} or self.regime not in {"balanced", "bounded"}):
            raise ValueError("unknown entry mode or regime")
        for name in ("entry_atr", "target_atr", "stop_atr", "risk_budget_yuan", "cash_budget_yuan"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 575 <= self.earliest_entry_minute <= self.latest_entry_minute < self.exit_minute <= 900:
            raise ValueError("invalid entry or exit window")
        if self.maximum_shares <= 0 or self.maximum_shares % 100 or self.settled_shares < self.maximum_shares:
            raise ValueError("invalid settled board-lot capacity")
        if self.confirmation_delay_bars < 1 or self.chase_atr < 0:
            raise ValueError("confirmation requires a later bar and non-negative chase limit")


@dataclass(frozen=True)
class TSession:
    date: pd.Timestamp
    times: np.ndarray
    minute: np.ndarray
    prices: np.ndarray
    vwap: np.ndarray
    turn_up: np.ndarray
    turn_down: np.ndarray
    range_position: np.ndarray
    reference: float
    atr: float
    balanced: bool
    bounded: bool


@dataclass
class ConfirmationTResult:
    trades: pd.DataFrame
    equity: pd.DataFrame


def confirmation_sessions(prepared: pd.DataFrame) -> list[TSession]:
    """Materialize past-only prepared bars once for bounded parameter searches."""
    base = BydTBacktestConfig(require_sideways_regime=True)
    broad = replace(base, sideways_max_abs_return_60=.20,
                    sideways_max_abs_ma20_slope_5=.04,
                    sideways_max_abs_ma20_ma60_gap=.12,
                    sideways_max_range_width_60=.50)
    frame = prepared.sort_values("datetime").copy()
    frame["balanced"] = sideways_regime_mask(frame, base)
    frame["bounded"] = sideways_regime_mask(frame, broad)
    sessions = []
    for day, rows in frame.groupby("date", sort=True):
        first = rows.iloc[0]
        sessions.append(TSession(
            date=pd.Timestamp(day), times=rows["datetime"].to_numpy(dtype="datetime64[ns]"),
            minute=rows["minute"].to_numpy(dtype=int),
            prices=rows[["open", "high", "low", "close"]].to_numpy(dtype=float),
            vwap=rows["vwap"].to_numpy(dtype=float),
            turn_up=rows["turn_up"].to_numpy(dtype=bool),
            turn_down=(rows["close"].lt(rows["close"].shift(1))
                       & rows["close"].le((rows["high"] + rows["low"]) / 2)).to_numpy(dtype=bool),
            range_position=rows["prior_range_position"].to_numpy(dtype=float),
            reference=float(first["previous_close"]),
            atr=float(first["prior_atr_pct"] * first["previous_close"]),
            balanced=bool(first["balanced"]), bounded=bool(first["bounded"]),
        ))
    return sessions


def risk_sized_shares(
    entry: float, stop: float, config: ConfirmationTConfig,
    fees: ChinaAStockFees, available_cash: float,
) -> int:
    """Reserve entry cash and fee-adjusted planned stop loss, in board lots."""
    positive = config.direction == "positive"
    if not (0 < stop < entry if positive else 0 < entry < stop):
        return 0
    buy = fees.execution_price(entry if positive else stop, "BUY")
    sell = fees.execution_price(stop if positive else entry, "SELL")
    for shares in range(config.maximum_shares, 0, -100):
        buy_fee = fees.order_fee(buy, shares, "BUY")
        loss = (buy - sell) * shares + buy_fee + fees.order_fee(sell, shares, "SELL")
        cash = buy * shares + buy_fee if positive else loss
        if cash <= min(available_cash, config.cash_budget_yuan) and loss <= config.risk_budget_yuan:
            return shares
    return 0


def simulate_confirmation_t(
    sessions: list[TSession], config: ConfirmationTConfig,
    fees: ChinaAStockFees = ChinaAStockFees(),
) -> ConfirmationTResult:
    """Replay one intraday T per day; account for exit fees in marked PnL.

    Timestamp labels are five-minute bar ends; confirmed entries use a later
    bar's OPEN, never the signal bar's low/close. An interruption longer than
    five minutes cancels pending confirmation, including lunch and overnight.
    Current fee assumptions on qfq bars are research assumptions, not a full
    historical corporate-action or broker execution reconstruction.
    """
    trades: list[dict] = []
    curves: list[pd.DataFrame] = []
    realized = 0.
    ledger = InventoryLedger(config.settled_shares)
    for day in sessions:
        ledger.start_session(day.date)
        curve = np.full(len(day.times), realized, dtype=float)
        if not getattr(day, config.regime) or not np.isfinite(day.atr) or day.atr <= 0:
            curves.append(pd.DataFrame({"datetime": day.times, "pnl": curve}))
            continue
        exits = np.flatnonzero(day.minute >= config.exit_minute)
        if not len(exits):
            # An incomplete day cannot prove the mandated liquidation.
            raise ValueError(f"missing scheduled exit for {day.date.date()}")
        terminal = int(exits[0])
        positive = config.direction == "positive"
        sign = 1 if positive else -1
        trigger = round(day.reference - sign * config.entry_atr * day.atr, 2)
        lower, upper = (-.03, .60) if positive else (.40, 1.03)
        allowed = ((day.minute >= config.earliest_entry_minute)
                   & (day.minute <= config.latest_entry_minute)
                   & (day.range_position >= lower) & (day.range_position <= upper))
        if config.entry_mode == "limit":
            touches = day.prices[:, 2] <= trigger if positive else day.prices[:, 1] >= trigger
            signals = np.flatnonzero(allowed & touches)
        else:
            if positive:
                touches = (day.prices[:, 2] <= trigger) & (day.prices[:, 2] <= day.vwap - config.entry_atr * day.atr)
                confirmed = day.turn_up & (day.prices[:, 3] <= np.minimum(day.reference, day.vwap))
            else:
                touches = (day.prices[:, 1] >= trigger) & (day.prices[:, 1] >= day.vwap + config.entry_atr * day.atr)
                confirmed = day.turn_down & (day.prices[:, 3] >= np.maximum(day.reference, day.vwap))
            indexes = np.arange(len(day.times))
            last_touch = np.maximum.accumulate(np.where(touches, indexes, -100))
            armed = last_touch >= 0
            age = (day.times - day.times[np.maximum(last_touch, 0)]) / np.timedelta64(1, "m")
            # The touch arms a short-lived opportunity; the rebound need not
            # remain at the extreme discount throughout confirmation.
            signals = np.flatnonzero(allowed & confirmed & armed & (age <= 15))
        opened = False
        for signal in signals:
            entry_index = int(signal) + (config.confirmation_delay_bars if config.entry_mode == "confirmed" else 0)
            if entry_index >= terminal or day.minute[entry_index] > config.latest_entry_minute:
                continue
            if config.entry_mode == "confirmed":
                elapsed = (day.times[entry_index] - day.times[signal]) / np.timedelta64(1, "m")
                if elapsed != config.confirmation_delay_bars * 5:
                    continue
                entry = float(day.prices[entry_index, 0])
                # A limit fixed at confirmation time prevents chasing a gap.
                if sign * (entry - day.prices[signal, 3]) > config.chase_atr * day.atr:
                    continue
            else:
                entry_open = float(day.prices[entry_index, 0])
                entry = min(entry_open, trigger) if positive else max(entry_open, trigger)
            entry_at_open = config.entry_mode == "confirmed" or (
                day.prices[entry_index, 0] <= trigger if positive else day.prices[entry_index, 0] >= trigger
            )
            target = round(entry + sign * config.target_atr * day.atr, 2)
            stop = round(entry - sign * config.stop_atr * day.atr, 2)
            available = config.cash_budget_yuan + realized
            shares = risk_sized_shares(entry, stop, config, fees, available)
            if not shares:
                continue
            if positive:
                ledger.buy(shares)
            else:
                ledger.sell(shares)
            entry_side = "BUY" if positive else "SELL"
            exit_side = "SELL" if positive else "BUY"
            entry_execution = fees.execution_price(entry, entry_side)
            entry_fee = fees.order_fee(entry_execution, shares, entry_side)
            cost = entry_execution * shares + sign * entry_fee
            stop_execution = fees.execution_price(stop, exit_side)
            planned_loss = sign * (cost - stop_execution * shares) + fees.order_fee(stop_execution, shares, exit_side)
            reason, exit_index, exit_raw = "SESSION_EXIT", terminal, float(day.prices[terminal, 3])
            for index in range(entry_index, terminal + 1):
                opening, high, low, close = day.prices[index]
                stopped = low <= stop if positive else high >= stop
                target_touched = high >= target if positive else low <= target
                if stopped:
                    reason, exit_index = "STOP", index
                    exit_raw = min(float(opening), stop) if positive else max(float(opening), stop)
                elif (index > entry_index or entry_at_open) and target_touched:
                    reason, exit_index, exit_raw = "TARGET", index, target
                elif index == terminal:
                    exit_index, exit_raw = terminal, float(close)
                else:
                    liquidation = fees.execution_price(float(close), exit_side)
                    curve[index] = realized + sign * (liquidation * shares - cost) - fees.order_fee(liquidation, shares, exit_side)
                    continue
                break
            exit_execution = fees.execution_price(exit_raw, exit_side)
            exit_fee = fees.order_fee(exit_execution, shares, exit_side)
            pnl = sign * (exit_execution * shares - cost) - exit_fee
            if positive:
                ledger.sell(shares)
            else:
                if available + pnl < 0:
                    raise ValueError("insufficient cash to restore reverse-T inventory")
                ledger.buy(shares)
            realized += pnl
            curve[exit_index:] = realized
            trades.append({
                "date": day.date, "signal_time": pd.Timestamp(day.times[signal]),
                "entry_bar_end": pd.Timestamp(day.times[entry_index]),
                "exit_bar_end": pd.Timestamp(day.times[exit_index]),
                "direction": config.direction, "entry_price": entry, "target_price": target, "stop_price": stop,
                "exit_price": exit_raw, "shares": shares, "pnl": pnl,
                "fees_paid": entry_fee + exit_fee, "cash_required": cost if positive else planned_loss,
                "planned_loss": planned_loss, "budget_exceeded": -pnl > config.risk_budget_yuan + .01,
                "exit_reason": reason, "position_after": ledger.position,
                "return_bps": pnl / (entry * shares) * 10000,
            })
            opened = True
            break
        if opened and ledger.position != config.settled_shares:
            raise RuntimeError("completed T must restore starting inventory")
        curves.append(pd.DataFrame({"datetime": day.times, "pnl": curve}))
    columns = ["date", "pnl", "shares", "exit_reason", "budget_exceeded", "fees_paid", "return_bps"]
    trades_frame = pd.DataFrame(trades) if trades else pd.DataFrame(columns=columns)
    equity = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["datetime", "pnl"])
    return ConfirmationTResult(trades_frame, equity)


def confirmation_metrics(result: ConfirmationTResult, first: str, last: str) -> dict:
    """Summarize incremental T PnL and five-minute close mark-to-market risk."""
    start, end = pd.Timestamp(first), pd.Timestamp(last)
    trades = result.trades.loc[pd.to_datetime(result.trades["date"]).between(start, end)]
    timestamps = pd.to_datetime(result.equity["datetime"])
    marked = result.equity.loc[timestamps.between(start, end + pd.Timedelta(days=1), inclusive="left"), "pnl"]
    previous = result.equity.loc[timestamps < start, "pnl"]
    anchor = float(previous.iloc[-1]) if len(previous) else 0.
    values = np.r_[anchor, marked.to_numpy(dtype=float)]
    gains = float(trades.loc[trades["pnl"] > 0, "pnl"].sum())
    losses = float(-trades.loc[trades["pnl"] < 0, "pnl"].sum())
    wins = int(trades["pnl"].gt(0).sum())
    return {
        "cycles": len(trades), "net_pnl": round(float(trades["pnl"].sum()), 2),
        "win_rate": wins / len(trades) if len(trades) else 0.,
        "win_rate_wilson_lower": wilson_lower_bound(wins, len(trades)),
        "profit_factor": gains / losses if losses else (None if gains else 0.),
        "mtm_drawdown_yuan": round(float((np.maximum.accumulate(values) - values).max()), 2),
        "worst_cycle": round(float(trades["pnl"].min()), 2) if len(trades) else 0.,
        "fees_paid": round(float(trades["fees_paid"].sum()), 2),
        "risk_budget_exceedances": int(trades["budget_exceeded"].sum()),
    }


def selection_qualified(development: dict, validation: dict, annual: list[dict]) -> bool:
    """Keep the prior minimum counts and profitability requirements."""
    def passed(metrics: dict, count: int) -> bool:
        factor = metrics["profit_factor"]
        return bool(metrics["cycles"] >= count and metrics["win_rate"] >= .60
                    and metrics["net_pnl"] > 0 and (factor is None or factor >= 1.2))
    return passed(development, 40) and passed(validation, 25) and all(item["net_pnl"] > 0 for item in annual)
