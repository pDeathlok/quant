"""Cost-aware, T+1-compliant intraday backtest utilities for BYD.

The simulator treats a "positive T" as buying an extra lot and later selling
settled inventory, and a "reverse T" as selling settled inventory and buying
it back later.  It supports repeated orders and cross-session lots while never
allowing same-day buys to increase the day's sellable inventory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Literal

import numpy as np
import pandas as pd


Side = Literal["BUY", "SELL"]
LotSide = Literal["positive", "reverse"]


@dataclass(frozen=True)
class ChinaAStockFees:
    """Conservative configurable fees for an A-share cash account."""

    commission_rate: float = 0.00025
    minimum_commission: float = 5.0
    stamp_tax_sell_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_rate: float = 0.0001

    def execution_price(self, raw_price: float, side: Side) -> float:
        multiplier = 1 + self.slippage_rate if side == "BUY" else 1 - self.slippage_rate
        return float(raw_price) * multiplier

    def order_fee(self, price: float, shares: int, side: Side) -> float:
        value = float(price) * int(shares)
        commission = max(value * self.commission_rate, self.minimum_commission)
        transfer = value * self.transfer_fee_rate
        stamp = value * self.stamp_tax_sell_rate if side == "SELL" else 0.0
        return commission + transfer + stamp


@dataclass(frozen=True)
class BydTBacktestConfig:
    base_shares: int = 10000
    minimum_shares: int = 8000
    maximum_shares: int = 12000
    lot_shares: int = 500
    max_positive_shares: int = 1500
    max_reverse_shares: int = 1500
    direction: Literal["both", "positive", "reverse"] = "both"
    entry_deviation: float = 0.008
    profit_target: float = 0.008
    stack_gap: float = 0.008
    stop_loss: float = 0.035
    max_holding_sessions: int = 3
    buy_rsi_max: float = 40.0
    sell_rsi_min: float = 60.0
    earliest_entry_minute: int = 9 * 60 + 45
    latest_entry_minute: int = 14 * 60 + 55
    tail_entry_minute: int = 14 * 60 + 35
    tail_extra_deviation: float = 0.002
    force_flat_minute: int | None = None
    require_sideways_regime: bool = False
    sideways_max_abs_return_60: float = 0.12
    sideways_max_abs_ma20_slope_5: float = 0.025
    sideways_max_abs_ma20_ma60_gap: float = 0.08
    sideways_min_range_width_60: float = 0.08
    sideways_max_range_width_60: float = 0.35
    sideways_min_atr_pct: float = 0.01
    sideways_max_atr_pct: float = 0.06
    positive_range_position_max: float = 0.40
    reverse_range_position_min: float = 0.60
    range_break_tolerance: float = 0.03
    positive_entry_max_prior_return_60: float = 10.0
    reverse_entry_min_prior_return_60: float = -10.0

    def __post_init__(self) -> None:
        if self.lot_shares <= 0 or self.lot_shares % 100:
            raise ValueError("lot_shares must be a positive multiple of 100")
        if self.minimum_shares > self.base_shares:
            raise ValueError("minimum_shares cannot exceed base_shares")
        if self.maximum_shares < self.base_shares:
            raise ValueError("maximum_shares cannot be below base_shares")
        if not 0 <= self.positive_range_position_max <= 1:
            raise ValueError("positive_range_position_max must be between 0 and 1")
        if not 0 <= self.reverse_range_position_min <= 1:
            raise ValueError("reverse_range_position_min must be between 0 and 1")
        if self.sideways_min_range_width_60 >= self.sideways_max_range_width_60:
            raise ValueError("sideways range-width bounds are invalid")
        if self.sideways_min_atr_pct >= self.sideways_max_atr_pct:
            raise ValueError("sideways ATR bounds are invalid")


@dataclass
class InventoryLedger:
    """Track total and sellable shares under the A-share T+1 rule."""

    position: int
    sellable: int = 0
    session: pd.Timestamp | None = None

    def start_session(self, session: Any) -> None:
        normalized = pd.Timestamp(session).normalize()
        if self.session is None or normalized != self.session:
            self.session = normalized
            self.sellable = self.position

    def buy(self, shares: int) -> None:
        if shares <= 0 or shares % 100:
            raise ValueError("buy shares must be a positive board lot")
        self.position += shares

    def sell(self, shares: int) -> None:
        if shares <= 0 or shares % 100:
            raise ValueError("sell shares must be a positive board lot")
        if shares > self.sellable:
            raise ValueError(
                f"T+1 sellability violation: tried {shares}, sellable {self.sellable}"
            )
        if shares > self.position:
            raise ValueError("cannot sell more shares than the current position")
        self.position -= shares
        self.sellable -= shares


@dataclass
class OpenTLot:
    lot_id: int
    side: LotSide
    shares: int
    open_time: pd.Timestamp
    open_session_index: int
    open_price: float
    open_fee: float
    open_reason: str


@dataclass
class BacktestResult:
    config: BydTBacktestConfig
    metrics: dict[str, Any]
    cycles: pd.DataFrame
    orders: pd.DataFrame
    equity: pd.DataFrame


def prepare_byd_intraday_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Create past-only signal features from normalized 5-minute bars."""
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"minute bars missing columns: {sorted(missing)}")

    out = raw.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "open", "high", "low", "close"])
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    out["date"] = out["datetime"].dt.normalize()
    out["minute"] = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute
    out["session_index"] = pd.factorize(out["date"], sort=True)[0]

    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0).clip(lower=0)
    typical = (out["high"] + out["low"] + out["close"]) / 3
    cumulative_volume = volume.groupby(out["date"]).cumsum()
    cumulative_value = (typical * volume).groupby(out["date"]).cumsum()
    out["vwap"] = cumulative_value / cumulative_volume.replace(0, np.nan)
    out["vwap"] = out["vwap"].fillna(out["close"])

    daily = out.groupby("date", sort=True).agg(
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    daily["previous_close"] = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - daily["previous_close"]).abs(),
            (daily["low"] - daily["previous_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["prior_atr_pct"] = (
        true_range.rolling(14, min_periods=5).mean().shift(1)
        / daily["close"].shift(1).replace(0, np.nan)
    )
    daily["ma20"] = daily["close"].rolling(20, min_periods=20).mean()
    daily["ma60"] = daily["close"].rolling(60, min_periods=60).mean()
    daily["prior_return_60"] = daily["close"].pct_change(60).shift(1)
    daily["prior_ma20_slope_5"] = daily["ma20"].pct_change(5).shift(1)
    daily["prior_ma20_ma60_gap"] = (daily["ma20"] / daily["ma60"] - 1).shift(1)
    daily["prior_high_60"] = daily["high"].rolling(60, min_periods=60).max().shift(1)
    daily["prior_low_60"] = daily["low"].rolling(60, min_periods=60).min().shift(1)
    daily["prior_range_width_60"] = daily["prior_high_60"] / daily[
        "prior_low_60"
    ].replace(0, np.nan) - 1
    past_daily_features = [
        "previous_close",
        "prior_atr_pct",
        "prior_return_60",
        "prior_ma20_slope_5",
        "prior_ma20_ma60_gap",
        "prior_high_60",
        "prior_low_60",
        "prior_range_width_60",
    ]
    for feature in past_daily_features:
        out[feature] = out["date"].map(daily[feature])
    prior_range = (out["prior_high_60"] - out["prior_low_60"]).replace(0, np.nan)
    out["prior_range_position"] = (out["close"] - out["prior_low_60"]) / prior_range
    out["deviation_vwap"] = out["close"] / out["vwap"].replace(0, np.nan) - 1
    out["deviation_previous_close"] = (
        out["close"] / out["previous_close"].replace(0, np.nan) - 1
    )

    previous_bar_close = out["close"].shift(1)
    previous_session = out["date"].shift(1)
    previous_bar_close = previous_bar_close.where(previous_session.eq(out["date"]), out["previous_close"])
    out["previous_bar_close"] = previous_bar_close
    price_delta = out["close"].diff()
    same_session = out["date"].eq(out["date"].shift(1))
    price_delta = price_delta.where(same_session, out["close"] - out["previous_close"])
    gain = price_delta.clip(lower=0).rolling(6, min_periods=6).mean()
    loss = (-price_delta.clip(upper=0)).rolling(6, min_periods=6).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi6"] = (100 - 100 / (1 + rs)).fillna(50.0)
    midpoint = (out["high"] + out["low"]) / 2
    out["turn_up"] = (out["close"] > out["previous_bar_close"]) & (out["close"] >= midpoint)
    out["turn_down"] = (out["close"] < out["previous_bar_close"]) & (out["close"] <= midpoint)
    return out.reset_index(drop=True)


def sideways_regime_mask(
    prepared_bars: pd.DataFrame,
    config: BydTBacktestConfig,
) -> pd.Series:
    """Return a past-only sideways flag for each bar under ``config``."""
    required = [
        "prior_return_60",
        "prior_ma20_slope_5",
        "prior_ma20_ma60_gap",
        "prior_range_width_60",
        "prior_atr_pct",
    ]
    missing = set(required).difference(prepared_bars.columns)
    if missing:
        raise ValueError(f"prepared bars missing regime features: {sorted(missing)}")
    finite = prepared_bars[required].notna().all(axis=1)
    return finite & (
        prepared_bars["prior_return_60"].abs().le(config.sideways_max_abs_return_60)
        & prepared_bars["prior_ma20_slope_5"].abs().le(
            config.sideways_max_abs_ma20_slope_5
        )
        & prepared_bars["prior_ma20_ma60_gap"].abs().le(
            config.sideways_max_abs_ma20_ma60_gap
        )
        & prepared_bars["prior_range_width_60"].between(
            config.sideways_min_range_width_60,
            config.sideways_max_range_width_60,
            inclusive="both",
        )
        & prepared_bars["prior_atr_pct"].between(
            config.sideways_min_atr_pct,
            config.sideways_max_atr_pct,
            inclusive="both",
        )
    )


def _row_is_sideways(row: Any, config: BydTBacktestConfig) -> bool:
    values = (
        row.prior_return_60,
        row.prior_ma20_slope_5,
        row.prior_ma20_ma60_gap,
        row.prior_range_width_60,
        row.prior_atr_pct,
    )
    if not all(np.isfinite(value) for value in values):
        return False
    return bool(
        abs(row.prior_return_60) <= config.sideways_max_abs_return_60
        and abs(row.prior_ma20_slope_5) <= config.sideways_max_abs_ma20_slope_5
        and abs(row.prior_ma20_ma60_gap) <= config.sideways_max_abs_ma20_ma60_gap
        and config.sideways_min_range_width_60
        <= row.prior_range_width_60
        <= config.sideways_max_range_width_60
        and config.sideways_min_atr_pct
        <= row.prior_atr_pct
        <= config.sideways_max_atr_pct
    )


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = wins / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def _entry_signal(row: Any, config: BydTBacktestConfig, side: LotSide) -> bool:
    if not config.earliest_entry_minute <= int(row.minute) <= config.latest_entry_minute:
        return False
    required_deviation = config.entry_deviation
    if int(row.minute) >= config.tail_entry_minute:
        required_deviation += config.tail_extra_deviation
    if config.require_sideways_regime:
        if not _row_is_sideways(row, config) or not np.isfinite(row.prior_range_position):
            return False
        lower_bound = -config.range_break_tolerance
        upper_bound = 1 + config.range_break_tolerance
        if side == "positive" and not (
            lower_bound <= row.prior_range_position <= config.positive_range_position_max
        ):
            return False
        if (
            side == "positive"
            and row.prior_return_60 > config.positive_entry_max_prior_return_60
        ):
            return False
        if side == "reverse" and not (
            config.reverse_range_position_min <= row.prior_range_position <= upper_bound
        ):
            return False
        if (
            side == "reverse"
            and row.prior_return_60 < config.reverse_entry_min_prior_return_60
        ):
            return False
    previous_close_requirement = required_deviation * 0.5
    if side == "positive":
        return bool(
            row.deviation_vwap <= -required_deviation
            and row.deviation_previous_close <= -previous_close_requirement
            and row.rsi6 <= config.buy_rsi_max
            and row.turn_up
        )
    return bool(
        row.deviation_vwap >= required_deviation
        and row.deviation_previous_close >= previous_close_requirement
        and row.rsi6 >= config.sell_rsi_min
        and row.turn_down
    )


def _marked_lot_pnl(lot: OpenTLot, close: float, fees: ChinaAStockFees) -> float:
    close_side: Side = "SELL" if lot.side == "positive" else "BUY"
    marked_price = fees.execution_price(close, close_side)
    close_fee = fees.order_fee(marked_price, lot.shares, close_side)
    gross = (
        (marked_price - lot.open_price) * lot.shares
        if lot.side == "positive"
        else (lot.open_price - marked_price) * lot.shares
    )
    return gross - lot.open_fee - close_fee


def backtest_byd_t(
    prepared_bars: pd.DataFrame,
    config: BydTBacktestConfig,
    fees: ChinaAStockFees = ChinaAStockFees(),
    *,
    track_equity: bool = True,
) -> BacktestResult:
    """Run a sequential next-bar-open simulation without future-price access."""
    if prepared_bars.empty:
        raise ValueError("prepared_bars cannot be empty")
    bars = prepared_bars.sort_values("datetime").reset_index(drop=True)
    sideways_sessions = 0
    if config.require_sideways_regime:
        session_regime = (
            pd.DataFrame(
                {
                    "date": bars["date"],
                    "sideways": sideways_regime_mask(bars, config),
                }
            )
            .groupby("date", sort=False)["sideways"]
            .first()
        )
        sideways_sessions = int(session_regime.sum())
    ledger = InventoryLedger(config.base_shares)
    positive_lots: list[OpenTLot] = []
    reverse_lots: list[OpenTLot] = []
    pending: dict[str, Any] | None = None
    next_lot_id = 1
    orders: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    realized_pnl = 0.0
    actual_fees = 0.0
    blocked_t1_sells = 0
    invariant_violations = 0
    minimum_position = ledger.position
    maximum_position = ledger.position

    def execute_order(row: Any, action: dict[str, Any]) -> None:
        nonlocal next_lot_id, realized_pnl, actual_fees, blocked_t1_sells, invariant_violations
        action_type = action["action"]
        shares = config.lot_shares
        side: Side = "BUY" if action_type in {"OPEN_POSITIVE", "CLOSE_REVERSE"} else "SELL"
        raw_price = float(row.open)
        execution_price = fees.execution_price(raw_price, side)
        fee = fees.order_fee(execution_price, shares, side)

        if side == "SELL" and ledger.sellable < shares:
            blocked_t1_sells += 1
            return
        if action_type == "OPEN_POSITIVE":
            if ledger.position + shares > config.maximum_shares:
                return
            ledger.buy(shares)
            positive_lots.append(
                OpenTLot(
                    lot_id=next_lot_id,
                    side="positive",
                    shares=shares,
                    open_time=pd.Timestamp(row.datetime),
                    open_session_index=int(row.session_index),
                    open_price=execution_price,
                    open_fee=fee,
                    open_reason=action["reason"],
                )
            )
            lot_id = next_lot_id
            next_lot_id += 1
        elif action_type == "OPEN_REVERSE":
            if ledger.position - shares < config.minimum_shares:
                return
            try:
                ledger.sell(shares)
            except ValueError:
                invariant_violations += 1
                return
            reverse_lots.append(
                OpenTLot(
                    lot_id=next_lot_id,
                    side="reverse",
                    shares=shares,
                    open_time=pd.Timestamp(row.datetime),
                    open_session_index=int(row.session_index),
                    open_price=execution_price,
                    open_fee=fee,
                    open_reason=action["reason"],
                )
            )
            lot_id = next_lot_id
            next_lot_id += 1
        else:
            lots = positive_lots if action_type == "CLOSE_POSITIVE" else reverse_lots
            lot = next((item for item in lots if item.lot_id == action["lot_id"]), None)
            if lot is None:
                return
            if action_type == "CLOSE_POSITIVE":
                try:
                    ledger.sell(shares)
                except ValueError:
                    invariant_violations += 1
                    return
                gross_pnl = (execution_price - lot.open_price) * shares
            else:
                if ledger.position + shares > config.maximum_shares:
                    return
                ledger.buy(shares)
                gross_pnl = (lot.open_price - execution_price) * shares
            lots.remove(lot)
            lot_id = lot.lot_id
            net_pnl = gross_pnl - lot.open_fee - fee
            realized_pnl += net_pnl
            cycles.append(
                {
                    "lot_id": lot.lot_id,
                    "side": lot.side,
                    "shares": shares,
                    "open_time": lot.open_time,
                    "close_time": pd.Timestamp(row.datetime),
                    "open_price": lot.open_price,
                    "close_price": execution_price,
                    "gross_pnl": gross_pnl,
                    "fees": lot.open_fee + fee,
                    "net_pnl": net_pnl,
                    "won": net_pnl > 0,
                    "holding_sessions": int(row.session_index) - lot.open_session_index,
                    "cross_session": pd.Timestamp(row.datetime).normalize()
                    != lot.open_time.normalize(),
                    "open_reason": lot.open_reason,
                    "close_reason": action["reason"],
                }
            )
        actual_fees += fee
        orders.append(
            {
                "time": pd.Timestamp(row.datetime),
                "action": action_type,
                "side": side,
                "shares": shares,
                "price": execution_price,
                "fee": fee,
                "lot_id": lot_id,
                "reason": action["reason"],
                "position_after": ledger.position,
                "sellable_after": ledger.sellable,
            }
        )

    for row in bars.itertuples(index=False):
        ledger.start_session(row.date)
        if pending is not None:
            execute_order(row, pending)
            pending = None
        minimum_position = min(minimum_position, ledger.position)
        maximum_position = max(maximum_position, ledger.position)

        if track_equity:
            marked_open_pnl = sum(
                _marked_lot_pnl(lot, float(row.close), fees)
                for lot in [*positive_lots, *reverse_lots]
            )
            equity_rows.append(
                {
                    "datetime": pd.Timestamp(row.datetime),
                    "realized_pnl": realized_pnl,
                    "open_lot_pnl": marked_open_pnl,
                    "economic_pnl": realized_pnl + marked_open_pnl,
                    "position": ledger.position,
                    "sellable": ledger.sellable,
                    "positive_open_shares": sum(lot.shares for lot in positive_lots),
                    "reverse_open_shares": sum(lot.shares for lot in reverse_lots),
                }
            )

        # Exits always take precedence over new entries.
        exit_action: dict[str, Any] | None = None
        for lot in positive_lots:
            holding_sessions = int(row.session_index) - lot.open_session_index
            if float(row.close) >= lot.open_price * (1 + config.profit_target):
                exit_action = {"action": "CLOSE_POSITIVE", "lot_id": lot.lot_id, "reason": "TARGET"}
                break
            if float(row.close) <= lot.open_price * (1 - config.stop_loss):
                exit_action = {"action": "CLOSE_POSITIVE", "lot_id": lot.lot_id, "reason": "STOP"}
                break
            if holding_sessions >= config.max_holding_sessions:
                exit_action = {"action": "CLOSE_POSITIVE", "lot_id": lot.lot_id, "reason": "MAX_HOLD"}
                break
        if exit_action is None:
            for lot in reverse_lots:
                holding_sessions = int(row.session_index) - lot.open_session_index
                if float(row.close) <= lot.open_price * (1 - config.profit_target):
                    exit_action = {"action": "CLOSE_REVERSE", "lot_id": lot.lot_id, "reason": "TARGET"}
                    break
                if float(row.close) >= lot.open_price * (1 + config.stop_loss):
                    exit_action = {"action": "CLOSE_REVERSE", "lot_id": lot.lot_id, "reason": "STOP"}
                    break
                if holding_sessions >= config.max_holding_sessions:
                    exit_action = {"action": "CLOSE_REVERSE", "lot_id": lot.lot_id, "reason": "MAX_HOLD"}
                    break
        if exit_action is None and config.force_flat_minute is not None:
            if int(row.minute) >= config.force_flat_minute:
                if positive_lots:
                    exit_action = {
                        "action": "CLOSE_POSITIVE",
                        "lot_id": positive_lots[0].lot_id,
                        "reason": "SESSION_CLOSE",
                    }
                elif reverse_lots:
                    exit_action = {
                        "action": "CLOSE_REVERSE",
                        "lot_id": reverse_lots[0].lot_id,
                        "reason": "SESSION_CLOSE",
                    }
        if exit_action is not None:
            pending = exit_action
            continue

        positive_signal = _entry_signal(row, config, "positive")
        reverse_signal = _entry_signal(row, config, "reverse")
        positive_quantity = sum(lot.shares for lot in positive_lots)
        reverse_quantity = sum(lot.shares for lot in reverse_lots)
        if positive_lots and not reverse_lots:
            last_price = positive_lots[-1].open_price
            if (
                config.direction in {"both", "positive"}
                and positive_signal
                and positive_quantity + config.lot_shares <= config.max_positive_shares
                and float(row.close) <= last_price * (1 - config.stack_gap)
            ):
                pending = {"action": "OPEN_POSITIVE", "reason": "STACKED_OVERSOLD"}
        elif reverse_lots and not positive_lots:
            last_price = reverse_lots[-1].open_price
            if (
                config.direction in {"both", "reverse"}
                and reverse_signal
                and reverse_quantity + config.lot_shares <= config.max_reverse_shares
                and float(row.close) >= last_price * (1 + config.stack_gap)
            ):
                pending = {"action": "OPEN_REVERSE", "reason": "STACKED_OVERBOUGHT"}
        elif not positive_lots and not reverse_lots:
            if config.direction in {"both", "positive"} and positive_signal:
                pending = {"action": "OPEN_POSITIVE", "reason": "OVERSOLD_REVERSAL"}
            elif config.direction in {"both", "reverse"} and reverse_signal:
                pending = {"action": "OPEN_REVERSE", "reason": "OVERBOUGHT_REVERSAL"}

    cycles_frame = pd.DataFrame(cycles)
    orders_frame = pd.DataFrame(orders)
    equity_frame = pd.DataFrame(equity_rows)
    final_open_lots = [*positive_lots, *reverse_lots]
    final_close = float(bars.iloc[-1]["close"])
    final_open_pnl = sum(_marked_lot_pnl(lot, final_close, fees) for lot in final_open_lots)
    economic_pnl = realized_pnl + final_open_pnl
    if cycles_frame.empty:
        winning_pnl = 0.0
        losing_pnl = 0.0
        wins = 0
    else:
        winning_pnl = float(cycles_frame.loc[cycles_frame["net_pnl"] > 0, "net_pnl"].sum())
        losing_pnl = float(-cycles_frame.loc[cycles_frame["net_pnl"] <= 0, "net_pnl"].sum())
        wins = int(cycles_frame["won"].sum())
    marked_losses = max(-final_open_pnl, 0.0)
    adjusted_losses = losing_pnl + marked_losses
    profit_factor = winning_pnl / adjusted_losses if adjusted_losses > 0 else float("inf")
    total_cycles = len(cycles_frame)
    if equity_frame.empty:
        max_drawdown = 0.0 if track_equity else float("nan")
    else:
        curve = equity_frame["economic_pnl"].astype(float)
        max_drawdown = float((curve.cummax() - curve).max())
    initial_value = config.base_shares * float(bars.iloc[0]["close"])
    metrics = {
        "start": str(pd.Timestamp(bars.iloc[0]["datetime"])),
        "end": str(pd.Timestamp(bars.iloc[-1]["datetime"])),
        "trading_sessions": int(bars["date"].nunique()),
        "sideways_sessions": sideways_sessions,
        "sideways_session_rate": sideways_sessions / int(bars["date"].nunique())
        if config.require_sideways_regime
        else 0.0,
        "cycles": total_cycles,
        "wins": wins,
        "losses": total_cycles - wins,
        "win_rate": wins / total_cycles if total_cycles else 0.0,
        "win_rate_wilson_lower": wilson_lower_bound(wins, total_cycles),
        "profit_factor": profit_factor,
        "gross_pnl": float(cycles_frame["gross_pnl"].sum()) if total_cycles else 0.0,
        "realized_net_pnl": realized_pnl,
        "final_open_lot_pnl": final_open_pnl,
        "net_pnl": economic_pnl,
        "fees_paid": actual_fees,
        "max_drawdown_yuan": max_drawdown,
        "max_drawdown_pct_of_base": max_drawdown / initial_value
        if initial_value
        else 0.0,
        "cost_reduction_per_base_share": economic_pnl / config.base_shares,
        "average_net_pnl": float(cycles_frame["net_pnl"].mean()) if total_cycles else 0.0,
        "median_net_pnl": float(cycles_frame["net_pnl"].median()) if total_cycles else 0.0,
        "cross_session_cycles": int(cycles_frame["cross_session"].sum()) if total_cycles else 0,
        "target_exits": int(cycles_frame["close_reason"].eq("TARGET").sum()) if total_cycles else 0,
        "stop_exits": int(cycles_frame["close_reason"].eq("STOP").sum()) if total_cycles else 0,
        "max_hold_exits": int(cycles_frame["close_reason"].eq("MAX_HOLD").sum()) if total_cycles else 0,
        "session_close_exits": int(cycles_frame["close_reason"].eq("SESSION_CLOSE").sum()) if total_cycles else 0,
        "open_lots_at_end": len(final_open_lots),
        "open_positive_shares_at_end": sum(lot.shares for lot in positive_lots),
        "open_reverse_shares_at_end": sum(lot.shares for lot in reverse_lots),
        "minimum_position": minimum_position,
        "maximum_position": maximum_position,
        "orders": len(orders_frame),
        "blocked_t1_sells": blocked_t1_sells,
        "t1_sellability_violations": invariant_violations,
    }
    return BacktestResult(config, metrics, cycles_frame, orders_frame, equity_frame)


def config_dict(config: BydTBacktestConfig) -> dict[str, Any]:
    return asdict(config)
