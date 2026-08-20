"""Causal primitives for low-frequency A-share swing-timing research.

The module deliberately separates a point-in-time signal snapshot from future
execution.  Signals are observed after the signal-date close, entries happen at
the next session's open, and A-share T+1 means the first possible exit is the
session after entry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant.backtest import AShareExecutionConfig
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.features.variable_library import build_continuous_ohlc


@dataclass(frozen=True)
class SwingEntryRule:
    """Frozen point-in-time gates for one entry-rule variant."""

    name: str
    min_good_stock_score: float = 60.0
    min_historical_value_score: float | None = None
    max_volatility_percentile: float = 1.0
    min_return_120d: float = 0.03
    max_return_120d: float = 0.35
    min_close_to_ma20: float = 0.0
    max_close_to_ma20: float = 0.04
    min_close_to_ma60: float = 0.0
    min_close_to_ma120: float = 0.0
    min_ma120_slope: float = 0.0
    min_market_return_13w: float = 0.0
    min_market_return_26w: float = 0.0
    min_market_drawdown_52w: float = -0.18
    max_market_volatility_13w: float = 0.30
    min_entry_gap: float = -0.03
    max_entry_gap: float = 0.015
    cooldown_calendar_days: int = 56

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SwingExitRule:
    """Fixed barrier and time-exit policy."""

    name: str
    take_profit: float
    stop_loss: float
    hold_days: int
    exit_grace_days: int = 5

    def __post_init__(self) -> None:
        if self.take_profit <= 0 or self.stop_loss <= 0:
            raise ValueError("take_profit and stop_loss must be positive")
        if self.hold_days < 1:
            raise ValueError("hold_days must allow at least one T+1 sell session")
        if self.exit_grace_days < 0:
            raise ValueError("exit_grace_days must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REQUIRED_SIGNAL_COLUMNS = {
    "date",
    "ts_code",
    "is_good_stock",
    "good_stock_score",
    "historical_value_score_5y",
    "close_to_ma20",
    "close_to_ma60",
    "close_to_ma120",
    "ma_120_slope_20d",
    "return_120d",
    "volatility_60d_cross_section_pct",
    "market_return_13w",
    "market_return_26w",
    "market_drawdown_52w",
    "market_volatility_13w",
}


def attach_future_price_paths(
    candidates: pd.DataFrame,
    daily_dir: str | Path,
    maximum_path_day: int,
) -> pd.DataFrame:
    """Attach only candidate-date future paths instead of a full wide market.

    This is equivalent to shifting each symbol's causal continuous OHLC series,
    but materializes the wide T+N columns only for actual candidate rows.
    """

    if candidates.empty:
        return candidates.copy()
    if maximum_path_day < 1:
        raise ValueError("maximum_path_day must be positive")
    daily_path_obj = Path(daily_dir)
    prepared = candidates.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    prepared["symbol"] = prepared.get("symbol", prepared["ts_code"]).astype(str)
    prepared["_candidate_order"] = np.arange(len(prepared))
    symbols = prepared["symbol"].dropna().unique().tolist()
    start_date = prepared["date"].min().strftime("%Y%m%d")
    end_date = (
        prepared["date"].max()
        + pd.Timedelta(days=maximum_path_day * 3 + 10)
    ).strftime("%Y%m%d")
    store = MarketDataStore(
        MarketDataStoreConfig.from_env(root=daily_path_obj.parent)
    )
    market = store.read_market_range(
        daily_path_obj.name,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
        ],
    )
    if market.empty:
        raise RuntimeError(f"No canonical daily rows found under {daily_path_obj.parent}")
    market["ts_code"] = market["ts_code"].astype(str)
    market_by_symbol = {
        symbol: frame for symbol, frame in market.groupby("ts_code", sort=False)
    }
    candidate_by_symbol = {
        symbol: frame for symbol, frame in prepared.groupby("symbol", sort=False)
    }
    frames: list[pd.DataFrame] = []

    for symbol in symbols:
        raw_daily = market_by_symbol.get(symbol)
        if raw_daily is None or raw_daily.empty:
            continue
        daily = raw_daily.copy()
        daily["date"] = pd.to_datetime(
            daily["trade_date"].astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
        daily = build_continuous_ohlc(
            daily.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        )
        daily_dates = pd.DatetimeIndex(daily["date"])
        symbol_candidates = candidate_by_symbol[symbol].copy().reset_index(drop=True)
        positions = daily_dates.get_indexer(
            pd.DatetimeIndex(symbol_candidates["date"])
        )
        matched = positions >= 0
        if not matched.any():
            continue
        symbol_candidates = symbol_candidates.loc[matched].reset_index(drop=True)
        positions = positions[matched]
        future: dict[str, object] = {
            "_adjusted_signal_close": daily["close"].to_numpy(dtype=float)[positions]
        }
        for day in range(1, maximum_path_day + 1):
            future_positions = positions + day
            valid = future_positions < len(daily)
            future_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
            future_dates[valid] = daily_dates.to_numpy(dtype="datetime64[ns]")[
                future_positions[valid]
            ]
            future[f"date_t{day}"] = future_dates
            for column in ("open", "high", "low", "close"):
                values = np.full(len(positions), np.nan)
                source = daily[column].to_numpy(dtype=float)
                values[valid] = source[future_positions[valid]]
                future[f"{column}_t{day}"] = values
        enriched = pd.concat(
            [symbol_candidates, pd.DataFrame(future)],
            axis=1,
        )
        enriched["entry_open"] = enriched["open_t1"]
        enriched["close"] = enriched["_adjusted_signal_close"]
        frames.append(enriched.drop(columns="_adjusted_signal_close"))

    if not frames:
        raise RuntimeError(
            f"No candidate symbols matched canonical daily data under {daily_path_obj.parent}"
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("_candidate_order")
        .drop(columns="_candidate_order")
        .dropna(subset=["entry_open"])
        .reset_index(drop=True)
    )


def entry_signal_mask(frame: pd.DataFrame, rule: SwingEntryRule) -> pd.Series:
    """Return a strict, NA-safe mask using signal-date information only."""

    missing = sorted(REQUIRED_SIGNAL_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"signal frame missing columns: {missing}")
    mask = (
        frame["is_good_stock"].fillna(False).astype(bool)
        & frame["good_stock_score"].ge(rule.min_good_stock_score)
        & frame["volatility_60d_cross_section_pct"].le(
            rule.max_volatility_percentile
        )
        & frame["return_120d"].between(
            rule.min_return_120d,
            rule.max_return_120d,
            inclusive="both",
        )
        & frame["close_to_ma20"].between(
            rule.min_close_to_ma20,
            rule.max_close_to_ma20,
            inclusive="both",
        )
        & frame["close_to_ma60"].gt(rule.min_close_to_ma60)
        & frame["close_to_ma120"].gt(rule.min_close_to_ma120)
        & frame["ma_120_slope_20d"].gt(rule.min_ma120_slope)
        & frame["market_return_13w"].gt(rule.min_market_return_13w)
        & frame["market_return_26w"].gt(rule.min_market_return_26w)
        & frame["market_drawdown_52w"].gt(rule.min_market_drawdown_52w)
        & frame["market_volatility_13w"].lt(
            rule.max_market_volatility_13w
        )
    )
    if rule.min_historical_value_score is not None:
        mask &= frame["historical_value_score_5y"].ge(
            rule.min_historical_value_score
        )
    return mask.fillna(False)


def apply_entry_execution_gates(
    frame: pd.DataFrame,
    rule: SwingEntryRule,
    *,
    one_price_tolerance: float = 1e-10,
    locked_limit_threshold: float = 0.048,
) -> pd.DataFrame:
    """Apply next-open gap and conservative one-price-limit entry gates."""

    required = {"close", "entry_open", "high_t1", "low_t1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"future-price frame missing columns: {missing}")
    out = frame.copy()
    out["entry_gap"] = out["entry_open"] / out["close"] - 1.0
    scale = out[["entry_open", "high_t1", "low_t1"]].abs().max(axis=1).clip(
        lower=1.0
    )
    one_price = (out["high_t1"] - out["low_t1"]).abs().le(
        scale * one_price_tolerance
    )
    locked_up = one_price & out["entry_gap"].ge(locked_limit_threshold)
    valid = (
        out["entry_open"].gt(0)
        & out["entry_gap"].between(
            rule.min_entry_gap,
            rule.max_entry_gap,
            inclusive="both",
        )
        & ~locked_up
    )
    out["entry_locked_up_proxy"] = locked_up
    return out.loc[valid].copy()


def apply_same_symbol_cooldown(
    frame: pd.DataFrame,
    cooldown_calendar_days: int,
    *,
    symbol_column: str = "ts_code",
    date_column: str = "date",
) -> pd.DataFrame:
    """Keep only the first signal inside each symbol's cooldown window."""

    if cooldown_calendar_days < 0:
        raise ValueError("cooldown_calendar_days must be non-negative")
    if frame.empty:
        return frame.copy()
    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column])
    ordered["_cooldown_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(
        [symbol_column, date_column, "_cooldown_order"]
    )
    keep = np.zeros(len(ordered), dtype=bool)
    last_kept: dict[str, pd.Timestamp] = {}
    for position, (_, row) in enumerate(ordered.iterrows()):
        symbol = str(row[symbol_column])
        signal_date = pd.Timestamp(row[date_column])
        previous = last_kept.get(symbol)
        if previous is None or (
            signal_date - previous
        ).days >= cooldown_calendar_days:
            keep[position] = True
            last_kept[symbol] = signal_date
    return (
        ordered.iloc[np.flatnonzero(keep)]
        .sort_values("_cooldown_order")
        .drop(columns="_cooldown_order")
        .reset_index(drop=True)
    )


def _is_locked_limit_down(
    open_price: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    previous_close: np.ndarray,
    *,
    one_price_tolerance: float,
    locked_limit_threshold: float,
) -> np.ndarray:
    scale = np.maximum.reduce(
        [np.abs(open_price), np.abs(high), np.abs(low), np.ones(len(open_price))]
    )
    one_price = np.abs(high - low) <= scale * one_price_tolerance
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = open_price / previous_close - 1.0
    return one_price & np.isfinite(gap) & (gap <= -locked_limit_threshold)


def _round_trip_net_return(
    entry_fill: float,
    exit_fill: float,
    config: AShareExecutionConfig,
    notional: float,
) -> tuple[float, float, float, int]:
    lots = int(notional // (entry_fill * config.lot_size))
    shares = lots * config.lot_size
    if shares <= 0:
        return np.nan, np.nan, np.nan, 0
    buy_value = shares * entry_fill
    sell_value = shares * exit_fill
    buy_commission = max(buy_value * config.commission_rate, config.min_commission)
    sell_commission = max(
        sell_value * config.commission_rate,
        config.min_commission,
    )
    buy_fee = buy_commission + buy_value * config.transfer_fee_rate
    sell_fee = (
        sell_commission
        + sell_value * config.transfer_fee_rate
        + sell_value * config.stamp_tax_rate
    )
    invested = buy_value + buy_fee
    net_return = (sell_value - sell_fee - invested) / invested
    gross_return = exit_fill / entry_fill - 1.0
    return net_return, gross_return, buy_fee + sell_fee, shares


def simulate_swing_exits(
    frame: pd.DataFrame,
    rule: SwingExitRule,
    *,
    execution: AShareExecutionConfig | None = None,
    notional: float = 100_000.0,
    one_price_tolerance: float = 1e-10,
    locked_limit_threshold: float = 0.048,
    result_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Simulate barriers with T+1, costs, slippage and limit-down delays.

    If stop and target are both touched on one bar, the stop is assumed first.
    A gap through the stop fills at the opening price.  A one-price limit-down
    proxy prevents selling and carries the position to the next available bar.
    """

    config = execution or AShareExecutionConfig(slippage=0.0005)
    if not config.t_plus_one:
        raise ValueError("swing research requires A-share T+1 execution")
    if notional <= 0:
        raise ValueError("notional must be positive")
    maximum_day = rule.hold_days + 1 + rule.exit_grace_days
    required = {"date", "ts_code", "entry_open"}
    for day in range(1, maximum_day + 1):
        required.update(
            {
                f"date_t{day}",
                f"open_t{day}",
                f"high_t{day}",
                f"low_t{day}",
                f"close_t{day}",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"future-price frame missing columns: {missing}")
    if frame.empty:
        return pd.DataFrame()

    entry_open = frame["entry_open"].to_numpy(dtype=float)
    entry_fill = entry_open * (1.0 + config.slippage)
    count = len(frame)
    exit_raw = np.full(count, np.nan)
    exit_day = np.full(count, -1, dtype=int)
    exit_date = np.full(count, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_type = np.full(count, "unresolved", dtype=object)
    valid_entry = np.isfinite(entry_fill) & (entry_fill > 0)
    stop_price = entry_fill * (1.0 - rule.stop_loss)
    target_price = entry_fill * (1.0 + rule.take_profit)

    # Day 1 is the buy day.  The loop starts at day 2 to enforce T+1.
    for day in range(2, maximum_day + 1):
        unresolved = valid_entry & ~np.isfinite(exit_raw)
        if not unresolved.any():
            break
        open_ = frame[f"open_t{day}"].to_numpy(dtype=float)
        high = frame[f"high_t{day}"].to_numpy(dtype=float)
        low = frame[f"low_t{day}"].to_numpy(dtype=float)
        close = frame[f"close_t{day}"].to_numpy(dtype=float)
        previous_close = frame[f"close_t{day - 1}"].to_numpy(dtype=float)
        dates = pd.to_datetime(frame[f"date_t{day}"]).to_numpy(
            dtype="datetime64[ns]"
        )
        valid_bar = (
            unresolved
            & np.isfinite(open_)
            & np.isfinite(high)
            & np.isfinite(low)
            & np.isfinite(close)
            & np.isfinite(previous_close)
            & (open_ > 0)
            & (high >= low)
        )
        locked_down = valid_bar & _is_locked_limit_down(
            open_,
            high,
            low,
            previous_close,
            one_price_tolerance=one_price_tolerance,
            locked_limit_threshold=locked_limit_threshold,
        )
        sellable = valid_bar & ~locked_down
        if not sellable.any():
            continue

        if day <= rule.hold_days + 1:
            stop_hit = sellable & (low <= stop_price)
            if stop_hit.any():
                exit_raw[stop_hit] = np.where(
                    open_[stop_hit] <= stop_price[stop_hit],
                    open_[stop_hit],
                    stop_price[stop_hit],
                )
                exit_day[stop_hit] = day
                exit_date[stop_hit] = dates[stop_hit]
                exit_type[stop_hit] = "stop_loss"

            still_open = sellable & ~np.isfinite(exit_raw)
            target_hit = still_open & (high >= target_price)
            if target_hit.any():
                # Conservative: do not grant price improvement on a gap-up.
                exit_raw[target_hit] = target_price[target_hit]
                exit_day[target_hit] = day
                exit_date[target_hit] = dates[target_hit]
                exit_type[target_hit] = "take_profit"

            expiry = (
                sellable
                & ~np.isfinite(exit_raw)
                & (day == rule.hold_days + 1)
            )
            if expiry.any():
                exit_raw[expiry] = close[expiry]
                exit_day[expiry] = day
                exit_date[expiry] = dates[expiry]
                exit_type[expiry] = "expiry"
        else:
            delayed = sellable & ~np.isfinite(exit_raw)
            if delayed.any():
                exit_raw[delayed] = open_[delayed]
                exit_day[delayed] = day
                exit_date[delayed] = dates[delayed]
                exit_type[delayed] = "delayed_expiry"

    rows: list[dict[str, object]] = []
    for position, (_, signal) in enumerate(frame.iterrows()):
        if not np.isfinite(exit_raw[position]):
            continue
        exit_fill = exit_raw[position] * (1.0 - config.slippage)
        net_return, gross_return, fees, shares = _round_trip_net_return(
            entry_fill[position],
            exit_fill,
            config,
            notional,
        )
        if not np.isfinite(net_return):
            continue
        signal_payload = (
            signal.to_dict()
            if result_columns is None
            else {
                column: signal[column]
                for column in result_columns
                if column in signal.index
            }
        )
        rows.append(
            {
                **signal_payload,
                "entry_fill": entry_fill[position],
                "exit_raw": exit_raw[position],
                "exit_fill": exit_fill,
                "exit_date": pd.Timestamp(exit_date[position]),
                "exit_day": int(exit_day[position]),
                "exit_type": str(exit_type[position]),
                "gross_return": gross_return,
                "net_return": net_return,
                "fees": fees,
                "shares": shares,
            }
        )
    return pd.DataFrame(rows)


def wilson_lower_bound(wins: int, total: int, *, z: float = 1.95996398454) -> float:
    """Two-sided 95% Wilson confidence interval lower bound."""

    if total <= 0:
        return np.nan
    probability = wins / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    adjustment = z * sqrt(
        probability * (1.0 - probability) / total
        + z * z / (4.0 * total * total)
    )
    return (center - adjustment) / denominator


def _event_equity_max_drawdown(trades: pd.DataFrame) -> float:
    event_returns = trades.groupby("entry_date")["net_return"].mean().sort_index()
    equity = (1.0 + event_returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else np.nan


def summarize_swing_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    """Summarize executable net trade returns without annualizing overlap."""

    if trades.empty or "net_return" not in trades:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": np.nan,
            "win_rate_wilson_lower_95": np.nan,
            "avg_net_return": np.nan,
            "profit_factor": np.nan,
            "event_equity_max_drawdown": np.nan,
        }
    returns = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    if returns.empty:
        return summarize_swing_trades(pd.DataFrame())
    wins = int((returns > 0).sum())
    gains = returns[returns > 0]
    losses = returns[returns <= 0]
    loss_sum = float(-losses.sum())
    years = pd.to_datetime(trades.loc[returns.index, "entry_date"]).dt.year
    annual = returns.groupby(years).mean()
    holding_sessions = (
        pd.to_numeric(
            trades.loc[returns.index, "exit_day"],
            errors="coerce",
        )
        - 1
    )
    return {
        "trades": int(len(returns)),
        "wins": wins,
        "win_rate": float(wins / len(returns)),
        "win_rate_wilson_lower_95": float(
            wilson_lower_bound(wins, len(returns))
        ),
        "avg_net_return": float(returns.mean()),
        "median_net_return": float(returns.median()),
        "avg_win": float(gains.mean()) if not gains.empty else np.nan,
        "avg_loss": float(losses.mean()) if not losses.empty else np.nan,
        "profit_factor": (
            float(gains.sum() / loss_sum) if loss_sum > 0 else np.nan
        ),
        "event_equity_max_drawdown": _event_equity_max_drawdown(
            trades.loc[returns.index]
        ),
        "average_holding_sessions": float(holding_sessions.mean()),
        "median_holding_sessions": float(holding_sessions.median()),
        "positive_year_share": float((annual > 0).mean()),
        "stop_rate": float((trades.loc[returns.index, "exit_type"] == "stop_loss").mean()),
        "target_rate": float((trades.loc[returns.index, "exit_type"] == "take_profit").mean()),
        "expiry_rate": float(
            trades.loc[returns.index, "exit_type"].isin(
                ["expiry", "delayed_expiry"]
            ).mean()
        ),
    }


def slice_trades(
    trades: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    dates = pd.to_datetime(trades["entry_date"])
    return trades.loc[
        dates.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
    ].copy()


def evaluate_periods(
    trades: pd.DataFrame,
    periods: Iterable[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, start, end in periods:
        sample = slice_trades(trades, start, end)
        rows.append(
            {
                "period": name,
                "start": start,
                "end": end,
                **summarize_swing_trades(sample),
            }
        )
    return pd.DataFrame(rows)


def select_rule_without_holdout(
    summary: pd.DataFrame,
    *,
    selection_period: str = "selection_2020_2023",
    minimum_trades: int = 80,
    minimum_profit_factor: float = 1.10,
    minimum_positive_year_share: float = 0.50,
) -> tuple[pd.Series, bool]:
    """Choose only from a named selection period; never inspect holdout columns."""

    sample = summary.loc[summary["period"].eq(selection_period)].copy()
    if sample.empty:
        raise ValueError(f"selection period not found: {selection_period}")
    eligible = sample.loc[
        sample["trades"].ge(minimum_trades)
        & sample["avg_net_return"].gt(0)
        & sample["profit_factor"].ge(minimum_profit_factor)
        & sample["positive_year_share"].ge(minimum_positive_year_share)
    ].copy()
    passed_constraints = not eligible.empty
    ranking = eligible if passed_constraints else sample
    ranking = ranking.sort_values(
        [
            "win_rate_wilson_lower_95",
            "profit_factor",
            "avg_net_return",
            "trades",
        ],
        ascending=[False, False, False, False],
        na_position="last",
    )
    return ranking.iloc[0], passed_constraints
