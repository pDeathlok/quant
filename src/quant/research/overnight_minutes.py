"""Point-in-time minute features and explicitly uncertain exit-path marks."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICES = ["open", "high", "low", "close"]


def daily_limit_exit(entry: float, next_pre_close: float, next_high: float, next_close: float,
                     take_profit: float, *, cost_bps: float = 20., penetration_bps: float = 10.) -> dict:
    """Pre-set sell-limit reachability proxy, no stop and no opening improvement."""
    if (not np.isfinite([take_profit, cost_bps, penetration_bps]).all() or not 0 < take_profit < 1
            or cost_bps < 0 or penetration_bps < 0):
        raise ValueError("invalid limit policy")
    if (not np.isfinite([entry, next_pre_close, next_high, next_close]).all()
            or min(entry, next_pre_close, next_high, next_close) <= 0 or next_high < next_close):
        return {"status": "missing_or_invalid_daily_price", "net": np.nan}
    if abs(next_pre_close - entry) > .011:
        return {"status": "corporate_action_unresolved", "net": np.nan}
    limit = np.ceil(entry * (1 + take_profit) * 100 - 1e-9) / 100
    reached = next_high >= limit + entry * penetration_bps / 10000
    price = limit if reached else next_close
    return {"status": "ok", "net": float(price / entry - 1 - cost_bps / 10000),
            "limit_price": float(limit), "penetrated": bool(reached),
            "exit_proxy": "preset_limit_price" if reached else "precommitted_close_price"}


def daily_exit_labels(entry: float, next_pre_close: float, next_open: float,
                      next_close: float, cost_bps: float = 20.) -> dict:
    """Official daily-price proxies, independent of minute closing marks."""
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("invalid cost")
    result = {}
    entry_known = np.isfinite([entry, next_pre_close]).all() and min(entry, next_pre_close) > 0
    for name, price in (("open", next_open), ("close", next_close)):
        known = entry_known and np.isfinite(price) and price > 0
        status = "ok" if known else "missing_or_invalid_daily_price"
        if known and abs(next_pre_close - entry) > .011:
            status = "corporate_action_unresolved"
        result[f"target_net_{name}"] = price / entry - 1 - cost_bps / 10000 if status == "ok" else np.nan
        result[f"label_quality_{name}"] = status
    return result


def lagged_factor_row(frame: pd.DataFrame, decision_date: pd.Timestamp) -> pd.Series | None:
    """Latest strictly earlier row; same-day factors are not known pre-close."""
    dates = pd.to_datetime(frame.date)
    if dates.duplicated().any():
        raise ValueError("factor dates must be unique within a stock")
    available = frame.loc[dates.lt(pd.Timestamp(decision_date).normalize())]
    if available.empty:
        return None
    return available.loc[pd.to_datetime(available.date).idxmax()]


def session_grid(day: pd.Timestamp, include_auction: bool = True) -> pd.DatetimeIndex:
    """Tushare five-minute convention including the opening-auction row.

    Provider samples must be audited against this contract before training;
    do not silently resample incompatible timestamps into these slots.
    """
    day = pd.Timestamp(day).normalize()
    return pd.date_range(day + pd.Timedelta(hours=9, minutes=30 if include_auction else 35),
                         day + pd.Timedelta(hours=11, minutes=30), freq="5min").append(
        pd.date_range(day + pd.Timedelta(hours=13, minutes=5),
                      day + pd.Timedelta(hours=15), freq="5min"))


def valid_prices(frame: pd.DataFrame) -> bool:
    prices = frame[PRICES].to_numpy(float)
    return bool(np.isfinite(prices).all() and (prices > 0).all()
                and frame.high.ge(frame[["open", "close"]].max(axis=1)).all()
                and frame.low.le(frame[["open", "close"]].min(axis=1)).all())


def preclose_snapshot(frame: pd.DataFrame, previous_close: float,
                      cutoff: pd.Timestamp, *, include_auction: bool = True) -> dict | None:
    """At 14:55 use only stamps through 14:50, conservatively delayed 5min."""
    if not np.isfinite(previous_close) or previous_close <= 0:
        raise ValueError("previous close must be known and positive")
    cutoff = pd.Timestamp(cutoff)
    times = pd.to_datetime(frame.trade_time)
    available = times + pd.Timedelta(minutes=5)
    part = frame[times.dt.normalize().eq(cutoff.normalize()) & available.le(cutoff)].copy()
    part["trade_time"] = pd.to_datetime(part.trade_time)
    part = part.sort_values("trade_time")
    expected = session_grid(cutoff, include_auction)
    expected = expected[expected + pd.Timedelta(minutes=5) <= cutoff]
    if len(expected) == 0 or not pd.DatetimeIndex(part.trade_time).equals(expected):
        return None
    if not valid_prices(part) or not np.isfinite(part[["vol", "amount"]].to_numpy(float)).all():
        return None
    if part.vol.lt(0).any() or part.amount.lt(0).any() or part.vol.sum() <= 0 or part.amount.sum() <= 0:
        return None
    close, high, low = float(part.close.iloc[-1]), float(part.high.max()), float(part.low.min())
    volume, amount = float(part.vol.sum()), float(part.amount.sum())
    recent = part[part.trade_time.gt(part.trade_time.iloc[-1] - pd.Timedelta(minutes=30))]
    # Do not inspect the final close or final daily volume, even for validation.
    return {"observed_bars": len(part), "snapshot_price": close,
            "return_so_far": close / previous_close - 1,
            "gap": float(part.open.iloc[0] / previous_close - 1),
            "range_so_far": (high - low) / previous_close,
            "close_position": (close - low) / (high - low) if high > low else .5,
            "vwap_deviation": close / (amount / volume) - 1,
            "volume_so_far": volume, "amount_so_far": amount,
            "last30_return": float(close / recent.open.iloc[0] - 1),
            "last30_volume_fraction": float(recent.vol.sum() / volume),
            "realized_variance": float(np.square(np.diff(np.log(part.close.to_numpy(float)))).sum())}


def minute_exit(frame: pd.DataFrame, entry_price: float, take_profit: float,
                stop_loss: float, cost_bps: float = 20., *, include_auction: bool = True) -> dict:
    """Path-based price proxies, not fill guarantees or a bounded-loss promise."""
    if (not np.isfinite([entry_price, take_profit, stop_loss, cost_bps]).all()
            or entry_price <= 0 or not 0 < take_profit < 1 or not 0 < stop_loss < 1 or cost_bps < 0):
        raise ValueError("invalid exit parameters")
    if frame.empty:
        return {"status": "incomplete_session"}
    part = frame.copy()
    part["trade_time"] = pd.to_datetime(part.trade_time)
    part = part.sort_values("trade_time")
    if not pd.DatetimeIndex(part.trade_time).equals(session_grid(part.trade_time.iloc[0], include_auction)):
        return {"status": "incomplete_session"}
    if not valid_prices(part) or not np.isfinite(part.vol).all() or part.vol.lt(0).any():
        return {"status": "invalid_prices_or_volume"}
    active = part[part.vol.gt(0)]
    if active.empty:
        return {"status": "no_observed_volume"}
    cost = cost_bps / 10000
    upper, lower = entry_price * (1 + take_profit), entry_price * (1 - stop_loss)
    for row in active.itertuples():
        if row.open >= upper or row.open <= lower:
            gross = row.open / entry_price - 1
            return {"status": "bar_open_price_proxy", "exit_time": str(row.trade_time),
                    "net_lower": gross - cost, "net_upper": gross - cost, "ambiguous": False}
        hit_up, hit_down = row.high >= upper, row.low <= lower
        if hit_up or hit_down:
            return {"status": "threshold_price_proxy", "exit_time": str(row.trade_time),
                    "net_lower": (-stop_loss if hit_down else take_profit) - cost,
                    "net_upper": (take_profit if hit_up else -stop_loss) - cost,
                    "ambiguous": bool(hit_up and hit_down)}
    gross = float(part.close.iloc[-1] / entry_price - 1)
    return {"status": "close_price_proxy", "exit_time": str(part.trade_time.iloc[-1]),
            "net_lower": gross - cost, "net_upper": gross - cost, "ambiguous": False}


def close_check_exit(frame: pd.DataFrame, entry_price: float, take_profit: float,
                     stop_loss: float, cost_bps: float = 20., *, include_auction: bool = False,
                     use_stop: bool = True) -> dict:
    """Check completed closes, execute at the next observed bar-open proxy.

    Intrabar extrema do not trigger orders. A closing exit is precommitted,
    not decided after seeing the final price. These marks still require fill
    and slippage validation; a stop does not bound gap losses.
    """
    if (not np.isfinite([entry_price, take_profit, stop_loss, cost_bps]).all()
            or entry_price <= 0 or not 0 < take_profit < 1 or not 0 < stop_loss < 1 or cost_bps < 0):
        raise ValueError("invalid exit parameters")
    if frame.empty:
        return {"status": "incomplete_session"}
    part = frame.copy()
    part["trade_time"] = pd.to_datetime(part.trade_time)
    part = part.sort_values("trade_time").reset_index(drop=True)
    if not pd.DatetimeIndex(part.trade_time).equals(session_grid(part.trade_time.iloc[0], include_auction)):
        return {"status": "incomplete_session"}
    values = part[["open", "close", "vol"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :2] <= 0).any() or (values[:, 2] < 0).any():
        return {"status": "invalid_prices_or_volume"}
    cost = cost_bps / 10000
    for index, row in part.iloc[:-1].iterrows():
        if row.vol <= 0:
            continue
        move = row.close / entry_price - 1
        if move >= take_profit or (use_stop and move <= -stop_loss):
            following = part.iloc[index + 1:]
            following = following[following.vol.gt(0)]
            if following.empty:
                return {"status": "no_observed_execution_volume", "trigger_time": str(row.trade_time)}
            execution = following.iloc[0]
            return {"status": "next_bar_open_price_proxy", "trigger_time": str(row.trade_time),
                    "execution_bar": str(execution.trade_time),
                    "net": float(execution.open / entry_price - 1 - cost)}
    if part.vol.iloc[-1] <= 0:
        return {"status": "no_observed_execution_volume"}
    return {"status": "precommitted_close_price_proxy", "execution_bar": str(part.trade_time.iloc[-1]),
            "net": float(part.close.iloc[-1] / entry_price - 1 - cost)}
