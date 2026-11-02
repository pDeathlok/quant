"""Finite, causal stock/entry/exit experiments for the weekly left-side control."""

from __future__ import annotations

import numpy as np
import pandas as pd

FILTERS = (
    "original", "stabilize", "dry_pullback", "low_volatility", "not_hot",
    "avoid_slide", "firm_close", "balanced",
)
ENTRIES = ("next_close", "wait_green", "reclaim_ma5", "recover_signal")
EXITS = ("fixed20", "stop8", "take15", "trail8", "ma10_two", "stale5", "stop_trail")
DEVELOPMENT = ("development_2024", "development_2025H1", "selection")


def stock_mask(frame: pd.DataFrame, rule: str) -> pd.Series:
    not_hot = frame.ma20_distance.le(0.08) & frame.return20.le(0.25)
    low_vol = frame.volatility20.le(0.035)
    rules = {
        "original": pd.Series(True, index=frame.index),
        "stabilize": frame.left_stabilize,
        "dry_pullback": frame.left_pullback,
        "low_volatility": low_vol,
        "not_hot": not_hot,
        "avoid_slide": frame.return5.ge(-0.10) & frame.return1.ge(-0.04),
        "firm_close": frame.close_position.ge(0.6) & frame.return1.ge(0),
        "balanced": not_hot & low_vol & frame.close_position.ge(0.4),
    }
    return rules[rule].fillna(False).astype(bool)


def schedule_paths(
    close: np.ndarray, ma5: np.ndarray, ma10: np.ndarray, entry: str, exit_rule: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paths include signal at 0 and deadline at 21; all triggers execute next close.

    Missing closes do not confirm a trigger. Inputs are already restricted to the
    frozen original complete-path event universe. -1 entry means expired order.
    """
    if entry not in ENTRIES or exit_rule not in EXITS or close.shape[1] != 22:
        raise ValueError("unsupported entry, exit or 20-session path")
    n = len(close)
    entry_at = np.ones(n, dtype=int)
    confirmation = np.zeros(n, dtype=int)
    if entry != "next_close":
        current, previous = close[:, 1:4], close[:, :3]
        if entry == "wait_green":
            confirmed = current > previous
        elif entry == "reclaim_ma5":
            confirmed = (current > ma5[:, 1:4]) & (previous <= ma5[:, :3])
        else:
            confirmed = (current >= close[:, [0]] * 1.01) & (current > previous)
        any_confirmed = confirmed.any(axis=1)
        confirmation = np.where(any_confirmed, confirmed.argmax(axis=1) + 1, -1)
        entry_at = np.where(any_confirmed, confirmation + 1, -1)
    safe_entry = entry_at.clip(min=0)
    entry_price = close[np.arange(n), safe_entry]
    exit_at = np.full(n, 21, dtype=int)
    exit_trigger = np.full(n, -1, dtype=int)
    peak = entry_price.copy()
    done = entry_at < 0
    for day in range(1, 21):
        held = day >= entry_at
        peak = np.where(held & ~done, np.maximum(peak, close[:, day]), peak)
        gain = close[:, day] / entry_price - 1
        peak_gain = peak / entry_price - 1
        drawdown = close[:, day] / peak - 1
        if exit_rule == "fixed20":
            trigger = np.zeros(n, dtype=bool)
        elif exit_rule == "stop8":
            trigger = gain <= -0.08
        elif exit_rule == "take15":
            trigger = gain >= 0.15
        elif exit_rule == "trail8":
            trigger = (peak_gain >= 0.08) & (drawdown <= -0.08)
        elif exit_rule == "ma10_two":
            trigger = (
                (close[:, day] < ma10[:, day])
                & (close[:, day - 1] < ma10[:, day - 1])
                & (day - 1 >= entry_at)
            )
        elif exit_rule == "stale5":
            trigger = (day - entry_at >= 5) & (gain <= 0)
        else:
            trigger = (gain <= -0.08) | ((peak_gain >= 0.10) & (drawdown <= -0.08))
        trigger &= (day > entry_at) & ~done
        exit_at[trigger] = day + 1
        exit_trigger[trigger] = day
        done |= trigger
    return entry_at, exit_at, confirmation, exit_trigger


def materialize(
    events: pd.DataFrame, paths: dict[str, np.ndarray], calendar: pd.DatetimeIndex,
    entry: str, exit_rule: str,
) -> tuple[pd.DataFrame, dict]:
    start, end, confirmed, trigger = schedule_paths(
        paths["close"], paths["ma5"], paths["ma10"], entry, exit_rule
    )
    keep = start >= 0
    out = events.loc[keep].copy()
    positions = calendar.get_indexer(out.date)
    start, end, confirmed, trigger = (x[keep] for x in (start, end, confirmed, trigger))
    out["origin_date"] = out.date
    out["origin_exit_date"] = out.exit_date
    out["date"] = calendar[positions + confirmed]
    out["entry_date"] = calendar[positions + start]
    out["exit_date"] = calendar[positions + end]
    out["exit_trigger_date"] = pd.to_datetime(
        [calendar[p + t] if t >= 0 else pd.NaT for p, t in zip(positions, trigger, strict=True)]
    )
    close = paths["close"][keep]
    row = np.arange(len(close))
    out["net_return"] = close[row, end] / close[row, start] - 1 - 0.002
    out["planned_exit_reason"] = np.where(trigger >= 0, exit_rule, "deadline")
    # Confirmation delays can map several original candidates to the same order.
    out = out.sort_values("origin_date", ascending=False).drop_duplicates(["symbol", "date"])
    audit = {
        "input_events": len(events), "expired_unconfirmed": int((~keep).sum()),
        "duplicate_confirmations_removed": int(keep.sum() - len(out)),
        "orders": len(out),
    }
    if not (out.date < out.entry_date).all() or not (out.entry_date < out.exit_date).all():
        raise ValueError("invalid confirmation or execution timing")
    dynamic = out.exit_trigger_date.notna()
    if not (out.loc[dynamic, "exit_trigger_date"] < out.loc[dynamic, "exit_date"]).all():
        raise ValueError("same-close trigger execution")
    if not (out.exit_date <= out.origin_exit_date).all():
        raise ValueError("extended original holding deadline")
    return out, audit


def development_rank(record: dict, *, enforce_drawdown: bool = True) -> tuple[float, float, float]:
    blocks = [record[name] for name in DEVELOPMENT]
    if any(
        x["trades"] < 40 or (enforce_drawdown and x["max_drawdown"] < -0.35)
        for x in blocks
    ):
        return (-float("inf"), -float("inf"), -float("inf"))
    returns = [x["total_return"] for x in blocks]
    return min(returns), float(np.mean(returns)), min(x["max_drawdown"] for x in blocks)
