"""Chronological score fusion, shallow rule extraction and cash-aware backtests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text


def purged_split(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if (events.exit_date <= events.date).any() or (events.entry_date <= events.date).any():
        raise ValueError("labels must occur strictly after the signal")
    return {
        "train": events[events.date.ge("2024-01-01") & events.exit_date.lt("2025-01-01")].copy(),
        "feedback": events[events.date.ge("2025-01-01") & events.exit_date.lt("2025-07-01")].copy(),
        "selection": events[
            events.date.ge("2025-07-01") & events.exit_date.lt("2026-01-01")
        ].copy(),
        "test": events[events.date.ge("2026-01-01")].copy(),
    }


def event_metrics(events: pd.DataFrame) -> dict[str, float | int | None]:
    values = events.net_return.to_numpy(float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {
        "trades": len(values),
        "win_rate": float((values > 0).mean()) if len(values) else None,
        "average_net_return": float(values.mean()) if len(values) else None,
        "profit_factor": float(gains / losses) if losses > 0 else None,
    }


def wilson_lower(wins: int, count: int, z: float = 1.96) -> float:
    if not count:
        return 0.0
    p = wins / count
    return float(
        (p + z * z / (2 * count) - z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)))
        / (1 + z * z / count)
    )


def choose_rule(
    train: pd.DataFrame,
    selection: pd.DataFrame,
    features: list[str],
    *,
    supervised: bool,
    minimum: int = 30,
) -> tuple[DecisionTreeRegressor, list[int], list[dict]]:
    """Fit only train. Admit leaves only on selection, with an explicit all-cash result."""
    candidates = []
    records = []
    for depth in (1, 2, 3):
        tree = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=minimum, random_state=20261021
        )
        target = train.net_return.gt(0).astype(float) if supervised else train.baseline_score
        tree.fit(train[features], target)
        leaves = tree.apply(selection[features])
        admitted = []
        for leaf in np.unique(leaves):
            part = selection[leaves == leaf]
            metrics = event_metrics(part)
            lower = wilson_lower(int(part.net_return.gt(0).sum()), len(part))
            eligible = (
                len(part) >= minimum
                and metrics["win_rate"] >= 0.60
                and lower >= 0.50
                and metrics["average_net_return"] > 0
                and (metrics["profit_factor"] or 0) > 1
            )
            records.append(
                {
                    "depth": depth,
                    "leaf": int(leaf),
                    **metrics,
                    "wilson_lower": lower,
                    "eligible": eligible,
                }
            )
            if eligible:
                admitted.append(int(leaf))
        picks = selection[np.isin(leaves, admitted)]
        metrics = event_metrics(picks)
        # Reward supported successful trades; do not rank an empty strategy as 100% successful.
        utility = float(picks.net_return.sum()) if len(picks) else -math.inf
        candidates.append((utility, -depth, tree, admitted))
    chosen = max(candidates, key=lambda item: item[:2])
    return chosen[2], chosen[3], records


def rule_text(tree: DecisionTreeRegressor, features: list[str], leaves: list[int]) -> str:
    return f"Admitted leaves: {leaves}\n" + export_text(tree, feature_names=features, decimals=5)


def top_picks(events: pd.DataFrame, score: str, top_n: int = 20) -> pd.DataFrame:
    return (
        events.sort_values(["date", score, "symbol"], ascending=[True, False, True], kind="stable")
        .groupby("date", sort=False)
        .head(top_n)
    )


def choose_confidence_gate(
    selection: pd.DataFrame, score: str, *, minimum: int = 30, top_n: int = 20
) -> tuple[float | None, list[dict]]:
    records = []
    eligible = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        picks = top_picks(selection[selection[score] >= threshold], score, top_n)
        metrics = event_metrics(picks)
        lower = wilson_lower(int(picks.net_return.gt(0).sum()), len(picks))
        passed = (
            len(picks) >= minimum
            and metrics["win_rate"] >= 0.60
            and lower >= 0.50
            and metrics["average_net_return"] > 0
            and (metrics["profit_factor"] or 0) > 1
        )
        records.append(
            {"threshold": threshold, **metrics, "wilson_lower": lower, "eligible": passed}
        )
        if passed:
            eligible.append((float(picks.net_return.sum()), threshold))
    return (max(eligible)[1] if eligible else None), records


def paired_block_gain(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    seed: int = 20261021,
    repetitions: int = 1000,
) -> dict:
    """Paired five-session blocks preserve some dependence from five-day holdings."""
    merged = baseline[["date", "equity"]].merge(
        candidate[["date", "equity"]], on="date", validate="one_to_one", suffixes=("_base", "_new")
    )
    base = np.diff(np.log(np.r_[1.0, merged.equity_base]))
    new = np.diff(np.log(np.r_[1.0, merged.equity_new]))
    generator = np.random.default_rng(seed)
    gains = []
    for _ in range(repetitions):
        starts = generator.integers(0, len(base), int(np.ceil(len(base) / 5)))
        indices = np.concatenate([(np.arange(start, start + 5) % len(base)) for start in starts])[
            : len(base)
        ]
        gains.append(np.expm1(new[indices].sum()) - np.expm1(base[indices].sum()))
    low, high = np.quantile(gains, [0.025, 0.975])
    return {
        "gain": float(merged.equity_new.iloc[-1] - merged.equity_base.iloc[-1]),
        "paired_block_95ci": [float(low), float(high)],
        "positive_gain_supported": bool(low > 0),
        "days": len(base),
        "block_sessions": 5,
        "bootstrap_repetitions": repetitions,
    }


def capital_backtest(
    events: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    *,
    slots: int = 20,
    cost: float = 0.002,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Close execution, half cost per side, no duplicate stock, delayed blocked exits.

    bars contain adjusted close for valuation and raw buy/sell eligibility for
    execution. Missing limit data fail closed. Unfinished positions stay marked
    and reported open; they are never force-filled at the end of the test.
    """
    if slots < 1 or not 0 <= cost < 1:
        raise ValueError("invalid capital settings")
    prices = {symbol: frame.set_index("date") for symbol, frame in bars.items()}
    pending = {date: part for date, part in events.groupby("entry_date", sort=False)}
    cash = 1.0
    positions: dict[str, dict] = {}
    curve, closed = [], []
    rejected = 0
    for date in calendar:
        marks = {}
        for symbol, position in list(positions.items()):
            history = prices[symbol]
            if date in history.index:
                row = history.loc[date]
                position["mark"] = float(row.close)
                if date >= position["exit_date"] and bool(row.sell_ok):
                    gross = position["shares"] * position["mark"]
                    proceeds = gross * (1 - cost / 2)
                    cash += proceeds
                    closed.append(
                        {
                            "symbol": symbol,
                            "entry_date": position["entry_date"],
                            "exit_date": date,
                            "signal_date": position["signal_date"],
                            "net_return": proceeds / position["invested"] - 1,
                            "pnl": proceeds - position["invested"],
                        }
                    )
                    del positions[symbol]
                    continue
            marks[symbol] = position["shares"] * position["mark"]
        equity = cash + sum(marks.values())
        if date in pending:
            ordered = pending[date].sort_values(
                ["execution_score", "symbol"], ascending=[False, True]
            )
            for event in ordered.itertuples():
                if event.symbol in positions or len(positions) >= slots:
                    rejected += 1
                    continue
                history = prices[event.symbol]
                if date not in history.index or not bool(history.loc[date].buy_ok):
                    rejected += 1
                    continue
                budget = min(cash, equity / slots)
                if budget <= 1e-10:
                    continue
                price = float(history.loc[date].close)
                shares = budget / (price * (1 + cost / 2))
                cash -= budget
                positions[event.symbol] = {
                    "shares": shares,
                    "mark": price,
                    "invested": budget,
                    "entry_date": date,
                    "exit_date": event.exit_date,
                    "signal_date": event.date,
                }
        holding_value = sum(p["shares"] * p["mark"] for p in positions.values())
        equity = cash + holding_value
        curve.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "exposure": holding_value / equity,
                "positions": len(positions),
            }
        )
    curve_frame = pd.DataFrame(curve)
    closed_frame = pd.DataFrame(
        closed, columns=["symbol", "entry_date", "exit_date", "signal_date", "net_return", "pnl"]
    )
    equity = curve_frame.equity
    # Initial cash must be included in the running peak, including first-day fees.
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    metrics = {
        **event_metrics(closed_frame),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float((equity / peak - 1).min()),
        "cash_day_ratio": float(curve_frame.positions.eq(0).mean()),
        "mean_exposure": float(curve_frame.exposure.mean()),
        "open_positions": len(positions),
        "rejected_orders": rejected,
    }
    return curve_frame, closed_frame, metrics
