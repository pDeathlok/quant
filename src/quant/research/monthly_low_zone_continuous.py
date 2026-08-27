"""Stateful position-management research for monthly low-zone anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContinuousPolicy:
    """One frozen capital-allocation and structural-stop policy."""

    name: str
    initial_weight: float
    grid_drawdowns: tuple[float, ...] = ()
    grid_weights: tuple[float, ...] = ()
    structural_stop: bool = False
    maximum_reentries: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.initial_weight <= 1.0:
            raise ValueError("initial_weight must be in (0, 1]")
        if len(self.grid_drawdowns) != len(self.grid_weights):
            raise ValueError("grid drawdowns and weights must have equal length")
        if tuple(sorted(self.grid_drawdowns)) != self.grid_drawdowns:
            raise ValueError("grid drawdowns must be increasing")
        if any(not 0.0 < value < 1.0 for value in self.grid_drawdowns):
            raise ValueError("grid drawdowns must be in (0, 1)")
        if any(not 0.0 < value <= 1.0 for value in self.grid_weights):
            raise ValueError("grid weights must be in (0, 1]")
        if self.initial_weight + sum(self.grid_weights) > 1.0 + 1e-12:
            raise ValueError("policy weights must sum to at most one")
        if self.maximum_reentries < 0:
            raise ValueError("maximum_reentries must be non-negative")
        if self.maximum_reentries and not self.structural_stop:
            raise ValueError("reentries require a structural stop")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CONTINUOUS_POLICIES: tuple[ContinuousPolicy, ...] = (
    ContinuousPolicy("lump_sum", 1.0),
    ContinuousPolicy(
        "grid_40_30_30_down5_down10",
        0.40,
        (0.05, 0.10),
        (0.30, 0.30),
    ),
    ContinuousPolicy(
        "grid_40_30_30_down10_down20",
        0.40,
        (0.10, 0.20),
        (0.30, 0.30),
    ),
    ContinuousPolicy(
        "grid_40_30_30_down15_down30",
        0.40,
        (0.15, 0.30),
        (0.30, 0.30),
    ),
    ContinuousPolicy(
        "base_stop_reentry_2",
        1.0,
        structural_stop=True,
        maximum_reentries=2,
    ),
    ContinuousPolicy(
        "grid_base_stop_reentry_2",
        0.40,
        (0.10, 0.20),
        (0.30, 0.30),
        structural_stop=True,
        maximum_reentries=2,
    ),
)


@dataclass(frozen=True)
class ContinuousConfig:
    """Frozen causal execution assumptions for stateful anchor paths."""

    target_return: float = 0.15
    horizon_sessions: int = 504
    round_trip_cost_bps: float = 20.0
    minimum_reentry_wait_sessions: int = 20
    maximum_reentry_entry_delay_sessions: int = 5
    minimum_sessions_since_new_low: int = 20
    minimum_reentry_base_position: float = 0.50
    minimum_reentry_amount_thousand: float = 30_000.0
    maximum_reentry_drawdown_from_prior_peak: float = -0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.target_return < 1.0:
            raise ValueError("target_return must be in (0, 1)")
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        if self.round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps must be non-negative")
        if self.minimum_reentry_wait_sessions < 1:
            raise ValueError("minimum_reentry_wait_sessions must be positive")
        if self.maximum_reentry_entry_delay_sessions < 1:
            raise ValueError(
                "maximum_reentry_entry_delay_sessions must be positive"
            )
        if self.minimum_sessions_since_new_low < 1:
            raise ValueError("minimum_sessions_since_new_low must be positive")
        if not 0.0 <= self.minimum_reentry_base_position <= 1.0:
            raise ValueError("minimum_reentry_base_position must be in [0, 1]")
        if self.minimum_reentry_amount_thousand < 0.0:
            raise ValueError("minimum_reentry_amount_thousand must be non-negative")
        if not -1.0 < self.maximum_reentry_drawdown_from_prior_peak < 0.0:
            raise ValueError(
                "maximum_reentry_drawdown_from_prior_peak must be in (-1, 0)"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _normalize_calendar(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce")).dropna()
    values = pd.DatetimeIndex(sorted(values.normalize().unique()))
    if values.empty:
        raise ValueError("market_calendar must not be empty")
    return values


def _event_value(event: Mapping[str, Any] | pd.Series, key: str) -> Any:
    return event.get(key) if hasattr(event, "get") else event[key]


def _one_price_bar(row: pd.Series) -> bool:
    raw_open = float(row["open"])
    raw_high = float(row["high"])
    raw_low = float(row["low"])
    tolerance = max(abs(raw_open), 1.0) * 1e-10
    return max(raw_high, raw_open) - min(raw_low, raw_open) <= tolerance


def _event_key(event: Mapping[str, Any] | pd.Series) -> str:
    source = str(_event_value(event, "source_sample") or "unknown")
    signal_id = _event_value(event, "signal_id")
    symbol = str(_event_value(event, "ts_code"))
    month = str(_event_value(event, "month_period"))
    return f"{source}|{signal_id}|{symbol}|{month}"


def simulate_continuous_anchor(
    symbol_daily: pd.DataFrame,
    event: Mapping[str, Any] | pd.Series,
    market_calendar: pd.DatetimeIndex,
    policy: ContinuousPolicy,
    config: ContinuousConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run one causal anchor state machine and return result, path and trades."""

    required_daily = {
        "date",
        "open",
        "high",
        "low",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "base_low",
        "base_position",
        "sessions_since_new_low",
        "return_20d",
        "prior_amount_median_20d",
        "prior_peak",
    }
    _require_columns(symbol_daily, required_daily, "symbol_daily")
    calendar = _normalize_calendar(market_calendar)
    calendar_positions = {
        pd.Timestamp(date): position for position, date in enumerate(calendar)
    }
    prices = symbol_daily.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    numeric_columns = required_daily - {"date"}
    for column in numeric_columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = (
        prices.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    price_by_date = {
        pd.Timestamp(row.date): row
        for row in prices.itertuples(index=False)
    }
    event_key = _event_key(event)
    symbol = str(_event_value(event, "ts_code"))
    signal_date = pd.Timestamp(_event_value(event, "signal_date")).normalize()
    entry_date = pd.Timestamp(_event_value(event, "entry_date")).normalize()
    entry_open = float(_event_value(event, "entry_open"))
    horizon_date = pd.Timestamp(_event_value(event, "target_date")).normalize()
    baseline_exit_date = pd.Timestamp(
        _event_value(event, "baseline_exit_date")
    ).normalize()
    baseline_exit_reason = str(_event_value(event, "baseline_exit_reason"))
    start_position = calendar_positions.get(entry_date)
    horizon_position = calendar_positions.get(horizon_date)
    resolution_position = calendar_positions.get(baseline_exit_date)
    base_payload = {
        "event_key": event_key,
        "policy": policy.name,
        "signal_id": _event_value(event, "signal_id"),
        "anchor_id": _event_value(event, "anchor_id"),
        "ts_code": symbol,
        "signal_date": signal_date,
        "month_period": _event_value(event, "month_period"),
        "source_sample": _event_value(event, "source_sample"),
        "entry_date": entry_date,
        "median_daily_amount": _event_value(event, "median_daily_amount"),
        "baseline_net_return": _event_value(event, "net_return"),
        "baseline_exit_reason": baseline_exit_reason,
    }
    invalid = (
        _event_value(event, "entry_status") != "accepted"
        or not bool(_event_value(event, "baseline_outcome_completed"))
        or start_position is None
        or horizon_position is None
        or resolution_position is None
        or not np.isfinite(entry_open)
        or entry_open <= 0.0
    )
    signal_rows = prices.loc[prices["date"].le(signal_date)]
    signal_row = signal_rows.iloc[-1] if not signal_rows.empty else None
    if invalid or signal_row is None:
        return (
            {
                **base_payload,
                "outcome_completed": False,
                "exit_date": pd.NaT,
                "exit_reason": "invalid_or_incomplete_anchor",
                "budget_return": np.nan,
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )
    initial_stop_level = float(signal_row["base_low"])
    initial_prior_peak = float(signal_row["prior_peak"])
    if policy.structural_stop and (
        not np.isfinite(initial_stop_level) or initial_stop_level <= 0.0
    ):
        return (
            {
                **base_payload,
                "outcome_completed": False,
                "exit_date": pd.NaT,
                "exit_reason": "missing_structural_stop_level",
                "budget_return": np.nan,
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    cash = 1.0
    tranches: list[dict[str, float | str]] = []
    state = "waiting_initial"
    cycle_number = 0
    reentries = 0
    stop_count = 0
    grid_add_count = 0
    target_hit = False
    stop_signal_date = pd.NaT
    last_stop_exit_position: int | None = None
    pending_reentry: dict[str, Any] | None = None
    stop_level = initial_stop_level
    cycle_reference = np.nan
    cycle_budget = np.nan
    final_date = pd.NaT
    final_reason = ""
    last_price = entry_open
    path_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    cost = config.round_trip_cost_bps / 10_000.0

    def stock_value(price: float) -> float:
        return float(
            sum(float(item["shares"]) * price for item in tranches)
        )

    def nav(price: float) -> float:
        return float(cash + stock_value(price))

    def weighted_cost() -> float:
        shares = sum(float(item["shares"]) for item in tranches)
        invested = sum(float(item["allocated_cash"]) for item in tranches)
        return float(invested / shares) if shares > 0.0 else np.nan

    def buy_tranche(
        *,
        date: pd.Timestamp,
        price: float,
        weight: float,
        label: str,
    ) -> bool:
        nonlocal cash
        allocation = min(float(cycle_budget) * weight, cash)
        if not np.isfinite(price) or price <= 0.0 or allocation <= 1e-12:
            return False
        shares = allocation / price
        cash -= allocation
        tranches.append(
            {
                "label": label,
                "allocated_cash": allocation,
                "entry_price": price,
                "shares": shares,
            }
        )
        trade_rows.append(
            {
                **base_payload,
                "date": date,
                "action": "buy",
                "trade_label": label,
                "cycle_number": cycle_number,
                "price": price,
                "allocated_cash": allocation,
                "shares": shares,
                "cash_after": cash,
            }
        )
        return True

    def start_cycle(
        *,
        date: pd.Timestamp,
        price: float,
        new_stop_level: float,
        is_reentry: bool,
    ) -> None:
        nonlocal cycle_number, cycle_budget, cycle_reference, stop_level, state
        nonlocal reentries
        cycle_number += 1
        if is_reentry:
            reentries += 1
        cycle_budget = cash
        cycle_reference = price
        stop_level = new_stop_level
        bought = buy_tranche(
            date=date,
            price=price,
            weight=policy.initial_weight,
            label="reentry_initial" if is_reentry else "initial",
        )
        state = "holding" if bought else "waiting_reentry"

    def liquidate(
        *,
        date: pd.Timestamp,
        price: float,
        reason: str,
    ) -> None:
        nonlocal cash, tranches
        allocated = sum(float(item["allocated_cash"]) for item in tranches)
        shares = sum(float(item["shares"]) for item in tranches)
        gross_proceeds = shares * price
        proceeds = max(gross_proceeds - allocated * cost, 0.0)
        cash += proceeds
        trade_rows.append(
            {
                **base_payload,
                "date": date,
                "action": "sell",
                "trade_label": reason,
                "cycle_number": cycle_number,
                "price": price,
                "allocated_cash": allocated,
                "shares": -shares,
                "cash_after": cash,
            }
        )
        tranches = []

    def complete(*, date: pd.Timestamp, reason: str) -> None:
        nonlocal state, final_date, final_reason
        state = "completed"
        final_date = date
        final_reason = reason

    start_cycle(
        date=entry_date,
        price=entry_open,
        new_stop_level=initial_stop_level,
        is_reentry=False,
    )

    simulation_end = max(horizon_position, resolution_position)
    for position in range(start_position, simulation_end + 1):
        date = pd.Timestamp(calendar[position])
        row_tuple = price_by_date.get(date)
        row = pd.Series(row_tuple._asdict()) if row_tuple is not None else None
        trading_allowed = position <= horizon_position
        added_today = False

        if state == "stop_pending" and row is not None and date > stop_signal_date:
            if not _one_price_bar(row):
                last_price = float(row["adjusted_open"])
                liquidate(date=date, price=last_price, reason="structural_stop")
                stop_count += 1
                last_stop_exit_position = position
                if reentries < policy.maximum_reentries and trading_allowed:
                    state = "waiting_reentry"
                else:
                    complete(date=date, reason="structural_stop_final")

        if state == "reentry_pending" and pending_reentry is not None and row is not None:
            if date > pending_reentry["signal_date"]:
                delay = position - int(pending_reentry["signal_position"])
                if delay > config.maximum_reentry_entry_delay_sessions:
                    pending_reentry = None
                    state = "waiting_reentry"
                elif not _one_price_bar(row):
                    last_price = float(row["adjusted_open"])
                    start_cycle(
                        date=date,
                        price=last_price,
                        new_stop_level=float(pending_reentry["base_low"]),
                        is_reentry=True,
                    )
                    pending_reentry = None

        if state == "waiting_reentry" and row is not None and trading_allowed:
            waited = (
                position - last_stop_exit_position
                if last_stop_exit_position is not None
                else -1
            )
            current_close = float(row["adjusted_close"])
            causal_reclaim = (
                waited >= config.minimum_reentry_wait_sessions
                and float(row["sessions_since_new_low"])
                >= config.minimum_sessions_since_new_low
                and float(row["return_20d"]) > 0.0
                and float(row["base_position"])
                >= config.minimum_reentry_base_position
                and float(row["prior_amount_median_20d"])
                >= config.minimum_reentry_amount_thousand
                and np.isfinite(initial_prior_peak)
                and initial_prior_peak > 0.0
                and current_close / initial_prior_peak - 1.0
                <= config.maximum_reentry_drawdown_from_prior_peak
                and np.isfinite(float(row["base_low"]))
                and float(row["base_low"]) > 0.0
            )
            if causal_reclaim:
                pending_reentry = {
                    "signal_date": date,
                    "signal_position": position,
                    "base_low": float(row["base_low"]),
                }
                state = "reentry_pending"

        if state == "holding" and row is not None and trading_allowed:
            last_price = float(row["adjusted_close"])
            if not _one_price_bar(row):
                labels = {str(item["label"]) for item in tranches}
                for drawdown, weight in zip(
                    policy.grid_drawdowns, policy.grid_weights
                ):
                    label = f"grid_down_{int(round(drawdown * 100))}"
                    level = float(cycle_reference) * (1.0 - drawdown)
                    if label in labels or float(row["adjusted_low"]) > level:
                        continue
                    if buy_tranche(
                        date=date,
                        price=level,
                        weight=weight,
                        label=label,
                    ):
                        labels.add(label)
                        grid_add_count += 1
                        added_today = True
            target = weighted_cost() * (1.0 + config.target_return)
            if (
                tranches
                and not added_today
                and np.isfinite(target)
                and float(row["adjusted_high"]) >= target
            ):
                liquidate(date=date, price=target, reason="take_profit")
                target_hit = True
                complete(date=date, reason="take_profit")
            elif (
                policy.structural_stop
                and tranches
                and np.isfinite(stop_level)
                and float(row["adjusted_close"]) < stop_level
            ):
                state = "stop_pending"
                stop_signal_date = date

        if state != "completed" and position >= horizon_position:
            if not tranches:
                complete(date=date, reason="flat_at_horizon")
            elif date >= baseline_exit_date:
                if baseline_exit_reason == "missing_bar_writeoff":
                    tranches = []
                    complete(date=date, reason="missing_bar_writeoff")
                elif row is not None:
                    terminal_price = float(
                        row["adjusted_open"]
                        if baseline_exit_reason
                        == "next_open_after_target_suspension"
                        else row["adjusted_close"]
                    )
                    last_price = terminal_price
                    liquidate(
                        date=date,
                        price=terminal_price,
                        reason=baseline_exit_reason,
                    )
                    complete(date=date, reason=baseline_exit_reason)

        current_nav = nav(last_price)
        current_stock_value = stock_value(last_price)
        path_rows.append(
            {
                **base_payload,
                "date": date,
                "anchor_nav": current_nav,
                "inner_cash": cash,
                "stock_value": current_stock_value,
                "invested_fraction": (
                    current_stock_value / current_nav if current_nav > 0.0 else 0.0
                ),
                "state": state,
                "cycle_number": cycle_number,
                "reentries": reentries,
                "stop_count": stop_count,
                "grid_add_count": grid_add_count,
            }
        )
        if state == "completed":
            break

    path = pd.DataFrame(path_rows)
    trades = pd.DataFrame(trade_rows)
    final_nav = float(path.iloc[-1]["anchor_nav"]) if not path.empty else np.nan
    drawdown = (
        path["anchor_nav"] / path["anchor_nav"].cummax() - 1.0
        if not path.empty
        else pd.Series(dtype=float)
    )
    result = {
        **base_payload,
        "outcome_completed": bool(state == "completed"),
        "exit_date": final_date,
        "exit_reason": final_reason,
        "budget_return": final_nav - 1.0 if np.isfinite(final_nav) else np.nan,
        "target_hit": target_hit,
        "stop_count": stop_count,
        "reentries": reentries,
        "cycles": cycle_number,
        "grid_add_count": grid_add_count,
        "trade_count": int(len(trades)),
        "mean_invested_fraction": (
            float(path["invested_fraction"].mean()) if not path.empty else np.nan
        ),
        "maximum_invested_fraction": (
            float(path["invested_fraction"].max()) if not path.empty else np.nan
        ),
        "maximum_anchor_drawdown": (
            float(drawdown.min()) if len(drawdown) else np.nan
        ),
    }
    return result, path, trades


def evaluate_continuous_policies(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: ContinuousConfig,
    policies: tuple[ContinuousPolicy, ...] = CONTINUOUS_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate all frozen policies on a common set of completed anchors."""

    _require_columns(daily, {"ts_code", "date"}, "daily")
    _require_columns(
        events,
        {
            "signal_id",
            "ts_code",
            "signal_date",
            "month_period",
            "entry_status",
            "entry_date",
            "entry_open",
            "target_date",
            "baseline_exit_date",
            "baseline_exit_reason",
            "baseline_outcome_completed",
            "net_return",
            "source_sample",
        },
        "events",
    )
    panel = daily.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    by_symbol = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in panel.groupby("ts_code", observed=True, sort=False)
    }
    result_rows: list[dict[str, Any]] = []
    path_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for _, event in events.iterrows():
        symbol_prices = by_symbol.get(str(event["ts_code"]), pd.DataFrame())
        for policy in policies:
            result, path, trades = simulate_continuous_anchor(
                symbol_prices, event, market_calendar, policy, config
            )
            result_rows.append(result)
            if not path.empty:
                path_parts.append(path)
            if not trades.empty:
                trade_parts.append(trades)
    return (
        pd.DataFrame(result_rows),
        pd.concat(path_parts, ignore_index=True, sort=False)
        if path_parts
        else pd.DataFrame(),
        pd.concat(trade_parts, ignore_index=True, sort=False)
        if trade_parts
        else pd.DataFrame(),
    )


def simulate_continuous_portfolio(
    anchor_paths: pd.DataFrame,
    anchor_results: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    *,
    policy: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float = 1_000_000.0,
    anchor_fraction: float = 0.025,
    maximum_anchors: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Mark a capacity-limited portfolio using each anchor's internal daily NAV."""

    if initial_cash <= 0.0:
        raise ValueError("initial_cash must be positive")
    if not 0.0 < anchor_fraction <= 1.0:
        raise ValueError("anchor_fraction must be in (0, 1]")
    if maximum_anchors < 1:
        raise ValueError("maximum_anchors must be positive")
    required_paths = {
        "event_key",
        "policy",
        "date",
        "anchor_nav",
        "stock_value",
    }
    required_results = {
        "event_key",
        "policy",
        "ts_code",
        "month_period",
        "entry_date",
        "exit_date",
        "median_daily_amount",
    }
    _require_columns(anchor_paths, required_paths, "anchor_paths")
    _require_columns(anchor_results, required_results, "anchor_results")
    calendar = _normalize_calendar(market_calendar)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    dates = calendar[(calendar >= start) & (calendar <= end)]
    if dates.empty:
        raise ValueError("portfolio period has no market sessions")
    results = anchor_results.loc[anchor_results["policy"].eq(policy)].copy()
    results["entry_date"] = pd.to_datetime(
        results["entry_date"], errors="coerce"
    ).dt.normalize()
    results["exit_date"] = pd.to_datetime(
        results["exit_date"], errors="coerce"
    ).dt.normalize()
    results["anchor_month"] = pd.PeriodIndex(
        results["month_period"], freq="M"
    ).to_timestamp(how="end").normalize()
    results = results.loc[
        results["anchor_month"].between(start, end, inclusive="both")
        & results["entry_date"].between(start, end, inclusive="both")
    ]
    paths = anchor_paths.loc[anchor_paths["policy"].eq(policy)].copy()
    paths["date"] = pd.to_datetime(paths["date"], errors="coerce").dt.normalize()
    path_lookup = paths.set_index(["event_key", "date"])[
        ["anchor_nav", "stock_value"]
    ]
    entries: dict[pd.Timestamp, list[Any]] = {}
    for row in results.itertuples(index=False):
        entries.setdefault(pd.Timestamp(row.entry_date), []).append(row)
    exits: dict[pd.Timestamp, list[str]] = {}
    for row in results.itertuples(index=False):
        if pd.notna(row.exit_date):
            exits.setdefault(pd.Timestamp(row.exit_date), []).append(str(row.event_key))

    cash = float(initial_cash)
    active: dict[str, dict[str, Any]] = {}
    active_symbols: set[str] = set()
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    audit = {
        "entered_anchors": 0,
        "exited_anchors": 0,
        "skipped_capacity": 0,
        "skipped_duplicate_symbol": 0,
        "skipped_cash": 0,
    }

    def marked_values(date: pd.Timestamp) -> tuple[float, float]:
        reserved = 0.0
        invested = 0.0
        for key, holding in active.items():
            lookup_key = (key, date)
            if lookup_key in path_lookup.index:
                state = path_lookup.loc[lookup_key]
                holding["last_anchor_nav"] = float(state["anchor_nav"])
                holding["last_stock_value"] = float(state["stock_value"])
            reserved += holding["allocation"] * holding["last_anchor_nav"]
            invested += holding["allocation"] * holding["last_stock_value"]
        return reserved, invested

    for date_value in dates:
        date = pd.Timestamp(date_value)
        marked_values(date)
        for key in exits.get(date, []):
            holding = active.pop(key, None)
            if holding is None:
                continue
            proceeds = holding["allocation"] * holding["last_anchor_nav"]
            cash += proceeds
            active_symbols.discard(holding["ts_code"])
            trade_rows.append(
                {
                    "date": date,
                    "action": "exit_anchor",
                    "event_key": key,
                    "policy": policy,
                    "ts_code": holding["ts_code"],
                    "cash_flow": proceeds,
                }
            )
            audit["exited_anchors"] += 1
        reserved, _ = marked_values(date)
        nav_before_entries = cash + reserved
        todays = sorted(
            entries.get(date, []),
            key=lambda row: (
                -float(row.median_daily_amount)
                if pd.notna(row.median_daily_amount)
                else np.inf,
                str(row.ts_code),
            ),
        )
        for row in todays:
            key = str(row.event_key)
            symbol = str(row.ts_code)
            if len(active) >= maximum_anchors:
                audit["skipped_capacity"] += 1
                continue
            if symbol in active_symbols:
                audit["skipped_duplicate_symbol"] += 1
                continue
            allocation = nav_before_entries * anchor_fraction
            if allocation > cash + 1e-8:
                audit["skipped_cash"] += 1
                continue
            first_key = (key, date)
            if first_key not in path_lookup.index:
                continue
            state = path_lookup.loc[first_key]
            cash -= allocation
            active[key] = {
                "allocation": allocation,
                "ts_code": symbol,
                "last_anchor_nav": float(state["anchor_nav"]),
                "last_stock_value": float(state["stock_value"]),
            }
            active_symbols.add(symbol)
            trade_rows.append(
                {
                    "date": date,
                    "action": "enter_anchor",
                    "event_key": key,
                    "policy": policy,
                    "ts_code": symbol,
                    "cash_flow": -allocation,
                }
            )
            audit["entered_anchors"] += 1
        reserved, invested = marked_values(date)
        total_nav = cash + reserved
        curve_rows.append(
            {
                "date": date,
                "policy": policy,
                "nav": total_nav,
                "cash": cash,
                "active_anchors": len(active),
                "reserved_fraction": reserved / total_nav if total_nav > 0.0 else 0.0,
                "invested_fraction": invested / total_nav if total_nav > 0.0 else 0.0,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(trade_rows), audit
