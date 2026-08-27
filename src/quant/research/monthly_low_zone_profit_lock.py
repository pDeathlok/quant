"""Causal profit-lock and staged-budget research for monthly low-zone anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd


PROFIT_LOCK_PERIODS = {
    "development_2013_2016": (
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2016-12-31"),
    ),
    "exposed_validation_2017_2020": (
        pd.Timestamp("2017-01-01"),
        pd.Timestamp("2020-12-31"),
    ),
    "seen_diagnostic_2021_2024": (
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2024-12-31"),
    ),
    "time_out_diagnostic_2025": (
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-12-31"),
    ),
}


@dataclass(frozen=True)
class ProfitLockConfig:
    """Frozen event-level exit assumptions for the path-management iteration."""

    horizon_sessions: int = 504
    target_returns: tuple[float, ...] = (0.10, 0.15, 0.20)
    round_trip_cost_bps: float = 20.0

    def __post_init__(self) -> None:
        if self.horizon_sessions < 1:
            raise ValueError("horizon_sessions must be positive")
        if not self.target_returns:
            raise ValueError("target_returns must not be empty")
        if any(not 0.0 < target < 1.0 for target in self.target_returns):
            raise ValueError("each target return must be in (0, 1)")
        if tuple(sorted(set(self.target_returns))) != self.target_returns:
            raise ValueError("target_returns must be unique and increasing")
        if self.round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProfitLockPortfolioConfig:
    """Frozen cash, capacity and partial-build assumptions."""

    initial_cash: float = 1_000_000.0
    maximum_anchors: int = 20
    target_anchor_fraction: float = 0.10
    probe_budget_fraction: float = 0.25
    add_budget_fraction: float = 0.25
    profit_lock_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_cash <= 0.0:
            raise ValueError("initial_cash must be positive")
        if self.maximum_anchors < 1:
            raise ValueError("maximum_anchors must be positive")
        if not 0.0 < self.target_anchor_fraction <= 1.0:
            raise ValueError("target_anchor_fraction must be in (0, 1]")
        if not 0.0 < self.probe_budget_fraction <= 1.0:
            raise ValueError("probe_budget_fraction must be in (0, 1]")
        if not 0.0 < self.add_budget_fraction <= 1.0:
            raise ValueError("add_budget_fraction must be in (0, 1]")
        if self.probe_budget_fraction + self.add_budget_fraction > 1.0:
            raise ValueError("probe and add budget fractions must sum to at most one")
        if not 0.0 < self.profit_lock_fraction <= 1.0:
            raise ValueError("profit_lock_fraction must be in (0, 1]")

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


def _normalize_benchmark(benchmark: pd.DataFrame | None) -> pd.DataFrame:
    if benchmark is None or benchmark.empty:
        return pd.DataFrame(columns=["date", "open", "close"]).set_index("date")
    _require_columns(benchmark, {"open", "close"}, "benchmark")
    out = benchmark.copy()
    if "trade_date" in out:
        source = out["trade_date"]
    elif "date" in out:
        source = out["date"]
    else:
        raise ValueError("benchmark missing columns: ['trade_date or date']")
    parsed = pd.to_datetime(source, errors="coerce")
    compact = pd.to_datetime(
        source.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    out["date"] = compact.fillna(parsed).dt.normalize()
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return (
        out.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")[["open", "close"]]
    )


def evaluate_profit_lock_events(
    daily: pd.DataFrame,
    baseline_events: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: ProfitLockConfig,
    benchmark: pd.DataFrame | None = None,
    *,
    resolve_observed_targets: bool = False,
) -> pd.DataFrame:
    """Apply first-touch limit exits without reading beyond each baseline event."""

    _require_columns(daily, {"ts_code", "date", "adjusted_high"}, "daily")
    required_events = {
        "signal_id",
        "ts_code",
        "rule",
        "signal_date",
        "horizon",
        "entry_status",
        "entry_date",
        "entry_open",
        "exit_date",
        "exit_reason",
        "outcome_completed",
        "net_return",
    }
    _require_columns(baseline_events, required_events, "baseline_events")
    calendar = _normalize_calendar(market_calendar)
    calendar_positions = {
        pd.Timestamp(date): position for position, date in enumerate(calendar)
    }
    market = _normalize_benchmark(benchmark)
    panel = daily.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["adjusted_high"] = pd.to_numeric(panel["adjusted_high"], errors="coerce")
    for optional in ("adjusted_low", "adjusted_close"):
        if optional in panel:
            panel[optional] = pd.to_numeric(panel[optional], errors="coerce")
    panel = (
        panel.dropna(subset=["ts_code", "date"])
        .sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
    )
    daily_by_symbol = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in panel.groupby("ts_code", observed=True, sort=False)
    }
    source = baseline_events.loc[
        pd.to_numeric(baseline_events["horizon"], errors="coerce").eq(
            config.horizon_sessions
        )
    ].copy()
    source["signal_date"] = pd.to_datetime(
        source["signal_date"], errors="coerce"
    ).dt.normalize()
    source["entry_date"] = pd.to_datetime(
        source["entry_date"], errors="coerce"
    ).dt.normalize()
    source["exit_date"] = pd.to_datetime(
        source["exit_date"], errors="coerce"
    ).dt.normalize()
    if "target_date" in source:
        source["target_date"] = pd.to_datetime(
            source["target_date"], errors="coerce"
        ).dt.normalize()

    rows: list[dict[str, Any]] = []
    cost = config.round_trip_cost_bps / 10_000.0
    for _, event in source.iterrows():
        base_payload = event.to_dict()
        base_payload.update(
            {
                "baseline_exit_date": event.get("exit_date"),
                "baseline_exit_reason": event.get("exit_reason"),
                "baseline_outcome_completed": bool(
                    event.get("outcome_completed", False)
                ),
                "baseline_gross_return": event.get("gross_return", np.nan),
                "baseline_net_return": event.get("net_return", np.nan),
                "baseline_mae": event.get("mae", np.nan),
                "baseline_mfe": event.get("mfe", np.nan),
            }
        )
        accepted = event.get("entry_status") == "accepted"
        completed = bool(event.get("outcome_completed", False))
        entry_date = event.get("entry_date")
        entry_open = pd.to_numeric(event.get("entry_open"), errors="coerce")
        path_end = event.get("exit_date")
        if pd.isna(path_end):
            path_end = event.get("target_date", pd.NaT)
        if pd.isna(path_end) and resolve_observed_targets:
            path_end = calendar[-1]
        prices = daily_by_symbol.get(str(event["ts_code"]))
        path = pd.DataFrame()
        if (
            accepted
            and (completed or resolve_observed_targets)
            and prices is not None
            and pd.notna(entry_date)
            and pd.notna(path_end)
        ):
            path = prices.loc[
                prices["date"].between(entry_date, path_end, inclusive="both")
            ].copy()
        for target_return in config.target_returns:
            record = dict(base_payload)
            record["target_return"] = target_return
            record["target_hit"] = False
            record["holding_sessions"] = np.nan
            if not accepted or (not completed and not resolve_observed_targets):
                record.update(
                    {
                        "outcome_completed": False,
                        "gross_return": np.nan,
                        "net_return": np.nan,
                        "benchmark_return": np.nan,
                        "excess_net_return": np.nan,
                        "mae": np.nan,
                        "mfe": np.nan,
                    }
                )
                rows.append(record)
                continue
            threshold = float(entry_open) * (1.0 + target_return)
            hits = path.loc[path["adjusted_high"].ge(threshold)]
            if hits.empty:
                if not completed:
                    record.update(
                        {
                            "outcome_completed": False,
                            "gross_return": np.nan,
                            "net_return": np.nan,
                            "benchmark_return": np.nan,
                            "excess_net_return": np.nan,
                            "mae": np.nan,
                            "mfe": np.nan,
                        }
                    )
                    rows.append(record)
                    continue
                record["holding_sessions"] = _holding_sessions(
                    entry_date,
                    event.get("exit_date"),
                    calendar_positions,
                )
                rows.append(record)
                continue
            hit = hits.iloc[0]
            exit_date = pd.Timestamp(hit["date"])
            benchmark_return = _benchmark_return(
                market,
                pd.Timestamp(entry_date),
                exit_date,
            )
            path_to_exit = path.loc[path["date"].le(exit_date)]
            mae = (
                float(path_to_exit["adjusted_low"].min() / float(entry_open) - 1.0)
                if "adjusted_low" in path_to_exit and not path_to_exit.empty
                else np.nan
            )
            record.update(
                {
                    "exit_date": exit_date,
                    "exit_reason": "take_profit",
                    "outcome_completed": True,
                    "gross_return": target_return,
                    "net_return": target_return - cost,
                    "benchmark_return": benchmark_return,
                    "excess_net_return": (
                        target_return - cost - benchmark_return
                        if np.isfinite(benchmark_return)
                        else np.nan
                    ),
                    "mae": mae,
                    "mfe": target_return,
                    "target_hit": True,
                    "holding_sessions": _holding_sessions(
                        pd.Timestamp(entry_date), exit_date, calendar_positions
                    ),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def _holding_sessions(
    entry_date: object,
    exit_date: object,
    calendar_positions: dict[pd.Timestamp, int],
) -> float:
    if pd.isna(entry_date) or pd.isna(exit_date):
        return np.nan
    start = calendar_positions.get(pd.Timestamp(entry_date))
    end = calendar_positions.get(pd.Timestamp(exit_date))
    if start is None or end is None or end < start:
        return np.nan
    return float(end - start + 1)


def _benchmark_return(
    benchmark: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> float:
    if entry_date not in benchmark.index or exit_date not in benchmark.index:
        return np.nan
    entry_open = float(benchmark.at[entry_date, "open"])
    exit_close = float(benchmark.at[exit_date, "close"])
    if not np.isfinite(entry_open) or not np.isfinite(exit_close) or entry_open <= 0.0:
        return np.nan
    return exit_close / entry_open - 1.0


def _date_cluster_summary(frame: pd.DataFrame) -> tuple[float, float, float]:
    cohorts = frame.groupby("signal_date", observed=True)["net_return"].mean().dropna()
    if cohorts.empty:
        return np.nan, np.nan, np.nan
    mean = float(cohorts.mean())
    if len(cohorts) < 2:
        return mean, np.nan, np.nan
    standard_error = float(cohorts.std(ddof=1) / sqrt(len(cohorts)))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def summarize_profit_lock_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize completed target events by honest data-exposure period."""

    required = {
        "rule",
        "signal_date",
        "horizon",
        "target_return",
        "entry_status",
        "outcome_completed",
        "net_return",
        "target_hit",
        "holding_sessions",
    }
    _require_columns(events, required, "events")
    frame = events.copy()
    frame["signal_date"] = pd.to_datetime(
        frame["signal_date"], errors="coerce"
    ).dt.normalize()
    rows: list[dict[str, Any]] = []
    group_columns = ["rule", "horizon", "target_return"]
    for period, (start, end) in PROFIT_LOCK_PERIODS.items():
        scoped = frame.loc[frame["signal_date"].between(start, end, inclusive="both")]
        for keys, group in scoped.groupby(group_columns, observed=True, sort=True):
            accepted = group.loc[group["entry_status"].eq("accepted")]
            eligible = accepted
            if "baseline_outcome_completed" in accepted:
                eligible = accepted.loc[
                    accepted["baseline_outcome_completed"].fillna(False)
                ]
            completed = eligible.loc[eligible["outcome_completed"].fillna(False)]
            returns = pd.to_numeric(completed["net_return"], errors="coerce").dropna()
            gains = float(returns.loc[returns > 0.0].sum())
            losses = float(-returns.loc[returns <= 0.0].sum())
            cluster_mean, cluster_low, cluster_high = _date_cluster_summary(completed)
            rows.append(
                {
                    "period": period,
                    "rule": keys[0],
                    "horizon": int(keys[1]),
                    "target_return": float(keys[2]),
                    "events": int(len(group)),
                    "accepted_entries": int(len(accepted)),
                    "completed_events": int(len(completed)),
                    "signal_dates": int(completed["signal_date"].nunique()),
                    "win_rate": float(returns.gt(0.0).mean())
                    if len(returns)
                    else np.nan,
                    "median_net_return": float(returns.median())
                    if len(returns)
                    else np.nan,
                    "mean_net_return": float(returns.mean())
                    if len(returns)
                    else np.nan,
                    "profit_factor": gains / losses if losses > 0.0 else np.nan,
                    "target_hit_rate": float(completed["target_hit"].mean())
                    if len(completed)
                    else np.nan,
                    "median_holding_sessions": float(
                        pd.to_numeric(
                            completed["holding_sessions"], errors="coerce"
                        ).median()
                    )
                    if len(completed)
                    else np.nan,
                    "tail_loss_rate": float(returns.le(-0.50).mean())
                    if len(returns)
                    else np.nan,
                    "date_equal_mean_net_return": cluster_mean,
                    "date_cluster_ci95_low": cluster_low,
                    "date_cluster_ci95_high": cluster_high,
                }
            )
    return pd.DataFrame(rows)


def assemble_staged_anchor_events(
    profit_events: pd.DataFrame,
    *,
    add_rule: str,
    probe_weight: float = 0.25,
    add_weight: float = 0.25,
) -> pd.DataFrame:
    """Combine a direct probe and one causal confirmation into anchor budgets."""

    required = {
        "anchor_id",
        "ts_code",
        "rule",
        "signal_date",
        "target_return",
        "horizon",
        "entry_status",
        "outcome_completed",
        "entry_date",
        "exit_date",
        "exit_reason",
        "net_return",
        "holding_sessions",
    }
    _require_columns(profit_events, required, "profit_events")
    if not 0.0 < probe_weight <= 1.0:
        raise ValueError("probe_weight must be in (0, 1]")
    if not 0.0 < add_weight <= 1.0 or probe_weight + add_weight > 1.0:
        raise ValueError("add weights must be positive and sum to at most one")
    frame = profit_events.copy()
    for column in ("signal_date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    rows: list[dict[str, Any]] = []
    group_columns = ["anchor_id", "target_return", "horizon"]
    for keys, group in frame.groupby(group_columns, observed=True, sort=True):
        probe_rows = group.loc[group["rule"].eq("anchor_direct")]
        if probe_rows.empty:
            continue
        probe = probe_rows.sort_values("signal_date").iloc[0]
        add_rows = group.loc[group["rule"].eq(add_rule)].sort_values("signal_date")
        add = add_rows.iloc[0] if not add_rows.empty else None
        probe_complete = bool(probe["outcome_completed"]) and (
            probe["entry_status"] == "accepted"
        )
        add_eligible = False
        if add is None:
            add_status = "confirmation_missing"
        elif not bool(add["outcome_completed"]) or add["entry_status"] != "accepted":
            add_status = "confirmation_unresolved_or_rejected"
        elif pd.isna(probe["exit_date"]) or pd.isna(add["entry_date"]):
            add_status = "confirmation_timing_unresolved"
        elif pd.Timestamp(add["entry_date"]) >= pd.Timestamp(probe["exit_date"]):
            add_status = "confirmation_after_probe_exit"
        else:
            add_status = "added_before_probe_exit"
            add_eligible = True
        probe_return = float(probe["net_return"]) if probe_complete else np.nan
        add_return = float(add["net_return"]) if add_eligible and add is not None else 0.0
        build_fraction = probe_weight + (add_weight if add_eligible else 0.0)
        budget_return = (
            probe_weight * probe_return + add_weight * add_return
            if np.isfinite(probe_return)
            else np.nan
        )
        committed_return = (
            budget_return / build_fraction
            if np.isfinite(budget_return) and build_fraction > 0.0
            else np.nan
        )
        exit_dates = [probe["exit_date"]]
        if add_eligible and add is not None:
            exit_dates.append(add["exit_date"])
        valid_exit_dates = [pd.Timestamp(value) for value in exit_dates if pd.notna(value)]
        rows.append(
            {
                "anchor_id": int(keys[0]),
                "ts_code": probe["ts_code"],
                "policy": f"probe_{int(probe_weight * 100)}_{add_rule}_{int(add_weight * 100)}",
                "add_rule": add_rule,
                "signal_date": probe["signal_date"],
                "target_return": float(keys[1]),
                "horizon": int(keys[2]),
                "outcome_completed": probe_complete,
                "probe_exit_date": probe["exit_date"],
                "probe_exit_reason": probe["exit_reason"],
                "probe_net_return": probe_return,
                "add_signal_date": add["signal_date"] if add is not None else pd.NaT,
                "add_entry_date": add["entry_date"] if add is not None else pd.NaT,
                "add_exit_date": add["exit_date"] if add_eligible and add is not None else pd.NaT,
                "add_net_return": add_return if add_eligible else np.nan,
                "add_status": add_status,
                "build_fraction": build_fraction,
                "budget_return": budget_return,
                "committed_return": committed_return,
                "exit_date": max(valid_exit_dates) if valid_exit_dates else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def simulate_profit_lock_portfolio(
    daily: pd.DataFrame,
    profit_events: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    target_return: float,
    add_rule: str | None,
    config: ProfitLockPortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run a daily marked-to-market portfolio with causal capacity ordering."""

    _require_columns(daily, {"ts_code", "date", "adjusted_close"}, "daily")
    required_events = {
        "signal_id",
        "anchor_id",
        "ts_code",
        "rule",
        "signal_date",
        "median_daily_amount",
        "target_return",
        "horizon",
        "entry_status",
        "outcome_completed",
        "entry_date",
        "entry_open",
        "exit_date",
        "exit_reason",
        "net_return",
    }
    _require_columns(profit_events, required_events, "profit_events")
    calendar = _normalize_calendar(market_calendar)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    simulation_dates = calendar[(calendar >= start) & (calendar <= end)]
    if simulation_dates.empty:
        raise ValueError("portfolio period has no market sessions")
    events = profit_events.loc[
        np.isclose(
            pd.to_numeric(profit_events["target_return"], errors="coerce"),
            target_return,
        )
        & profit_events["rule"].isin(
            ["anchor_direct"] if add_rule is None else ["anchor_direct", add_rule]
        )
    ].copy()
    if events.empty:
        raise ValueError("profit_events contain no requested target/rules")
    horizons = pd.to_numeric(events["horizon"], errors="coerce").dropna().unique()
    if len(horizons) != 1:
        raise ValueError("portfolio requires exactly one event horizon")
    for column in ("signal_date", "entry_date", "exit_date"):
        events[column] = pd.to_datetime(events[column], errors="coerce").dt.normalize()
    direct_anchor_ids = set(
        events.loc[
            events["rule"].eq("anchor_direct")
            & events["signal_date"].between(start, end, inclusive="both"),
            "anchor_id",
        ]
    )
    events = events.loc[events["anchor_id"].isin(direct_anchor_ids)].copy()
    events = events.loc[
        events["entry_status"].eq("accepted")
        & events["entry_date"].between(start, end, inclusive="both")
    ]
    event_by_key = {
        (int(row.anchor_id), str(row.rule)): row
        for row in events.sort_values("signal_date").itertuples(index=False)
    }
    entries_by_date: dict[pd.Timestamp, list[Any]] = {}
    for event in event_by_key.values():
        entries_by_date.setdefault(pd.Timestamp(event.entry_date), []).append(event)

    symbols = {str(event.ts_code) for event in event_by_key.values()}
    prices = daily.loc[daily["ts_code"].astype(str).isin(symbols)].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["adjusted_close"] = pd.to_numeric(
        prices["adjusted_close"], errors="coerce"
    )
    prices = (
        prices.dropna(subset=["ts_code", "date", "adjusted_close"])
        .sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
    )
    close_by_symbol = {
        str(symbol): group.set_index("date")["adjusted_close"]
        for symbol, group in prices.groupby("ts_code", observed=True, sort=False)
    }

    cash = float(config.initial_cash)
    anchors: dict[int, dict[str, Any]] = {}
    active_symbols: set[str] = set()
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {
        "skipped_capacity": 0,
        "skipped_duplicate_symbol": 0,
        "skipped_cash": 0,
        "skipped_inactive_confirmation": 0,
        "entered_anchors": 0,
        "added_tranches": 0,
        "exited_tranches": 0,
    }

    def marked_value(anchor: dict[str, Any]) -> float:
        total = 0.0
        for tranche in anchor["tranches"].values():
            total += tranche["allocated_cash"] * (
                tranche["last_price"] / tranche["entry_open"]
            )
        return total

    def current_nav() -> float:
        return cash + sum(marked_value(anchor) for anchor in anchors.values())

    def exit_due_tranches(date: pd.Timestamp) -> None:
        nonlocal cash
        removals: list[tuple[int, str]] = []
        for anchor_id, anchor in list(anchors.items()):
            for rule, tranche in list(anchor["tranches"].items()):
                event = tranche["event"]
                is_runner = bool(tranche.get("is_runner", False))
                outcome_completed = (
                    bool(tranche["runner_outcome_completed"])
                    if is_runner
                    else bool(event.outcome_completed)
                )
                exit_date = (
                    tranche["runner_exit_date"] if is_runner else event.exit_date
                )
                if not outcome_completed or pd.isna(exit_date):
                    continue
                if pd.Timestamp(exit_date) != date:
                    continue
                if (
                    not is_runner
                    and config.profit_lock_fraction < 1.0
                    and bool(getattr(event, "target_hit", False))
                ):
                    core_cash = (
                        tranche["allocated_cash"] * config.profit_lock_fraction
                    )
                    runner_cash = tranche["allocated_cash"] - core_cash
                    core_return = float(event.net_return)
                    cash += max(core_cash * (1.0 + core_return), 0.0)
                    trade_rows.append(
                        {
                            "date": date,
                            "action": "exit_core",
                            "anchor_id": anchor_id,
                            "ts_code": anchor["ts_code"],
                            "rule": rule,
                            "allocated_cash": core_cash,
                            "net_return": core_return,
                            "exit_reason": event.exit_reason,
                        }
                    )
                    audit["exited_tranches"] += 1
                    tranche.update(
                        {
                            "allocated_cash": runner_cash,
                            "is_runner": True,
                            "runner_outcome_completed": bool(
                                getattr(event, "baseline_outcome_completed", False)
                            ),
                            "runner_exit_date": getattr(
                                event, "baseline_exit_date", pd.NaT
                            ),
                            "runner_net_return": getattr(
                                event, "baseline_net_return", np.nan
                            ),
                            "runner_exit_reason": getattr(
                                event, "baseline_exit_reason", ""
                            ),
                        }
                    )
                    runner_exit_date = tranche["runner_exit_date"]
                    if (
                        tranche["runner_outcome_completed"]
                        and pd.notna(runner_exit_date)
                        and pd.Timestamp(runner_exit_date) <= date
                    ):
                        runner_return = float(tranche["runner_net_return"])
                        cash += max(runner_cash * (1.0 + runner_return), 0.0)
                        trade_rows.append(
                            {
                                "date": date,
                                "action": "exit_runner",
                                "anchor_id": anchor_id,
                                "ts_code": anchor["ts_code"],
                                "rule": rule,
                                "allocated_cash": runner_cash,
                                "net_return": runner_return,
                                "exit_reason": tranche["runner_exit_reason"],
                            }
                        )
                        audit["exited_tranches"] += 1
                        removals.append((anchor_id, rule))
                    continue
                net_return = float(
                    tranche["runner_net_return"] if is_runner else event.net_return
                )
                exit_reason = (
                    tranche["runner_exit_reason"] if is_runner else event.exit_reason
                )
                proceeds = tranche["allocated_cash"] * (1.0 + net_return)
                cash += max(proceeds, 0.0)
                trade_rows.append(
                    {
                        "date": date,
                        "action": "exit_runner" if is_runner else "exit",
                        "anchor_id": anchor_id,
                        "ts_code": anchor["ts_code"],
                        "rule": rule,
                        "allocated_cash": tranche["allocated_cash"],
                        "net_return": net_return,
                        "exit_reason": exit_reason,
                    }
                )
                audit["exited_tranches"] += 1
                removals.append((anchor_id, rule))
        for anchor_id, rule in removals:
            anchor = anchors.get(anchor_id)
            if anchor is None:
                continue
            anchor["tranches"].pop(rule, None)
            if not anchor["tranches"]:
                active_symbols.discard(anchor["ts_code"])
                anchors.pop(anchor_id, None)

    for date_value in simulation_dates:
        date = pd.Timestamp(date_value)
        # Conservative same-day convention: an existing exit frees capacity before
        # a confirmation entry. A confirmation on the probe exit date cannot add.
        exit_due_tranches(date)
        todays = entries_by_date.get(date, [])
        direct = sorted(
            (event for event in todays if event.rule == "anchor_direct"),
            key=lambda event: (
                -float(event.median_daily_amount)
                if pd.notna(event.median_daily_amount)
                else np.inf,
                str(event.ts_code),
            ),
        )
        newly_entered: set[tuple[int, str]] = set()
        for event in direct:
            anchor_id = int(event.anchor_id)
            symbol = str(event.ts_code)
            if anchor_id in anchors or symbol in active_symbols:
                audit["skipped_duplicate_symbol"] += 1
                continue
            if len(anchors) >= config.maximum_anchors:
                audit["skipped_capacity"] += 1
                continue
            nav = current_nav()
            target_budget = nav * config.target_anchor_fraction
            allocation = min(
                target_budget * config.probe_budget_fraction,
                cash,
            )
            if allocation <= 0.0:
                audit["skipped_cash"] += 1
                continue
            entry_open = float(event.entry_open)
            cash -= allocation
            anchors[anchor_id] = {
                "ts_code": symbol,
                "target_budget": target_budget,
                "tranches": {
                    "anchor_direct": {
                        "event": event,
                        "allocated_cash": allocation,
                        "entry_open": entry_open,
                        "last_price": entry_open,
                        "is_runner": False,
                    }
                },
            }
            active_symbols.add(symbol)
            audit["entered_anchors"] += 1
            newly_entered.add((anchor_id, "anchor_direct"))
            trade_rows.append(
                {
                    "date": date,
                    "action": "entry",
                    "anchor_id": anchor_id,
                    "ts_code": symbol,
                    "rule": "anchor_direct",
                    "allocated_cash": allocation,
                    "net_return": np.nan,
                    "exit_reason": pd.NA,
                }
            )
        if add_rule is not None:
            additions = sorted(
                (event for event in todays if event.rule == add_rule),
                key=lambda event: (int(event.anchor_id), str(event.ts_code)),
            )
            for event in additions:
                anchor_id = int(event.anchor_id)
                anchor = anchors.get(anchor_id)
                if anchor is None or "anchor_direct" not in anchor["tranches"]:
                    audit["skipped_inactive_confirmation"] += 1
                    continue
                if add_rule in anchor["tranches"]:
                    continue
                allocation = min(
                    anchor["target_budget"] * config.add_budget_fraction,
                    cash,
                )
                if allocation <= 0.0:
                    audit["skipped_cash"] += 1
                    continue
                entry_open = float(event.entry_open)
                cash -= allocation
                anchor["tranches"][add_rule] = {
                    "event": event,
                    "allocated_cash": allocation,
                    "entry_open": entry_open,
                    "last_price": entry_open,
                    "is_runner": False,
                }
                audit["added_tranches"] += 1
                newly_entered.add((anchor_id, add_rule))
                trade_rows.append(
                    {
                        "date": date,
                        "action": "add",
                        "anchor_id": anchor_id,
                        "ts_code": anchor["ts_code"],
                        "rule": add_rule,
                        "allocated_cash": allocation,
                        "net_return": np.nan,
                        "exit_reason": pd.NA,
                    }
                )
        # A newly opened tranche can touch its target later on its entry day.
        if newly_entered:
            exit_due_tranches(date)
        for anchor in anchors.values():
            series = close_by_symbol.get(anchor["ts_code"])
            if series is None or date not in series.index:
                continue
            close = float(series.at[date])
            if not np.isfinite(close) or close <= 0.0:
                continue
            for tranche in anchor["tranches"].values():
                tranche["last_price"] = close
        invested_value = sum(marked_value(anchor) for anchor in anchors.values())
        nav = cash + invested_value
        curve_rows.append(
            {
                "date": date,
                "nav": nav,
                "cash": cash,
                "invested_value": invested_value,
                "invested_fraction": invested_value / nav if nav > 0.0 else np.nan,
                "active_anchors": len(anchors),
                "active_tranches": sum(
                    len(anchor["tranches"]) for anchor in anchors.values()
                ),
            }
        )
    audit.update(
        {
            "start_date": start,
            "end_date": end,
            "target_return": target_return,
            "add_rule": add_rule,
            "ending_active_anchors": len(anchors),
            "ending_cash": cash,
        }
    )
    return pd.DataFrame(curve_rows), pd.DataFrame(trade_rows), audit
