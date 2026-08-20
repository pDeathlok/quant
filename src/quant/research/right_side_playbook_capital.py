"""Capital-constrained descriptive evaluation for frozen playbook selections.

This module does not fit a model or choose a policy.  It applies one explicit
cash-and-capacity contract to already-frozen B-fold selections.  Daily raw bars
are used only to mark open positions; realized returns and stop/target ordering
remain exactly those in the immutable playbook outcome table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant.research.right_side_playbook_policy import NO_TRADE_PLAYBOOK_ID


CAPITAL_BACKTEST_SCHEMA_VERSION = "right-side-playbook-capital-b-development-v1"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_TARGET_POSITION_CASH = DEFAULT_INITIAL_CAPITAL / 20.0
DEFAULT_MAX_CONCURRENT_POSITIONS = 20
DEFAULT_MAX_NEW_POSITIONS_PER_SESSION = 5
DEFAULT_BOARD_LOT_SIZE = 100
DEFAULT_COST_BPS = 15.0
TRADING_SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class CapitalBacktestSpec:
    """A frozen capital contract; scenario differences are descriptive only."""

    scenario_id: str = "base_max20_cost15bps"
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    target_position_cash: float = DEFAULT_TARGET_POSITION_CASH
    max_concurrent_positions: int = DEFAULT_MAX_CONCURRENT_POSITIONS
    max_new_positions_per_session: int = DEFAULT_MAX_NEW_POSITIONS_PER_SESSION
    board_lot_size: int = DEFAULT_BOARD_LOT_SIZE
    cost_bps: float = DEFAULT_COST_BPS

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive and finite")
        if not np.isfinite(self.target_position_cash) or self.target_position_cash <= 0:
            raise ValueError("target_position_cash must be positive and finite")
        if self.max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")
        if self.max_new_positions_per_session <= 0:
            raise ValueError("max_new_positions_per_session must be positive")
        if self.board_lot_size <= 0:
            raise ValueError("board_lot_size must be positive")
        if not np.isfinite(self.cost_bps) or self.cost_bps < 0:
            raise ValueError("cost_bps must be finite and non-negative")

    def manifest(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidate_priority": "first_layer_score_desc_symbol_asc_event_id_asc",
            "daily_order": "process_exits_before_entries",
            "cash_rule": "no_leverage_and_only_cash_from_completed_exits_is_reusable",
            "duplicate_symbol_rule": "independent_event_lots_allowed",
            "position_cash_rule": "fixed_initial_capital_divided_by_20_in_all_scenarios",
            "rounding_rule": "floor_to_100_share_board_lot_when_raw_entry_price_exists",
            "missing_entry_price_rule": "equal_target_slot_fractional_fallback_explicitly_flagged",
            "realized_return_rule": "gross_return_minus_scenario_cost_bps",
            "intrabar_ambiguity_rule": "reuse_frozen_outcome_sl_first",
        }


@dataclass(frozen=True)
class CapitalSimulationResult:
    curve: pd.DataFrame
    orders: pd.DataFrame
    metrics: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def build_continuous_daily_marks(daily: pd.DataFrame) -> pd.DataFrame:
    """Build causal continuous open/close marks from raw OHLC and pre-close.

    The multiplicative continuity factor removes ex-date discontinuities.  Its
    absolute level is arbitrary; position marking uses ratios from entry.
    """

    _require_columns(
        daily,
        {"symbol", "date", "open", "close", "pre_close"},
        name="daily market bars",
    )
    out = daily[["symbol", "date", "open", "close", "pre_close"]].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("open", "close", "pre_close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["symbol", "date"]).sort_values(
        ["symbol", "date"], kind="stable"
    )
    if out.duplicated(["symbol", "date"]).any():
        raise ValueError("daily market bars contain duplicate symbol/date keys")
    previous_close = out.groupby("symbol", sort=False)["close"].shift(1)
    ratio = previous_close.div(out["pre_close"])
    valid = ratio.notna() & np.isfinite(ratio.to_numpy(dtype=float)) & ratio.gt(0)
    ratio = ratio.where(valid, 1.0)
    out["continuity_factor"] = ratio.groupby(out["symbol"], sort=False).cumprod()
    out["continuous_open"] = out["open"] * out["continuity_factor"]
    out["continuous_close"] = out["close"] * out["continuity_factor"]
    return out[
        [
            "symbol",
            "date",
            "open",
            "close",
            "continuous_open",
            "continuous_close",
        ]
    ].reset_index(drop=True)


def prepare_capital_candidates(
    selections: pd.DataFrame,
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    arms: tuple[str, ...] = ("shared_playbook_model", "static_per_signal"),
) -> pd.DataFrame:
    """Join frozen B selections to ranking scores and exact realized outcomes."""

    _require_columns(
        selections,
        {
            "arm",
            "fold",
            "event_id",
            "symbol",
            "date",
            "playbook_id",
            "entry_mode",
            "eligible",
            "mature",
            "entry_date",
            "exit_date",
            "net_return",
            "round_trip_cost",
            "ambiguous_bar",
        },
        name="policy selections",
    )
    _require_columns(
        events,
        {"fold", "event_id", "first_layer_score"},
        name="playbook events",
    )
    _require_columns(
        outcomes,
        {
            "fold",
            "event_id",
            "playbook_id",
            "entry_raw_price",
            "exit_raw_price",
            "gross_return",
            "net_return",
            "round_trip_cost",
            "entry_date",
            "exit_date",
            "ambiguous_bar",
        },
        name="playbook outcomes",
    )
    selected = selections.loc[
        selections["arm"].astype(str).isin(arms)
        & selections["playbook_id"].astype(str).ne(NO_TRADE_PLAYBOOK_ID)
    ].copy()
    if set(selected["fold"].astype(str)) != {"B"}:
        raise ValueError("capital candidates must contain B only")
    if selected.duplicated(["arm", "fold", "event_id"]).any():
        raise ValueError("policy selections contain duplicate arm/event rows")

    event_scores = events[["fold", "event_id", "first_layer_score"]].copy()
    if event_scores.duplicated(["fold", "event_id"]).any():
        raise ValueError("playbook events contain duplicate event scores")
    selected = selected.merge(
        event_scores,
        on=["fold", "event_id"],
        how="left",
        validate="many_to_one",
    )
    if selected["first_layer_score"].isna().any():
        raise ValueError("capital candidates lack first-layer ranking scores")

    outcome_columns = [
        "fold",
        "event_id",
        "playbook_id",
        "entry_raw_price",
        "exit_raw_price",
        "gross_return",
        "net_return",
        "round_trip_cost",
        "entry_date",
        "exit_date",
        "ambiguous_bar",
    ]
    exact_outcomes = outcomes[outcome_columns].rename(
        columns={
            "net_return": "outcome_net_return",
            "round_trip_cost": "outcome_round_trip_cost",
            "entry_date": "outcome_entry_date",
            "exit_date": "outcome_exit_date",
            "ambiguous_bar": "outcome_ambiguous_bar",
        }
    )
    if exact_outcomes.duplicated(["fold", "event_id", "playbook_id"]).any():
        raise ValueError("playbook outcomes contain duplicate event/action rows")
    joined = selected.merge(
        exact_outcomes,
        on=["fold", "event_id", "playbook_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise ValueError("selected capital candidates lack exact outcome rows")
    joined = joined.drop(columns="_merge")
    selected_net = pd.to_numeric(joined["net_return"], errors="coerce")
    outcome_net = pd.to_numeric(joined["outcome_net_return"], errors="coerce")
    common_net = selected_net.notna() & outcome_net.notna()
    if not np.allclose(
        selected_net.loc[common_net].to_numpy(float),
        outcome_net.loc[common_net].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("selected policy net returns differ from outcome source")
    selected_entry = pd.to_datetime(joined["entry_date"], errors="coerce")
    outcome_entry = pd.to_datetime(joined["outcome_entry_date"], errors="coerce")
    selected_exit = pd.to_datetime(joined["exit_date"], errors="coerce")
    outcome_exit = pd.to_datetime(joined["outcome_exit_date"], errors="coerce")
    if not selected_entry.fillna(pd.Timestamp.min).eq(
        outcome_entry.fillna(pd.Timestamp.min)
    ).all() or not selected_exit.fillna(pd.Timestamp.min).eq(
        outcome_exit.fillna(pd.Timestamp.min)
    ).all():
        raise ValueError("selected policy entry/exit dates differ from outcome source")
    joined["date"] = pd.to_datetime(joined["date"], errors="coerce")
    joined["entry_date"] = selected_entry
    joined["exit_date"] = selected_exit
    joined["gross_return"] = pd.to_numeric(joined["gross_return"], errors="coerce")
    joined["source_net_return"] = outcome_net
    joined["source_round_trip_cost"] = pd.to_numeric(
        joined["outcome_round_trip_cost"], errors="coerce"
    )
    known = (
        joined["eligible"].fillna(False).astype(bool)
        & joined["mature"].fillna(False).astype(bool)
        & joined["gross_return"].notna()
        & joined["entry_date"].notna()
        & joined["exit_date"].notna()
    )
    joined["capital_evaluable"] = known
    return joined.sort_values(
        ["arm", "entry_date", "first_layer_score", "symbol", "event_id"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _scenario_net_return(gross_return: float, cost_bps: float) -> float:
    return float(gross_return) - float(cost_bps) / 10_000.0


def _metric_payload(
    *,
    arm: str,
    spec: CapitalBacktestSpec,
    curve: pd.DataFrame,
    orders: pd.DataFrame,
) -> dict[str, Any]:
    final_equity = float(curve["equity"].iloc[-1])
    total_return = final_equity / spec.initial_capital - 1.0
    sessions = int(len(curve))
    cagr = (
        float((final_equity / spec.initial_capital) ** (TRADING_SESSIONS_PER_YEAR / sessions) - 1.0)
        if sessions and final_equity > 0
        else np.nan
    )
    daily_returns = pd.to_numeric(curve["daily_return"], errors="coerce").dropna()
    volatility = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else np.nan
    sharpe = (
        float(daily_returns.mean() / volatility * np.sqrt(TRADING_SESSIONS_PER_YEAR))
        if np.isfinite(volatility) and volatility > 0
        else np.nan
    )
    drawdown = curve["equity"].div(curve["equity"].cummax()).sub(1.0)
    executed = orders["status"].eq("executed")
    rejected = orders["status"].eq("rejected")
    unevaluated = orders["status"].eq("unevaluated")
    known_orders = executed | rejected
    executed_returns = pd.to_numeric(
        orders.loc[executed, "scenario_net_return"], errors="coerce"
    )
    realized_pnl = pd.to_numeric(orders.loc[executed, "realized_pnl"], errors="coerce")
    accounting_error = final_equity - (spec.initial_capital + float(realized_pnl.sum()))
    rejection_counts = {
        str(reason): int(count)
        for reason, count in orders.loc[~executed].groupby("reason").size().items()
    }
    return {
        "schema_version": CAPITAL_BACKTEST_SCHEMA_VERSION,
        "evaluation_role": "B_development_descriptive_not_selection_not_promotion",
        "fold": "B",
        "arm": arm,
        "scenario_id": spec.scenario_id,
        "cost_bps": spec.cost_bps,
        "initial_capital": spec.initial_capital,
        "target_position_cash": spec.target_position_cash,
        "max_concurrent_positions": spec.max_concurrent_positions,
        "max_new_positions_per_session": spec.max_new_positions_per_session,
        "sessions": sessions,
        "final_equity": final_equity,
        "minimum_cash": float(curve["cash"].min()),
        "no_leverage_cash_non_negative": bool(curve["cash"].min() >= -1e-8),
        "final_open_positions": int(curve["open_positions"].iloc[-1]),
        "accounting_reconciliation_error": accounting_error,
        "accounting_reconciliation_pass": bool(abs(accounting_error) <= 1e-6),
        "total_return": total_return,
        "cagr_252": cagr,
        "max_drawdown": float(drawdown.min()),
        "daily_sharpe_252": sharpe,
        "candidate_orders": int(len(orders)),
        "known_candidate_orders": int(known_orders.sum()),
        "unevaluated_candidate_orders": int(unevaluated.sum()),
        "executed_trades": int(executed.sum()),
        "rejected_orders": int(rejected.sum()),
        "rejection_rate_all_candidates": float((rejected | unevaluated).mean()),
        "rejection_rate_known_candidates": (
            float(rejected.sum() / known_orders.sum()) if known_orders.any() else np.nan
        ),
        "executed_win_rate": (
            float(executed_returns.gt(0).mean()) if len(executed_returns) else np.nan
        ),
        "average_executed_net_return": (
            float(executed_returns.mean()) if len(executed_returns) else np.nan
        ),
        "maximum_observed_concurrent_positions": int(curve["open_positions"].max()),
        "maximum_observed_new_positions": int(curve["entries"].max()),
        "average_cash_utilization": float(curve["cash_utilization"].mean()),
        "board_lot_allocations": int(
            (orders.loc[executed, "allocation_mode"] == "raw_price_100_share_lot").sum()
        ),
        "equal_slot_price_fallback_allocations": int(
            (orders.loc[executed, "allocation_mode"] == "equal_slot_price_fallback").sum()
        ),
        "ambiguous_sl_first_executions": int(
            orders.loc[executed, "ambiguous_bar"].fillna(False).astype(bool).sum()
        ),
        "rejection_reason_counts": rejection_counts,
        "mark_to_market": {
            "method": "causal_continuous_raw_daily_close_ratio_then_exact_outcome_at_exit",
            "missing_position_day_marks": int(curve["missing_position_day_marks"].sum()),
            "warning": (
                "outcome net return is realized exactly at exit; daily marks reconstruct "
                "corporate-action continuity from raw close/pre_close"
            ),
        },
    }


def simulate_capital_constrained_policy(
    candidates: pd.DataFrame,
    daily_marks: pd.DataFrame,
    *,
    arm: str,
    spec: CapitalBacktestSpec = CapitalBacktestSpec(),
    sessions: pd.DatetimeIndex | None = None,
) -> CapitalSimulationResult:
    """Simulate one frozen policy arm under the supplied capital contract."""

    required_candidates = {
        "arm",
        "fold",
        "event_id",
        "symbol",
        "playbook_id",
        "entry_mode",
        "first_layer_score",
        "entry_date",
        "exit_date",
        "entry_raw_price",
        "gross_return",
        "source_net_return",
        "source_round_trip_cost",
        "capital_evaluable",
        "outcome_ambiguous_bar",
    }
    _require_columns(candidates, required_candidates, name="capital candidates")
    _require_columns(
        daily_marks,
        {"symbol", "date", "continuous_open", "continuous_close"},
        name="daily continuous marks",
    )
    arm_candidates = candidates.loc[candidates["arm"].astype(str).eq(arm)].copy()
    if arm_candidates.empty:
        raise ValueError(f"no capital candidates for arm={arm}")
    if set(arm_candidates["fold"].astype(str)) != {"B"}:
        raise ValueError("capital simulation is B-only")
    arm_candidates["entry_date"] = pd.to_datetime(
        arm_candidates["entry_date"], errors="coerce"
    )
    arm_candidates["exit_date"] = pd.to_datetime(
        arm_candidates["exit_date"], errors="coerce"
    )
    arm_candidates["first_layer_score"] = pd.to_numeric(
        arm_candidates["first_layer_score"], errors="raise"
    )
    source_cost = pd.to_numeric(
        arm_candidates.loc[arm_candidates["capital_evaluable"], "source_round_trip_cost"],
        errors="coerce",
    )
    if not np.isclose(source_cost.to_numpy(float), 0.0015, rtol=0.0, atol=1e-12).all():
        raise ValueError("source capital candidates do not carry exact 15 bps cost")
    if spec.cost_bps == DEFAULT_COST_BPS:
        source_net = pd.to_numeric(
            arm_candidates.loc[arm_candidates["capital_evaluable"], "source_net_return"],
            errors="coerce",
        )
        derived = pd.to_numeric(
            arm_candidates.loc[arm_candidates["capital_evaluable"], "gross_return"],
            errors="coerce",
        ).sub(spec.cost_bps / 10_000.0)
        if not np.allclose(
            source_net.to_numpy(float), derived.to_numpy(float), rtol=0.0, atol=1e-12
        ):
            raise ValueError("15 bps scenario cannot be reproduced from gross return")

    marks = daily_marks.copy()
    marks["symbol"] = marks["symbol"].astype(str)
    marks["date"] = pd.to_datetime(marks["date"], errors="coerce")
    if marks.duplicated(["symbol", "date"]).any():
        raise ValueError("daily marks contain duplicate symbol/date keys")
    mark_lookup = marks.set_index(["symbol", "date"])[
        ["continuous_open", "continuous_close"]
    ].to_dict("index")

    if sessions is None:
        sessions = pd.DatetimeIndex(
            sorted(
                set(marks["date"].dropna())
                | set(arm_candidates["entry_date"].dropna())
                | set(arm_candidates["exit_date"].dropna())
            )
        )
    else:
        sessions = pd.DatetimeIndex(pd.to_datetime(sessions)).sort_values().unique()
    minimum_entry = arm_candidates["entry_date"].min()
    maximum_exit = arm_candidates["exit_date"].max()
    sessions = sessions[(sessions >= minimum_entry) & (sessions <= maximum_exit)]
    if len(sessions) == 0:
        raise ValueError("capital simulation has no sessions")

    evaluable = arm_candidates["capital_evaluable"].fillna(False).astype(bool)
    known_candidates = arm_candidates.loc[evaluable].copy()
    unknown_candidates = arm_candidates.loc[~evaluable].copy()
    grouped_candidates = {
        pd.Timestamp(day): group.sort_values(
            ["first_layer_score", "symbol", "event_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        for day, group in known_candidates.groupby("entry_date", sort=True)
    }

    cash = float(spec.initial_capital)
    previous_equity = float(spec.initial_capital)
    open_positions: dict[str, dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    for row in unknown_candidates.itertuples(index=False):
        order_rows.append(
            {
                "scenario_id": spec.scenario_id,
                "arm": arm,
                "event_id": str(row.event_id),
                "symbol": str(row.symbol),
                "playbook_id": str(row.playbook_id),
                "entry_date": row.entry_date,
                "exit_date": row.exit_date,
                "first_layer_score": float(row.first_layer_score),
                "status": "unevaluated",
                "reason": "immature_or_missing_outcome",
                "principal": np.nan,
                "shares": np.nan,
                "allocation_mode": "not_allocated",
                "scenario_net_return": np.nan,
                "realized_pnl": np.nan,
                "ambiguous_bar": bool(row.outcome_ambiguous_bar)
                if pd.notna(row.outcome_ambiguous_bar)
                else False,
            }
        )

    for session in sessions:
        session = pd.Timestamp(session)
        exits_today = [
            event_id
            for event_id, position in open_positions.items()
            if position["exit_date"] == session
        ]
        realized_pnl_today = 0.0
        for event_id in exits_today:
            position = open_positions.pop(event_id)
            proceeds = position["principal"] * (1.0 + position["scenario_net_return"])
            cash += proceeds
            realized_pnl = proceeds - position["principal"]
            realized_pnl_today += realized_pnl
            order_rows[position["order_row"]]["status"] = "executed"
            order_rows[position["order_row"]]["reason"] = "filled_and_exited"
            order_rows[position["order_row"]]["realized_pnl"] = realized_pnl

        entries_today = 0
        rejected_today = 0
        for row in grouped_candidates.get(session, pd.DataFrame()).itertuples(index=False):
            reason: str | None = None
            if entries_today >= spec.max_new_positions_per_session:
                reason = "daily_new_position_limit"
            elif len(open_positions) >= spec.max_concurrent_positions:
                reason = "concurrent_position_limit"
            raw_entry_price = _finite_float(row.entry_raw_price)
            allocation_mode = "raw_price_100_share_lot"
            shares = np.nan
            principal = np.nan
            if reason is None:
                if np.isfinite(raw_entry_price) and raw_entry_price > 0:
                    shares = float(
                        np.floor(
                            min(spec.target_position_cash, cash)
                            / (raw_entry_price * spec.board_lot_size)
                        )
                        * spec.board_lot_size
                    )
                    if shares < spec.board_lot_size:
                        reason = "insufficient_cash_or_board_lot"
                    else:
                        principal = shares * raw_entry_price
                else:
                    allocation_mode = "equal_slot_price_fallback"
                    principal = min(spec.target_position_cash, cash)
                    if principal <= 0:
                        reason = "insufficient_cash"
            scenario_return = _scenario_net_return(row.gross_return, spec.cost_bps)
            order_index = len(order_rows)
            order_rows.append(
                {
                    "scenario_id": spec.scenario_id,
                    "arm": arm,
                    "event_id": str(row.event_id),
                    "symbol": str(row.symbol),
                    "playbook_id": str(row.playbook_id),
                    "entry_date": session,
                    "exit_date": pd.Timestamp(row.exit_date),
                    "first_layer_score": float(row.first_layer_score),
                    "status": "rejected" if reason else "open",
                    "reason": reason or "filled_pending_exit",
                    "principal": principal,
                    "shares": shares,
                    "allocation_mode": allocation_mode,
                    "scenario_net_return": scenario_return,
                    "realized_pnl": np.nan,
                    "ambiguous_bar": bool(row.outcome_ambiguous_bar)
                    if pd.notna(row.outcome_ambiguous_bar)
                    else False,
                }
            )
            if reason is not None:
                rejected_today += 1
                continue
            assert np.isfinite(principal)
            if principal > cash + 1e-8:
                raise RuntimeError("capital simulator attempted to use unavailable cash")
            cash -= principal
            entry_marks = mark_lookup.get((str(row.symbol), session), {})
            entry_mark_column = (
                "continuous_open" if str(row.entry_mode) == "next_open" else "continuous_close"
            )
            entry_continuous_price = _finite_float(
                entry_marks.get(entry_mark_column, np.nan)
            )
            open_positions[str(row.event_id)] = {
                "event_id": str(row.event_id),
                "symbol": str(row.symbol),
                "entry_date": session,
                "exit_date": pd.Timestamp(row.exit_date),
                "principal": float(principal),
                "entry_continuous_price": entry_continuous_price,
                "last_mark_value": float(principal),
                "scenario_net_return": scenario_return,
                "order_row": order_index,
            }
            entries_today += 1

        market_value = 0.0
        missing_marks = 0
        for position in open_positions.values():
            current = mark_lookup.get((position["symbol"], session), {})
            continuous_close = _finite_float(current.get("continuous_close", np.nan))
            entry_continuous_price = _finite_float(position["entry_continuous_price"])
            if (
                np.isfinite(continuous_close)
                and continuous_close > 0
                and np.isfinite(entry_continuous_price)
                and entry_continuous_price > 0
            ):
                mark_value = position["principal"] * (
                    continuous_close / entry_continuous_price
                )
                position["last_mark_value"] = float(mark_value)
            else:
                mark_value = float(position["last_mark_value"])
                missing_marks += 1
            market_value += mark_value
        equity = cash + market_value
        daily_return = equity / previous_equity - 1.0 if previous_equity else np.nan
        curve_rows.append(
            {
                "scenario_id": spec.scenario_id,
                "arm": arm,
                "date": session,
                "cash": cash,
                "market_value": market_value,
                "equity": equity,
                "daily_return": daily_return,
                "cumulative_return": equity / spec.initial_capital - 1.0,
                "open_positions": len(open_positions),
                "entries": entries_today,
                "exits": len(exits_today),
                "rejections": rejected_today,
                "realized_pnl": realized_pnl_today,
                "cash_utilization": market_value / equity if equity > 0 else np.nan,
                "missing_position_day_marks": missing_marks,
            }
        )
        previous_equity = equity

    if open_positions:
        raise ValueError("capital simulation ended with unclosed mature positions")
    curve = pd.DataFrame(curve_rows)
    orders = pd.DataFrame(order_rows).sort_values(
        ["entry_date", "first_layer_score", "symbol", "event_id"],
        ascending=[True, False, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    if orders["status"].eq("open").any():
        raise RuntimeError("capital simulation order log contains unresolved open orders")
    metrics = _metric_payload(arm=arm, spec=spec, curve=curve, orders=orders)
    return CapitalSimulationResult(curve=curve, orders=orders, metrics=metrics)


__all__ = [
    "CAPITAL_BACKTEST_SCHEMA_VERSION",
    "CapitalBacktestSpec",
    "CapitalSimulationResult",
    "build_continuous_daily_marks",
    "prepare_capital_candidates",
    "simulate_capital_constrained_policy",
]
