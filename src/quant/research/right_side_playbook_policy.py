"""Research-only executable playbooks for unified right-side signals.

The first-layer model ranks stocks after the signal-date close.  This module
materializes counterfactual outcomes for a deliberately small, preregistered
menu of entry/exit actions.  Entry gates use raw point-in-time prices while
returns and barriers use the project's causal continuous-price scale.

Important execution conventions
--------------------------------

* entry is on the next *global market* session, never the next stock-local bar;
* an entry-session purchase cannot be sold until the following market session;
* a one-price limit-up is never executable;
* ``next_open`` also rejects an exact open-at-limit print, even if the stock
  trades below the limit later that day;
* ``next_close`` can independently reject an exact close-at-limit print;
* an intraday stop and target on the same daily bar are ordered stop first and
  the row is marked ambiguous for sensitivity analysis;
* an incomplete tail remains unlabeled instead of being treated as a loss.

This is an event-outcome engine, not a capital-aware backtest.  It deliberately
does not model portfolio cash, overlapping positions, queue priority, or an
unobservable intraday path inside one daily OHLC bar.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc
from quant.research.right_side_unified_labels import detect_next_locked_limit_up


EntryMode = Literal["next_open", "next_close", "no_trade"]
ExitKind = Literal["expiry", "fixed", "trailing", "no_trade"]

PLAYBOOK_POLICY_VERSION = "right_side_playbook_v1"
DEFAULT_ROUND_TRIP_COST_BPS = 15.0


@dataclass(frozen=True)
class EntryConstraint:
    """Preregistered entry mode and raw next-session execution gates."""

    mode: Literal["next_open", "next_close"]
    min_open_gap: float | None = None
    max_open_gap: float | None = None
    reject_locked_limit_up: bool = True
    reject_open_at_up_limit: bool = False
    reject_close_at_up_limit: bool = False
    exact_limit_tolerance: float = 0.0051

    def __post_init__(self) -> None:
        if self.mode not in {"next_open", "next_close"}:
            raise ValueError(f"unsupported entry mode: {self.mode}")
        if self.min_open_gap is not None and not np.isfinite(self.min_open_gap):
            raise ValueError("min_open_gap must be finite when provided")
        if self.max_open_gap is not None and not np.isfinite(self.max_open_gap):
            raise ValueError("max_open_gap must be finite when provided")
        if (
            self.min_open_gap is not None
            and self.max_open_gap is not None
            and self.min_open_gap > self.max_open_gap
        ):
            raise ValueError("min_open_gap cannot exceed max_open_gap")
        if self.exact_limit_tolerance < 0:
            raise ValueError("exact_limit_tolerance must be non-negative")


@dataclass(frozen=True)
class ExitPolicy:
    """One fixed exit rule evaluated from T+2 through its expiry session.

    ``hold_sessions`` counts global market sessions after the entry session.
    Thus an entry on T+1 with ``hold_sessions=2`` expires at the T+3 close.
    """

    policy_id: str
    kind: ExitKind
    hold_sessions: int
    take_profit: float | None = None
    stop_loss: float | None = None
    trailing_drawdown: float | None = None

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id cannot be empty")
        if self.kind not in {"expiry", "fixed", "trailing", "no_trade"}:
            raise ValueError(f"unsupported exit kind: {self.kind}")
        if self.kind == "no_trade":
            if self.hold_sessions != 0:
                raise ValueError("no_trade must have hold_sessions=0")
            return
        if self.hold_sessions <= 0:
            raise ValueError("hold_sessions must be positive")
        if self.kind == "expiry":
            if any(
                value is not None
                for value in (self.take_profit, self.stop_loss, self.trailing_drawdown)
            ):
                raise ValueError("expiry policy cannot define barriers")
            return
        if self.take_profit is None or self.stop_loss is None:
            raise ValueError(f"{self.kind} policy requires take_profit and stop_loss")
        if not np.isfinite(self.take_profit) or self.take_profit <= 0:
            raise ValueError("take_profit must be positive and finite")
        if not np.isfinite(self.stop_loss) or self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive and finite")
        if self.kind == "trailing":
            if self.trailing_drawdown is None:
                raise ValueError("trailing policy requires trailing_drawdown")
            if not np.isfinite(self.trailing_drawdown) or self.trailing_drawdown <= 0:
                raise ValueError("trailing_drawdown must be positive and finite")
        elif self.trailing_drawdown is not None:
            raise ValueError("fixed policy cannot define trailing_drawdown")


@dataclass(frozen=True)
class PlaybookSpec:
    """A stable action identifier plus its entry, exit, and cost contract."""

    playbook_id: str
    entry: EntryConstraint | None
    exit: ExitPolicy
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS
    version: str = PLAYBOOK_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.playbook_id:
            raise ValueError("playbook_id cannot be empty")
        if not np.isfinite(self.round_trip_cost_bps) or self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps must be finite and non-negative")
        if not self.version:
            raise ValueError("version cannot be empty")
        if self.exit.kind == "no_trade":
            if self.entry is not None or self.round_trip_cost_bps != 0:
                raise ValueError("no_trade requires entry=None and zero cost")
        elif self.entry is None:
            raise ValueError("trade playbook requires an entry constraint")

    @property
    def is_no_trade(self) -> bool:
        return self.exit.kind == "no_trade"


EXPIRY_T3_CLOSE = ExitPolicy("expiry_t3_close", "expiry", hold_sessions=2)
EXPIRY_T5_CLOSE = ExitPolicy("expiry_t5_close", "expiry", hold_sessions=4)
FIXED_TP4_SL2_T5 = ExitPolicy(
    "fixed_tp4_sl2_t5",
    "fixed",
    hold_sessions=4,
    take_profit=0.04,
    stop_loss=0.02,
)
FIXED_TP6_SL3_T5 = ExitPolicy(
    "fixed_tp6_sl3_t5",
    "fixed",
    hold_sessions=4,
    take_profit=0.06,
    stop_loss=0.03,
)

DEFAULT_EXIT_POLICIES: tuple[ExitPolicy, ...] = (
    EXPIRY_T3_CLOSE,
    EXPIRY_T5_CLOSE,
    FIXED_TP4_SL2_T5,
    FIXED_TP6_SL3_T5,
)

DEFAULT_OPEN_ENTRY = EntryConstraint(
    "next_open",
    reject_open_at_up_limit=True,
)
DEFAULT_CLOSE_ENTRY = EntryConstraint(
    "next_close",
    reject_close_at_up_limit=True,
)

DEFAULT_TRADE_PLAYBOOKS: tuple[PlaybookSpec, ...] = tuple(
    PlaybookSpec(
        f"{entry.mode}__{exit_policy.policy_id}",
        entry,
        exit_policy,
    )
    for entry in (DEFAULT_OPEN_ENTRY, DEFAULT_CLOSE_ENTRY)
    for exit_policy in DEFAULT_EXIT_POLICIES
)

NO_TRADE_PLAYBOOK = PlaybookSpec(
    "no_trade",
    None,
    ExitPolicy("no_trade", "no_trade", hold_sessions=0),
    round_trip_cost_bps=0.0,
)
NO_TRADE_PLAYBOOK_ID = NO_TRADE_PLAYBOOK.playbook_id

DEFAULT_PLAYBOOK_CATALOG: tuple[PlaybookSpec, ...] = (
    *DEFAULT_TRADE_PLAYBOOKS,
    NO_TRADE_PLAYBOOK,
)


def default_playbook_catalog(*, include_no_trade: bool = True) -> tuple[PlaybookSpec, ...]:
    """Return the frozen v1 catalog in deterministic action order."""

    return DEFAULT_PLAYBOOK_CATALOG if include_no_trade else DEFAULT_TRADE_PLAYBOOKS


def serialize_playbook_catalog(
    catalog: Sequence[PlaybookSpec] | None = None,
) -> list[dict[str, object]]:
    """Return an ordered JSON-safe catalog payload for an artifact manifest."""

    selected = _validate_catalog(DEFAULT_PLAYBOOK_CATALOG if catalog is None else catalog)
    return [asdict(spec) for spec in selected]


def playbook_catalog_hash(catalog: Sequence[PlaybookSpec] | None = None) -> str:
    """Hash every action parameter and the deterministic catalog order."""

    encoded = json.dumps(
        serialize_playbook_catalog(catalog),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CORE_OUTPUT_COLUMNS = [
    "event_id",
    "symbol",
    "date",
    "playbook_id",
    "playbook_version",
    "entry_mode",
    "exit_policy_id",
    "eligible",
    "eligibility_reason",
    "mature",
    "maturity_reason",
    "entry_date",
    "entry_price",
    "entry_raw_price",
    "exit_date",
    "exit_price",
    "exit_raw_price",
    "exit_reason",
    "gross_return",
    "net_return",
    "mae",
    "holding_sessions",
    "ambiguous_bar",
    "round_trip_cost",
    "round_trip_cost_bps",
    "locked_limit_up",
    "locked_limit_source",
    "open_at_up_limit",
    "close_at_up_limit",
    "open_gap",
]


def _normalize_dates(values: pd.Series, *, name: str) -> pd.Series:
    text = values.astype("string").str.strip()
    compact = text.str.replace("-", "", regex=False)
    eight_digit = compact.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(text, errors="coerce")
    if eight_digit.any():
        parsed.loc[eight_digit] = pd.to_datetime(
            compact.loc[eight_digit], format="%Y%m%d", errors="coerce"
        )
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        samples = values.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(f"{name} contains invalid dates: {samples}")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _normalize_calendar(values: Sequence[object]) -> pd.DatetimeIndex:
    raw = pd.Series(list(values), dtype="object")
    if raw.empty:
        raise ValueError("market_calendar must contain at least one trading date")
    dates = _normalize_dates(raw, name="market_calendar")
    if dates.isna().any():
        raise ValueError("market_calendar cannot contain missing dates")
    if dates.duplicated().any():
        raise ValueError("market_calendar cannot contain duplicate dates")
    return pd.DatetimeIndex(dates.sort_values().to_numpy())


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def _prepare_daily(
    daily: pd.DataFrame,
    tradability: pd.DataFrame | None,
    *,
    daily_symbol_column: str,
    daily_date_column: str,
    tradability_symbol_column: str,
    tradability_date_column: str,
) -> pd.DataFrame:
    required = {
        daily_symbol_column,
        daily_date_column,
        "open",
        "high",
        "low",
        "close",
        "pre_close",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily missing columns: {missing}")

    out = pd.DataFrame(index=daily.index)
    out["symbol"] = daily[daily_symbol_column].astype("string").str.strip()
    out["date"] = _normalize_dates(daily[daily_date_column], name=daily_date_column)
    for column in ("open", "high", "low", "close", "pre_close"):
        out[f"raw_{column}"] = pd.to_numeric(daily[column], errors="coerce")
    out["daily_up_limit"] = (
        pd.to_numeric(daily["up_limit"], errors="coerce")
        if "up_limit" in daily.columns
        else np.nan
    )
    out["daily_suspended"] = (
        _as_bool(daily["is_suspended"])
        if "is_suspended" in daily.columns
        else False
    )
    out["has_daily_bar"] = True
    if out["symbol"].isna().any() or out["symbol"].eq("").any():
        raise ValueError("daily contains missing symbols")
    if out["date"].isna().any():
        raise ValueError("daily contains missing dates")
    if out.duplicated(["symbol", "date"]).any():
        raise ValueError("daily contains duplicate symbol/date rows")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)

    for column in ("open", "high", "low", "close"):
        out[f"adjusted_{column}"] = np.nan
    for _, positions in out.groupby("symbol", sort=False).groups.items():
        idx = list(positions)
        source = pd.DataFrame(
            {
                "date": out.loc[idx, "date"],
                "open": out.loc[idx, "raw_open"],
                "high": out.loc[idx, "raw_high"],
                "low": out.loc[idx, "raw_low"],
                "close": out.loc[idx, "raw_close"],
                "pre_close": out.loc[idx, "raw_pre_close"],
            },
            index=idx,
        )
        adjusted = build_continuous_ohlc(source)
        for column in ("open", "high", "low", "close"):
            out.loc[idx, f"adjusted_{column}"] = pd.to_numeric(
                adjusted[column], errors="coerce"
            ).to_numpy()

    if tradability is None or tradability.empty:
        out["tradability_up_limit"] = np.nan
        out["tradability_suspended"] = False
    else:
        required_tradability = {tradability_symbol_column, tradability_date_column}
        missing = sorted(required_tradability - set(tradability.columns))
        if missing:
            raise ValueError(f"tradability missing columns: {missing}")
        available = pd.DataFrame(index=tradability.index)
        available["symbol"] = (
            tradability[tradability_symbol_column].astype("string").str.strip()
        )
        available["date"] = _normalize_dates(
            tradability[tradability_date_column], name=tradability_date_column
        )
        available["tradability_up_limit"] = (
            pd.to_numeric(tradability["up_limit"], errors="coerce")
            if "up_limit" in tradability.columns
            else np.nan
        )
        available["tradability_suspended"] = (
            _as_bool(tradability["is_suspended"])
            if "is_suspended" in tradability.columns
            else False
        )
        if available.duplicated(["symbol", "date"]).any():
            raise ValueError("tradability contains duplicate symbol/date rows")
        # Keep tradability-only suspension rows.  A stock can legitimately have
        # no OHLC bar on a globally open session while the point-in-time table
        # explicitly records that it was suspended.
        out = out.merge(
            available,
            on=["symbol", "date"],
            how="outer",
            validate="one_to_one",
        )
        out["daily_suspended"] = out["daily_suspended"].fillna(False)
        out["has_daily_bar"] = out["has_daily_bar"].fillna(False)
        out["tradability_suspended"] = out["tradability_suspended"].fillna(False)

    out["up_limit"] = out["tradability_up_limit"].combine_first(out["daily_up_limit"])
    out["is_suspended"] = (
        out["daily_suspended"].fillna(False).astype(bool)
        | out["tradability_suspended"].fillna(False).astype(bool)
    )
    detection = detect_next_locked_limit_up(
        pd.DataFrame(
            {
                "open": out["raw_open"],
                "high": out["raw_high"],
                "low": out["raw_low"],
                "close": out["raw_close"],
                "pre_close": out["raw_pre_close"],
                "up_limit": out["up_limit"],
            },
            index=out.index,
        )
    )
    out["locked_limit_up"] = detection["locked_limit_up"]
    out["locked_limit_source"] = detection["locked_limit_source"]
    exact_limit = np.isfinite(out["up_limit"].to_numpy(dtype=float)) & out["up_limit"].gt(0)
    out["open_at_up_limit"] = exact_limit & np.isclose(
        out["raw_open"].to_numpy(dtype=float),
        out["up_limit"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=0.0051,
        equal_nan=False,
    )
    out["close_at_up_limit"] = exact_limit & np.isclose(
        out["raw_close"].to_numpy(dtype=float),
        out["up_limit"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=0.0051,
        equal_nan=False,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["open_gap"] = out["raw_open"] / out["raw_pre_close"] - 1.0
    return out


def _validate_catalog(catalog: Sequence[PlaybookSpec]) -> tuple[PlaybookSpec, ...]:
    normalized = tuple(catalog)
    if not normalized:
        raise ValueError("catalog cannot be empty")
    if any(not isinstance(spec, PlaybookSpec) for spec in normalized):
        raise TypeError("catalog must contain PlaybookSpec values")
    ids = [spec.playbook_id for spec in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("catalog contains duplicate playbook_id values")
    return normalized


def _finite_positive(*values: object) -> bool:
    numbers = np.asarray(values, dtype=float)
    return bool(np.isfinite(numbers).all() and (numbers > 0).all())


def _is_at_limit(value: float, limit: float, tolerance: float) -> bool:
    return bool(
        np.isfinite(value)
        and np.isfinite(limit)
        and limit > 0
        and np.isclose(value, limit, rtol=1e-9, atol=tolerance)
    )


def _entry_snapshot(
    entry_row: pd.Series | None,
    entry: EntryConstraint,
    *,
    has_next_market_date: bool,
) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "eligible": False,
        "eligibility_reason": "no_next_market_date",
        "entry_price": np.nan,
        "entry_raw_price": np.nan,
        "locked_limit_up": False,
        "locked_limit_source": "unavailable",
        "open_at_up_limit": False,
        "close_at_up_limit": False,
        "open_gap": np.nan,
    }
    if not has_next_market_date:
        return snapshot
    if entry_row is None:
        snapshot["eligibility_reason"] = "missing_entry_bar"
        return snapshot

    snapshot.update(
        {
            "locked_limit_up": bool(entry_row["locked_limit_up"]),
            "locked_limit_source": str(entry_row["locked_limit_source"]),
            "open_at_up_limit": bool(entry_row["open_at_up_limit"]),
            "close_at_up_limit": bool(entry_row["close_at_up_limit"]),
            "open_gap": float(entry_row["open_gap"]),
        }
    )
    if bool(entry_row["is_suspended"]):
        snapshot["eligibility_reason"] = "suspended_entry"
        return snapshot
    if not bool(entry_row["has_daily_bar"]):
        snapshot["eligibility_reason"] = "missing_entry_bar"
        return snapshot
    if entry.reject_locked_limit_up and bool(entry_row["locked_limit_up"]):
        snapshot["eligibility_reason"] = "locked_limit_up"
        return snapshot

    raw_column = "raw_open" if entry.mode == "next_open" else "raw_close"
    adjusted_column = "adjusted_open" if entry.mode == "next_open" else "adjusted_close"
    raw_price = float(entry_row[raw_column])
    adjusted_price = float(entry_row[adjusted_column])
    snapshot["entry_raw_price"] = raw_price
    snapshot["entry_price"] = adjusted_price
    if not _finite_positive(raw_price, adjusted_price):
        snapshot["eligibility_reason"] = "invalid_entry_price"
        return snapshot

    up_limit = float(entry_row["up_limit"])
    if entry.reject_open_at_up_limit and _is_at_limit(
        float(entry_row["raw_open"]), up_limit, entry.exact_limit_tolerance
    ):
        snapshot["eligibility_reason"] = "open_at_up_limit"
        return snapshot
    if entry.reject_close_at_up_limit and _is_at_limit(
        float(entry_row["raw_close"]), up_limit, entry.exact_limit_tolerance
    ):
        snapshot["eligibility_reason"] = "close_at_up_limit"
        return snapshot

    gap = float(entry_row["open_gap"])
    if entry.min_open_gap is not None and (not np.isfinite(gap) or gap < entry.min_open_gap):
        snapshot["eligibility_reason"] = "open_gap_below_min"
        return snapshot
    if entry.max_open_gap is not None and (not np.isfinite(gap) or gap > entry.max_open_gap):
        snapshot["eligibility_reason"] = "open_gap_above_max"
        return snapshot

    snapshot["eligible"] = True
    snapshot["eligibility_reason"] = "eligible"
    return snapshot


def _incomplete_result(reason: str, *, mae: float) -> dict[str, object]:
    return {
        "mature": False,
        "maturity_reason": reason,
        "exit_date": pd.NaT,
        "exit_price": np.nan,
        "exit_raw_price": np.nan,
        "exit_reason": pd.NA,
        "gross_return": np.nan,
        "net_return": np.nan,
        "mae": np.nan,
        "holding_sessions": pd.NA,
        "ambiguous_bar": False,
    }


def _mature_result(
    *,
    exit_date: pd.Timestamp,
    exit_price: float,
    exit_raw_price: float,
    exit_reason: str,
    entry_price: float,
    mae: float,
    holding_sessions: int,
    ambiguous_bar: bool,
    cost: float,
) -> dict[str, object]:
    gross_return = exit_price / entry_price - 1.0
    return {
        "mature": True,
        "maturity_reason": "mature",
        "exit_date": exit_date,
        "exit_price": float(exit_price),
        "exit_raw_price": float(exit_raw_price),
        "exit_reason": exit_reason,
        "gross_return": float(gross_return),
        "net_return": float(gross_return - cost),
        "mae": float(mae),
        "holding_sessions": int(holding_sessions),
        "ambiguous_bar": bool(ambiguous_bar),
    }


def _simulate_exit(
    *,
    policy: ExitPolicy,
    entry: EntryConstraint,
    entry_row: pd.Series,
    entry_price: float,
    entry_calendar_position: int,
    calendar: pd.DatetimeIndex,
    daily_lookup: dict[tuple[str, pd.Timestamp], pd.Series],
    symbol: str,
    cost: float,
) -> dict[str, object]:
    # A next-open position is exposed to the entry day's low, but A-share T+1
    # still prevents liquidation on that day.  A next-close entry is not.
    mae = 0.0
    if entry.mode == "next_open":
        entry_low = float(entry_row["adjusted_low"])
        if np.isfinite(entry_low):
            mae = min(0.0, entry_low / entry_price - 1.0)

    stop_price = (
        entry_price * (1.0 - float(policy.stop_loss))
        if policy.stop_loss is not None
        else np.nan
    )
    target_price = (
        entry_price * (1.0 + float(policy.take_profit))
        if policy.take_profit is not None
        else np.nan
    )
    peak = entry_price
    trailing_active = False
    if policy.kind == "trailing" and entry.mode == "next_open":
        entry_high = float(entry_row["adjusted_high"])
        if np.isfinite(entry_high):
            peak = max(peak, entry_high)
            # The position cannot be sold on its purchase session, but its
            # observable high can arm a trailing stop for the first sellable
            # session.
            trailing_active = peak >= target_price
    trailing_ambiguity_seen = False

    for holding_session in range(1, policy.hold_sessions + 1):
        calendar_position = entry_calendar_position + holding_session
        if calendar_position >= len(calendar):
            return _incomplete_result("incomplete_market_window", mae=mae)
        exit_date = pd.Timestamp(calendar[calendar_position])
        row = daily_lookup.get((symbol, exit_date))
        if row is None:
            return _incomplete_result("missing_future_bar", mae=mae)
        if bool(row["is_suspended"]):
            # Suspension prevents any barrier fill, but it does not move the
            # preregistered global-session clock.  An unsellable expiry itself
            # remains unlabeled; extending to a later date would be a distinct
            # playbook rather than an ex-post repair.
            if holding_session == policy.hold_sessions:
                return _incomplete_result("suspended_expiry_session", mae=mae)
            continue
        if not bool(row["has_daily_bar"]):
            return _incomplete_result("missing_future_bar", mae=mae)

        adjusted_open = float(row["adjusted_open"])
        adjusted_high = float(row["adjusted_high"])
        adjusted_low = float(row["adjusted_low"])
        adjusted_close = float(row["adjusted_close"])
        raw_open = float(row["raw_open"])
        raw_close = float(row["raw_close"])
        if not _finite_positive(
            adjusted_open,
            adjusted_high,
            adjusted_low,
            adjusted_close,
            raw_open,
            raw_close,
        ):
            return _incomplete_result("invalid_future_price", mae=mae)

        if policy.kind == "expiry":
            mae = min(mae, adjusted_low / entry_price - 1.0)
            if holding_session == policy.hold_sessions:
                return _mature_result(
                    exit_date=exit_date,
                    exit_price=adjusted_close,
                    exit_raw_price=raw_close,
                    exit_reason="expiry",
                    entry_price=entry_price,
                    mae=mae,
                    holding_sessions=holding_session,
                    ambiguous_bar=False,
                    cost=cost,
                )
            continue

        stop_hit = adjusted_low <= stop_price
        target_hit = adjusted_high >= target_price
        same_bar_ambiguous = bool(stop_hit and target_hit)

        # This matches the established b1_backtest convention: initial stop is
        # evaluated before target/trailing logic; a gap through the stop fills
        # at the opening price rather than the unreachable stop threshold.
        if stop_hit:
            fill = adjusted_open if adjusted_open <= stop_price else stop_price
            raw_fill = raw_open * (fill / adjusted_open)
            mae = min(mae, fill / entry_price - 1.0)
            return _mature_result(
                exit_date=exit_date,
                exit_price=fill,
                exit_raw_price=raw_fill,
                exit_reason="stop_loss",
                entry_price=entry_price,
                mae=mae,
                holding_sessions=holding_session,
                ambiguous_bar=same_bar_ambiguous,
                cost=cost,
            )

        if policy.kind == "fixed" and target_hit:
            mae = min(mae, adjusted_low / entry_price - 1.0)
            # Target fills at the preregistered threshold even after a gap-up;
            # this keeps the optimistic side conservative and matches the
            # established research simulator.
            raw_target = raw_open * (target_price / adjusted_open)
            return _mature_result(
                exit_date=exit_date,
                exit_price=target_price,
                exit_raw_price=raw_target,
                exit_reason="take_profit",
                entry_price=entry_price,
                mae=mae,
                holding_sessions=holding_session,
                ambiguous_bar=False,
                cost=cost,
            )

        if policy.kind == "trailing":
            previous_peak = peak
            previous_active = trailing_active
            if previous_active:
                previous_trailing_price = previous_peak * (
                    1.0 - float(policy.trailing_drawdown)
                )
                if adjusted_low <= previous_trailing_price:
                    fill = (
                        adjusted_open
                        if adjusted_open <= previous_trailing_price
                        else previous_trailing_price
                    )
                    raw_fill = raw_open * (fill / adjusted_open)
                    mae = min(mae, fill / entry_price - 1.0)
                    return _mature_result(
                        exit_date=exit_date,
                        exit_price=fill,
                        exit_raw_price=raw_fill,
                        exit_reason="trailing_stop",
                        entry_price=entry_price,
                        mae=mae,
                        holding_sessions=holding_session,
                        ambiguous_bar=(
                            trailing_ambiguity_seen or adjusted_high > previous_peak
                        ),
                        cost=cost,
                    )
            peak = max(peak, adjusted_high)
            if not trailing_active and peak >= target_price:
                # A newly activated trailing stop becomes executable only on
                # the following session; daily OHLC cannot prove high-before-
                # low ordering within its activation bar.
                trailing_active = True
                trailing_price = peak * (1.0 - float(policy.trailing_drawdown))
                trailing_ambiguity_seen |= adjusted_low <= trailing_price

        mae = min(mae, adjusted_low / entry_price - 1.0)
        if holding_session == policy.hold_sessions:
            return _mature_result(
                exit_date=exit_date,
                exit_price=adjusted_close,
                exit_raw_price=raw_close,
                exit_reason="expiry",
                entry_price=entry_price,
                mae=mae,
                holding_sessions=holding_session,
                ambiguous_bar=trailing_ambiguity_seen,
                cost=cost,
            )

    raise AssertionError("unreachable exit simulation state")


def _no_trade_row(event: dict[str, object], spec: PlaybookSpec) -> dict[str, object]:
    return {
        **event,
        "playbook_id": spec.playbook_id,
        "playbook_version": spec.version,
        "entry_mode": "no_trade",
        "exit_policy_id": spec.exit.policy_id,
        "eligible": True,
        "eligibility_reason": "eligible",
        "mature": True,
        "maturity_reason": "not_applicable",
        "entry_date": pd.NaT,
        "entry_price": np.nan,
        "entry_raw_price": np.nan,
        "exit_date": pd.NaT,
        "exit_price": np.nan,
        "exit_raw_price": np.nan,
        "exit_reason": "no_trade",
        "gross_return": 0.0,
        "net_return": 0.0,
        "mae": 0.0,
        "holding_sessions": 0,
        "ambiguous_bar": False,
        "round_trip_cost": 0.0,
        "round_trip_cost_bps": 0.0,
        "locked_limit_up": False,
        "locked_limit_source": "not_applicable",
        "open_at_up_limit": False,
        "close_at_up_limit": False,
        "open_gap": np.nan,
    }


def build_playbook_outcomes(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    market_calendar: Sequence[object],
    tradability: pd.DataFrame | None = None,
    *,
    catalog: Sequence[PlaybookSpec] | None = None,
    event_id_column: str = "event_id",
    signal_symbol_column: str = "symbol",
    signal_date_column: str = "date",
    daily_symbol_column: str = "ts_code",
    daily_date_column: str = "trade_date",
    tradability_symbol_column: str = "ts_code",
    tradability_date_column: str = "trade_date",
) -> pd.DataFrame:
    """Build one outcome row per signal and preregistered playbook.

    Original signal columns are repeated on every action row, allowing causal
    event factors and first-layer OOF predictions to flow into a downstream
    policy model.  Output fields owned by this function may not already appear
    in ``signals`` (apart from the configured event/symbol/date columns).

    This in-memory builder is intended for tests, smoke runs, or an already
    narrow event table.  A full 217-factor research build should stream events
    by symbol/date partition and keep factors in a separate event-keyed table
    rather than copying every factor into every action row.
    """

    required_signals = {signal_symbol_column, signal_date_column}
    missing = sorted(required_signals - set(signals.columns))
    if missing:
        raise ValueError(f"signals missing columns: {missing}")
    selected_catalog = _validate_catalog(
        DEFAULT_PLAYBOOK_CATALOG if catalog is None else catalog
    )
    owned = set(_CORE_OUTPUT_COLUMNS) - {
        event_id_column,
        signal_symbol_column,
        signal_date_column,
    }
    collisions = sorted(owned & set(signals.columns))
    if collisions:
        raise ValueError(f"signals collide with outcome columns: {collisions}")

    calendar = _normalize_calendar(market_calendar)
    prepared_daily = _prepare_daily(
        daily,
        tradability,
        daily_symbol_column=daily_symbol_column,
        daily_date_column=daily_date_column,
        tradability_symbol_column=tradability_symbol_column,
        tradability_date_column=tradability_date_column,
    )
    outside = ~prepared_daily["date"].isin(calendar)
    if outside.any():
        samples = (
            prepared_daily.loc[outside, "date"]
            .dt.strftime("%Y-%m-%d")
            .drop_duplicates()
            .head(3)
            .tolist()
        )
        raise ValueError(f"daily dates are absent from market_calendar: {samples}")

    base = signals.copy().reset_index(drop=True)
    base["symbol"] = base[signal_symbol_column].astype("string").str.strip()
    base["date"] = _normalize_dates(base[signal_date_column], name=signal_date_column)
    if base["symbol"].isna().any() or base["symbol"].eq("").any():
        raise ValueError("signals contains missing symbols")
    if base["date"].isna().any():
        raise ValueError("signals contains missing dates")
    if (~base["date"].isin(calendar)).any():
        samples = (
            base.loc[~base["date"].isin(calendar), "date"]
            .dt.strftime("%Y-%m-%d")
            .head(3)
            .tolist()
        )
        raise ValueError(f"signal dates are absent from market_calendar: {samples}")
    if event_id_column in base.columns:
        base["event_id"] = base[event_id_column].astype("string")
        if base["event_id"].isna().any() or base["event_id"].eq("").any():
            raise ValueError("signals contains missing event_id values")
    else:
        base["event_id"] = (
            base["symbol"].astype("string")
            + "|"
            + base["date"].dt.strftime("%Y%m%d").astype("string")
        )
    if base["event_id"].duplicated().any():
        raise ValueError("signals contains duplicate event_id values")

    # Configured aliases are normalized into the stable public names.  Drop an
    # alias only when it is distinct so signal factors remain otherwise intact.
    aliases_to_drop = {
        column
        for column in (event_id_column, signal_symbol_column, signal_date_column)
        if column in base.columns and column not in {"event_id", "symbol", "date"}
    }
    base = base.drop(columns=sorted(aliases_to_drop))

    daily_lookup = {
        (str(row["symbol"]), pd.Timestamp(row["date"])): row
        for _, row in prepared_daily.iterrows()
    }
    rows: list[dict[str, object]] = []
    for _, signal in base.iterrows():
        signal_date = pd.Timestamp(signal["date"])
        symbol = str(signal["symbol"])
        signal_position = int(calendar.searchsorted(signal_date, side="left"))
        entry_position = signal_position + 1
        has_next_market_date = entry_position < len(calendar)
        entry_date = (
            pd.Timestamp(calendar[entry_position]) if has_next_market_date else pd.NaT
        )
        entry_row = (
            daily_lookup.get((symbol, entry_date)) if has_next_market_date else None
        )
        event = signal.to_dict()

        for spec in selected_catalog:
            if spec.is_no_trade:
                rows.append(_no_trade_row(event, spec))
                continue
            assert spec.entry is not None
            entry_state = _entry_snapshot(
                entry_row,
                spec.entry,
                has_next_market_date=has_next_market_date,
            )
            cost = float(spec.round_trip_cost_bps) / 10_000.0
            outcome: dict[str, object]
            if not bool(entry_state["eligible"]):
                outcome = _incomplete_result("ineligible_entry", mae=np.nan)
            else:
                assert entry_row is not None
                outcome = _simulate_exit(
                    policy=spec.exit,
                    entry=spec.entry,
                    entry_row=entry_row,
                    entry_price=float(entry_state["entry_price"]),
                    entry_calendar_position=entry_position,
                    calendar=calendar,
                    daily_lookup=daily_lookup,
                    symbol=symbol,
                    cost=cost,
                )
            rows.append(
                {
                    **event,
                    "playbook_id": spec.playbook_id,
                    "playbook_version": spec.version,
                    "entry_mode": spec.entry.mode,
                    "exit_policy_id": spec.exit.policy_id,
                    **entry_state,
                    **outcome,
                    "entry_date": entry_date,
                    "round_trip_cost": cost,
                    "round_trip_cost_bps": float(spec.round_trip_cost_bps),
                }
            )

    extra_signal_columns = [
        column
        for column in base.columns
        if column not in {"event_id", "symbol", "date"}
    ]
    columns = [*_CORE_OUTPUT_COLUMNS, *extra_signal_columns]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["eligible"] = result["eligible"].astype(bool)
    result["mature"] = result["mature"].astype(bool)
    result["ambiguous_bar"] = result["ambiguous_bar"].astype(bool)
    result["holding_sessions"] = pd.array(result["holding_sessions"], dtype="Int64")
    result["entry_date"] = pd.to_datetime(result["entry_date"])
    result["exit_date"] = pd.to_datetime(result["exit_date"])
    return result.loc[:, columns].reset_index(drop=True)


__all__ = [
    "DEFAULT_CLOSE_ENTRY",
    "DEFAULT_EXIT_POLICIES",
    "DEFAULT_OPEN_ENTRY",
    "DEFAULT_PLAYBOOK_CATALOG",
    "DEFAULT_ROUND_TRIP_COST_BPS",
    "DEFAULT_TRADE_PLAYBOOKS",
    "EntryConstraint",
    "ExitPolicy",
    "FIXED_TP4_SL2_T5",
    "FIXED_TP6_SL3_T5",
    "NO_TRADE_PLAYBOOK",
    "NO_TRADE_PLAYBOOK_ID",
    "PLAYBOOK_POLICY_VERSION",
    "PlaybookSpec",
    "build_playbook_outcomes",
    "default_playbook_catalog",
    "playbook_catalog_hash",
    "serialize_playbook_catalog",
]
