"""Causal confirmation state machine for completed monthly low-zone anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from quant.research.monthly_low_zone import EVENT_PERIODS


PRICE_CONFIRMATION_RULES = (
    "anchor_direct",
    "no_new_low_20",
    "range_mid_reclaim",
    "range_mid_relative",
    "range_mid_weekly",
    "range_mid_relative_weekly",
    "confirmed_market",
    "confirmed_market_exhaustion",
    "breadth_repair",
    "breadth_relative",
    "breadth_relative_weekly",
    "breadth_relative_weekly_exhaustion",
)
SURVIVAL_CONFIRMATION_RULES = (
    "breadth_relative_weekly_survival",
    "breadth_relative_weekly_exhaustion_survival",
)
CONFIRMATION_RULES = (*PRICE_CONFIRMATION_RULES, *SURVIVAL_CONFIRMATION_RULES)


@dataclass(frozen=True)
class MonthlyConfirmationConfig:
    """Frozen gates for confirming a monthly low-zone anchor."""

    minimum_wait_sessions: int = 5
    maximum_wait_sessions: int = 126
    minimum_sessions_since_new_low: int = 20
    minimum_confirmation_amount_thousand: float = 30_000.0
    maximum_confirmation_drawdown_from_anchor_peak: float = -0.40
    weekly_j_repair_threshold: float = -10.0
    benchmark_ma_sessions: int = 120
    minimum_breadth_constituents: int = 500
    minimum_breadth_positive_share: float = 0.55
    round_trip_cost_bps: float = 20.0
    horizons: tuple[int, int, int] = (126, 252, 504)

    def __post_init__(self) -> None:
        if self.minimum_wait_sessions < 1:
            raise ValueError("minimum_wait_sessions must be positive")
        if self.maximum_wait_sessions < self.minimum_wait_sessions:
            raise ValueError("maximum_wait_sessions must cover the minimum wait")
        if self.minimum_sessions_since_new_low < 1:
            raise ValueError("minimum_sessions_since_new_low must be positive")
        if self.minimum_confirmation_amount_thousand < 0.0:
            raise ValueError("minimum_confirmation_amount_thousand must be non-negative")
        if not -1.0 < self.maximum_confirmation_drawdown_from_anchor_peak < 0.0:
            raise ValueError(
                "maximum_confirmation_drawdown_from_anchor_peak must be in (-1, 0)"
            )
        if self.benchmark_ma_sessions < 2:
            raise ValueError("benchmark_ma_sessions must be at least two")
        if self.minimum_breadth_constituents < 1:
            raise ValueError("minimum_breadth_constituents must be positive")
        if not 0.0 < self.minimum_breadth_positive_share <= 1.0:
            raise ValueError("minimum_breadth_positive_share must be in (0, 1]")
        if self.round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps must be non-negative")
        if len(self.horizons) != 3 or tuple(sorted(self.horizons)) != self.horizons:
            raise ValueError("horizons must contain three increasing session counts")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    compact = pd.to_datetime(
        values.astype("string").str.replace(r"\.0$", "", regex=True).str[:8],
        format="%Y%m%d",
        errors="coerce",
    )
    return compact.fillna(parsed).dt.normalize()


def _normalize_calendar(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce")).dropna()
    values = pd.DatetimeIndex(sorted(values.normalize().unique()))
    if values.empty:
        raise ValueError("market_calendar must not be empty")
    return values


def build_benchmark_confirmation_features(
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Return causal 20-session return and moving-average market state."""

    _require_columns(benchmark, {"close"}, "benchmark")
    if "trade_date" in benchmark:
        date_values = benchmark["trade_date"]
    elif "date" in benchmark:
        date_values = benchmark["date"]
    else:
        raise ValueError("benchmark missing columns: ['trade_date or date']")
    out = benchmark.copy()
    out["date"] = _parse_dates(date_values)
    out["benchmark_close"] = pd.to_numeric(out["close"], errors="coerce")
    out = (
        out.dropna(subset=["date", "benchmark_close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    out["benchmark_return_20d"] = out["benchmark_close"].pct_change(
        20, fill_method=None
    )
    out["benchmark_ma120"] = out["benchmark_close"].rolling(
        120, min_periods=120
    ).mean()
    return out[
        ["date", "benchmark_close", "benchmark_return_20d", "benchmark_ma120"]
    ]


def build_market_breadth_features(
    daily_features: pd.DataFrame,
    config: MonthlyConfirmationConfig,
) -> pd.DataFrame:
    """Return same-date liquid-universe median return and positive share."""

    _require_columns(
        daily_features,
        {"date", "return_20d", "prior_amount_median_20d"},
        "daily_features",
    )
    panel = daily_features[["date", "return_20d", "prior_amount_median_20d"]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["return_20d"] = pd.to_numeric(panel["return_20d"], errors="coerce")
    panel["prior_amount_median_20d"] = pd.to_numeric(
        panel["prior_amount_median_20d"], errors="coerce"
    )
    panel = panel.loc[
        panel["date"].notna()
        & panel["return_20d"].notna()
        & panel["prior_amount_median_20d"].ge(
            config.minimum_confirmation_amount_thousand
        )
    ].copy()
    panel["positive"] = panel["return_20d"].gt(0.0).astype(float)
    breadth = (
        panel.groupby("date", as_index=False, observed=True)
        .agg(
            breadth_constituents=("return_20d", "count"),
            breadth_median_return_20d=("return_20d", "median"),
            breadth_positive_share_20d=("positive", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    insufficient = breadth["breadth_constituents"].lt(
        config.minimum_breadth_constituents
    )
    breadth.loc[
        insufficient,
        ["breadth_median_return_20d", "breadth_positive_share_20d"],
    ] = np.nan
    return breadth


def _weekly_state_for_dates(
    weekly: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    state = pd.DataFrame(
        {
            "weekly_available_date": pd.NaT,
            "weekly_j": np.nan,
            "weekly_prev_j": np.nan,
        },
        index=np.arange(len(dates)),
    )
    if weekly is None or weekly.empty or dates.empty:
        return state
    available_dates = pd.DatetimeIndex(weekly["weekly_available_date"])
    positions = available_dates.searchsorted(dates, side="right") - 1
    valid = positions >= 0
    if valid.any():
        selected = weekly.iloc[positions[valid]]
        state.loc[valid, "weekly_available_date"] = selected[
            "weekly_available_date"
        ].to_numpy()
        state.loc[valid, "weekly_j"] = pd.to_numeric(
            selected["weekly_j"], errors="coerce"
        ).to_numpy()
        state.loc[valid, "weekly_prev_j"] = pd.to_numeric(
            selected["weekly_prev_j"], errors="coerce"
        ).to_numpy()
    return state


def _anchor_signal_payload(
    anchor: pd.Series,
    *,
    anchor_id: int,
    rule: str,
    confirmation_date: pd.Timestamp,
    confirmation_close: float,
    confirmation_drawdown: float,
    wait_sessions: int,
    waiting_path_drawdown: float,
    weekly_available_date: pd.Timestamp | pd.NaT,
    weekly_j: float,
    weekly_prev_j: float,
) -> dict[str, Any]:
    payload = anchor.to_dict()
    payload.update(
        {
            "_confirmation_key": f"{anchor_id}|{rule}",
            "anchor_id": anchor_id,
            "anchor_rule": anchor.get("rule"),
            "anchor_date": pd.Timestamp(anchor["signal_date"]),
            "anchor_close": float(anchor.get("adjusted_close", np.nan)),
            "anchor_prior_peak": float(anchor.get("prior_peak", np.nan)),
            "anchor_drawdown_from_prior_peak": float(
                anchor.get("drawdown_from_prior_peak", np.nan)
            ),
            "rule": rule,
            "signal_date": confirmation_date,
            "confirmation_close": confirmation_close,
            "confirmation_drawdown_from_anchor_peak": confirmation_drawdown,
            "confirmation_wait_sessions": wait_sessions,
            "waiting_path_drawdown": waiting_path_drawdown,
            "weekly_available_date": weekly_available_date,
            "weekly_j": weekly_j,
            "weekly_prev_j": weekly_prev_j,
        }
    )
    return payload


def generate_monthly_confirmation_signals(
    daily_features: pd.DataFrame,
    weekly_features: pd.DataFrame,
    monthly_anchors: pd.DataFrame,
    benchmark_features: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: MonthlyConfirmationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return first confirmation per anchor/rule and expiry diagnostics."""

    required_daily = {
        "ts_code",
        "date",
        "adjusted_close",
        "adjusted_low",
        "prior_amount_median_20d",
        "sessions_since_new_low",
        "return_20d",
        "base_position",
        "down_amount_share_ratio",
        "volatility_contraction_ratio",
    }
    required_anchors = {
        "ts_code",
        "signal_date",
        "rule",
        "adjusted_close",
        "prior_peak",
        "drawdown_from_prior_peak",
    }
    required_benchmark = {
        "date",
        "benchmark_close",
        "benchmark_return_20d",
        "benchmark_ma120",
    }
    _require_columns(daily_features, required_daily, "daily_features")
    _require_columns(monthly_anchors, required_anchors, "monthly_anchors")
    _require_columns(benchmark_features, required_benchmark, "benchmark_features")
    if not weekly_features.empty:
        _require_columns(
            weekly_features,
            {"ts_code", "weekly_available_date", "weekly_j", "weekly_prev_j"},
            "weekly_features",
        )

    calendar = _normalize_calendar(market_calendar)
    calendar_positions = {
        pd.Timestamp(date): position for position, date in enumerate(calendar)
    }
    daily = daily_features.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = (
        daily.dropna(subset=["ts_code", "date"])
        .sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
    )
    daily_by_symbol = {
        str(code): group.reset_index(drop=True)
        for code, group in daily.groupby("ts_code", observed=True, sort=False)
    }
    weekly_by_symbol: dict[str, pd.DataFrame] = {}
    if not weekly_features.empty:
        weekly = weekly_features.copy()
        weekly["weekly_available_date"] = pd.to_datetime(
            weekly["weekly_available_date"], errors="coerce"
        ).dt.normalize()
        weekly = weekly.dropna(subset=["ts_code", "weekly_available_date"])
        weekly_by_symbol = {
            str(code): group.sort_values("weekly_available_date").reset_index(drop=True)
            for code, group in weekly.groupby("ts_code", observed=True, sort=False)
        }
    benchmark = benchmark_features.copy()
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="coerce"
    ).dt.normalize()
    for column in (
        "breadth_constituents",
        "breadth_median_return_20d",
        "breadth_positive_share_20d",
    ):
        if column not in benchmark:
            benchmark[column] = np.nan
    benchmark = benchmark.drop_duplicates("date", keep="last").set_index("date")

    anchors = monthly_anchors.copy().reset_index(drop=True)
    anchors["signal_date"] = pd.to_datetime(
        anchors["signal_date"], errors="coerce"
    ).dt.normalize()
    if "signal_id" not in anchors:
        anchors["signal_id"] = np.arange(1, len(anchors) + 1, dtype=np.int64)

    signal_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for _, anchor in anchors.iterrows():
        anchor_id = int(anchor["signal_id"])
        symbol = str(anchor["ts_code"])
        anchor_date = pd.Timestamp(anchor["signal_date"])
        anchor_position = calendar_positions.get(anchor_date)
        anchor_close = float(anchor["adjusted_close"])
        anchor_peak = float(anchor["prior_peak"])
        direct_drawdown = anchor_close / anchor_peak - 1.0
        direct_payload = _anchor_signal_payload(
            anchor,
            anchor_id=anchor_id,
            rule="anchor_direct",
            confirmation_date=anchor_date,
            confirmation_close=anchor_close,
            confirmation_drawdown=direct_drawdown,
            wait_sessions=0,
            waiting_path_drawdown=0.0,
            weekly_available_date=pd.NaT,
            weekly_j=float(anchor.get("weekly_j", np.nan)),
            weekly_prev_j=np.nan,
        )
        signal_rows.append(direct_payload)
        diagnostic_rows.append(
            {
                "_confirmation_key": f"{anchor_id}|anchor_direct",
                "anchor_id": anchor_id,
                "ts_code": symbol,
                "anchor_rule": anchor.get("rule"),
                "anchor_date": anchor_date,
                "rule": "anchor_direct",
                "confirmation_status": "confirmed",
                "confirmation_date": anchor_date,
                "confirmation_wait_sessions": 0,
                "waiting_path_drawdown": 0.0,
            }
        )

        prices = daily_by_symbol.get(symbol)
        empty_status = "missing_anchor_calendar" if anchor_position is None else "expired"
        candidate_frame = pd.DataFrame()
        if prices is not None and not prices.empty and anchor_position is not None:
            start_position = anchor_position + config.minimum_wait_sessions
            end_position = min(
                anchor_position + config.maximum_wait_sessions,
                len(calendar) - 1,
            )
            if start_position <= end_position:
                start_date = pd.Timestamp(calendar[start_position])
                end_date = pd.Timestamp(calendar[end_position])
                candidate_frame = prices.loc[
                    prices["date"].between(start_date, end_date, inclusive="both")
                ].copy()
        if candidate_frame.empty:
            for rule in PRICE_CONFIRMATION_RULES[1:]:
                diagnostic_rows.append(
                    {
                        "_confirmation_key": f"{anchor_id}|{rule}",
                        "anchor_id": anchor_id,
                        "ts_code": symbol,
                        "anchor_rule": anchor.get("rule"),
                        "anchor_date": anchor_date,
                        "rule": rule,
                        "confirmation_status": empty_status,
                        "confirmation_date": pd.NaT,
                        "confirmation_wait_sessions": np.nan,
                        "waiting_path_drawdown": np.nan,
                    }
                )
            continue

        candidate_frame = candidate_frame.sort_values("date").reset_index(drop=True)
        candidate_dates = pd.DatetimeIndex(candidate_frame["date"])
        candidate_frame["confirmation_wait_sessions"] = [
            calendar_positions.get(pd.Timestamp(date), -10**9) - anchor_position
            for date in candidate_dates
        ]
        benchmark_state = benchmark.reindex(candidate_dates).reset_index(drop=True)
        for column in (
            *(required_benchmark - {"date"}),
            "breadth_constituents",
            "breadth_median_return_20d",
            "breadth_positive_share_20d",
        ):
            candidate_frame[column] = pd.to_numeric(
                benchmark_state[column], errors="coerce"
            ).to_numpy()
        weekly_state = _weekly_state_for_dates(
            weekly_by_symbol.get(symbol), candidate_dates
        )
        for column in ("weekly_available_date", "weekly_j", "weekly_prev_j"):
            candidate_frame[column] = weekly_state[column].to_numpy()

        numeric_columns = (
            "adjusted_close",
            "adjusted_low",
            "prior_amount_median_20d",
            "sessions_since_new_low",
            "return_20d",
            "base_position",
            "down_amount_share_ratio",
            "volatility_contraction_ratio",
            "benchmark_close",
            "benchmark_return_20d",
            "benchmark_ma120",
            "weekly_j",
            "weekly_prev_j",
            "breadth_constituents",
            "breadth_median_return_20d",
            "breadth_positive_share_20d",
        )
        for column in numeric_columns:
            candidate_frame[column] = pd.to_numeric(
                candidate_frame[column], errors="coerce"
            )
        candidate_frame["confirmation_drawdown_from_anchor_peak"] = (
            candidate_frame["adjusted_close"] / anchor_peak - 1.0
        )
        common = (
            candidate_frame["prior_amount_median_20d"].ge(
                config.minimum_confirmation_amount_thousand
            )
            & candidate_frame["confirmation_drawdown_from_anchor_peak"].le(
                config.maximum_confirmation_drawdown_from_anchor_peak
            )
        )
        no_new_low = (
            common
            & candidate_frame["sessions_since_new_low"].ge(
                config.minimum_sessions_since_new_low
            )
            & candidate_frame["return_20d"].gt(0.0)
        )
        range_mid = no_new_low & candidate_frame["base_position"].ge(0.50)
        relative = range_mid & candidate_frame["return_20d"].gt(
            candidate_frame["benchmark_return_20d"]
        )
        weekly_repair = (
            candidate_frame["weekly_j"].gt(config.weekly_j_repair_threshold)
            & candidate_frame["weekly_j"].gt(candidate_frame["weekly_prev_j"])
        )
        range_weekly = range_mid & weekly_repair
        relative_weekly = relative & weekly_repair
        market_repair = (
            candidate_frame["benchmark_close"].ge(candidate_frame["benchmark_ma120"])
            & candidate_frame["benchmark_return_20d"].gt(0.0)
        )
        confirmed_market = relative_weekly & market_repair
        exhaustion = (
            candidate_frame["down_amount_share_ratio"].le(1.0)
            & candidate_frame["volatility_contraction_ratio"].le(1.0)
        )
        breadth_repaired = (
            candidate_frame["breadth_constituents"].ge(
                config.minimum_breadth_constituents
            )
            & candidate_frame["breadth_median_return_20d"].gt(0.0)
            & candidate_frame["breadth_positive_share_20d"].ge(
                config.minimum_breadth_positive_share
            )
        )
        breadth_repair = range_mid & breadth_repaired
        breadth_relative = breadth_repair & candidate_frame["return_20d"].gt(
            candidate_frame["breadth_median_return_20d"]
        )
        breadth_relative_weekly = breadth_relative & weekly_repair
        rule_masks = {
            "no_new_low_20": no_new_low,
            "range_mid_reclaim": range_mid,
            "range_mid_relative": relative,
            "range_mid_weekly": range_weekly,
            "range_mid_relative_weekly": relative_weekly,
            "confirmed_market": confirmed_market,
            "confirmed_market_exhaustion": confirmed_market & exhaustion,
            "breadth_repair": breadth_repair,
            "breadth_relative": breadth_relative,
            "breadth_relative_weekly": breadth_relative_weekly,
            "breadth_relative_weekly_exhaustion": (
                breadth_relative_weekly & exhaustion
            ),
        }
        path = prices.loc[
            prices["date"].between(anchor_date, candidate_dates[-1], inclusive="both")
        ].copy()
        for rule in PRICE_CONFIRMATION_RULES[1:]:
            matches = candidate_frame.loc[rule_masks[rule].fillna(False)]
            if matches.empty:
                diagnostic_rows.append(
                    {
                        "_confirmation_key": f"{anchor_id}|{rule}",
                        "anchor_id": anchor_id,
                        "ts_code": symbol,
                        "anchor_rule": anchor.get("rule"),
                        "anchor_date": anchor_date,
                        "rule": rule,
                        "confirmation_status": "expired",
                        "confirmation_date": pd.NaT,
                        "confirmation_wait_sessions": np.nan,
                        "waiting_path_drawdown": np.nan,
                    }
                )
                continue
            match = matches.iloc[0]
            confirmation_date = pd.Timestamp(match["date"])
            wait_sessions = int(match["confirmation_wait_sessions"])
            waiting_path = path.loc[path["date"].le(confirmation_date)]
            waiting_path_drawdown = float(
                waiting_path["adjusted_low"].min() / anchor_close - 1.0
            )
            payload = _anchor_signal_payload(
                anchor,
                anchor_id=anchor_id,
                rule=rule,
                confirmation_date=confirmation_date,
                confirmation_close=float(match["adjusted_close"]),
                confirmation_drawdown=float(
                    match["confirmation_drawdown_from_anchor_peak"]
                ),
                wait_sessions=wait_sessions,
                waiting_path_drawdown=waiting_path_drawdown,
                weekly_available_date=match["weekly_available_date"],
                weekly_j=float(match["weekly_j"]),
                weekly_prev_j=float(match["weekly_prev_j"]),
            )
            for column in (
                "return_20d",
                "base_position",
                "down_amount_share_ratio",
                "volatility_contraction_ratio",
                "benchmark_return_20d",
                "breadth_constituents",
                "breadth_median_return_20d",
                "breadth_positive_share_20d",
            ):
                payload[column] = match[column]
            signal_rows.append(payload)
            diagnostic_rows.append(
                {
                    "_confirmation_key": f"{anchor_id}|{rule}",
                    "anchor_id": anchor_id,
                    "ts_code": symbol,
                    "anchor_rule": anchor.get("rule"),
                    "anchor_date": anchor_date,
                    "rule": rule,
                    "confirmation_status": "confirmed",
                    "confirmation_date": confirmation_date,
                    "confirmation_wait_sessions": wait_sessions,
                    "waiting_path_drawdown": waiting_path_drawdown,
                }
            )

    signals = pd.DataFrame(signal_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if signals.empty:
        signals = pd.DataFrame(columns=["signal_id", "ts_code", "signal_date", "rule"])
        diagnostics["signal_id"] = pd.Series(dtype="Int64")
        return signals, diagnostics
    signals = signals.sort_values(["signal_date", "ts_code", "rule"]).reset_index(
        drop=True
    )
    signals["signal_id"] = np.arange(1, len(signals) + 1, dtype=np.int64)
    signal_ids = signals.set_index("_confirmation_key")["signal_id"].to_dict()
    diagnostics["signal_id"] = diagnostics["_confirmation_key"].map(signal_ids).astype(
        "Int64"
    )
    signals = signals.drop(columns="_confirmation_key")
    diagnostics = diagnostics.drop(columns="_confirmation_key")
    return signals, diagnostics


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return np.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    spread = z * sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (center - spread) / denominator


def _date_cluster_summary(frame: pd.DataFrame) -> tuple[float, float, float]:
    cohorts = frame.groupby("signal_date", observed=True)["net_return"].mean().dropna()
    if cohorts.empty:
        return np.nan, np.nan, np.nan
    mean = float(cohorts.mean())
    if len(cohorts) < 2:
        return mean, np.nan, np.nan
    standard_error = float(cohorts.std(ddof=1) / sqrt(len(cohorts)))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def summarize_confirmation_events(
    events: pd.DataFrame,
    anchor_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return period/rule/horizon metrics with clustered uncertainty."""

    _require_columns(
        events,
        {
            "signal_id",
            "rule",
            "signal_date",
            "horizon",
            "entry_status",
            "outcome_completed",
            "net_return",
            "excess_net_return",
            "mae",
            "mfe",
            "exit_reason",
        },
        "events",
    )
    _require_columns(
        anchor_diagnostics,
        {
            "signal_id",
            "anchor_id",
            "anchor_date",
            "rule",
            "confirmation_status",
            "confirmation_wait_sessions",
            "waiting_path_drawdown",
        },
        "anchor_diagnostics",
    )
    diagnostics = anchor_diagnostics.copy()
    diagnostics["anchor_date"] = pd.to_datetime(
        diagnostics["anchor_date"], errors="coerce"
    )
    event_frame = events.copy()
    event_frame["signal_date"] = pd.to_datetime(
        event_frame["signal_date"], errors="coerce"
    )
    confirmed_lookup = diagnostics.loc[
        diagnostics["confirmation_status"].eq("confirmed") & diagnostics["signal_id"].notna(),
        ["signal_id", "anchor_date"],
    ].drop_duplicates("signal_id")
    event_frame = event_frame.merge(
        confirmed_lookup,
        on="signal_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    horizons = sorted(pd.to_numeric(event_frame["horizon"], errors="coerce").dropna().unique())
    for period, (start, end) in EVENT_PERIODS.items():
        period_diagnostics = diagnostics.loc[
            diagnostics["anchor_date"].between(start, end, inclusive="both")
        ]
        period_events = event_frame.loc[
            event_frame["anchor_date"].between(start, end, inclusive="both")
        ]
        for rule in CONFIRMATION_RULES:
            rule_diagnostics = period_diagnostics.loc[
                period_diagnostics["rule"].eq(rule)
            ]
            confirmed_diagnostics = rule_diagnostics.loc[
                rule_diagnostics["confirmation_status"].eq("confirmed")
            ]
            for horizon in horizons:
                all_events = period_events.loc[
                    period_events["rule"].eq(rule)
                    & period_events["horizon"].eq(horizon)
                ]
                accepted = all_events.loc[all_events["entry_status"].eq("accepted")]
                completed = accepted.loc[accepted["outcome_completed"].fillna(False)]
                returns = pd.to_numeric(completed["net_return"], errors="coerce").dropna()
                excess = pd.to_numeric(
                    completed["excess_net_return"], errors="coerce"
                ).dropna()
                wins = int(returns.gt(0.0).sum())
                losses = float(-returns.loc[returns <= 0.0].sum())
                cluster_mean, cluster_low, cluster_high = _date_cluster_summary(completed)
                rows.append(
                    {
                        "period": period,
                        "rule": rule,
                        "horizon": int(horizon),
                        "anchors": int(rule_diagnostics["anchor_id"].nunique()),
                        "confirmed_anchors": int(
                            confirmed_diagnostics["anchor_id"].nunique()
                        ),
                        "confirmation_rate": float(
                            confirmed_diagnostics["anchor_id"].nunique()
                            / rule_diagnostics["anchor_id"].nunique()
                        )
                        if rule_diagnostics["anchor_id"].nunique()
                        else np.nan,
                        "median_confirmation_wait_sessions": float(
                            pd.to_numeric(
                                confirmed_diagnostics["confirmation_wait_sessions"],
                                errors="coerce",
                            ).median()
                        )
                        if not confirmed_diagnostics.empty
                        else np.nan,
                        "median_waiting_path_drawdown": float(
                            pd.to_numeric(
                                confirmed_diagnostics["waiting_path_drawdown"],
                                errors="coerce",
                            ).median()
                        )
                        if not confirmed_diagnostics.empty
                        else np.nan,
                        "accepted_entries": int(len(accepted)),
                        "completed_events": int(len(completed)),
                        "symbols": int(completed["ts_code"].nunique())
                        if "ts_code" in completed
                        else 0,
                        "signal_dates": int(completed["signal_date"].nunique()),
                        "win_rate": wins / len(returns) if len(returns) else np.nan,
                        "win_rate_wilson_lower_95": _wilson_lower(wins, len(returns)),
                        "mean_net_return": float(returns.mean())
                        if len(returns)
                        else np.nan,
                        "median_net_return": float(returns.median())
                        if len(returns)
                        else np.nan,
                        "profit_factor": float(
                            returns.loc[returns > 0.0].sum() / losses
                        )
                        if losses > 0.0
                        else np.nan,
                        "mean_excess_net_return": float(excess.mean())
                        if len(excess)
                        else np.nan,
                        "excess_win_rate": float(excess.gt(0.0).mean())
                        if len(excess)
                        else np.nan,
                        "mean_mae": float(
                            pd.to_numeric(completed["mae"], errors="coerce").mean()
                        )
                        if len(completed)
                        else np.nan,
                        "mean_mfe": float(
                            pd.to_numeric(completed["mfe"], errors="coerce").mean()
                        )
                        if len(completed)
                        else np.nan,
                        "writeoff_rate": float(
                            completed["exit_reason"].eq("missing_bar_writeoff").mean()
                        )
                        if len(completed)
                        else np.nan,
                        "date_equal_mean_net_return": cluster_mean,
                        "date_cluster_ci95_low": cluster_low,
                        "date_cluster_ci95_high": cluster_high,
                    }
                )
    return pd.DataFrame(rows)
