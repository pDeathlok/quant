"""Causal chart-pattern confirmations for monthly low-zone anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


PATTERN_RULES = (
    "double_bottom_breakout",
    "inverse_head_shoulders_breakout",
)


@dataclass(frozen=True)
class ChartPatternConfig:
    """Frozen geometry and execution-time gates for causal bottom patterns."""

    pivot_radius_sessions: int = 3
    formation_lookback_sessions: int = 126
    maximum_wait_sessions: int = 126
    double_bottom_minimum_gap_sessions: int = 20
    double_bottom_maximum_gap_sessions: int = 90
    double_bottom_maximum_difference: float = 0.08
    minimum_neckline_rebound: float = 0.10
    head_shoulders_minimum_gap_sessions: int = 10
    head_shoulders_maximum_gap_sessions: int = 60
    shoulder_maximum_difference: float = 0.10
    head_minimum_depth: float = 0.08
    neckline_breakout_buffer: float = 0.01
    minimum_sessions_since_new_low: int = 20
    minimum_base_position: float = 0.50
    minimum_confirmation_amount_thousand: float = 30_000.0
    maximum_confirmation_drawdown_from_anchor_peak: float = -0.40

    def __post_init__(self) -> None:
        if self.pivot_radius_sessions < 1:
            raise ValueError("pivot_radius_sessions must be positive")
        if self.formation_lookback_sessions < 1:
            raise ValueError("formation_lookback_sessions must be positive")
        if self.maximum_wait_sessions < 1:
            raise ValueError("maximum_wait_sessions must be positive")
        if not (
            1
            <= self.double_bottom_minimum_gap_sessions
            <= self.double_bottom_maximum_gap_sessions
        ):
            raise ValueError("invalid double-bottom gap range")
        if not (
            1
            <= self.head_shoulders_minimum_gap_sessions
            <= self.head_shoulders_maximum_gap_sessions
        ):
            raise ValueError("invalid head-and-shoulders gap range")
        for name in (
            "double_bottom_maximum_difference",
            "minimum_neckline_rebound",
            "shoulder_maximum_difference",
            "head_minimum_depth",
            "neckline_breakout_buffer",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.minimum_sessions_since_new_low < 1:
            raise ValueError("minimum_sessions_since_new_low must be positive")
        if not 0.0 <= self.minimum_base_position <= 1.0:
            raise ValueError("minimum_base_position must be in [0, 1]")
        if self.minimum_confirmation_amount_thousand < 0.0:
            raise ValueError("minimum_confirmation_amount_thousand must be non-negative")
        if not -1.0 < self.maximum_confirmation_drawdown_from_anchor_peak < 0.0:
            raise ValueError(
                "maximum_confirmation_drawdown_from_anchor_peak must be in (-1, 0)"
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


def causal_pivot_lows(
    prices: pd.DataFrame,
    *,
    radius: int,
) -> pd.DataFrame:
    """Return strict local lows and the date each pivot first becomes knowable."""

    _require_columns(prices, {"date", "adjusted_low"}, "prices")
    if radius < 1:
        raise ValueError("radius must be positive")
    frame = prices[["date", "adjusted_low"]].copy().reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["adjusted_low"] = pd.to_numeric(frame["adjusted_low"], errors="coerce")
    rows: list[dict[str, Any]] = []
    values = frame["adjusted_low"].to_numpy(float)
    dates = pd.DatetimeIndex(frame["date"])
    for position in range(radius, len(frame) - radius):
        window = values[position - radius : position + radius + 1]
        value = values[position]
        if not np.isfinite(value) or not np.isfinite(window).all():
            continue
        minimum = float(window.min())
        if value != minimum or int(np.isclose(window, minimum).sum()) != 1:
            continue
        rows.append(
            {
                "position": position,
                "pivot_date": pd.Timestamp(dates[position]),
                "recognition_date": pd.Timestamp(dates[position + radius]),
                "pivot_price": value,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["position", "pivot_date", "recognition_date", "pivot_price"],
    )


def _calendar_gap(
    left: pd.Timestamp,
    right: pd.Timestamp,
    positions: dict[pd.Timestamp, int],
) -> int | None:
    left_position = positions.get(pd.Timestamp(left))
    right_position = positions.get(pd.Timestamp(right))
    if left_position is None or right_position is None:
        return None
    return right_position - left_position


def _confirmation_candidates(
    prices: pd.DataFrame,
    *,
    earliest_date: pd.Timestamp,
    end_date: pd.Timestamp,
    recognition_date: pd.Timestamp,
    neckline: float,
    anchor_peak: float,
    config: ChartPatternConfig,
) -> pd.DataFrame:
    start_date = max(pd.Timestamp(earliest_date), pd.Timestamp(recognition_date))
    confirmation_drawdown = prices["adjusted_close"] / anchor_peak - 1.0
    common = (
        prices["date"].between(start_date, end_date, inclusive="both")
        & prices["adjusted_close"].ge(
            neckline * (1.0 + config.neckline_breakout_buffer)
        )
        & prices["sessions_since_new_low"].ge(
            config.minimum_sessions_since_new_low
        )
        & prices["return_20d"].gt(0.0)
        & prices["base_position"].ge(config.minimum_base_position)
        & prices["prior_amount_median_20d"].ge(
            config.minimum_confirmation_amount_thousand
        )
        & confirmation_drawdown.le(
            config.maximum_confirmation_drawdown_from_anchor_peak
        )
    )
    return prices.loc[common].copy()


def _find_double_bottom(
    prices: pd.DataFrame,
    pivots: pd.DataFrame,
    *,
    anchor_date: pd.Timestamp,
    end_date: pd.Timestamp,
    anchor_peak: float,
    calendar_positions: dict[pd.Timestamp, int],
    config: ChartPatternConfig,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(pivots)), 2):
        left = pivots.iloc[left_index]
        right = pivots.iloc[right_index]
        gap = _calendar_gap(left["pivot_date"], right["pivot_date"], calendar_positions)
        if gap is None or not (
            config.double_bottom_minimum_gap_sessions
            <= gap
            <= config.double_bottom_maximum_gap_sessions
        ):
            continue
        lower = min(float(left["pivot_price"]), float(right["pivot_price"]))
        difference = abs(float(left["pivot_price"]) - float(right["pivot_price"])) / lower
        if difference > config.double_bottom_maximum_difference:
            continue
        between = prices.loc[
            prices["date"].between(
                left["pivot_date"], right["pivot_date"], inclusive="neither"
            )
        ]
        if between.empty:
            continue
        neckline = float(between["adjusted_close"].max())
        rebound = neckline / lower - 1.0
        if not np.isfinite(neckline) or rebound < config.minimum_neckline_rebound:
            continue
        candidates = _confirmation_candidates(
            prices,
            earliest_date=anchor_date,
            end_date=end_date,
            recognition_date=pd.Timestamp(right["recognition_date"]),
            neckline=neckline,
            anchor_peak=anchor_peak,
            config=config,
        )
        if candidates.empty:
            continue
        confirmation = candidates.iloc[0]
        matches.append(
            {
                "confirmation_date": pd.Timestamp(confirmation["date"]),
                "confirmation_close": float(confirmation["adjusted_close"]),
                "pattern_start_date": pd.Timestamp(left["pivot_date"]),
                "pattern_middle_date": pd.NaT,
                "pattern_end_date": pd.Timestamp(right["pivot_date"]),
                "pattern_start_price": float(left["pivot_price"]),
                "pattern_middle_price": np.nan,
                "pattern_end_price": float(right["pivot_price"]),
                "pattern_neckline": neckline,
                "pattern_rebound": rebound,
                "pattern_symmetry": difference,
                "pattern_head_depth": np.nan,
            }
        )
    if not matches:
        return None
    return min(
        matches,
        key=lambda match: (
            match["confirmation_date"],
            match["pattern_end_date"],
            match["pattern_start_date"],
        ),
    )


def _find_inverse_head_shoulders(
    prices: pd.DataFrame,
    pivots: pd.DataFrame,
    *,
    anchor_date: pd.Timestamp,
    end_date: pd.Timestamp,
    anchor_peak: float,
    calendar_positions: dict[pd.Timestamp, int],
    config: ChartPatternConfig,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for left_index, head_index, right_index in combinations(range(len(pivots)), 3):
        left = pivots.iloc[left_index]
        head = pivots.iloc[head_index]
        right = pivots.iloc[right_index]
        left_gap = _calendar_gap(
            left["pivot_date"], head["pivot_date"], calendar_positions
        )
        right_gap = _calendar_gap(
            head["pivot_date"], right["pivot_date"], calendar_positions
        )
        if left_gap is None or right_gap is None:
            continue
        if not (
            config.head_shoulders_minimum_gap_sessions
            <= left_gap
            <= config.head_shoulders_maximum_gap_sessions
            and config.head_shoulders_minimum_gap_sessions
            <= right_gap
            <= config.head_shoulders_maximum_gap_sessions
        ):
            continue
        left_price = float(left["pivot_price"])
        head_price = float(head["pivot_price"])
        right_price = float(right["pivot_price"])
        shoulder_difference = abs(left_price - right_price) / min(
            left_price, right_price
        )
        shoulder_mean = (left_price + right_price) / 2.0
        head_depth = 1.0 - head_price / shoulder_mean
        if shoulder_difference > config.shoulder_maximum_difference:
            continue
        if head_depth < config.head_minimum_depth:
            continue
        left_interval = prices.loc[
            prices["date"].between(
                left["pivot_date"], head["pivot_date"], inclusive="neither"
            )
        ]
        right_interval = prices.loc[
            prices["date"].between(
                head["pivot_date"], right["pivot_date"], inclusive="neither"
            )
        ]
        if left_interval.empty or right_interval.empty:
            continue
        neckline = max(
            float(left_interval["adjusted_close"].max()),
            float(right_interval["adjusted_close"].max()),
        )
        lower = min(left_price, head_price, right_price)
        rebound = neckline / lower - 1.0
        if not np.isfinite(neckline) or rebound < config.minimum_neckline_rebound:
            continue
        candidates = _confirmation_candidates(
            prices,
            earliest_date=anchor_date,
            end_date=end_date,
            recognition_date=pd.Timestamp(right["recognition_date"]),
            neckline=neckline,
            anchor_peak=anchor_peak,
            config=config,
        )
        if candidates.empty:
            continue
        confirmation = candidates.iloc[0]
        matches.append(
            {
                "confirmation_date": pd.Timestamp(confirmation["date"]),
                "confirmation_close": float(confirmation["adjusted_close"]),
                "pattern_start_date": pd.Timestamp(left["pivot_date"]),
                "pattern_middle_date": pd.Timestamp(head["pivot_date"]),
                "pattern_end_date": pd.Timestamp(right["pivot_date"]),
                "pattern_start_price": left_price,
                "pattern_middle_price": head_price,
                "pattern_end_price": right_price,
                "pattern_neckline": neckline,
                "pattern_rebound": rebound,
                "pattern_symmetry": shoulder_difference,
                "pattern_head_depth": head_depth,
            }
        )
    if not matches:
        return None
    return min(
        matches,
        key=lambda match: (
            match["confirmation_date"],
            match["pattern_end_date"],
            match["pattern_start_date"],
        ),
    )


def generate_chart_pattern_signals(
    daily_features: pd.DataFrame,
    monthly_anchors: pd.DataFrame,
    market_calendar: pd.DatetimeIndex,
    config: ChartPatternConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return first causal bottom-pattern breakout per monthly low-zone anchor."""

    required_daily = {
        "ts_code",
        "date",
        "adjusted_low",
        "adjusted_close",
        "prior_amount_median_20d",
        "sessions_since_new_low",
        "return_20d",
        "base_position",
    }
    required_anchors = {
        "ts_code",
        "signal_date",
        "adjusted_close",
        "prior_peak",
        "drawdown_from_prior_peak",
    }
    _require_columns(daily_features, required_daily, "daily_features")
    _require_columns(monthly_anchors, required_anchors, "monthly_anchors")
    calendar = _normalize_calendar(market_calendar)
    calendar_positions = {
        pd.Timestamp(date): position for position, date in enumerate(calendar)
    }
    daily = daily_features.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    for column in required_daily - {"ts_code", "date"}:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = (
        daily.dropna(subset=["ts_code", "date"])
        .sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
    )
    daily_by_symbol = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in daily.groupby("ts_code", observed=True, sort=False)
    }
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
        anchor_date = pd.Timestamp(anchor["signal_date"])
        anchor_position = calendar_positions.get(anchor_date)
        symbol = str(anchor["ts_code"])
        prices = daily_by_symbol.get(symbol)
        matches: dict[str, dict[str, Any] | None] = {
            rule: None for rule in PATTERN_RULES
        }
        status = "expired"
        if anchor_position is None:
            status = "missing_anchor_calendar"
        elif prices is None or prices.empty:
            status = "missing_symbol_prices"
        else:
            start_position = max(
                0, anchor_position - config.formation_lookback_sessions
            )
            end_position = min(
                len(calendar) - 1, anchor_position + config.maximum_wait_sessions
            )
            start_date = pd.Timestamp(calendar[start_position])
            end_date = pd.Timestamp(calendar[end_position])
            formation = prices.loc[
                prices["date"].between(start_date, end_date, inclusive="both")
            ].copy().reset_index(drop=True)
            if not formation.empty:
                pivots = causal_pivot_lows(
                    formation, radius=config.pivot_radius_sessions
                )
                anchor_peak = float(anchor["prior_peak"])
                if np.isfinite(anchor_peak) and anchor_peak > 0.0:
                    matches["double_bottom_breakout"] = _find_double_bottom(
                        formation,
                        pivots,
                        anchor_date=anchor_date,
                        end_date=end_date,
                        anchor_peak=anchor_peak,
                        calendar_positions=calendar_positions,
                        config=config,
                    )
                    matches["inverse_head_shoulders_breakout"] = (
                        _find_inverse_head_shoulders(
                            formation,
                            pivots,
                            anchor_date=anchor_date,
                            end_date=end_date,
                            anchor_peak=anchor_peak,
                            calendar_positions=calendar_positions,
                            config=config,
                        )
                    )
        for rule, match in matches.items():
            if match is None:
                diagnostic_rows.append(
                    {
                        "anchor_id": anchor_id,
                        "ts_code": symbol,
                        "anchor_date": anchor_date,
                        "rule": rule,
                        "confirmation_status": status,
                        "confirmation_date": pd.NaT,
                        "confirmation_wait_sessions": np.nan,
                    }
                )
                continue
            confirmation_date = pd.Timestamp(match["confirmation_date"])
            confirmation_position = calendar_positions[confirmation_date]
            wait_sessions = confirmation_position - int(anchor_position)
            anchor_close = float(anchor["adjusted_close"])
            path = prices.loc[
                prices["date"].between(
                    anchor_date, confirmation_date, inclusive="both"
                )
            ]
            waiting_drawdown = (
                float(path["adjusted_low"].min() / anchor_close - 1.0)
                if not path.empty and anchor_close > 0.0
                else np.nan
            )
            payload = anchor.to_dict()
            payload.update(match)
            payload.update(
                {
                    "_pattern_key": f"{anchor_id}|{rule}",
                    "anchor_id": anchor_id,
                    "anchor_date": anchor_date,
                    "anchor_close": anchor_close,
                    "anchor_prior_peak": float(anchor["prior_peak"]),
                    "anchor_drawdown_from_prior_peak": float(
                        anchor["drawdown_from_prior_peak"]
                    ),
                    "rule": rule,
                    "signal_date": confirmation_date,
                    "confirmation_drawdown_from_anchor_peak": (
                        float(match["confirmation_close"])
                        / float(anchor["prior_peak"])
                        - 1.0
                    ),
                    "confirmation_wait_sessions": wait_sessions,
                    "waiting_path_drawdown": waiting_drawdown,
                }
            )
            signal_rows.append(payload)
            diagnostic_rows.append(
                {
                    "_pattern_key": f"{anchor_id}|{rule}",
                    "anchor_id": anchor_id,
                    "ts_code": symbol,
                    "anchor_date": anchor_date,
                    "rule": rule,
                    "confirmation_status": "confirmed",
                    "confirmation_date": confirmation_date,
                    "confirmation_wait_sessions": wait_sessions,
                }
            )
    signals = pd.DataFrame(signal_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    if signals.empty:
        signals = pd.DataFrame(columns=["signal_id", "ts_code", "signal_date", "rule"])
        diagnostics["signal_id"] = pd.Series(pd.NA, index=diagnostics.index, dtype="Int64")
        return signals, diagnostics.drop(columns=["_pattern_key"], errors="ignore")
    signals = signals.sort_values(["signal_date", "ts_code", "rule"]).reset_index(
        drop=True
    )
    signals["signal_id"] = np.arange(1, len(signals) + 1, dtype=np.int64)
    lookup = signals.set_index("_pattern_key")["signal_id"].to_dict()
    diagnostics["signal_id"] = diagnostics.get("_pattern_key", pd.Series()).map(
        lookup
    ).astype("Int64")
    return (
        signals.drop(columns="_pattern_key"),
        diagnostics.drop(columns="_pattern_key", errors="ignore"),
    )
