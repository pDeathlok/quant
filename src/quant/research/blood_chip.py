"""Point-in-time price-volume research for suspected A-share fire sales.

The module deliberately separates three questions:

1. Did price, turnover and price impact look like constrained selling?
2. Did the marginal downside impact subsequently decay while price absorbed volume?
3. Would a causal next-open, T+1 portfolio have survived realistic costs and stops?

The output is a research ranking and backtest, not proof of seller identity or a
standalone investment recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "vol",
    "amount",
]

CASE_FEATURES = [
    "return_120d",
    "volatility_60d",
    "market_return_60d",
    "shock_score",
    "absorption_score",
    "impact_decay",
    "rebound_from_event_low",
    "clv_3d",
    "shock_realized_volatility_5d",
    "prior_realized_volatility_20d",
    "shock_volatility_expansion_ratio",
    "confirmation_realized_volatility_3d",
    "volatility_decay_ratio",
    "confirmation_volatility_vs_prior_ratio",
    "shock_range_expansion_ratio",
    "range_decay_ratio",
    "shock_amount_expansion_ratio",
    "amount_decay_ratio",
    "confirmation_amount_vs_prior_ratio",
    "downside_amount_decay_ratio",
    "confirmation_downside_amount_share",
    "downside_amount_share_decay_ratio",
    "sell_pressure_decay_ratio",
    "event_low_age_sessions",
    "event_close_location",
    "shock_to_signal_residual_return",
    "shock_kdj_daily_j",
    "shock_kdj_weekly_j",
    "shock_kdj_monthly_j",
    "shock_kdj_negative_count",
    "confirmation_kdj_daily_j",
    "confirmation_kdj_weekly_j",
    "confirmation_kdj_monthly_j",
]


@dataclass(frozen=True)
class BloodChipSignalConfig:
    """Frozen gates for pressure and absorption signals."""

    minimum_history_days: int = 120
    minimum_prior_amount_thousand: float = 30_000.0
    shock_score_threshold: float = 0.80
    maximum_residual_5d_percentile: float = 0.05
    minimum_amount_ratio_5d: float = 1.25
    minimum_impact_ratio_5d: float = 1.25
    event_quiet_days: int = 5
    minimum_absorption_day: int = 2
    maximum_absorption_day: int = 10
    minimum_rebound_from_event_low: float = 0.02
    maximum_impact_decay: float = 0.90
    minimum_residual_3d: float = 0.0
    minimum_clv_3d: float = -0.25
    absorption_score_threshold: float = 0.60
    minimum_return_120d: float | None = None
    maximum_return_120d: float | None = None
    maximum_volatility_60d: float | None = None
    minimum_market_return_60d: float | None = None

    def __post_init__(self) -> None:
        if self.minimum_history_days < 60:
            raise ValueError("minimum_history_days must be at least 60")
        if self.minimum_prior_amount_thousand <= 0:
            raise ValueError("minimum_prior_amount_thousand must be positive")
        for field_name in (
            "shock_score_threshold",
            "maximum_residual_5d_percentile",
            "absorption_score_threshold",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.minimum_absorption_day < 1:
            raise ValueError("minimum_absorption_day must be positive")
        if self.maximum_absorption_day < self.minimum_absorption_day:
            raise ValueError("maximum_absorption_day must not precede minimum_absorption_day")
        if self.event_quiet_days < 1:
            raise ValueError("event_quiet_days must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BloodChipBacktestConfig:
    """Execution and portfolio policy for long-horizon research."""

    initial_cash: float = 1_000_000.0
    maximum_positions: int = 10
    stop_loss: float = 0.10
    maximum_holding_days: int = 120
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    minimum_commission: float = 5.0
    slippage: float = 0.0005
    minimum_entry_gap: float = -0.07
    maximum_entry_gap: float = 0.07
    locked_limit_threshold: float = 0.048
    allow_reentry_after_stop: bool = True
    require_new_event_for_reentry: bool = True
    lot_size: int = 100

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.maximum_positions < 1:
            raise ValueError("maximum_positions must be positive")
        if not 0 < self.stop_loss < 1:
            raise ValueError("stop_loss must be in (0, 1)")
        if self.maximum_holding_days < 1:
            raise ValueError("maximum_holding_days must be positive")
        if self.lot_size < 1:
            raise ValueError("lot_size must be positive")
        for field_name in (
            "commission_rate",
            "stamp_tax_rate",
            "transfer_fee_rate",
            "minimum_commission",
            "slippage",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.maximum_entry_gap < self.minimum_entry_gap:
            raise ValueError("maximum_entry_gap must not be below minimum_entry_gap")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BloodChipBacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    rejected_entries: pd.DataFrame


@dataclass
class _Position:
    symbol: str
    event_id: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_raw: float
    entry_fill: float
    entry_factor: float
    adjusted_units: float
    raw_shares: int
    entry_value: float
    invested_value: float
    buy_fee: float
    stop_price: float
    sessions_held: int
    last_close: float
    maximum_adverse_excursion: float
    maximum_favorable_excursion: float
    reentry_number: int
    signal_payload: dict[str, object]


def _normalize_date(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _rolling(
    values: pd.Series,
    groups: pd.Series,
    window: int,
    operation: str,
    *,
    minimum_periods: int,
) -> pd.Series:
    rolling = values.groupby(groups, observed=True, sort=False).rolling(
        window,
        min_periods=minimum_periods,
    )
    result = getattr(rolling, operation)()
    return result.reset_index(level=0, drop=True).reindex(values.index)


def _cross_section_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date", observed=True, sort=False)[column].rank(
        method="average",
        pct=True,
    )


def load_canonical_daily(
    root: str | Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Read only monthly canonical partitions intersecting the requested range."""

    start = str(start_date).replace("-", "")
    end = str(end_date).replace("-", "")
    if end < start:
        raise ValueError("end_date must not precede start_date")
    root_path = Path(root)
    start_month = start[:6]
    end_month = end[:6]
    frames: list[pd.DataFrame] = []
    for path in sorted(root_path.glob("year_month=*/data.parquet")):
        month = path.parent.name.split("=", 1)[-1]
        if month < start_month or month > end_month:
            continue
        frame = pd.read_parquet(path, columns=DAILY_COLUMNS)
        dates = frame["trade_date"].astype(str)
        frame = frame.loc[dates.between(start, end, inclusive="both")]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no canonical daily rows found in {root_path}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_benchmark(
    path: str | Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load a point-in-time daily benchmark return series."""

    frame = pd.read_parquet(path)
    required = {"trade_date", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"benchmark missing columns: {missing}")
    start = str(start_date).replace("-", "")
    end = str(end_date).replace("-", "")
    dates = frame["trade_date"].astype(str)
    return frame.loc[dates.between(start, end, inclusive="both")].copy()


def _prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(DAILY_COLUMNS) - set(daily.columns))
    if missing:
        raise ValueError(f"daily data missing columns: {missing}")
    out = daily[DAILY_COLUMNS].copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    for column in DAILY_COLUMNS[2:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid_symbol = out["ts_code"].str.endswith((".SH", ".SZ", ".BJ"))
    out = out.loc[
        valid_symbol
        & out["date"].notna()
        & out["open"].gt(0)
        & out["high"].gt(0)
        & out["low"].gt(0)
        & out["close"].gt(0)
        & out["amount"].gt(0)
    ].copy()
    return (
        out.sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "date"], keep="last")
        .reset_index(drop=True)
    )


def _add_causal_continuous_prices(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    groups = out["ts_code"]
    previous_raw_close = out.groupby(groups, observed=True, sort=False)["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        action_ratio = previous_raw_close / out["pre_close"]
    action_ratio = action_ratio.where(action_ratio.gt(0) & np.isfinite(action_ratio), 1.0)
    out["adjustment_factor"] = action_ratio.groupby(
        groups,
        observed=True,
        sort=False,
    ).cumprod()
    for column in ("open", "high", "low", "close"):
        out[f"adjusted_{column}"] = out[column] * out["adjustment_factor"]
    return out


def _prepare_benchmark(benchmark: pd.DataFrame) -> pd.DataFrame:
    out = benchmark.copy()
    out["trade_date"] = out["trade_date"].astype(str).str.replace("-", "", regex=False)
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d", errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "pct_chg" in out:
        out["market_return_1d"] = pd.to_numeric(out["pct_chg"], errors="coerce") / 100.0
    else:
        out["market_return_1d"] = out["close"].pct_change(fill_method=None)
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    out["market_return_mean_60d_prior"] = out["market_return_1d"].shift(1).rolling(
        60,
        min_periods=40,
    ).mean()
    out["market_return_var_60d_prior"] = out["market_return_1d"].shift(1).rolling(
        60,
        min_periods=40,
    ).var()
    out["market_return_60d"] = out["close"].pct_change(60, fill_method=None)
    return out[
        [
            "date",
            "market_return_1d",
            "market_return_mean_60d_prior",
            "market_return_var_60d_prior",
            "market_return_60d",
        ]
    ]


def _validate_terminal_benchmark_coverage(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> None:
    """Require complete benchmark inputs for the latest stock observation date."""

    if daily.empty:
        raise ValueError("daily data has no valid observations")
    terminal_date = daily["date"].max()
    terminal_benchmark = benchmark.loc[benchmark["date"].eq(terminal_date)]
    if terminal_benchmark.empty:
        latest_benchmark_date = benchmark["date"].max() if not benchmark.empty else pd.NaT
        latest_text = (
            pd.Timestamp(latest_benchmark_date).date().isoformat()
            if pd.notna(latest_benchmark_date)
            else "none"
        )
        raise ValueError(
            "benchmark does not cover terminal daily date: "
            f"daily={terminal_date.date().isoformat()} benchmark={latest_text}"
        )
    terminal_return = pd.to_numeric(
        terminal_benchmark["market_return_1d"],
        errors="coerce",
    )
    if terminal_return.isna().any() or not np.isfinite(terminal_return).all():
        raise ValueError(
            "terminal benchmark return is missing: "
            f"date={terminal_date.date().isoformat()}"
        )
    required_rolling = [
        "market_return_mean_60d_prior",
        "market_return_var_60d_prior",
        "market_return_60d",
    ]
    missing_rolling = [
        column
        for column in required_rolling
        if pd.to_numeric(terminal_benchmark[column], errors="coerce").isna().any()
    ]
    if missing_rolling:
        raise ValueError(
            "terminal benchmark rolling features are missing: "
            f"date={terminal_date.date().isoformat()} columns={missing_rolling}"
        )


def build_blood_chip_features(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Build causal pressure and absorption features from daily bars."""

    out = _add_causal_continuous_prices(_prepare_daily(daily))
    market = _prepare_benchmark(benchmark)
    _validate_terminal_benchmark_coverage(out, market)
    out = out.merge(market, on="date", how="left", validate="many_to_one")
    groups = out["ts_code"]
    out["return_1d"] = out["pct_chg"] / 100.0
    product = out["return_1d"] * out["market_return_1d"]
    stock_mean = _rolling(
        out.groupby(groups, observed=True, sort=False)["return_1d"].shift(1),
        groups,
        60,
        "mean",
        minimum_periods=40,
    )
    product_mean = _rolling(
        product.groupby(groups, observed=True, sort=False).shift(1),
        groups,
        60,
        "mean",
        minimum_periods=40,
    )
    covariance = product_mean - stock_mean * out["market_return_mean_60d_prior"]
    out["market_beta_60d"] = (
        covariance / out["market_return_var_60d_prior"].replace(0, np.nan)
    ).clip(lower=-1.0, upper=3.0)
    effective_beta = out["market_beta_60d"].fillna(1.0)
    out["residual_return_1d"] = (
        out["return_1d"] - effective_beta * out["market_return_1d"]
    )
    out["residual_return_3d"] = _rolling(
        out["residual_return_1d"], groups, 3, "sum", minimum_periods=3
    )
    out["residual_return_5d"] = _rolling(
        out["residual_return_1d"], groups, 5, "sum", minimum_periods=5
    )
    out["history_days"] = out.groupby(groups, observed=True, sort=False).cumcount() + 1

    out["prior_amount_median_20d"] = _rolling(
        out.groupby(groups, observed=True, sort=False)["amount"].shift(1),
        groups,
        20,
        "median",
        minimum_periods=15,
    )
    amount_5d = _rolling(out["amount"], groups, 5, "mean", minimum_periods=5)
    amount_baseline = _rolling(
        out.groupby(groups, observed=True, sort=False)["amount"].shift(5),
        groups,
        60,
        "median",
        minimum_periods=40,
    )
    out["amount_ratio_5d"] = amount_5d / amount_baseline.replace(0, np.nan)

    negative_residual = (-out["residual_return_1d"]).clip(lower=0.0)
    negative_3d = _rolling(negative_residual, groups, 3, "sum", minimum_periods=3)
    negative_5d = _rolling(negative_residual, groups, 5, "sum", minimum_periods=5)
    amount_3d_sum = _rolling(out["amount"], groups, 3, "sum", minimum_periods=3)
    amount_5d_sum = _rolling(out["amount"], groups, 5, "sum", minimum_periods=5)
    out["down_impact_3d"] = negative_3d / amount_3d_sum.replace(0, np.nan)
    out["down_impact_5d"] = negative_5d / amount_5d_sum.replace(0, np.nan)
    impact_baseline = _rolling(
        out.groupby(groups, observed=True, sort=False)["down_impact_5d"].shift(5),
        groups,
        60,
        "median",
        minimum_periods=40,
    )
    out["impact_ratio_5d"] = out["down_impact_5d"] / impact_baseline.replace(0, np.nan)

    rolling_high = _rolling(
        out["adjusted_close"], groups, 20, "max", minimum_periods=15
    )
    out["drawdown_20d"] = out["adjusted_close"] / rolling_high.replace(0, np.nan) - 1.0
    prior_120 = out.groupby(groups, observed=True, sort=False)["adjusted_close"].shift(120)
    out["return_120d"] = out["adjusted_close"] / prior_120.replace(0, np.nan) - 1.0
    out["volatility_60d"] = _rolling(
        out["residual_return_1d"], groups, 60, "std", minimum_periods=40
    ) * sqrt(252.0)

    price_range = (out["adjusted_high"] - out["adjusted_low"]).replace(0, np.nan)
    out["clv_1d"] = (
        2.0 * out["adjusted_close"] - out["adjusted_high"] - out["adjusted_low"]
    ) / price_range
    out["clv_3d"] = _rolling(out["clv_1d"], groups, 3, "mean", minimum_periods=3)

    out["residual_5d_percentile"] = _cross_section_percentile(
        out, "residual_return_5d"
    )
    out["amount_ratio_5d_percentile"] = _cross_section_percentile(
        out, "amount_ratio_5d"
    )
    out["impact_ratio_5d_percentile"] = _cross_section_percentile(
        out, "impact_ratio_5d"
    )
    out["drawdown_20d_percentile"] = _cross_section_percentile(out, "drawdown_20d")
    out["shock_score"] = (
        0.35 * (1.0 - out["residual_5d_percentile"])
        + 0.25 * out["amount_ratio_5d_percentile"]
        + 0.25 * out["impact_ratio_5d_percentile"]
        + 0.15 * (1.0 - out["drawdown_20d_percentile"])
    )
    return out


def generate_blood_chip_signals(
    features: pd.DataFrame,
    config: BloodChipSignalConfig,
    *,
    include_pending_entry: bool = False,
) -> pd.DataFrame:
    """Generate one first-confirmation signal for each separated shock event.

    ``include_pending_entry`` keeps a confirmation on the final available bar so
    a live plan can describe the next-open conditions.  The default remains
    strict for backtests: rows without a known next bar are excluded.
    """

    required = {
        "ts_code",
        "date",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "history_days",
        "prior_amount_median_20d",
        "residual_5d_percentile",
        "amount_ratio_5d",
        "impact_ratio_5d",
        "drawdown_20d_percentile",
        "shock_score",
        "residual_return_3d",
        "down_impact_3d",
        "clv_3d",
        "return_120d",
        "volatility_60d",
        "market_return_60d",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    out = features.copy().sort_values(["ts_code", "date"]).reset_index(drop=True)
    groups = out["ts_code"]
    shock_flag = (
        out["history_days"].ge(config.minimum_history_days)
        & out["prior_amount_median_20d"].ge(config.minimum_prior_amount_thousand)
        & out["residual_5d_percentile"].le(config.maximum_residual_5d_percentile)
        & out["amount_ratio_5d"].ge(config.minimum_amount_ratio_5d)
        & out["impact_ratio_5d"].ge(config.minimum_impact_ratio_5d)
        & out["shock_score"].ge(config.shock_score_threshold)
    ).fillna(False)
    previous_shock = _rolling(
        shock_flag.astype(float).groupby(groups, observed=True, sort=False).shift(1),
        groups,
        config.event_quiet_days,
        "max",
        minimum_periods=1,
    ).fillna(0.0)
    new_event = shock_flag & previous_shock.eq(0.0)
    out["shock_event_id"] = new_event.astype(int).groupby(
        groups,
        observed=True,
        sort=False,
    ).cumsum()
    positions = out.groupby(groups, observed=True, sort=False).cumcount().astype(float)
    anchor_position = positions.where(new_event).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    out["days_since_shock"] = positions - anchor_position
    out["shock_date"] = out["date"].where(new_event).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    out["shock_score_anchor"] = out["shock_score"].where(new_event).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    anchor_impact_source = (
        out["down_impact_5d"] if "down_impact_5d" in out else out["impact_ratio_5d"]
    )
    out["shock_down_impact"] = anchor_impact_source.where(new_event).groupby(
        groups,
        observed=True,
        sort=False,
    ).ffill()
    event_groups = [out["ts_code"], out["shock_event_id"]]
    out["event_low"] = out["adjusted_low"].groupby(
        event_groups,
        observed=True,
        sort=False,
    ).cummin()
    out["rebound_from_event_low"] = (
        out["adjusted_close"] / out["event_low"].replace(0, np.nan) - 1.0
    )
    out["impact_decay"] = out["down_impact_3d"] / out["shock_down_impact"].replace(
        0, np.nan
    )
    out["rebound_percentile"] = _cross_section_percentile(
        out, "rebound_from_event_low"
    )
    out["impact_decay_percentile"] = _cross_section_percentile(out, "impact_decay")
    out["residual_3d_percentile"] = _cross_section_percentile(
        out, "residual_return_3d"
    )
    out["clv_3d_percentile"] = _cross_section_percentile(out, "clv_3d")
    out["absorption_score"] = (
        0.30 * out["rebound_percentile"]
        + 0.25 * (1.0 - out["impact_decay_percentile"])
        + 0.25 * out["residual_3d_percentile"]
        + 0.20 * out["clv_3d_percentile"]
    )
    eligible = (
        out["shock_event_id"].gt(0)
        & out["days_since_shock"].between(
            config.minimum_absorption_day,
            config.maximum_absorption_day,
            inclusive="both",
        )
        & out["rebound_from_event_low"].ge(config.minimum_rebound_from_event_low)
        & out["impact_decay"].le(config.maximum_impact_decay)
        & out["residual_return_3d"].gt(config.minimum_residual_3d)
        & out["clv_3d"].ge(config.minimum_clv_3d)
        & out["absorption_score"].ge(config.absorption_score_threshold)
    ).fillna(False)
    if config.minimum_return_120d is not None:
        eligible &= out["return_120d"].ge(config.minimum_return_120d).fillna(False)
    if config.maximum_return_120d is not None:
        eligible &= out["return_120d"].le(config.maximum_return_120d).fillna(False)
    if config.maximum_volatility_60d is not None:
        eligible &= out["volatility_60d"].le(config.maximum_volatility_60d).fillna(False)
    if config.minimum_market_return_60d is not None:
        eligible &= out["market_return_60d"].ge(
            config.minimum_market_return_60d
        ).fillna(False)
    eligible_number = eligible.astype(int).groupby(
        event_groups,
        observed=True,
        sort=False,
    ).cumsum()
    signal_flag = eligible & eligible_number.eq(1)
    out["entry_date"] = out.groupby(groups, observed=True, sort=False)["date"].shift(-1)
    out["entry_open"] = out.groupby(groups, observed=True, sort=False)[
        "adjusted_open"
    ].shift(-1)
    out["signal_score"] = (
        0.55 * out["shock_score_anchor"] + 0.45 * out["absorption_score"]
    )
    out["signal_date"] = out["date"]
    out["signal_close"] = out["adjusted_close"]
    keep = [
        "ts_code",
        "shock_event_id",
        "shock_date",
        "signal_date",
        "entry_date",
        "entry_open",
        "signal_close",
        "signal_score",
        "shock_score_anchor",
        "absorption_score",
        "days_since_shock",
        "rebound_from_event_low",
        "impact_decay",
        "residual_return_3d",
        "clv_3d",
        "return_120d",
        "volatility_60d",
        "market_return_60d",
        "amount_ratio_5d",
        "impact_ratio_5d",
    ]
    signals = out.loc[signal_flag, keep].copy()
    signals = signals.rename(columns={"shock_score_anchor": "shock_score"})
    if not include_pending_entry:
        signals = signals.dropna(subset=["entry_date", "entry_open"])
    return signals.reset_index(drop=True)


def add_blood_chip_path_features(
    features: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach causal shock-to-confirmation exhaustion features to signals."""

    path_columns = [
        "shock_realized_volatility_5d",
        "prior_realized_volatility_20d",
        "shock_volatility_expansion_ratio",
        "confirmation_realized_volatility_3d",
        "volatility_decay_ratio",
        "confirmation_volatility_vs_prior_ratio",
        "prior_range_pct_20d",
        "shock_range_pct_5d",
        "shock_range_expansion_ratio",
        "confirmation_range_pct_3d",
        "range_decay_ratio",
        "prior_amount_mean_20d",
        "shock_amount_mean_5d",
        "shock_amount_expansion_ratio",
        "confirmation_amount_mean_3d",
        "amount_decay_ratio",
        "confirmation_amount_vs_prior_ratio",
        "shock_downside_amount_5d",
        "shock_downside_amount_share",
        "confirmation_downside_amount_3d",
        "confirmation_downside_amount_share",
        "downside_amount_decay_ratio",
        "downside_amount_share_decay_ratio",
        "shock_sell_pressure_5d",
        "confirmation_sell_pressure_3d",
        "sell_pressure_decay_ratio",
        "event_low_age_sessions",
        "event_close_location",
        "shock_to_signal_residual_return",
    ]
    out = signals.copy()
    if out.empty:
        for column in path_columns:
            out[column] = pd.Series(dtype=float)
        return out
    required_features = {
        "ts_code",
        "date",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "amount",
        "residual_return_1d",
    }
    missing_features = sorted(required_features - set(features.columns))
    if missing_features:
        raise ValueError(f"feature frame missing path columns: {missing_features}")
    required_signals = {"ts_code", "shock_date", "signal_date"}
    missing_signals = sorted(required_signals - set(out.columns))
    if missing_signals:
        raise ValueError(f"signal frame missing path columns: {missing_signals}")

    panel_columns = [
        "ts_code",
        "date",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "amount",
        "residual_return_1d",
    ]
    if "volatility_60d" in features:
        panel_columns.append("volatility_60d")
    panel = features[panel_columns].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ts_code", "date"]).drop_duplicates(
        ["ts_code", "date"], keep="last"
    ).reset_index(drop=True)
    panel["_panel_index"] = np.arange(len(panel), dtype=np.int64)
    lookup = panel[["ts_code", "date", "_panel_index"]]

    out["shock_date"] = pd.to_datetime(out["shock_date"])
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["_signal_order"] = np.arange(len(out), dtype=np.int64)
    shock_lookup = lookup.rename(
        columns={"date": "shock_date", "_panel_index": "_shock_index"}
    )
    signal_lookup = lookup.rename(
        columns={"date": "signal_date", "_panel_index": "_signal_index"}
    )
    out = out.merge(
        shock_lookup,
        on=["ts_code", "shock_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        signal_lookup,
        on=["ts_code", "signal_date"],
        how="left",
        validate="many_to_one",
    )

    symbols = panel["ts_code"].astype(str).to_numpy()
    highs = pd.to_numeric(panel["adjusted_high"], errors="coerce").to_numpy()
    lows = pd.to_numeric(panel["adjusted_low"], errors="coerce").to_numpy()
    closes = pd.to_numeric(panel["adjusted_close"], errors="coerce").to_numpy()
    amounts = pd.to_numeric(panel["amount"], errors="coerce").to_numpy()
    residuals = pd.to_numeric(
        panel["residual_return_1d"], errors="coerce"
    ).to_numpy()
    group_starts = np.zeros(len(panel), dtype=np.int64)
    if len(panel):
        indices = np.arange(len(panel), dtype=np.int64)
        new_group = np.concatenate(([True], symbols[1:] != symbols[:-1]))
        group_starts = np.maximum.accumulate(np.where(new_group, indices, 0))

    def window_statistics(indices: np.ndarray) -> tuple[float, float, float, float, float]:
        if len(indices) == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        returns = residuals[indices]
        window_amounts = amounts[indices]
        valid_returns = returns[np.isfinite(returns)]
        realized_volatility = (
            float(np.std(valid_returns, ddof=1) * sqrt(252.0))
            if len(valid_returns) >= 2
            else np.nan
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ranges = (highs[indices] - lows[indices]) / closes[indices]
        range_mean = float(np.nanmean(ranges)) if np.isfinite(ranges).any() else np.nan
        amount_mean = (
            float(np.nanmean(window_amounts))
            if np.isfinite(window_amounts).any()
            else np.nan
        )
        downside_mask = np.isfinite(returns) & (returns < 0)
        downside_amount = np.where(downside_mask, window_amounts, 0.0)
        downside_amount_mean = float(np.nansum(downside_amount) / len(indices))
        total_amount = float(np.nansum(window_amounts))
        pressure_numerator = float(
            np.nansum(window_amounts * np.clip(-returns, 0.0, None))
        )
        sell_pressure = pressure_numerator / total_amount if total_amount > 0 else np.nan
        return (
            realized_volatility,
            range_mean,
            amount_mean,
            downside_amount_mean,
            sell_pressure,
        )

    def safe_ratio(numerator: float, denominator: float) -> float:
        return (
            float(numerator / denominator)
            if np.isfinite(numerator) and np.isfinite(denominator) and denominator > 0
            else np.nan
        )

    rows: list[dict[str, float]] = []
    for _, signal in out.iterrows():
        shock_value = signal["_shock_index"]
        signal_value = signal["_signal_index"]
        if pd.isna(shock_value) or pd.isna(signal_value):
            rows.append({column: np.nan for column in path_columns})
            continue
        shock_index = int(shock_value)
        signal_index = int(signal_value)
        if (
            shock_index > signal_index
            or symbols[shock_index] != symbols[signal_index]
            or symbols[signal_index] != str(signal["ts_code"])
        ):
            rows.append({column: np.nan for column in path_columns})
            continue
        shock_start = max(int(group_starts[shock_index]), shock_index - 4)
        prior_start = max(int(group_starts[shock_index]), shock_index - 20)
        confirmation_start = max(shock_index + 1, signal_index - 2)
        prior_indices = np.arange(prior_start, shock_index)
        shock_indices = np.arange(shock_start, shock_index + 1)
        confirmation_indices = np.arange(confirmation_start, signal_index + 1)
        event_indices = np.arange(shock_index, signal_index + 1)
        prior_stats = window_statistics(prior_indices)
        shock_stats = window_statistics(shock_indices)
        confirmation_stats = window_statistics(confirmation_indices)
        event_lows = lows[event_indices]
        event_highs = highs[event_indices]
        finite_lows = event_lows[np.isfinite(event_lows)]
        finite_highs = event_highs[np.isfinite(event_highs)]
        if len(finite_lows):
            event_low = float(np.min(finite_lows))
            low_positions = np.flatnonzero(np.isclose(event_lows, event_low, equal_nan=False))
            low_age = float(len(event_indices) - 1 - int(low_positions[-1]))
        else:
            event_low = np.nan
            low_age = np.nan
        event_high = float(np.max(finite_highs)) if len(finite_highs) else np.nan
        event_span = event_high - event_low
        close_location = (
            float((closes[signal_index] - event_low) / event_span)
            if np.isfinite(event_span) and event_span > 0
            else np.nan
        )
        post_shock_residual = float(
            np.nansum(residuals[shock_index + 1 : signal_index + 1])
        )
        rows.append(
            {
                "shock_realized_volatility_5d": shock_stats[0],
                "prior_realized_volatility_20d": prior_stats[0],
                "shock_volatility_expansion_ratio": safe_ratio(
                    shock_stats[0], prior_stats[0]
                ),
                "confirmation_realized_volatility_3d": confirmation_stats[0],
                "volatility_decay_ratio": safe_ratio(confirmation_stats[0], shock_stats[0]),
                "confirmation_volatility_vs_prior_ratio": safe_ratio(
                    confirmation_stats[0], prior_stats[0]
                ),
                "prior_range_pct_20d": prior_stats[1],
                "shock_range_pct_5d": shock_stats[1],
                "shock_range_expansion_ratio": safe_ratio(
                    shock_stats[1], prior_stats[1]
                ),
                "confirmation_range_pct_3d": confirmation_stats[1],
                "range_decay_ratio": safe_ratio(confirmation_stats[1], shock_stats[1]),
                "prior_amount_mean_20d": prior_stats[2],
                "shock_amount_mean_5d": shock_stats[2],
                "shock_amount_expansion_ratio": safe_ratio(
                    shock_stats[2], prior_stats[2]
                ),
                "confirmation_amount_mean_3d": confirmation_stats[2],
                "amount_decay_ratio": safe_ratio(confirmation_stats[2], shock_stats[2]),
                "confirmation_amount_vs_prior_ratio": safe_ratio(
                    confirmation_stats[2], prior_stats[2]
                ),
                "shock_downside_amount_5d": shock_stats[3],
                "shock_downside_amount_share": safe_ratio(
                    shock_stats[3], shock_stats[2]
                ),
                "confirmation_downside_amount_3d": confirmation_stats[3],
                "confirmation_downside_amount_share": safe_ratio(
                    confirmation_stats[3], confirmation_stats[2]
                ),
                "downside_amount_decay_ratio": safe_ratio(
                    confirmation_stats[3], shock_stats[3]
                ),
                "downside_amount_share_decay_ratio": safe_ratio(
                    safe_ratio(confirmation_stats[3], confirmation_stats[2]),
                    safe_ratio(shock_stats[3], shock_stats[2]),
                ),
                "shock_sell_pressure_5d": shock_stats[4],
                "confirmation_sell_pressure_3d": confirmation_stats[4],
                "sell_pressure_decay_ratio": safe_ratio(
                    confirmation_stats[4], shock_stats[4]
                ),
                "event_low_age_sessions": low_age,
                "event_close_location": close_location,
                "shock_to_signal_residual_return": post_shock_residual,
            }
        )
    path_frame = pd.DataFrame(rows, index=out.index)
    for column in path_columns:
        out[column] = path_frame[column]
    return out.sort_values("_signal_order").drop(
        columns=["_signal_order", "_shock_index", "_signal_index"]
    ).reset_index(drop=True)


def _is_locked_limit(
    open_price: float,
    high: float,
    low: float,
    previous_close: float,
    threshold: float,
    *,
    direction: str,
) -> bool:
    scale = max(abs(open_price), abs(high), abs(low), 1.0)
    one_price = abs(high - low) <= scale * 1e-10
    if not one_price or previous_close <= 0:
        return False
    gap = open_price / previous_close - 1.0
    if direction == "down":
        return gap <= -threshold
    return gap >= threshold


def _buy_fee(value: float, config: BloodChipBacktestConfig) -> float:
    return max(value * config.commission_rate, config.minimum_commission) + (
        value * config.transfer_fee_rate
    )


def _sell_fee(value: float, config: BloodChipBacktestConfig) -> float:
    return (
        max(value * config.commission_rate, config.minimum_commission)
        + value * config.transfer_fee_rate
        + value * config.stamp_tax_rate
    )


def _empty_backtest_result(initial_cash: float) -> BloodChipBacktestResult:
    equity = pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "cash": pd.Series(dtype=float),
            "positions": pd.Series(dtype=int),
            "equity": pd.Series(dtype=float),
            "daily_return": pd.Series(dtype=float),
        }
    )
    return BloodChipBacktestResult(equity, pd.DataFrame(), pd.DataFrame())


def run_blood_chip_backtest(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    config: BloodChipBacktestConfig,
    entry_start: str,
    entry_end: str,
) -> BloodChipBacktestResult:
    """Run a causal next-open portfolio with stops and event-based re-entry."""

    if signals.empty:
        return _empty_backtest_result(config.initial_cash)
    panel = daily.copy()
    if "date" not in panel:
        panel["date"] = pd.to_datetime(
            panel["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
    else:
        panel["date"] = pd.to_datetime(panel["date"])
    if not {"adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"} <= set(
        panel.columns
    ):
        panel = _add_causal_continuous_prices(_prepare_daily(panel))
    if "adjustment_factor" not in panel:
        panel["adjustment_factor"] = 1.0
    panel = panel.sort_values(["date", "ts_code"]).drop_duplicates(
        ["date", "ts_code"], keep="last"
    )
    signal_frame = signals.copy()
    for column in ("signal_date", "entry_date", "shock_date"):
        signal_frame[column] = pd.to_datetime(signal_frame[column])
    start = _normalize_date(entry_start)
    end = _normalize_date(entry_end)
    signal_frame = signal_frame.loc[
        signal_frame["entry_date"].between(start, end, inclusive="both")
    ].copy()
    if signal_frame.empty:
        return _empty_backtest_result(config.initial_cash)
    first_entry_date = pd.Timestamp(signal_frame["entry_date"].min())
    last_entry_date = pd.Timestamp(signal_frame["entry_date"].max())
    if "signal_close" not in signal_frame:
        closes = panel[["ts_code", "date", "adjusted_close"]].rename(
            columns={"date": "signal_date", "adjusted_close": "signal_close"}
        )
        signal_frame = signal_frame.merge(
            closes,
            on=["ts_code", "signal_date"],
            how="left",
            validate="many_to_one",
        )
    panel = panel.loc[panel["date"].ge(first_entry_date)].copy()
    signal_frame["signal_score"] = pd.to_numeric(
        signal_frame.get("signal_score", 0.0), errors="coerce"
    ).fillna(0.0)
    signals_by_date = {
        pd.Timestamp(date): group.sort_values(
            ["signal_score", "ts_code"], ascending=[False, True]
        )
        for date, group in signal_frame.groupby("entry_date", sort=True)
    }

    cash = float(config.initial_cash)
    positions: dict[str, _Position] = {}
    last_event_used: dict[str, int] = {}
    trade_count_by_symbol: dict[str, int] = {}
    ever_traded: set[str] = set()
    trade_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    previous_equity = float(config.initial_cash)
    last_date: pd.Timestamp | None = None
    last_bars: dict[str, pd.Series] = {}

    def reject(signal: pd.Series, reason: str) -> None:
        rejected_rows.append(
            {
                "ts_code": str(signal["ts_code"]),
                "signal_date": pd.Timestamp(signal["signal_date"]),
                "entry_date": pd.Timestamp(signal["entry_date"]),
                "shock_event_id": int(signal["shock_event_id"]),
                "reason": reason,
            }
        )

    def close_position(
        position: _Position,
        *,
        date: pd.Timestamp,
        exit_raw_adjusted: float,
        reason: str,
    ) -> None:
        nonlocal cash
        exit_fill = exit_raw_adjusted * (1.0 - config.slippage)
        sell_value = position.adjusted_units * exit_fill
        sell_fee = _sell_fee(sell_value, config)
        cash += sell_value - sell_fee
        gross_return = exit_fill / position.entry_fill - 1.0
        net_return = (sell_value - sell_fee - position.invested_value) / position.invested_value
        payload = dict(position.signal_payload)
        trade_rows.append(
            {
                **payload,
                "ts_code": position.symbol,
                "shock_event_id": position.event_id,
                "signal_date": position.signal_date,
                "entry_date": position.entry_date,
                "exit_date": date,
                "entry_raw": position.entry_raw,
                "entry_fill": position.entry_fill,
                "exit_raw": exit_raw_adjusted,
                "exit_fill": exit_fill,
                "entry_value": position.entry_value,
                "invested_value": position.invested_value,
                "exit_value": sell_value,
                "fees": position.buy_fee + sell_fee,
                "raw_shares": position.raw_shares,
                "holding_sessions": position.sessions_held,
                "exit_reason": reason,
                "gross_return": gross_return,
                "net_return": net_return,
                "maximum_adverse_excursion": position.maximum_adverse_excursion,
                "maximum_favorable_excursion": position.maximum_favorable_excursion,
                "reentry_number": position.reentry_number,
            }
        )
        positions.pop(position.symbol, None)

    for date, day_frame in panel.groupby("date", observed=True, sort=True):
        date = pd.Timestamp(date)
        last_date = date
        bars = day_frame.set_index("ts_code", drop=False)
        for symbol, position in list(positions.items()):
            if symbol not in bars.index:
                continue
            row = bars.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            position.sessions_held += 1
            open_adjusted = float(row["adjusted_open"])
            high_adjusted = float(row["adjusted_high"])
            low_adjusted = float(row["adjusted_low"])
            close_adjusted = float(row["adjusted_close"])
            previous_close = position.last_close
            position.maximum_adverse_excursion = min(
                position.maximum_adverse_excursion,
                low_adjusted / position.entry_fill - 1.0,
            )
            position.maximum_favorable_excursion = max(
                position.maximum_favorable_excursion,
                high_adjusted / position.entry_fill - 1.0,
            )
            position.last_close = close_adjusted
            locked_down = _is_locked_limit(
                open_adjusted,
                high_adjusted,
                low_adjusted,
                previous_close,
                config.locked_limit_threshold,
                direction="down",
            )
            stop_hit = low_adjusted <= position.stop_price
            time_hit = position.sessions_held >= config.maximum_holding_days
            if locked_down and (stop_hit or time_hit):
                continue
            if stop_hit:
                exit_raw = (
                    open_adjusted if open_adjusted <= position.stop_price else position.stop_price
                )
                close_position(position, date=date, exit_raw_adjusted=exit_raw, reason="stop_loss")
            elif time_hit:
                close_position(
                    position,
                    date=date,
                    exit_raw_adjusted=open_adjusted,
                    reason="time_exit",
                )

        candidates = signals_by_date.get(date)
        if candidates is not None:
            for _, signal in candidates.iterrows():
                symbol = str(signal["ts_code"])
                event_id = int(signal["shock_event_id"])
                if symbol in positions:
                    reject(signal, "already_holding")
                    continue
                if len(positions) >= config.maximum_positions:
                    reject(signal, "portfolio_full")
                    continue
                if symbol not in bars.index:
                    reject(signal, "missing_entry_bar")
                    continue
                if not config.allow_reentry_after_stop and symbol in ever_traded:
                    reject(signal, "reentry_disabled")
                    continue
                if config.require_new_event_for_reentry and event_id <= last_event_used.get(
                    symbol, 0
                ):
                    reject(signal, "event_reused")
                    continue
                row = bars.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                signal_close = float(signal["signal_close"])
                adjusted_open = float(row["adjusted_open"])
                adjusted_high = float(row["adjusted_high"])
                adjusted_low = float(row["adjusted_low"])
                entry_gap = adjusted_open / signal_close - 1.0
                if not config.minimum_entry_gap <= entry_gap <= config.maximum_entry_gap:
                    reject(signal, "entry_gap")
                    continue
                if _is_locked_limit(
                    adjusted_open,
                    adjusted_high,
                    adjusted_low,
                    signal_close,
                    config.locked_limit_threshold,
                    direction="up",
                ):
                    reject(signal, "locked_limit_up")
                    continue
                factor = float(row.get("adjustment_factor", 1.0) or 1.0)
                raw_open = float(row["open"])
                raw_fill = raw_open * (1.0 + config.slippage)
                marked_positions = 0.0
                for held_symbol, held in positions.items():
                    if held_symbol in bars.index:
                        held_row = bars.loc[held_symbol]
                        if isinstance(held_row, pd.DataFrame):
                            held_row = held_row.iloc[-1]
                        marked_positions += held.adjusted_units * float(
                            held_row["adjusted_open"]
                        )
                    else:
                        marked_positions += held.adjusted_units * held.last_close
                opening_equity = cash + marked_positions
                budget = min(cash, opening_equity / config.maximum_positions)
                lots = int(budget // (raw_fill * config.lot_size))
                raw_shares = lots * config.lot_size
                if raw_shares <= 0:
                    reject(signal, "insufficient_cash")
                    continue
                entry_value = raw_shares * raw_fill
                buy_fee = _buy_fee(entry_value, config)
                if entry_value + buy_fee > cash:
                    lots = int((cash - config.minimum_commission) // (raw_fill * config.lot_size))
                    raw_shares = max(lots, 0) * config.lot_size
                    entry_value = raw_shares * raw_fill
                    buy_fee = _buy_fee(entry_value, config) if raw_shares else 0.0
                if raw_shares <= 0 or entry_value + buy_fee > cash:
                    reject(signal, "insufficient_cash")
                    continue
                adjusted_entry_fill = raw_fill * factor
                reentry_number = trade_count_by_symbol.get(symbol, 0)
                payload = {
                    key: signal[key]
                    for key in CASE_FEATURES
                    if key in signal.index
                }
                payload["signal_score"] = float(signal.get("signal_score", np.nan))
                cash -= entry_value + buy_fee
                positions[symbol] = _Position(
                    symbol=symbol,
                    event_id=event_id,
                    signal_date=pd.Timestamp(signal["signal_date"]),
                    entry_date=date,
                    entry_raw=raw_open,
                    entry_fill=adjusted_entry_fill,
                    entry_factor=factor,
                    adjusted_units=raw_shares / factor,
                    raw_shares=raw_shares,
                    entry_value=entry_value,
                    invested_value=entry_value + buy_fee,
                    buy_fee=buy_fee,
                    stop_price=adjusted_entry_fill * (1.0 - config.stop_loss),
                    sessions_held=0,
                    last_close=float(row["adjusted_close"]),
                    maximum_adverse_excursion=min(
                        0.0, float(row["adjusted_low"]) / adjusted_entry_fill - 1.0
                    ),
                    maximum_favorable_excursion=max(
                        0.0, float(row["adjusted_high"]) / adjusted_entry_fill - 1.0
                    ),
                    reentry_number=reentry_number,
                    signal_payload=payload,
                )
                last_event_used[symbol] = event_id
                trade_count_by_symbol[symbol] = reentry_number + 1
                ever_traded.add(symbol)

        marked_value = 0.0
        for symbol, position in positions.items():
            if symbol in bars.index:
                row = bars.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                position.last_close = float(row["adjusted_close"])
            marked_value += position.adjusted_units * position.last_close
            if symbol in bars.index:
                last_bars[symbol] = row
        equity = cash + marked_value
        equity_rows.append(
            {
                "date": date,
                "cash": cash,
                "positions": len(positions),
                "equity": equity,
                "daily_return": equity / previous_equity - 1.0,
            }
        )
        previous_equity = equity
        if date > last_entry_date and not positions:
            break

    if positions and last_date is not None:
        for position in list(positions.values()):
            close_position(
                position,
                date=last_date,
                exit_raw_adjusted=position.last_close,
                reason="end_of_data",
            )
        if equity_rows:
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["positions"] = 0
            equity_rows[-1]["equity"] = cash
            previous = (
                config.initial_cash if len(equity_rows) == 1 else equity_rows[-2]["equity"]
            )
            equity_rows[-1]["daily_return"] = cash / float(previous) - 1.0

    return BloodChipBacktestResult(
        equity_curve=pd.DataFrame(equity_rows),
        trades=pd.DataFrame(trade_rows),
        rejected_entries=pd.DataFrame(rejected_rows),
    )


def _wilson_lower_bound(wins: int, total: int, z: float = 1.95996398454) -> float:
    if total <= 0:
        return np.nan
    probability = wins / total
    denominator = 1.0 + z * z / total
    center = probability + z * z / (2.0 * total)
    adjustment = z * sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    )
    return (center - adjustment) / denominator


def summarize_blood_chip_result(
    result: BloodChipBacktestResult,
    benchmark: pd.DataFrame,
) -> dict[str, float | int]:
    """Summarize net portfolio and trade results under one stable schema."""

    if result.equity_curve.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": np.nan,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "maximum_drawdown": np.nan,
        }
    equity = result.equity_curve.copy().sort_values("date")
    daily_returns = pd.to_numeric(equity["daily_return"], errors="coerce").fillna(0.0)
    wealth = (1.0 + daily_returns).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    elapsed_days = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days, 1)
    annualized_return = (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0
    wealth_with_initial = np.concatenate(([1.0], wealth.to_numpy(dtype=float)))
    running_peak = np.maximum.accumulate(wealth_with_initial)
    drawdown = wealth_with_initial / running_peak - 1.0
    daily_std = float(daily_returns.std(ddof=1))
    sharpe = (
        float(daily_returns.mean() / daily_std * sqrt(252.0))
        if daily_std > 0
        else np.nan
    )
    trades = result.trades.copy()
    returns = pd.to_numeric(
        trades.get("net_return", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    wins = int((returns > 0).sum())
    gains = returns[returns > 0]
    losses = returns[returns <= 0]
    loss_sum = float(-losses.sum())

    market = _prepare_benchmark(benchmark)
    market = market.loc[
        market["date"].between(equity["date"].min(), equity["date"].max(), inclusive="both")
    ]
    benchmark_return = float(
        (1.0 + market["market_return_1d"].fillna(0.0)).prod() - 1.0
    )
    return {
        "trades": int(len(returns)),
        "wins": wins,
        "win_rate": float(wins / len(returns)) if len(returns) else np.nan,
        "win_rate_wilson_lower_95": _wilson_lower_bound(wins, len(returns)),
        "average_net_return": float(returns.mean()) if len(returns) else np.nan,
        "median_net_return": float(returns.median()) if len(returns) else np.nan,
        "average_win": float(gains.mean()) if len(gains) else np.nan,
        "average_loss": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(gains.sum() / loss_sum) if loss_sum > 0 else np.nan,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "maximum_drawdown": float(np.min(drawdown)),
        "sharpe": sharpe,
        "benchmark_total_return": benchmark_return,
        "excess_total_return": total_return - benchmark_return,
        "average_holding_sessions": float(trades["holding_sessions"].mean())
        if not trades.empty
        else np.nan,
        "stop_rate": float(trades["exit_reason"].eq("stop_loss").mean())
        if not trades.empty
        else np.nan,
        "reentry_trades": int(trades["reentry_number"].gt(0).sum())
        if not trades.empty
        else 0,
        "successful_reentries": int(
            (trades["reentry_number"].gt(0) & trades["net_return"].gt(0)).sum()
        )
        if not trades.empty
        else 0,
        "rejected_entries": int(len(result.rejected_entries)),
    }


def analyze_blood_chip_cases(trades: pd.DataFrame) -> pd.DataFrame:
    """Compare outcome groups using an explicit allow-list of entry-time features."""

    if trades.empty:
        return pd.DataFrame(
            columns=["case_type", "feature", "count", "mean", "median", "q25", "q75"]
        )
    masks = {
        "winner": pd.to_numeric(trades["net_return"], errors="coerce").gt(0),
        "loss": pd.to_numeric(trades["net_return"], errors="coerce").le(0),
        "stop_loss": trades["exit_reason"].eq("stop_loss"),
        "successful_reentry": trades["reentry_number"].gt(0)
        & pd.to_numeric(trades["net_return"], errors="coerce").gt(0),
        "failed_reentry": trades["reentry_number"].gt(0)
        & pd.to_numeric(trades["net_return"], errors="coerce").le(0),
    }
    rows: list[dict[str, Any]] = []
    for case_type, mask in masks.items():
        if not mask.any():
            continue
        for feature in CASE_FEATURES:
            if feature not in trades:
                continue
            values = pd.to_numeric(trades.loc[mask, feature], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "case_type": case_type,
                    "feature": feature,
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)
