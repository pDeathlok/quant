"""Canonical project factor calculation and sample-admission contracts.

This module is the only supported entry point for the complete daily market
factor set.  Cheap strategy gates may still consume the smaller cached base
layer, but every model-facing row is built here after the gate has passed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quant.data.factors import (
    ATR,
    BIAS,
    CCI,
    EMA,
    MA,
    MACD,
    OBV,
    RSI,
    Amplitude,
    BollingerBands,
    DownsideVolatility,
    KDJ,
    Volatility,
    WilliamsR,
)
from quant.data.factors.alpha101 import (
    Alpha001Factor,
    Alpha002Factor,
    Alpha003Factor,
    Alpha004Factor,
    Alpha005Factor,
    Alpha006Factor,
    Alpha007Factor,
    Alpha008Factor,
    Alpha009Factor,
    Alpha010Factor,
)
from quant.data.factors.alpha191 import (
    Alpha191_01Factor,
    Alpha191_02Factor,
    Alpha191_03Factor,
    Alpha191_04Factor,
    Alpha191_05Factor,
    Alpha191_06Factor,
    Alpha191_07Factor,
    Alpha191_08Factor,
    Alpha191_09Factor,
    Alpha191_10Factor,
    Alpha191_11Factor,
    Alpha191_12Factor,
    Alpha191_13Factor,
    Alpha191_14Factor,
    Alpha191_15Factor,
)
from quant.data.factors.momentum import (
    MomentumSkip5Factor,
    ReversalFactor,
    RiskAdjustedMomentumFactor,
    ReturnFactor,
)
from quant.data.factors.technical import (
    ChaikinMoneyFlow,
    EaseOfMovement,
    KeltnerChannel,
    MassIndex,
    PSY,
    ParabolicSAR,
    VR,
    VortexIndicator,
)
from quant.data.source_merge import normalize_tushare_daily
from quant.features.variable_library import (
    DAILY_BASIC_SOURCE_COLUMNS,
    PROJECT_FACTOR_COLUMNS,
    build_continuous_ohlc,
    build_latest_scale_ohlc,
    calculate_project_extra_features,
)


PROJECT_FACTOR_SCHEMA_VERSION = "project-v5-canonical-alias-free"
LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION = "project-v1-latest-scale-global-rank"
KEY_COLUMNS = ("ts_code", "symbol", "trade_date", "date")


def _last_percentile_rank(values: np.ndarray) -> float:
    """Match pandas' average percentile rank for the last rolling value."""

    last = values[-1]
    less = np.count_nonzero(values < last)
    equal = np.count_nonzero(values == last)
    return float(less + (equal + 1) / 2) / len(values)


def resolve_project_factor_schema(requested: str | None = None) -> str:
    mode = requested or os.getenv("PROJECT_FACTOR_COMPATIBILITY_MODE", "")
    if mode in {"legacy", "legacy-production", LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION}:
        return LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
    if mode in {"", "current", PROJECT_FACTOR_SCHEMA_VERSION}:
        return PROJECT_FACTOR_SCHEMA_VERSION
    raise ValueError(f"unknown project factor schema/mode: {mode}")


@dataclass(frozen=True)
class FactorAdmission:
    """A sample-size-only factor decision made before model fitting."""

    factor: str
    non_null_rows: int
    coverage: float
    unique_values: int
    admitted: bool
    reason: str


def _prepare_daily(daily: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    out = normalize_tushare_daily(daily, symbol)
    out = out.sort_values("date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    if "symbol" not in out.columns:
        out["symbol"] = out.get("ts_code", symbol)
    return out


def _round_half_up_to_cent(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.floor(numeric * 100.0 + 0.50000001) / 100.0


def calculate_limit_up_flags(frame: pd.DataFrame) -> pd.Series:
    """Return board-, date-, and ST-aware A-share limit-up closes.

    Prefer an exchange-provided ``up_limit`` column when available. The local
    fallback covers ordinary continuous-trading days; rows without a usable
    previous close remain false instead of fabricating an IPO limit.
    """

    close = pd.to_numeric(frame["close"], errors="coerce")
    pre_close = pd.to_numeric(
        frame.get("pre_close", close.shift(1)),
        errors="coerce",
    )
    if "up_limit" in frame.columns:
        provided = pd.to_numeric(frame["up_limit"], errors="coerce")
    else:
        provided = pd.Series(np.nan, index=frame.index, dtype=float)

    code_source = frame.get("ts_code", frame.get("symbol", ""))
    if isinstance(code_source, pd.Series):
        codes = code_source.fillna("").astype(str).str.upper()
    else:
        codes = pd.Series(str(code_source).upper(), index=frame.index)
    names = frame.get("name", pd.Series("", index=frame.index))
    if not isinstance(names, pd.Series):
        names = pd.Series(str(names), index=frame.index)
    names = names.fillna("").astype(str).str.upper()
    dates = pd.to_datetime(frame.get("date"), errors="coerce")

    is_beijing = codes.str.endswith(".BJ") | codes.str.match(r"^(4|8|92)")
    is_star = codes.str.match(r"^(688|689)")
    is_chinext = codes.str.match(r"^(300|301)")
    chinext_reform = is_chinext & dates.ge(pd.Timestamp("2020-08-24"))
    is_main_st = names.str.contains("ST", regex=False) & ~(
        is_beijing | is_star | chinext_reform
    )

    ratio = pd.Series(0.10, index=frame.index, dtype=float)
    ratio.loc[is_main_st] = 0.05
    ratio.loc[is_star | chinext_reform] = 0.20
    ratio.loc[is_beijing] = 0.30
    fallback = _round_half_up_to_cent(pre_close * (1.0 + ratio))
    limit_price = provided.combine_first(fallback)
    valid = close.notna() & pre_close.gt(0) & limit_price.notna()
    return (valid & close.ge(limit_price - 1e-8)).astype(bool)


def _calculate_legacy_alpha101(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact pre-v4 single-symbol Alpha101 semantics for pinned models only."""

    out = pd.DataFrame(index=frame.index)
    returns = frame["close"].pct_change()
    standard_deviation = returns.rolling(20).std()
    alpha001_value = pd.Series(
        np.where(returns < 0, standard_deviation, frame["close"]),
        index=frame.index,
    ).pow(2)
    out["alpha001"] = (
        alpha001_value.rolling(5).apply(np.argmax, raw=True).rank(pct=True) - 0.5
    )
    log_volume_delta = np.log(frame["volume"]).diff(2)
    open_return = (frame["close"] - frame["open"]) / frame["open"]
    out["alpha002"] = -log_volume_delta.rank(pct=True).rolling(6).corr(open_return.rank(pct=True))
    rank_open = frame["open"].rank(pct=True)
    rank_volume = frame["volume"].rank(pct=True)
    covariance = pd.DataFrame({"open": rank_open, "volume": rank_volume}).rolling(5).cov()
    covariance_values = covariance.loc[(slice(None), "open"), "volume"]
    covariance_values.index = covariance_values.index.droplevel(1)
    out["alpha003"] = -covariance_values.rank(pct=True)
    volume_rank_3 = frame["volume"].rolling(3).apply(
        _last_percentile_rank,
        raw=True,
    )
    negative_return_rank_3 = (-frame["close"].diff()).rolling(3).apply(
        _last_percentile_rank,
        raw=True,
    )
    out["alpha004"] = (volume_rank_3 * negative_return_rank_3).rank(pct=True)
    vwap = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    open_difference = frame["open"] - vwap.rolling(10).mean()
    close_difference = frame["close"] - vwap
    out["alpha005"] = open_difference.rank(pct=True) * -close_difference.abs().rank(pct=True)
    out["alpha006"] = -(frame["open"] - frame["close"]).rolling(20).mean().rank(pct=True)
    out["alpha007"] = frame["high"].rolling(5).corr(frame["volume"]).rank(pct=True)
    open_return_sum = open_return.rolling(20).sum()
    volume_ratio = frame["volume"].rolling(20).std() / frame["volume"].rolling(20).mean().replace(0, np.nan)
    out["alpha008"] = (open_return_sum.rank(pct=True) * volume_ratio.rank(pct=True)).rank(pct=True)
    volume_delta = frame["volume"].diff()
    negative_close_delta = -frame["close"].diff()
    out["alpha009"] = -(volume_delta.rank(pct=True) * negative_close_delta.rank(pct=True)).rank(pct=True)
    out["alpha010"] = -rank_volume.rolling(3).apply(
        _last_percentile_rank,
        raw=True,
    )
    return out


def calculate_legacy_market_factors(
    daily: pd.DataFrame,
    *,
    factor_schema_version: str | None = None,
) -> pd.DataFrame:
    """Calculate the historical B1 factor family without strategy-local code."""

    frame = daily.sort_values("date").reset_index(drop=True).copy()
    limit_up = calculate_limit_up_flags(frame)
    schema = resolve_project_factor_schema(factor_schema_version)
    price_builder = (
        build_latest_scale_ohlc
        if schema == LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
        else build_continuous_ohlc
    )
    price = price_builder(frame)
    for column in ("open", "high", "low", "close"):
        frame[column] = price[column]
    frame["pre_close"] = frame["close"].shift(1)
    frame["pct_chg"] = frame["close"].pct_change() * 100
    factors = pd.DataFrame(index=frame.index)
    gt_9p5pct = frame["pct_chg"] > 9.5

    for window in (3, 5, 10, 20, 60):
        factors[f"limit_up_cnt_{window}d"] = limit_up.rolling(window, min_periods=window).sum()
        factors[f"gt_9p5pct_cnt_{window}d"] = gt_9p5pct.rolling(
            window, min_periods=window
        ).sum()

    for window in (5, 10, 20, 60, 120):
        factors[f"ma_{window}"] = MA(window).compute(frame)
    for window in (5, 10, 20):
        factors[f"ema_{window}"] = EMA(window).compute(frame)
    factors["macd"] = MACD().compute(frame)
    for window in (6, 12, 24):
        factors[f"rsi_{window}"] = RSI(window).compute(frame)
    bands = BollingerBands().compute(frame)
    if isinstance(bands, pd.DataFrame):
        factors["bb_upper"] = bands.iloc[:, 0]
        factors["bb_lower"] = bands.iloc[:, 2]
    factors["atr_14"] = ATR(14).compute(frame)
    factors["williams_r_14"] = WilliamsR(14).compute(frame)
    factors["cci"] = CCI().compute(frame)
    for window in (6, 12, 24):
        factors[f"bias_{window}"] = BIAS(window).compute(frame)
    factors["obv"] = OBV().compute(frame)
    factors["psy_12"] = PSY(12).compute(frame)
    factors["psy_24"] = PSY(24).compute(frame)
    for window in (6, 12, 24):
        factors[f"vr_{window}"] = VR(window).compute(frame)
    factors["mass_index"] = MassIndex().compute(frame)
    factors["parabolic_sar"] = ParabolicSAR().compute(frame)
    vortex = VortexIndicator().compute(frame)
    if isinstance(vortex, pd.DataFrame):
        factors["vortex_plus"] = vortex.iloc[:, 0]
        factors["vortex_minus"] = vortex.iloc[:, 1]
    factors["cmf"] = ChaikinMoneyFlow().compute(frame)
    factors["eom"] = EaseOfMovement().compute(frame)
    channel = KeltnerChannel().compute(frame)
    if isinstance(channel, pd.DataFrame):
        factors["keltner_upper"] = channel.iloc[:, 0]
        factors["keltner_lower"] = channel.iloc[:, 1]
        factors["keltner_width"] = (channel.iloc[:, 0] - channel.iloc[:, 1]) / channel.iloc[:, 2]
    factors["amplitude_1"] = Amplitude(1).compute(frame)
    factors["amplitude_20"] = Amplitude(20).compute(frame)

    if schema == LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION:
        factors = pd.concat([factors, _calculate_legacy_alpha101(frame)], axis=1)
    else:
        alpha101 = (
            Alpha001Factor,
            Alpha002Factor,
            Alpha003Factor,
            Alpha004Factor,
            Alpha005Factor,
            Alpha006Factor,
            Alpha007Factor,
            Alpha008Factor,
            Alpha009Factor,
            Alpha010Factor,
        )
        for number, factory in enumerate(alpha101, start=1):
            factors[f"alpha{number:03d}"] = factory().compute(frame)
    alpha191 = (
        Alpha191_01Factor,
        Alpha191_02Factor,
        Alpha191_03Factor,
        Alpha191_04Factor,
        Alpha191_05Factor,
        Alpha191_06Factor,
        Alpha191_07Factor,
        Alpha191_08Factor,
        Alpha191_09Factor,
        Alpha191_10Factor,
        Alpha191_11Factor,
        Alpha191_12Factor,
        Alpha191_13Factor,
        Alpha191_14Factor,
        Alpha191_15Factor,
    )
    for number, factory in enumerate(alpha191, start=1):
        factors[f"alpha191_{number:02d}"] = factory().compute(frame)
    for window in (1, 5, 10, 20, 60, 120):
        factors[f"return_{window}d"] = ReturnFactor(window).compute(frame)
    for window in (5, 20, 60):
        factors[f"momentum_{window}d"] = MomentumSkip5Factor(window).compute(frame)
    factors["risk_adjusted_momentum"] = RiskAdjustedMomentumFactor().compute(frame)
    factors["reversal_5d"] = ReversalFactor(5).compute(frame)
    factors["reversal_20d"] = ReversalFactor(20).compute(frame)
    for window in (20, 60):
        factors[f"volatility_{window}d"] = Volatility(window).compute(frame)
        factors[f"downside_volatility_{window}d"] = DownsideVolatility(window).compute(frame)
    factors["price_log"] = np.log(frame["close"] + 1)
    factors["price_volume_ratio"] = frame["close"] / (frame["volume"] + 1)
    factors["volume_relative_60d"] = frame["volume"] / frame["volume"].rolling(60).mean().replace(0, np.nan)
    return factors.replace([np.inf, -np.inf], np.nan)


def calculate_project_market_factors(
    daily: pd.DataFrame,
    symbol: str = "",
    *,
    shared_factors: pd.DataFrame | None = None,
    factor_schema_version: str | None = None,
) -> pd.DataFrame:
    """Return model-ready keys plus every project OHLCV-derived factor."""

    prepared = _prepare_daily(daily, symbol)
    if prepared.empty:
        return prepared
    schema = resolve_project_factor_schema(factor_schema_version)
    price_builder = (
        build_latest_scale_ohlc
        if schema == LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
        else build_continuous_ohlc
    )
    legacy = calculate_legacy_market_factors(
        prepared,
        factor_schema_version=schema,
    )
    shared = (
        calculate_project_extra_features(prepared, price_builder=price_builder)
        if shared_factors is None or schema == LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
        else shared_factors.reset_index(drop=True)
    )
    raw = prepared.copy()
    price = price_builder(prepared)
    for column in ("open", "high", "low", "close"):
        raw[column] = price[column]
    raw["change"] = raw["close"] - raw["close"].shift(1)
    raw["pre_close"] = raw["close"].shift(1)
    raw["pct_chg"] = raw["close"].pct_change() * 100
    combined = pd.concat([raw, legacy, shared], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    market_columns = [
        column
        for column in PROJECT_FACTOR_COLUMNS
        if column in combined.columns and column not in DAILY_BASIC_SOURCE_COLUMNS
    ]
    result = combined[[*KEY_COLUMNS, *market_columns]].copy()
    result["factor_schema_version"] = schema
    return result.replace([np.inf, -np.inf], np.nan)


def calculate_project_factor_frame(
    daily: pd.DataFrame,
    symbol: str = "",
    *,
    shared_factors: pd.DataFrame | None = None,
    daily_basic_features: pd.DataFrame | None = None,
    factor_schema_version: str | None = None,
) -> pd.DataFrame:
    """Return the complete canonical project contract when daily_basic is supplied.

    The daily_basic frame must already be point-in-time derived by
    ``load_daily_basic_features``.  Applications may defer this merge until
    after a cheap candidate gate, but the final model frame always follows this
    contract.
    """

    market = calculate_project_market_factors(
        daily,
        symbol,
        shared_factors=shared_factors,
        factor_schema_version=factor_schema_version,
    )
    out = market.copy()
    if daily_basic_features is not None and not daily_basic_features.empty:
        right = daily_basic_features.copy()
        right["trade_date"] = right["trade_date"].astype(str)
        if "ts_code" not in right.columns and "symbol" in right.columns:
            right["ts_code"] = right["symbol"].astype(str)
        duplicate_columns = [
            column
            for column in right.columns
            if column not in {"ts_code", "trade_date"} and column in out.columns
        ]
        if duplicate_columns:
            out = out.drop(columns=duplicate_columns)
        out = out.merge(right, on=["ts_code", "trade_date"], how="left")
    for column in PROJECT_FACTOR_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[[*KEY_COLUMNS, *PROJECT_FACTOR_COLUMNS, "factor_schema_version"]].replace(
        [np.inf, -np.inf], np.nan
    )


def admit_factors_by_sample(
    frame: pd.DataFrame,
    candidates: Iterable[str],
    *,
    minimum_non_null_rows: int,
    minimum_coverage: float,
    minimum_unique_values: int = 2,
) -> tuple[list[str], pd.DataFrame]:
    """Admit all sufficiently populated factors, without performance screening."""

    if minimum_non_null_rows < 1:
        raise ValueError("minimum_non_null_rows must be positive")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between zero and one")
    decisions: list[FactorAdmission] = []
    admitted: list[str] = []
    total = max(len(frame), 1)
    for factor in dict.fromkeys(candidates):
        if factor not in frame.columns:
            decisions.append(FactorAdmission(factor, 0, 0.0, 0, False, "missing_column"))
            continue
        values = pd.to_numeric(frame[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
        non_null = int(values.notna().sum())
        coverage = non_null / total
        unique = int(values.nunique(dropna=True))
        if non_null < minimum_non_null_rows:
            reason = "insufficient_rows"
        elif coverage < minimum_coverage:
            reason = "insufficient_coverage"
        elif unique < minimum_unique_values:
            reason = "constant_or_degenerate"
        else:
            reason = "admitted"
            admitted.append(factor)
        decisions.append(
            FactorAdmission(factor, non_null, coverage, unique, reason == "admitted", reason)
        )
    report = pd.DataFrame([decision.__dict__ for decision in decisions])
    return admitted, report
