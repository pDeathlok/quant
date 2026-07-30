"""Versioned, reusable daily technical-factor layer.

The layer owns factors shared by B1, family-rule, and z-skill consumers.  Raw
market data remains the source of truth; this module only provides an
idempotent, versioned materialization so a routine run computes common rolling
features once.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.data.source_merge import normalize_tushare_daily
from quant.data.factors.technical import KDJ
from quant.features.variable_library import (
    EXTRA_FEATURE_COLUMNS,
    build_continuous_ohlc,
    calc_bbi,
    calculate_project_extra_features,
)


FACTOR_LAYER_VERSION = "v1"
SIGNAL_FACTOR_LAYER_VERSION = "signal-v1"
SIGNAL_STATE_SCHEMA_VERSION = 1
DEFAULT_FACTOR_ROOT = Path("data/features/daily_factor_layer")
KEY_COLUMNS = ["ts_code", "symbol", "trade_date", "date"]
Z_FACTOR_COLUMNS = [
    "z_amplitude",
    "z_close_pos",
    "z_vol_ratio_prev",
    "z_vol_ratio_5",
    "z_vol_ma10",
    "z_vol_ma20",
    "z_is_rise",
    "z_is_shrink",
    "z_is_beidou",
    "z_is_big_yin",
    "z_ma3",
    "z_ma6",
    "z_ma12",
    "z_ma24",
    "z_bbi",
    "z_white",
    "z_yellow",
    "z_kdj_j",
]
Z_CONSUMER_ALIASES = {
    "amplitude": "z_amplitude",
    "close_pos": "z_close_pos",
    "vol_ratio_5": "z_vol_ratio_5",
    "vol_ratio_prev": "z_vol_ratio_prev",
    "vol_ma10": "z_vol_ma10",
    "vol_ma20": "z_vol_ma20",
    "is_rise": "z_is_rise",
    "is_shrink": "z_is_shrink",
    "is_beidou": "z_is_beidou",
    "is_big_yin": "z_is_big_yin",
    "ma3": "z_ma3",
    "ma6": "z_ma6",
    "ma12": "z_ma12",
    "ma24": "z_ma24",
    "bbi": "z_bbi",
    "dg_yellow": "z_yellow",
}
BASE_FACTOR_COLUMNS = [*EXTRA_FEATURE_COLUMNS, "ma_5", "ma_10", "ma_20", "ma_60", "ma_120", *Z_FACTOR_COLUMNS]
SIGNAL_FACTOR_COLUMNS = [
    "kdj_d_j",
    "bbi",
    "volume_relative_5d",
    "volume_relative_20d",
    "ma_60",
    *Z_FACTOR_COLUMNS,
]
SIGNAL_PRICE_FACTOR_COLUMNS = [
    "bbi",
    "ma_60",
    "z_ma3",
    "z_ma6",
    "z_ma12",
    "z_ma24",
    "z_bbi",
    "z_white",
    "z_yellow",
]
SIGNAL_SOURCE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "pct_chg",
]


def _prepare_daily(daily: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    out = normalize_tushare_daily(daily, symbol)
    out = out.sort_values("date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    if "symbol" not in out.columns:
        out["symbol"] = out.get("ts_code", symbol)
    return out


def calculate_z_base_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate the existing z-skill primitives with namespaced columns."""

    out = pd.DataFrame(index=daily.index)
    price = build_continuous_ohlc(daily)
    open_ = price["open"]
    high = price["high"]
    low = price["low"]
    close = price["close"]
    volume = daily["volume"]
    pre_close = close.shift(1).replace(0, np.nan)
    pct_chg = close.pct_change() * 100

    out["z_amplitude"] = (high - low) / pre_close * 100
    out["z_close_pos"] = (close - low) / (high - low).replace(0, np.nan)
    out["z_vol_ratio_prev"] = volume / volume.shift(1).replace(0, np.nan)
    out["z_vol_ratio_5"] = volume / volume.shift(1).rolling(5, min_periods=1).mean().replace(0, np.nan)
    out["z_vol_ma10"] = volume.rolling(10, min_periods=3).mean()
    out["z_vol_ma20"] = volume.rolling(20, min_periods=5).mean()
    out["z_is_rise"] = close > open_
    out["z_is_shrink"] = volume < volume.shift(1) * 0.75
    out["z_is_beidou"] = (pct_chg >= 3) & (out["z_vol_ratio_5"] >= 1.5)
    out["z_is_big_yin"] = (close < open_) & (out["z_vol_ratio_5"] >= 1.5) & (pct_chg <= -2)
    out["z_ma3"] = close.rolling(3, min_periods=1).mean()
    out["z_ma6"] = close.rolling(6, min_periods=2).mean()
    out["z_ma12"] = close.rolling(12, min_periods=4).mean()
    out["z_ma24"] = close.rolling(24, min_periods=8).mean()
    out["z_bbi"] = (out["z_ma3"] + out["z_ma6"] + out["z_ma12"] + out["z_ma24"]) / 4
    out["z_white"] = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    out["z_yellow"] = (
        close.rolling(14, min_periods=8).mean()
        + close.rolling(28, min_periods=14).mean()
        + close.rolling(57, min_periods=28).mean()
        + close.rolling(114, min_periods=60).mean()
    ) / 4
    low9 = low.rolling(9, min_periods=3).min()
    high9 = high.rolling(9, min_periods=3).max()
    rsv = ((close - low9) / (high9 - low9).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    out["z_kdj_j"] = 3 * k - 2 * d
    return out


def calculate_daily_base_factors(daily: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Calculate every strategy-neutral factor shared by routine consumers."""

    prepared = _prepare_daily(daily, symbol)
    if prepared.empty:
        return prepared
    price = build_continuous_ohlc(prepared)
    common = calculate_project_extra_features(prepared)
    for window in (5, 10, 20, 60, 120):
        common[f"ma_{window}"] = price["close"].rolling(window).mean()
    factors = pd.concat([prepared[KEY_COLUMNS], common, calculate_z_base_factors(prepared)], axis=1)
    factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
    factors["factor_version"] = FACTOR_LAYER_VERSION
    return factors


def _signal_source_hashes(prepared: pd.DataFrame) -> pd.Series:
    source = pd.DataFrame(index=prepared.index)
    for column in SIGNAL_SOURCE_COLUMNS:
        if column == "date":
            source[column] = pd.to_datetime(
                prepared.get(column),
                errors="coerce",
            ).astype("int64")
        else:
            values = prepared[column] if column in prepared.columns else np.nan
            source[column] = pd.to_numeric(values, errors="coerce")
    return pd.util.hash_pandas_object(source, index=False).astype("uint64")


def _ewm_adjust_false(
    values: pd.Series,
    alpha: float,
) -> tuple[pd.Series, dict[str, float | None]]:
    """Match pandas adjust=False EWM while retaining its online state."""

    weighted = np.nan
    old_weight = 1.0
    output: list[float] = []
    old_weight_factor = 1.0 - alpha
    for raw_value in pd.to_numeric(values, errors="coerce"):
        is_observation = pd.notna(raw_value)
        if pd.isna(weighted):
            if is_observation:
                weighted = float(raw_value)
        else:
            old_weight *= old_weight_factor
            if is_observation:
                value = float(raw_value)
                if weighted != value:
                    weighted = (
                        old_weight * weighted + alpha * value
                    ) / (old_weight + alpha)
                old_weight = 1.0
        output.append(float(weighted) if pd.notna(weighted) else np.nan)
    state = {
        "value": float(weighted) if pd.notna(weighted) else None,
        "old_weight": float(old_weight),
    }
    return pd.Series(output, index=values.index, dtype=float), state


def _ewm_adjust_false_step(
    state: dict[str, Any],
    value: float,
    alpha: float,
) -> tuple[float, dict[str, float | None]]:
    weighted = state.get("value")
    old_weight = float(state.get("old_weight", 1.0))
    is_observation = pd.notna(value)
    if weighted is None or pd.isna(weighted):
        weighted = float(value) if is_observation else np.nan
    else:
        old_weight *= 1.0 - alpha
        if is_observation:
            observed = float(value)
            if weighted != observed:
                weighted = (
                    old_weight * float(weighted) + alpha * observed
                ) / (old_weight + alpha)
            old_weight = 1.0
    next_state = {
        "value": float(weighted) if pd.notna(weighted) else None,
        "old_weight": float(old_weight),
    }
    return (
        float(weighted) if pd.notna(weighted) else np.nan,
        next_state,
    )


def calculate_daily_signal_factors(
    daily: pd.DataFrame,
    symbol: str = "",
) -> pd.DataFrame:
    """Calculate only the factors consumed by the fused daily signal scan."""

    prepared = _prepare_daily(daily, symbol)
    if prepared.empty:
        return prepared
    price = build_continuous_ohlc(prepared)
    volume = pd.to_numeric(prepared["volume"], errors="coerce")
    factors = prepared[KEY_COLUMNS].copy()
    factors["kdj_d_j"] = KDJ().compute(price)["J"]
    factors["bbi"] = calc_bbi(price["close"])
    factors["volume_relative_5d"] = (
        volume / volume.rolling(5).mean().replace(0, np.nan)
    )
    factors["volume_relative_20d"] = (
        volume / volume.rolling(20).mean().replace(0, np.nan)
    )
    factors["ma_60"] = price["close"].rolling(60).mean()
    factors = pd.concat(
        [factors, calculate_z_base_factors(prepared)],
        axis=1,
    )
    factors = factors.loc[:, ~factors.columns.duplicated(keep="last")]
    factors["factor_basis_close"] = pd.to_numeric(
        price["close"],
        errors="coerce",
    )
    factors["source_hash"] = _signal_source_hashes(prepared).to_numpy()
    factors["factor_version"] = SIGNAL_FACTOR_LAYER_VERSION
    return factors


def _signal_state_from_prepared(prepared: pd.DataFrame) -> dict[str, Any]:
    price = build_continuous_ohlc(prepared)
    close = pd.to_numeric(price["close"], errors="coerce")
    high = pd.to_numeric(price["high"], errors="coerce")
    low = pd.to_numeric(price["low"], errors="coerce")
    volume = pd.to_numeric(prepared["volume"], errors="coerce")

    daily_low9 = low.rolling(9).min()
    daily_high9 = high.rolling(9).max()
    daily_rsv = (
        (close - daily_low9)
        / (daily_high9 - daily_low9).replace(0, np.nan)
        * 100
    )
    daily_k, daily_k_state = _ewm_adjust_false(daily_rsv, 1 / 3)
    _, daily_d_state = _ewm_adjust_false(daily_k, 1 / 3)

    z_low9 = low.rolling(9, min_periods=3).min()
    z_high9 = high.rolling(9, min_periods=3).max()
    z_rsv = (
        (close - z_low9)
        / (z_high9 - z_low9).replace(0, np.nan)
        * 100
    ).fillna(50)
    z_k, z_k_state = _ewm_adjust_false(z_rsv, 1 / 3)
    _, z_d_state = _ewm_adjust_false(z_k, 1 / 3)

    white_first, white_first_state = _ewm_adjust_false(close, 2 / 11)
    _, white_second_state = _ewm_adjust_false(white_first, 2 / 11)
    last = prepared.iloc[-1]
    return {
        "schema_version": SIGNAL_STATE_SCHEMA_VERSION,
        "factor_version": SIGNAL_FACTOR_LAYER_VERSION,
        "last_date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "last_raw_close": float(last["close"]),
        "close": [float(value) for value in close.tail(120)],
        "high": [float(value) for value in high.tail(9)],
        "low": [float(value) for value in low.tail(9)],
        "volume": [float(value) for value in volume.tail(20)],
        "daily_k": daily_k_state,
        "daily_d": daily_d_state,
        "z_k": z_k_state,
        "z_d": z_d_state,
        "z_white_first": white_first_state,
        "z_white_second": white_second_state,
    }


def factor_symbol_dir(factor_root: Path, symbol: str) -> Path:
    safe_symbol = str(symbol).replace("/", "_").replace(":", "")
    return factor_root / FACTOR_LAYER_VERSION / safe_symbol


def load_daily_base_factors(
    symbol: str,
    factor_root: Path = DEFAULT_FACTOR_ROOT,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    directory = factor_symbol_dir(Path(factor_root), symbol)
    if start_date is not None and end_date is not None:
        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year
        paths = [directory / f"{year}.parquet" for year in range(start_year, end_year + 1)]
        paths = [path for path in paths if path.exists()]
    else:
        paths = sorted(directory.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return out.sort_values("date").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def signal_factor_symbol_dir(factor_root: Path, symbol: str) -> Path:
    safe_symbol = str(symbol).replace("/", "_").replace(":", "")
    return factor_root / SIGNAL_FACTOR_LAYER_VERSION / safe_symbol


def load_daily_signal_factors(
    symbol: str,
    factor_root: Path = DEFAULT_FACTOR_ROOT,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    directory = signal_factor_symbol_dir(Path(factor_root), symbol)
    if start_date is not None and end_date is not None:
        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year
        years = range(start_year, end_year + 1)
    else:
        years = sorted(
            int(path.stem)
            for path in directory.glob("[0-9][0-9][0-9][0-9].parquet")
        )
    paths = [
        path
        for year in years
        for path in (
            directory / f"{year}.parquet",
            directory / f"{year}.delta.parquet",
        )
        if path.exists()
    ]
    if not paths:
        return pd.DataFrame()
    out = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
        sort=False,
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out["date"] <= pd.Timestamp(end_date)]
    return (
        out.drop_duplicates(["symbol", "date"], keep="last")
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )


@contextmanager
def _symbol_cache_lock(factor_root: Path, symbol: str):
    """Serialize first-writer computation across parallel routine consumers."""

    directory = factor_symbol_dir(factor_root, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".refresh.lock"
    with lock_path.open("a+") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError, UnboundLocalError):
                pass


@contextmanager
def _signal_cache_lock(factor_root: Path, symbol: str):
    directory = signal_factor_symbol_dir(factor_root, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".refresh.lock"
    with lock_path.open("a+") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError, UnboundLocalError):
                pass


def _cache_covers(cached: pd.DataFrame, dates: pd.Series) -> bool:
    requested = set(pd.to_datetime(dates, errors="coerce").dropna().dt.normalize())
    if not requested:
        return True
    if cached.empty:
        return False
    available = set(pd.to_datetime(cached["date"], errors="coerce").dropna().dt.normalize())
    return requested <= available


def attach_daily_base_factors(
    daily: pd.DataFrame,
    symbol: str,
    factor_root: Path = DEFAULT_FACTOR_ROOT,
    compute_if_missing: bool = True,
    persist_missing: bool = True,
) -> pd.DataFrame:
    """Attach cached factors by date, calculating only when cache is unavailable."""

    factor_root = Path(os.getenv("DAILY_FACTOR_ROOT", str(factor_root)))
    prepared = _prepare_daily(daily, symbol)
    if prepared.empty:
        return prepared
    cached = load_daily_base_factors(
        symbol,
        factor_root=factor_root,
        start_date=prepared["date"].min(),
        end_date=prepared["date"].max(),
    )
    if not _cache_covers(cached, prepared["date"]) and compute_if_missing:
        if not persist_missing:
            cached = calculate_daily_base_factors(prepared, symbol)
        else:
            with _symbol_cache_lock(factor_root, symbol):
                cached = load_daily_base_factors(
                    symbol,
                    factor_root=factor_root,
                    start_date=prepared["date"].min(),
                    end_date=prepared["date"].max(),
                )
                if not _cache_covers(cached, prepared["date"]):
                    computed = calculate_daily_base_factors(prepared, symbol)
                    for year, group in computed.groupby(computed["date"].dt.year):
                        _write_year_partition(group, factor_root, symbol, int(year))
                    cached = computed
    if cached.empty:
        return prepared
    feature_cols = [col for col in BASE_FACTOR_COLUMNS if col in cached.columns]
    existing = [col for col in feature_cols if col in prepared.columns]
    if existing:
        prepared = prepared.drop(columns=existing)
    return prepared.merge(cached[["date", *feature_cols]], on="date", how="left")


def _mean_tail(values: list[float], window: int, min_periods: int) -> float:
    selected = values[-window:]
    if len(selected) < min_periods:
        return np.nan
    return float(np.mean(selected))


def _signal_state_step(
    state: dict[str, Any],
    row: pd.Series,
    source_hash: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_raw_close = float(state["last_raw_close"])
    current_pre_close = pd.to_numeric(
        pd.Series([row.get("pre_close")]),
        errors="coerce",
    ).iloc[0]
    scale = (
        float(current_pre_close) / last_raw_close
        if pd.notna(current_pre_close) and last_raw_close
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    closes = [float(value) * scale for value in state["close"]]
    highs = [float(value) * scale for value in state["high"]]
    lows = [float(value) * scale for value in state["low"]]
    volumes = [float(value) for value in state["volume"]]
    for key in ("z_white_first", "z_white_second"):
        value = state[key].get("value")
        if value is not None:
            state[key]["value"] = float(value) * scale

    current_open = float(row["open"])
    current_high = float(row["high"])
    current_low = float(row["low"])
    current_close = float(row["close"])
    current_volume = float(row["volume"])
    previous_close = closes[-1] if closes else np.nan
    previous_volume = volumes[-1] if volumes else np.nan
    previous_five_volume = _mean_tail(volumes, 5, 1)

    closes.append(current_close)
    highs.append(current_high)
    lows.append(current_low)
    volumes.append(current_volume)
    low9 = min(lows[-9:]) if len(lows) >= 9 else np.nan
    high9 = max(highs[-9:]) if len(highs) >= 9 else np.nan
    daily_rsv = (
        (current_close - low9) / (high9 - low9) * 100
        if pd.notna(low9) and high9 != low9
        else np.nan
    )
    daily_k, state["daily_k"] = _ewm_adjust_false_step(
        state["daily_k"],
        daily_rsv,
        1 / 3,
    )
    daily_d, state["daily_d"] = _ewm_adjust_false_step(
        state["daily_d"],
        daily_k,
        1 / 3,
    )

    z_low9 = min(lows[-9:]) if len(lows) >= 3 else np.nan
    z_high9 = max(highs[-9:]) if len(highs) >= 3 else np.nan
    z_rsv = (
        (current_close - z_low9) / (z_high9 - z_low9) * 100
        if pd.notna(z_low9) and z_high9 != z_low9
        else 50.0
    )
    z_k, state["z_k"] = _ewm_adjust_false_step(
        state["z_k"],
        z_rsv,
        1 / 3,
    )
    z_d, state["z_d"] = _ewm_adjust_false_step(
        state["z_d"],
        z_k,
        1 / 3,
    )
    white_first, state["z_white_first"] = _ewm_adjust_false_step(
        state["z_white_first"],
        current_close,
        2 / 11,
    )
    white_second, state["z_white_second"] = _ewm_adjust_false_step(
        state["z_white_second"],
        white_first,
        2 / 11,
    )

    pct_chg = (
        (current_close / previous_close - 1) * 100
        if pd.notna(previous_close) and previous_close
        else np.nan
    )
    z_vol_ratio_5 = (
        current_volume / previous_five_volume
        if pd.notna(previous_five_volume) and previous_five_volume
        else np.nan
    )
    z_vol_ratio_prev = (
        current_volume / previous_volume
        if pd.notna(previous_volume) and previous_volume
        else np.nan
    )
    z_ma3 = _mean_tail(closes, 3, 1)
    z_ma6 = _mean_tail(closes, 6, 2)
    z_ma12 = _mean_tail(closes, 12, 4)
    z_ma24 = _mean_tail(closes, 24, 8)
    yellow_parts = [
        _mean_tail(closes, 14, 8),
        _mean_tail(closes, 28, 14),
        _mean_tail(closes, 57, 28),
        _mean_tail(closes, 114, 60),
    ]
    z_yellow = (
        float(sum(yellow_parts) / 4)
        if all(pd.notna(value) for value in yellow_parts)
        else np.nan
    )
    record = {
        "ts_code": row["ts_code"],
        "symbol": row["symbol"],
        "trade_date": row["trade_date"],
        "date": pd.Timestamp(row["date"]),
        "kdj_d_j": 3 * daily_k - 2 * daily_d,
        "bbi": np.mean(
            [
                _mean_tail(closes, 3, 3),
                _mean_tail(closes, 6, 6),
                _mean_tail(closes, 12, 12),
                _mean_tail(closes, 24, 24),
            ]
        ),
        "volume_relative_5d": (
            current_volume / _mean_tail(volumes, 5, 5)
            if _mean_tail(volumes, 5, 5)
            else np.nan
        ),
        "volume_relative_20d": (
            current_volume / _mean_tail(volumes, 20, 20)
            if _mean_tail(volumes, 20, 20)
            else np.nan
        ),
        "ma_60": _mean_tail(closes, 60, 60),
        "z_amplitude": (
            (current_high - current_low) / previous_close * 100
            if pd.notna(previous_close) and previous_close
            else np.nan
        ),
        "z_close_pos": (
            (current_close - current_low) / (current_high - current_low)
            if current_high != current_low
            else np.nan
        ),
        "z_vol_ratio_prev": z_vol_ratio_prev,
        "z_vol_ratio_5": z_vol_ratio_5,
        "z_vol_ma10": _mean_tail(volumes, 10, 3),
        "z_vol_ma20": _mean_tail(volumes, 20, 5),
        "z_is_rise": current_close > current_open,
        "z_is_shrink": (
            current_volume < previous_volume * 0.75
            if pd.notna(previous_volume)
            else False
        ),
        "z_is_beidou": bool(
            pd.notna(pct_chg)
            and pd.notna(z_vol_ratio_5)
            and pct_chg >= 3
            and z_vol_ratio_5 >= 1.5
        ),
        "z_is_big_yin": bool(
            current_close < current_open
            and pd.notna(z_vol_ratio_5)
            and pd.notna(pct_chg)
            and z_vol_ratio_5 >= 1.5
            and pct_chg <= -2
        ),
        "z_ma3": z_ma3,
        "z_ma6": z_ma6,
        "z_ma12": z_ma12,
        "z_ma24": z_ma24,
        "z_bbi": (z_ma3 + z_ma6 + z_ma12 + z_ma24) / 4,
        "z_white": white_second,
        "z_yellow": z_yellow,
        "z_kdj_j": 3 * z_k - 2 * z_d,
        "factor_basis_close": current_close,
        "source_hash": np.uint64(source_hash),
        "factor_version": SIGNAL_FACTOR_LAYER_VERSION,
    }
    state["last_date"] = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    state["last_raw_close"] = current_close
    state["close"] = closes[-120:]
    state["high"] = highs[-9:]
    state["low"] = lows[-9:]
    state["volume"] = volumes[-20:]
    return state, record


def _load_signal_state(
    factor_root: Path,
    symbol: str,
) -> dict[str, Any] | None:
    path = signal_factor_symbol_dir(factor_root, symbol) / "state.json"
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        state.get("schema_version") != SIGNAL_STATE_SCHEMA_VERSION
        or state.get("factor_version") != SIGNAL_FACTOR_LAYER_VERSION
    ):
        return None
    try:
        last_date = pd.Timestamp(state["last_date"])
        last_raw_close = float(state["last_raw_close"])
        if pd.isna(last_date) or not np.isfinite(last_raw_close) or last_raw_close <= 0:
            return None
        for key, maximum in (
            ("close", 120),
            ("high", 9),
            ("low", 9),
            ("volume", 20),
        ):
            values = state[key]
            if (
                not isinstance(values, list)
                or not values
                or len(values) > maximum
            ):
                return None
            numeric = np.asarray(values, dtype=float)
            if not np.isfinite(numeric).all():
                return None
        for key in (
            "daily_k",
            "daily_d",
            "z_k",
            "z_d",
            "z_white_first",
            "z_white_second",
        ):
            ewm_state = state[key]
            old_weight = float(ewm_state["old_weight"])
            value = ewm_state.get("value")
            if not np.isfinite(old_weight) or old_weight <= 0:
                return None
            if value is not None and not np.isfinite(float(value)):
                return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return state


def _write_signal_state(
    state: dict[str, Any],
    factor_root: Path,
    symbol: str,
) -> Path:
    directory = signal_factor_symbol_dir(factor_root, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state.json"
    temp_path = directory / f".state.{os.getpid()}.tmp"
    temp_path.write_text(
        json.dumps(state, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temp_path, path)
    return path


def _write_signal_year_partition(
    frame: pd.DataFrame,
    factor_root: Path,
    symbol: str,
    year: int,
    *,
    replace_start: pd.Timestamp | None = None,
    replace_end: pd.Timestamp | None = None,
) -> Path:
    directory = signal_factor_symbol_dir(factor_root, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{year}.parquet"
    delta_path = directory / f"{year}.delta.parquet"
    existing_frames = [
        pd.read_parquet(existing_path)
        for existing_path in (path, delta_path)
        if existing_path.exists()
    ]
    existing = (
        pd.concat(existing_frames, ignore_index=True, sort=False)
        if existing_frames
        else pd.DataFrame()
    )
    if not existing.empty:
        existing = existing.drop_duplicates(
            ["symbol", "date"],
            keep="last",
        )
    if not existing.empty and replace_start is not None:
        existing_dates = pd.to_datetime(
            existing["date"],
            errors="coerce",
        )
        end = replace_end if replace_end is not None else replace_start
        existing = existing[
            (existing_dates < replace_start)
            | (existing_dates > end)
        ].copy()
    combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    if combined.empty and not len(combined.columns):
        return path
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined.sort_values("date")
        .drop_duplicates(["symbol", "date"], keep="last")
    )
    temp_path = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    combined.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)
    if delta_path.exists():
        delta_path.unlink()
    return path


def _write_signal_delta_partition(
    frame: pd.DataFrame,
    factor_root: Path,
    symbol: str,
    year: int,
) -> Path:
    directory = signal_factor_symbol_dir(factor_root, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{year}.delta.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = (
        combined.sort_values("date", kind="mergesort")
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    compact_rows = max(
        2,
        int(os.getenv("SIGNAL_FACTOR_DELTA_COMPACT_ROWS", "64")),
    )
    if len(combined) >= compact_rows:
        _write_signal_year_partition(
            combined,
            factor_root,
            symbol,
            year,
        )
        return directory / f"{year}.parquet"
    temp_path = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    combined.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)
    return path


def _bootstrap_signal_factor_cache(
    prepared: pd.DataFrame,
    symbol: str,
    factor_root: Path,
    *,
    reconcile_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    factors = calculate_daily_signal_factors(prepared, symbol)
    replace_start = pd.Timestamp(prepared["date"].min())
    replace_end = max(
        pd.Timestamp(prepared["date"].max()),
        pd.Timestamp(reconcile_end)
        if reconcile_end is not None and pd.notna(reconcile_end)
        else pd.Timestamp(prepared["date"].max()),
    )
    for year in range(replace_start.year, replace_end.year + 1):
        group = factors[factors["date"].dt.year == year]
        path = signal_factor_symbol_dir(factor_root, symbol) / f"{year}.parquet"
        if group.empty and not path.exists():
            continue
        _write_signal_year_partition(
            group,
            factor_root,
            symbol,
            int(year),
            replace_start=replace_start,
            replace_end=replace_end,
        )
    _write_signal_state(
        _signal_state_from_prepared(prepared),
        factor_root,
        symbol,
    )
    return factors


def _refresh_signal_factor_cache(
    prepared: pd.DataFrame,
    symbol: str,
    factor_root: Path,
) -> tuple[pd.DataFrame, str]:
    cached = load_daily_signal_factors(
        symbol,
        factor_root=factor_root,
        start_date=prepared["date"].min(),
        end_date=prepared["date"].max(),
    )
    current_hashes = pd.DataFrame(
        {
            "date": prepared["date"],
            "current_source_hash": _signal_source_hashes(prepared).to_numpy(),
        }
    )
    state = _load_signal_state(factor_root, symbol)
    state_date = (
        pd.Timestamp(state["last_date"])
        if state is not None and state.get("last_date")
        else pd.NaT
    )
    rebuild_mode = "bootstrap"
    if not cached.empty:
        required = {
            "source_hash",
            "factor_basis_close",
            "factor_version",
            *SIGNAL_FACTOR_COLUMNS,
        }
        if required <= set(cached.columns):
            overlap = cached[["date", "source_hash"]].merge(
                current_hashes,
                on="date",
                how="inner",
            )
            hashes_match = (
                overlap["source_hash"].astype("uint64")
                == overlap["current_source_hash"].astype("uint64")
            ).all()
            versions_match = cached["factor_version"].eq(
                SIGNAL_FACTOR_LAYER_VERSION
            ).all()
            cached_dates = set(cached["date"])
            prepared_dates = set(prepared["date"])
            cached_dates_still_exist = cached_dates.issubset(
                prepared_dates
            )
            cache_state_matches = (
                state is not None
                and pd.notna(state_date)
                and cached["date"].max() == state_date
            )
            if (
                hashes_match
                and versions_match
                and cached_dates_still_exist
                and cache_state_matches
            ):
                missing = prepared[~prepared["date"].isin(cached_dates)]
                if missing.empty:
                    return cached, "cache_hit"
                missing_is_append = (
                    pd.notna(state_date)
                    and missing["date"].min() > state_date
                    and cached["date"].max() == state_date
                    and prepared.loc[
                        prepared["date"] <= state_date,
                        "date",
                    ].isin(cached_dates).all()
                )
                if state is not None and missing_is_append:
                    hash_by_date = current_hashes.set_index("date")[
                        "current_source_hash"
                    ]
                    records: list[dict[str, Any]] = []
                    for _, row in missing.sort_values("date").iterrows():
                        state, record = _signal_state_step(
                            state,
                            row,
                            int(hash_by_date.loc[row["date"]]),
                        )
                        records.append(record)
                    incremental = pd.DataFrame(records)
                    for year, group in incremental.groupby(
                        incremental["date"].dt.year
                    ):
                        _write_signal_delta_partition(
                            group,
                            factor_root,
                            symbol,
                            int(year),
                        )
                    _write_signal_state(state, factor_root, symbol)
                    return (
                        pd.concat(
                            [cached, incremental],
                            ignore_index=True,
                            sort=False,
                        ),
                        "incremental",
                    )
            rebuild_mode = "invalidated_rebuild"
        else:
            rebuild_mode = "invalidated_rebuild"
    return (
        _bootstrap_signal_factor_cache(
            prepared,
            symbol,
            factor_root,
            reconcile_end=state_date,
        ),
        rebuild_mode,
    )


def attach_daily_signal_factors(
    daily: pd.DataFrame,
    symbol: str,
    factor_root: Path = DEFAULT_FACTOR_ROOT,
    persist_missing: bool = True,
) -> pd.DataFrame:
    """Attach exact signal factors, advancing versioned state when possible."""

    factor_root = Path(os.getenv("DAILY_FACTOR_ROOT", str(factor_root)))
    prepared = _prepare_daily(daily, symbol)
    if prepared.empty:
        return prepared
    if persist_missing:
        with _signal_cache_lock(factor_root, symbol):
            cached, cache_mode = _refresh_signal_factor_cache(
                prepared,
                symbol,
                factor_root,
            )
    else:
        cached = calculate_daily_signal_factors(prepared, symbol)
        cache_mode = "memory_full"
    if cached.empty:
        return prepared

    current_price = build_continuous_ohlc(prepared)
    current_basis = pd.DataFrame(
        {
            "date": prepared["date"],
            "current_basis_close": pd.to_numeric(
                current_price["close"],
                errors="coerce",
            ),
        }
    )
    factors = cached[
        [
            "date",
            "factor_basis_close",
            *[
                column
                for column in SIGNAL_FACTOR_COLUMNS
                if column in cached.columns
            ],
        ]
    ].merge(current_basis, on="date", how="inner")
    scale = (
        factors["current_basis_close"]
        / factors["factor_basis_close"].replace(0, np.nan)
    )
    for column in SIGNAL_PRICE_FACTOR_COLUMNS:
        if column in factors.columns:
            factors[column] = factors[column] * scale
    feature_columns = [
        column
        for column in SIGNAL_FACTOR_COLUMNS
        if column in factors.columns
    ]
    existing = [
        column
        for column in feature_columns
        if column in prepared.columns
    ]
    if existing:
        prepared = prepared.drop(columns=existing)
    result = prepared.merge(
        factors[["date", *feature_columns]],
        on="date",
        how="left",
    )
    result.attrs["signal_factor_cache_mode"] = cache_mode
    return result


def attach_z_skill_base_factors(
    daily: pd.DataFrame,
    symbol: str,
    factor_root: Path = DEFAULT_FACTOR_ROOT,
    persist_missing: bool = True,
) -> pd.DataFrame:
    """Attach the shared layer and expose legacy z-skill field names."""

    out = attach_daily_base_factors(
        daily,
        symbol,
        factor_root=factor_root,
        compute_if_missing=True,
        persist_missing=persist_missing,
    )
    for target, source in Z_CONSUMER_ALIASES.items():
        if source in out.columns:
            out[target] = out[source]
    return out


def _write_year_partition(frame: pd.DataFrame, factor_root: Path, symbol: str, year: int) -> Path:
    directory = factor_symbol_dir(factor_root, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{year}.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values("date").drop_duplicates(["symbol", "date"], keep="last")
    temp_path = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    combined.to_parquet(temp_path, index=False)
    os.replace(temp_path, path)
    return path


def refresh_symbol_factor_cache(
    daily_path: Path,
    factor_root: Path,
    incremental_start_date: str | pd.Timestamp,
    source_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    daily = source_frame.copy() if source_frame is not None else pd.read_parquet(daily_path)
    symbol = daily_path.stem
    for column in ("ts_code", "symbol"):
        if column in daily.columns:
            values = daily[column].dropna().astype(str)
            if not values.empty:
                symbol = values.iloc[0]
                break
    start = pd.Timestamp(incremental_start_date)
    prepared = _prepare_daily(daily, symbol)
    target_dates = prepared.loc[prepared["date"] >= start, "date"]
    cached = load_daily_base_factors(
        symbol,
        factor_root=factor_root,
        start_date=start,
        end_date=prepared["date"].max(),
    )
    if _cache_covers(cached, target_dates):
        return {
            "symbol": symbol,
            "rows": int(len(target_dates)),
            "date_max": target_dates.max().strftime("%Y-%m-%d") if not target_dates.empty else None,
            "paths": [],
            "cache_hit": True,
            "elapsed_seconds": perf_counter() - started,
        }
    factors = calculate_daily_base_factors(prepared, symbol)
    recent = factors[factors["date"] >= start].copy()
    paths: list[str] = []
    for year, group in recent.groupby(recent["date"].dt.year):
        paths.append(str(_write_year_partition(group, factor_root, symbol, int(year))))
    return {
        "symbol": symbol,
        "rows": int(len(recent)),
        "date_max": recent["date"].max().strftime("%Y-%m-%d") if not recent.empty else None,
        "paths": paths,
        "cache_hit": False,
        "elapsed_seconds": perf_counter() - started,
    }


def refresh_daily_factor_layer(
    daily_dir: Path,
    factor_root: Path = DEFAULT_FACTOR_ROOT,
    incremental_start_date: str | pd.Timestamp = "2020-01-01",
    workers: int = 8,
    executor_type: str = "processes",
    limit: int | None = None,
) -> dict[str, Any]:
    """Idempotently refresh the shared layer and write a freshness manifest."""

    daily_dir = Path(daily_dir)
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    market = store.read_market_range(daily_dir.name)
    tasks: list[tuple[Path, pd.DataFrame | None]]
    if not market.empty and "ts_code" in market.columns:
        grouped = [
            (daily_dir / f"{symbol}.parquet", group.reset_index(drop=True))
            for symbol, group in market.groupby("ts_code", sort=True)
        ]
        tasks = grouped[:limit] if limit else grouped
    else:
        suffixes = (".SZ.parquet", ".SH.parquet", ".BJ.parquet")
        files = [
            path
            for path in sorted(daily_dir.glob("*.parquet"))
            if path.name.endswith(suffixes)
        ]
        if limit:
            files = files[:limit]
        tasks = [(path, None) for path in files]
    if not tasks:
        raise RuntimeError(f"No standard daily data found in {daily_dir}")
    executor_cls = ProcessPoolExecutor if executor_type == "processes" else ThreadPoolExecutor
    results: list[dict[str, Any]] = []
    started = perf_counter()
    with executor_cls(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                refresh_symbol_factor_cache,
                path,
                Path(factor_root),
                incremental_start_date,
                source_frame,
            )
            for path, source_frame in tasks
        ]
        for n, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if n % 500 == 0 or n == len(futures):
                print(f"daily factor layer: {n}/{len(futures)} symbols", flush=True)
    max_dates = [item["date_max"] for item in results if item["date_max"]]
    manifest = {
        "status": "success",
        "factor_version": FACTOR_LAYER_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "incremental_start_date": str(pd.Timestamp(incremental_start_date).date()),
        "symbols": len(results),
        "rows": sum(int(item["rows"]) for item in results),
        "cache_hits": sum(bool(item.get("cache_hit")) for item in results),
        "date_max": max(max_dates) if max_dates else None,
        "elapsed_seconds": perf_counter() - started,
    }
    manifest_path = Path(factor_root) / FACTOR_LAYER_VERSION / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
