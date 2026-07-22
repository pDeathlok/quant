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

from quant.data.source_merge import normalize_tushare_daily
from quant.features.variable_library import (
    EXTRA_FEATURE_COLUMNS,
    build_continuous_ohlc,
    calculate_project_extra_features,
)


FACTOR_LAYER_VERSION = "v1"
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
) -> dict[str, Any]:
    started = perf_counter()
    daily = pd.read_parquet(daily_path)
    start = pd.Timestamp(incremental_start_date)
    prepared = _prepare_daily(daily, daily_path.stem)
    target_dates = prepared.loc[prepared["date"] >= start, "date"]
    cached = load_daily_base_factors(
        daily_path.stem,
        factor_root=factor_root,
        start_date=start,
        end_date=prepared["date"].max(),
    )
    if _cache_covers(cached, target_dates):
        return {
            "symbol": daily_path.stem,
            "rows": int(len(target_dates)),
            "date_max": target_dates.max().strftime("%Y-%m-%d") if not target_dates.empty else None,
            "paths": [],
            "cache_hit": True,
            "elapsed_seconds": perf_counter() - started,
        }
    factors = calculate_daily_base_factors(prepared, daily_path.stem)
    recent = factors[factors["date"] >= start].copy()
    paths: list[str] = []
    for year, group in recent.groupby(recent["date"].dt.year):
        paths.append(str(_write_year_partition(group, factor_root, daily_path.stem, int(year))))
    return {
        "symbol": daily_path.stem,
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

    suffixes = (".SZ.parquet", ".SH.parquet", ".BJ.parquet")
    files = [path for path in sorted(Path(daily_dir).glob("*.parquet")) if path.name.endswith(suffixes)]
    if limit:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No standard daily files found in {daily_dir}")
    executor_cls = ProcessPoolExecutor if executor_type == "processes" else ThreadPoolExecutor
    results: list[dict[str, Any]] = []
    started = perf_counter()
    with executor_cls(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(refresh_symbol_factor_cache, path, Path(factor_root), incremental_start_date)
            for path in files
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
