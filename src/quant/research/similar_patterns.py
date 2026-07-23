"""Historical similar-pattern retrieval and distributional stock forecasts."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from quant.data import list_partitioned_symbol_paths, read_partitioned_symbol_file


@dataclass(frozen=True)
class SimilarPatternConfig:
    cache_schema_version: int = 5
    lookback_days: int = 60
    weekly_lookback: int = 26
    monthly_lookback: int = 12
    volume_price_interaction_days: int = 60
    min_history_days: int = 260
    forward_days: tuple[int, ...] = (1, 20, 60)
    max_candidates_per_symbol: int = 120
    candidate_step_days: int = 5
    candidate_start_date: str | None = None
    similarity_threshold: float | None = None
    take_profit_3d: float = 0.03
    stop_loss_3d: float = 0.03
    top_k: int = 80
    min_candidate_rows: int = 500
    signal_bearish_max: float = 45.0
    signal_bullish_min: float = 55.0
    min_effective_cases: int = 100
    max_effective_cases: int = 800
    max_events_per_date: int = 3
    similarity_weight_power: float = 2.0
    same_industry_weight: float = 1.25
    cross_industry_weight: float = 0.85
    same_regime_weight: float = 1.15
    regime_mismatch_weight: float = 0.75
    same_industry_regime_weight: float = 1.10
    industry_regime_mismatch_weight: float = 0.80
    recency_half_life_days: int = 1095
    transaction_cost: float = 0.001
    enable_risk_gate: bool = True


@dataclass(frozen=True)
class TargetSpec:
    symbol: str
    name: str
    target_date: pd.Timestamp


@dataclass(frozen=True)
class SimilarPatternResult:
    target: TargetSpec
    latest_snapshot: dict[str, float | str | None]
    similar_cases: pd.DataFrame
    forecast: pd.DataFrame
    status_probs: dict[str, float]
    t1_scenario_plan: pd.DataFrame | None = None
    sell_model_plan: pd.DataFrame | None = None
    sell_model_summary: dict[str, int | float | str | None] | None = None
    match_mode: str = "top_k"
    scan_summary: dict[str, int | float | str | None] | None = None


def normalize_daily_frame(frame: pd.DataFrame, symbol_hint: str | None = None) -> pd.DataFrame:
    """Normalize local Tushare/cache daily frames into ascending OHLCV rows."""
    if frame.empty:
        return frame
    out = frame.copy()
    if "trade_date" in out.columns:
        trade_dates = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        if "date" in out.columns:
            date_values = pd.to_datetime(out["date"], errors="coerce")
            out["date"] = date_values.fillna(trade_dates)
        else:
            out["date"] = trade_dates
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        raise ValueError("daily frame requires date or trade_date")

    if "volume" in out.columns and "vol" in out.columns:
        out["volume"] = out["volume"].fillna(out["vol"])
    elif "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    elif "volume" not in out.columns:
        out["volume"] = np.nan
    if "pct_change" not in out.columns:
        if "pct_chg" in out.columns:
            out["pct_change"] = out["pct_chg"]
        else:
            out["pct_change"] = out["close"].pct_change() * 100
    if "symbol" not in out.columns:
        out["symbol"] = symbol_hint or out.get("ts_code", pd.Series([""])).iloc[0]
    if "name" not in out.columns:
        out["name"] = ""

    cols = ["date", "symbol", "name", "open", "high", "low", "close", "volume", "amount", "pct_change"]
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    out = out[cols].dropna(subset=["date", "open", "high", "low", "close"]).copy()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["symbol"] = out["symbol"].fillna(symbol_hint or "").astype(str)
    out = _continuous_ohlc_from_pct_change(out)
    return out


def _continuous_ohlc_from_pct_change(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove ex-right jumps using Tushare's continuous pct_change series."""
    out = frame.copy()
    pct_change = pd.to_numeric(out["pct_change"], errors="coerce") / 100.0
    raw_close = pd.to_numeric(out["close"], errors="coerce")
    if len(out) < 2 or pct_change.notna().sum() < 2 or raw_close.iloc[-1] <= 0:
        return out
    growth = 1.0 + pct_change
    raw_growth = raw_close.pct_change() + 1.0
    growth = growth.where(growth.gt(0) & np.isfinite(growth), raw_growth)
    growth.iloc[0] = 1.0
    continuous = growth.fillna(1.0).cumprod()
    continuous = continuous / continuous.iloc[-1] * raw_close.iloc[-1]
    scale = continuous / raw_close.replace(0, np.nan)
    for column in ["open", "high", "low", "close"]:
        out[column] = pd.to_numeric(out[column], errors="coerce") * scale
    return out


def load_stock_basic(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ts_code", "name", "industry", "list_date"])
    basic = pd.read_parquet(path).copy()
    return basic[["ts_code", "name", "industry", "list_date"]].drop_duplicates("ts_code")


def load_daily_file(path: Path) -> pd.DataFrame:
    return normalize_daily_frame(read_partitioned_symbol_file(path), path.stem)


def vector_cache_key(config: SimilarPatternConfig) -> str:
    payload = {
        "cache_schema_version": config.cache_schema_version,
        "lookback_days": config.lookback_days,
        "weekly_lookback": config.weekly_lookback,
        "monthly_lookback": config.monthly_lookback,
        "volume_price_interaction_days": config.volume_price_interaction_days,
        "min_history_days": config.min_history_days,
        "forward_days": config.forward_days,
        "candidate_step_days": config.candidate_step_days,
        "candidate_start_date": config.candidate_start_date,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def vector_cache_path(cache_dir: Path, symbol: str, config: SimilarPatternConfig) -> Path:
    safe_symbol = symbol.replace(".", "_")
    return cache_dir / vector_cache_key(config) / f"{safe_symbol}.npz"


def partitioned_daily_source_fingerprint(daily_dir: Path) -> str | None:
    """Return one stable fingerprint for the cross-sectional parquet partitions."""
    partition_root = daily_dir.parent / f"{daily_dir.name}_partitioned"
    partition_paths = sorted(partition_root.glob("year_month=*/data.parquet"))
    if not partition_paths:
        return None
    digest = hashlib.sha1()
    for path in partition_paths:
        source_stat = path.stat()
        digest.update(path.parent.name.encode("utf-8"))
        digest.update(str(source_stat.st_mtime_ns).encode("ascii"))
        digest.update(str(source_stat.st_size).encode("ascii"))
    return f"partitioned:{digest.hexdigest()}"


def save_stock_vector_cache(
    cache_path: Path,
    symbol: str,
    name: str,
    industry: str,
    daily: pd.DataFrame,
    indices: list[int],
    matrix: np.ndarray,
    config: SimilarPatternConfig,
    source_mtime_ns: int,
    source_size: int,
    source_fingerprint: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dates = np.array([pd.Timestamp(daily.iloc[idx]["date"]).strftime("%Y-%m-%d") for idx in indices])
    close = daily["close"].to_numpy(dtype=float)
    high = daily["high"].to_numpy(dtype=float)
    low = daily["low"].to_numpy(dtype=float)
    volume = daily["volume"].replace(0, np.nan).to_numpy(dtype=float)
    volume_ma20 = pd.Series(volume).rolling(20).mean().to_numpy(dtype=float)
    future: dict[str, np.ndarray] = {}
    for days in config.forward_days:
        future[f"fwd_{days}d"] = np.array(
            [close[idx + days] / close[idx] - 1 if idx + days < len(close) else np.nan for idx in indices],
            dtype=np.float32,
        )
    max_forward = max(config.forward_days)
    future["fwd_1d_volume_ratio"] = np.array(
        [
            volume[idx + 1] / volume_ma20[idx]
            if idx + 1 < len(volume) and np.isfinite(volume[idx + 1]) and np.isfinite(volume_ma20[idx]) and volume_ma20[idx] > 0
            else np.nan
            for idx in indices
        ],
        dtype=np.float32,
    )
    future["max_runup_3d"] = np.array(
        [np.nanmax(high[idx + 1 : idx + 4]) / close[idx] - 1 if idx + 3 < len(high) else np.nan for idx in indices],
        dtype=np.float32,
    )
    future["max_drawdown_3d"] = np.array(
        [np.nanmin(low[idx + 1 : idx + 4]) / close[idx] - 1 if idx + 3 < len(low) else np.nan for idx in indices],
        dtype=np.float32,
    )
    future["max_drawdown_60d"] = np.array(
        [
            np.nanmin(close[idx + 1 : idx + max_forward + 1]) / close[idx] - 1
            if idx + max_forward < len(close)
            else np.nan
            for idx in indices
        ],
        dtype=np.float32,
    )
    future["max_runup_60d"] = np.array(
        [
            np.nanmax(close[idx + 1 : idx + max_forward + 1]) / close[idx] - 1
            if idx + max_forward < len(close)
            else np.nan
            for idx in indices
        ],
        dtype=np.float32,
    )
    np.savez(
        cache_path,
        symbol=np.array(symbol),
        name=np.array(name),
        industry=np.array(industry),
        indices=np.array(indices, dtype=np.int32),
        dates=dates,
        close=np.array([close[idx] for idx in indices], dtype=np.float32),
        vectors=matrix.astype(np.float32),
        source_mtime_ns=np.array(source_mtime_ns, dtype=np.int64),
        source_size=np.array(source_size, dtype=np.int64),
        source_fingerprint=np.array(source_fingerprint),
        **future,
    )


def load_stock_vector_cache(cache_path: Path) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "symbol": str(data["symbol"].item()),
            "name": str(data["name"].item()),
            "industry": str(data["industry"].item()),
            "indices": data["indices"],
            "dates": data["dates"],
            "close": data["close"],
            "vectors": data["vectors"],
            "fwd_1d": data["fwd_1d"],
            "fwd_1d_volume_ratio": data["fwd_1d_volume_ratio"],
            "fwd_20d": data["fwd_20d"],
            "fwd_60d": data["fwd_60d"],
            "max_runup_3d": data["max_runup_3d"],
            "max_drawdown_3d": data["max_drawdown_3d"],
            "max_drawdown_60d": data["max_drawdown_60d"],
            "max_runup_60d": data["max_runup_60d"],
            "source_mtime_ns": int(data["source_mtime_ns"].item())
            if "source_mtime_ns" in data.files
            else None,
            "source_size": int(data["source_size"].item())
            if "source_size" in data.files
            else None,
            "source_fingerprint": str(data["source_fingerprint"].item())
            if "source_fingerprint" in data.files
            else None,
        }


def build_stock_vector_cache(
    path: Path,
    info: dict[str, object],
    config: SimilarPatternConfig,
    cache_dir: Path,
    force: bool = False,
    source_fingerprint: str | None = None,
) -> dict[str, object]:
    symbol = path.stem
    cache_path = vector_cache_path(cache_dir, symbol, config)
    source_stat = path.stat() if path.exists() else None
    effective_source_fingerprint = source_fingerprint
    if effective_source_fingerprint is None and source_stat is not None:
        effective_source_fingerprint = (
            f"file:{source_stat.st_mtime_ns}:{source_stat.st_size}"
        )
    if effective_source_fingerprint is None:
        effective_source_fingerprint = partitioned_daily_source_fingerprint(path.parent)
    if cache_path.exists() and not force:
        cached = load_stock_vector_cache(cache_path)
        fingerprint_matches = (
            effective_source_fingerprint is not None
            and cached.get("source_fingerprint") == effective_source_fingerprint
        )
        legacy_file_stat_matches = (
            source_stat is not None
            and cached.get("source_fingerprint") is None
            and cached.get("source_mtime_ns") == source_stat.st_mtime_ns
            and cached.get("source_size") == source_stat.st_size
        )
        if fingerprint_matches or legacy_file_stat_matches:
            return {
                "symbol": symbol,
                "status": "cache_hit",
                "cache_path": str(cache_path),
                "vectors": int(cached["vectors"].shape[0]),
                "elapsed_sec": 0.0,
            }

    started = perf_counter()
    try:
        daily = load_daily_file(path)
        if len(daily) < config.min_history_days + min(config.forward_days) + 1:
            return {"symbol": symbol, "status": "too_short", "cache_path": str(cache_path), "vectors": 0, "elapsed_sec": 0.0}
        industry = str(info.get("industry", ""))
        if info.get("name"):
            daily["name"] = daily["name"].replace("", np.nan).fillna(str(info["name"]))
        name = stock_name_from_basic_or_daily(info, daily)
        if is_excluded_stock(name):
            return {"symbol": symbol, "status": "excluded", "cache_path": str(cache_path), "vectors": 0, "elapsed_sec": 0.0}
        weekly_close, monthly_close = resample_close_series(daily)
        indices, matrix = build_stock_candidate_matrix(daily, config, weekly_close, monthly_close)
        if len(indices) == 0:
            return {"symbol": symbol, "status": "no_vectors", "cache_path": str(cache_path), "vectors": 0, "elapsed_sec": 0.0}
        save_stock_vector_cache(
            cache_path,
            symbol,
            name,
            industry,
            daily,
            indices,
            matrix,
            config,
            source_mtime_ns=source_stat.st_mtime_ns if source_stat is not None else -1,
            source_size=source_stat.st_size if source_stat is not None else -1,
            source_fingerprint=effective_source_fingerprint or "unavailable",
        )
        return {
            "symbol": symbol,
            "status": "built",
            "cache_path": str(cache_path),
            "vectors": int(matrix.shape[0]),
            "elapsed_sec": round(perf_counter() - started, 4),
        }
    except Exception as exc:
        return {
            "symbol": symbol,
            "status": "error",
            "error": str(exc),
            "cache_path": str(cache_path),
            "vectors": 0,
            "elapsed_sec": round(perf_counter() - started, 4),
        }


def _build_stock_vector_cache_worker(
    args: tuple[str, dict[str, object], SimilarPatternConfig, str, bool, str | None],
) -> dict[str, object]:
    path_text, info, config, cache_dir_text, force, source_fingerprint = args
    return build_stock_vector_cache(
        Path(path_text),
        info,
        config,
        Path(cache_dir_text),
        force,
        source_fingerprint,
    )


def _should_use_thread_pool_for_vector_cache() -> bool:
    """Avoid nested process pools when the current worker cannot safely spawn children."""
    try:
        return bool(mp.current_process().daemon)
    except Exception:
        return False


def build_vector_caches_parallel(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    cache_dir: Path,
    target_symbols: set[str] | None = None,
    max_symbols: int | None = None,
    workers: int = 1,
    force: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    files = list_partitioned_symbol_paths(daily_dir)
    if max_symbols is not None:
        files = files[:max_symbols]
    target_symbols = {symbol.upper() for symbol in (target_symbols or set())}
    files = [path for path in files if path.stem.upper() not in target_symbols]
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    source_fingerprint = partitioned_daily_source_fingerprint(daily_dir)
    tasks = [
        (
            str(path),
            basic_map.get(path.stem, {}),
            config,
            str(cache_dir),
            force,
            source_fingerprint,
        )
        for path in files
    ]

    records: list[dict[str, object]] = []
    started = perf_counter()
    if workers <= 1:
        for n, task in enumerate(tasks, start=1):
            records.append(_build_stock_vector_cache_worker(task))
            if n % 200 == 0 or n == len(tasks):
                built = sum(1 for record in records if record.get("status") in {"built", "cache_hit"})
                message = f"vector cache {n}/{len(tasks)} files usable={built:,} elapsed={perf_counter() - started:.1f}s"
                print(f"  {message}", flush=True)
                if progress_callback:
                    progress_callback(message)
    else:
        executor_cls = ThreadPoolExecutor if _should_use_thread_pool_for_vector_cache() else ProcessPoolExecutor
        try:
            executor = executor_cls(max_workers=workers)
        except (AssertionError, BrokenPipeError, PermissionError):
            print("  process pool unavailable; falling back to thread pool", flush=True)
            executor_cls = ThreadPoolExecutor
            executor = executor_cls(max_workers=workers)
        with executor:
            futures = [executor.submit(_build_stock_vector_cache_worker, task) for task in tasks]
            for n, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if n % 200 == 0 or n == len(futures):
                    built = sum(1 for record in records if record.get("status") in {"built", "cache_hit"})
                    message = f"vector cache {n}/{len(futures)} files usable={built:,} elapsed={perf_counter() - started:.1f}s"
                    print(f"  {message}", flush=True)
                    if progress_callback:
                        progress_callback(message)
    return pd.DataFrame(records)


def latest_snapshot(daily: pd.DataFrame, idx: int) -> dict[str, float | str | None]:
    row = daily.iloc[idx]
    hist = daily.iloc[: idx + 1].copy()
    close = hist["close"]
    volume = hist["volume"].replace(0, np.nan)
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    high60 = close.rolling(60).max().iloc[-1]
    low60 = close.rolling(60).min().iloc[-1]
    vol20 = volume.rolling(20).mean().iloc[-1]
    return {
        "date": row["date"].strftime("%Y-%m-%d"),
        "close": _round(row["close"]),
        "ret_1d": _round(close.pct_change().iloc[-1] * 100),
        "ret_20d": _round(close.pct_change(20).iloc[-1] * 100),
        "ret_60d": _round(close.pct_change(60).iloc[-1] * 100),
        "drawdown_60d": _round((row["close"] / high60 - 1) * 100 if high60 else np.nan),
        "range_pos_60d": _round((row["close"] - low60) / (high60 - low60) if high60 != low60 else np.nan),
        "dist_ma20": _round((row["close"] / ma20 - 1) * 100 if ma20 else np.nan),
        "dist_ma60": _round((row["close"] / ma60 - 1) * 100 if ma60 else np.nan),
        "vol_ratio20": _round(row["volume"] / vol20 if vol20 else np.nan),
    }


def build_pattern_vector(
    daily: pd.DataFrame,
    end_idx: int,
    config: SimilarPatternConfig,
    weekly_close: pd.Series | None = None,
    monthly_close: pd.Series | None = None,
) -> np.ndarray | None:
    """Build a daily/weekly/monthly volume-price shape vector ending at end_idx."""
    if end_idx < config.min_history_days or end_idx >= len(daily):
        return None

    close_all = daily["close"].to_numpy(dtype=float)
    open_all = daily["open"].to_numpy(dtype=float)
    high_all = daily["high"].to_numpy(dtype=float)
    low_all = daily["low"].to_numpy(dtype=float)
    volume_all = daily["volume"].to_numpy(dtype=float)

    daily_close = close_all[end_idx - config.lookback_days : end_idx + 1]
    if len(daily_close) < config.lookback_days + 1 or np.any(daily_close[:-1] == 0):
        return None

    daily_ret = daily_close[1:] / daily_close[:-1] - 1
    close_path = daily_close[1:] / daily_close[0] - 1
    volume_tail = volume_all[end_idx - config.lookback_days + 1 : end_idx + 1].astype(float)
    volume_tail[volume_tail <= 0] = np.nan
    finite_volume = volume_tail[np.isfinite(volume_tail)]
    volume_fill = float(np.median(finite_volume)) if len(finite_volume) else 0.0
    volume_tail = np.nan_to_num(volume_tail, nan=volume_fill, posinf=volume_fill, neginf=0.0)
    vol_log = np.log1p(np.clip(volume_tail, 0, None))
    vol_z = _zscore(vol_log)
    volume_price_interaction = _zscore(daily_ret) * vol_z

    end_date = pd.Timestamp(daily["date"].iloc[end_idx])
    if weekly_close is None or monthly_close is None:
        weekly_close, monthly_close = resample_close_series(daily)
    weekly_hist = weekly_close[weekly_close.index <= end_date]
    monthly_hist = monthly_close[monthly_close.index <= end_date]
    if len(weekly_hist) < config.weekly_lookback + 1 or len(monthly_hist) < config.monthly_lookback + 1:
        return None
    weekly_ret = weekly_hist.pct_change().tail(config.weekly_lookback).to_numpy()
    monthly_ret = monthly_hist.pct_change().tail(config.monthly_lookback).to_numpy()

    price = close_all
    high = high_all
    low = low_all
    open_price = open_all
    if any(end_idx - window + 1 < 0 for window in (10, 20, 60, 120)):
        return None
    ma20 = np.nanmean(price[end_idx - 19 : end_idx + 1])
    ma60 = np.nanmean(price[end_idx - 59 : end_idx + 1])
    ma120 = np.nanmean(price[end_idx - 119 : end_idx + 1])
    max60 = np.nanmax(price[end_idx - 59 : end_idx + 1])
    min120 = np.nanmin(price[end_idx - 119 : end_idx + 1])
    max120 = np.nanmax(price[end_idx - 119 : end_idx + 1])
    ret_window = price[end_idx - 20 : end_idx + 1]
    ret_20_values = ret_window[1:] / ret_window[:-1] - 1
    range_20 = (high[end_idx - 19 : end_idx + 1] - low[end_idx - 19 : end_idx + 1]) / np.where(
        price[end_idx - 19 : end_idx + 1] == 0, np.nan, price[end_idx - 19 : end_idx + 1]
    )
    body_denominator = np.where(open_price[end_idx - 9 : end_idx + 1] == 0, np.nan, open_price[end_idx - 9 : end_idx + 1])
    body_10 = (price[end_idx - 9 : end_idx + 1] - open_price[end_idx - 9 : end_idx + 1]) / body_denominator
    range_denominator = max120 - min120
    candle = np.array(
        [
            price[end_idx] / price[end_idx - 5] - 1,
            price[end_idx] / price[end_idx - 20] - 1,
            price[end_idx] / price[end_idx - 60] - 1,
            price[end_idx] / price[end_idx - 120] - 1,
            price[end_idx] / ma20 - 1,
            price[end_idx] / ma60 - 1,
            price[end_idx] / ma120 - 1,
            price[end_idx] / max60 - 1,
            (price[end_idx] - min120) / range_denominator if range_denominator else np.nan,
            _nan_stat(ret_20_values, "std"),
            _nan_stat(range_20, "mean"),
            _nan_stat(body_10, "mean"),
        ],
        dtype=float,
    )

    vector = np.concatenate(
        [
            _zscore(daily_ret),
            _zscore(close_path),
            vol_z,
            _zscore(volume_price_interaction),
            _zscore(weekly_ret),
            _zscore(monthly_ret),
            _zscore(candle),
        ]
    )
    if not np.isfinite(vector).all():
        return None
    return vector.astype(np.float32)


def resample_close_series(daily: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    indexed = daily.set_index("date").sort_index()
    weekly_close = indexed["close"].resample("W-FRI").last().dropna()
    monthly_close = indexed["close"].resample("ME").last().dropna()
    return weekly_close, monthly_close


def make_candidate_row(
    daily: pd.DataFrame,
    end_idx: int,
    vector: np.ndarray,
    config: SimilarPatternConfig,
    industry: str = "",
) -> dict[str, object] | None:
    max_forward = max(config.forward_days)
    if end_idx + min(config.forward_days) >= len(daily):
        return None
    row = daily.iloc[end_idx]
    close = float(row["close"])
    if close <= 0:
        return None
    future = {}
    for days in config.forward_days:
        future[f"fwd_{days}d"] = (
            float(daily.iloc[end_idx + days]["close"] / close - 1)
            if end_idx + days < len(daily)
            else np.nan
        )
    volume = daily["volume"].replace(0, np.nan)
    vol_ma20 = volume.rolling(20).mean().iloc[end_idx]
    future["fwd_1d_volume_ratio"] = (
        float(volume.iloc[end_idx + 1] / vol_ma20)
        if np.isfinite(volume.iloc[end_idx + 1]) and np.isfinite(vol_ma20) and vol_ma20 > 0
        else np.nan
    )
    high_3d = daily.iloc[end_idx + 1 : end_idx + 4]["high"].astype(float)
    low_3d = daily.iloc[end_idx + 1 : end_idx + 4]["low"].astype(float)
    future["max_runup_3d"] = float(high_3d.max() / close - 1) if len(high_3d) == 3 else np.nan
    future["max_drawdown_3d"] = float(low_3d.min() / close - 1) if len(low_3d) == 3 else np.nan
    future_window = daily.iloc[end_idx + 1 : end_idx + max_forward + 1]["close"].astype(float)
    future["max_drawdown_60d"] = float(future_window.min() / close - 1) if len(future_window) == max_forward else np.nan
    future["max_runup_60d"] = float(future_window.max() / close - 1) if len(future_window) == max_forward else np.nan
    return {
        "symbol": str(row["symbol"]),
        "name": str(row.get("name", "")),
        "industry": industry,
        "date": pd.Timestamp(row["date"]),
        "close": close,
        "vector": vector,
        **future,
    }


def make_cached_candidate_row(
    cached: dict[str, object],
    position: int,
    distance: float,
    similarity: float,
) -> dict[str, object]:
    return {
        "symbol": cached["symbol"],
        "name": cached["name"],
        "industry": cached["industry"],
        "date": pd.Timestamp(str(cached["dates"][position])),
        "close": float(cached["close"][position]),
        "fwd_1d": float(cached["fwd_1d"][position]),
        "fwd_1d_volume_ratio": float(cached["fwd_1d_volume_ratio"][position]),
        "fwd_20d": float(cached["fwd_20d"][position]),
        "fwd_60d": float(cached["fwd_60d"][position]),
        "max_runup_3d": float(cached["max_runup_3d"][position]),
        "max_drawdown_3d": float(cached["max_drawdown_3d"][position]),
        "max_drawdown_60d": float(cached["max_drawdown_60d"][position]),
        "max_runup_60d": float(cached["max_runup_60d"][position]),
        "distance": distance,
        "similarity": similarity,
    }


def candidate_end_indices(daily: pd.DataFrame, config: SimilarPatternConfig) -> range:
    max_end = len(daily) - min(config.forward_days) - 1
    if max_end < config.min_history_days:
        return range(0)
    if config.candidate_start_date:
        start_date = pd.Timestamp(config.candidate_start_date)
        eligible = daily.index[daily["date"] >= start_date]
        first_date_idx = int(eligible[0]) if len(eligible) else max_end + 1
        start_end = max(config.min_history_days, first_date_idx)
    else:
        start_end = max(
            config.min_history_days,
            max_end - config.max_candidates_per_symbol * config.candidate_step_days,
        )
    if start_end > max_end:
        return range(0)
    return range(start_end, max_end + 1, max(1, config.candidate_step_days))


def build_stock_candidate_matrix(
    daily: pd.DataFrame,
    config: SimilarPatternConfig,
    weekly_close: pd.Series | None = None,
    monthly_close: pd.Series | None = None,
) -> tuple[list[int], np.ndarray]:
    """Build one stock's candidate vectors so target comparisons can be vectorized."""
    if weekly_close is None or monthly_close is None:
        weekly_close, monthly_close = resample_close_series(daily)
    indices: list[int] = []
    vectors: list[np.ndarray] = []
    for end_idx in candidate_end_indices(daily, config):
        vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
        if vector is None:
            continue
        indices.append(end_idx)
        vectors.append(vector)
    if not vectors:
        return [], np.empty((0, 0), dtype=np.float32)
    return indices, np.vstack(vectors).astype(np.float32)


def select_best_positions_from_contiguous_matches(
    candidate_indices: list[int],
    similarities: np.ndarray,
    threshold: float,
    candidate_step_days: int,
) -> list[int]:
    """Keep the best match in each contiguous threshold-passing run for one stock."""
    passing_positions = np.flatnonzero(similarities >= threshold)
    if len(passing_positions) == 0:
        return []

    max_gap = max(1, candidate_step_days)
    selected: list[int] = []
    run_positions: list[int] = [int(passing_positions[0])]
    previous_idx = candidate_indices[int(passing_positions[0])]

    for raw_position in passing_positions[1:]:
        position = int(raw_position)
        current_idx = candidate_indices[position]
        if current_idx - previous_idx <= max_gap:
            run_positions.append(position)
        else:
            best = max(run_positions, key=lambda pos: float(similarities[pos]))
            selected.append(best)
            run_positions = [position]
        previous_idx = current_idx

    best = max(run_positions, key=lambda pos: float(similarities[pos]))
    selected.append(best)
    return selected


def stock_name_from_basic_or_daily(info: dict[str, object], daily: pd.DataFrame) -> str:
    if info.get("name"):
        return str(info["name"])
    if "name" in daily.columns and daily["name"].notna().any():
        return str(daily["name"].dropna().iloc[-1])
    return ""


def is_excluded_stock(name: str) -> bool:
    return "ST" in name.upper() or "退" in name


def build_candidate_library(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    target_symbols: set[str],
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Build historical pattern candidates from local daily parquet files."""
    files = list_partitioned_symbol_paths(daily_dir)
    if max_symbols is not None:
        files = files[:max_symbols]
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    rows: list[dict[str, object]] = []
    started = perf_counter()
    for n, path in enumerate(files, start=1):
        try:
            daily = load_daily_file(path)
        except Exception as exc:
            print(f"skip {path.name}: {exc}", flush=True)
            continue
        if len(daily) < config.min_history_days + max(config.forward_days) + 1:
            continue
        symbol = path.stem
        info = basic_map.get(symbol, {})
        industry = str(info.get("industry", ""))
        if info.get("name"):
            daily["name"] = daily["name"].replace("", np.nan).fillna(str(info["name"]))
        name = stock_name_from_basic_or_daily(info, daily)
        if is_excluded_stock(name):
            continue

        weekly_close, monthly_close = resample_close_series(daily)
        for end_idx in candidate_end_indices(daily, config):
            vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
            if vector is None:
                continue
            row = make_candidate_row(daily, end_idx, vector, config, industry)
            if row is not None:
                rows.append(row)

        if n % 500 == 0:
            elapsed = perf_counter() - started
            print(f"  library scan {n}/{len(files)} files, candidates={len(rows):,}, elapsed={elapsed:.1f}s", flush=True)

    library = pd.DataFrame(rows)
    if library.empty:
        return library
    target_symbols = {symbol.upper() for symbol in target_symbols}
    library = library[~library["symbol"].str.upper().isin(target_symbols)].reset_index(drop=True)
    return library


def analyze_target(
    symbol: str,
    daily: pd.DataFrame,
    library: pd.DataFrame,
    config: SimilarPatternConfig,
    basic: pd.DataFrame,
    target_date: str | None = None,
) -> SimilarPatternResult:
    if target_date:
        target_ts = pd.Timestamp(target_date)
        eligible = daily[daily["date"] <= target_ts]
        if eligible.empty:
            raise ValueError(f"{symbol} has no rows on or before {target_date}")
        end_idx = int(eligible.index[-1])
    else:
        end_idx = len(daily) - 1
        target_ts = pd.Timestamp(daily.iloc[end_idx]["date"])

    weekly_close, monthly_close = resample_close_series(daily)
    vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
    if vector is None:
        raise ValueError(f"{symbol} has insufficient history for target date {target_ts.date()}")
    if len(library) < config.min_candidate_rows:
        raise ValueError(f"candidate library too small: {len(library)} rows")

    matrix = np.vstack(library["vector"].to_numpy())
    distances = np.linalg.norm(matrix - vector, axis=1)
    similarity = 1 / (1 + distances)
    top_idx = np.argsort(distances)[: config.top_k]
    cases = library.iloc[top_idx].drop(columns=["vector"]).copy()
    cases["distance"] = distances[top_idx]
    cases["similarity"] = similarity[top_idx]
    cases = cases.sort_values(["distance", "date"]).reset_index(drop=True)
    cases["rank"] = np.arange(1, len(cases) + 1)

    forecast = summarize_forecast(cases)
    status_probs = summarize_status_probs(cases)
    basic_row = basic[basic["ts_code"] == symbol]
    name = str(basic_row["name"].iloc[0]) if not basic_row.empty else str(daily["name"].dropna().iloc[-1])
    target = TargetSpec(symbol=symbol, name=name, target_date=target_ts)
    return add_trade_plans(SimilarPatternResult(
        target=target,
        latest_snapshot=latest_snapshot(daily, end_idx),
        similar_cases=cases,
        forecast=forecast,
        status_probs=status_probs,
        match_mode="top_k",
        scan_summary={"library_rows": int(len(library)), "top_k": int(config.top_k)},
    ), config)


def prepare_target_context(
    symbol: str,
    daily: pd.DataFrame,
    config: SimilarPatternConfig,
    basic: pd.DataFrame,
    target_date: str | None = None,
) -> dict[str, object]:
    if target_date:
        target_ts = pd.Timestamp(target_date)
        eligible = daily[daily["date"] <= target_ts]
        if eligible.empty:
            raise ValueError(f"{symbol} has no rows on or before {target_date}")
        end_idx = int(eligible.index[-1])
    else:
        end_idx = len(daily) - 1
        target_ts = pd.Timestamp(daily.iloc[end_idx]["date"])
    weekly_close, monthly_close = resample_close_series(daily)
    vector = build_pattern_vector(daily, end_idx, config, weekly_close, monthly_close)
    if vector is None:
        raise ValueError(f"{symbol} has insufficient history for target date {target_ts.date()}")
    basic_row = basic[basic["ts_code"] == symbol]
    name = str(basic_row["name"].iloc[0]) if not basic_row.empty else str(daily["name"].dropna().iloc[-1])
    return {
        "symbol": symbol,
        "target": TargetSpec(symbol=symbol, name=name, target_date=target_ts),
        "snapshot": latest_snapshot(daily, end_idx),
        "vector": vector,
    }


def analyze_targets_by_threshold(
    daily_dir: Path,
    basic: pd.DataFrame,
    config: SimilarPatternConfig,
    target_symbols: list[str],
    target_date: str | None = None,
    max_symbols: int | None = None,
    vector_cache_dir: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, SimilarPatternResult]:
    if config.similarity_threshold is None:
        raise ValueError("similarity_threshold is required for threshold mode")

    files = list_partitioned_symbol_paths(daily_dir)
    symbol_paths = {path.stem.upper(): path for path in files}
    target_contexts: dict[str, dict[str, object]] = {}
    for symbol in target_symbols:
        target_path = symbol_paths.get(symbol.upper())
        if target_path is None:
            print(f"missing target daily data: {symbol}", flush=True)
            continue
        daily = load_daily_file(target_path)
        target_contexts[symbol] = prepare_target_context(symbol, daily, config, basic, target_date)
    if not target_contexts:
        return {}

    target_vectors = {symbol: context["vector"] for symbol, context in target_contexts.items()}
    target_symbol_set = {symbol.upper() for symbol in target_contexts}
    target_matches: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in target_contexts}

    if max_symbols is not None:
        files = files[:max_symbols]
    basic_map = basic.set_index("ts_code").to_dict("index") if not basic.empty else {}
    scanned_candidates = 0
    started = perf_counter()

    for n, path in enumerate(files, start=1):
        symbol = path.stem
        if symbol.upper() in target_symbol_set:
            continue
        cached: dict[str, object] | None = None
        daily: pd.DataFrame | None = None
        if vector_cache_dir is not None:
            cache_path = vector_cache_path(vector_cache_dir, symbol, config)
            if not cache_path.exists():
                continue
            cached = load_stock_vector_cache(cache_path)
            candidate_indices = [int(value) for value in cached["indices"]]
            candidate_matrix = cached["vectors"]
        else:
            try:
                daily = load_daily_file(path)
            except Exception as exc:
                print(f"skip {path.name}: {exc}", flush=True)
                continue
            if len(daily) < config.min_history_days + max(config.forward_days) + 1:
                continue
            info = basic_map.get(symbol, {})
            industry = str(info.get("industry", ""))
            if info.get("name"):
                daily["name"] = daily["name"].replace("", np.nan).fillna(str(info["name"]))
            name = stock_name_from_basic_or_daily(info, daily)
            if is_excluded_stock(name):
                continue

            weekly_close, monthly_close = resample_close_series(daily)
            candidate_indices, candidate_matrix = build_stock_candidate_matrix(daily, config, weekly_close, monthly_close)
        if len(candidate_indices) == 0:
            continue
        scanned_candidates += len(candidate_indices)

        for target_symbol, target_vector in target_vectors.items():
            deltas = candidate_matrix - target_vector
            distances = np.linalg.norm(deltas, axis=1)
            similarities = 1 / (1 + distances)
            selected_positions = select_best_positions_from_contiguous_matches(
                candidate_indices,
                similarities,
                config.similarity_threshold,
                config.candidate_step_days,
            )
            for position in selected_positions:
                if cached is not None:
                    row = make_cached_candidate_row(
                        cached,
                        position,
                        float(distances[position]),
                        float(similarities[position]),
                    )
                else:
                    end_idx = candidate_indices[position]
                    row = make_candidate_row(daily, end_idx, candidate_matrix[position], config, industry)
                    if row is None:
                        continue
                    row.pop("vector", None)
                    row["distance"] = float(distances[position])
                    row["similarity"] = float(similarities[position])
                target_matches[target_symbol].append(row)

        if n % 500 == 0:
            elapsed = perf_counter() - started
            counts = ", ".join(f"{symbol}={len(rows):,}" for symbol, rows in target_matches.items())
            message = (
                f"threshold scan {n}/{len(files)} files, candidates={scanned_candidates:,}, "
                f"matches: {counts}, elapsed={elapsed:.1f}s"
            )
            print(f"  {message}", flush=True)
            if progress_callback:
                progress_callback(message)

    results: dict[str, SimilarPatternResult] = {}
    for symbol, context in target_contexts.items():
        cases = pd.DataFrame(target_matches[symbol])
        if cases.empty:
            cases = pd.DataFrame(
                columns=[
                    "symbol",
                    "name",
                    "industry",
                    "date",
                    "close",
                    "fwd_1d",
                    "fwd_20d",
                    "fwd_60d",
                    "max_drawdown_60d",
                    "max_runup_60d",
                    "distance",
                    "similarity",
                ]
            )
            forecast = summarize_forecast(cases)
            status_probs = {"上升": 0.0, "震荡": 0.0, "下跌": 0.0, "高波动": 0.0}
        else:
            cases = cases.sort_values(["similarity", "date"], ascending=[False, True]).reset_index(drop=True)
            cases["rank"] = np.arange(1, len(cases) + 1)
            forecast = summarize_forecast(cases)
            status_probs = summarize_status_probs(cases)
        results[symbol] = add_trade_plans(SimilarPatternResult(
            target=context["target"],
            latest_snapshot=context["snapshot"],
            similar_cases=cases,
            forecast=forecast,
            status_probs=status_probs,
            match_mode="threshold",
            scan_summary={
                "candidate_start_date": config.candidate_start_date,
                "candidate_step_days": int(config.candidate_step_days),
                "similarity_threshold": float(config.similarity_threshold),
                "scanned_candidates": int(scanned_candidates),
                "matched_cases": int(len(cases)),
                "contiguous_dedupe": "best_per_stock_run",
                "vector_cache": str(vector_cache_dir) if vector_cache_dir else None,
                "max_symbols": max_symbols,
            },
        ), config)
    return results


def optimize_similar_cases(
    cases: pd.DataFrame,
    config: SimilarPatternConfig,
    *,
    target_date: pd.Timestamp,
    target_industry: str,
    target_market_regime: str,
    target_industry_regime: str = "",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Deduplicate correlated events and assign nonlinear forecast weights."""
    raw_cases = int(len(cases))
    if cases.empty:
        return cases.copy(), {
            "raw_cases": 0,
            "deduplicated_cases": 0,
            "effective_sample_size": 0.0,
            "sample_status": "insufficient",
        }

    out = cases.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["industry"] = out.get("industry", "").fillna("").astype(str)
    out["similarity"] = pd.to_numeric(out["similarity"], errors="coerce")
    out = out.dropna(subset=["date", "similarity"])
    out = out.sort_values(["similarity", "date"], ascending=[False, False])
    out = out.drop_duplicates(["date", "industry"], keep="first")
    out = (
        out.sort_values(["date", "similarity"], ascending=[False, False])
        .groupby("date", group_keys=False)
        .head(max(1, config.max_events_per_date))
    )

    threshold = float(config.similarity_threshold or 0.0)
    margins = (out["similarity"] - threshold).clip(lower=0.0001)
    max_margin = float(margins.max()) if not margins.empty else 0.0001
    base_weight = (margins / max(max_margin, 0.0001)).pow(config.similarity_weight_power)
    industry_factor = np.where(
        out["industry"].eq(str(target_industry)),
        config.same_industry_weight,
        config.cross_industry_weight,
    )
    if "market_regime" in out.columns and target_market_regime:
        regime_factor = np.where(
            out["market_regime"].fillna("neutral").astype(str).eq(str(target_market_regime)),
            config.same_regime_weight,
            config.regime_mismatch_weight,
        )
    else:
        regime_factor = np.ones(len(out), dtype=float)
    if "industry_regime" in out.columns and target_industry_regime:
        same_industry = out["industry"].eq(str(target_industry))
        industry_regime_factor = np.where(
            ~same_industry,
            1.0,
            np.where(
                out["industry_regime"].fillna("neutral").astype(str).eq(str(target_industry_regime)),
                config.same_industry_regime_weight,
                config.industry_regime_mismatch_weight,
            ),
        )
    else:
        industry_regime_factor = np.ones(len(out), dtype=float)
    ages = (pd.Timestamp(target_date) - out["date"]).dt.days.clip(lower=0)
    recency_factor = np.power(0.5, ages / max(1, config.recency_half_life_days))
    out["forecast_weight"] = (
        base_weight.to_numpy(dtype=float)
        * industry_factor
        * regime_factor
        * industry_regime_factor
        * recency_factor
    )
    out = out.sort_values(["forecast_weight", "similarity"], ascending=[False, False]).head(
        max(1, config.max_effective_cases)
    )
    weight = out["forecast_weight"].to_numpy(dtype=float)
    weight_sum = float(np.sum(weight))
    effective_sample_size = (
        float(weight_sum**2 / np.sum(np.square(weight)))
        if weight_sum > 0 and float(np.sum(np.square(weight))) > 0
        else 0.0
    )
    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out, {
        "raw_cases": raw_cases,
        "deduplicated_cases": int(len(out)),
        "effective_sample_size": round(effective_sample_size, 2),
        "sample_status": "ready" if effective_sample_size >= config.min_effective_cases else "insufficient",
        "max_events_per_date": int(config.max_events_per_date),
        "max_effective_cases": int(config.max_effective_cases),
        "weight_power": float(config.similarity_weight_power),
    }


def build_probability_variant_cases(
    cases: pd.DataFrame,
    config: SimilarPatternConfig,
    *,
    target_date: pd.Timestamp,
    target_industry: str,
    target_market_regime: str,
    target_industry_regime: str = "",
) -> dict[str, pd.DataFrame]:
    """Build transferable single-condition variants used by global model selection."""
    if cases.empty:
        return {name: cases.copy() for name in ("event_dedupe", "nonlinear", "regime_industry", "recency")}

    normalized = cases.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["industry"] = normalized.get("industry", "").fillna("").astype(str)
    normalized["similarity"] = pd.to_numeric(normalized["similarity"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "similarity"])

    event_dedupe = normalized.sort_values(["similarity", "date"], ascending=[False, False]).drop_duplicates(
        ["date", "industry"], keep="first"
    )
    event_dedupe = (
        event_dedupe.sort_values(["date", "similarity"], ascending=[False, False])
        .groupby("date", group_keys=False)
        .head(max(1, config.max_events_per_date))
    )

    nonlinear = normalized.copy()
    threshold = float(config.similarity_threshold or 0.0)
    margin = (nonlinear["similarity"] - threshold).clip(lower=0.0001)
    nonlinear["forecast_weight"] = np.square(margin / max(float(margin.max()), 0.0001))

    regime_industry = normalized.copy()
    weight = regime_industry["similarity"].fillna(0.0)
    weight *= np.where(
        regime_industry["industry"].eq(str(target_industry)),
        config.same_industry_weight,
        config.cross_industry_weight,
    )
    if "market_regime" in regime_industry.columns and target_market_regime:
        weight *= np.where(
            regime_industry["market_regime"].fillna("neutral").astype(str).eq(str(target_market_regime)),
            config.same_regime_weight,
            config.regime_mismatch_weight,
        )
    if "industry_regime" in regime_industry.columns and target_industry_regime:
        same_industry = regime_industry["industry"].eq(str(target_industry))
        weight *= np.where(
            ~same_industry,
            1.0,
            np.where(
                regime_industry["industry_regime"].fillna("neutral").astype(str).eq(str(target_industry_regime)),
                config.same_industry_regime_weight,
                config.industry_regime_mismatch_weight,
            ),
        )
    regime_industry["forecast_weight"] = weight

    recency = normalized.copy()
    ages = (pd.Timestamp(target_date) - recency["date"]).dt.days.clip(lower=0)
    recency["forecast_weight"] = recency["similarity"].fillna(0.0) * np.power(
        0.5,
        ages / max(1, config.recency_half_life_days),
    )
    return {
        "event_dedupe": event_dedupe,
        "nonlinear": nonlinear,
        "regime_industry": regime_industry,
        "recency": recency,
    }


def classify_forecast_signal(
    up_probability: float | None,
    snapshot: dict[str, float | str | None],
    market_regime: str,
    config: SimilarPatternConfig,
) -> dict[str, object]:
    """Classify a forecast and veto weak bullish signals during breakdowns."""
    if up_probability is None or not np.isfinite(float(up_probability)):
        return {"signal": "observe", "risk_gate": "missing", "reasons": ["缺少有效概率"]}
    probability = float(up_probability)
    if probability >= config.signal_bullish_min:
        signal = "bullish"
    elif probability <= config.signal_bearish_max:
        signal = "bearish"
    else:
        signal = "observe"

    def value(key: str) -> float:
        raw = snapshot.get(key)
        try:
            parsed = float(raw) if raw is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
        return parsed

    reasons: list[str] = []
    dist_ma20 = value("dist_ma20")
    dist_ma60 = value("dist_ma60")
    drawdown = value("drawdown_60d")
    volume_ratio = value("vol_ratio20")
    breakdown = (
        (np.isfinite(dist_ma20) and dist_ma20 <= -3.0)
        and (np.isfinite(dist_ma60) and dist_ma60 <= -5.0)
        and (
            (np.isfinite(drawdown) and drawdown <= -12.0)
            or (np.isfinite(volume_ratio) and volume_ratio >= 1.3)
        )
    )
    if breakdown:
        reasons.append("放量或深回撤破位")
    if market_regime == "risk_off":
        reasons.append("沪深300处于风险规避状态")
    blocked = config.enable_risk_gate and signal == "bullish" and (
        breakdown or (market_regime == "risk_off" and probability < 60.0)
    )
    return {
        "signal": "observe" if blocked else signal,
        "raw_signal": signal,
        "risk_gate": "blocked" if blocked else "passed",
        "reasons": reasons,
        "probability": round(probability, 2),
        "bearish_max": float(config.signal_bearish_max),
        "bullish_min": float(config.signal_bullish_min),
    }


def fit_probability_calibration(
    probabilities: list[float],
    outcomes: list[bool],
    *,
    min_samples: int = 20,
) -> dict[str, object]:
    """Fit a small monotonic reliability curve that is JSON serializable."""
    frame = pd.DataFrame({"probability": probabilities, "outcome": outcomes}).dropna()
    if len(frame) < min_samples or frame["outcome"].nunique() < 2:
        return {"status": "identity", "sample_count": int(len(frame)), "x": [0.0, 100.0], "y": [0.0, 100.0]}
    frame["bin"] = pd.qcut(frame["probability"], q=min(6, frame["probability"].nunique()), duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        x=("probability", "mean"),
        positives=("outcome", "sum"),
        count=("outcome", "size"),
    )
    grouped["y"] = (grouped["positives"] + 1.0) / (grouped["count"] + 2.0) * 100.0
    x_values = grouped["x"].astype(float).to_numpy()
    y_values = np.maximum.accumulate(grouped["y"].astype(float).to_numpy())
    x_values = np.r_[0.0, x_values, 100.0]
    y_values = np.r_[max(0.0, y_values[0] - 10.0), y_values, min(100.0, y_values[-1] + 10.0)]
    return {
        "status": "fitted",
        "sample_count": int(len(frame)),
        "x": [round(float(value), 4) for value in x_values],
        "y": [round(float(value), 4) for value in y_values],
    }


def apply_probability_calibration(
    probability: float | None,
    calibration: dict[str, object] | None,
) -> float | None:
    """Apply a serialized monotonic reliability curve."""
    if probability is None or not np.isfinite(float(probability)):
        return None
    if not calibration:
        return _round(float(probability))
    x_values = np.asarray(calibration.get("x") or [0.0, 100.0], dtype=float)
    y_values = np.asarray(calibration.get("y") or [0.0, 100.0], dtype=float)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return _round(float(probability))
    return _round(float(np.interp(float(probability), x_values, y_values)))


def summarize_forecast(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col, label in [("fwd_1d", "next_1d"), ("fwd_20d", "next_1m"), ("fwd_60d", "next_3m")]:
        if cases.empty or col not in cases.columns or "similarity" not in cases.columns:
            rows.append(
                {
                    "horizon": label,
                    "sample_count": 0,
                    "up_probability": None,
                    "mean_return": None,
                    "p10": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "p90": None,
                }
            )
            continue
        values = cases[col].astype(float)
        weight_col = "forecast_weight" if "forecast_weight" in cases.columns else "similarity"
        weights = cases[weight_col].astype(float)
        if values.notna().sum() == 0 or weights.sum() <= 0:
            rows.append(
                {
                    "horizon": label,
                    "sample_count": 0,
                    "up_probability": None,
                    "mean_return": None,
                    "p10": None,
                    "p25": None,
                    "median": None,
                    "p75": None,
                    "p90": None,
                }
            )
            continue
        rows.append(
            {
                "horizon": label,
                "sample_count": int(values.notna().sum()),
                "up_probability": _round(float((weights[values > 0].sum() / weights.sum()) * 100)),
                "mean_return": _round(float(np.average(values, weights=weights)) * 100),
                "p10": _round(float(values.quantile(0.10)) * 100),
                "p25": _round(float(values.quantile(0.25)) * 100),
                "median": _round(float(values.quantile(0.50)) * 100),
                "p75": _round(float(values.quantile(0.75)) * 100),
                "p90": _round(float(values.quantile(0.90)) * 100),
            }
        )
    return pd.DataFrame(rows)


def summarize_status_probs(cases: pd.DataFrame) -> dict[str, float]:
    if cases.empty:
        return {"上升": 0.0, "震荡": 0.0, "下跌": 0.0, "高波动": 0.0}
    returns = cases["fwd_60d"].astype(float)
    drawdowns = cases["max_drawdown_60d"].astype(float)
    statuses = np.select(
        [
            returns >= 0.10,
            (returns > -0.05) & (returns < 0.10) & (drawdowns > -0.12),
            returns <= -0.10,
        ],
        ["上升", "震荡", "下跌"],
        default="高波动",
    )
    weights = cases["similarity"].astype(float).to_numpy()
    total = weights.sum()
    if total <= 0:
        return {"上升": 0.0, "震荡": 0.0, "下跌": 0.0, "高波动": 0.0}
    return {
        status: _round(float(weights[statuses == status].sum() / total * 100))
        for status in ["上升", "震荡", "下跌", "高波动"]
    }


def add_trade_plans(result: SimilarPatternResult, config: SimilarPatternConfig) -> SimilarPatternResult:
    scenario_plan = build_t1_scenario_plan(result.similar_cases, config.take_profit_3d, config.stop_loss_3d)
    model_plan, model_summary = build_sell_model_plan(
        result.similar_cases,
        config.take_profit_3d,
        config.stop_loss_3d,
    )
    return SimilarPatternResult(
        target=result.target,
        latest_snapshot=result.latest_snapshot,
        similar_cases=result.similar_cases,
        forecast=result.forecast,
        status_probs=result.status_probs,
        t1_scenario_plan=scenario_plan,
        sell_model_plan=model_plan,
        sell_model_summary=model_summary,
        match_mode=result.match_mode,
        scan_summary=result.scan_summary,
    )


def classify_t1_return(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    percent = value * 100
    if percent <= -3:
        return "大跌<=-3%"
    if percent <= -1:
        return "小跌-3~-1%"
    if percent < 1:
        return "震荡-1~1%"
    if percent < 3:
        return "小涨1~3%"
    return "大涨>=3%"


def classify_volume_ratio(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < 0.8:
        return "缩量<0.8"
    if value <= 1.3:
        return "平量0.8~1.3"
    if value <= 2.0:
        return "放量1.3~2"
    return "巨量>2"


def action_from_probs(sample_count: int, up_prob: float, down_prob: float) -> str:
    if sample_count < 20:
        return "观察-样本不足"
    if down_prob >= 55 and down_prob > up_prob + 10:
        return "卖出/减仓"
    if up_prob >= 55 and down_prob <= 35:
        return "持有/可低吸"
    if down_prob >= 45:
        return "谨慎持有"
    return "持有观察"


def build_t1_scenario_plan(cases: pd.DataFrame, take_profit_3d: float, stop_loss_3d: float) -> pd.DataFrame:
    columns = [
        "t1_return_bucket",
        "t1_volume_bucket",
        "sample_count",
        "hit_up_3d_prob",
        "hit_down_3d_prob",
        "mean_fwd_20d",
        "median_fwd_20d",
        "action",
    ]
    required = {"fwd_1d", "fwd_1d_volume_ratio", "max_runup_3d", "max_drawdown_3d", "fwd_20d"}
    if cases.empty or not required.issubset(cases.columns):
        return pd.DataFrame(columns=columns)
    data = cases.copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["t1_return_bucket"] = data["fwd_1d"].map(classify_t1_return)
    data["t1_volume_bucket"] = data["fwd_1d_volume_ratio"].map(classify_volume_ratio)
    data["hit_up_3d"] = data["max_runup_3d"] >= take_profit_3d
    data["hit_down_3d"] = data["max_drawdown_3d"] <= -stop_loss_3d
    rows: list[dict[str, object]] = []
    for (return_bucket, volume_bucket), group in data.groupby(["t1_return_bucket", "t1_volume_bucket"], sort=True):
        up_prob = float(group["hit_up_3d"].mean() * 100)
        down_prob = float(group["hit_down_3d"].mean() * 100)
        sample_count = int(len(group))
        rows.append(
            {
                "t1_return_bucket": return_bucket,
                "t1_volume_bucket": volume_bucket,
                "sample_count": sample_count,
                "hit_up_3d_prob": _round(up_prob),
                "hit_down_3d_prob": _round(down_prob),
                "mean_fwd_20d": _round(float(group["fwd_20d"].mean()) * 100),
                "median_fwd_20d": _round(float(group["fwd_20d"].median()) * 100),
                "action": action_from_probs(sample_count, up_prob, down_prob),
            }
        )
    return pd.DataFrame(rows).sort_values(["t1_return_bucket", "t1_volume_bucket"]).reset_index(drop=True)


def build_sell_model_plan(
    cases: pd.DataFrame,
    take_profit_3d: float,
    stop_loss_3d: float,
) -> tuple[pd.DataFrame, dict[str, int | float | str | None]]:
    columns = [
        "scenario",
        "t1_return",
        "t1_volume_ratio",
        "model_hit_up_3d_prob",
        "model_hit_down_3d_prob",
        "recommendation",
    ]
    required = {"fwd_1d", "fwd_1d_volume_ratio", "max_runup_3d", "max_drawdown_3d", "similarity", "distance"}
    if cases.empty or not required.issubset(cases.columns):
        return pd.DataFrame(columns=columns), {"status": "insufficient_data", "sample_count": 0}

    data = cases.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).copy()
    if len(data) < 80:
        return pd.DataFrame(columns=columns), {"status": "insufficient_data", "sample_count": int(len(data))}

    feature_cols = ["fwd_1d", "fwd_1d_volume_ratio", "similarity", "distance"]
    x = data[feature_cols].to_numpy(dtype=float)
    y_up = (data["max_runup_3d"].to_numpy(dtype=float) >= take_profit_3d).astype(int)
    y_down = (data["max_drawdown_3d"].to_numpy(dtype=float) <= -stop_loss_3d).astype(int)
    if len(np.unique(y_up)) < 2 or len(np.unique(y_down)) < 2:
        return pd.DataFrame(columns=columns), {"status": "single_class_label", "sample_count": int(len(data))}

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return pd.DataFrame(columns=columns), {"status": "sklearn_missing", "sample_count": int(len(data))}

    up_model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=20, random_state=42)
    down_model = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=20, random_state=43)
    up_model.fit(x, y_up)
    down_model.fit(x, y_down)

    median_similarity = float(data["similarity"].median())
    median_distance = float(data["distance"].median())
    scenarios = [
        ("大跌缩量", -0.035, 0.7),
        ("大跌放量", -0.035, 1.6),
        ("小跌平量", -0.015, 1.0),
        ("震荡平量", 0.0, 1.0),
        ("小涨放量", 0.015, 1.6),
        ("大涨放量", 0.035, 1.8),
    ]
    rows = []
    for name, t1_ret, volume_ratio in scenarios:
        scenario_x = np.array([[t1_ret, volume_ratio, median_similarity, median_distance]], dtype=float)
        up_prob = float(up_model.predict_proba(scenario_x)[0, 1] * 100)
        down_prob = float(down_model.predict_proba(scenario_x)[0, 1] * 100)
        rows.append(
            {
                "scenario": name,
                "t1_return": _round(t1_ret * 100),
                "t1_volume_ratio": _round(volume_ratio),
                "model_hit_up_3d_prob": _round(up_prob),
                "model_hit_down_3d_prob": _round(down_prob),
                "recommendation": action_from_probs(len(data), up_prob, down_prob),
            }
        )
    summary = {
        "status": "trained",
        "sample_count": int(len(data)),
        "take_profit_3d": _round(take_profit_3d * 100),
        "stop_loss_3d": _round(stop_loss_3d * 100),
        "up_positive_rate": _round(float(y_up.mean() * 100)),
        "down_positive_rate": _round(float(y_down.mean() * 100)),
    }
    return pd.DataFrame(rows), summary


def write_result(result: SimilarPatternResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = result.target.symbol.replace(".", "_")
    cases_path = output_dir / f"{slug}_similar_cases.csv"
    forecast_path = output_dir / f"{slug}_forecast.csv"
    meta_path = output_dir / f"{slug}_summary.json"
    report_path = output_dir / f"{slug}_report.md"

    cases_out = result.similar_cases.copy()
    if "date" in cases_out.columns and not cases_out.empty:
        cases_out["date"] = pd.to_datetime(cases_out["date"]).dt.strftime("%Y-%m-%d")
    cases_out.to_csv(cases_path, index=False, encoding="utf-8-sig")
    result.forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")
    if result.t1_scenario_plan is not None:
        result.t1_scenario_plan.to_csv(output_dir / f"{slug}_t1_scenario_plan.csv", index=False, encoding="utf-8-sig")
    if result.sell_model_plan is not None:
        result.sell_model_plan.to_csv(output_dir / f"{slug}_sell_model_plan.csv", index=False, encoding="utf-8-sig")
    meta = {
        "target": {
            "symbol": result.target.symbol,
            "name": result.target.name,
            "target_date": result.target.target_date.strftime("%Y-%m-%d"),
        },
        "latest_snapshot": result.latest_snapshot,
        "status_probs": result.status_probs,
        "match_mode": result.match_mode,
        "scan_summary": result.scan_summary,
        "sell_model_summary": result.sell_model_summary,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_markdown_report(result), encoding="utf-8")
    return {"cases": cases_path, "forecast": forecast_path, "summary": meta_path, "report": report_path}


def render_markdown_report(result: SimilarPatternResult) -> str:
    top_cases = result.similar_cases.head(20).copy()
    display_cols = [
        "rank",
        "symbol",
        "name",
        "industry",
        "date",
        "similarity",
        "fwd_1d",
        "fwd_20d",
        "fwd_60d",
        "max_drawdown_60d",
    ]
    if top_cases.empty:
        top_cases = pd.DataFrame(columns=display_cols)
    else:
        top_cases = top_cases[display_cols].copy()
        for col in ["fwd_1d", "fwd_20d", "fwd_60d", "max_drawdown_60d"]:
            top_cases[col] = (top_cases[col].astype(float) * 100).round(2)
        top_cases["similarity"] = top_cases["similarity"].round(4)
        top_cases["date"] = pd.to_datetime(top_cases["date"]).dt.strftime("%Y-%m-%d")

    lines = [
        f"# {result.target.name}（{result.target.symbol}）历史相似走势预测报告",
        "",
        f"- 目标日期：{result.target.target_date.strftime('%Y-%m-%d')}",
        "- 数据口径：本地 `data/raw/daily` 日线，按量价形态、量价交互、周/月趋势构建相似向量。",
        f"- 匹配模式：{result.match_mode}",
        "- 说明：该报告是研究原型输出，不构成投资建议；后续应切换统一前复权数据并做滚动回测。",
        "",
    ]
    if result.scan_summary:
        lines.extend(
            [
                "## 扫描设置",
                "",
                pd.DataFrame([result.scan_summary]).to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
        "## 当前走势画像",
        "",
        pd.DataFrame([result.latest_snapshot]).to_markdown(index=False),
        "",
        "## 相似案例后验预测",
        "",
        result.forecast.to_markdown(index=False),
        "",
        "## 未来三个月阶段概率",
        "",
        pd.DataFrame([result.status_probs]).to_markdown(index=False),
        "",
        "## T+1 量价情景操作计划",
        "",
        (result.t1_scenario_plan if result.t1_scenario_plan is not None else pd.DataFrame()).to_markdown(index=False),
        "",
        "## 3日卖出/持有模型建议",
        "",
        pd.DataFrame([result.sell_model_summary or {}]).to_markdown(index=False),
        "",
        (result.sell_model_plan if result.sell_model_plan is not None else pd.DataFrame()).to_markdown(index=False),
        "",
        "## Top 20 历史相似片段",
        "",
        top_cases.to_markdown(index=False),
        "",
        ]
    )
    return "\n".join(lines)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return np.zeros_like(arr, dtype=float)
    mean = float(finite.mean())
    std = float(finite.std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _nan_stat(values: np.ndarray, kind: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    if kind == "std":
        return float(finite.std())
    return float(finite.mean())


def _round(value: object, ndigits: int = 2) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, ndigits)
