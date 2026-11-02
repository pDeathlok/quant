#!/usr/bin/env python
"""Materialize causal candidate factors for the left-side v5 research run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_json, atomic_write_parquet
from quant.features.active_market_value import ACTIVE_SHARE_DECAY


DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
)
DEFAULT_DAILY_PARTITIONS = PROJECT_ROOT / "data/raw/daily_partitioned"
DEFAULT_DAILY_BASIC = PROJECT_ROOT / "data/raw/daily_basic"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/research/left_side_unified_v5/candidate_factors.parquet"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data/research/left_side_unified_v5/candidate_factors.manifest.json"
)

CANDLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "rs_upper_shadow_range_share",
    "rs_upper_shadow_body_ratio",
)
BREADTH_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_breadth_up_ratio_1d",
    "market_breadth_down_ratio_1d",
    "market_breadth_flat_ratio_1d",
    "market_breadth_advance_decline_ratio_1d",
    "market_breadth_advance_decline_spread_1d",
    "market_breadth_return_mean_1d_pct",
    "market_breadth_return_median_1d_pct",
    "market_breadth_return_dispersion_1d_pct",
    "market_breadth_up5_ratio_1d",
    "market_breadth_down5_ratio_1d",
    "market_breadth_up5_down5_spread_1d",
    "market_breadth_limit_up_ratio_proxy",
    "market_breadth_limit_down_ratio_proxy",
    "market_breadth_limit_up_down_spread_proxy",
    "market_breadth_up_ratio_ma5",
    "market_breadth_up_ratio_ma20",
    "market_breadth_up_ratio_change_1d",
    "market_breadth_up_ratio_change_5d",
    "market_breadth_advance_decline_spread_ma5",
    "market_breadth_advance_decline_spread_ma20",
)
ACTIVE_VALUE_FEATURE_COLUMNS: tuple[str, ...] = (
    "stock_active_share_ratio_13d_proxy",
    "stock_active_mv_log",
    "stock_active_mv_return_1d_pct",
    "stock_active_mv_return_5d_pct",
    "market_active_mv_log",
    "market_active_mv_ratio_proxy",
    "market_active_mv_return_1d_pct",
    "market_active_mv_return_5d_pct",
)
CANDIDATE_FACTOR_COLUMNS: tuple[str, ...] = (
    *CANDLE_FEATURE_COLUMNS,
    *BREADTH_FEATURE_COLUMNS,
    *ACTIVE_VALUE_FEATURE_COLUMNS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce").div(
        pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    )


def _event_frame(path: Path) -> pd.DataFrame:
    columns = ["symbol", "date", "open", "high", "low", "close"]
    frame = pd.read_parquet(path, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("left-side event keys are duplicated")
    return frame.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _candlestick_features(events: pd.DataFrame) -> pd.DataFrame:
    open_ = pd.to_numeric(events["open"], errors="coerce")
    high = pd.to_numeric(events["high"], errors="coerce")
    low = pd.to_numeric(events["low"], errors="coerce")
    close = pd.to_numeric(events["close"], errors="coerce")
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    return pd.DataFrame(
        {
            "symbol": events["symbol"].astype(str),
            "date": events["date"],
            "rs_upper_shadow_range_share": _safe_div(upper_shadow, high - low),
            "rs_upper_shadow_body_ratio": _safe_div(upper_shadow, (close - open_).abs()),
        }
    ).replace([np.inf, -np.inf], np.nan)


def _daily_partition_paths(root: Path, start: str, end: str) -> list[Path]:
    start_month = pd.Timestamp(start).strftime("%Y%m")
    end_month = pd.Timestamp(end).strftime("%Y%m")
    paths = [
        path
        for path in sorted(root.glob("year_month=*/data.parquet"))
        if start_month <= path.parent.name.split("=", 1)[-1] <= end_month
    ]
    if not paths:
        raise FileNotFoundError(f"no daily partitions for {start}..{end}: {root}")
    return paths


def _read_market(root: Path, start: str, end: str) -> pd.DataFrame:
    start_text = pd.Timestamp(start).strftime("%Y%m%d")
    end_text = pd.Timestamp(end).strftime("%Y%m%d")
    columns = ("ts_code", "trade_date", "close", "pre_close", "pct_chg", "vol", "volume")
    frames: list[pd.DataFrame] = []
    for path in _daily_partition_paths(root, start, end):
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(
            path, columns=[column for column in columns if column in available]
        )
        dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
        frame = frame[dates.between(start_text, end_text)]
        if not frame.empty:
            frames.append(frame)
    market = pd.concat(frames, ignore_index=True, sort=False)
    market["trade_date"] = (
        market["trade_date"].astype(str).str.replace("-", "", regex=False)
    )
    market = market.sort_values(["ts_code", "trade_date"], kind="stable")
    if market.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("daily market rows are duplicated")
    return market.reset_index(drop=True)


def _market_breadth_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce")
    if "pct_chg" in market:
        returns = pd.to_numeric(market["pct_chg"], errors="coerce")
    else:
        previous = pd.to_numeric(market["pre_close"], errors="coerce")
        returns = close.div(previous.replace(0.0, np.nan)).sub(1.0).mul(100.0)
    volume_column = "volume" if "volume" in market else "vol"
    traded = pd.to_numeric(market[volume_column], errors="coerce").gt(0.0)
    eligible = returns.notna() & close.notna() & traded
    work = pd.DataFrame(
        {
            "trade_date": market["trade_date"],
            "ret": returns.where(eligible),
            "eligible": eligible,
            "up": eligible & returns.gt(0.0),
            "down": eligible & returns.lt(0.0),
            "flat": eligible & returns.abs().le(1e-12),
            "up5": eligible & returns.ge(5.0),
            "down5": eligible & returns.le(-5.0),
            "limit_up": eligible & returns.ge(9.5),
            "limit_down": eligible & returns.le(-9.5),
        }
    )
    out = (
        work.groupby("trade_date", sort=True)
        .agg(
            eligible=("eligible", "sum"),
            up=("up", "sum"),
            down=("down", "sum"),
            flat=("flat", "sum"),
            up5=("up5", "sum"),
            down5=("down5", "sum"),
            limit_up=("limit_up", "sum"),
            limit_down=("limit_down", "sum"),
            return_mean=("ret", "mean"),
            return_median=("ret", "median"),
            return_dispersion=("ret", "std"),
        )
        .reset_index()
    )
    denominator = out["eligible"]
    out["market_breadth_up_ratio_1d"] = _safe_div(out["up"], denominator)
    out["market_breadth_down_ratio_1d"] = _safe_div(out["down"], denominator)
    out["market_breadth_flat_ratio_1d"] = _safe_div(out["flat"], denominator)
    out["market_breadth_advance_decline_ratio_1d"] = _safe_div(out["up"], out["down"])
    out["market_breadth_advance_decline_spread_1d"] = _safe_div(
        out["up"] - out["down"], denominator
    )
    out["market_breadth_return_mean_1d_pct"] = out["return_mean"]
    out["market_breadth_return_median_1d_pct"] = out["return_median"]
    out["market_breadth_return_dispersion_1d_pct"] = out["return_dispersion"]
    out["market_breadth_up5_ratio_1d"] = _safe_div(out["up5"], denominator)
    out["market_breadth_down5_ratio_1d"] = _safe_div(out["down5"], denominator)
    out["market_breadth_up5_down5_spread_1d"] = _safe_div(
        out["up5"] - out["down5"], denominator
    )
    out["market_breadth_limit_up_ratio_proxy"] = _safe_div(out["limit_up"], denominator)
    out["market_breadth_limit_down_ratio_proxy"] = _safe_div(out["limit_down"], denominator)
    out["market_breadth_limit_up_down_spread_proxy"] = _safe_div(
        out["limit_up"] - out["limit_down"], denominator
    )
    up_ratio = out["market_breadth_up_ratio_1d"]
    spread = out["market_breadth_advance_decline_spread_1d"]
    out["market_breadth_up_ratio_ma5"] = up_ratio.rolling(5, min_periods=5).mean()
    out["market_breadth_up_ratio_ma20"] = up_ratio.rolling(20, min_periods=20).mean()
    out["market_breadth_up_ratio_change_1d"] = up_ratio.diff(1)
    out["market_breadth_up_ratio_change_5d"] = up_ratio.diff(5)
    out["market_breadth_advance_decline_spread_ma5"] = spread.rolling(5, min_periods=5).mean()
    out["market_breadth_advance_decline_spread_ma20"] = spread.rolling(20, min_periods=20).mean()
    out["date"] = pd.to_datetime(out["trade_date"], format="%Y%m%d")
    return out[["date", *BREADTH_FEATURE_COLUMNS]].replace([np.inf, -np.inf], np.nan)


def _daily_basic_paths(root: Path, start: str, end: str) -> list[Path]:
    start_text = pd.Timestamp(start).strftime("%Y%m%d")
    end_text = pd.Timestamp(end).strftime("%Y%m%d")
    paths = [
        path
        for path in sorted(root.glob("*.parquet"))
        if start_text <= path.stem <= end_text
    ]
    if not paths:
        raise FileNotFoundError(f"no daily_basic files for {start}..{end}: {root}")
    return paths


def _read_daily_basic(root: Path, start: str, end: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(path, columns=["ts_code", "trade_date", "turnover_rate", "circ_mv"])
        for path in _daily_basic_paths(root, start, end)
    ]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["trade_date"] = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    frame = frame.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("daily_basic rows are duplicated")
    return frame


def _active_value_features(basic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    turnover = pd.to_numeric(basic["turnover_rate"], errors="coerce").to_numpy(float) / 100.0
    circ_mv = pd.to_numeric(basic["circ_mv"], errors="coerce").to_numpy(float)
    active_share = np.full(len(basic), np.nan, dtype=float)
    for positions in basic.groupby("ts_code", sort=False).indices.values():
        state = np.nan
        for position in positions:
            raw = turnover[position]
            if not np.isfinite(raw):
                continue
            current = float(np.clip(raw, 0.0, 1.0))
            state = current if not np.isfinite(state) else current + ACTIVE_SHARE_DECAY * (1.0 - current) * state
            active_share[position] = float(np.clip(state, 0.0, 1.0))
    active_mv = circ_mv * 10_000.0 * active_share
    stock = basic[["ts_code", "trade_date"]].copy()
    stock["stock_active_share_ratio_13d_proxy"] = active_share
    stock["stock_active_mv_proxy_cny"] = active_mv
    stock["stock_active_mv_log"] = np.log(pd.Series(active_mv).where(active_mv > 0.0))
    grouped = stock.groupby("ts_code", sort=False)["stock_active_mv_proxy_cny"]
    stock["stock_active_mv_return_1d_pct"] = grouped.pct_change(fill_method=None).mul(100.0)
    stock["stock_active_mv_return_5d_pct"] = grouped.pct_change(5, fill_method=None).mul(100.0)

    basic_with_active = pd.DataFrame(
        {
            "trade_date": basic["trade_date"],
            "active_mv": active_mv,
            "float_mv": circ_mv * 10_000.0,
        }
    )
    market = (
        basic_with_active.groupby("trade_date", sort=True)
        .agg(active_mv=("active_mv", "sum"), float_mv=("float_mv", "sum"))
        .reset_index()
    )
    market["market_active_mv_log"] = np.log(market["active_mv"].where(market["active_mv"] > 0.0))
    market["market_active_mv_ratio_proxy"] = _safe_div(market["active_mv"], market["float_mv"])
    market["market_active_mv_return_1d_pct"] = market["active_mv"].pct_change(fill_method=None).mul(100.0)
    market["market_active_mv_return_5d_pct"] = market["active_mv"].pct_change(5, fill_method=None).mul(100.0)
    market["date"] = pd.to_datetime(market["trade_date"], format="%Y%m%d")
    stock["symbol"] = stock.pop("ts_code").astype(str)
    stock["date"] = pd.to_datetime(stock.pop("trade_date"), format="%Y%m%d")
    return (
        stock[["symbol", "date", *ACTIVE_VALUE_FEATURE_COLUMNS[:4]]].replace([np.inf, -np.inf], np.nan),
        market[["date", *ACTIVE_VALUE_FEATURE_COLUMNS[4:]]].replace([np.inf, -np.inf], np.nan),
    )


def _coverage(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    return {
        column: round(float(frame[column].notna().mean()), 6)
        for column in columns
    }


def materialize(args: argparse.Namespace) -> dict[str, object]:
    events = _event_frame(args.events)
    start = args.start_date or events["date"].min().date().isoformat()
    end = args.end_date or events["date"].max().date().isoformat()
    factors = _candlestick_features(events)

    market = _read_market(args.daily_partitions, start, end)
    breadth = _market_breadth_features(market)
    factors = factors.merge(breadth, on="date", how="left", validate="many_to_one")
    del market, breadth
    gc.collect()

    warmup_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).date().isoformat()
    basic = _read_daily_basic(args.daily_basic, warmup_start, end)
    stock_active, market_active = _active_value_features(basic)
    del basic
    gc.collect()
    factors = factors.merge(
        stock_active, on=["symbol", "date"], how="left", validate="one_to_one"
    ).merge(market_active, on="date", how="left", validate="many_to_one")
    factors = factors[["symbol", "date", *CANDIDATE_FACTOR_COLUMNS]]
    if len(factors) != len(events):
        raise RuntimeError("candidate factor join changed event cardinality")
    if factors.duplicated(["symbol", "date"]).any():
        raise RuntimeError("candidate factor output contains duplicate keys")

    atomic_write_parquet(factors, args.output, index=False)
    payload: dict[str, object] = {
        "status": "success",
        "schema_version": "left-side-v5-candidate-factors-v1",
        "start_date": start,
        "end_date": end,
        "rows": len(factors),
        "factor_count": len(CANDIDATE_FACTOR_COLUMNS),
        "factor_columns": list(CANDIDATE_FACTOR_COLUMNS),
        "coverage": _coverage(factors, CANDIDATE_FACTOR_COLUMNS),
        "events_sha256": _sha256(args.events),
        "output_sha256": _sha256(args.output),
    }
    atomic_write_json(payload, args.manifest, indent=2)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--daily-partitions", type=Path, default=DEFAULT_DAILY_PARTITIONS)
    parser.add_argument("--daily-basic", type=Path, default=DEFAULT_DAILY_BASIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser


def main() -> None:
    payload = materialize(_parser().parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
