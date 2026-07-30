#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a leakage-safe selector model history from rule caches and market bars.

This builder is intentionally lighter than the page-snapshot replay.  It keeps
the information needed by the buy/hold return models: candidate identity,
matched strategy groups, and forward labels calculated from Tushare's
continuous pct_chg/pre_close price relationship.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.strategies.custom.z_skill_patterns import EXTENDED_STRATEGIES
from quant.webapp.services import FAMILY_SIGNAL_COLUMNS, STRATEGY_MEMBER_TO_GROUP


DEFAULT_FAMILY = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
DEFAULT_EXTENDED = PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"
DEFAULT_DAILY = PROJECT_ROOT / "data/raw/daily_partitioned/year_month=*/data.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/research/selector_model_history_2020.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selector model history from cached strategy candidates.")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-06-12")
    parser.add_argument("--family-cache", type=Path, default=DEFAULT_FAMILY)
    parser.add_argument("--extended-cache", type=Path, default=DEFAULT_EXTENDED)
    parser.add_argument("--daily-glob", default=str(DEFAULT_DAILY))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _signal_keys(frame: pd.DataFrame, definitions: list[tuple[str, str]]) -> pd.DataFrame:
    hits: list[pd.DataFrame] = []
    for column, group in definitions:
        if column not in frame.columns:
            continue
        mask = frame[column].fillna(False).astype(bool)
        if not mask.any():
            continue
        part = frame.loc[mask, ["symbol", "date"]].copy()
        part["strategy_group"] = group
        hits.append(part)
    return pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(columns=["symbol", "date", "strategy_group"])


def load_candidate_keys(args: argparse.Namespace) -> pd.DataFrame:
    family_defs = [(definition[0], definition[1]) for definition in FAMILY_SIGNAL_COLUMNS.values()]
    parquet = __import__("pyarrow.parquet", fromlist=["read_schema"])
    family_schema = set(parquet.read_schema(args.family_cache).names)
    family_columns = ["symbol", "date", *sorted({column for column, _ in family_defs if column in family_schema})]
    family = pd.read_parquet(args.family_cache, columns=family_columns)
    family["date"] = pd.to_datetime(family["date"], errors="coerce")
    family_hits = _signal_keys(family, family_defs)

    extended_defs = [
        (str(item["key"]), STRATEGY_MEMBER_TO_GROUP.get(str(item["key"]), str(item["key"])))
        for item in EXTENDED_STRATEGIES
    ]
    extended_schema = set(parquet.read_schema(args.extended_cache).names)
    extended_columns = ["symbol", "date", *[column for column, _ in extended_defs if column in extended_schema]]
    extended = pd.read_parquet(args.extended_cache, columns=extended_columns)
    extended["date"] = pd.to_datetime(extended["date"], errors="coerce")
    extended_hits = _signal_keys(extended, extended_defs)

    hits = pd.concat([family_hits, extended_hits], ignore_index=True)
    hits = hits[
        (hits["date"] >= pd.Timestamp(args.start_date)) & (hits["date"] <= pd.Timestamp(args.end_date))
    ].drop_duplicates(["symbol", "date", "strategy_group"])
    grouped = hits.groupby(["symbol", "date"], as_index=False).agg(
        matched_groups=("strategy_group", lambda values: sorted(set(values))),
        matched_count=("strategy_group", "nunique"),
    )
    grouped["best_profit_factor"] = np.nan
    grouped["best_avg_return_pct"] = np.nan
    return grouped


def add_group_realized_history(history: pd.DataFrame, market_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Add leakage-safe trailing realized returns for every matched strategy group."""
    out = history.copy()
    out["_row_id"] = np.arange(len(out))
    exploded = out[
        ["_row_id", "date", "matched_groups", "future_return_t5_pct", "future_max_high_t5_pct"]
    ].explode("matched_groups")
    exploded = exploded.rename(columns={"matched_groups": "_strategy_group"})
    exploded = exploded.dropna(subset=["_strategy_group"])
    session_dates = pd.DatetimeIndex(pd.to_datetime(market_dates)).normalize().drop_duplicates().sort_values()
    session_index = {value: index for index, value in enumerate(session_dates)}
    signal_index = exploded["date"].dt.normalize().map(session_index)
    available_index = signal_index + 5
    exploded["_available_date"] = available_index.map(
        lambda index: session_dates[int(index)] if pd.notna(index) and int(index) < len(session_dates) else pd.NaT
    )
    realized = exploded.dropna(subset=["_available_date"]).groupby(
        ["_strategy_group", "_available_date"], as_index=False
    ).agg(
        _group_hold_return=("future_return_t5_pct", "mean"),
        _group_buy_return=("future_max_high_t5_pct", "mean"),
    )
    feature_columns = [
        f"selector_group_{mode}_realized_{window}d"
        for mode in ("hold", "buy")
        for window in (20, 60)
    ]
    joined_parts = []
    for group, current in exploded.groupby("_strategy_group", sort=False):
        snapshots = realized[realized["_strategy_group"] == group].sort_values("_available_date").copy()
        for target_name, prefix in (
            ("_group_hold_return", "selector_group_hold_realized"),
            ("_group_buy_return", "selector_group_buy_realized"),
        ):
            for window in (20, 60):
                column = f"{prefix}_{window}d"
                snapshots[column] = snapshots[target_name].rolling(window, min_periods=5).mean()
        left = current[["_row_id", "date"]].sort_values("date")
        right = snapshots[["_available_date", *feature_columns]].sort_values("_available_date")
        joined_parts.append(
            pd.merge_asof(
                left,
                right,
                left_on="date",
                right_on="_available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    if joined_parts:
        group_features = pd.concat(joined_parts, ignore_index=True)
        group_features = group_features.groupby("_row_id", as_index=False)[sorted(set(feature_columns))].mean()
        out = out.merge(group_features, on="_row_id", how="left")
    return out.drop(columns="_row_id")


def build_forward_labels(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    # Keep enough pre-window history for the longest 120-session factor while
    # still emitting only requested candidate dates.
    start = (pd.Timestamp(args.start_date) - pd.Timedelta(days=250)).date().isoformat()
    end = (pd.Timestamp(args.end_date) + pd.Timedelta(days=20)).date().isoformat()
    market = (
        pl.scan_parquet(args.daily_glob, hive_partitioning=True)
        .select(["symbol", "date", "open", "high", "low", "close", "pre_close", "pct_chg", "volume", "turnover"])
        .filter((pl.col("date") >= pl.lit(start).str.to_date()) & (pl.col("date") <= pl.lit(end).str.to_date()))
        .sort(["symbol", "date"])
    )
    market_daily = market.group_by("date").agg(
        pl.col("pct_chg").mean().alias("selector_market_mean_1d"),
        pl.col("pct_chg").median().alias("selector_market_median_1d"),
        pl.col("pct_chg").std().alias("selector_market_dispersion_1d"),
        (pl.col("pct_chg") > 0).mean().alias("selector_market_up_ratio_1d"),
        (pl.col("pct_chg") >= 5).mean().alias("selector_market_up5_ratio_1d"),
        (pl.col("pct_chg") <= -5).mean().alias("selector_market_down5_ratio_1d"),
    ).sort("date")
    market_daily = market_daily.with_columns(
        pl.col("selector_market_mean_1d").rolling_mean(5).alias("selector_market_mean_5d"),
        pl.col("selector_market_mean_1d").rolling_mean(20).alias("selector_market_mean_20d"),
        pl.col("selector_market_up_ratio_1d").rolling_mean(5).alias("selector_market_up_ratio_5d"),
        pl.col("selector_market_up_ratio_1d").rolling_mean(20).alias("selector_market_up_ratio_20d"),
    )
    market = market.join(market_daily, on="date", how="left").sort(["symbol", "date"])
    daily_return = 1.0 + pl.col("pct_chg").cast(pl.Float64) / 100.0
    log_return = daily_return.log()
    pct_return = pl.col("pct_chg").cast(pl.Float64)
    amplitude = (pl.col("high") - pl.col("low")) / pl.col("pre_close") * 100.0
    close_position = (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
    gap_return = (pl.col("open") / pl.col("pre_close") - 1.0) * 100.0
    close_factors = [daily_return.shift(-day).over("symbol") for day in range(1, 6)]
    future_close = pl.fold(acc=pl.lit(1.0), function=lambda acc, item: acc * item, exprs=close_factors)
    high_ratio = pl.col("high").cast(pl.Float64) / pl.col("pre_close").cast(pl.Float64)
    future_highs = []
    for day in range(1, 6):
        prior_returns = close_factors[: day - 1]
        prior_growth = (
            pl.fold(acc=pl.lit(1.0), function=lambda acc, item: acc * item, exprs=prior_returns)
            if prior_returns
            else pl.lit(1.0)
        )
        future_highs.append(prior_growth * high_ratio.shift(-day).over("symbol"))
    labels = market.select(
        "symbol",
        pl.col("date").cast(pl.Date),
        ((future_close - 1.0) * 100.0).alias("future_return_t5_pct"),
        pl.when(future_close.is_not_null())
        .then((pl.max_horizontal(future_highs) - 1.0) * 100.0)
        .otherwise(None)
        .alias("future_max_high_t5_pct"),
        pl.col("pct_chg").cast(pl.Float64).alias("selector_return_1d"),
        *[
            ((log_return.rolling_sum(window).over("symbol").exp() - 1.0) * 100.0).alias(
                f"selector_return_{window}d"
            )
            for window in (2, 3, 5, 10, 15, 20, 30, 60, 120)
        ],
        *[
            pl.col("pct_chg").cast(pl.Float64).rolling_std(window).over("symbol").alias(
                f"selector_volatility_{window}d"
            )
            for window in (3, 5, 10, 20, 30, 60, 120)
        ],
        *[
            (pl.col("pct_chg") > 0).cast(pl.Float64).rolling_mean(window).over("symbol").alias(
                f"selector_positive_ratio_{window}d"
            )
            for window in (3, 5, 10, 20, 60)
        ],
        *[
            pct_return.rolling_skew(window).over("symbol").alias(f"selector_return_skew_{window}d")
            for window in (10, 20, 60)
        ],
        *[
            pct_return.rolling_max(window).over("symbol").alias(f"selector_best_day_{window}d")
            for window in (5, 20, 60)
        ],
        *[
            pct_return.rolling_min(window).over("symbol").alias(f"selector_worst_day_{window}d")
            for window in (5, 20, 60)
        ],
        *[
            pl.when(pct_return < 0).then(pct_return).otherwise(0.0).rolling_std(window).over("symbol").alias(
                f"selector_downside_volatility_{window}d"
            )
            for window in (10, 20, 60)
        ],
        amplitude.alias("selector_amplitude_1d"),
        *[
            amplitude.rolling_mean(window).over("symbol").alias(f"selector_amplitude_mean_{window}d")
            for window in (5, 20)
        ],
        *[
            amplitude.rolling_std(window).over("symbol").alias(f"selector_amplitude_std_{window}d")
            for window in (5, 20)
        ],
        close_position.alias("selector_close_pos"),
        *[
            close_position.rolling_mean(window).over("symbol").alias(f"selector_close_pos_mean_{window}d")
            for window in (5, 20)
        ],
        gap_return.alias("selector_gap_1d"),
        *[
            gap_return.rolling_mean(window).over("symbol").alias(f"selector_gap_mean_{window}d")
            for window in (5, 20)
        ],
        ((pl.col("high") / pl.col("pre_close") - 1.0) * 100.0).alias("selector_high_1d"),
        ((pl.col("low") / pl.col("pre_close") - 1.0) * 100.0).alias("selector_low_1d"),
        *[
            (pl.col("volume") / pl.col("volume").rolling_mean(window).over("symbol")).alias(
                f"selector_volume_relative_{window}d"
            )
            for window in (5, 10, 20, 60)
        ],
        *[
            (pl.col("turnover") / pl.col("turnover").rolling_mean(window).over("symbol")).alias(
                f"selector_turnover_relative_{window}d"
            )
            for window in (5, 20)
        ],
        (pl.col("turnover") / pl.col("turnover").shift(1).over("symbol")).alias(
            "selector_turnover_change_1d"
        ),
        (
            (pl.col("turnover") - pl.col("turnover").rolling_mean(20).over("symbol"))
            / pl.col("turnover").rolling_std(20).over("symbol")
        ).alias("selector_turnover_zscore_20d"),
        pl.rolling_corr(pct_return, pl.col("volume").cast(pl.Float64), window_size=20).over("symbol").alias(
            "selector_return_volume_corr_20d"
        ),
        (pl.col("pct_chg") - pl.col("selector_market_mean_1d")).alias("selector_excess_return_1d"),
        *[pl.col(column) for column in market_daily.collect_schema().names() if column != "date"],
    )
    keys = pl.from_pandas(candidates[["symbol", "date"]]).lazy().with_columns(pl.col("date").cast(pl.Date))
    selected = keys.join(labels, on=["symbol", "date"], how="left").collect().to_pandas()
    selected["date"] = pd.to_datetime(selected["date"])
    history = candidates.merge(selected, on=["symbol", "date"], how="left")
    trading_dates = (
        pl.scan_parquet(args.daily_glob, hive_partitioning=True)
        .select(pl.col("date").unique())
        .sort("date")
        .collect()["date"]
        .to_list()
    )
    return add_group_realized_history(history, trading_dates)


def main() -> None:
    args = parse_args()
    candidates = load_candidate_keys(args)
    history = build_forward_labels(candidates, args)
    history = history.sort_values(["date", "symbol"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.parquet")
    history.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    summary: dict[str, Any] = {
        "status": "success",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output": str(args.output),
        "rows": int(len(history)),
        "dates": int(history["date"].nunique()),
        "symbols": int(history["symbol"].nunique()),
        "date_min": history["date"].min().date().isoformat(),
        "date_max": history["date"].max().date().isoformat(),
        "buy_label_coverage": float(history["future_max_high_t5_pct"].notna().mean()),
        "hold_label_coverage": float(history["future_return_t5_pct"].notna().mean()),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
