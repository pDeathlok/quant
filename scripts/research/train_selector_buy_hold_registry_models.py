#!/usr/bin/env python
"""Train canonical-registry selector buy/hold return models.

The sample universe is the union of the released right-side and left-side
ranking candidate datasets.  Labels use next-close entry, exclude locked
limit-up entries, and retain the two display semantics:

* buy: maximum high return over the following five sessions;
* hold: terminal close return after five sessions.

The complete governed registry is the discovery input contract.  Historical
materializers must provide every factor before fitting; constant factors are
still passed to discovery and are naturally removed when the final non-zero
tree-gain input contract is frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    LEGACY_TO_CANONICAL_FACTOR_NAMES,
    assert_no_forbidden_factor_names,
)
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.selector_buy_hold_factor_contract import (
    SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
    SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
    SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION,
    SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS,
    SELECTOR_BUY_HOLD_RELEASE_ID,
    selector_buy_hold_factor_contract_payload,
    selector_buy_hold_model_input_sha256,
    validate_selector_buy_hold_artifact,
)
from quant.features.selector_buy_hold_materialization import (
    audit_selector_buy_hold_training_materialization,
    require_complete_selector_buy_hold_training_materialization,
    selector_buy_hold_training_materialization_plan,
    training_materialization_audit_payload,
)
from quant.features.long_weekly_factors import (
    MONTHLY_VALUATION_HISTORY_COLUMNS,
    add_monthly_valuation_history,
)
from quant.features.market_sentiment import (
    build_limit_proxy_features,
    read_top_list_features,
)
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, YearFold
from quant.research.short_side_groups import ALL_SHORT_GROUPS, GROUP_MEMBERS


RESEARCH_ROOT = PROJECT_ROOT / "data/research/selector_buy_hold_registry_v2"
REPORT_ROOT = PROJECT_ROOT / "reports/research/selector_buy_hold_registry_v3"
CANDIDATE_MODEL_ROOT = PROJECT_ROOT / "models/candidates/selector_buy_hold_registry_v3"
PRODUCTION_MODEL_ROOT = PROJECT_ROOT / "models/production/selector_buy_hold_registry_v3"
DATASET_PATH = RESEARCH_ROOT / "selector_buy_hold_registry_dataset.parquet"
DATASET_MANIFEST_PATH = RESEARCH_ROOT / "dataset_manifest.json"
REPORT_PATH = REPORT_ROOT / "training_report.json"
FEATURE_COVERAGE_PATH = REPORT_ROOT / "feature_coverage.csv"
MATERIALIZATION_AUDIT_PATH = REPORT_ROOT / "materialization_audit.json"

RIGHT_EVENTS = (
    PROJECT_ROOT
    / "data/research/right_side_unified_canonical_v5_rule113/unified_right_side_dataset.parquet"
)
RIGHT_LABELS = (
    PROJECT_ROOT
    / "data/research/right_side_unified_canonical_v5_rule113/unified_right_side_labels.parquet"
)
LEFT_EVENTS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
)
LEFT_LABELS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/labels.parquet"
)

EXACT_FACTOR_SOURCES: tuple[Path, ...] = (
    PROJECT_ROOT / "data/research/selector_model_history_2020.parquet",
    PROJECT_ROOT / "data/research/right_side_registry_607/daily_basic_features.parquet",
    PROJECT_ROOT / "data/research/right_side_registry_607/daily_extras.parquet",
)
WEEKLY_FACTOR_SOURCES: tuple[Path, ...] = (
    PROJECT_ROOT / "data/research/right_side_registry_607/analyst_weekly.parquet",
    PROJECT_ROOT / "data/features/long_entry/weekly_external_v1.parquet",
    PROJECT_ROOT / "data/features/long_entry/weekly_quality_factors_v1.parquet",
    PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet",
)
MARKET_DAILY_DIR = PROJECT_ROOT / "data/raw/daily"
TOP_LIST_DIR = PROJECT_ROOT / "data/raw/top_list"
VALUATION_HISTORY_SOURCE = (
    PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet"
)

MARKET_SENTIMENT_FACTOR_COLUMNS: tuple[str, ...] = (
    "limit_down_ratio_proxy",
    "limit_up_count_proxy",
    "limit_up_ratio_proxy",
    "market_median_ret_1d",
    "market_panic_5d",
    "market_sentiment_5d",
    "market_up_ratio",
    "strong_up_ratio_proxy",
)
TOP_LIST_FACTOR_COLUMNS: tuple[str, ...] = (
    "top_list_count",
    "top_net_amount_ratio",
    "top_net_rate",
)

KEY_COLUMNS: tuple[str, ...] = ("symbol", "date")
TARGET_COLUMNS: tuple[str, ...] = (
    "future_max_high_t5_pct",
    "future_return_t5_pct",
)
METADATA_COLUMNS: tuple[str, ...] = (
    "label_end_date",
    "right_candidate",
    "left_candidate",
)
RIGHT_RAW_SIGNALS: tuple[str, ...] = tuple(
    dict.fromkeys(member for group in ALL_SHORT_GROUPS for member in GROUP_MEMBERS[group])
)
LEFT_RAW_SIGNALS: tuple[str, ...] = ("B1", "SB1", "SUPER_B1", "LOW_PULLBACK")


@dataclass(frozen=True)
class RankingMetrics:
    rows: int
    days: int
    global_spearman: float
    daily_spearman: float
    positive_daily_spearman_ratio: float
    decile_spread: float
    decile_trend: float
    top20_avg_return: float
    top20_hit_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train selector buy/hold models on the canonical factor registry."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--right-daily-cap", type=int, default=320)
    parser.add_argument("--left-daily-cap", type=int, default=192)
    parser.add_argument("--minimum-factor-observations", type=int, default=2)
    parser.add_argument("--folds", default="B,C")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def _parquet_alias_migrations(path: Path) -> tuple[tuple[str, str], ...]:
    import pyarrow.parquet as pq

    available = tuple(str(value) for value in pq.ParquetFile(path).schema.names)
    migrations = tuple(
        (alias, canonical)
        for alias, canonical in LEGACY_TO_CANONICAL_FACTOR_NAMES.items()
        if alias in available
    )
    if not migrations:
        return ()
    pl = _polars()
    for alias, canonical in migrations:
        if canonical not in available:
            continue
        left = pl.col(alias)
        right = pl.col(canonical)
        schema = pl.scan_parquet(path).collect_schema()
        if schema[alias] in (pl.Float32, pl.Float64):
            left = left.fill_nan(None)
        if schema[canonical] in (pl.Float32, pl.Float64):
            right = right.fill_nan(None)
        mismatches = (
            pl.scan_parquet(path)
            .select((~left.eq_missing(right)).sum().alias("mismatches"))
            .collect(engine="streaming")["mismatches"][0]
        )
        if int(mismatches or 0) > 0:
            raise RuntimeError(
                f"legacy factor boundary mismatch {alias}->{canonical}: "
                f"{path}: rows={int(mismatches)}"
            )
    return migrations


def _parquet_columns(path: Path) -> tuple[str, ...]:
    import pyarrow.parquet as pq

    available = tuple(str(value) for value in pq.ParquetFile(path).schema.names)
    migrations = dict(_parquet_alias_migrations(path))
    canonical: list[str] = []
    seen: set[str] = set()
    for column in available:
        name = migrations.get(column, column)
        if name not in seen:
            canonical.append(name)
            seen.add(name)
    forbidden = sorted(set(canonical) & set(FORBIDDEN_COMPATIBILITY_ALIASES))
    if forbidden:
        raise RuntimeError(f"canonical parquet boundary retained aliases: {path}: {forbidden}")
    return tuple(canonical)


def _scan_parquet_canonical(path: Path) -> Any:
    """Read one legacy cache once, validate aliases, and expose canonical columns."""

    pl = _polars()
    frame = pl.scan_parquet(path)
    schema_names = set(frame.collect_schema().names())
    for alias, canonical in _parquet_alias_migrations(path):
        if canonical in schema_names:
            frame = frame.drop(alias)
        else:
            frame = frame.rename({alias: canonical})
        schema_names.discard(alias)
        schema_names.add(canonical)
    return frame


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "columns": len(_parquet_columns(path)),
        "legacy_aliases_migrated": [
            alias for alias, _ in _parquet_alias_migrations(path)
        ],
    }


def _polars() -> Any:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - local research dependency
        raise RuntimeError(
            "dataset construction requires polars; install the project ml extra"
        ) from exc
    return pl


def _join_materialized_exact_source(
    frame: Any,
    source: Any,
    *,
    keys: Sequence[str],
    columns: Sequence[str],
    source_name: str,
    ownership: dict[str, str],
    fill_null_with_zero: Sequence[str] = (),
) -> Any:
    """Join one canonical training materializer and fail on value drift."""

    pl = _polars()
    current = set(frame.collect_schema().names())
    existing = [column for column in columns if column in current]
    frame = frame.join(
        source.select([*keys, *columns]),
        on=list(keys),
        how="left",
        suffix="__materialized",
        validate="m:1",
    )
    if existing:
        mismatch_counts = frame.select(
            [
                (
                    pl.col(column).is_not_null()
                    & pl.col(f"{column}__materialized").is_not_null()
                    & ~pl.col(column)
                    .cast(pl.Float32, strict=False)
                    .eq_missing(
                        pl.col(f"{column}__materialized").cast(
                            pl.Float32, strict=False
                        )
                    )
                )
                .sum()
                .alias(column)
                for column in existing
            ]
        ).collect(engine="streaming")
        mismatches = [
            column
            for column in existing
            if int(mismatch_counts[column][0] or 0) > 0
        ]
        if mismatches:
            raise RuntimeError(
                f"training materializer disagrees with canonical values: "
                f"{source_name}: {mismatches}"
            )
        frame = frame.with_columns(
            [
                pl.coalesce(
                    pl.col(column), pl.col(f"{column}__materialized")
                ).alias(column)
                for column in existing
            ]
        ).drop([f"{column}__materialized" for column in existing])
    zero_columns = [column for column in fill_null_with_zero if column in columns]
    if zero_columns:
        frame = frame.with_columns(
            [
                pl.col(column).fill_null(0.0).cast(pl.Float32).alias(column)
                for column in zero_columns
            ]
        )
    for column in columns:
        previous = ownership.get(column)
        ownership[column] = f"{previous} | fill:{source_name}" if previous else source_name
    return frame


def _candidate_dates(frame: Any) -> tuple[pd.Timestamp, pd.Timestamp, tuple[str, ...]]:
    pl = _polars()
    dates = (
        frame.select(pl.col("date").cast(pl.Date).unique().sort())
        .collect(engine="streaming")["date"]
        .to_list()
    )
    if not dates:
        raise RuntimeError("selector training candidates contain no dates")
    date_text = tuple(pd.Timestamp(value).strftime("%Y%m%d") for value in dates)
    return pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]), date_text


def _top_list_partition_dates(directory: Path) -> set[str]:
    return {
        path.stem.rsplit("_", 1)[-1]
        for path in directory.glob("*top_list_*.parquet")
    }


def _with_full_history_derived_factors(
    frame: Any,
    ownership: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Materialize registry factors not stored in the side sample caches."""

    pl = _polars()
    date_min, date_max, candidate_dates = _candidate_dates(frame)
    market = build_limit_proxy_features(MARKET_DAILY_DIR, start=date_min)
    if market.empty:
        raise RuntimeError("market sentiment history materializer returned no rows")
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    market = market.loc[
        market["date"].between(date_min, date_max),
        ["date", *MARKET_SENTIMENT_FACTOR_COLUMNS],
    ].drop_duplicates("date", keep="last")
    missing_market_dates = sorted(set(candidate_dates) - set(
        market["date"].dt.strftime("%Y%m%d")
    ))
    market_source = (
        pl.from_pandas(market)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .lazy()
    )
    frame = _join_materialized_exact_source(
        frame,
        market_source,
        keys=("date",),
        columns=MARKET_SENTIMENT_FACTOR_COLUMNS,
        source_name="quant.features.market_sentiment.build_limit_proxy_features",
        ownership=ownership,
    )

    top_partition_dates = _top_list_partition_dates(TOP_LIST_DIR)
    missing_top_dates = sorted(set(candidate_dates) - top_partition_dates)
    top = read_top_list_features(TOP_LIST_DIR, start=date_min)
    if top.empty:
        top = pd.DataFrame(columns=["ts_code", "date", *TOP_LIST_FACTOR_COLUMNS])
    top["date"] = pd.to_datetime(top["date"], errors="coerce")
    top = top.loc[
        top["date"].between(date_min, date_max),
        ["ts_code", "date", *TOP_LIST_FACTOR_COLUMNS],
    ].copy()
    top["symbol"] = top["ts_code"].astype(str)
    top_source = (
        pl.from_pandas(top[["symbol", "date", *TOP_LIST_FACTOR_COLUMNS]])
        .with_columns(
            pl.col("symbol").cast(pl.String),
            pl.col("date").cast(pl.Datetime("ns")),
        )
        .unique(KEY_COLUMNS, keep="last")
        .lazy()
    )
    frame = _join_materialized_exact_source(
        frame,
        top_source,
        keys=KEY_COLUMNS,
        columns=TOP_LIST_FACTOR_COLUMNS,
        source_name="quant.features.market_sentiment.read_top_list_features",
        ownership=ownership,
    )
    covered_candidate_dates = sorted(set(candidate_dates) & top_partition_dates)
    available_top_dates = pd.DataFrame(
        {
            "date": pd.to_datetime(
                covered_candidate_dates,
                format="%Y%m%d",
            ),
            "_top_list_partition_available": [True] * len(covered_candidate_dates),
        }
    )
    top_availability = (
        pl.from_pandas(available_top_dates)
        .with_columns(pl.col("date").cast(pl.Datetime("ns")))
        .lazy()
    )
    frame = frame.join(top_availability, on="date", how="left", validate="m:1")
    frame = frame.with_columns(
        [
            pl.when(pl.col("_top_list_partition_available").fill_null(False))
            .then(pl.col(column).fill_null(0.0))
            .otherwise(pl.col(column))
            .cast(pl.Float32)
            .alias(column)
            for column in TOP_LIST_FACTOR_COLUMNS
        ]
    ).drop("_top_list_partition_available")

    valuation_columns = ["date", "ts_code", "industry", "roe", "pe_ttm", "pb"]
    valuation = pd.read_parquet(
        VALUATION_HISTORY_SOURCE,
        columns=valuation_columns,
    )
    valuation["date"] = pd.to_datetime(valuation["date"], errors="raise")
    history_start = date_min - pd.DateOffset(months=84)
    valuation = valuation[valuation["date"].between(history_start, date_max)].copy()
    valuation["month"] = valuation["date"].dt.to_period("M")
    valuation = (
        valuation.sort_values(["ts_code", "date"])
        .drop_duplicates(["ts_code", "month"], keep="last")
        .drop(columns="month")
    )
    valuation = add_monthly_valuation_history(valuation)
    valuation_source = (
        pl.from_pandas(
            valuation[["ts_code", "date", *MONTHLY_VALUATION_HISTORY_COLUMNS]]
        )
        .rename({"ts_code": "symbol"})
        .with_columns(
            pl.col("symbol").cast(pl.String),
            pl.col("date").cast(pl.Datetime("ns")),
        )
        .lazy()
        .sort(["symbol", "date"])
    )
    current = set(frame.collect_schema().names())
    existing = [
        column for column in MONTHLY_VALUATION_HISTORY_COLUMNS if column in current
    ]
    frame = frame.sort(["symbol", "date"]).join_asof(
        valuation_source,
        on="date",
        by="symbol",
        strategy="backward",
        tolerance="45d",
        suffix="__materialized",
        check_sortedness=False,
    )
    if existing:
        mismatch_counts = frame.select(
            [
                (
                    pl.col(column).is_not_null()
                    & pl.col(f"{column}__materialized").is_not_null()
                    & ~pl.col(column)
                    .cast(pl.Float32, strict=False)
                    .eq_missing(
                        pl.col(f"{column}__materialized").cast(
                            pl.Float32, strict=False
                        )
                    )
                )
                .sum()
                .alias(column)
                for column in existing
            ]
        ).collect(engine="streaming")
        mismatches = [
            column
            for column in existing
            if int(mismatch_counts[column][0] or 0) > 0
        ]
        if mismatches:
            raise RuntimeError(
                "monthly valuation materializer disagrees with canonical values: "
                f"{mismatches}"
            )
        frame = frame.with_columns(
            [
                pl.coalesce(
                    pl.col(column), pl.col(f"{column}__materialized")
                ).alias(column)
                for column in existing
            ]
        ).drop([f"{column}__materialized" for column in existing])
    valuation_source_name = (
        "quant.features.long_weekly_factors.add_monthly_valuation_history"
    )
    for column in MONTHLY_VALUATION_HISTORY_COLUMNS:
        previous = ownership.get(column)
        ownership[column] = (
            f"{previous} | fill:{valuation_source_name}"
            if previous
            else valuation_source_name
        )
    return frame, {
        "market_sentiment": {
            "source": str(MARKET_DAILY_DIR.relative_to(PROJECT_ROOT)),
            "materializer": "quant.features.market_sentiment.build_limit_proxy_features",
            "rows": int(len(market)),
            "date_min": market["date"].min().date().isoformat(),
            "date_max": market["date"].max().date().isoformat(),
            "missing_candidate_date_count": len(missing_market_dates),
            "missing_candidate_dates": missing_market_dates,
            "missing_date_semantics": "preserve_null_do_not_forward_fill",
            "factors": list(MARKET_SENTIMENT_FACTOR_COLUMNS),
        },
        "top_list": {
            "source": str(TOP_LIST_DIR.relative_to(PROJECT_ROOT)),
            "partition_count": len(top_partition_dates),
            "event_rows": int(len(top)),
            "missing_candidate_date_count": len(missing_top_dates),
            "missing_candidate_dates": missing_top_dates,
            "missing_partition_semantics": "preserve_null_do_not_assume_no_event",
            "factors": list(TOP_LIST_FACTOR_COLUMNS),
        },
        "monthly_valuation_history": {
            "source": str(VALUATION_HISTORY_SOURCE.relative_to(PROJECT_ROOT)),
            "rows": int(len(valuation)),
            "factors": list(MONTHLY_VALUATION_HISTORY_COLUMNS),
            "window_months": 84,
            "minimum_months": 24,
        },
    }


def _scan_labeled_candidates(
    event_path: Path,
    label_path: Path,
    *,
    side: str,
    raw_signals: Sequence[str],
    daily_cap: int,
) -> Any:
    pl = _polars()
    event_columns = set(_parquet_columns(event_path))
    factor_columns = [
        column
        for column in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
        if column in event_columns
    ]
    signal_columns = [column for column in raw_signals if column in event_columns]
    events = (
        _scan_parquet_canonical(event_path)
        .select([*KEY_COLUMNS, *factor_columns, *signal_columns])
        .with_columns(
            pl.col("symbol").cast(pl.String),
            pl.col("date").cast(pl.Datetime("ns")),
        )
    )
    labels = (
        _scan_parquet_canonical(label_path)
        .filter(
            (pl.col("entry_mode") == "next_close")
            & (pl.col("horizon") == 5)
            & pl.col("mature").fill_null(False)
            & pl.col("entry_executable").fill_null(False)
            & ~pl.col("locked_limit_up").fill_null(False)
        )
        .select(
            pl.col("symbol").cast(pl.String),
            pl.col("date").cast(pl.Datetime("ns")),
            pl.col("label_end_date").cast(pl.Datetime("ns")),
            (pl.col("mfe").cast(pl.Float64) * 100.0).alias(
                "future_max_high_t5_pct"
            ),
            (pl.col("terminal_return").cast(pl.Float64) * 100.0).alias(
                "future_return_t5_pct"
            ),
        )
        .filter(
            pl.col("future_max_high_t5_pct").is_not_null()
            & pl.col("future_return_t5_pct").is_not_null()
        )
        .unique(KEY_COLUMNS, keep="last")
    )
    joined = events.join(labels, on=list(KEY_COLUMNS), how="inner", validate="1:1")
    if daily_cap > 0:
        joined = (
            joined.with_columns(
                pl.col("symbol")
                .hash(seed=20260827)
                .rank(method="ordinal")
                .over("date")
                .alias("_daily_sample_rank")
            )
            .filter(pl.col("_daily_sample_rank") <= daily_cap)
            .drop("_daily_sample_rank")
        )
    return joined.with_columns(
        pl.lit(side == "right").alias("right_candidate"),
        pl.lit(side == "left").alias("left_candidate"),
    )


def _normalise_nan(frame: Any, columns: Iterable[str]) -> Any:
    pl = _polars()
    schema = frame.collect_schema()
    expressions = []
    for column in columns:
        if column not in schema:
            continue
        dtype = schema[column]
        if dtype in (pl.Float32, pl.Float64):
            expressions.append(pl.col(column).fill_nan(None))
    return frame.with_columns(expressions) if expressions else frame


def _merge_candidate_sides(right: Any, left: Any) -> tuple[Any, dict[str, Any]]:
    pl = _polars()
    right_names = tuple(right.collect_schema().names())
    left_names = tuple(left.collect_schema().names())
    side_columns = {"right_candidate", "left_candidate"}
    common = sorted(
        (set(right_names) & set(left_names)) - set(KEY_COLUMNS) - side_columns
    )
    right = _normalise_nan(right, common)
    left = _normalise_nan(left, common)
    audit_join = right.select([*KEY_COLUMNS, *common]).join(
        left.select([*KEY_COLUMNS, *common]),
        on=list(KEY_COLUMNS),
        how="inner",
        suffix="__left",
        validate="1:1",
    )
    mismatch_columns: list[str] = []
    if common:
        right_schema = right.collect_schema()
        left_schema = left.collect_schema()
        comparisons = []
        for column in common:
            left_value = pl.col(column)
            right_value = pl.col(f"{column}__left")
            if right_schema[column].is_numeric() and left_schema[column].is_numeric():
                # The right research table is persisted as float32 and the left
                # table as float64.  Compare the exact model-facing float32
                # representation; this rejects semantic drift while ignoring
                # storage-only roundoff below the released input precision.
                left_value = left_value.cast(pl.Float32)
                right_value = right_value.cast(pl.Float32)
            comparisons.append(
                (~left_value.eq_missing(right_value)).sum().alias(column)
            )
        counts = audit_join.select(
            comparisons
        ).collect(engine="streaming")
        mismatch_columns = [
            column for column in common if int(counts[column][0] or 0) > 0
        ]
    if mismatch_columns:
        raise RuntimeError(
            "left/right candidate sources disagree on canonical values: "
            f"{mismatch_columns}"
        )
    joined = right.join(
        left,
        on=list(KEY_COLUMNS),
        how="full",
        coalesce=True,
        suffix="__left",
        validate="1:1",
    )
    expressions = [
        pl.coalesce(pl.col(column), pl.col(f"{column}__left")).alias(column)
        for column in common
    ]
    for column in sorted(side_columns):
        expressions.append(
            (
                pl.col(column).fill_null(False)
                | pl.col(f"{column}__left").fill_null(False)
            ).alias(column)
        )
    joined = joined.with_columns(expressions).drop(
        [
            f"{column}__left"
            for column in (*common, *sorted(side_columns))
        ]
    )
    audit = {
        "overlap_common_column_count": len(common),
        "overlap_mismatch_columns": mismatch_columns,
        "numeric_comparison_precision": "exact_float32_model_input",
    }
    return joined, audit


def _join_exact_factor_source(
    frame: Any,
    path: Path,
    remaining: set[str],
    ownership: dict[str, str],
) -> tuple[Any, set[str]]:
    pl = _polars()
    available = set(_parquet_columns(path))
    selector_history = PROJECT_ROOT / "data/research/selector_model_history_2020.parquet"
    is_selector_history = path.resolve() == selector_history.resolve()
    daily_basic = PROJECT_ROOT / "data/research/right_side_registry_607/daily_basic_features.parquet"
    daily_extras = PROJECT_ROOT / "data/research/right_side_registry_607/daily_extras.parquet"
    factor_layers = {definition.name: definition.layer for definition in FACTOR_REGISTRY}

    def owned_by_source(column: str) -> bool:
        if is_selector_history:
            return factor_layers.get(column) == "selector_live"
        if path.resolve() == daily_basic.resolve():
            return factor_layers.get(column) in {"project_daily", "chan_live"}
        if path.resolve() == daily_extras.resolve():
            return factor_layers.get(column) in {
                "project_daily_candidate",
                "chan_live",
                "long_external_candidate",
            }
        return True

    additions = [
        column
        for column in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
        if column in available
        and owned_by_source(column)
    ]
    if not additions:
        return frame, remaining
    source = (
        _scan_parquet_canonical(path)
        .select([*KEY_COLUMNS, *additions])
        .with_columns(
            pl.col("symbol").cast(pl.String),
            pl.col("date").cast(pl.Datetime("ns")),
        )
        .unique(KEY_COLUMNS, keep="last")
    )
    current = set(frame.collect_schema().names())
    existing = [column for column in additions if column in current]
    frame = frame.join(
        source,
        on=list(KEY_COLUMNS),
        how="left",
        suffix="__source",
        validate="m:1",
    )
    if existing:
        mismatch_counts = frame.select(
            [
                (
                    pl.col(column).is_not_null()
                    & pl.col(f"{column}__source").is_not_null()
                    & ~pl.col(column)
                    .cast(pl.Float32, strict=False)
                    .eq_missing(
                        pl.col(f"{column}__source").cast(
                            pl.Float32, strict=False
                        )
                    )
                )
                .sum()
                .alias(column)
                for column in existing
            ]
        ).collect(engine="streaming")
        mismatches = [
            column
            for column in existing
            if int(mismatch_counts[column][0] or 0) > 0
        ]
        if mismatches:
            raise RuntimeError(
                f"exact PIT source disagrees with existing canonical values: "
                f"{path}: {mismatches}"
            )
        frame = frame.with_columns(
            [
                pl.coalesce(
                    pl.col(column), pl.col(f"{column}__source")
                ).alias(column)
                for column in existing
            ]
        ).drop([f"{column}__source" for column in existing])
    relative = str(path.relative_to(PROJECT_ROOT))
    for column in additions:
        previous = ownership.get(column)
        ownership[column] = (
            f"{previous} | fill:{relative}" if previous else relative
        )
    return frame, remaining - set(additions)


def _join_weekly_factor_source(
    frame: Any,
    path: Path,
    remaining: set[str],
    ownership: dict[str, str],
) -> tuple[Any, set[str]]:
    pl = _polars()
    available = set(_parquet_columns(path))
    factor_layers = {definition.name: definition.layer for definition in FACTOR_REGISTRY}
    analyst = PROJECT_ROOT / "data/research/right_side_registry_607/analyst_weekly.parquet"
    external = PROJECT_ROOT / "data/features/long_entry/weekly_external_v1.parquet"
    quality = PROJECT_ROOT / "data/features/long_entry/weekly_quality_factors_v1.parquet"
    training = PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet"

    def owned_by_source(column: str) -> bool:
        layer = factor_layers.get(column)
        resolved = path.resolve()
        if resolved == analyst.resolve():
            return (
                layer in {"long_snapshot", "long_research"}
                and column.startswith("analyst_")
            )
        if resolved == external.resolve():
            return layer == "long_external_candidate"
        if resolved == quality.resolve():
            return layer == "long_research"
        if resolved == training.resolve():
            return layer in {"long_snapshot", "long_research"} and not column.startswith(
                "analyst_"
            )
        return True

    additions = [
        column
        for column in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
        if column in available
        and owned_by_source(column)
    ]
    if not additions:
        return frame, remaining
    symbol_key = "symbol" if "symbol" in available else "ts_code"
    source = (
        _scan_parquet_canonical(path)
        .select(
            pl.col(symbol_key).cast(pl.String).alias("symbol"),
            pl.col("date").cast(pl.Datetime("ns")),
            *additions,
        )
        .unique(KEY_COLUMNS, keep="last")
        .sort(["symbol", "date"])
    )
    current = set(frame.collect_schema().names())
    existing = [column for column in additions if column in current]
    frame = frame.sort(["symbol", "date"]).join_asof(
        source,
        on="date",
        by="symbol",
        strategy="backward",
        tolerance="14d",
        suffix="__source",
        check_sortedness=False,
    )
    if existing:
        mismatch_counts = frame.select(
            [
                (
                    pl.col(column).is_not_null()
                    & pl.col(f"{column}__source").is_not_null()
                    & ~pl.col(column)
                    .cast(pl.Float32, strict=False)
                    .eq_missing(
                        pl.col(f"{column}__source").cast(
                            pl.Float32, strict=False
                        )
                    )
                )
                .sum()
                .alias(column)
                for column in existing
            ]
        ).collect(engine="streaming")
        mismatches = [
            column
            for column in existing
            if int(mismatch_counts[column][0] or 0) > 0
        ]
        if mismatches:
            raise RuntimeError(
                f"weekly PIT source disagrees with existing canonical values: "
                f"{path}: {mismatches}"
            )
        frame = frame.with_columns(
            [
                pl.coalesce(
                    pl.col(column), pl.col(f"{column}__source")
                ).alias(column)
                for column in existing
            ]
        ).drop([f"{column}__source" for column in existing])
    relative = str(path.relative_to(PROJECT_ROOT))
    for column in additions:
        previous = ownership.get(column)
        ownership[column] = (
            f"{previous} | fill:{relative}" if previous else relative
        )
    return frame, remaining - set(additions)


def _with_strategy_group_features(frame: Any) -> Any:
    pl = _polars()
    names = set(frame.collect_schema().names())
    raw_members = set(member for values in GROUP_MEMBERS.values() for member in values)
    missing_members = sorted(raw_members - names)
    if missing_members:
        frame = frame.with_columns(
            [pl.lit(False).alias(column) for column in missing_members]
        )
    group_expressions: dict[str, Any] = {}
    for group in ALL_SHORT_GROUPS:
        members = list(GROUP_MEMBERS[group])
        inputs = [
            pl.col(member).fill_null(False).cast(pl.Boolean)
            for member in members
        ]
        if group in names:
            inputs.append(pl.col(group).fill_null(False).cast(pl.Boolean))
        group_expressions[group] = pl.any_horizontal(
            inputs
        )
    expressions = []
    for group, expression in group_expressions.items():
        feature = f"group__{group}"
        if feature in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS:
            expressions.append(expression.cast(pl.Float32).alias(feature))
    expressions.append(
        pl.sum_horizontal(
            [expression.cast(pl.Int16) for expression in group_expressions.values()]
        )
        .cast(pl.Float32)
        .alias("matched_count")
    )
    return frame.with_columns(expressions)


def build_dataset(
    output: Path,
    *,
    right_daily_cap: int,
    left_daily_cap: int,
) -> dict[str, Any]:
    pl = _polars()
    for path in (
        RIGHT_EVENTS,
        RIGHT_LABELS,
        LEFT_EVENTS,
        LEFT_LABELS,
        *EXACT_FACTOR_SOURCES,
        *WEEKLY_FACTOR_SOURCES,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        _parquet_columns(path)
    right = _scan_labeled_candidates(
        RIGHT_EVENTS,
        RIGHT_LABELS,
        side="right",
        raw_signals=RIGHT_RAW_SIGNALS,
        daily_cap=right_daily_cap,
    )
    left = _scan_labeled_candidates(
        LEFT_EVENTS,
        LEFT_LABELS,
        side="left",
        raw_signals=LEFT_RAW_SIGNALS,
        daily_cap=left_daily_cap,
    )
    frame, overlap_audit = _merge_candidate_sides(right, left)
    initial_names = set(frame.collect_schema().names())
    ownership = {
        column: "left/right unified candidate datasets"
        for column in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
        if column in initial_names
    }
    remaining = set(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS) - initial_names
    for path in EXACT_FACTOR_SOURCES:
        frame, remaining = _join_exact_factor_source(
            frame, path, remaining, ownership
        )
    for path in WEEKLY_FACTOR_SOURCES:
        frame, remaining = _join_weekly_factor_source(
            frame, path, remaining, ownership
        )
    frame, derived_materialization = _with_full_history_derived_factors(
        frame,
        ownership,
    )
    frame = _with_strategy_group_features(frame)
    final_names = set(frame.collect_schema().names())
    missing = [
        column
        for column in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
        if column not in final_names
    ]
    if missing:
        raise RuntimeError(
            "selector training dataset has registry factors without a historical "
            f"materializer: {missing}"
        )
    factor_expressions = []
    schema = frame.collect_schema()
    for column in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS:
        expression = pl.col(column)
        if schema[column] in (pl.Float32, pl.Float64):
            expression = expression.fill_nan(None)
        factor_expressions.append(expression.cast(pl.Float32, strict=False).alias(column))
    frame = frame.with_columns(factor_expressions)
    selected_columns = [
        *KEY_COLUMNS,
        *METADATA_COLUMNS,
        *TARGET_COLUMNS,
        *SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS,
    ]
    frame = frame.select(selected_columns).sort(["date", "symbol"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    frame.sink_parquet(
        temporary,
        compression="zstd",
        statistics=True,
        engine="streaming",
    )
    os.replace(temporary, output)
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(output)
    date_frame = pl.scan_parquet(output).select(
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        pl.col("right_candidate").sum().alias("right_rows"),
        pl.col("left_candidate").sum().alias("left_rows"),
    ).collect(engine="streaming")
    coverage = _coverage_rows(output, pd.Timestamp(date_frame["date_max"][0]).year)
    materialization_audit = audit_selector_buy_hold_training_materialization(coverage)
    manifest = {
        "status": (
            "success" if materialization_audit.complete else "incomplete_materialization"
        ),
        "schema_version": "selector-buy-hold-registry-dataset-v2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_contract": {
            "candidate_sources": "released right-side and left-side unified ranker samples",
            "entry_mode": "next_close",
            "horizon": 5,
            "locked_limit_up_excluded": True,
            "mature_only": True,
            "entry_executable_only": True,
            "right_daily_cap": right_daily_cap,
            "left_daily_cap": left_daily_cap,
            "sample_hash_seed": 20260827,
        },
        "rows": int(parquet.metadata.num_rows),
        "columns": int(len(parquet.schema.names)),
        "date_min": pd.Timestamp(date_frame["date_min"][0]).date().isoformat(),
        "date_max": pd.Timestamp(date_frame["date_max"][0]).date().isoformat(),
        "right_candidate_rows": int(date_frame["right_rows"][0]),
        "left_candidate_rows": int(date_frame["left_rows"][0]),
        **selector_buy_hold_factor_contract_payload(),
        "source_signatures": [
            _source_signature(path)
            for path in (
                RIGHT_EVENTS,
                RIGHT_LABELS,
                LEFT_EVENTS,
                LEFT_LABELS,
                *EXACT_FACTOR_SOURCES,
                *WEEKLY_FACTOR_SOURCES,
            )
        ],
        "feature_ownership": ownership,
        "unmaterialized_candidate_features": [
            issue.factor for issue in materialization_audit.issues
        ],
        "training_materialization_audit": training_materialization_audit_payload(
            materialization_audit
        ),
        "training_materialization_plan": selector_buy_hold_training_materialization_plan(),
        "derived_materialization": derived_materialization,
        "overlap_audit": overlap_audit,
        "output": str(output.relative_to(PROJECT_ROOT)),
        "output_sha256": _sha256(output),
    }
    atomic_write_json(manifest, DATASET_MANIFEST_PATH)
    require_complete_selector_buy_hold_training_materialization(coverage)
    return manifest


def historical_scores(
    predictions: np.ndarray,
    reference: np.ndarray,
    width: float,
) -> np.ndarray:
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("historical score reference is empty")
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = max(float((q75 - q25) / 1.349), 1e-6)
    z_score = (np.asarray(predictions, dtype=float) - median) / scale
    return np.clip(
        50.0 + 100.0 / np.pi * np.arctan(z_score / max(width, 0.1)),
        0.0,
        100.0,
    )


def evaluate_ranking(
    dates: pd.Series,
    target: pd.Series,
    predictions: np.ndarray,
    *,
    hit_threshold: float,
) -> RankingMetrics:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "target": pd.to_numeric(target, errors="coerce").to_numpy(),
            "prediction": np.asarray(predictions, dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("ranking evaluation frame is empty")
    frame["target_rank"] = frame.groupby("date")["target"].rank(
        method="average", pct=True
    )
    frame["prediction_rank"] = frame.groupby("date")["prediction"].rank(
        method="average", pct=True
    )
    daily = frame.groupby("date", sort=True).apply(
        lambda part: part["target_rank"].corr(part["prediction_rank"]),
        include_groups=False,
    ).dropna()
    frame["decile"] = np.ceil(frame["prediction_rank"] * 10).clip(1, 10).astype(int)
    deciles = frame.groupby("decile", as_index=False)["target"].mean()
    low = float(deciles.loc[deciles["decile"].eq(1), "target"].iloc[0])
    high = float(deciles.loc[deciles["decile"].eq(10), "target"].iloc[0])
    trend = spearmanr(deciles["decile"], deciles["target"]).statistic
    top20 = frame.sort_values(
        ["date", "prediction"], ascending=[True, False]
    ).groupby("date", sort=False).head(20)
    global_spearman = spearmanr(frame["prediction"], frame["target"]).statistic
    return RankingMetrics(
        rows=int(len(frame)),
        days=int(frame["date"].nunique()),
        global_spearman=float(global_spearman) if np.isfinite(global_spearman) else 0.0,
        daily_spearman=float(daily.mean()) if len(daily) else 0.0,
        positive_daily_spearman_ratio=float((daily > 0).mean()) if len(daily) else 0.0,
        decile_spread=high - low,
        decile_trend=float(trend) if np.isfinite(trend) else 0.0,
        top20_avg_return=float(top20["target"].mean()),
        top20_hit_rate=float((top20["target"] > hit_threshold).mean()),
    )


def _coverage_rows(dataset: Path, train_end_year: int) -> pd.DataFrame:
    pl = _polars()
    frame = pl.scan_parquet(dataset).filter(pl.col("date").dt.year() <= train_end_year)
    expressions = []
    for index, column in enumerate(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS):
        expressions.extend(
            (
                pl.col(column).is_not_null().sum().alias(f"n{index}"),
                pl.col(column).drop_nulls().n_unique().alias(f"u{index}"),
            )
        )
    summary = frame.select(pl.len().alias("rows"), *expressions).collect(
        engine="streaming"
    )
    row_count = int(summary["rows"][0])
    rows = []
    for index, column in enumerate(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS):
        non_null = int(summary[f"n{index}"][0])
        unique = int(summary[f"u{index}"][0])
        rows.append(
            {
                "factor": column,
                "train_rows": row_count,
                "non_null": non_null,
                "coverage": float(non_null / row_count) if row_count else 0.0,
                "unique_values": unique,
            }
        )
    return pd.DataFrame(rows)


def _tree_selected_features(
    models: Mapping[str, XGBRegressor],
    features: tuple[str, ...],
    *,
    production_features: tuple[str, ...] = SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS,
) -> tuple[str, ...]:
    """Freeze tree-used factors that have a complete production materializer."""

    used: set[str] = set()
    for model in models.values():
        importances = np.asarray(model.feature_importances_, dtype=float)
        if len(importances) != len(features):
            raise RuntimeError("selector discovery feature importance shape drifted")
        used.update(
            feature
            for feature, importance in zip(features, importances)
            if np.isfinite(importance) and float(importance) > 0.0
        )
    production_set = set(production_features)
    selected = tuple(
        feature for feature in features
        if feature in used and feature in production_set
    )
    if not selected:
        raise RuntimeError("selector discovery models selected no production factors")
    assert_no_forbidden_factor_names(
        selected,
        context="selector buy/hold selected production features",
    )
    return selected


def _model(n_jobs: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=360,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=30,
        subsample=0.80,
        colsample_bytree=0.72,
        reg_lambda=10.0,
        reg_alpha=0.05,
        max_bin=96,
        tree_method="hist",
        random_state=20260827,
        n_jobs=n_jobs,
    )


def _date_weights(dates: pd.Series) -> np.ndarray:
    counts = dates.groupby(dates).transform("size").astype(float)
    weights = 1.0 / counts
    return (weights / weights.mean()).to_numpy(dtype=np.float32)


def _fold_masks(data: pd.DataFrame, fold: YearFold) -> dict[str, np.ndarray]:
    years = data["date"].dt.year.to_numpy()
    label_end = data["label_end_date"].dt.normalize()
    train_boundary = pd.Timestamp(f"{fold.train_end_year}-12-31")
    valid_boundary = pd.Timestamp(f"{fold.validation_year}-12-31")
    return {
        "train": (
            (years >= fold.train_start_year)
            & (years <= fold.train_end_year)
            & label_end.le(train_boundary).to_numpy()
        ),
        "validation": (
            (years == fold.validation_year)
            & label_end.le(valid_boundary).to_numpy()
        ),
        "test": years == fold.test_year,
    }


def _training_target(values: pd.Series, dates: pd.Series, mode: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if mode == "hold":
        return (
            numeric.groupby(dates).rank(method="average", pct=True).sub(0.5)
        ).to_numpy(dtype=np.float32)
    return numeric.clip(-5.0, 30.0).to_numpy(dtype=np.float32)


def _predict_artifact(artifact: Mapping[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = [str(value) for value in artifact.get("features") or ()]
    values = frame.reindex(columns=features)
    imputer = artifact.get("imputer")
    transformed: Any = imputer.transform(values) if imputer is not None else values
    if isinstance(artifact.get("models"), Mapping):
        component_scores = {}
        for component, model in artifact["models"].items():
            component_scores[component] = historical_scores(
                model.predict(transformed),
                np.asarray(artifact["score_references"][component]),
                float((artifact.get("normalization_widths") or {}).get(component, 2.0)),
            )
        weight = float(artifact.get("buy_weight", 0.0))
        return weight * component_scores["buy"] + (1.0 - weight) * component_scores["hold"]
    predictions = artifact["model"].predict(transformed)
    return historical_scores(
        predictions,
        np.asarray(artifact["score_reference"]),
        float(artifact.get("normalization_width", 2.0)),
    )


def _ranking_objective(metrics: RankingMetrics) -> float:
    return (
        metrics.daily_spearman * 8.0
        + metrics.decile_spread * 0.20
        + metrics.decile_trend * 0.10
    )


def _train_fold(
    data: pd.DataFrame,
    features: tuple[str, ...],
    fold: YearFold,
    *,
    n_jobs: int,
    incumbent: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[str, ...]]:
    masks = _fold_masks(data, fold)
    x_columns = list(features)
    train_frame = data.loc[masks["train"], x_columns]
    valid_frame = data.loc[masks["validation"], x_columns]
    weights = _date_weights(data.loc[masks["train"], "date"])
    dev_models: dict[str, XGBRegressor] = {}
    dev_references: dict[str, np.ndarray] = {}
    valid_scores: dict[str, np.ndarray] = {}
    for mode, target in (
        ("buy", "future_max_high_t5_pct"),
        ("hold", "future_return_t5_pct"),
    ):
        model = _model(n_jobs)
        model.fit(
            train_frame,
            _training_target(
                data.loc[masks["train"], target],
                data.loc[masks["train"], "date"],
                mode,
            ),
            sample_weight=weights,
            verbose=False,
        )
        reference = np.sort(model.predict(train_frame).astype(float))
        dev_models[mode] = model
        dev_references[mode] = reference
        valid_scores[mode] = historical_scores(
            model.predict(valid_frame), reference, 6.0 if mode == "buy" else 2.0
        )
    selected_features = _tree_selected_features(dev_models, features)
    if selected_features != features:
        del train_frame, valid_frame, dev_models, dev_references, valid_scores
        gc.collect()
        x_columns = list(selected_features)
        train_frame = data.loc[masks["train"], x_columns]
        valid_frame = data.loc[masks["validation"], x_columns]
        dev_models = {}
        dev_references = {}
        valid_scores = {}
        for mode, target in (
            ("buy", "future_max_high_t5_pct"),
            ("hold", "future_return_t5_pct"),
        ):
            model = _model(n_jobs)
            model.fit(
                train_frame,
                _training_target(
                    data.loc[masks["train"], target],
                    data.loc[masks["train"], "date"],
                    mode,
                ),
                sample_weight=weights,
                verbose=False,
            )
            reference = np.sort(model.predict(train_frame).astype(float))
            dev_models[mode] = model
            dev_references[mode] = reference
            valid_scores[mode] = historical_scores(
                model.predict(valid_frame),
                reference,
                6.0 if mode == "buy" else 2.0,
            )
    blend_rows = []
    for buy_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        prediction = (
            buy_weight * valid_scores["buy"]
            + (1.0 - buy_weight) * valid_scores["hold"]
        )
        metrics = evaluate_ranking(
            data.loc[masks["validation"], "date"],
            data.loc[masks["validation"], "future_return_t5_pct"],
            prediction,
            hit_threshold=0.0,
        )
        blend_rows.append(
            {"buy_weight": buy_weight, "metrics": asdict(metrics), "objective": _ranking_objective(metrics)}
        )
    selected_blend = max(blend_rows, key=lambda row: float(row["objective"]))
    buy_weight = float(selected_blend["buy_weight"])
    del train_frame, valid_frame, dev_models, dev_references, valid_scores
    gc.collect()

    fit_mask = masks["train"] | masks["validation"]
    fit_frame = data.loc[fit_mask, x_columns]
    test_frame = data.loc[masks["test"], x_columns]
    fit_weights = _date_weights(data.loc[fit_mask, "date"])
    models: dict[str, XGBRegressor] = {}
    references: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for mode, target in (
        ("buy", "future_max_high_t5_pct"),
        ("hold", "future_return_t5_pct"),
    ):
        model = _model(n_jobs)
        model.fit(
            fit_frame,
            _training_target(
                data.loc[fit_mask, target], data.loc[fit_mask, "date"], mode
            ),
            sample_weight=fit_weights,
            verbose=False,
        )
        reference = np.sort(model.predict(fit_frame).astype(float))
        models[mode] = model
        references[mode] = reference
        test_scores[mode] = historical_scores(
            model.predict(test_frame), reference, 6.0 if mode == "buy" else 2.0
        )
    challenger_predictions = {
        "buy": test_scores["buy"],
        "hold": buy_weight * test_scores["buy"] + (1.0 - buy_weight) * test_scores["hold"],
    }
    reports: dict[str, dict[str, Any]] = {}
    for mode, target, threshold in (
        ("buy", "future_max_high_t5_pct", 5.0),
        ("hold", "future_return_t5_pct", 0.0),
    ):
        challenger_metrics = evaluate_ranking(
            data.loc[masks["test"], "date"],
            data.loc[masks["test"], target],
            challenger_predictions[mode],
            hit_threshold=threshold,
        )
        incumbent_prediction = _predict_artifact(
            incumbent[mode], data.loc[masks["test"]]
        )
        incumbent_metrics = evaluate_ranking(
            data.loc[masks["test"], "date"],
            data.loc[masks["test"], target],
            incumbent_prediction,
            hit_threshold=threshold,
        )
        reports[mode] = {
            "target": target,
            "challenger": asdict(challenger_metrics),
            "incumbent_49_factor": asdict(incumbent_metrics),
            "daily_spearman_delta": challenger_metrics.daily_spearman
            - incumbent_metrics.daily_spearman,
            "decile_spread_delta": challenger_metrics.decile_spread
            - incumbent_metrics.decile_spread,
            "non_inferiority_passed": bool(
                challenger_metrics.daily_spearman + 1e-12
                >= incumbent_metrics.daily_spearman
                and challenger_metrics.decile_spread > 0.0
            ),
        }
    fold_payload = {
        "fold": fold.name,
        "train_years": [fold.train_start_year, fold.train_end_year],
        "validation_year": fold.validation_year,
        "test_year": fold.test_year,
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "discovery_factor_count": len(features),
        "selected_model_input_count": len(selected_features),
        "selected_model_input_columns": list(selected_features),
        "selection_method": (
            "full_registry_nonzero_tree_gain_union_then_"
            "production_materializer_gate_then_retrain"
        ),
        "production_materializable_factor_count": len(
            SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS
        ),
        "selected_hold_buy_weight": buy_weight,
        "hold_blend_validation": blend_rows,
        "models": reports,
    }
    artifacts = {
        "buy": {
            "model": models["buy"],
            "score_reference": references["buy"],
            "normalization_width": 6.0,
        },
        "hold": {
            "models": models,
            "score_references": references,
            "normalization_widths": {"buy": 6.0, "hold": 2.0},
            "buy_weight": buy_weight,
        },
    }
    del fit_frame, test_frame, test_scores, challenger_predictions
    gc.collect()
    return fold_payload, artifacts, selected_features


def _artifact_payload(
    *,
    mode: str,
    features: tuple[str, ...],
    exclusions: list[dict[str, Any]],
    sample_manifest: Mapping[str, Any],
    fold: YearFold,
    fitted: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SELECTOR_BUY_HOLD_ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": SELECTOR_BUY_HOLD_FEATURE_SCHEMA_VERSION,
        "release_id": SELECTOR_BUY_HOLD_RELEASE_ID,
        "mode": mode,
        "target": (
            "future_max_high_t5_pct" if mode == "buy" else "future_return_t5_pct"
        ),
        "candidate_features": list(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        "features": list(features),
        "factor_contract_sha256": SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256,
        "model_input_contract_sha256": selector_buy_hold_model_input_sha256(features),
        "feature_exclusions": [
            *exclusions,
            *(
                {
                    "factor": candidate,
                    "reason": "zero_tree_gain_after_full_registry_discovery",
                }
                for candidate in SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
                if candidate not in set(features)
            ),
        ],
        "discovery_factor_count": len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        "selection_method": (
            "full_registry_nonzero_tree_gain_union_then_"
            "production_materializer_gate_then_retrain"
        ),
        "production_materializable_factor_count": len(
            SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS
        ),
        "preprocessing": "xgboost_native_nan_float32_v1",
        "score_definition": "fixed robust historical transform of predicted next-close five-session return target",
        "trained_through": str(fold.validation_year),
        "sample_schema_version": sample_manifest.get("schema_version"),
        "sample_sha256": sample_manifest.get("output_sha256"),
        **dict(fitted),
    }
    validate_selector_buy_hold_artifact(payload)
    return payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    sample_manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    if sample_manifest.get("factor_contract_sha256") != SELECTOR_BUY_HOLD_FACTOR_CONTRACT_SHA256:
        raise RuntimeError("selector buy/hold dataset factor contract drifted")
    folds_by_name = {fold.name: fold for fold in DEFAULT_YEAR_FOLDS}
    requested = tuple(value.strip().upper() for value in args.folds.split(",") if value.strip())
    folds = [folds_by_name[name] for name in requested]
    coverage = _coverage_rows(args.dataset, max(fold.test_year for fold in folds))
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    audit = audit_selector_buy_hold_training_materialization(
        coverage,
        minimum_observations=args.minimum_factor_observations,
    )
    issue_status = {
        issue.factor: issue.status for issue in (*audit.issues, *audit.warnings)
    }
    coverage["materialization_status"] = coverage["factor"].map(
        issue_status
    ).fillna("usable_informative")
    atomic_write_csv(coverage, FEATURE_COVERAGE_PATH, index=False)
    atomic_write_json(
        training_materialization_audit_payload(audit),
        MATERIALIZATION_AUDIT_PATH,
    )
    require_complete_selector_buy_hold_training_materialization(
        coverage,
        minimum_observations=args.minimum_factor_observations,
    )
    features = SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS
    exclusions: list[dict[str, Any]] = []
    old_root = PROJECT_ROOT / "models/production/selector_buy_hold"
    incumbent = {
        mode: joblib.load(old_root / f"{mode}.joblib") for mode in ("buy", "hold")
    }
    incumbent_features = tuple(
        dict.fromkeys(
            str(value)
            for artifact in incumbent.values()
            for value in artifact.get("features") or ()
        )
    )
    read_features = tuple(dict.fromkeys((*features, *incumbent_features)))
    data = pd.read_parquet(
        args.dataset,
        columns=[*KEY_COLUMNS, *METADATA_COLUMNS, *TARGET_COLUMNS, *read_features],
    )
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["label_end_date"] = pd.to_datetime(data["label_end_date"], errors="raise")
    for column in read_features:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype(np.float32)
    fold_reports = []
    final_artifacts: dict[str, Any] = {}
    final_features: tuple[str, ...] = ()
    final_fold: YearFold | None = None
    for fold in folds:
        report, fitted, selected_features = _train_fold(
            data,
            features,
            fold,
            n_jobs=args.n_jobs,
            incumbent=incumbent,
        )
        fold_reports.append(report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if fold == folds[-1]:
            final_fold = fold
            final_features = selected_features
            final_artifacts = {
                mode: _artifact_payload(
                    mode=mode,
                    features=selected_features,
                    exclusions=exclusions,
                    sample_manifest=sample_manifest,
                    fold=fold,
                    fitted=fitted[mode],
                )
                for mode in ("buy", "hold")
            }
    if final_fold is None:
        raise RuntimeError("no selector buy/hold folds were trained")
    CANDIDATE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_hashes = {}
    for mode, artifact in final_artifacts.items():
        path = CANDIDATE_MODEL_ROOT / f"{mode}.joblib"
        joblib.dump(artifact, path, compress=3)
        candidate_hashes[mode] = _sha256(path)
    promotion_eligible = bool(
        fold_reports
        and all(
            model["non_inferiority_passed"]
            for fold in fold_reports
            for model in fold["models"].values()
        )
    )
    production_paths: dict[str, str] = {}
    production_hashes: dict[str, str] = {}
    if args.promote and promotion_eligible:
        PRODUCTION_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        for mode, artifact in final_artifacts.items():
            path = PRODUCTION_MODEL_ROOT / f"{mode}.joblib"
            joblib.dump(artifact, path, compress=3)
            production_paths[mode] = str(path.relative_to(PROJECT_ROOT))
            production_hashes[mode] = _sha256(path)
        manifest = {
            "status": "success",
            "schema_version": SELECTOR_BUY_HOLD_MANIFEST_SCHEMA_VERSION,
            "release_id": SELECTOR_BUY_HOLD_RELEASE_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "lifecycle": "production",
            **selector_buy_hold_factor_contract_payload(),
            "discovery_factor_count": len(features),
            "production_materializable_factor_count": len(
                SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS
            ),
            "model_input_count": len(final_features),
            "model_input_columns": list(final_features),
            "model_input_contract_sha256": selector_buy_hold_model_input_sha256(final_features),
            "sample_manifest": str(DATASET_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "sample_sha256": sample_manifest.get("output_sha256"),
            "validation": fold_reports,
            "models": {
                mode: {
                    "path": production_paths[mode],
                    "sha256": production_hashes[mode],
                    "target": final_artifacts[mode]["target"],
                }
                for mode in ("buy", "hold")
            },
            "rollback": {
                "manifest": "models/production/selector_buy_hold/manifest.json",
                "artifacts": [
                    "models/production/selector_buy_hold/buy.joblib",
                    "models/production/selector_buy_hold/hold.joblib",
                ],
            },
        }
        atomic_write_json(manifest, PRODUCTION_MODEL_ROOT / "manifest.json")
    payload = {
        "status": "success",
        "schema_version": "selector-buy-hold-registry-training-report-v3",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "release_id": SELECTOR_BUY_HOLD_RELEASE_ID,
        "candidate_factor_count": len(SELECTOR_BUY_HOLD_CANDIDATE_FACTOR_COLUMNS),
        "production_materializable_factor_count": len(
            SELECTOR_BUY_HOLD_PRODUCTION_FACTOR_COLUMNS
        ),
        "materialized_feature_count": audit.usable_factor_count,
        "informative_feature_count": audit.informative_factor_count,
        "materialization_complete": audit.complete,
        "model_input_count": len(final_features),
        "model_input_columns": list(final_features),
        "model_input_contract_sha256": selector_buy_hold_model_input_sha256(final_features),
        "selection_excluded_feature_count": len(features) - len(final_features),
        "folds": fold_reports,
        "promotion_eligible": promotion_eligible,
        "promoted": bool(production_paths),
        "candidate_artifact_hashes": candidate_hashes,
        "production_artifact_hashes": production_hashes,
        "forbidden_alias_intersection": sorted(
            set(final_features) & set(FORBIDDEN_COMPATIBILITY_ALIASES)
        ),
    }
    atomic_write_json(payload, REPORT_PATH)
    return payload


def main() -> None:
    args = parse_args()
    if args.build_dataset or not args.dataset.exists():
        manifest = build_dataset(
            args.dataset,
            right_daily_cap=args.right_daily_cap,
            left_daily_cap=args.left_daily_cap,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if args.dataset_only:
        return
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
