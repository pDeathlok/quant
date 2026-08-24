#!/usr/bin/env python
"""Build and validate pooled models for current right-side and mixed strategies."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet, atomic_write_text
from quant.data.source_merge import normalize_tushare_daily
from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    LEGACY_TO_CANONICAL_FACTOR_NAMES,
    assert_no_forbidden_factor_names,
    migrate_legacy_factor_columns,
    stable_canonical_feature_union,
)
from quant.features.project_factor_layer import (
    PROJECT_FACTOR_SCHEMA_VERSION,
    calculate_project_market_factors,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.features.right_side_factor_contract import (
    RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
    RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
    RIGHT_SIDE_SHADOW_MODEL_INPUT_CONTRACT_SHA256,
    factor_contract_sha256,
)
from quant.ml.xgb_research import XGBResearchModel
from quant.research.right_side_long_task import (
    DEFAULT_XGB_CLASSIFIER_SPEC,
    LONG_TASK_ARM_SPECS,
    LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
    LONG_TASK_SCHEMA_VERSION,
    LONG_TASK_FEATURE_COLUMNS,
    XGBClassifierSpec,
    LongTaskUnifiedModel,
    aggregate_long_task_predictions,
    expand_long_task_rows,
    inverse_sqrt_task_weights,
    merge_prediction_artifacts,
    apply_event_calibrator,
)
from quant.research.right_side_unified import (
    B2_SOURCE_COLUMNS,
    B3_SOURCE_COLUMNS,
    DEFAULT_YEAR_FOLDS,
    Float32NaNTransformer,
    ProbabilityCalibratedModel,
    RIGHT_SIDE_SIGNALS,
    aggregate_independent_predictions,
    balanced_sample_weights,
    binary_metrics,
    choose_validation_threshold,
    daily_top_k_trading_metrics,
    ensure_model_features,
    signal_metrics,
    split_by_year_fold,
)
from quant.research.right_side_unified_features import (
    ADDED_RULE_FEATURE_COLUMNS_V2,
    LEGACY_RULE_FEATURE_COLUMNS_V1,
    RULE_FEATURE_COLUMNS_SHA256,
    RULE_FEATURE_SCHEMA_VERSION,
    RULE_FEATURE_COLUMNS,
    SIGNAL_FEATURE_REQUIREMENTS,
    audit_factor_coverage,
    compute_right_side_rule_features,
    rule_feature_columns_sha256,
    validate_signal_factor_contract,
)
from quant.research.right_side_beam_feature_selection import (
    BEAM_SCHEMA_VERSION,
    BeamEvaluation,
    BeamSettings,
    backward_beam_search,
    build_validation_search_and_pipeline_select,
    deterministic_stratified_event_sample,
    evaluation_record,
    evaluate_pipeline_select_variants,
    feature_columns_sha256 as beam_feature_columns_sha256,
    combine_residual_probabilities,
    ranking_lift,
    score_candidate as score_beam_candidate,
    search_manifest as beam_search_manifest,
    visited_candidate_permutation_test,
)
from quant.research.right_side_unified_labels import build_right_side_unified_labels
from quant.research.right_side_paired_comparison import paired_model_comparisons
from quant.research.right_side_factor_increment_comparison import (
    compare_rule_feature_versions,
)
from quant.research.right_side_legacy_ranking_comparison import (
    compare_candidate_with_legacy_artifact,
)
from quant.research.right_side_targets import (
    SUPPORTED_TRAINING_TARGETS,
    materialize_training_target,
    target_contract,
    target_metadata,
    target_source_columns,
    validate_target_cost,
)
from quant.research.right_side_unified_signals import (
    CANONICAL_SIGNAL_SCHEMA_VERSION,
    SIGNAL_CONTRACT_NOTES,
    compute_canonical_z_signal_flags,
    merge_canonical_signal_flags,
)


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-08-12"
DEFAULT_Z_CACHE = PROJECT_ROOT / "data/features/z_skill_daily_candidates.parquet"
DEFAULT_FAMILY_CACHE = PROJECT_ROOT / "data/features/b1/b1_family_rule_candidates.parquet"
DEFAULT_DAILY_PARTITIONS = PROJECT_ROOT / "data/raw/daily_partitioned"
DEFAULT_DAILY_BASIC = PROJECT_ROOT / "data/raw/daily_basic"
DEFAULT_TRADABILITY = PROJECT_ROOT / "data/raw/tradability"
CANONICAL_RELEASE_SLUG = "right_side_unified_canonical_v5_rule113"
DEFAULT_RESEARCH_ROOT = PROJECT_ROOT / "data/research" / CANONICAL_RELEASE_SLUG
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research" / CANONICAL_RELEASE_SLUG
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research" / CANONICAL_RELEASE_SLUG

DATASET_PATH = DEFAULT_RESEARCH_ROOT / "unified_right_side_dataset.parquet"
LABEL_DATASET_PATH = DEFAULT_RESEARCH_ROOT / "unified_right_side_labels.parquet"
MANIFEST_PATH = DEFAULT_RESEARCH_ROOT / "dataset_manifest.json"
FACTOR_AUDIT_PATH = DEFAULT_REPORT_ROOT / "factor_coverage.csv"
SAMPLE_AUDIT_PATH = DEFAULT_REPORT_ROOT / "sample_audit.json"
METRICS_PATH = DEFAULT_REPORT_ROOT / "model_metrics.csv"
SIGNAL_METRICS_PATH = DEFAULT_REPORT_ROOT / "signal_metrics.csv"
PREDICTIONS_PATH = DEFAULT_REPORT_ROOT / "test_predictions.parquet"
PAIRED_COMPARISON_PATH = DEFAULT_REPORT_ROOT / "paired_model_comparison.csv"
REPORT_PATH = DEFAULT_REPORT_ROOT / "validation_report.md"
FACTOR_INCREMENT_PATH = DEFAULT_REPORT_ROOT / "rule_factor_increment_ab.csv"
RANKING_DECISION_PATH = DEFAULT_REPORT_ROOT / "ranking_promotion_decision_ab.json"
BEAM_REPORT_ROOT = DEFAULT_REPORT_ROOT / "beam"
LEGACY_ARTIFACT_PREDICTIONS_PATH = (
    DEFAULT_REPORT_ROOT / "legacy_artifact_baseline" / "event_predictions.parquet"
)
LEGACY_RANKING_COMPARISON_PATH = (
    DEFAULT_REPORT_ROOT / "legacy_artifact_ranking_comparison_ab.csv"
)

SUBVARIANT_COLUMNS: tuple[str, ...] = (*B2_SOURCE_COLUMNS, *B3_SOURCE_COLUMNS)
IDENTITY_COLUMNS: tuple[str, ...] = (
    *RIGHT_SIDE_SIGNALS,
    *SUBVARIANT_COLUMNS,
    "signal_count",
    "has_right_signal",
    "has_mixed_signal",
)
LABEL_COLUMNS: tuple[str, ...] = (
    "mfe",
    "mae",
    "terminal",
    "terminal_return",
    "hit_up3",
    "hit_up5",
    "hit_up8",
    "hit_down3",
    "good_path5",
)
EXPERIMENTS: tuple[str, ...] = (
    "rule_only",
    "independent",
    "unified_without_signal_id",
    "unified_with_signal_id",
    "unified_balanced",
    "unified_long_task",
    "unified_long_task_balanced",
    "unified_long_task_deep_rule105",
    "unified_long_task_deep",
    "unified_long_task_deep_beam",
)
DEFAULT_TRAIN_EXPERIMENTS: tuple[str, ...] = ("unified_long_task_deep",)
PAIRED_EXPERIMENTS: tuple[str, ...] = (
    "unified_with_signal_id",
    "unified_balanced",
    "unified_long_task",
    "unified_long_task_balanced",
    "unified_long_task_deep_rule105",
    "unified_long_task_deep",
    "unified_long_task_deep_beam",
)


LABEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "date",
    "entry_mode",
    "horizon",
    "entry_date",
    "label_end_date",
    "entry_executable",
    "locked_limit_up",
    "locked_limit_source",
    "entry_raw_price",
    "entry_price",
    "mature",
    "maturity_reason",
    *LABEL_COLUMNS,
)


def _read_canonical_feature_frame(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read one cache through the strict legacy-to-canonical boundary."""

    import pyarrow.parquet as pq

    available = tuple(pq.ParquetFile(path).schema.names)
    requested = None if columns is None else tuple(dict.fromkeys(columns))
    if requested is None:
        read_columns = list(available)
    else:
        alias_by_canonical = {
            canonical: alias
            for alias, canonical in LEGACY_TO_CANONICAL_FACTOR_NAMES.items()
        }
        missing = sorted(
            column
            for column in requested
            if column not in available
            and alias_by_canonical.get(column) not in available
        )
        if missing:
            raise RuntimeError(f"feature dataset missing required columns: {missing}")
        read_columns = [column for column in requested if column in available]
        read_columns.extend(
            alias
            for alias in sorted(FORBIDDEN_COMPATIBILITY_ALIASES)
            if alias in available and alias not in read_columns
        )
    frame = pd.read_parquet(path, columns=read_columns)
    frame = migrate_legacy_factor_columns(
        frame,
        context=f"right-side dataset boundary {path}",
        copy=False,
    )
    if requested is not None:
        frame = frame.loc[:, requested]
    if len(frame.columns) != len(set(frame.columns)):
        raise RuntimeError("canonical feature dataset contains duplicate columns")
    assert_no_forbidden_factor_names(
        frame.columns,
        context="right-side training dataframe",
    )
    return frame


class _StreamingParquetWriter:
    """Write homogeneous pandas batches and atomically publish on success."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self._writer: object | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temp_path,
                table.schema,
                compression="zstd",
            )
        else:
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)
        self.rows += len(frame)

    def close(self, *, commit: bool) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if commit:
            if self.rows <= 0 or not self.temp_path.exists():
                raise RuntimeError(f"streaming parquet writer produced no rows: {self.path}")
            os.replace(self.temp_path, self.path)
        elif self.temp_path.exists():
            self.temp_path.unlink()


class _CoverageAccumulator:
    """Accumulate factor coverage without retaining the full event table."""

    def __init__(self) -> None:
        self.total_rows: dict[str, int] = {"ALL": 0, **{signal: 0 for signal in RIGHT_SIDE_SIGNALS}}
        self.stats: dict[tuple[str, str], dict[str, object]] = {}

    def _update_values(self, scope: str, frame: pd.DataFrame, features: Sequence[str]) -> None:
        for feature in features:
            key = (scope, feature)
            stat = self.stats.setdefault(key, {"non_null": 0, "values": set()})
            if feature not in frame.columns:
                continue
            values = frame[feature]
            present = values[values.notna()]
            stat["non_null"] = int(stat["non_null"]) + int(len(present))
            unique_values = stat["values"]
            if len(unique_values) <= 1 and not present.empty:
                for value in pd.unique(present):
                    if isinstance(value, np.generic):
                        value = value.item()
                    unique_values.add(value)
                    if len(unique_values) > 1:
                        break

    def update(self, frame: pd.DataFrame) -> None:
        self.total_rows["ALL"] += len(frame)
        self._update_values("ALL", frame, (*PROJECT_FACTOR_COLUMNS, *RULE_FEATURE_COLUMNS))
        for signal, required in SIGNAL_FEATURE_REQUIREMENTS.items():
            subset = frame[frame[signal].fillna(False).astype(bool)]
            self.total_rows[signal] += len(subset)
            self._update_values(signal, subset, required)

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        scopes = [("ALL", (*PROJECT_FACTOR_COLUMNS, *RULE_FEATURE_COLUMNS)), *SIGNAL_FEATURE_REQUIREMENTS.items()]
        for scope, features in scopes:
            row_count = self.total_rows[scope]
            for feature in features:
                stat = self.stats.get((scope, feature), {"non_null": 0, "values": set()})
                non_null = int(stat["non_null"])
                unique = len(stat["values"])
                coverage = float(non_null / row_count) if row_count else 0.0
                if row_count == 0:
                    status = "empty_frame"
                elif non_null == 0:
                    status = "all_null"
                elif unique <= 1:
                    status = "constant"
                elif coverage < 0.80:
                    status = "sparse"
                else:
                    status = "ok"
                rows.append(
                    {
                        "signal": scope,
                        "signal_rows": row_count,
                        "feature": feature,
                        "non_null": non_null,
                        "coverage": coverage,
                        "unique": unique if unique <= 1 else 2,
                        "status": status,
                    }
                )
        return pd.DataFrame(rows)


def _daily_partition_paths(start_date: str, end_date: str, root: Path) -> list[Path]:
    start_month = pd.Timestamp(start_date).strftime("%Y%m")
    end_month = pd.Timestamp(end_date).strftime("%Y%m")
    return [
        path
        for path in sorted(root.glob("year_month=*/data.parquet"))
        if start_month <= path.parent.name.partition("=")[2] <= end_month
    ]


def _read_market(
    paths: Sequence[Path],
    *,
    symbols: set[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    columns = [
        "ts_code",
        "trade_date",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "volume",
        "name",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=[column for column in columns if column in available])
        if symbols is not None:
            frame = frame[frame["ts_code"].astype(str).isin(symbols)]
        if start_date is not None:
            frame = frame[frame["trade_date"].astype(str) >= pd.Timestamp(start_date).strftime("%Y%m%d")]
        if end_date is not None:
            frame = frame[frame["trade_date"].astype(str) <= pd.Timestamp(end_date).strftime("%Y%m%d")]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        ["ts_code", "trade_date"], keep="last"
    )


def _load_tradability(root: Path, start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    for path in sorted(root.glob("*.parquet")):
        if start <= path.stem <= end:
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _market_calendar(paths: Sequence[Path], start_date: str, end_date: str) -> pd.DatetimeIndex:
    values: list[pd.Series] = []
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    for path in paths:
        dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"].astype(str)
        values.append(dates[dates.between(start, end)])
    if not values:
        raise RuntimeError("daily partitions do not contain a market calendar")
    return pd.DatetimeIndex(
        pd.to_datetime(pd.concat(values).drop_duplicates(), format="%Y%m%d", errors="raise")
    ).sort_values()


def _process_symbol_features(
    task: tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        pd.DatetimeIndex,
        tuple[int, ...],
        tuple[str, ...],
    ],
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    (
        symbol,
        daily,
        family_rows,
        tradability_rows,
        start,
        end,
        calendar,
        horizons,
        entry_modes,
    ) = task
    try:
        normalized = normalize_tushare_daily(daily, symbol).sort_values("date").reset_index(drop=True)
        if normalized.empty:
            return pd.DataFrame(), pd.DataFrame(), None
        rules = compute_right_side_rule_features(normalized).reset_index(drop=True)
        z_flags = compute_canonical_z_signal_flags(
            normalized,
            rule_features=rules,
        )
        if family_rows.empty:
            family_rows = pd.DataFrame(columns=["symbol", "date"])
        signal_rows = merge_canonical_signal_flags(z_flags, family_rows)
        signal_rows = signal_rows[
            signal_rows["date"].between(start, end)
            & signal_rows[list(RIGHT_SIDE_SIGNALS)].any(axis=1)
        ].copy()
        if signal_rows.empty:
            return pd.DataFrame(), pd.DataFrame(), None
        for column in (*B2_SOURCE_COLUMNS, *B3_SOURCE_COLUMNS):
            if column not in family_rows.columns:
                signal_rows[column] = False
                continue
            source = family_rows[["symbol", "date", column]].copy()
            source["date"] = pd.to_datetime(source["date"], errors="coerce")
            source[column] = source[column].fillna(False).astype(bool)
            source = source.groupby(["symbol", "date"], as_index=False)[column].max()
            signal_rows = signal_rows.merge(
                source,
                on=["symbol", "date"],
                how="left",
                validate="one_to_one",
            )
            signal_rows[column] = signal_rows[column].fillna(False).astype(bool)
        signal_rows["signal_count"] = signal_rows[list(RIGHT_SIDE_SIGNALS)].sum(axis=1).astype("int16")
        signal_rows["has_right_signal"] = signal_rows[
            [signal for signal in RIGHT_SIDE_SIGNALS if signal not in {"GOLDEN_BOWL", "ZAIHOU", "BREATHING", "YUEYUE"}]
        ].any(axis=1)
        signal_rows["has_mixed_signal"] = signal_rows[
            ["GOLDEN_BOWL", "ZAIHOU", "BREATHING", "YUEYUE"]
        ].any(axis=1)

        project = calculate_project_market_factors(
            normalized,
            symbol=symbol,
            factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
        )
        if set(project["factor_schema_version"].dropna().astype(str)) != {
            PROJECT_FACTOR_SCHEMA_VERSION
        }:
            raise RuntimeError("project factor schema drifted from the causal v4 contract")
        project_columns = [column for column in PROJECT_FACTOR_COLUMNS if column in project.columns]
        base = project[["ts_code", "symbol", "trade_date", "date", *project_columns, "factor_schema_version"]].copy()
        rules = rules.reset_index(drop=True)
        base = pd.concat([base.reset_index(drop=True), rules], axis=1)
        merged = base.merge(signal_rows, on=["symbol", "date"], how="inner", validate="one_to_one")
        for column in PROJECT_FACTOR_COLUMNS:
            if column not in merged.columns:
                merged[column] = pd.Series(np.nan, index=merged.index, dtype="float32")
        factor_columns = [
            column
            for column in (*PROJECT_FACTOR_COLUMNS, *RULE_FEATURE_COLUMNS)
            if column in merged.columns
        ]
        for column in factor_columns:
            if pd.api.types.is_numeric_dtype(merged[column]) and not pd.api.types.is_bool_dtype(merged[column]):
                merged[column] = pd.to_numeric(merged[column], errors="coerce").astype("float32")
        event_columns = [
            "ts_code",
            "symbol",
            "trade_date",
            "date",
            *PROJECT_FACTOR_COLUMNS,
            "factor_schema_version",
            *RULE_FEATURE_COLUMNS,
            *RIGHT_SIDE_SIGNALS,
            *SUBVARIANT_COLUMNS,
            "signal_count",
            "has_right_signal",
            "has_mixed_signal",
        ]
        events = merged[event_columns].sort_values("date", kind="stable").reset_index(drop=True)
        assert_no_forbidden_factor_names(
            events.columns,
            context="new right-side event dataset",
        )

        label_daily = normalized[
            normalized["date"].between(calendar[0], calendar[-1])
        ].copy()
        labels = build_right_side_unified_labels(
            signal_rows,
            label_daily,
            calendar,
            tradability_rows if not tradability_rows.empty else None,
            horizons=horizons,
            entry_modes=entry_modes,
        )
        labels = labels[list(LABEL_OUTPUT_COLUMNS)].sort_values(
            ["date", "entry_mode", "horizon"], kind="stable"
        ).reset_index(drop=True)
        return events, labels, None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{symbol}: {exc}"


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.max_pending_per_worker <= 0:
        raise ValueError("max_pending_per_worker must be positive")

    family_columns = (
        "symbol",
        "date",
        *SUBVARIANT_COLUMNS,
        "signal_vegas_tunnel",
        "signal_tvb_merged",
    )
    available_family_columns = set(pq.ParquetFile(args.family_cache).schema.names)
    missing_family_columns = set(family_columns) - available_family_columns
    if missing_family_columns:
        raise RuntimeError(
            f"family cache missing Web signal contract columns: {sorted(missing_family_columns)}"
        )
    family = pd.read_parquet(args.family_cache, columns=list(family_columns))
    family["symbol"] = family["symbol"].astype(str)
    family["date"] = pd.to_datetime(family["date"], errors="coerce")
    family = family[
        family["date"].between(pd.Timestamp(args.start_date), pd.Timestamp(args.end_date))
    ].dropna(subset=["symbol", "date"])

    # Six years covers the longest current weekly/monthly project factors while
    # still computing every candidate strictly from information available by
    # its signal date.
    history_start = (
        pd.Timestamp(args.start_date) - pd.DateOffset(years=6)
    ).strftime("%Y-%m-%d")
    label_end = (pd.Timestamp(args.end_date) + pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    paths = _daily_partition_paths(history_start, label_end, args.daily_partitions)
    label_paths = _daily_partition_paths(args.start_date, label_end, args.daily_partitions)
    calendar = _market_calendar(label_paths, args.start_date, label_end)
    tradability = _load_tradability(args.tradability, args.start_date, label_end)
    market = _read_market(
        paths,
        start_date=history_start,
        end_date=label_end,
    )
    if market.empty:
        raise RuntimeError("no canonical daily market rows matched the research range")
    market["ts_code"] = market["ts_code"].astype(str)
    selected_symbols = sorted(market["ts_code"].unique())
    if args.symbol_limit:
        selected_symbols = selected_symbols[: args.symbol_limit]
        market = market[market["ts_code"].isin(selected_symbols)].copy()
        family = family[family["symbol"].isin(selected_symbols)].copy()
    symbols = set(selected_symbols)

    by_family = {
        str(symbol): group.reset_index(drop=True)
        for symbol, group in family.groupby("symbol", sort=False)
    }
    empty_family = pd.DataFrame(columns=list(family_columns))
    by_tradability = (
        {
            str(symbol): group.reset_index(drop=True)
            for symbol, group in tradability.groupby("ts_code", sort=False)
        }
        if not tradability.empty
        else {}
    )
    empty_tradability = pd.DataFrame()

    def iter_tasks() -> Iterator[tuple[object, ...]]:
        for symbol, group in market.groupby("ts_code", sort=True):
            symbol_text = str(symbol)
            yield (
                symbol_text,
                group.reset_index(drop=True),
                by_family.get(symbol_text, empty_family),
                by_tradability.get(symbol_text, empty_tradability),
                pd.Timestamp(args.start_date),
                pd.Timestamp(args.end_date),
                calendar,
                tuple(args.horizons),
                tuple(args.entry_modes),
            )

    event_writer = _StreamingParquetWriter(args.dataset_out)
    label_writer = _StreamingParquetWriter(args.labels_out)
    coverage = _CoverageAccumulator()
    errors: list[str] = []
    event_symbols = 0
    event_date_min: pd.Timestamp | None = None
    event_date_max: pd.Timestamp | None = None
    multi_hit_events = 0
    locked_limit_events = 0
    signal_counts: Counter[str] = Counter()
    maturity_reason: Counter[str] = Counter()
    locked_limit_source: Counter[str] = Counter()
    entry_modes: Counter[str] = Counter()
    horizons: Counter[int] = Counter()

    def consume(result: tuple[pd.DataFrame, pd.DataFrame, str | None]) -> None:
        nonlocal event_symbols, event_date_min, event_date_max, multi_hit_events, locked_limit_events
        events, labels, error = result
        if error is not None:
            errors.append(error)
            return
        if events.empty:
            if not labels.empty:
                errors.append("worker returned labels without events")
            return
        if events.duplicated(["symbol", "date"]).any():
            errors.append(f"duplicate events from worker: {events['symbol'].iloc[0]}")
            return
        if labels.duplicated(["symbol", "date", "entry_mode", "horizon"]).any():
            errors.append(f"duplicate labels from worker: {events['symbol'].iloc[0]}")
            return
        validate_signal_factor_contract(events.columns)
        coverage.update(events)
        event_writer.write(events)
        label_writer.write(labels)
        event_symbols += 1
        current_min = pd.Timestamp(events["date"].min())
        current_max = pd.Timestamp(events["date"].max())
        event_date_min = current_min if event_date_min is None else min(event_date_min, current_min)
        event_date_max = current_max if event_date_max is None else max(event_date_max, current_max)
        multi_hit_events += int(events["signal_count"].gt(1).sum())
        for signal in RIGHT_SIDE_SIGNALS:
            signal_counts[signal] += int(events[signal].sum())
        locked_limit_events += int(
            labels.drop_duplicates(["symbol", "date"])["locked_limit_up"].sum()
        )
        maturity_reason.update(labels["maturity_reason"].astype(str).value_counts().to_dict())
        locked_limit_source.update(labels["locked_limit_source"].astype(str).value_counts().to_dict())
        entry_modes.update(labels["entry_mode"].astype(str).value_counts().to_dict())
        horizons.update(labels["horizon"].astype(int).value_counts().to_dict())

    task_iterator = iter(iter_tasks())
    completed = 0
    total_tasks = len(selected_symbols)
    try:
        if args.workers <= 1:
            for task in task_iterator:
                consume(_process_symbol_features(task))
                completed += 1
                if completed % 250 == 0:
                    print(f"features {completed:,}/{total_tasks:,}", flush=True)
        else:
            max_pending = max(args.workers, args.workers * args.max_pending_per_worker)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pending = set()
                for _ in range(min(max_pending, total_tasks)):
                    try:
                        pending.add(executor.submit(_process_symbol_features, next(task_iterator)))
                    except StopIteration:
                        break
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            consume(future.result())
                        except Exception as exc:
                            errors.append(f"worker process failure: {exc}")
                        completed += 1
                        try:
                            pending.add(executor.submit(_process_symbol_features, next(task_iterator)))
                        except StopIteration:
                            pass
                        if completed % 250 == 0:
                            print(f"features {completed:,}/{total_tasks:,}", flush=True)
        if errors:
            raise RuntimeError(
                f"feature/label build failed for {len(errors)} symbols; first errors={errors[:5]}"
            )
        event_writer.close(commit=True)
        label_writer.close(commit=True)
    except Exception:
        event_writer.close(commit=False)
        label_writer.close(commit=False)
        raise

    del market, family, by_family, by_tradability
    gc.collect()
    audit = coverage.to_frame()
    atomic_write_csv(audit, args.factor_audit_out, index=False)
    overall_audit = audit[audit["signal"].eq("ALL")]
    materialized_project_count = int(
        overall_audit[
            overall_audit["feature"].isin(PROJECT_FACTOR_COLUMNS)
            & ~overall_audit["status"].isin(["missing", "all_null"])
        ]["feature"].nunique()
    )
    sample_audit = {
        "rows": int(label_writer.rows),
        "unique_events": int(event_writer.rows),
        "unique_symbols": int(event_symbols),
        "date_min": event_date_min,
        "date_max": event_date_max,
        "multi_hit_events": int(multi_hit_events),
        "locked_limit_events": int(locked_limit_events),
        "locked_limit_rows": int(
            sum(count for reason, count in maturity_reason.items() if reason == "locked_limit_up")
        ),
        "mature_rows": int(maturity_reason.get("mature", 0)),
        "maturity_reason": dict(maturity_reason),
        "locked_limit_source": dict(locked_limit_source),
        "entry_modes": dict(entry_modes),
        "horizons": {str(key): value for key, value in horizons.items()},
        "signal_counts": {signal: int(signal_counts[signal]) for signal in RIGHT_SIDE_SIGNALS},
    }
    atomic_write_json(sample_audit, args.sample_audit_out)
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "right-side-unified-dataset-v4-rule113-canonical-current",
        "signal_schema_version": CANONICAL_SIGNAL_SCHEMA_VERSION,
        "signal_contract_notes": dict(SIGNAL_CONTRACT_NOTES),
        "factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "feature_schema_version": RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
        "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
        "model_input_contract_sha256": RIGHT_SIDE_SHADOW_MODEL_INPUT_CONTRACT_SHA256,
        "forbidden_aliases": [],
        "daily_basic_included": False,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "signals": list(RIGHT_SIDE_SIGNALS),
        "project_factor_contract_count": len(PROJECT_FACTOR_COLUMNS),
        "project_factor_columns": list(PROJECT_FACTOR_COLUMNS),
        "project_factor_materialized_count": materialized_project_count,
        "rule_feature_count": len(RULE_FEATURE_COLUMNS),
        "rule_feature_schema_version": RULE_FEATURE_SCHEMA_VERSION,
        "rule_feature_columns": list(RULE_FEATURE_COLUMNS),
        "rule_feature_columns_sha256": RULE_FEATURE_COLUMNS_SHA256,
        "event_dataset": str(args.dataset_out),
        "label_dataset": str(args.labels_out),
        "sample_audit": sample_audit,
    }
    atomic_write_json(manifest, args.manifest_out)
    return manifest


def audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    events = _read_canonical_feature_frame(args.dataset)
    labels = pd.read_parquet(args.labels)
    validate_signal_factor_contract(events.columns)
    required_events = {
        "symbol",
        "date",
        *RIGHT_SIDE_SIGNALS,
    }
    required_labels = {
        "symbol",
        "date",
        "entry_mode",
        "horizon",
        "entry_date",
        "label_end_date",
        "mature",
        "locked_limit_up",
        *LABEL_COLUMNS,
    }
    missing_events = required_events - set(events.columns)
    missing_labels = required_labels - set(labels.columns)
    if missing_events or missing_labels:
        raise RuntimeError(
            "dataset audit failed; "
            f"missing_event_columns={sorted(missing_events)} "
            f"missing_label_columns={sorted(missing_labels)}"
        )
    duplicated_events = int(events.duplicated(["symbol", "date"]).sum())
    duplicated_labels = int(labels.duplicated(["symbol", "date", "entry_mode", "horizon"]).sum())
    impossible = int((labels["mature"] & labels["label_end_date"].isna()).sum())
    mature_with_missing_label = int(
        (
            labels["mature"]
            & labels[list(LABEL_COLUMNS)].isna().any(axis=1)
        ).sum()
    )
    mislabeled_tail = int((~labels["mature"] & labels[list(LABEL_COLUMNS)].notna().any(axis=1)).sum())
    event_level = events
    signal_event_rows = {
        signal: int(event_level[signal].fillna(False).astype(bool).sum())
        for signal in RIGHT_SIDE_SIGNALS
    }
    signals_without_events = [
        signal for signal, row_count in signal_event_rows.items() if row_count == 0
    ]
    factor_contract_failures: dict[str, list[str]] = {}
    for signal, required_features in SIGNAL_FEATURE_REQUIREMENTS.items():
        subset = event_level[event_level[signal].fillna(False).astype(bool)]
        coverage = audit_factor_coverage(subset, required_features)
        failed = coverage.loc[
            coverage["status"].isin(["missing", "all_null"]),
            "feature",
        ].astype(str).tolist()
        if failed:
            factor_contract_failures[signal] = failed
    summary = {
        "event_rows": int(len(events)),
        "label_rows": int(len(labels)),
        "duplicate_events": duplicated_events,
        "duplicate_labels": duplicated_labels,
        "mature_without_label_end": impossible,
        "mature_with_missing_label": mature_with_missing_label,
        "immature_with_any_label": mislabeled_tail,
        "locked_limit_in_mature": int((labels["locked_limit_up"] & labels["mature"]).sum()),
        "signal_event_rows": signal_event_rows,
        "signals_without_events": signals_without_events,
        "factor_contract_complete": not signals_without_events,
        "signal_factor_failures": factor_contract_failures,
    }
    if factor_contract_failures:
        summary["factor_contract_complete"] = False
    if any(
        summary[key]
        for key in (
            "duplicate_events",
            "duplicate_labels",
            "mature_without_label_end",
            "mature_with_missing_label",
            "immature_with_any_label",
            "locked_limit_in_mature",
        )
    ) or signals_without_events or factor_contract_failures:
        raise RuntimeError(f"dataset audit failed: {summary}")
    return summary


def _fit_base_classifier(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    sample_weight: pd.Series | None,
    n_jobs: int,
    classifier_spec: XGBClassifierSpec = DEFAULT_XGB_CLASSIFIER_SPEC,
) -> XGBResearchModel:
    imputer = Float32NaNTransformer()
    # Float32NaNTransformer already normalizes non-finite values.  Avoid an
    # additional full-size DataFrame copy here: the pooled research sample can
    # contain hundreds of features and more than a million rows.
    train_x = imputer.fit_transform(train[list(features)])
    early_stop_x = imputer.transform(early_stop[list(features)])
    y_train = train[label].astype(int)
    classifier = XGBClassifier(**classifier_spec.classifier_kwargs(n_jobs=n_jobs))
    classifier.fit(
        train_x,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(early_stop_x, early_stop[label].astype(int))],
        verbose=False,
    )
    best_iteration = int(classifier.best_iteration) if classifier.best_iteration is not None else None
    return XGBResearchModel(
        feature_names_in_=list(features),
        selected_features_=list(features),
        imputer=imputer,
        selector=None,
        classifier=classifier,
        best_iteration=best_iteration,
        factor_schema_version_=PROJECT_FACTOR_SCHEMA_VERSION,
    )


def _predict_xgb_raw_margin(model: XGBResearchModel, values: pd.DataFrame) -> np.ndarray:
    """Return the persisted XGBoost model's raw margin deterministically."""

    transformed = model.transform(values)
    iteration_range = (
        None
        if model.best_iteration is None
        else (0, int(model.best_iteration) + 1)
    )
    kwargs: dict[str, Any] = {"output_margin": True}
    if iteration_range is not None:
        kwargs["iteration_range"] = iteration_range
    margin = np.asarray(model.classifier.predict(transformed, **kwargs), dtype=float)
    if margin.ndim != 1 or len(margin) != len(values) or not np.isfinite(margin).all():
        raise RuntimeError("beam XGBoost raw margin contract failed")
    return margin


def _fit_residual_base_classifier(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    train_base_margin: Sequence[float],
    early_stop_base_margin: Sequence[float],
    n_jobs: int,
    classifier_spec: XGBClassifierSpec,
) -> XGBResearchModel:
    """Fit R1 with the matched R0 margin as XGBoost ``base_margin``.

    The fitted booster predicts the total R1 margin when the same R0 margin is
    supplied at inference.  Subtracting that margin from R0 yields the learned
    incremental component used by Beam Residual v3.
    """

    imputer = Float32NaNTransformer()
    train_x = imputer.fit_transform(train[list(features)])
    early_stop_x = imputer.transform(early_stop[list(features)])
    train_margin = np.asarray(train_base_margin, dtype=float)
    early_margin = np.asarray(early_stop_base_margin, dtype=float)
    if (
        train_margin.shape != (len(train),)
        or early_margin.shape != (len(early_stop),)
        or not np.isfinite(train_margin).all()
        or not np.isfinite(early_margin).all()
    ):
        raise ValueError("beam residual base margins must be finite and row-aligned")
    classifier = XGBClassifier(**classifier_spec.classifier_kwargs(n_jobs=n_jobs))
    classifier.fit(
        train_x,
        train[label].astype(int),
        base_margin=train_margin,
        eval_set=[(early_stop_x, early_stop[label].astype(int))],
        base_margin_eval_set=[early_margin],
        verbose=False,
    )
    best_iteration = (
        int(classifier.best_iteration)
        if classifier.best_iteration is not None
        else None
    )
    return XGBResearchModel(
        feature_names_in_=list(features),
        selected_features_=list(features),
        imputer=imputer,
        selector=None,
        classifier=classifier,
        best_iteration=best_iteration,
        factor_schema_version_=PROJECT_FACTOR_SCHEMA_VERSION,
    )


def _predict_residual_raw_margin(
    model: XGBResearchModel,
    values: pd.DataFrame,
    *,
    base_margin: Sequence[float],
) -> np.ndarray:
    transformed = model.transform(values)
    margin = np.asarray(base_margin, dtype=float)
    if margin.shape != (len(values),) or not np.isfinite(margin).all():
        raise ValueError("beam residual inference margin must be finite and row-aligned")
    kwargs: dict[str, Any] = {
        "output_margin": True,
        "base_margin": margin,
    }
    if model.best_iteration is not None:
        kwargs["iteration_range"] = (0, int(model.best_iteration) + 1)
    output = np.asarray(model.classifier.predict(transformed, **kwargs), dtype=float)
    if output.shape != margin.shape or not np.isfinite(output).all():
        raise RuntimeError("beam residual candidate margin contract failed")
    return output


def _fit_probability_calibrator(
    labels: Sequence[int | bool],
    probabilities: Sequence[float],
) -> LogisticRegression:
    validation_raw = np.asarray(probabilities, dtype=float)
    if not np.isfinite(validation_raw).all():
        raise ValueError("calibration probabilities must be finite")
    validation_raw = np.clip(validation_raw, 1e-6, 1.0 - 1e-6)
    validation_logit = np.log(validation_raw / (1.0 - validation_raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(validation_logit, np.asarray(labels, dtype=int))
    return calibrator


def _fit_classifier(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    sample_weight: pd.Series | None,
    n_jobs: int,
) -> ProbabilityCalibratedModel:
    base_model = _fit_base_classifier(
        train,
        early_stop,
        features,
        label,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )
    validation_raw = np.asarray(base_model.predict_proba(calibration), dtype=float)[:, 1]
    calibrator = _fit_probability_calibrator(
        calibration[label].astype(int),
        validation_raw,
    )
    return ProbabilityCalibratedModel(base_model=base_model, calibrator=calibrator)


def _fit_long_task_classifier(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    common_features: Sequence[str],
    label: str,
    *,
    n_jobs: int,
    task_weighting: str,
    classifier_spec: XGBClassifierSpec,
) -> tuple[LongTaskUnifiedModel, int, dict[str, float | str]]:
    retained = [*common_features, label]
    train_long = expand_long_task_rows(train, retained_columns=retained)
    early_stop_long = expand_long_task_rows(early_stop, retained_columns=retained)
    calibration_long = expand_long_task_rows(
        calibration,
        retained_columns=common_features,
    )
    if train_long.empty or early_stop_long.empty or calibration_long.empty:
        raise ValueError("long-task split contains no active task rows")
    if task_weighting == "one_vote":
        sample_weight = None
        weight_metadata: dict[str, float | str] = {
            "task_weighting": task_weighting,
            "sample_weight_min": 1.0,
            "sample_weight_mean": 1.0,
            "sample_weight_max": 1.0,
        }
    elif task_weighting == "inverse_sqrt_task_frequency_clip_0p5_3p0":
        sample_weight = inverse_sqrt_task_weights(train_long)
        weight_metadata = {
            "task_weighting": task_weighting,
            "sample_weight_min": float(sample_weight.min()),
            "sample_weight_mean": float(sample_weight.mean()),
            "sample_weight_max": float(sample_weight.max()),
        }
    else:
        raise ValueError(f"unknown long-task weighting: {task_weighting}")
    task_features = [*common_features, *LONG_TASK_FEATURE_COLUMNS]
    base_model = _fit_base_classifier(
        train_long,
        early_stop_long,
        task_features,
        label,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
        classifier_spec=classifier_spec,
    )
    task_probability = np.asarray(
        base_model.predict_proba(calibration_long), dtype=float
    )[:, 1]
    event_probability = aggregate_long_task_predictions(
        calibration_long,
        task_probability,
        event_count=len(calibration),
    )
    calibrator = _fit_probability_calibrator(
        calibration[label].astype(int),
        event_probability,
    )
    return (
        LongTaskUnifiedModel(
            base_model=base_model,
            event_calibrator=calibrator,
            common_features=tuple(common_features),
        ),
        len(train_long),
        weight_metadata,
    )


def _expand_long_task_for_beam(
    events: pd.DataFrame,
    *,
    common_features: Sequence[str],
    label: str | None,
) -> pd.DataFrame:
    retained = [*common_features]
    if label is not None:
        retained.append(label)
    expanded = expand_long_task_rows(events, retained_columns=retained)
    if expanded.empty:
        raise ValueError("beam split contains no active task rows")
    return expanded


def _aggregate_long_task_raw_margin(
    expanded: pd.DataFrame,
    task_margins: Sequence[float],
    *,
    event_count: int,
) -> np.ndarray:
    return aggregate_long_task_predictions(
        expanded,
        task_margins,
        event_count=event_count,
    )


class _BeamControlFit:
    """Internal fitted R0 bundle (plain class for dynamic-script imports)."""

    def __init__(
        self,
        *,
        model: XGBResearchModel,
        calibrator: LogisticRegression,
        common_features: tuple[str, ...],
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.common_features = common_features


def _fit_beam_control(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    common_features: Sequence[str],
    label: str,
    *,
    n_jobs: int,
    classifier_spec: XGBClassifierSpec,
) -> _BeamControlFit:
    train_long = _expand_long_task_for_beam(
        train, common_features=common_features, label=label
    )
    early_long = _expand_long_task_for_beam(
        early_stop, common_features=common_features, label=label
    )
    calibration_long = _expand_long_task_for_beam(
        calibration, common_features=common_features, label=None
    )
    task_features = [*common_features, *LONG_TASK_FEATURE_COLUMNS]
    model = _fit_base_classifier(
        train_long,
        early_long,
        task_features,
        label,
        sample_weight=None,
        n_jobs=n_jobs,
        classifier_spec=classifier_spec,
    )
    calibration_probability = np.asarray(
        model.predict_proba(calibration_long), dtype=float
    )[:, 1]
    calibration_event = aggregate_long_task_predictions(
        calibration_long,
        calibration_probability,
        event_count=len(calibration),
    )
    calibrator = _fit_probability_calibrator(
        calibration[label].astype(int), calibration_event
    )
    return _BeamControlFit(
        model=model,
        calibrator=calibrator,
        common_features=tuple(common_features),
    )


def _beam_control_predictions(
    fit: _BeamControlFit,
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    expanded = _expand_long_task_for_beam(
        events, common_features=fit.common_features, label=None
    )
    task_probability = np.asarray(fit.model.predict_proba(expanded), dtype=float)[:, 1]
    event_probability = aggregate_long_task_predictions(
        expanded,
        task_probability,
        event_count=len(events),
    )
    calibrated = apply_event_calibrator(event_probability, fit.calibrator)
    task_margin = _predict_xgb_raw_margin(fit.model, expanded)
    event_margin = _aggregate_long_task_raw_margin(
        expanded, task_margin, event_count=len(events)
    )
    if not np.isfinite(calibrated).all() or not np.isfinite(event_margin).all():
        raise RuntimeError("beam control produced non-finite event scores")
    return calibrated, event_margin, expanded, task_margin


def _fit_beam_residual_candidate(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    control: _BeamControlFit,
    candidate_common_features: Sequence[str],
    label: str,
    *,
    n_jobs: int,
    classifier_spec: XGBClassifierSpec,
    train_control_base_margin: Sequence[float] | None = None,
    early_stop_control_base_margin: Sequence[float] | None = None,
) -> XGBResearchModel:
    if train_control_base_margin is None:
        train_control_long = _expand_long_task_for_beam(
            train, common_features=control.common_features, label=None
        )
        train_margin = _predict_xgb_raw_margin(control.model, train_control_long)
    else:
        train_margin = np.asarray(train_control_base_margin, dtype=float)
    if early_stop_control_base_margin is None:
        early_control_long = _expand_long_task_for_beam(
            early_stop, common_features=control.common_features, label=None
        )
        early_margin = _predict_xgb_raw_margin(control.model, early_control_long)
    else:
        early_margin = np.asarray(early_stop_control_base_margin, dtype=float)
    train_candidate_long = _expand_long_task_for_beam(
        train, common_features=candidate_common_features, label=label
    )
    early_candidate_long = _expand_long_task_for_beam(
        early_stop, common_features=candidate_common_features, label=label
    )
    return _fit_residual_base_classifier(
        train_candidate_long,
        early_candidate_long,
        [*candidate_common_features, *LONG_TASK_FEATURE_COLUMNS],
        label,
        train_base_margin=train_margin,
        early_stop_base_margin=early_margin,
        n_jobs=n_jobs,
        classifier_spec=classifier_spec,
    )


def _beam_residual_event_components(
    candidate: XGBResearchModel,
    control: _BeamControlFit,
    events: pd.DataFrame,
    *,
    candidate_common_features: Sequence[str],
    precomputed_control: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ) = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if precomputed_control is None:
        (
            baseline_probability,
            baseline_margin,
            control_long,
            row_base_margin,
        ) = _beam_control_predictions(control, events)
        control_task_identity = control_long[
            list(LONG_TASK_FEATURE_COLUMNS)
        ].to_numpy(dtype=bool)
    else:
        (
            baseline_probability,
            baseline_margin,
            row_base_margin,
            control_task_identity,
        ) = precomputed_control
    candidate_long = _expand_long_task_for_beam(
        events, common_features=candidate_common_features, label=None
    )
    candidate_task_identity = candidate_long[list(LONG_TASK_FEATURE_COLUMNS)].to_numpy(
        dtype=bool
    )
    if not np.array_equal(control_task_identity, candidate_task_identity):
        raise RuntimeError("beam R0/R1 task-row alignment contract failed")
    candidate_task_margin = _predict_residual_raw_margin(
        candidate,
        candidate_long,
        base_margin=row_base_margin,
    )
    candidate_event_margin = _aggregate_long_task_raw_margin(
        candidate_long,
        candidate_task_margin,
        event_count=len(events),
    )
    return baseline_probability, baseline_margin, candidate_event_margin


def _beam_residual_event_probability(
    candidate: XGBResearchModel,
    control: _BeamControlFit,
    events: pd.DataFrame,
    *,
    candidate_common_features: Sequence[str],
    reliability: float = 1.0,
    precomputed_control: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    ) = None,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_probability, baseline_margin, candidate_event_margin = (
        _beam_residual_event_components(
            candidate,
            control,
            events,
            candidate_common_features=candidate_common_features,
            precomputed_control=precomputed_control,
        )
    )
    combined = combine_residual_probabilities(
        baseline_probability,
        baseline_margin,
        candidate_event_margin,
        reliability=reliability,
    )
    return baseline_probability, combined


def _split_validation_stages(
    validation: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split validation chronologically into early-stop/calibration/threshold stages."""

    ordered_dates = pd.DatetimeIndex(pd.to_datetime(validation["date"]).unique()).sort_values()
    if len(ordered_dates) < 60:
        raise ValueError("validation needs at least 60 trade dates for three-stage model selection")
    first = ordered_dates[max(1, int(len(ordered_dates) * 0.50)) - 1]
    second = ordered_dates[max(2, int(len(ordered_dates) * 0.75)) - 1]
    early_stop = validation[pd.to_datetime(validation["date"]) <= first].copy()
    calibration = validation[
        pd.to_datetime(validation["date"]).gt(first)
        & pd.to_datetime(validation["date"]).le(second)
    ].copy()
    threshold = validation[pd.to_datetime(validation["date"]).gt(second)].copy()
    for name, frame in (
        ("early_stop", early_stop),
        ("calibration", calibration),
        ("threshold", threshold),
    ):
        if frame.empty or frame[label].nunique() < 2:
            raise ValueError(f"{name} validation stage lacks both label classes")
    return early_stop, calibration, threshold


def _valid_local_model(frame: pd.DataFrame, label: str, minimum_rows: int) -> bool:
    if len(frame) < minimum_rows or frame[label].nunique() < 2:
        return False
    counts = frame[label].value_counts()
    return int(counts.min()) >= min(200, max(20, minimum_rows // 10)) and frame["date"].nunique() >= 120


def _split_beam_history_stages(
    history: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the most recent history dates for search fit controls.

    These stages are strictly before the rolling evaluation block.  Candidate
    selection therefore sees labels from development windows only and cannot
    indirectly early-stop on the outer test year.
    """

    ordered_dates = pd.DatetimeIndex(
        pd.to_datetime(history["date"]).unique()
    ).sort_values()
    if len(ordered_dates) < 120:
        raise ValueError("beam history needs at least 120 trade dates")
    train_end = ordered_dates[max(1, int(len(ordered_dates) * 0.80)) - 1]
    early_end = ordered_dates[max(2, int(len(ordered_dates) * 0.90)) - 1]
    train = history[pd.to_datetime(history["date"]).le(train_end)].copy()
    early_stop = history[
        pd.to_datetime(history["date"]).gt(train_end)
        & pd.to_datetime(history["date"]).le(early_end)
    ].copy()
    calibration = history[pd.to_datetime(history["date"]).gt(early_end)].copy()
    for stage, frame in (
        ("train", train),
        ("early_stop", early_stop),
        ("calibration", calibration),
    ):
        if frame.empty or frame[label].nunique() < 2:
            raise ValueError(f"beam {stage} stage lacks both target classes")
    return train, early_stop, calibration


def _beam_evaluation_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    label: str,
    history_max_rows: int,
    evaluation_max_rows: int,
) -> tuple[
    tuple[
        tuple[
            object,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            dict[str, Any],
        ],
        ...,
    ],
    tuple[
        object,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
    ],
]:
    """Materialize search folds plus an unseen ``pipeline_select`` tail."""

    windows, pipeline_window = build_validation_search_and_pipeline_select(
        pd.to_datetime(train["date"]).unique(),
        pd.to_datetime(validation["date"]).unique(),
    )
    development = pd.concat([train, validation], ignore_index=True, sort=False)
    dates = pd.to_datetime(development["date"])
    label_ends = pd.to_datetime(development["label_end_date"])
    output: list[
        tuple[
            object,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            dict[str, Any],
        ]
    ] = []

    def materialize(
        window: object,
    ) -> tuple[
        object,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
    ]:
        evaluation_start = min(window.evaluation_dates)
        history_mask = dates.isin(window.history_dates) & label_ends.lt(evaluation_start)
        evaluation_mask = dates.isin(window.evaluation_dates)
        history = development.loc[history_mask].copy()
        evaluation = development.loc[evaluation_mask].copy()
        if history.empty or evaluation.empty or evaluation[label].nunique() < 2:
            raise ValueError(f"beam window {window.name} is not evaluable")
        history, history_sampling = deterministic_stratified_event_sample(
            history,
            maximum_rows=history_max_rows,
            label_column=label,
            signal_columns=RIGHT_SIDE_SIGNALS,
            random_seed=42,
        )
        evaluation, evaluation_sampling = deterministic_stratified_event_sample(
            evaluation,
            maximum_rows=evaluation_max_rows,
            label_column=label,
            signal_columns=RIGHT_SIDE_SIGNALS,
            random_seed=42,
        )
        model_train, early_stop, calibration = _split_beam_history_stages(
            history, label
        )
        return (
            window,
            model_train,
            early_stop,
            calibration,
            evaluation,
            {
                "history": history_sampling,
                "evaluation": evaluation_sampling,
            },
        )

    for window in windows:
        output.append(materialize(window))
    return tuple(output), materialize(pipeline_window)


def _run_beam_development_search(
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    admitted_project_features: Sequence[str],
    label: str,
    outer_fold: str,
    n_jobs: int,
    settings: BeamSettings,
    search_classifier_spec: XGBClassifierSpec,
    report_root: Path,
    history_max_rows: int,
    evaluation_max_rows: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Select a 105+increment subset using only an outer fold's dev data."""

    candidate_features = tuple(ADDED_RULE_FEATURE_COLUMNS_V2)
    context_features = [
        *admitted_project_features,
        *LEGACY_RULE_FEATURE_COLUMNS_V1,
    ]
    frame_specs, pipeline_spec = _beam_evaluation_frames(
        train,
        validation,
        label=label,
        history_max_rows=history_max_rows,
        evaluation_max_rows=evaluation_max_rows,
    )
    print(
        json.dumps(
            {
                "event": "beam_v3_search_start",
                "outer_fold": outer_fold,
                "candidate_count": len(candidate_features),
                "rolling_windows": [item[0].name for item in frame_specs],
                "pipeline_window": pipeline_spec[0].name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    control_fits: dict[str, _BeamControlFit] = {}
    control_fit_margins: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    control_probabilities: dict[str, np.ndarray] = {}
    control_evaluation_components: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    evaluation_labels: dict[str, np.ndarray] = {}
    for window, model_train, early_stop, calibration, evaluation, _ in frame_specs:
        print(
            json.dumps(
                {
                    "event": "beam_v3_control_fit_start",
                    "outer_fold": outer_fold,
                    "window": window.name,
                    "train_rows": len(model_train),
                    "early_stop_rows": len(early_stop),
                    "calibration_rows": len(calibration),
                    "evaluation_rows": len(evaluation),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        control = _fit_beam_control(
            model_train,
            early_stop,
            calibration,
            context_features,
            label,
            n_jobs=n_jobs,
            classifier_spec=search_classifier_spec,
        )
        control_fits[window.name] = control
        train_control_long = _expand_long_task_for_beam(
            model_train,
            common_features=control.common_features,
            label=None,
        )
        early_control_long = _expand_long_task_for_beam(
            early_stop,
            common_features=control.common_features,
            label=None,
        )
        control_fit_margins[window.name] = (
            _predict_xgb_raw_margin(control.model, train_control_long),
            _predict_xgb_raw_margin(control.model, early_control_long),
        )
        (
            baseline_probability,
            baseline_event_margin,
            evaluation_control_long,
            evaluation_task_margin,
        ) = _beam_control_predictions(control, evaluation)
        control_probabilities[window.name] = baseline_probability
        control_evaluation_components[window.name] = (
            baseline_probability,
            baseline_event_margin,
            evaluation_task_margin,
            evaluation_control_long[list(LONG_TASK_FEATURE_COLUMNS)].to_numpy(
                dtype=bool
            ),
        )
        evaluation_labels[window.name] = (
            evaluation[label].astype(int).to_numpy()
        )
        print(
            json.dumps(
                {
                    "event": "beam_v3_control_fit_done",
                    "outer_fold": outer_fold,
                    "window": window.name,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del train_control_long, early_control_long, evaluation_control_long
        gc.collect()

    evaluation_cache: dict[tuple[str, ...], BeamEvaluation] = {}

    def evaluate_selected(selected: Sequence[str]) -> BeamEvaluation:
        selected_tuple = tuple(
            feature for feature in candidate_features if feature in set(selected)
        )
        if selected_tuple in evaluation_cache:
            return evaluation_cache[selected_tuple]
        selected_set = set(selected_tuple)
        removed = tuple(sorted(set(candidate_features) - selected_set))
        fold_lifts = []
        fold_predictions: list[dict[str, Any]] = []
        for window, model_train, early_stop, calibration, evaluation, _ in frame_specs:
            del calibration
            control = control_fits[window.name]
            candidate_common_features = [*context_features, *selected_tuple]
            candidate_model = _fit_beam_residual_candidate(
                model_train,
                early_stop,
                control,
                candidate_common_features,
                label,
                n_jobs=n_jobs,
                classifier_spec=search_classifier_spec,
                train_control_base_margin=control_fit_margins[window.name][0],
                early_stop_control_base_margin=control_fit_margins[window.name][1],
            )
            _, candidate_probability = (
                _beam_residual_event_probability(
                    candidate_model,
                    control,
                    evaluation,
                    candidate_common_features=candidate_common_features,
                    precomputed_control=control_evaluation_components[
                        window.name
                    ],
                )
            )
            baseline_probability = control_probabilities[window.name]
            labels = evaluation_labels[window.name]
            fold_lifts.append(
                ranking_lift(
                    labels,
                    baseline_probability,
                    candidate_probability,
                    fold=window.name,
                )
            )
            fold_predictions.append(
                {
                    "fold": window.name,
                    "labels": labels,
                    "baseline_probability": baseline_probability,
                    "candidate_probability": candidate_probability,
                }
            )
            del candidate_model
            gc.collect()
        candidate_score = score_beam_candidate(
            fold_lifts,
            n_features=len(selected_tuple),
            settings=settings,
        )
        result = BeamEvaluation(
            removed=removed,
            selected=selected_tuple,
            candidate_score=candidate_score,
            fold_lifts=tuple(fold_lifts),
            fold_predictions=tuple(fold_predictions),
        )
        evaluation_cache[selected_tuple] = result
        visited_count = len(evaluation_cache)
        if visited_count == 1 or visited_count % 10 == 0:
            print(
                json.dumps(
                    {
                        "event": "beam_v3_candidate_progress",
                        "outer_fold": outer_fold,
                        "visited_combinations": visited_count,
                        "selected_increment_count": len(selected_tuple),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return result

    def evaluate_removed(removed: tuple[str, ...]) -> BeamEvaluation:
        selected = tuple(feature for feature in candidate_features if feature not in removed)
        source = evaluate_selected(selected)
        return BeamEvaluation(
            removed=removed,
            selected=selected,
            candidate_score=source.candidate_score,
            fold_lifts=source.fold_lifts,
            fold_predictions=source.fold_predictions,
        )

    result = backward_beam_search(
        candidate_features=candidate_features,
        evaluate=evaluate_removed,
        settings=settings,
    )
    all_evaluations = result.evaluations
    permutation = visited_candidate_permutation_test(
        all_evaluations,
        settings=settings,
        observed_best_score=result.best.score,
        random_seed=42,
    )
    permutation_gate_passed = bool(
        permutation["empirical_p_value"]
        <= settings.maximum_permutation_p_value
    )
    print(
        json.dumps(
            {
                "event": "beam_v3_search_frozen",
                "outer_fold": outer_fold,
                "visited_combinations": len(all_evaluations),
                "selected_increment_count": len(result.best.selected),
                "permutation_p_value": permutation["empirical_p_value"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    # Top1 is now frozen.  Only at this point may it see the disjoint
    # pipeline_select tail; the outer test year remains structurally absent.
    (
        pipeline_window,
        pipeline_train,
        pipeline_early_stop,
        pipeline_calibration,
        pipeline_evaluation,
        pipeline_sampling,
    ) = pipeline_spec
    pipeline_control = _fit_beam_control(
        pipeline_train,
        pipeline_early_stop,
        pipeline_calibration,
        context_features,
        label,
        n_jobs=n_jobs,
        classifier_spec=search_classifier_spec,
    )
    pipeline_candidate_features = [*context_features, *result.best.selected]
    pipeline_candidate = _fit_beam_residual_candidate(
        pipeline_train,
        pipeline_early_stop,
        pipeline_control,
        pipeline_candidate_features,
        label,
        n_jobs=n_jobs,
        classifier_spec=search_classifier_spec,
    )
    (
        pipeline_baseline_probability,
        pipeline_baseline_margin,
        pipeline_candidate_margin,
    ) = _beam_residual_event_components(
        pipeline_candidate,
        pipeline_control,
        pipeline_evaluation,
        candidate_common_features=pipeline_candidate_features,
    )
    pipeline_soft_probability = combine_residual_probabilities(
        pipeline_baseline_probability,
        pipeline_baseline_margin,
        pipeline_candidate_margin,
        reliability=0.5,
    )
    pipeline_no_gate_probability = combine_residual_probabilities(
        pipeline_baseline_probability,
        pipeline_baseline_margin,
        pipeline_candidate_margin,
        reliability=1.0,
    )
    pipeline_decision = evaluate_pipeline_select_variants(
        pipeline_evaluation[label].astype(int).to_numpy(),
        pipeline_baseline_probability,
        soft_gate_probability=pipeline_soft_probability,
        no_gate_probability=pipeline_no_gate_probability,
    )
    pipeline_decision.update(
        {
            "status": "evaluated_after_top1_freeze",
            "window": pipeline_window.name,
            "history_date_min": str(min(pipeline_window.history_dates).date()),
            "history_date_max": str(max(pipeline_window.history_dates).date()),
            "evaluation_date_min": str(
                min(pipeline_window.evaluation_dates).date()
            ),
            "evaluation_date_max": str(
                max(pipeline_window.evaluation_dates).date()
            ),
            "sampling": pipeline_sampling,
            "test_data_used": False,
        }
    )
    pipeline_select_gate_passed = bool(pipeline_decision["gate_passed"])
    del pipeline_candidate, pipeline_control
    gc.collect()
    fold_root = report_root / str(outer_fold)
    fold_root.mkdir(parents=True, exist_ok=True)
    ranked = sorted(all_evaluations, key=lambda item: (int(item.eligible), item.score), reverse=True)
    atomic_write_csv(
        pd.DataFrame(
            [evaluation_record(item, rank=index + 1) for index, item in enumerate(ranked)]
        ),
        fold_root / "visited_candidates.csv",
        index=False,
    )
    manifest = beam_search_manifest(
        result,
        settings=settings,
        outer_fold=outer_fold,
        context_features=context_features,
        candidate_features=candidate_features,
        rolling_windows=[item[0] for item in frame_specs],
        permutation=permutation,
    )
    manifest.update(
        {
            "all_increment_candidates": list(candidate_features),
            "prefilter_method": "none_full_13_candidate_universe",
            "all_accessed_combinations": len(all_evaluations),
            "search_classifier_spec": asdict(search_classifier_spec),
            "sampling": {
                item[0].name: item[5]
                for item in frame_specs
            },
            "permutation_gate_passed": permutation_gate_passed,
            "pipeline_select": pipeline_decision,
            "pipeline_select_gate_passed": pipeline_select_gate_passed,
            "development_gates_passed": bool(
                result.best.eligible
                and permutation_gate_passed
                and pipeline_select_gate_passed
            ),
        }
    )
    atomic_write_json(manifest, fold_root / "search_manifest.json")
    print(
        json.dumps(
            {
                "event": "beam_v3_development_done",
                "outer_fold": outer_fold,
                "development_gates_passed": manifest["development_gates_passed"],
                "pipeline_selected_variant": pipeline_decision[
                    "selected_variant"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result.best.selected, manifest


def train_models(args: argparse.Namespace) -> dict[str, Any]:
    requested_experiments = tuple(
        dict.fromkeys(getattr(args, "experiments", DEFAULT_TRAIN_EXPERIMENTS))
    )
    unknown_experiments = set(requested_experiments) - set(EXPERIMENTS)
    if unknown_experiments:
        raise ValueError(f"unknown experiments: {sorted(unknown_experiments)}")
    requested = set(requested_experiments)
    fold_by_name = {fold.name: fold for fold in DEFAULT_YEAR_FOLDS}
    unknown_folds = set(args.folds) - set(fold_by_name)
    if unknown_folds:
        raise ValueError(f"unknown fold names: {sorted(unknown_folds)}")
    selected_folds = [fold_by_name[name] for name in args.folds]
    validate_target_cost(args.label, args.round_trip_cost_bps)
    current_target_metadata = target_metadata(args.label)
    fixed_features = list(
        stable_canonical_feature_union(PROJECT_FACTOR_COLUMNS, RULE_FEATURE_COLUMNS)
    )
    label_input_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "date",
                "label_end_date",
                "mature",
                "locked_limit_up",
                *target_source_columns(args.label),
                "terminal_return",
                "mfe",
                "mae",
            ]
        )
    )
    label_frame = pd.read_parquet(
        args.labels,
        columns=label_input_columns,
        filters=[
            ("entry_mode", "==", args.entry_mode),
            ("horizon", "==", args.horizon),
        ],
    )
    label_frame["date"] = pd.to_datetime(label_frame["date"])
    label_frame["label_end_date"] = pd.to_datetime(label_frame["label_end_date"])
    label_frame = materialize_training_target(label_frame, args.label)
    label_frame = label_frame[
        label_frame["mature"]
        & label_frame[args.label].notna()
        & ~label_frame["locked_limit_up"]
    ].copy()
    event_columns = list(
        dict.fromkeys(
            [
                "symbol",
                "date",
                *RIGHT_SIDE_SIGNALS,
                *SUBVARIANT_COLUMNS,
                "signal_count",
                "has_right_signal",
                "has_mixed_signal",
                *fixed_features,
            ]
        )
    )
    events = _read_canonical_feature_frame(args.dataset, columns=event_columns)
    events["date"] = pd.to_datetime(events["date"])
    selected = events.merge(
        label_frame,
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )
    # The dataset builder may concatenate process results in completion order.
    # Fix the row order before stable tie-breaking, weighting, and model fits so
    # reruns remain deterministic.
    selected = selected.sort_values(["date", "symbol"], kind="stable").reset_index(
        drop=True
    )
    del events, label_frame
    gc.collect()
    selected[args.label] = selected[args.label].astype(int)
    if args.label == "hit_down3":
        raise ValueError(
            "hit_down3 is a risk model and cannot be ranked as a buy model; "
            "train an upside/path label and use hit_down3 only for risk diagnostics"
        )
    missing_features = set(fixed_features) - set(selected.columns)
    if missing_features:
        raise RuntimeError(f"dataset missing fixed model features: {sorted(missing_features)}")
    args.model_root.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    signal_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []
    feature_counts: dict[str, dict[str, int]] = {}

    for fold in selected_folds:
        splits = split_by_year_fold(selected, fold)
        if splits.test.empty or splits.validation.empty or splits.train.empty:
            continue
        early_stop_stage, calibration_stage, threshold_stage = _split_validation_stages(
            splits.validation,
            args.label,
        )
        admitted_project_features = ensure_model_features(
            splits.train,
            PROJECT_FACTOR_COLUMNS,
            minimum_coverage=args.minimum_project_feature_coverage,
        )
        generic_features = list(admitted_project_features)
        common_features = [*generic_features, *RULE_FEATURE_COLUMNS]
        conditioned_features = [
            *common_features,
            *RIGHT_SIDE_SIGNALS,
            *SUBVARIANT_COLUMNS,
            "signal_count",
            "has_right_signal",
            "has_mixed_signal",
        ]
        feature_counts[fold.name] = {
            "project": len(admitted_project_features),
            "generic": len(generic_features),
            "mandatory_rule": len(RULE_FEATURE_COLUMNS),
            "common": len(common_features),
            "conditioned": len(conditioned_features),
            "long_task": len(common_features) + len(LONG_TASK_FEATURE_COLUMNS),
        }
        rule_probability = np.full(len(splits.test), float(splits.train[args.label].mean()))
        if "rule_only" in requested:
            rule_metrics = binary_metrics(splits.test[args.label], rule_probability, threshold=0.0)
            metrics_rows.append({
                "fold": fold.name,
                "entry_mode": args.entry_mode,
                "horizon": args.horizon,
                "label": args.label,
                "experiment": "rule_only",
                "signal": "ALL",
                "status": "ok",
                **rule_metrics,
            })
            rule_signal = signal_metrics(
                splits.test,
                rule_probability,
                label_column=args.label,
                threshold=None,
            )
            rule_signal.insert(0, "label", args.label)
            rule_signal.insert(0, "horizon", args.horizon)
            rule_signal.insert(0, "entry_mode", args.entry_mode)
            rule_signal.insert(0, "experiment", "rule_only")
            rule_signal.insert(0, "fold", fold.name)
            signal_rows.append(rule_signal)

        predictions: dict[str, np.ndarray] = {}
        for experiment, features, balanced in (
            # Keep the factor set identical for the signal-ID ablation.  The
            # only difference between these two arms must be identity fields.
            ("unified_without_signal_id", common_features, False),
            ("unified_with_signal_id", conditioned_features, False),
            ("unified_balanced", conditioned_features, True),
        ):
            if experiment not in requested:
                continue
            weight = balanced_sample_weights(splits.train) if balanced else None
            model = _fit_classifier(
                splits.train,
                early_stop_stage,
                calibration_stage,
                features,
                args.label,
                sample_weight=weight,
                n_jobs=args.model_jobs,
            )
            validation_probability = model.predict_proba(threshold_stage)[:, 1]
            threshold = choose_validation_threshold(threshold_stage[args.label], validation_probability)
            test_probability = model.predict_proba(splits.test)[:, 1]
            predictions[experiment] = test_probability
            metrics = binary_metrics(splits.test[args.label], test_probability, threshold=threshold)
            metrics_rows.append({
                "fold": fold.name,
                "entry_mode": args.entry_mode,
                "horizon": args.horizon,
                "label": args.label,
                "experiment": experiment,
                "signal": "ALL",
                "status": "ok",
                "train_rows": len(splits.train),
                "validation_rows": len(splits.validation),
                "project_feature_count": len(admitted_project_features),
                "mandatory_rule_feature_count": len(RULE_FEATURE_COLUMNS),
                **metrics,
                **daily_top_k_trading_metrics(
                    splits.test,
                    test_probability,
                    top_k=args.daily_top_k,
                    round_trip_cost_bps=args.round_trip_cost_bps,
                ),
            })
            per_signal = signal_metrics(splits.test, test_probability, label_column=args.label, threshold=threshold)
            per_signal.insert(0, "label", args.label)
            per_signal.insert(0, "horizon", args.horizon)
            per_signal.insert(0, "entry_mode", args.entry_mode)
            per_signal.insert(0, "experiment", experiment)
            per_signal.insert(0, "fold", fold.name)
            signal_rows.append(per_signal)
            model_path = args.model_root / args.entry_mode / f"h{args.horizon}" / args.label / fold.name / f"{experiment}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            del model
            gc.collect()

        for arm_spec in LONG_TASK_ARM_SPECS:
            experiment = arm_spec.experiment
            if experiment not in requested:
                continue
            arm_rule_features = list(arm_spec.rule_feature_columns)
            arm_common_features = [*generic_features, *arm_rule_features]
            (
                long_task_model,
                long_task_train_rows,
                weight_metadata,
            ) = _fit_long_task_classifier(
                splits.train,
                early_stop_stage,
                calibration_stage,
                arm_common_features,
                args.label,
                n_jobs=args.model_jobs,
                task_weighting=arm_spec.task_weighting,
                classifier_spec=arm_spec.classifier_spec,
            )
            validation_probability = long_task_model.predict_proba(threshold_stage)[:, 1]
            threshold = choose_validation_threshold(
                threshold_stage[args.label],
                validation_probability,
            )
            test_probability = long_task_model.predict_proba(splits.test)[:, 1]
            predictions[experiment] = test_probability
            metrics_rows.append({
                "fold": fold.name,
                "entry_mode": args.entry_mode,
                "horizon": args.horizon,
                "label": args.label,
                "experiment": experiment,
                "signal": "ALL",
                "status": "ok",
                "train_rows": len(splits.train),
                "long_task_train_rows": long_task_train_rows,
                "validation_rows": len(splits.validation),
                "project_feature_count": len(admitted_project_features),
                "mandatory_rule_feature_count": len(arm_rule_features),
                "task_feature_count": len(LONG_TASK_FEATURE_COLUMNS),
                "xgb_max_depth": arm_spec.classifier_spec.max_depth,
                "xgb_min_child_weight": arm_spec.classifier_spec.min_child_weight,
                **weight_metadata,
                **binary_metrics(
                    splits.test[args.label],
                    test_probability,
                    threshold=threshold,
                ),
                **daily_top_k_trading_metrics(
                    splits.test,
                    test_probability,
                    top_k=args.daily_top_k,
                    round_trip_cost_bps=args.round_trip_cost_bps,
                ),
            })
            per_signal = signal_metrics(
                splits.test,
                test_probability,
                label_column=args.label,
                threshold=threshold,
            )
            per_signal.insert(0, "label", args.label)
            per_signal.insert(0, "horizon", args.horizon)
            per_signal.insert(0, "entry_mode", args.entry_mode)
            per_signal.insert(0, "experiment", experiment)
            per_signal.insert(0, "fold", fold.name)
            signal_rows.append(per_signal)
            model_path = (
                args.model_root
                / args.entry_mode
                / f"h{args.horizon}"
                / args.label
                / fold.name
                / f"{experiment}.joblib"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(long_task_model, model_path)
            atomic_write_json(
                {
                    "schema_version": LONG_TASK_SCHEMA_VERSION,
                    "experiment": experiment,
                    "fold": asdict(fold),
                    "common_features": list(arm_common_features),
                    "feature_schema_version": RIGHT_SIDE_SHADOW_FEATURE_SCHEMA_VERSION,
                    "factor_contract_sha256": RIGHT_SIDE_SHADOW_FACTOR_CONTRACT_SHA256,
                    "model_input_contract_sha256": factor_contract_sha256(
                        [*arm_common_features, *LONG_TASK_FEATURE_COLUMNS],
                        schema_version=LONG_TASK_SCHEMA_VERSION,
                    ),
                    "forbidden_aliases": [],
                    "rule_feature_schema_version": arm_spec.rule_feature_schema_version,
                    "rule_feature_count": len(arm_rule_features),
                    "rule_feature_columns": arm_rule_features,
                    "rule_feature_columns_sha256": rule_feature_columns_sha256(
                        arm_rule_features
                    ),
                    "task_features": list(LONG_TASK_FEATURE_COLUMNS),
                    "train_event_rows": len(splits.train),
                    "train_task_rows": long_task_train_rows,
                    "task_row_weighting": arm_spec.task_weighting,
                    "sample_weight": weight_metadata,
                    "xgb_classifier_spec": asdict(arm_spec.classifier_spec),
                    "event_aggregation": "max_active_task_probability",
                    "calibration": "platt_after_event_max_on_calibration_stage",
                    "threshold_selection": "event_max_on_threshold_stage",
                    "target": {
                        "name": args.label,
                        **current_target_metadata,
                    },
                },
                model_path.with_suffix(".manifest.json"),
            )
            del long_task_model
            gc.collect()

        beam_experiment = "unified_long_task_deep_beam"
        if beam_experiment in requested:
            beam_settings = BeamSettings(
                width=args.beam_width,
                min_features=args.beam_min_features,
                max_remove=args.beam_max_remove,
                permutation_rounds=args.beam_permutation_rounds,
                maximum_permutation_p_value=args.beam_maximum_permutation_p_value,
            )
            beam_search_spec = replace(
                LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
                n_estimators=args.beam_search_estimators,
                max_depth=args.beam_search_max_depth,
                early_stopping_rounds=args.beam_search_early_stopping_rounds,
            )
            selected_increment_features, beam_manifest = _run_beam_development_search(
                train=splits.train,
                validation=splits.validation,
                admitted_project_features=admitted_project_features,
                label=args.label,
                outer_fold=fold.name,
                n_jobs=args.model_jobs,
                settings=beam_settings,
                search_classifier_spec=beam_search_spec,
                report_root=args.beam_report_root,
                history_max_rows=args.beam_history_max_rows,
                evaluation_max_rows=args.beam_evaluation_max_rows,
            )
            beam_common_features = [
                *generic_features,
                *LEGACY_RULE_FEATURE_COLUMNS_V1,
                *selected_increment_features,
            ]
            (
                beam_model,
                beam_train_rows,
                beam_weight_metadata,
            ) = _fit_long_task_classifier(
                splits.train,
                early_stop_stage,
                calibration_stage,
                beam_common_features,
                args.label,
                n_jobs=args.model_jobs,
                task_weighting="one_vote",
                classifier_spec=LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
            )
            beam_validation_probability = beam_model.predict_proba(threshold_stage)[:, 1]
            beam_threshold = choose_validation_threshold(
                threshold_stage[args.label], beam_validation_probability
            )
            beam_test_probability = beam_model.predict_proba(splits.test)[:, 1]
            predictions[beam_experiment] = beam_test_probability
            metrics_rows.append(
                {
                    "fold": fold.name,
                    "entry_mode": args.entry_mode,
                    "horizon": args.horizon,
                    "label": args.label,
                    "experiment": beam_experiment,
                    "signal": "ALL",
                    "status": "ok",
                    "train_rows": len(splits.train),
                    "long_task_train_rows": beam_train_rows,
                    "validation_rows": len(splits.validation),
                    "project_feature_count": len(admitted_project_features),
                    "mandatory_rule_feature_count": len(LEGACY_RULE_FEATURE_COLUMNS_V1)
                    + len(selected_increment_features),
                    "beam_selected_increment_count": len(selected_increment_features),
                    "beam_visited_combinations": beam_manifest["all_accessed_combinations"],
                    "beam_permutation_p_value": beam_manifest["permutation"]["empirical_p_value"],
                    "beam_permutation_gate_passed": beam_manifest["permutation_gate_passed"],
                    "task_feature_count": len(LONG_TASK_FEATURE_COLUMNS),
                    "xgb_max_depth": LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC.max_depth,
                    "xgb_min_child_weight": LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC.min_child_weight,
                    **beam_weight_metadata,
                    **binary_metrics(
                        splits.test[args.label],
                        beam_test_probability,
                        threshold=beam_threshold,
                    ),
                    **daily_top_k_trading_metrics(
                        splits.test,
                        beam_test_probability,
                        top_k=args.daily_top_k,
                        round_trip_cost_bps=args.round_trip_cost_bps,
                    ),
                }
            )
            beam_signal = signal_metrics(
                splits.test,
                beam_test_probability,
                label_column=args.label,
                threshold=beam_threshold,
            )
            beam_signal.insert(0, "label", args.label)
            beam_signal.insert(0, "horizon", args.horizon)
            beam_signal.insert(0, "entry_mode", args.entry_mode)
            beam_signal.insert(0, "experiment", beam_experiment)
            beam_signal.insert(0, "fold", fold.name)
            signal_rows.append(beam_signal)
            beam_model_path = (
                args.model_root
                / args.entry_mode
                / f"h{args.horizon}"
                / args.label
                / fold.name
                / f"{beam_experiment}.joblib"
            )
            beam_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(beam_model, beam_model_path)
            atomic_write_json(
                {
                    "schema_version": LONG_TASK_SCHEMA_VERSION,
                    "experiment": beam_experiment,
                    "fold": asdict(fold),
                    "common_features": beam_common_features,
                    "rule_feature_schema_version": (
                        "right_side_rule_features_v1_105_plus_beam_v2_increment"
                    ),
                    "beam_schema_version": BEAM_SCHEMA_VERSION,
                    "rule_feature_count": len(LEGACY_RULE_FEATURE_COLUMNS_V1)
                    + len(selected_increment_features),
                    "rule_feature_columns": [
                        *LEGACY_RULE_FEATURE_COLUMNS_V1,
                        *selected_increment_features,
                    ],
                    "rule_feature_columns_sha256": rule_feature_columns_sha256(
                        [*LEGACY_RULE_FEATURE_COLUMNS_V1, *selected_increment_features]
                    ),
                    "beam_search": beam_manifest,
                    "task_features": list(LONG_TASK_FEATURE_COLUMNS),
                    "train_event_rows": len(splits.train),
                    "train_task_rows": beam_train_rows,
                    "task_row_weighting": "one_vote",
                    "sample_weight": beam_weight_metadata,
                    "xgb_classifier_spec": asdict(LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC),
                    "event_aggregation": "max_active_task_probability",
                    "target": {"name": args.label, **current_target_metadata},
                },
                beam_model_path.with_suffix(".manifest.json"),
            )
            del beam_model
            gc.collect()

        local_test_predictions: dict[str, np.ndarray] = {}
        local_threshold_predictions: dict[str, np.ndarray] = {}
        local_status: dict[str, str] = {}
        independent_signals = RIGHT_SIDE_SIGNALS if "independent" in requested else ()
        for signal in independent_signals:
            train_local = splits.train[splits.train[signal]].copy()
            validation_local = splits.validation[splits.validation[signal]].copy()
            if not _valid_local_model(train_local, args.label, args.minimum_local_rows) or not _valid_local_model(
                validation_local, args.label, max(200, args.minimum_local_rows // 4)
            ):
                local_status[signal] = "no_valid_local_model"
                continue
            # Use the exact same calendar boundaries as the unified arms.  A
            # per-signal quantile split can move the boundary and let the
            # aggregate independent threshold reuse calibration/early-stop
            # observations.
            local_early_stop = early_stop_stage[
                early_stop_stage[signal].fillna(False).astype(bool)
            ].copy()
            local_calibration = calibration_stage[
                calibration_stage[signal].fillna(False).astype(bool)
            ].copy()
            local_threshold = threshold_stage[
                threshold_stage[signal].fillna(False).astype(bool)
            ].copy()
            if any(
                frame.empty or frame[args.label].nunique() < 2
                for frame in (local_early_stop, local_calibration, local_threshold)
            ):
                local_status[signal] = "no_valid_local_validation_stages"
                continue
            model = _fit_classifier(
                train_local,
                local_early_stop,
                local_calibration,
                common_features,
                args.label,
                sample_weight=None,
                n_jobs=args.model_jobs,
            )
            validation_probability = model.predict_proba(local_threshold)[:, 1]
            threshold = choose_validation_threshold(local_threshold[args.label], validation_probability)
            active = splits.test[signal].fillna(False).astype(bool)
            test_probability = np.full(len(splits.test), np.nan, dtype=float)
            if active.any():
                test_probability[active.to_numpy()] = model.predict_proba(
                    splits.test.loc[active]
                )[:, 1]
            threshold_active = threshold_stage[signal].fillna(False).astype(bool)
            threshold_probability = np.full(len(threshold_stage), np.nan, dtype=float)
            threshold_probability[threshold_active.to_numpy()] = validation_probability
            local_test_predictions[signal] = test_probability
            local_threshold_predictions[signal] = threshold_probability
            metrics = binary_metrics(
                splits.test.loc[active, args.label],
                test_probability[active.to_numpy()],
                threshold=threshold,
            )
            metrics_rows.append({
                "fold": fold.name,
                "entry_mode": args.entry_mode,
                "horizon": args.horizon,
                "label": args.label,
                "experiment": "independent_member",
                "signal": signal,
                "status": "ok",
                "train_rows": len(train_local),
                "validation_rows": len(validation_local),
                **metrics,
            })
            local_status[signal] = "ok"
            model_path = args.model_root / args.entry_mode / f"h{args.horizon}" / args.label / fold.name / "independent" / f"{signal}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            del model
            gc.collect()

        missing_independent: np.ndarray | None = None
        if "independent" in requested:
            independent = aggregate_independent_predictions(splits.test, local_test_predictions)
            independent_threshold_probability = aggregate_independent_predictions(
                threshold_stage,
                local_threshold_predictions,
            )
            missing_independent = ~np.isfinite(independent)
            missing_threshold = ~np.isfinite(independent_threshold_probability)
            independent[missing_independent] = rule_probability[missing_independent]
            independent_threshold_probability[missing_threshold] = float(
                splits.train[args.label].mean()
            )
            independent_threshold = choose_validation_threshold(
                threshold_stage[args.label],
                independent_threshold_probability,
            )
            predictions["independent"] = independent
            metrics_rows.append({
                "fold": fold.name,
                "entry_mode": args.entry_mode,
                "horizon": args.horizon,
                "label": args.label,
                "experiment": "independent",
                "signal": "ALL",
                "status": "ok_with_constant_fallback" if missing_independent.any() else "ok",
                "fallback_rows": int(missing_independent.sum()),
                "threshold_fallback_rows": int(missing_threshold.sum()),
                **binary_metrics(
                    splits.test[args.label],
                    independent,
                    threshold=independent_threshold,
                ),
                **daily_top_k_trading_metrics(
                    splits.test,
                    independent,
                    top_k=args.daily_top_k,
                    round_trip_cost_bps=args.round_trip_cost_bps,
                ),
            })
            independent_signal = signal_metrics(
                splits.test,
                independent,
                label_column=args.label,
                threshold=independent_threshold,
            )
            independent_signal.insert(0, "label", args.label)
            independent_signal.insert(0, "horizon", args.horizon)
            independent_signal.insert(0, "entry_mode", args.entry_mode)
            independent_signal.insert(0, "experiment", "independent")
            independent_signal.insert(0, "fold", fold.name)
            signal_rows.append(independent_signal)
            local_manifest = {
                "fold": asdict(fold),
                "local_status": local_status,
                "fallback_rows": int(missing_independent.sum()),
                "threshold_fallback_rows": int(missing_threshold.sum()),
                "target": {
                    "name": args.label,
                    **current_target_metadata,
                },
            }
            atomic_write_json(
                local_manifest,
                args.model_root / args.entry_mode / f"h{args.horizon}" / args.label / fold.name / "manifest.json",
            )
        prediction_frame = splits.test[["symbol", "date", *RIGHT_SIDE_SIGNALS, args.label, "terminal_return", "mfe", "mae"]].copy()
        prediction_frame["fold"] = fold.name
        prediction_frame["entry_mode"] = args.entry_mode
        prediction_frame["horizon"] = args.horizon
        prediction_frame["label"] = args.label
        for experiment, probability in predictions.items():
            prediction_frame[f"pred_{experiment}"] = probability
        if "rule_only" in requested:
            prediction_frame["pred_rule_only"] = rule_probability
        if missing_independent is not None:
            prediction_frame["independent_model_available"] = ~missing_independent
        prediction_rows.append(prediction_frame)

    if not metrics_rows:
        raise RuntimeError("no complete time fold could be trained")
    metrics_frame = pd.DataFrame(metrics_rows)
    signal_frame = pd.concat(signal_rows, ignore_index=True, sort=False) if signal_rows else pd.DataFrame()
    prediction_frame = pd.concat(prediction_rows, ignore_index=True, sort=False)
    for column, value in current_target_metadata.items():
        metrics_frame[column] = value
        if not signal_frame.empty:
            signal_frame[column] = value
        prediction_frame[column] = value
    scope_columns = ["entry_mode", "horizon", "label"]
    produced_metric_experiments = set(metrics_frame["experiment"].astype(str))
    if args.metrics_out.exists():
        existing = pd.read_csv(args.metrics_out)
        keep = ~(
            existing["entry_mode"].eq(args.entry_mode)
            & existing["horizon"].eq(args.horizon)
            & existing["label"].eq(args.label)
            & existing["fold"].isin(args.folds)
            & existing["experiment"].isin(produced_metric_experiments)
        )
        metrics_frame = pd.concat([existing.loc[keep], metrics_frame], ignore_index=True, sort=False)
    if args.signal_metrics_out.exists() and not signal_frame.empty:
        existing_signal = pd.read_csv(args.signal_metrics_out)
        if set(scope_columns) <= set(existing_signal.columns):
            keep = ~(
                existing_signal["entry_mode"].eq(args.entry_mode)
                & existing_signal["horizon"].eq(args.horizon)
                & existing_signal["label"].eq(args.label)
                & existing_signal["fold"].isin(args.folds)
                & existing_signal["experiment"].isin(produced_metric_experiments)
            )
            signal_frame = pd.concat([existing_signal.loc[keep], signal_frame], ignore_index=True, sort=False)
    if args.predictions_out.exists():
        existing_predictions = pd.read_parquet(args.predictions_out)
        replace_prediction_columns = [
            f"pred_{experiment}"
            for experiment in requested_experiments
            if f"pred_{experiment}" in prediction_frame.columns
        ]
        if "independent_model_available" in prediction_frame.columns:
            replace_prediction_columns.append("independent_model_available")
        prediction_frame = merge_prediction_artifacts(
            existing_predictions,
            prediction_frame,
            replace_columns=replace_prediction_columns,
        )
    atomic_write_csv(metrics_frame, args.metrics_out, index=False)
    atomic_write_csv(signal_frame, args.signal_metrics_out, index=False)
    atomic_write_parquet(prediction_frame, args.predictions_out, index=False, compression="zstd")
    return {
        "metrics_rows": len(metrics_frame),
        "signal_metrics_rows": len(signal_frame),
        "prediction_rows": len(prediction_frame),
        "feature_counts": feature_counts,
        "trained_folds": list(args.folds),
        "trained_experiments": list(requested_experiments),
        "target": current_target_metadata,
    }


def _format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def compare_factor_increment(args: argparse.Namespace) -> dict[str, Any]:
    """Compare candidate rule schemas using ranking metrics only.

    This entry point intentionally never loads a return column.  It reads only
    A/B rows by default and fails if callers try to turn the already-seen C
    period into a development decision.
    """

    folds = tuple(dict.fromkeys(str(value) for value in args.folds))
    if not folds or set(folds) - {"A", "B"}:
        raise ValueError("factor-increment promotion decisions may use folds A/B only")
    ranking_columns = list(
        dict.fromkeys(
            [
                "date",
                "entry_mode",
                "horizon",
                "label",
                "fold",
                args.label,
                *RIGHT_SIDE_SIGNALS,
                f"pred_{args.baseline_experiment}",
                *[f"pred_{value}" for value in args.candidate_experiments],
            ]
        )
    )
    # Deliberately project only ranking/identity columns.  Return, MFE, MAE and
    # costs cannot accidentally influence the first-layer replacement gate.
    predictions = pd.read_parquet(args.predictions, columns=ranking_columns)
    predictions = predictions[predictions["fold"].astype(str).isin(folds)].copy()
    if predictions.empty:
        raise ValueError(f"prediction artifact contains no A/B rows for folds={folds}")

    comparison_frames: list[pd.DataFrame] = []
    decision_candidates: dict[str, Any] = {}
    for experiment in args.candidate_experiments:
        candidate_column = f"pred_{experiment}"
        baseline_column = f"pred_{args.baseline_experiment}"
        overall = compare_rule_feature_versions(
            predictions,
            candidate_column=candidate_column,
            baseline_column=baseline_column,
            label_column=args.label,
            top_fraction=args.top_fraction,
            bootstrap_iterations=args.bootstrap_iterations,
            random_seed=args.random_seed,
            daily_top_k=args.daily_top_k,
            confidence_level=args.confidence_level,
        )
        overall.insert(0, "signal", "ALL")
        overall.insert(0, "candidate_experiment", experiment)
        signal_frames: list[pd.DataFrame] = []
        for signal in RIGHT_SIDE_SIGNALS:
            if signal not in predictions.columns:
                raise ValueError(f"prediction artifact missing signal identity {signal}")
            active = predictions[predictions[signal].fillna(False).astype(bool)]
            if active.empty:
                continue
            signal_comparison = compare_rule_feature_versions(
                active,
                candidate_column=candidate_column,
                baseline_column=baseline_column,
                label_column=args.label,
                top_fraction=args.top_fraction,
                bootstrap_iterations=0,
                random_seed=args.random_seed,
                daily_top_k=args.daily_top_k,
                confidence_level=args.confidence_level,
            )
            signal_comparison.insert(0, "signal", signal)
            signal_comparison.insert(0, "candidate_experiment", experiment)
            signal_frames.append(signal_comparison)
        candidate_comparison = pd.concat(
            [overall, *signal_frames], ignore_index=True, sort=False
        )
        comparison_frames.append(candidate_comparison)

        fold_rows = overall[overall["status"].isin(["ok", "exact_only"])].copy()
        signal_rows = (
            pd.concat(signal_frames, ignore_index=True, sort=False)
            if signal_frames
            else pd.DataFrame()
        )
        fold_positive = bool(
            len(fold_rows) == len(folds)
            and fold_rows["delta_pr_auc"].gt(0).all()
        )
        mean_delta_pr = float(fold_rows["delta_pr_auc"].mean())
        mean_delta_lift = float(fold_rows["delta_top10_lift"].mean())
        macro_signal_delta = float(signal_rows["delta_pr_auc"].mean())
        coverage = float(fold_rows["coverage"].min())
        criteria = {
            "each_fold_delta_pr_auc_positive": fold_positive,
            "mean_delta_pr_auc_positive": bool(mean_delta_pr > 0),
            "mean_delta_top10_lift_nonnegative": bool(mean_delta_lift >= 0),
            "macro_signal_delta_pr_auc_nonnegative": bool(macro_signal_delta >= 0),
            "paired_coverage_at_least_0p99": bool(coverage >= 0.99),
        }
        decision_candidates[experiment] = {
            "baseline_experiment": args.baseline_experiment,
            "folds": list(folds),
            "mean_delta_pr_auc": mean_delta_pr,
            "mean_delta_top10_lift": mean_delta_lift,
            "macro_signal_delta_pr_auc": macro_signal_delta,
            "minimum_coverage": coverage,
            "criteria": criteria,
            "ranking_gate_passed": bool(all(criteria.values())),
        }

    output = pd.concat(comparison_frames, ignore_index=True, sort=False)
    atomic_write_csv(output, args.comparison_out, index=False)
    for experiment, values in decision_candidates.items():
        values["all_candidate_gates_passed"] = bool(values["ranking_gate_passed"])
        if experiment != "unified_long_task_deep_beam":
            continue
        manifests = [
            args.beam_report_root / fold / "search_manifest.json"
            for fold in folds
        ]
        missing_manifests = [str(path) for path in manifests if not path.exists()]
        if missing_manifests:
            values["beam_permutation_gate"] = {
                "passed": False,
                "missing_manifests": missing_manifests,
            }
            values["all_candidate_gates_passed"] = False
            continue
        payload_by_fold = {
            fold: json.loads(path.read_text(encoding="utf-8"))
            for fold, path in zip(folds, manifests)
        }
        provenance_by_fold: dict[str, bool] = {}
        permutation_by_fold: dict[str, bool] = {}
        pipeline_select_by_fold: dict[str, bool] = {}
        for fold, payload in payload_by_fold.items():
            selected_features = tuple(payload.get("selected_features", ()))
            provenance_by_fold[fold] = bool(
                payload.get("schema_version") == BEAM_SCHEMA_VERSION
                and payload.get("test_data_used") is False
                and payload.get("test_data_used_for_search") is False
                and tuple(payload.get("candidate_features", ()))
                == tuple(ADDED_RULE_FEATURE_COLUMNS_V2)
                and payload.get("selected_features_sha256")
                == beam_feature_columns_sha256(selected_features)
            )
            permutation_by_fold[fold] = bool(
                payload.get("permutation_gate_passed")
            )
            pipeline_select_by_fold[fold] = bool(
                payload.get("pipeline_select_gate_passed")
                and payload.get("pipeline_select", {}).get("test_data_used")
                is False
            )
        provenance_passed = bool(all(provenance_by_fold.values()))
        beam_permutation_passed = bool(all(permutation_by_fold.values()))
        pipeline_select_passed = bool(all(pipeline_select_by_fold.values()))
        values["beam_provenance_gate"] = {
            "passed": provenance_passed,
            "by_fold": provenance_by_fold,
            "schema_version": BEAM_SCHEMA_VERSION,
        }
        values["beam_permutation_gate"] = {
            "passed": beam_permutation_passed,
            "by_fold": permutation_by_fold,
        }
        values["beam_pipeline_select_gate"] = {
            "passed": pipeline_select_passed,
            "by_fold": pipeline_select_by_fold,
        }
        values["all_candidate_gates_passed"] = bool(
            values["ranking_gate_passed"]
            and provenance_passed
            and beam_permutation_passed
            and pipeline_select_passed
        )
    eligible = [
        (experiment, values)
        for experiment, values in decision_candidates.items()
        if values["all_candidate_gates_passed"]
    ]
    selected_candidate: str | None = None
    replace_online = False
    decision_reason = "no_candidate_passed_ranking_gate"
    if eligible:
        selected_candidate, _ = max(
            eligible,
            key=lambda item: float(item[1]["mean_delta_pr_auc"]),
        )
        replace_online = True
        decision_reason = (
            "ranking_and_beam_v3_development_gates_passed"
            if selected_candidate == "unified_long_task_deep_beam"
            else "ranking_gate_passed"
        )
    elif any(
        values.get("ranking_gate_passed")
        and not values.get("all_candidate_gates_passed")
        for values in decision_candidates.values()
    ):
        decision_reason = "ranking_passed_but_candidate_specific_gate_failed"
    decision = {
        "schema_version": "right-side-rule-factor-ranking-decision-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_metric": "ranking_only_no_returns",
        "label": args.label,
        "folds": list(folds),
        "selected_candidate": selected_candidate,
        "replace_online": replace_online,
        "decision_reason": decision_reason,
        "candidates": decision_candidates,
    }
    atomic_write_json(decision, args.decision_out)
    return decision


def compare_legacy_artifact_ranking(args: argparse.Namespace) -> dict[str, Any]:
    candidate_columns = [
        "symbol",
        "date",
        "entry_mode",
        "horizon",
        "label",
        "fold",
        "good_path5",
        f"pred_{args.candidate_experiment}",
    ]
    legacy_columns = [
        "symbol",
        "date",
        "entry_mode",
        "horizon",
        "fold",
        "good_path5",
        "legacy_quality_score",
        "legacy_signal_timing_match_any",
        "legacy_temporal_status",
    ]
    candidate = pd.read_parquet(args.predictions, columns=candidate_columns)
    legacy = pd.read_parquet(args.legacy_predictions, columns=legacy_columns)
    comparison = compare_candidate_with_legacy_artifact(
        candidate,
        legacy,
        candidate_column=f"pred_{args.candidate_experiment}",
        folds=args.folds,
        top_fraction=args.top_fraction,
        daily_top_k=args.daily_top_k,
        bootstrap_iterations=args.bootstrap_iterations,
        random_seed=args.random_seed,
    )
    atomic_write_csv(comparison, args.comparison_out, index=False)
    return {
        "schema_version": "right-side-legacy-artifact-ranking-comparison-v1",
        "candidate_experiment": args.candidate_experiment,
        "folds": list(args.folds),
        "rows": len(comparison),
        "comparison_out": str(args.comparison_out),
        "coverage_scope": "exact_symbol_date_legacy_overlap_only_no_extrapolation",
        "label_contract": [
            "candidate_training_contract_good_path5",
            "legacy_next_open_good_path5",
        ],
    }


def _format_delta_interval(row: pd.Series, metric: str, digits: int = 4) -> str:
    estimate = _format_number(row.get(f"delta_{metric}"), digits)
    low = _format_number(row.get(f"delta_{metric}_ci_low"), digits)
    high = _format_number(row.get(f"delta_{metric}_ci_high"), digits)
    return f"{estimate} [{low}, {high}]"


def render_report(args: argparse.Namespace) -> Path:
    report_folds = tuple(
        dict.fromkeys(
            getattr(args, "folds", [fold.name for fold in DEFAULT_YEAR_FOLDS])
        )
    )
    metrics = pd.read_csv(args.metrics)
    metrics = metrics[metrics["fold"].astype(str).isin(report_folds)].copy()
    sample = json.loads(args.sample_audit.read_text(encoding="utf-8"))
    factors = pd.read_csv(args.factor_audit)
    rule_factor_audit = factors[factors["signal"].astype(str).ne("ALL")]
    all_factor_audit = factors[factors["signal"].astype(str).eq("ALL")]
    missing_rule_factor_rows = int(
        rule_factor_audit["status"].isin(["missing", "all_null"]).sum()
    )
    nonvarying_rule_factor_rows = int(
        rule_factor_audit["status"].isin(["constant", "sparse"]).sum()
    )
    excluded_daily_basic_columns = int(
        all_factor_audit["status"].eq("all_null").sum()
    )
    predictions = pd.read_parquet(
        args.predictions,
        filters=[("fold", "in", list(report_folds))],
    )
    if metrics.empty or predictions.empty:
        raise ValueError(f"no report rows matched folds={list(report_folds)}")
    paired_candidates = [
        experiment
        for experiment in PAIRED_EXPERIMENTS
        if f"pred_{experiment}" in predictions.columns
    ]
    if not paired_candidates:
        raise ValueError("prediction artifact contains no unified paired-comparison arm")
    paired_all = paired_model_comparisons(
        predictions,
        candidate_experiments=paired_candidates,
        top_fraction=args.paired_top_fraction,
        daily_top_k=args.daily_top_k,
        bootstrap_iterations=args.paired_bootstrap_iterations,
        confidence_level=args.paired_confidence_level,
        random_state=args.paired_random_state,
    )
    paired_all.insert(0, "comparison_scope", "all_events")
    paired_frames = [paired_all]
    availability_column = "independent_model_available"
    if availability_column in predictions.columns:
        comparable_predictions = predictions[
            predictions[availability_column].fillna(False).astype(bool)
        ].copy()
        if not comparable_predictions.empty:
            paired_comparable = paired_model_comparisons(
                comparable_predictions,
                candidate_experiments=paired_candidates,
                top_fraction=args.paired_top_fraction,
                daily_top_k=args.daily_top_k,
                bootstrap_iterations=args.paired_bootstrap_iterations,
                confidence_level=args.paired_confidence_level,
                random_state=args.paired_random_state,
            )
            paired_comparable.insert(0, "comparison_scope", "independent_model_rows")
            paired_frames.append(paired_comparable)
    paired = pd.concat(paired_frames, ignore_index=True, sort=False)
    atomic_write_csv(paired, args.paired_comparison_out, index=False)
    pooled = metrics[(metrics["signal"] == "ALL") & metrics["experiment"].isin(EXPERIMENTS)].copy()
    summary = (
        pooled.groupby(["entry_mode", "horizon", "label", "experiment"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            rows=("rows", "sum"),
            roc_auc=("roc_auc", "mean"),
            average_precision=("average_precision", "mean"),
            top_lift=("top_lift", "mean"),
            brier=("brier", "mean"),
            average_net_return=("average_net_return", "mean"),
            # Profit factor is a ratio and must not be arithmetically averaged
            # across folds.  The fold median is a robust descriptive summary;
            # fold-level and paired tables remain the decision authority.
            profit_factor=("profit_factor", "median"),
        )
        .sort_values(["entry_mode", "horizon", "label", "average_precision"], ascending=[True, True, True, False])
    )
    lines = [
        "# 右侧/混合统一模型走步验证",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"开发折：{', '.join(report_folds)}",
        "",
        "## 样本审计",
        "",
        f"- 唯一事件：{sample.get('unique_events', 0):,}",
        f"- 多信号事件：{sample.get('multi_hit_events', 0):,}",
        f"- 一字涨停行：{sample.get('locked_limit_rows', 0):,}",
        f"- 成熟标签行：{sample.get('mature_rows', 0):,}",
        f"- 通用价格/技术因子：{len(PROJECT_FACTOR_COLUMNS) - excluded_daily_basic_columns}；规则专属因子：{len(RULE_FEATURE_COLUMNS)}（按策略切片缺失/全空：{missing_rule_factor_rows}；常数/稀疏：{nonvarying_rule_factor_rows}）",
        f"- daily_basic 扩展因子：{excluded_daily_basic_columns} 列按主实验合同排除并保留全空占位，不进入训练；不属于右侧筛选规则因子。",
        "",
        "## 标签定义",
        "",
        *[
            f"- `{label}`：{target_contract(str(label)).definition}。"
            for label in sorted(summary["label"].astype(str).unique())
        ],
        "",
        "## 模型结果（各时间折等权均值）",
        "",
        "| 入场 | 窗口 | 标签 | 实验 | 折数 | 样本 | ROC-AUC | PR-AUC | Top10% Lift | Top-K单笔净收益 | 交易级PF中位数 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['entry_mode']} | {int(row['horizon'])} | {row['label']} | {row['experiment']} | {int(row['folds'])} | {int(row['rows'])} | "
            f"{_format_number(row['roc_auc'])} | {_format_number(row['average_precision'])} | "
            f"{_format_number(row['top_lift'])} | {_format_number(row['average_net_return'])} | "
            f"{_format_number(row['profit_factor'])} |"
        )
    availability_column = "independent_model_available"
    if availability_column in predictions.columns:
        coverage = predictions.copy()
        coverage[availability_column] = coverage[availability_column].fillna(False).astype(bool)
        coverage = (
            coverage.groupby(["entry_mode", "horizon", "label", "fold"], as_index=False)
            .agg(
                test_rows=(availability_column, "size"),
                model_rows=(availability_column, "sum"),
                independent_model_coverage=(availability_column, "mean"),
            )
            .sort_values(["entry_mode", "horizon", "label", "fold"])
        )
        lines.extend(
            [
                "",
                "## 独立成员模型覆盖率",
                "",
                "| 入场 | 窗口 | 标签 | 折 | 测试行 | 有独立模型行 | 覆盖率 |",
                "|---|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in coverage.to_dict("records"):
            lines.append(
                f"| {row['entry_mode']} | {int(row['horizon'])} | {row['label']} | {row['fold']} | "
                f"{int(row['test_rows'])} | {int(row['model_rows'])} | "
                f"{_format_number(row['independent_model_coverage'])} |"
            )
    confidence_percent = int(round(args.paired_confidence_level * 100))
    lines.extend(
        [
            "",
            "## 统一模型相对独立模型的测试集配对差值",
            "",
            f"正值表示统一模型更优；区间为自然月整块、同测试行成对重采样的 {confidence_percent}% percentile CI。该区间衡量测试期时间块不确定性，不包含重新训练的不确定性。",
            "",
            "`all_events` 同时衡量统一模型的覆盖收益与模型收益；`independent_model_rows` 只保留独立成员模型可训练的行，更接近纯模型能力比较。",
            "",
            f"| 口径 | 入场 | 窗口 | 标签 | 折 | 统一模型 | 成对行 | 月块 | Δ PR-AUC [{confidence_percent}% CI] | Δ Top{args.paired_top_fraction:.0%} Lift [{confidence_percent}% CI] | Δ 日Top-{args.daily_top_k}平均终值收益 [{confidence_percent}% CI] |",
            "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in paired.sort_values(
        ["comparison_scope", "entry_mode", "horizon", "label", "fold", "candidate"]
    ).iterrows():
        lines.append(
            f"| {row['comparison_scope']} | {row['entry_mode']} | {int(row['horizon'])} | {row['label']} | {row['fold']} | {row['candidate']} | "
            f"{int(row['paired_rows'])} | {int(row['month_blocks'])} | "
            f"{_format_delta_interval(row, 'pr_auc')} | "
            f"{_format_delta_interval(row, 'top_lift')} | "
            f"{_format_delta_interval(row, 'daily_top_k_avg_terminal_return')} |"
        )
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            "表格严格按入场、窗口和标签分组；不同标签的基准率不同，禁止跨标签直接比较 PR-AUC。是否发布统一模型应优先看同折配对差值及其月度区块置信区间，再结合逐信号结果与交易成本。",
            "",
            "`rule_only` 表示规则已经形成的候选池，并给所有候选赋训练期基准率常数；它只衡量“不做模型排序”的候选池基线，不是真实的规则强弱排序器。",
            "",
            "`unified_long_task` 将每个事件按 active signal 展成长表，只加入当前 task one-hot；训练仍只有一个共享模型。校准和阈值选择前均按事件取各 active task 最大分，与独立模型的 max 聚合口径一致。",
            "",
            "`unified_long_task_balanced` 仅在训练长表使用 task 频次负二分之一次方权重（均值 1、裁剪至 [0.5, 3.0]）；`unified_long_task_deep` 仅将共享树改为 max_depth=6、min_child_weight=6，其余训练、事件聚合、校准和阈值契约不变。",
            "",
            "Top-K 收益与 PF 是重叠事件的单笔统计，不是资金曲线；当前报告不展示无效的复利收益和最大回撤。组合级结论需另跑逐仓位、逐日盯市的资金回测。",
            "",
            "独立模型样本不足的成员采用训练期基准率常数回退，并在每折 manifest 与预测表中显式记录；这类行反映统一模型对稀疏成员的覆盖收益，不等同于两边都有可训练模型时的纯模型增益。CHANGAN 等极稀疏策略不得单独宣称有效。",
            "",
            "所有标签均要求完整未来窗口，T+1 一字涨停不进入成熟训练样本，测试年份未用于早停或阈值选择。",
        ]
    )
    return atomic_write_text("\n".join(lines) + "\n", args.report_out)


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--labels", type=Path, default=LABEL_DATASET_PATH)
    parser.add_argument("--factor-audit", type=Path, default=FACTOR_AUDIT_PATH)
    parser.add_argument("--sample-audit", type=Path, default=SAMPLE_AUDIT_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="build fresh factors and T+1 labels")
    build.add_argument("--start-date", default=DEFAULT_START_DATE)
    build.add_argument("--end-date", default=DEFAULT_END_DATE)
    build.add_argument("--family-cache", type=Path, default=DEFAULT_FAMILY_CACHE)
    build.add_argument("--daily-partitions", type=Path, default=DEFAULT_DAILY_PARTITIONS)
    build.add_argument("--daily-basic", type=Path, default=DEFAULT_DAILY_BASIC)
    build.add_argument("--tradability", type=Path, default=DEFAULT_TRADABILITY)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument(
        "--max-pending-per-worker",
        type=int,
        default=2,
        help="有界任务队列；限制已序列化但未完成的单股任务数量。",
    )
    build.add_argument("--symbol-limit", type=int, default=None)
    build.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 10])
    build.add_argument("--entry-modes", nargs="+", choices=["next_open", "next_close"], default=["next_open", "next_close"])
    build.add_argument("--dataset-out", type=Path, default=DATASET_PATH)
    build.add_argument("--labels-out", type=Path, default=LABEL_DATASET_PATH)
    build.add_argument("--manifest-out", type=Path, default=MANIFEST_PATH)
    build.add_argument("--factor-audit-out", type=Path, default=FACTOR_AUDIT_PATH)
    build.add_argument("--sample-audit-out", type=Path, default=SAMPLE_AUDIT_PATH)

    audit = subparsers.add_parser("audit", help="fail on leakage/tradability/factor contract violations")
    _add_common_paths(audit)

    train = subparsers.add_parser("train", help="train fair independent and unified walk-forward arms")
    train.add_argument("--dataset", type=Path, default=DATASET_PATH)
    train.add_argument("--labels", type=Path, default=LABEL_DATASET_PATH)
    train.add_argument("--entry-mode", choices=["next_open", "next_close"], required=True)
    train.add_argument("--horizon", type=int, default=5)
    train.add_argument(
        "--label",
        choices=list(SUPPORTED_TRAINING_TARGETS),
        default="good_path5",
    )
    train.add_argument(
        "--folds",
        nargs="+",
        choices=[fold.name for fold in DEFAULT_YEAR_FOLDS],
        default=[fold.name for fold in DEFAULT_YEAR_FOLDS],
        help="A/B用于方案开发；方案冻结后再单独运行C作最终确认。",
    )
    train.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS),
        default=list(DEFAULT_TRAIN_EXPERIMENTS),
        help="仅替换所列实验臂在所列fold的指标和预测列；其他实验臂保持不变。",
    )
    train.add_argument("--minimum-local-rows", type=int, default=2000)
    train.add_argument("--minimum-project-feature-coverage", type=float, default=0.50)
    train.add_argument("--model-jobs", type=int, default=4)
    train.add_argument("--daily-top-k", type=int, default=10)
    train.add_argument("--round-trip-cost-bps", type=float, default=15.0)
    train.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    train.add_argument("--metrics-out", type=Path, default=METRICS_PATH)
    train.add_argument("--signal-metrics-out", type=Path, default=SIGNAL_METRICS_PATH)
    train.add_argument("--predictions-out", type=Path, default=PREDICTIONS_PATH)
    train.add_argument("--beam-report-root", type=Path, default=BEAM_REPORT_ROOT)
    train.add_argument("--beam-width", type=int, default=4)
    train.add_argument("--beam-min-features", type=int, default=6)
    train.add_argument("--beam-max-remove", type=int, default=10)
    train.add_argument(
        "--beam-prefilter-keep",
        type=int,
        default=13,
        choices=[13],
        help=(
            "deprecated compatibility option; Beam Residual v3 always searches "
            "the complete 13-factor candidate universe"
        ),
    )
    train.add_argument("--beam-permutation-rounds", type=int, default=20)
    train.add_argument("--beam-maximum-permutation-p-value", type=float, default=0.05)
    train.add_argument("--beam-search-estimators", type=int, default=120)
    train.add_argument("--beam-search-max-depth", type=int, default=4)
    train.add_argument("--beam-search-early-stopping-rounds", type=int, default=20)
    train.add_argument("--beam-history-max-rows", type=int, default=80000)
    train.add_argument("--beam-evaluation-max-rows", type=int, default=30000)

    increment = subparsers.add_parser(
        "compare-factor-increment",
        help="compare 118/Beam candidates with the 105 arm using ranking only",
    )
    increment.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    increment.add_argument("--label", default="good_path5")
    increment.add_argument("--folds", nargs="+", choices=["A", "B"], default=["A", "B"])
    increment.add_argument(
        "--candidate-experiments",
        nargs="+",
        default=["unified_long_task_deep"],
    )
    increment.add_argument(
        "--baseline-experiment",
        default="unified_long_task_deep_rule105",
    )
    increment.add_argument("--top-fraction", type=float, default=0.10)
    increment.add_argument("--daily-top-k", type=int, default=10)
    increment.add_argument("--bootstrap-iterations", type=int, default=500)
    increment.add_argument("--confidence-level", type=float, default=0.95)
    increment.add_argument("--random-seed", type=int, default=42)
    increment.add_argument("--comparison-out", type=Path, default=FACTOR_INCREMENT_PATH)
    increment.add_argument("--decision-out", type=Path, default=RANKING_DECISION_PATH)
    increment.add_argument("--beam-report-root", type=Path, default=BEAM_REPORT_ROOT)

    legacy = subparsers.add_parser(
        "compare-legacy-artifact",
        help="read-only A/B ranking comparison on exact old-artifact overlap",
    )
    legacy.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    legacy.add_argument(
        "--legacy-predictions",
        type=Path,
        default=LEGACY_ARTIFACT_PREDICTIONS_PATH,
    )
    legacy.add_argument("--candidate-experiment", default="unified_long_task_deep")
    legacy.add_argument("--folds", nargs="+", choices=["A", "B"], default=["A", "B"])
    legacy.add_argument("--top-fraction", type=float, default=0.10)
    legacy.add_argument("--daily-top-k", type=int, default=10)
    legacy.add_argument("--bootstrap-iterations", type=int, default=500)
    legacy.add_argument("--random-seed", type=int, default=42)
    legacy.add_argument("--comparison-out", type=Path, default=LEGACY_RANKING_COMPARISON_PATH)

    report = subparsers.add_parser("report", help="render a compact validation report")
    report.add_argument("--metrics", type=Path, default=METRICS_PATH)
    report.add_argument("--sample-audit", type=Path, default=SAMPLE_AUDIT_PATH)
    report.add_argument("--factor-audit", type=Path, default=FACTOR_AUDIT_PATH)
    report.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    report.add_argument(
        "--folds",
        nargs="+",
        choices=[fold.name for fold in DEFAULT_YEAR_FOLDS],
        default=[fold.name for fold in DEFAULT_YEAR_FOLDS],
        help="报告及配对比较只读取指定fold；开发选择必须显式使用 A B。",
    )
    report.add_argument("--paired-comparison-out", type=Path, default=PAIRED_COMPARISON_PATH)
    report.add_argument("--paired-bootstrap-iterations", type=int, default=500)
    report.add_argument("--paired-confidence-level", type=float, default=0.95)
    report.add_argument("--paired-random-state", type=int, default=42)
    report.add_argument("--paired-top-fraction", type=float, default=0.10)
    report.add_argument("--daily-top-k", type=int, default=10)
    report.add_argument("--report-out", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-dataset":
        result = build_dataset(args)
    elif args.command == "audit":
        result = audit_dataset(args)
    elif args.command == "train":
        result = train_models(args)
    elif args.command == "compare-factor-increment":
        result = compare_factor_increment(args)
    elif args.command == "compare-legacy-artifact":
        result = compare_legacy_artifact_ranking(args)
    elif args.command == "report":
        result = {"report": str(render_report(args))}
    else:  # pragma: no cover
        raise RuntimeError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
