#!/usr/bin/env python
"""Build, train, and compare one canonical unified left-side ranking model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.data.source_merge import normalize_tushare_daily
from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
    migrate_legacy_factor_columns,
    stable_canonical_feature_union,
)
from quant.features.left_side_factor_contract import (
    LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
    LEFT_SIDE_FACTOR_COLUMNS,
    LEFT_SIDE_FACTOR_CONTRACT_SHA256,
    LEFT_SIDE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
    LEFT_SIDE_TASK_FEATURE_COLUMNS,
    left_side_contract_payload,
)
from quant.features.project_factor_layer import (
    PROJECT_FACTOR_SCHEMA_VERSION,
    calculate_project_market_factors,
)
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.ml.xgb_research import XGBResearchModel
from quant.research.left_side_unified_features import (
    LEFT_SIDE_RULE_FEATURE_COLUMNS,
    LEFT_SIDE_RULE_FEATURE_COLUMNS_SHA256,
    LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
    LEFT_SIDE_SIGNAL_SCHEMA_VERSION,
    LEFT_SIDE_SIGNALS,
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
    compute_left_side_rule_features,
    compute_left_side_signal_flags,
    validate_left_side_factor_contract,
)
from quant.research.right_side_unified_features import compute_right_side_rule_features
from quant.research.right_side_long_task import (
    LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
    LongTaskUnifiedModel,
    aggregate_long_task_predictions,
    expand_long_task_rows,
)
from quant.research.right_side_targets import (
    materialize_training_target,
    target_metadata,
    target_source_columns,
    validate_target_cost,
)
from quant.research.right_side_unified import (
    DEFAULT_YEAR_FOLDS,
    Float32NaNTransformer,
    ProbabilityCalibratedModel,
    binary_metrics,
    daily_top_k_trading_metrics,
    split_by_year_fold,
)
from quant.research.right_side_unified_labels import build_right_side_unified_labels


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-08-22"
DEFAULT_DAILY_PARTITIONS = PROJECT_ROOT / "data/raw/daily_partitioned"
DEFAULT_TRADABILITY = PROJECT_ROOT / "data/raw/tradability"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/research/left_side_unified_v3_group4_input_parity"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research/left_side_unified_v3_group4_input_parity"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/left_side_unified_v3_group4_input_parity"
DEFAULT_DATASET = DEFAULT_DATA_ROOT / "events.parquet"
DEFAULT_LABELS = DEFAULT_DATA_ROOT / "labels.parquet"
DEFAULT_DATASET_MANIFEST = DEFAULT_DATA_ROOT / "manifest.json"
DEFAULT_METRICS = DEFAULT_REPORT_ROOT / "model_metrics.csv"
DEFAULT_PREDICTIONS = DEFAULT_REPORT_ROOT / "test_predictions.parquet"
DEFAULT_DECISION = DEFAULT_REPORT_ROOT / "ranking_replacement_decision.json"
DEFAULT_ALIAS_AUDIT = DEFAULT_REPORT_ROOT / "canonical_alias_audit.json"
MODEL_NAME = "unified_left_long_task_deep"
BASELINE_NAME = "independent_left_members"
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


class _StreamingParquetWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary, table.schema, compression="zstd"
            )
        else:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self, *, commit: bool) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if commit:
            if self.rows <= 0 or not self.temporary.exists():
                raise RuntimeError(f"left-side writer produced no rows: {self.path}")
            os.replace(self.temporary, self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _daily_paths(root: Path, start: str, end: str) -> list[Path]:
    start_month = pd.Timestamp(start).strftime("%Y%m")
    end_month = pd.Timestamp(end).strftime("%Y%m")
    paths = [
        path
        for path in sorted(root.glob("year_month=*/data.parquet"))
        if start_month <= path.parent.name.split("=", 1)[-1] <= end_month
    ]
    if not paths:
        raise FileNotFoundError(f"no daily partitions in {root} for {start}..{end}")
    return paths


def _read_market(paths: Sequence[Path], start: str, end: str) -> pd.DataFrame:
    columns = (
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "volume",
        "amount",
        "name",
    )
    start_compact = pd.Timestamp(start).strftime("%Y%m%d")
    end_compact = pd.Timestamp(end).strftime("%Y%m%d")
    frames: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(
            path, columns=[column for column in columns if column in available]
        )
        dates = frame["trade_date"].astype(str)
        frame = frame[dates.between(start_compact, end_compact)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(["ts_code", "trade_date"], kind="stable")
        .drop_duplicates(["ts_code", "trade_date"], keep="last")
    )


def _market_calendar(paths: Sequence[Path], start: str, end: str) -> pd.DatetimeIndex:
    start_compact = pd.Timestamp(start).strftime("%Y%m%d")
    end_compact = pd.Timestamp(end).strftime("%Y%m%d")
    dates = [
        pd.read_parquet(path, columns=["trade_date"])["trade_date"].astype(str)
        for path in paths
    ]
    selected = pd.concat(dates).drop_duplicates()
    selected = selected[selected.between(start_compact, end_compact)]
    return pd.DatetimeIndex(
        pd.to_datetime(selected, format="%Y%m%d", errors="raise")
    ).sort_values()


def _load_tradability(root: Path, start: str, end: str) -> pd.DataFrame:
    start_compact = pd.Timestamp(start).strftime("%Y%m%d")
    end_compact = pd.Timestamp(end).strftime("%Y%m%d")
    frames = [
        pd.read_parquet(path)
        for path in sorted(root.glob("*.parquet"))
        if start_compact <= path.stem <= end_compact
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _process_symbol(
    task: tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        pd.DatetimeIndex,
        tuple[int, ...],
        tuple[str, ...],
    ]
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    symbol, daily, tradability, start, end, calendar, horizons, entry_modes = task
    try:
        normalized = normalize_tushare_daily(daily, symbol).sort_values("date").reset_index(
            drop=True
        )
        if normalized.empty:
            return pd.DataFrame(), pd.DataFrame(), None
        rules = compute_left_side_rule_features(normalized).reset_index(drop=True)
        shared_rules = compute_right_side_rule_features(normalized).reset_index(
            drop=True
        )[list(LEFT_SIDE_SHARED_RULE_REQUIREMENTS)]
        flags = compute_left_side_signal_flags(
            normalized, rule_features=rules
        ).reset_index(drop=True)
        flags = flags[
            flags["date"].between(start, end)
            & flags[list(LEFT_SIDE_SIGNALS)].any(axis=1)
        ].copy()
        if flags.empty:
            return pd.DataFrame(), pd.DataFrame(), None
        flags["signal_count"] = flags[list(LEFT_SIDE_SIGNALS)].sum(axis=1).astype(
            "int8"
        )
        project = calculate_project_market_factors(
            normalized,
            symbol=symbol,
            factor_schema_version=PROJECT_FACTOR_SCHEMA_VERSION,
        )
        project = migrate_legacy_factor_columns(
            project,
            context=f"left-side project factors {symbol}",
            copy=False,
        )
        for column in PROJECT_FACTOR_COLUMNS:
            if column not in project:
                project[column] = pd.Series(
                    np.nan, index=project.index, dtype="float32"
                )
        base = pd.concat(
            [
                project[
                    [
                        "ts_code",
                        "symbol",
                        "trade_date",
                        "date",
                        *PROJECT_FACTOR_COLUMNS,
                        "factor_schema_version",
                    ]
                ].reset_index(drop=True),
                rules,
                shared_rules,
            ],
            axis=1,
        )
        events = base.merge(
            flags,
            on=["symbol", "date"],
            how="inner",
            validate="one_to_one",
        )
        event_columns = [
            "ts_code",
            "symbol",
            "trade_date",
            "date",
            *PROJECT_FACTOR_COLUMNS,
            *LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
            *LEFT_SIDE_RULE_FEATURE_COLUMNS,
            *LEFT_SIDE_SIGNALS,
            "signal_count",
            "factor_schema_version",
        ]
        events = events[event_columns].sort_values("date", kind="stable")
        label_daily = normalized[
            normalized["date"].between(calendar[0], calendar[-1])
        ].copy()
        labels = build_right_side_unified_labels(
            flags,
            label_daily,
            calendar,
            tradability if not tradability.empty else None,
            horizons=horizons,
            entry_modes=entry_modes,
        )
        labels = labels[list(LABEL_OUTPUT_COLUMNS)].sort_values(
            ["date", "entry_mode", "horizon"], kind="stable"
        )
        return events.reset_index(drop=True), labels.reset_index(drop=True), None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), f"{symbol}: {exc}"


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    history_start = (
        pd.Timestamp(args.start_date) - pd.DateOffset(years=6)
    ).strftime("%Y-%m-%d")
    label_end = (pd.Timestamp(args.end_date) + pd.Timedelta(days=40)).strftime(
        "%Y-%m-%d"
    )
    paths = _daily_paths(args.daily_partitions, history_start, label_end)
    calendar = _market_calendar(paths, args.start_date, label_end)
    market = _read_market(paths, history_start, label_end)
    if market.empty:
        raise RuntimeError("left-side dataset found no market rows")
    market["ts_code"] = market["ts_code"].astype(str)
    tradability = _load_tradability(args.tradability, args.start_date, label_end)
    by_tradability = (
        {
            str(symbol): group.reset_index(drop=True)
            for symbol, group in tradability.groupby("ts_code", sort=False)
        }
        if not tradability.empty
        else {}
    )
    symbols = sorted(market["ts_code"].unique())
    if args.symbol_limit:
        symbols = symbols[: args.symbol_limit]
        market = market[market["ts_code"].isin(symbols)]

    def tasks() -> Iterator[tuple[object, ...]]:
        for symbol, frame in market.groupby("ts_code", sort=True):
            yield (
                str(symbol),
                frame.reset_index(drop=True),
                by_tradability.get(str(symbol), pd.DataFrame()),
                pd.Timestamp(args.start_date),
                pd.Timestamp(args.end_date),
                calendar,
                tuple(args.horizons),
                tuple(args.entry_modes),
            )

    event_writer = _StreamingParquetWriter(args.dataset_out)
    label_writer = _StreamingParquetWriter(args.labels_out)
    errors: list[str] = []
    signal_counts: Counter[str] = Counter()
    locked_count = 0

    def consume(result: tuple[pd.DataFrame, pd.DataFrame, str | None]) -> None:
        nonlocal locked_count
        events, labels, error = result
        if error:
            errors.append(error)
            return
        if events.empty:
            return
        assert_no_forbidden_factor_names(
            events.columns, context="left-side training dataset"
        )
        validate_left_side_factor_contract(events.columns)
        event_writer.write(events)
        label_writer.write(labels)
        for signal in LEFT_SIDE_SIGNALS:
            signal_counts[signal] += int(events[signal].sum())
        locked_count += int(
            labels.drop_duplicates(["symbol", "date"])["locked_limit_up"].sum()
        )

    iterator = iter(tasks())
    completed = 0
    try:
        if args.workers <= 1:
            for task in iterator:
                consume(_process_symbol(task))
                completed += 1
        else:
            max_pending = max(args.workers, args.workers * 2)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                pending = set()
                for _ in range(min(max_pending, len(symbols))):
                    try:
                        pending.add(executor.submit(_process_symbol, next(iterator)))
                    except StopIteration:
                        break
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume(future.result())
                        completed += 1
                        try:
                            pending.add(executor.submit(_process_symbol, next(iterator)))
                        except StopIteration:
                            pass
                        if completed % 250 == 0:
                            print(f"left features {completed:,}/{len(symbols):,}", flush=True)
        if errors:
            raise RuntimeError(
                f"left-side dataset failed for {len(errors)} symbols: {errors[:5]}"
            )
        event_writer.close(commit=True)
        label_writer.close(commit=True)
    except BaseException:
        event_writer.close(commit=False)
        label_writer.close(commit=False)
        raise

    manifest = {
        "status": "success",
        "schema_version": "left-side-unified-dataset-v3-group4-input-parity",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "event_rows": event_writer.rows,
        "label_rows": label_writer.rows,
        "signals": list(LEFT_SIDE_SIGNALS),
        "signal_counts": dict(signal_counts),
        "locked_limit_events": locked_count,
        "project_factor_schema_version": PROJECT_FACTOR_SCHEMA_VERSION,
        "project_factor_count": len(PROJECT_FACTOR_COLUMNS),
        "project_factor_columns": list(PROJECT_FACTOR_COLUMNS),
        "rule_factor_schema_version": LEFT_SIDE_RULE_FEATURE_SCHEMA_VERSION,
        "shared_right_rule_factor_count": len(LEFT_SIDE_SHARED_RULE_REQUIREMENTS),
        "shared_right_rule_factors": list(LEFT_SIDE_SHARED_RULE_REQUIREMENTS),
        "rule_factor_count": len(LEFT_SIDE_RULE_FEATURE_COLUMNS),
        "rule_factor_columns": list(LEFT_SIDE_RULE_FEATURE_COLUMNS),
        "rule_factor_columns_sha256": LEFT_SIDE_RULE_FEATURE_COLUMNS_SHA256,
        "factor_count": len(LEFT_SIDE_FACTOR_COLUMNS),
        "factor_columns": list(LEFT_SIDE_FACTOR_COLUMNS),
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "model_input_count": len(LEFT_SIDE_FACTOR_COLUMNS)
        + len(LEFT_SIDE_TASK_FEATURE_COLUMNS),
        "model_input_contract_sha256": LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
        "signal_schema_version": LEFT_SIDE_SIGNAL_SCHEMA_VERSION,
        "forbidden_aliases": [],
        "event_dataset": str(args.dataset_out),
        "event_dataset_sha256": _sha256(args.dataset_out),
        "label_dataset": str(args.labels_out),
        "label_dataset_sha256": _sha256(args.labels_out),
    }
    if find_forbidden_aliases_in_payload(manifest):
        raise RuntimeError("left-side dataset manifest contains forbidden aliases")
    atomic_write_json(manifest, args.manifest_out)
    return manifest


def _split_validation_stages(
    validation: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(validation["date"].unique()).sort_values()
    if len(dates) < 60:
        raise ValueError("left-side validation requires at least 60 dates")
    first = dates[max(1, int(len(dates) * 0.50)) - 1]
    second = dates[max(2, int(len(dates) * 0.75)) - 1]
    stages = (
        validation[validation["date"].le(first)].copy(),
        validation[
            validation["date"].gt(first) & validation["date"].le(second)
        ].copy(),
        validation[validation["date"].gt(second)].copy(),
    )
    if any(frame.empty or frame[label].nunique() < 2 for frame in stages):
        raise ValueError("left-side validation stage lacks both classes")
    return stages


def _fit_base_model(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    n_jobs: int,
) -> ProbabilityCalibratedModel:
    transformer = Float32NaNTransformer()
    train_x = transformer.fit_transform(train[list(features)])
    early_x = transformer.transform(early_stop[list(features)])
    classifier = XGBClassifier(
        **LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC.classifier_kwargs(n_jobs=n_jobs)
    )
    classifier.fit(
        train_x,
        train[label].astype(int),
        eval_set=[(early_x, early_stop[label].astype(int))],
        verbose=False,
    )
    best_iteration = (
        int(classifier.best_iteration)
        if classifier.best_iteration is not None
        else None
    )
    base = XGBResearchModel(
        feature_names_in_=list(features),
        selected_features_=list(features),
        imputer=transformer,
        selector=None,
        classifier=classifier,
        best_iteration=best_iteration,
        factor_schema_version_=LEFT_SIDE_FEATURE_SCHEMA_VERSION,
    )
    raw = np.asarray(base.predict_proba(calibration[list(features)]), dtype=float)[:, 1]
    raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
    logits = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(logits, calibration[label].astype(int))
    return ProbabilityCalibratedModel(base_model=base, calibrator=calibrator)


def _fit_unified_model(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    n_jobs: int,
) -> LongTaskUnifiedModel:
    retained = [*features, label]
    train_long = expand_long_task_rows(
        train,
        retained_columns=retained,
        signal_columns=LEFT_SIDE_SIGNALS,
        task_feature_columns=LEFT_SIDE_TASK_FEATURE_COLUMNS,
    )
    early_long = expand_long_task_rows(
        early_stop,
        retained_columns=retained,
        signal_columns=LEFT_SIDE_SIGNALS,
        task_feature_columns=LEFT_SIDE_TASK_FEATURE_COLUMNS,
    )
    model_features = [*features, *LEFT_SIDE_TASK_FEATURE_COLUMNS]
    transformer = Float32NaNTransformer()
    train_x = transformer.fit_transform(train_long[model_features])
    early_x = transformer.transform(early_long[model_features])
    classifier = XGBClassifier(
        **LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC.classifier_kwargs(n_jobs=n_jobs)
    )
    classifier.fit(
        train_x,
        train_long[label].astype(int),
        eval_set=[(early_x, early_long[label].astype(int))],
        verbose=False,
    )
    best_iteration = (
        int(classifier.best_iteration)
        if classifier.best_iteration is not None
        else None
    )
    base = XGBResearchModel(
        feature_names_in_=model_features,
        selected_features_=model_features,
        imputer=transformer,
        selector=None,
        classifier=classifier,
        best_iteration=best_iteration,
        factor_schema_version_=LEFT_SIDE_FEATURE_SCHEMA_VERSION,
    )
    calibration_long = expand_long_task_rows(
        calibration,
        retained_columns=features,
        signal_columns=LEFT_SIDE_SIGNALS,
        task_feature_columns=LEFT_SIDE_TASK_FEATURE_COLUMNS,
    )
    task_probability = np.asarray(
        base.predict_proba(calibration_long[model_features]), dtype=float
    )[:, 1]
    event_probability = aggregate_long_task_predictions(
        calibration_long, task_probability, event_count=len(calibration)
    )
    valid = np.isfinite(event_probability)
    clipped = np.clip(event_probability[valid], 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(logits, calibration.loc[valid, label].astype(int))
    return LongTaskUnifiedModel(
        base_model=base,
        event_calibrator=calibrator,
        common_features=tuple(features),
        signal_columns=LEFT_SIDE_SIGNALS,
        task_feature_columns=LEFT_SIDE_TASK_FEATURE_COLUMNS,
    )


def _fit_independent_members(
    train: pd.DataFrame,
    early_stop: pd.DataFrame,
    calibration: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    n_jobs: int,
) -> dict[str, ProbabilityCalibratedModel]:
    models: dict[str, ProbabilityCalibratedModel] = {}
    for signal in LEFT_SIDE_SIGNALS:
        train_local = train[train[signal]].copy()
        early_local = early_stop[early_stop[signal]].copy()
        calibration_local = calibration[calibration[signal]].copy()
        if any(
            len(frame) < 100 or frame[label].nunique() < 2
            for frame in (train_local, early_local, calibration_local)
        ):
            raise RuntimeError(f"left independent member has insufficient rows: {signal}")
        models[signal] = _fit_base_model(
            train_local,
            early_local,
            calibration_local,
            features,
            label,
            n_jobs=n_jobs,
        )
    return models


def _predict_independent(
    frame: pd.DataFrame,
    models: dict[str, ProbabilityCalibratedModel],
) -> np.ndarray:
    probability = np.full(len(frame), np.nan, dtype=float)
    for signal, model in models.items():
        mask = frame[signal].fillna(False).to_numpy(bool)
        if not mask.any():
            continue
        local = np.asarray(model.predict_proba(frame.loc[mask]), dtype=float)[:, 1]
        positions = np.flatnonzero(mask)
        current = probability[positions]
        probability[positions] = np.where(
            np.isfinite(current), np.maximum(current, local), local
        )
    return probability


def _ranking_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    label: str,
    *,
    top_k: int,
    cost_bps: float,
) -> dict[str, Any]:
    metrics = binary_metrics(frame[label], probability, threshold=0.5)
    return {
        **metrics,
        **daily_top_k_trading_metrics(
            frame,
            probability,
            top_k=top_k,
            round_trip_cost_bps=cost_bps,
        ),
    }


def train_and_compare(args: argparse.Namespace) -> dict[str, Any]:
    validate_target_cost(args.label, args.round_trip_cost_bps)
    factor_columns = stable_canonical_feature_union(
        PROJECT_FACTOR_COLUMNS,
        LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
        LEFT_SIDE_RULE_FEATURE_COLUMNS,
    )
    if factor_columns != LEFT_SIDE_FACTOR_COLUMNS:
        raise RuntimeError(
            "left-side training union drifted from the versioned factor contract"
        )
    assert_no_forbidden_factor_names(
        factor_columns, context="left-side training input"
    )
    event_columns = [
        "symbol",
        "date",
        *factor_columns,
        *LEFT_SIDE_SIGNALS,
        "signal_count",
        "factor_schema_version",
    ]
    events = pd.read_parquet(args.dataset, columns=event_columns)
    events = migrate_legacy_factor_columns(
        events, context="left-side event dataset", copy=False
    )
    events["date"] = pd.to_datetime(events["date"], errors="raise")
    schemas = set(events["factor_schema_version"].dropna().astype(str))
    if schemas != {PROJECT_FACTOR_SCHEMA_VERSION}:
        raise RuntimeError(
            "left-side dataset project schema mismatch: "
            f"expected={PROJECT_FACTOR_SCHEMA_VERSION} actual={sorted(schemas)}"
        )
    label_columns = list(
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
    labels = pd.read_parquet(
        args.labels,
        columns=label_columns,
        filters=[
            ("entry_mode", "==", args.entry_mode),
            ("horizon", "==", args.horizon),
        ],
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="raise")
    labels["label_end_date"] = pd.to_datetime(
        labels["label_end_date"], errors="raise"
    )
    labels = materialize_training_target(labels, args.label)
    labels = labels[
        labels["mature"]
        & labels[args.label].notna()
        & ~labels["locked_limit_up"]
    ]
    selected = events.merge(
        labels, on=["symbol", "date"], how="inner", validate="one_to_one"
    ).sort_values(["date", "symbol"], kind="stable")
    selected[args.label] = selected[args.label].astype(int)
    del events, labels
    gc.collect()

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fold_models: dict[str, Path] = {}
    fold_by_name = {fold.name: fold for fold in DEFAULT_YEAR_FOLDS}
    for fold_name in args.folds:
        fold = fold_by_name[fold_name]
        splits = split_by_year_fold(selected, fold)
        early, calibration, _threshold = _split_validation_stages(
            splits.validation, args.label
        )
        independent = _fit_independent_members(
            splits.train,
            early,
            calibration,
            factor_columns,
            args.label,
            n_jobs=args.model_jobs,
        )
        unified = _fit_unified_model(
            splits.train,
            early,
            calibration,
            factor_columns,
            args.label,
            n_jobs=args.model_jobs,
        )
        independent_probability = _predict_independent(splits.test, independent)
        unified_probability = np.asarray(
            unified.predict_proba(splits.test), dtype=float
        )[:, 1]
        if not np.isfinite(independent_probability).all() or not np.isfinite(
            unified_probability
        ).all():
            raise RuntimeError("left-side comparison produced missing probabilities")
        for experiment, probability in (
            (BASELINE_NAME, independent_probability),
            (MODEL_NAME, unified_probability),
        ):
            metrics.append(
                {
                    "fold": fold.name,
                    "entry_mode": args.entry_mode,
                    "horizon": args.horizon,
                    "label": args.label,
                    "experiment": experiment,
                    "rows": len(splits.test),
                    **_ranking_metrics(
                        splits.test,
                        probability,
                        args.label,
                        top_k=args.daily_top_k,
                        cost_bps=args.round_trip_cost_bps,
                    ),
                }
            )
        prediction = splits.test[
            ["symbol", "date", args.label, *LEFT_SIDE_SIGNALS]
        ].copy()
        prediction.insert(0, "fold", fold.name)
        prediction["entry_mode"] = args.entry_mode
        prediction["horizon"] = args.horizon
        prediction["label"] = args.label
        prediction[f"pred_{BASELINE_NAME}"] = independent_probability
        prediction[f"pred_{MODEL_NAME}"] = unified_probability
        predictions.append(prediction)

        fold_root = (
            args.model_root
            / args.entry_mode
            / f"h{args.horizon}"
            / args.label
            / fold.name
        )
        fold_root.mkdir(parents=True, exist_ok=True)
        model_path = fold_root / f"{MODEL_NAME}.joblib"
        joblib.dump(unified, model_path)
        fold_models[fold.name] = model_path
        model_features = tuple(unified.feature_names_in_)
        assert_no_forbidden_factor_names(
            model_features, context=f"left-side fold {fold.name} artifact"
        )
        manifest = {
            "status": "success",
            "schema_version": LEFT_SIDE_ARTIFACT_SCHEMA_VERSION,
            "lifecycle": "research_only",
            "fold": asdict(fold),
            "entry_mode": args.entry_mode,
            "horizon": args.horizon,
            "target": {"name": args.label, **target_metadata(args.label)},
            "signals": list(LEFT_SIDE_SIGNALS),
            "features": list(model_features),
            "feature_names_in": list(model_features),
            "selected_feature_columns": list(model_features),
            "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
            "model_input_contract_sha256": LEFT_SIDE_MODEL_INPUT_CONTRACT_SHA256,
            **left_side_contract_payload(),
            "artifact": str(model_path),
            "artifact_sha256": _sha256(model_path),
            "forbidden_aliases": [],
        }
        if find_forbidden_aliases_in_payload(manifest):
            raise RuntimeError("left-side model manifest contains forbidden aliases")
        atomic_write_json(manifest, model_path.with_suffix(".manifest.json"))
        for signal, model in independent.items():
            legacy_comparison_path = fold_root / "independent" / f"{signal}.joblib"
            legacy_comparison_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, legacy_comparison_path)

    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    atomic_write_csv(metric_frame, args.metrics_out, index=False)
    atomic_write_parquet(prediction_frame, args.predictions_out, index=False)
    pivot = metric_frame.pivot(
        index="fold",
        columns="experiment",
        values="average_precision",
    )
    fold_deltas = {
        fold: float(row[MODEL_NAME] - row[BASELINE_NAME])
        for fold, row in pivot.iterrows()
    }
    # A is the development comparison.  B is the selection window and C is
    # the untouched confirmation window.  A production decision is only
    # complete when both B and C are present and the unified ranker is not
    # worse on either window, matching the project's ranking-only gate.
    decision_folds = [fold for fold in ("B", "C") if fold in fold_deltas]
    replace_online = decision_folds == ["B", "C"] and all(
        fold_deltas[fold] >= 0.0 for fold in decision_folds
    )
    decision = {
        "status": "success",
        "schema_version": "left-side-ranking-replacement-decision-v2-group4",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate": MODEL_NAME,
        "baseline": BASELINE_NAME,
        "primary_metric": "average_precision",
        "decision_folds": decision_folds,
        "fold_ap_delta": fold_deltas,
        "replace_online": replace_online,
        "replacement_rule": "candidate_ap_not_lower_on_selection_B_and_confirmation_C",
        "development_fold": "A",
        "selection_fold": "B",
        "confirmation_fold": "C",
        "sample_contract": "same_event_keys_same_target_same_fold",
        "locked_limit_up_excluded": True,
        "entry_mode": args.entry_mode,
        "horizon": args.horizon,
        "label": args.label,
        "production_candidate_fold": "C",
        "production_candidate_artifact": str(fold_models.get("C", "")),
        "factor_contract_sha256": LEFT_SIDE_FACTOR_CONTRACT_SHA256,
        "forbidden_aliases": [],
    }
    if find_forbidden_aliases_in_payload(decision):
        raise RuntimeError("left-side ranking decision contains forbidden aliases")
    atomic_write_json(decision, args.decision_out)
    return decision


def audit_aliases(args: argparse.Namespace) -> dict[str, Any]:
    schema = set(pq.ParquetFile(args.dataset).schema.names)
    pairs = {
        "price_level": "close",
        "bb_middle": "ma_20",
        "rs_pct_chg_1d": "pct_chg",
        "rs_amplitude_pct": "amplitude_1",
        "rs_vol_ratio_5_inclusive": "volume_relative_5d",
    }
    checked: dict[str, str] = {}
    columns: list[str] = []
    for alias, canonical in pairs.items():
        if alias in schema:
            columns.append(alias)
            if canonical in schema:
                columns.append(canonical)
    if columns:
        frame = pd.read_parquet(args.dataset, columns=list(dict.fromkeys(columns)))
        migrate_legacy_factor_columns(
            frame, context="left-side historical alias audit", copy=False
        )
    for alias, canonical in pairs.items():
        if alias not in schema:
            checked[alias] = "absent"
        elif canonical in schema:
            checked[alias] = "identical_and_migratable"
        else:
            checked[alias] = "alias_only_migratable"
    payload = {
        "status": "success",
        "schema_version": "canonical-alias-boundary-audit-v1",
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256(args.dataset),
        "checked": checked,
        "training_frame_forbidden_intersection": [],
    }
    atomic_write_json(payload, args.alias_audit_out)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--start-date", default=DEFAULT_START_DATE)
    build.add_argument("--end-date", default=DEFAULT_END_DATE)
    build.add_argument("--daily-partitions", type=Path, default=DEFAULT_DAILY_PARTITIONS)
    build.add_argument("--tradability", type=Path, default=DEFAULT_TRADABILITY)
    build.add_argument("--dataset-out", type=Path, default=DEFAULT_DATASET)
    build.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS)
    build.add_argument("--manifest-out", type=Path, default=DEFAULT_DATASET_MANIFEST)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--symbol-limit", type=int, default=None)
    build.add_argument("--horizons", type=int, nargs="+", default=[3, 5, 7])
    build.add_argument(
        "--entry-modes", nargs="+", default=["next_open", "next_close"]
    )

    train = subparsers.add_parser("train")
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    train.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    train.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    train.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS)
    train.add_argument("--decision-out", type=Path, default=DEFAULT_DECISION)
    train.add_argument("--entry-mode", choices=["next_open", "next_close"], default="next_close")
    train.add_argument("--horizon", type=int, default=5)
    train.add_argument("--label", default="good_path5")
    train.add_argument("--folds", nargs="+", default=["A", "B", "C"])
    train.add_argument("--model-jobs", type=int, default=4)
    train.add_argument("--daily-top-k", type=int, default=10)
    train.add_argument("--round-trip-cost-bps", type=float, default=20.0)

    audit = subparsers.add_parser("audit-aliases")
    audit.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    audit.add_argument("--alias-audit-out", type=Path, default=DEFAULT_ALIAS_AUDIT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        result = build_dataset(args)
    elif args.command == "train":
        result = train_and_compare(args)
    else:
        result = audit_aliases(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
