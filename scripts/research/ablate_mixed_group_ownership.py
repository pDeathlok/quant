#!/usr/bin/env python
"""Compare whether mixed short strategies belong with right, left, or alone."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json, atomic_write_parquet
from quant.features.canonical_factor_names import (
    assert_no_forbidden_factor_names,
    find_forbidden_aliases_in_payload,
    stable_canonical_feature_union,
)
from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.right_side_factor_contract import factor_contract_sha256
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS
from quant.ml.xgb_research import XGBResearchModel
from quant.research.left_side_unified_features import (
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
)
from quant.research.right_side_long_task import (
    LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC,
    LongTaskUnifiedModel,
    aggregate_long_task_predictions,
    expand_long_task_rows,
)
from quant.research.right_side_unified import (
    DEFAULT_YEAR_FOLDS,
    Float32NaNTransformer,
    binary_metrics,
    daily_top_k_trading_metrics,
    split_by_year_fold,
)
from quant.research.short_side_groups import (
    ALL_SHORT_GROUPS,
    GROUP_MEMBERS,
    LEFT_GROUPS,
    LEFT_WITH_MIXED_GROUPS,
    MIXED_GROUPS,
    RIGHT_GROUPS,
    RIGHT_WITH_MIXED_GROUPS,
)


DEFAULT_RIGHT_EVENTS = (
    PROJECT_ROOT
    / "data/research/right_side_unified_canonical_v5_rule113/unified_right_side_dataset.parquet"
)
DEFAULT_RIGHT_LABELS = (
    PROJECT_ROOT
    / "data/research/right_side_unified_canonical_v5_rule113/unified_right_side_labels.parquet"
)
DEFAULT_LEFT_EVENTS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
)
DEFAULT_LEFT_LABELS = (
    PROJECT_ROOT
    / "data/research/left_side_unified_v3_group4_input_parity/labels.parquet"
)
DEFAULT_ROOT = PROJECT_ROOT / "data/research/short_side_mixed_ownership_ablation"
DEFAULT_MASTER = DEFAULT_ROOT / "events_next_close_h5_good_path5.parquet"
DEFAULT_MANIFEST = DEFAULT_ROOT / "dataset_manifest.json"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models/research/short_side_mixed_ownership_ablation"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "reports/research/short_side_mixed_ownership_ablation"

FACTOR_COLUMNS: tuple[str, ...] = stable_canonical_feature_union(
    PROJECT_FACTOR_COLUMNS,
    LEFT_SIDE_SHARED_RULE_REQUIREMENTS,
)
SCHEMA_VERSION = "short-side-mixed-ownership-ablation-v2-common-canonical"
MODEL_SCHEMA_VERSION = "short-side-group-long-task-common-v2"

POOL_GROUPS: dict[str, tuple[str, ...]] = {
    "right": RIGHT_GROUPS,
    "left": LEFT_GROUPS,
    "mixed": MIXED_GROUPS,
    "right_with_mixed": RIGHT_WITH_MIXED_GROUPS,
    "left_with_mixed": LEFT_WITH_MIXED_GROUPS,
}
ARCHITECTURES: dict[str, tuple[str, ...]] = {
    "mixed_with_right": ("right_with_mixed", "left"),
    "mixed_with_left": ("right", "left_with_mixed"),
    "mixed_independent": ("right", "left", "mixed"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _right_group_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for group in RIGHT_WITH_MIXED_GROUPS:
        members = GROUP_MEMBERS[group]
        expressions.append(
            pl.any_horizontal([pl.col(member).fill_null(False) for member in members])
            .cast(pl.Boolean)
            .alias(group)
        )
    return expressions


def _standardized_side(
    *,
    events_path: Path,
    labels_path: Path,
    side: str,
) -> pl.LazyFrame:
    events = pl.scan_parquet(events_path)
    labels = (
        pl.scan_parquet(labels_path)
        .filter(
            (pl.col("entry_mode") == "next_close")
            & (pl.col("horizon") == 5)
            & pl.col("mature").fill_null(False)
            & pl.col("entry_executable").fill_null(False)
            & ~pl.col("locked_limit_up").fill_null(True)
            & pl.col("good_path5").is_not_null()
        )
        .select(
            "symbol",
            "date",
            "label_end_date",
            pl.col("good_path5").cast(pl.Int8),
            pl.col("terminal_return").cast(pl.Float32),
            pl.col("mfe").cast(pl.Float32),
            pl.col("mae").cast(pl.Float32),
        )
    )
    if side == "right":
        events = events.with_columns(_right_group_expressions())
        available_factors = set(pl.scan_parquet(events_path).collect_schema().names())
        group_columns = RIGHT_WITH_MIXED_GROUPS
    elif side == "left":
        available_factors = set(pl.scan_parquet(events_path).collect_schema().names())
        group_columns = LEFT_GROUPS
    else:
        raise ValueError(f"unknown side: {side}")

    factor_exprs = [
        (
            pl.col(column).cast(pl.Float32)
            if column in available_factors
            else pl.lit(None, dtype=pl.Float32)
        ).alias(column)
        for column in FACTOR_COLUMNS
    ]
    group_exprs = [
        (
            pl.col(group).fill_null(False).cast(pl.Boolean)
            if group in group_columns
            else pl.lit(False, dtype=pl.Boolean)
        ).alias(group)
        for group in ALL_SHORT_GROUPS
    ]
    standardized = events.select(
        pl.col("symbol").cast(pl.String),
        pl.col("date").cast(pl.Datetime("us")),
        *factor_exprs,
        *group_exprs,
    )
    return standardized.join(
        labels.with_columns(pl.col("date").cast(pl.Datetime("us"))),
        on=["symbol", "date"],
        how="inner",
        validate="1:1",
    )


def _overlap_factor_conflict_count(
    right: pl.LazyFrame,
    left: pl.LazyFrame,
) -> int:
    overlap = right.select("symbol", "date", *FACTOR_COLUMNS).join(
        left.select("symbol", "date", *FACTOR_COLUMNS),
        on=["symbol", "date"],
        how="inner",
        validate="1:1",
        suffix="_left",
    )
    mismatches: list[pl.Expr] = []
    for column in FACTOR_COLUMNS:
        right_value = pl.col(column)
        left_value = pl.col(f"{column}_left")
        right_missing = right_value.is_null() | right_value.is_nan()
        left_missing = left_value.is_null() | left_value.is_nan()
        mismatches.append(
            (right_missing != left_missing)
            | (~right_missing & ~left_missing & (right_value != left_value))
        )
    return int(
        overlap.select(pl.any_horizontal(mismatches).sum())
        .collect(engine="streaming")
        .item()
    )


def _overlap_label_conflict_count(
    right: pl.LazyFrame,
    left: pl.LazyFrame,
) -> int:
    label_columns = (
        "label_end_date",
        "good_path5",
        "terminal_return",
        "mfe",
        "mae",
    )
    overlap = right.select("symbol", "date", *label_columns).join(
        left.select("symbol", "date", *label_columns),
        on=["symbol", "date"],
        how="inner",
        validate="1:1",
        suffix="_left",
    )
    mismatches: list[pl.Expr] = [
        pl.col("label_end_date") != pl.col("label_end_date_left"),
        pl.col("good_path5") != pl.col("good_path5_left"),
    ]
    for column in ("terminal_return", "mfe", "mae"):
        right_value = pl.col(column)
        left_value = pl.col(f"{column}_left")
        right_missing = right_value.is_null() | right_value.is_nan()
        left_missing = left_value.is_null() | left_value.is_nan()
        mismatches.append(
            (right_missing != left_missing)
            | (~right_missing & ~left_missing & (right_value != left_value))
        )
    return int(
        overlap.select(pl.any_horizontal(mismatches).sum())
        .collect(engine="streaming")
        .item()
    )


def build_master(args: argparse.Namespace) -> dict[str, Any]:
    assert_no_forbidden_factor_names(FACTOR_COLUMNS, context="ownership ablation factors")
    right = _standardized_side(
        events_path=args.right_events,
        labels_path=args.right_labels,
        side="right",
    )
    left = _standardized_side(
        events_path=args.left_events,
        labels_path=args.left_labels,
        side="left",
    )
    if args.start_date is not None:
        start = pd.Timestamp(args.start_date)
        right = right.filter(pl.col("date") >= start)
        left = left.filter(pl.col("date") >= start)
    if args.end_date is not None:
        end = pd.Timestamp(args.end_date)
        right = right.filter(pl.col("date") <= end)
        left = left.filter(pl.col("date") <= end)
    factor_conflict_count = _overlap_factor_conflict_count(right, left)
    if factor_conflict_count:
        raise RuntimeError(
            "right/left common factor builders disagree on "
            f"{factor_conflict_count} overlapping events"
        )
    label_conflict_count = _overlap_label_conflict_count(right, left)
    if label_conflict_count:
        raise RuntimeError(
            "right/left label builders disagree on "
            f"{label_conflict_count} overlapping events"
        )
    keys = ("symbol", "date")
    left_group_flags = left.select(*keys, *LEFT_GROUPS)
    right_with_left_groups = right.join(
        left_group_flags,
        on=list(keys),
        how="left",
        validate="1:1",
        suffix="_from_left",
    ).with_columns(
        *[
            pl.col(f"{group}_from_left").fill_null(False).alias(group)
            for group in LEFT_GROUPS
        ]
    ).drop(*[f"{group}_from_left" for group in LEFT_GROUPS])
    left_only = left.join(
        right.select(*keys),
        on=list(keys),
        how="anti",
    )
    master = pl.concat([right_with_left_groups, left_only], how="vertical_relaxed")
    args.master.parent.mkdir(parents=True, exist_ok=True)
    master.sink_parquet(
        args.master,
        compression="zstd",
        statistics=True,
        engine="streaming",
        mkdir=True,
    )
    audit = pl.scan_parquet(args.master).select(
        pl.len().alias("rows"),
        pl.col("date").min().alias("date_min"),
        pl.col("date").max().alias("date_max"),
        *[pl.col(group).sum().alias(group) for group in ALL_SHORT_GROUPS],
    ).collect()
    row = audit.row(0, named=True)
    payload = {
        "status": "success",
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "requested_start_date": args.start_date,
        "requested_end_date": args.end_date,
        "sample_contract": "union_of_canonical_group_events_same_t1_close_h5_good_path5",
        "entry_mode": "next_close",
        "horizon": 5,
        "label": "good_path5",
        "entry_executable_required": True,
        "locked_limit_up_excluded": True,
        "rows": int(row["rows"]),
        "date_min": str(row["date_min"]),
        "date_max": str(row["date_max"]),
        "group_counts": {group: int(row[group]) for group in ALL_SHORT_GROUPS},
        "factor_count": len(FACTOR_COLUMNS),
        "overlap_factor_conflict_count": factor_conflict_count,
        "overlap_label_conflict_count": label_conflict_count,
        "factor_columns": list(FACTOR_COLUMNS),
        "factor_contract_sha256": factor_contract_sha256(
            FACTOR_COLUMNS, schema_version=SCHEMA_VERSION
        ),
        "forbidden_aliases": [],
        "master": str(args.master),
        "master_sha256": _sha256(args.master),
    }
    if find_forbidden_aliases_in_payload(payload):
        raise RuntimeError("ownership ablation manifest contains forbidden aliases")
    atomic_write_json(payload, args.manifest)
    return payload


def _validation_stages(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(validation["date"].unique()).sort_values()
    first = dates[max(1, int(len(dates) * 0.50)) - 1]
    second = dates[max(2, int(len(dates) * 0.75)) - 1]
    early = validation[validation["date"].le(first)].copy()
    calibration = validation[
        validation["date"].gt(first) & validation["date"].le(second)
    ].copy()
    if any(frame.empty or frame["good_path5"].nunique() < 2 for frame in (early, calibration)):
        raise RuntimeError("ownership ablation validation stages lack both labels")
    return early, calibration


def _pool_frame(frame: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    return frame[frame[list(groups)].fillna(False).astype(bool).any(axis=1)].copy()


def _usable_features(train: pd.DataFrame) -> tuple[str, ...]:
    values = train[list(FACTOR_COLUMNS)]
    counts = values.count()
    minimum = values.min(numeric_only=True)
    maximum = values.max(numeric_only=True)
    usable = tuple(
        column
        for column in FACTOR_COLUMNS
        if int(counts.get(column, 0)) > 0
        and minimum.get(column) != maximum.get(column)
    )
    assert_no_forbidden_factor_names(usable, context="ownership ablation usable factors")
    return usable


def _fit_pool_model(
    train: pd.DataFrame,
    early: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    groups: Sequence[str],
    features: Sequence[str],
    model_jobs: int,
) -> LongTaskUnifiedModel:
    task_features = tuple(f"task_{group}" for group in groups)
    retained = [*features, "good_path5"]
    train_long = expand_long_task_rows(
        train,
        retained_columns=retained,
        signal_columns=groups,
        task_feature_columns=task_features,
    )
    early_long = expand_long_task_rows(
        early,
        retained_columns=retained,
        signal_columns=groups,
        task_feature_columns=task_features,
    )
    model_features = [*features, *task_features]
    transformer = Float32NaNTransformer()
    train_x = transformer.fit_transform(train_long[model_features])
    early_x = transformer.transform(early_long[model_features])
    classifier_spec = LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC
    classifier = XGBClassifier(
        **classifier_spec.classifier_kwargs(n_jobs=model_jobs)
    )
    classifier.fit(
        train_x,
        train_long["good_path5"].astype(int),
        eval_set=[(early_x, early_long["good_path5"].astype(int))],
        verbose=False,
    )
    base = XGBResearchModel(
        feature_names_in_=model_features,
        selected_features_=model_features,
        imputer=transformer,
        selector=None,
        classifier=classifier,
        best_iteration=(
            int(classifier.best_iteration)
            if classifier.best_iteration is not None
            else None
        ),
        factor_schema_version_=MODEL_SCHEMA_VERSION,
    )
    calibration_long = expand_long_task_rows(
        calibration,
        retained_columns=features,
        signal_columns=groups,
        task_feature_columns=task_features,
    )
    task_probability = np.asarray(
        base.predict_proba(calibration_long[model_features]), dtype=float
    )[:, 1]
    event_probability = aggregate_long_task_predictions(
        calibration_long,
        task_probability,
        event_count=len(calibration),
    )
    valid = np.isfinite(event_probability)
    clipped = np.clip(event_probability[valid], 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=42)
    calibrator.fit(logits, calibration.loc[valid, "good_path5"].astype(int))
    return LongTaskUnifiedModel(
        base_model=base,
        event_calibrator=calibrator,
        common_features=tuple(features),
        signal_columns=tuple(groups),
        task_feature_columns=task_features,
    )


def _predict_pool(
    frame: pd.DataFrame,
    model: LongTaskUnifiedModel,
    groups: Sequence[str],
) -> np.ndarray:
    mask = frame[list(groups)].fillna(False).astype(bool).any(axis=1).to_numpy()
    output = np.full(len(frame), np.nan, dtype=float)
    if mask.any():
        output[mask] = np.asarray(
            model.predict_proba(frame.loc[mask]), dtype=float
        )[:, 1]
    return output


def _combine_predictions(columns: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(columns)
    if np.isnan(matrix).all(axis=1).any():
        raise RuntimeError("ownership architecture left candidate events unscored")
    return np.nanmax(matrix, axis=1)


def _daily_ndcg_at_k(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    k: int = 10,
) -> float:
    working = frame[["date", "good_path5"]].copy()
    working["probability"] = probability
    values: list[float] = []
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    for _, day in working.groupby("date", sort=False):
        ranked = day.sort_values("probability", ascending=False).head(k)
        relevance = ranked["good_path5"].to_numpy(dtype=float)
        dcg = float(np.sum(relevance * discount[: len(relevance)]))
        ideal = np.sort(day["good_path5"].to_numpy(dtype=float))[::-1][:k]
        idcg = float(np.sum(ideal * discount[: len(ideal)]))
        if idcg > 0.0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else float("nan")


def _daily_precision_at_k(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    k: int = 10,
) -> float:
    working = frame[["date", "good_path5"]].copy()
    working["probability"] = probability
    values = [
        float(day.nlargest(k, "probability")["good_path5"].mean())
        for _, day in working.groupby("date", sort=False)
        if not day.empty
    ]
    return float(np.mean(values)) if values else float("nan")


def _metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    top_k: int,
    cost_bps: float,
) -> dict[str, Any]:
    return {
        **binary_metrics(frame["good_path5"], probability, threshold=0.5),
        **daily_top_k_trading_metrics(
            frame,
            probability,
            top_k=top_k,
            round_trip_cost_bps=cost_bps,
        ),
        "daily_precision_at10": _daily_precision_at_k(
            frame, probability, k=10
        ),
        "daily_ndcg_at10": _daily_ndcg_at_k(frame, probability, k=10),
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    frozen_architecture = args.frozen_architecture
    if frozen_architecture is not None:
        decision_path = args.report_root / "ownership_decision.json"
        if not decision_path.exists():
            raise FileNotFoundError(
                "frozen holdout requires the A/B ownership decision first"
            )
        prior_decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if prior_decision.get("selected_architecture") != frozen_architecture:
            raise RuntimeError(
                "frozen architecture does not match the persisted A/B decision"
            )
        architecture_components = {
            frozen_architecture: ARCHITECTURES[frozen_architecture]
        }
    else:
        architecture_components = ARCHITECTURES
    required_pools = tuple(
        dict.fromkeys(
            component
            for components in architecture_components.values()
            for component in components
        )
    )
    columns = [
        "symbol",
        "date",
        "label_end_date",
        "good_path5",
        "terminal_return",
        "mfe",
        "mae",
        *FACTOR_COLUMNS,
        *ALL_SHORT_GROUPS,
    ]
    fold_by_name = {fold.name: fold for fold in DEFAULT_YEAR_FOLDS}
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    feature_audit: dict[str, Any] = {}

    for fold_name in args.folds:
        # Reload once per fold so the full master can be released after the
        # chronological split. This trades sequential parquet reads for a much
        # lower peak RSS during the long-task expansion and XGBoost fit.
        frame = pd.read_parquet(args.master, columns=columns)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["label_end_date"] = pd.to_datetime(
            frame["label_end_date"], errors="raise"
        )
        frame["good_path5"] = frame["good_path5"].astype(int)
        fold = fold_by_name[fold_name]
        splits = split_by_year_fold(frame, fold)
        del frame
        gc.collect()
        early_all, calibration_all = _validation_stages(splits.validation)
        usable = _usable_features(splits.train)
        feature_audit[fold_name] = {
            "input_factor_count": len(FACTOR_COLUMNS),
            "usable_factor_count": len(usable),
            "usable_factor_contract_sha256": factor_contract_sha256(
                usable, schema_version=MODEL_SCHEMA_VERSION
            ),
        }
        pool_predictions: dict[str, np.ndarray] = {}
        for pool in required_pools:
            groups = POOL_GROUPS[pool]
            train = _pool_frame(splits.train, groups)
            early = _pool_frame(early_all, groups)
            calibration = _pool_frame(calibration_all, groups)
            if any(
                len(part) < 500 or part["good_path5"].nunique() < 2
                for part in (train, early, calibration)
            ):
                raise RuntimeError(f"ownership pool {pool}/{fold_name} lacks samples")
            model = _fit_pool_model(
                train,
                early,
                calibration,
                groups=groups,
                features=usable,
                model_jobs=args.model_jobs,
            )
            pool_predictions[pool] = _predict_pool(splits.test, model, groups)
            model_path = args.model_root / fold_name / f"{pool}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            manifest = {
                "status": "success",
                "schema_version": MODEL_SCHEMA_VERSION,
                "lifecycle": "research_only",
                "fold": asdict(fold),
                "pool": pool,
                "groups": list(groups),
                "factor_count": len(usable),
                "features": list(model.feature_names_in_),
                "classifier_spec": asdict(LONG_TASK_DEEP_XGB_CLASSIFIER_SPEC),
                "factor_contract_sha256": feature_audit[fold_name][
                    "usable_factor_contract_sha256"
                ],
                "artifact": str(model_path),
                "artifact_sha256": _sha256(model_path),
                "forbidden_aliases": [],
            }
            if find_forbidden_aliases_in_payload(manifest):
                raise RuntimeError("ownership model manifest contains forbidden aliases")
            atomic_write_json(manifest, model_path.with_suffix(".manifest.json"))
            del model, train, early, calibration
            gc.collect()

        fold_prediction = splits.test[
            ["symbol", "date", "good_path5", "terminal_return", *ALL_SHORT_GROUPS]
        ].copy()
        fold_prediction.insert(0, "fold", fold_name)
        mixed_mask = splits.test[list(MIXED_GROUPS)].any(axis=1).to_numpy()
        right_mask = splits.test[list(RIGHT_GROUPS)].any(axis=1).to_numpy()
        left_mask = splits.test[list(LEFT_GROUPS)].any(axis=1).to_numpy()
        scopes = {
            "all": np.ones(len(splits.test), dtype=bool),
            "mixed": mixed_mask,
            "mixed_exclusive": mixed_mask & ~right_mask & ~left_mask,
            "right": right_mask,
            "left": left_mask,
            **{
                group: splits.test[group].fillna(False).astype(bool).to_numpy()
                for group in MIXED_GROUPS
            },
        }
        for architecture, components in architecture_components.items():
            probability = _combine_predictions(
                [pool_predictions[component] for component in components]
            )
            fold_prediction[f"pred_{architecture}"] = probability
            for scope, mask in scopes.items():
                metrics_rows.append(
                    {
                        "fold": fold_name,
                        "architecture": architecture,
                        "components": "+".join(components),
                        "model_count": len(components),
                        "scope": scope,
                        "rows": int(mask.sum()),
                        **_metrics(
                            splits.test.loc[mask],
                            probability[mask],
                            top_k=args.daily_top_k,
                            cost_bps=args.round_trip_cost_bps,
                        ),
                    }
                )
        prediction_rows.append(fold_prediction)
        del splits, early_all, calibration_all, pool_predictions
        gc.collect()

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    metrics_name = (
        "frozen_holdout_metrics.csv"
        if frozen_architecture is not None
        else "architecture_metrics.csv"
    )
    predictions_name = (
        "frozen_holdout_predictions.parquet"
        if frozen_architecture is not None
        else "test_predictions.parquet"
    )
    atomic_write_csv(metrics, args.report_root / metrics_name, index=False)
    atomic_write_parquet(
        predictions,
        args.report_root / predictions_name,
        index=False,
    )

    if frozen_architecture is not None:
        holdout = {
            "status": "success",
            "schema_version": "short-side-mixed-ownership-holdout-v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frozen_architecture": frozen_architecture,
            "frozen_components": list(architecture_components[frozen_architecture]),
            "folds": list(args.folds),
            "metrics": metrics.to_dict(orient="records"),
            "feature_audit": feature_audit,
            "sample_contract": "same_event_keys_same_target_same_fold_same_factor_union",
            "entry_executable_required": True,
            "locked_limit_up_excluded": True,
            "forbidden_aliases": [],
        }
        if find_forbidden_aliases_in_payload(holdout):
            raise RuntimeError("ownership holdout contains forbidden aliases")
        atomic_write_json(holdout, args.report_root / "frozen_holdout.json")
        return holdout

    decision_folds = [fold for fold in ("A", "B") if fold in set(args.folds)]
    architecture_scores: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        rows = metrics[
            metrics["architecture"].eq(architecture)
            & metrics["fold"].isin(decision_folds)
        ]
        mixed = rows[rows["scope"].eq("mixed")]
        mixed_groups = rows[rows["scope"].isin(MIXED_GROUPS)]
        overall = rows[rows["scope"].eq("all")]
        architecture_scores[architecture] = {
            "model_count": len(ARCHITECTURES[architecture]),
            "mixed_ap_by_fold": dict(
                zip(mixed["fold"], mixed["average_precision"], strict=True)
            ),
            "overall_ap_by_fold": dict(
                zip(overall["fold"], overall["average_precision"], strict=True)
            ),
            "mixed_daily_precision_at10_by_fold": dict(
                zip(mixed["fold"], mixed["daily_precision_at10"], strict=True)
            ),
            "mixed_ndcg_at10_by_fold": dict(
                zip(mixed["fold"], mixed["daily_ndcg_at10"], strict=True)
            ),
            "mixed_group_ap_by_fold": {
                group: dict(
                    zip(
                        mixed_groups.loc[mixed_groups["scope"].eq(group), "fold"],
                        mixed_groups.loc[
                            mixed_groups["scope"].eq(group), "average_precision"
                        ],
                        strict=True,
                    )
                )
                for group in MIXED_GROUPS
            },
            "worst_mixed_group_ap": float(
                mixed_groups["average_precision"].min()
            ),
            "worst_mixed_ap": float(mixed["average_precision"].min()),
            "median_mixed_ap": float(mixed["average_precision"].median()),
            "worst_overall_ap": float(overall["average_precision"].min()),
        }
    selected = max(
        architecture_scores,
        key=lambda name: (
            architecture_scores[name]["worst_mixed_group_ap"],
            architecture_scores[name]["worst_mixed_ap"],
            architecture_scores[name]["median_mixed_ap"],
            architecture_scores[name]["worst_overall_ap"],
            -architecture_scores[name]["model_count"],
        ),
    )
    decision = {
        "status": "success",
        "schema_version": "short-side-mixed-ownership-decision-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision_folds": decision_folds,
        "primary_order": [
            "worst_per_mixed_group_average_precision",
            "worst_mixed_average_precision",
            "median_mixed_average_precision",
            "worst_overall_average_precision",
            "fewer_production_models_on_tie",
        ],
        "architectures": architecture_scores,
        "selected_architecture": selected,
        "selected_components": list(ARCHITECTURES[selected]),
        "production_model_count": len(ARCHITECTURES[selected]),
        "sample_contract": "same_event_keys_same_target_same_fold_same_factor_union",
        "factor_input_count": len(FACTOR_COLUMNS),
        "feature_audit": feature_audit,
        "entry_executable_required": True,
        "locked_limit_up_excluded": True,
        "forbidden_aliases": [],
    }
    if find_forbidden_aliases_in_payload(decision):
        raise RuntimeError("ownership decision contains forbidden aliases")
    atomic_write_json(decision, args.report_root / "ownership_decision.json")
    return decision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--right-events", type=Path, default=DEFAULT_RIGHT_EVENTS)
    build.add_argument("--right-labels", type=Path, default=DEFAULT_RIGHT_LABELS)
    build.add_argument("--left-events", type=Path, default=DEFAULT_LEFT_EVENTS)
    build.add_argument("--left-labels", type=Path, default=DEFAULT_LEFT_LABELS)
    build.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--start-date", default=None)
    build.add_argument("--end-date", default=None)

    run = subparsers.add_parser("run")
    run.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    run.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    run.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    run.add_argument("--folds", nargs="+", default=["A", "B"])
    run.add_argument("--model-jobs", type=int, default=4)
    run.add_argument("--daily-top-k", type=int, default=10)
    run.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    run.add_argument(
        "--frozen-architecture",
        choices=tuple(ARCHITECTURES),
        default=None,
        help="Evaluate only the persisted A/B winner on untouched holdout folds.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = build_master(args) if args.command == "build" else run_ablation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
