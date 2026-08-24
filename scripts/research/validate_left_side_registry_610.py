#!/usr/bin/env python
"""Evaluate the complete canonical registry as a left-ranker candidate arm.

The governed registry is the candidate universe.  Only point-in-time columns
with observed training coverage and more than one value enter XGBoost; the
manifest retains every unavailable/constant candidate and its exclusion
reason.  Fold A is a fail-fast development screen.  A non-improving full arm
must not consume B/C or replace the released compact contract.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from quant.data.atomic_io import atomic_write_csv, atomic_write_json
from quant.features.canonical_factor_names import (
    FORBIDDEN_COMPATIBILITY_ALIASES,
    assert_no_forbidden_factor_names,
    stable_canonical_feature_union,
)
from quant.features.factor_registry import FACTOR_REGISTRY
from quant.features.left_side_factor_contract import LEFT_SIDE_FACTOR_COLUMNS
from quant.research.left_side_unified_features import LEFT_SIDE_SIGNALS
from quant.research.right_side_targets import materialize_training_target
from quant.research.right_side_unified import DEFAULT_YEAR_FOLDS, split_by_year_fold

from scripts.research.validate_unified_left_side_models import (
    _fit_unified_model,
    _split_validation_stages,
)


EVENTS = PROJECT_ROOT / "data/research/left_side_unified_v3_group4_input_parity/events.parquet"
LABELS = PROJECT_ROOT / "data/research/left_side_unified_v3_group4_input_parity/labels.parquet"
BASELINE_PREDICTIONS = PROJECT_ROOT / "reports/research/left_side_unified_v3_group4_input_parity/test_predictions.parquet"
REPORT_ROOT = PROJECT_ROOT / "reports/research/left_side_registry_610"

DAILY_SOURCES = (
    PROJECT_ROOT / "data/research/right_side_registry_607/daily_basic_features.parquet",
    PROJECT_ROOT / "data/research/right_side_registry_607/daily_extras.parquet",
)
WEEKLY_SOURCES = (
    PROJECT_ROOT / "data/research/right_side_registry_607/analyst_weekly.parquet",
    PROJECT_ROOT / "data/features/long_entry/weekly_external_v1.parquet",
    PROJECT_ROOT / "data/features/long_entry/weekly_quality_factors_v1.parquet",
    PROJECT_ROOT / "data/features/long_entry/weekly_training_v2.parquet",
)


def _registry_feature_columns() -> tuple[str, ...]:
    columns = stable_canonical_feature_union(
        definition.name
        for definition in FACTOR_REGISTRY
        if definition.role == "feature"
    )
    assert_no_forbidden_factor_names(columns, context="left registry candidate universe")
    return columns


def _read_source(
    path: Path,
    *,
    candidates: set[str],
    keys: tuple[str, ...],
    end: pd.Timestamp,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    columns = [*keys, *sorted(candidates & available)]
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    return frame[frame["date"].le(end)].copy()


def _merge_daily(events: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    output = events
    remaining = set(candidates) - set(output.columns)
    for path in DAILY_SOURCES:
        source = _read_source(
            path,
            candidates=remaining,
            keys=("symbol", "date"),
            end=output["date"].max(),
        )
        additions = [column for column in source if column not in {"symbol", "date"}]
        if additions:
            output = output.merge(
                source[["symbol", "date", *additions]],
                on=["symbol", "date"],
                how="left",
                validate="one_to_one",
            )
            remaining.difference_update(additions)
    return output


def _merge_weekly(events: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    remaining = set(candidates) - set(events.columns)
    weekly: pd.DataFrame | None = None
    for path in WEEKLY_SOURCES:
        source = _read_source(
            path,
            candidates=remaining,
            keys=("ts_code", "date"),
            end=events["date"].max(),
        ).rename(columns={"ts_code": "symbol"})
        additions = [column for column in source if column not in {"symbol", "date"}]
        if not additions:
            continue
        source = source[["symbol", "date", *additions]].drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        weekly = (
            source
            if weekly is None
            else weekly.merge(
                source,
                on=["symbol", "date"],
                how="outer",
                validate="one_to_one",
            )
        )
        remaining.difference_update(additions)
    if weekly is None:
        return events
    left = events.sort_values(["date", "symbol"], kind="stable")
    right = weekly.sort_values(["date", "symbol"], kind="stable")
    return pd.merge_asof(
        left,
        right,
        on="date",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
        tolerance=pd.Timedelta(days=14),
    ).sort_values(["date", "symbol"], kind="stable")


def main() -> None:
    registry = _registry_feature_columns()
    end = pd.Timestamp("2026-08-22")
    base_columns = [
        "symbol",
        "date",
        *LEFT_SIDE_FACTOR_COLUMNS,
        *LEFT_SIDE_SIGNALS,
        "signal_count",
    ]
    events = pd.read_parquet(
        EVENTS,
        columns=base_columns,
        filters=[("date", "<=", end)],
    )
    events["date"] = pd.to_datetime(events["date"], errors="raise").dt.normalize()
    labels = pd.read_parquet(
        LABELS,
        columns=(
            "symbol",
            "date",
            "mature",
            "locked_limit_up",
            "good_path5",
            "label_end_date",
        ),
        filters=[
            ("entry_mode", "==", "next_close"),
            ("horizon", "==", 5),
            ("date", "<=", end),
        ],
    )
    labels["date"] = pd.to_datetime(labels["date"], errors="raise").dt.normalize()
    labels["label_end_date"] = pd.to_datetime(labels["label_end_date"], errors="raise")
    labels = materialize_training_target(labels, "good_path5")
    labels = labels[
        labels["mature"]
        & ~labels["locked_limit_up"]
        & labels["good_path5"].notna()
    ]
    selected = events.merge(
        labels[["symbol", "date", "good_path5", "label_end_date"]],
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )
    del events, labels
    selected = _merge_daily(selected, registry)
    selected = _merge_weekly(selected, registry)
    gc.collect()

    baseline = pd.read_parquet(BASELINE_PREDICTIONS)
    baseline["date"] = pd.to_datetime(baseline["date"], errors="raise").dt.normalize()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, object]] = []
    passed_prior_gate = True
    for fold in DEFAULT_YEAR_FOLDS:
        if not passed_prior_gate:
            break
        splits = split_by_year_fold(selected, fold)
        coverage_rows = []
        usable = []
        for feature in registry:
            if feature not in splits.train:
                coverage_rows.append((feature, 0.0, 0, "not_materialized"))
                continue
            values = pd.to_numeric(splits.train[feature], errors="coerce")
            coverage = float(values.notna().mean())
            unique = int(values.nunique(dropna=True))
            status = (
                "usable"
                if coverage >= 0.01 and unique >= 2
                else "low_coverage_or_constant"
            )
            coverage_rows.append((feature, coverage, unique, status))
            if status == "usable":
                usable.append(feature)
        usable_features = stable_canonical_feature_union(usable)
        assert_no_forbidden_factor_names(
            usable_features,
            context=f"left registry usable features fold {fold.name}",
        )
        coverage = pd.DataFrame(
            coverage_rows,
            columns=("factor", "coverage", "unique_values", "status"),
        )
        atomic_write_csv(
            coverage,
            REPORT_ROOT / f"factor_coverage_{fold.name}.csv",
            index=False,
        )
        early, calibration, _ = _split_validation_stages(
            splits.validation, "good_path5"
        )
        model = _fit_unified_model(
            splits.train,
            early,
            calibration,
            usable_features,
            "good_path5",
            n_jobs=6,
        )
        probability = np.asarray(model.predict_proba(splits.test), dtype=float)[:, 1]
        full_ap = float(
            average_precision_score(splits.test["good_path5"], probability)
        )
        fold_baseline = baseline[baseline["fold"].eq(fold.name)][
            ["symbol", "date", "pred_unified_left_long_task_deep"]
        ].copy()
        paired = splits.test[["symbol", "date", "good_path5"]].merge(
            fold_baseline,
            on=["symbol", "date"],
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(splits.test):
            raise RuntimeError(
                f"registry comparison baseline keys are incomplete for fold {fold.name}"
            )
        baseline_ap = float(
            average_precision_score(
                paired["good_path5"],
                paired["pred_unified_left_long_task_deep"],
            )
        )
        delta = full_ap - baseline_ap
        fold_result = {
            "fold": fold.name,
            "test_year": fold.test_year,
            "rows": len(splits.test),
            "candidate_contract_count": len(registry),
            "usable_feature_count": len(usable_features),
            "usable_features": list(usable_features),
            "baseline_compact_ap": baseline_ap,
            "registry_full_ap": full_ap,
            "delta_ap": delta,
            "passed_non_inferiority_gate": bool(delta >= 0.0),
        }
        fold_results.append(fold_result)
        atomic_write_json(
            fold_result,
            REPORT_ROOT / f"validation_summary_{fold.name}.json",
        )
        print(json.dumps(fold_result, ensure_ascii=False, indent=2), flush=True)
        passed_prior_gate = bool(delta >= 0.0)
        del splits, model, probability, coverage
        gc.collect()

    promotion_eligible = (
        [row["fold"] for row in fold_results] == ["A", "B", "C"]
        and all(bool(row["passed_non_inferiority_gate"]) for row in fold_results)
    )
    result = {
        "status": "success",
        "schema_version": "left-side-registry-610-pit-abc-v2",
        "candidate_contract_count": len(registry),
        "fold_results": fold_results,
        "promotion_eligible": promotion_eligible,
        "stopped_after_fold": fold_results[-1]["fold"],
        "gate": "candidate_average_precision_greater_than_or_equal_to_compact_per_fold",
        "promotion_modified": False,
        "forbidden_alias_intersection": sorted(
            set(registry) & set(FORBIDDEN_COMPATIBILITY_ALIASES)
        ),
    }
    atomic_write_json(result, REPORT_ROOT / "validation_summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
