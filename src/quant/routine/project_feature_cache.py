"""Strict loader for the shared exact-date project-factor cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant.features.project_factor_layer import PROJECT_FACTOR_SCHEMA_VERSION
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


@dataclass(frozen=True)
class ProjectFeatureCacheSnapshot:
    features: pd.DataFrame
    eligible_signals: pd.DataFrame
    policy_excluded_symbols: tuple[str, ...]
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact_date_project_feature_cache(
    target: pd.Timestamp,
    signals: pd.DataFrame,
    *,
    feature_path: Path,
    manifest_path: Path,
    required_value_columns: Sequence[str] = (),
    context: str,
) -> ProjectFeatureCacheSnapshot:
    """Load one complete shared cache snapshot and explain every omission.

    Consumers declare only their additional non-null requirements. The common
    date, checksum, schema, factor, key, and candidate-coverage rules remain
    centralized so a future model cannot silently bypass them.
    """

    target = pd.Timestamp(target).normalize()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"{context} project feature cache manifest is unavailable"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{context} project feature cache manifest must be a mapping")
    if manifest.get("status") != "success":
        raise RuntimeError(f"{context} project feature cache did not succeed")
    if manifest.get("target_date") != target.date().isoformat():
        raise RuntimeError(f"{context} project feature cache is stale")
    if manifest.get("candidate_coverage_status") != "complete":
        raise RuntimeError(
            f"{context} project feature candidate coverage is incomplete"
        )
    if manifest.get("factor_schema_version") != PROJECT_FACTOR_SCHEMA_VERSION:
        raise RuntimeError(f"{context} project feature schema mismatch")
    manifest_factor_count = manifest.get("factor_count")
    if manifest_factor_count is not None and int(manifest_factor_count) != len(
        PROJECT_FACTOR_COLUMNS
    ):
        raise RuntimeError(f"{context} project feature factor count mismatch")
    expected_sha256 = str(manifest.get("output_sha256") or "")
    try:
        actual_sha256 = _sha256(feature_path)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"{context} project feature cache is unavailable") from exc
    if not expected_sha256 or expected_sha256 != actual_sha256:
        raise RuntimeError(f"{context} project feature checksum mismatch")

    features = pd.read_parquet(feature_path)
    required = {
        "ts_code",
        "symbol",
        "trade_date",
        "date",
        "factor_schema_version",
        *PROJECT_FACTOR_COLUMNS,
    }
    missing_columns = required - set(features.columns)
    if missing_columns:
        raise RuntimeError(
            f"{context} project feature cache missing columns: "
            f"{sorted(missing_columns)}"
        )
    missing_required = set(required_value_columns) - set(PROJECT_FACTOR_COLUMNS)
    if missing_required:
        raise ValueError(
            f"{context} required project values are not registered factors: "
            f"{sorted(missing_required)}"
        )

    features = features.copy()
    features["symbol"] = features["symbol"].astype(str)
    features["date"] = pd.to_datetime(
        features["date"], errors="coerce"
    ).dt.normalize()
    if features["date"].isna().any() or not features["date"].eq(target).all():
        raise RuntimeError(f"{context} project feature cache is not exact-date")
    if features.duplicated(["symbol", "date"]).any():
        raise RuntimeError(f"{context} project feature cache contains duplicate keys")
    schemas = set(features["factor_schema_version"].dropna().astype(str))
    if features.empty or schemas != {PROJECT_FACTOR_SCHEMA_VERSION}:
        raise RuntimeError(f"{context} project feature cache row schema mismatch")
    all_null = [
        column
        for column in PROJECT_FACTOR_COLUMNS
        if features[column].notna().sum() == 0
    ]
    if all_null:
        raise RuntimeError(
            f"{context} project feature cache has all-null factors: {all_null[:20]}"
        )
    if required_value_columns:
        incomplete_values = features[list(required_value_columns)].isna().any(axis=1)
        if incomplete_values.any():
            samples = (
                features.loc[incomplete_values, "symbol"]
                .astype(str)
                .head(20)
                .tolist()
            )
            raise RuntimeError(
                f"{context} project feature cache has incomplete daily_basic values: "
                f"count={int(incomplete_values.sum())} samples={samples}"
            )

    expected_symbols = set(signals["symbol"].astype(str))
    available_symbols = set(features["symbol"])
    policy_excluded = {
        str(symbol)
        for symbol in manifest.get("policy_excluded_candidate_symbols") or ()
        if str(symbol)
    }
    overlap = available_symbols & policy_excluded
    if overlap:
        raise RuntimeError(
            f"{context} project feature policy exclusions overlap cached rows: "
            f"{sorted(overlap)[:20]}"
        )
    missing_symbols = expected_symbols - available_symbols
    unexplained = missing_symbols - policy_excluded
    if unexplained:
        raise RuntimeError(
            f"{context} project feature cache has unexplained missing candidates: "
            f"{sorted(unexplained)[:20]}"
        )

    excluded = tuple(sorted(expected_symbols & policy_excluded))
    eligible_signals = signals[
        ~signals["symbol"].astype(str).isin(excluded)
    ].reset_index(drop=True)
    eligible_symbols = set(eligible_signals["symbol"].astype(str))
    features = features[
        features["symbol"].isin(eligible_symbols)
    ].reset_index(drop=True)
    return ProjectFeatureCacheSnapshot(
        features=features,
        eligible_signals=eligible_signals,
        policy_excluded_symbols=excluded,
        manifest=manifest,
    )


__all__ = [
    "ProjectFeatureCacheSnapshot",
    "load_exact_date_project_feature_cache",
]
