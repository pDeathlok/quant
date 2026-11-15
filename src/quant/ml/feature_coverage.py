"""Runtime validation for model-required feature frames.

Model preprocessors may legitimately impute occasional stock-level gaps.  They
must not, however, turn a missing runtime column or an entirely unavailable
daily feature into a seemingly valid score.  This module keeps that boundary
explicit and returns a JSON-ready coverage report for run manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_FEATURE_HISTORY_YEARS = 6
_LOSS_NULL_PE_FEATURES = frozenset({"pe_ttm", "pe_ttm_inv"})


class RequiredFeatureCoverageError(ValueError):
    """Raised when a model-required feature is unavailable for a score batch."""

    def __init__(self, context: str, report: dict[str, Any]) -> None:
        self.context = context
        self.report = report
        missing = report["missing_columns"]
        all_null = report["all_null_features"]
        super().__init__(
            f"{context} required-feature coverage failed: "
            f"missing_columns={missing}; all_null_features={all_null}"
        )


def model_feature_history_start(
    target_date: str | pd.Timestamp,
    *,
    years: int = MODEL_FEATURE_HISTORY_YEARS,
) -> pd.Timestamp:
    """Return the history boundary needed by the longest production factors."""

    if years < 1:
        raise ValueError("model feature history years must be positive")
    return pd.Timestamp(target_date) - pd.DateOffset(years=years)


def inspect_required_feature_coverage(
    frame: pd.DataFrame,
    required_features: Iterable[str],
    *,
    target_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Describe non-null coverage for one model-scoring batch.

    Missing columns are represented with zero coverage, while also remaining
    distinct in ``missing_columns``.  Infinite numeric values count as missing
    because every production prediction path normalizes them to ``NaN``.
    """

    required = list(dict.fromkeys(str(feature) for feature in required_features))
    missing_columns = [feature for feature in required if feature not in frame.columns]
    present = [feature for feature in required if feature in frame.columns]
    normalized = (
        frame[present].replace([np.inf, -np.inf], np.nan)
        if present
        else pd.DataFrame(index=frame.index)
    )
    non_null_counts = {
        feature: int(normalized[feature].notna().sum()) if feature in normalized else 0
        for feature in required
    }
    row_count = int(len(frame))
    coverage = {
        feature: (float(non_null_counts[feature] / row_count) if row_count else 0.0)
        for feature in required
    }
    all_null_features = [
        feature for feature in present if non_null_counts[feature] == 0
    ]
    partial_features = [
        feature
        for feature in present
        if 0 < non_null_counts[feature] < row_count
    ]
    coverage_values = list(coverage.values())
    target_timestamp = (
        pd.Timestamp(target_date) if target_date is not None else None
    )
    return {
        "status": "invalid" if missing_columns or all_null_features else "valid",
        "target_date": (
            target_timestamp.date().isoformat()
            if target_timestamp is not None and pd.notna(target_timestamp)
            else None
        ),
        "row_count": row_count,
        "required_feature_count": len(required),
        "present_feature_count": len(present),
        "covered_feature_count": sum(value > 0 for value in coverage_values),
        "missing_columns": missing_columns,
        "all_null_features": all_null_features,
        "partial_features": partial_features,
        "minimum_coverage": min(coverage_values) if coverage_values else 1.0,
        "mean_coverage": float(np.mean(coverage_values)) if coverage_values else 1.0,
        "coverage": coverage,
        "non_null_counts": non_null_counts,
    }


def _verified_loss_batch(
    frame: pd.DataFrame,
    target_date: str | pd.Timestamp | None,
    daily_basic_dir: Path,
) -> bool:
    from quant.data.daily_basic_valuation import resolved_feature_mask
    from quant.data.market_snapshot import current_market_snapshot

    if frame.empty or target_date is None or not {"symbol", "date"} <= set(frame.columns):
        return False
    target = pd.Timestamp(target_date).normalize()
    dates = pd.to_datetime(frame["date"].astype("string"), format="mixed", errors="coerce")
    if not dates.dt.normalize().eq(target).all() or frame["symbol"].isna().any():
        return False
    symbols = sorted(set(frame["symbol"].astype(str)))
    trade_date = target.strftime("%Y%m%d")
    snapshot = current_market_snapshot()
    if snapshot is not None:
        source = snapshot.read("daily_basic", trade_date, trade_date, symbols=symbols)
    else:
        path = daily_basic_dir / f"{trade_date}.parquet"
        if not path.is_file():
            return False
        source = pd.read_parquet(path)
    if not {"ts_code", "trade_date", "pe_ttm"} <= set(source.columns):
        return False
    source = source.loc[source["ts_code"].isin(symbols)].reset_index(drop=True)
    if (
        len(source) != len(symbols)
        or source["ts_code"].duplicated().any()
        or not source["trade_date"].astype(str).eq(trade_date).all()
        or not source["pe_ttm"].isna().all()
    ):
        return False
    return bool(resolved_feature_mask(source, "pe_ttm").all())


def validate_required_feature_coverage(
    frame: pd.DataFrame,
    required_features: Iterable[str],
    *,
    target_date: str | pd.Timestamp | None = None,
    context: str = "model scoring",
    daily_basic_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail on unavailable features, except PE nulls proven by same-day losses."""

    report = inspect_required_feature_coverage(
        frame,
        required_features,
        target_date=target_date,
    )
    all_null = report["all_null_features"]
    if (
        daily_basic_dir is not None
        and all_null
        and not report["missing_columns"]
        and set(all_null) <= _LOSS_NULL_PE_FEATURES
        and frame[all_null].isna().all().all()
        and _verified_loss_batch(frame, target_date, daily_basic_dir)
    ):
        report["verified_loss_null_features"] = all_null
        report["all_null_features"] = []
        report["status"] = "valid"
    if report["status"] != "valid":
        raise RequiredFeatureCoverageError(context, report)
    return report
