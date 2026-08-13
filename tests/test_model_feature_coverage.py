from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.ml.feature_coverage import (
    RequiredFeatureCoverageError,
    inspect_required_feature_coverage,
    model_feature_history_start,
    validate_required_feature_coverage,
)


def test_feature_coverage_distinguishes_missing_and_all_null_features() -> None:
    frame = pd.DataFrame(
        {
            "partial": [1.0, np.nan],
            "all_null": [np.nan, np.inf],
        }
    )

    report = inspect_required_feature_coverage(
        frame,
        ["partial", "all_null", "missing"],
        target_date="2026-08-12",
    )

    assert report["status"] == "invalid"
    assert report["target_date"] == "2026-08-12"
    assert report["missing_columns"] == ["missing"]
    assert report["all_null_features"] == ["all_null"]
    assert report["partial_features"] == ["partial"]
    assert report["coverage"] == {
        "partial": 0.5,
        "all_null": 0.0,
        "missing": 0.0,
    }


def test_strict_feature_coverage_allows_partial_rows_but_rejects_daily_outages() -> None:
    valid = pd.DataFrame({"feature": [1.0, np.nan]})
    assert validate_required_feature_coverage(valid, ["feature"])["status"] == "valid"

    invalid = pd.DataFrame({"feature": [np.nan, np.inf]})
    with pytest.raises(RequiredFeatureCoverageError) as exc_info:
        validate_required_feature_coverage(
            invalid,
            ["feature", "missing"],
            context="test scoring",
        )

    assert exc_info.value.report["missing_columns"] == ["missing"]
    assert exc_info.value.report["all_null_features"] == ["feature"]
    assert "test scoring" in str(exc_info.value)


def test_model_feature_history_uses_six_calendar_years() -> None:
    assert model_feature_history_start("2026-08-12") == pd.Timestamp("2020-08-12")
    assert model_feature_history_start("2024-02-29") == pd.Timestamp("2018-02-28")
