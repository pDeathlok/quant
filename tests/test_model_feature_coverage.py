from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.data.daily_basic_valuation import REFERENCE_SOURCE
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


@pytest.mark.parametrize(
    "failure", [None, "no_evidence", "stale", "positive_pe", "missing_symbol",
                "duplicate", "mixed_dates", "missing_feature", "other_null", "infinity"],
)
def test_all_null_pe_requires_same_day_loss_for_every_candidate(tmp_path, failure) -> None:
    day = "20261105"
    symbols = ["601956.SH", "603028.SH"]
    frame = pd.DataFrame({
        "symbol": symbols, "date": pd.Timestamp(day),
        "pe_ttm": np.nan, "pe_ttm_inv": np.nan, "pb": 1.0,
    })
    source = pd.DataFrame({
        "ts_code": symbols, "trade_date": day, "pe_ttm": np.nan,
        "pe_ttm_reference": [-22.0, -16.0],
        "pe_ttm_reference_date": day, "pe_ttm_reference_source": REFERENCE_SOURCE,
    })
    if failure == "no_evidence":
        source.loc[1, "pe_ttm_reference_source"] = None
    elif failure == "stale":
        source.loc[1, "pe_ttm_reference_date"] = "20261104"
    elif failure == "positive_pe":
        source.loc[1, "pe_ttm"] = 16.0
    elif failure == "missing_symbol":
        source = source.iloc[:1]
    elif failure == "duplicate":
        source = pd.concat([source, source.iloc[:1]])
    elif failure == "mixed_dates":
        frame.loc[1, "date"] = pd.Timestamp("20261104")
    elif failure == "missing_feature":
        frame = frame.drop(columns="pe_ttm_inv")
    elif failure == "other_null":
        frame["pb"] = np.nan
    elif failure == "infinity":
        frame.loc[1, "pe_ttm_inv"] = np.inf
    source.to_parquet(tmp_path / f"{day}.parquet", index=False)
    options = {"target_date": day, "daily_basic_dir": tmp_path}
    features = ["pe_ttm", "pe_ttm_inv", "pb"]
    if failure is not None:
        with pytest.raises(RequiredFeatureCoverageError):
            validate_required_feature_coverage(frame, features, **options)
    else:
        report = validate_required_feature_coverage(frame, features, **options)
        assert report["status"] == "valid"
        assert report["verified_loss_null_features"] == ["pe_ttm", "pe_ttm_inv"]
        assert report["non_null_counts"]["pe_ttm"] == 0
        assert frame[["pe_ttm", "pe_ttm_inv"]].isna().all().all()
