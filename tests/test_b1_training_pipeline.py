from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


RESEARCH_SCRIPTS = Path(__file__).parents[1] / "scripts" / "research"
if str(RESEARCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SCRIPTS))

import refresh_b1_feature_cache as b1_refresh  # noqa: E402
import train_b1_tushare_models as b1_training  # noqa: E402
from quant.ml.feature_coverage import RequiredFeatureCoverageError  # noqa: E402
from train_b1_tushare_models import combine_training_frames  # noqa: E402


def test_b1_incremental_build_reads_six_years_of_history(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakeStore:
        def read_market_range(self, dataset_name, *, start_date, symbols):
            captured.update(
                dataset_name=dataset_name,
                start_date=start_date,
                symbols=symbols,
            )
            return pd.DataFrame()

    monkeypatch.setattr(
        b1_training.MarketDataStoreConfig,
        "from_env",
        lambda root: object(),
    )
    monkeypatch.setattr(b1_training, "MarketDataStore", lambda config: FakeStore())
    monkeypatch.setattr(
        b1_training,
        "process_daily_frame",
        lambda args: pd.DataFrame(),
    )

    path = tmp_path / "daily" / "000001.SZ.parquet"
    b1_training.process_daily_file((str(path), "20260812"))

    assert captured == {
        "dataset_name": "daily",
        "start_date": "20200812",
        "symbols": ["000001.SZ"],
    }


def test_b1_refresh_reports_latest_required_feature_coverage() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-08-11", "2026-08-12", "2026-08-12"]
            ),
            "weekly_ma233": [None, 10.0, None],
        }
    )

    report = b1_refresh._validate_latest_model_features(
        data,
        ["weekly_ma233"],
    )

    assert report["status"] == "valid"
    assert report["target_date"] == "2026-08-12"
    assert report["coverage"] == {"weekly_ma233": 0.5}


def test_b1_refresh_rejects_latest_all_null_required_feature() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-11", "2026-08-12"]),
            "weekly_ma233": [10.0, None],
        }
    )

    with pytest.raises(RequiredFeatureCoverageError) as exc_info:
        b1_refresh._validate_latest_model_features(data, ["weekly_ma233"])

    assert exc_info.value.report["target_date"] == "2026-08-12"
    assert exc_info.value.report["all_null_features"] == ["weekly_ma233"]


def _project_feature_row(symbol: str, target_date: pd.Timestamp) -> dict[str, object]:
    row: dict[str, object] = {
        column: 1.0 for column in b1_refresh.PROJECT_FACTOR_COLUMNS
    }
    row.update(
        symbol=symbol,
        ts_code=symbol,
        trade_date=target_date.strftime("%Y%m%d"),
        date=target_date,
        factor_schema_version=b1_refresh.resolve_project_factor_schema(),
    )
    return row


def test_active_candidate_sidecar_unions_b1_and_z_without_duplicate_work() -> None:
    target_date = pd.Timestamp("2026-08-12")
    b1_gate = pd.DataFrame(
        {"symbol": ["000001.SZ"], "date": [target_date]}
    )
    z_gate = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [target_date, target_date],
        }
    )

    assert b1_refresh._additional_only_symbols(
        b1_gate,
        z_gate,
        target_date,
    ) == ["000002.SZ"]

    active, stats = b1_refresh._assemble_active_candidate_cache(
        pd.DataFrame([_project_feature_row("000001.SZ", target_date)]),
        pd.DataFrame([_project_feature_row("000002.SZ", target_date)]),
        b1_gate,
        z_gate,
        target_date=target_date,
        factor_schema_version=b1_refresh.resolve_project_factor_schema(),
    )

    assert active["symbol"].tolist() == ["000001.SZ", "000002.SZ"]
    assert active.set_index("symbol")["candidate_sources"].to_dict() == {
        "000001.SZ": "b1,z_skill",
        "000002.SZ": "z_skill",
    }
    assert set(b1_refresh.PROJECT_FACTOR_COLUMNS) <= set(active.columns)
    assert stats == {
        "target_date": "2026-08-12",
        "b1_candidate_count": 1,
        "z_candidate_count": 2,
        "union_candidate_count": 2,
        "overlap_candidate_count": 1,
        "computed_candidate_count": 2,
        "missing_candidate_count": 0,
        "missing_candidate_symbols": [],
        "missing_candidate_samples": [],
    }

    coverage = b1_refresh._validate_active_candidate_coverage(stats, [])
    assert coverage["candidate_coverage_status"] == "complete"
    assert coverage["unexplained_missing_candidate_count"] == 0


def test_active_candidate_coverage_allows_policy_exclusion_but_not_silent_miss() -> None:
    stats = {
        "union_candidate_count": 3,
        "missing_candidate_symbols": ["000003.SZ"],
    }

    coverage = b1_refresh._validate_active_candidate_coverage(
        stats,
        ["000003.SZ: target signal belongs to ST/delisting stock"],
    )

    assert coverage["eligible_candidate_count"] == 2
    assert coverage["policy_excluded_candidate_symbols"] == ["000003.SZ"]
    with pytest.raises(RuntimeError, match="unexplained missing symbols"):
        b1_refresh._validate_active_candidate_coverage(stats, [])


def test_combine_training_frames_prefers_causal_factor_price_columns() -> None:
    daily = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": pd.to_datetime(["2026-07-31"]),
            "close": [10.0],
            "pct_chg": [1.0],
        }
    )
    factors = pd.DataFrame(
        {
            "close": [9.5],
            "pct_chg": [0.5],
            "factor_schema_version": ["project-v4-causal-price-alpha"],
        }
    )
    labels = pd.DataFrame({"label_t1_open_max_high_5pct": [1]})

    combined = combine_training_frames(daily, factors, labels)

    assert not combined.columns.has_duplicates
    assert combined.loc[0, "close"] == 9.5
    assert combined.loc[0, "pct_chg"] == 0.5
    assert combined.loc[0, "factor_schema_version"] == "project-v4-causal-price-alpha"
