from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.ml.feature_coverage import RequiredFeatureCoverageError
from scripts.research import score_latest_strategy_models as scoring


def test_build_latest_dataset_allows_explicitly_excluded_candidates(monkeypatch, tmp_path):
    target_date = pd.Timestamp("2026-07-27")
    signal_rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "600228.SH"],
            "date": [target_date, target_date],
            "signal": [True, True],
        }
    )
    paths = [tmp_path / "000001.SZ.parquet", tmp_path / "600228.SH.parquet"]
    scored_frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [target_date],
            "signal": [True],
        }
    )

    monkeypatch.setattr(scoring, "_load_signal_cache", lambda signals, start_date: signal_rows)
    monkeypatch.setattr(scoring, "list_partitioned_symbol_paths", lambda daily_dir: paths)

    def fake_process(args):
        path = Path(args[0])
        if path.stem == "600228.SH":
            return None, None, "600228.SH: target signal belongs to ST/delisting stock"
        return scored_frame, None, None

    monkeypatch.setattr(scoring, "_process_symbol", fake_process)

    result = scoring.build_latest_dataset(
        tmp_path,
        ["signal"],
        "2020-01-01",
        "2026-07-27",
        workers=1,
    )

    assert result["symbol"].tolist() == ["000001.SZ"]
    assert result.attrs["excluded_candidates"] == [
        "600228.SH: target signal belongs to ST/delisting stock"
    ]


@pytest.mark.parametrize(
    ("frame", "missing_columns", "all_null_features"),
    [
        (pd.DataFrame({"present": [1.0]}), ["required"], []),
        (
            pd.DataFrame({"present": [1.0], "required": [np.nan]}),
            [],
            ["required"],
        ),
    ],
)
def test_ensure_model_features_fails_instead_of_silently_imputing_daily_outages(
    frame,
    missing_columns,
    all_null_features,
):
    class Model:
        feature_names_in_ = ["present", "required"]

    frame["date"] = pd.Timestamp("2026-08-12")
    with pytest.raises(RequiredFeatureCoverageError) as exc_info:
        scoring.ensure_model_features(frame, {("signal", "up5"): Model()})

    assert exc_info.value.report["missing_columns"] == missing_columns
    assert exc_info.value.report["all_null_features"] == all_null_features


def test_ensure_model_features_emits_manifest_ready_coverage() -> None:
    class Model:
        feature_names_in_ = ["feature"]

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-12", "2026-08-12"]),
            "feature": [1.0, np.nan],
        }
    )

    result = scoring.ensure_model_features(
        frame,
        {("signal", "up5"): Model()},
    )

    assert result.attrs["feature_coverage"]["status"] == "valid"
    assert result.attrs["feature_coverage"]["coverage"] == {"feature": 0.5}
    assert "pred_up5" in result


def test_latest_z_skill_scoring_reads_six_years_of_history(monkeypatch, tmp_path):
    captured: dict[str, pd.Timestamp] = {}

    def fake_read(path, *, start_date, end_date):
        captured["start_date"] = pd.Timestamp(start_date)
        captured["end_date"] = pd.Timestamp(end_date)
        return pd.DataFrame()

    monkeypatch.setattr(scoring, "read_partitioned_symbol_file", fake_read)
    target_date = pd.Timestamp("2026-08-12")
    scoring._process_symbol(
        (
            str(tmp_path / "000001.SZ.parquet"),
            pd.DataFrame(),
            ["signal"],
            target_date,
        )
    )

    assert captured == {
        "start_date": pd.Timestamp("2020-08-12"),
        "end_date": target_date,
    }


def _active_cache_row(symbol: str, target_date: pd.Timestamp) -> dict[str, object]:
    row: dict[str, object] = {
        column: 1.0 for column in scoring.PROJECT_FACTOR_COLUMNS
    }
    row.update(
        symbol=symbol,
        ts_code=symbol,
        trade_date=target_date.strftime("%Y%m%d"),
        date=target_date,
        factor_schema_version=scoring.resolve_project_factor_schema(),
    )
    return row


def test_latest_scoring_reuses_active_cache_and_only_calculates_misses(
    monkeypatch,
    tmp_path,
) -> None:
    target_date = pd.Timestamp("2026-08-12")
    signal_rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000002.SZ"],
            "date": [target_date, target_date],
            "signal": [True, True],
        }
    )
    cache_path = tmp_path / "active.parquet"
    pd.DataFrame([_active_cache_row("000001.SZ", target_date)]).to_parquet(
        cache_path,
        index=False,
    )
    missing_path = tmp_path / "000002.SZ.parquet"
    processed: list[str] = []

    monkeypatch.setattr(scoring, "_load_signal_cache", lambda signals, start_date: signal_rows)
    monkeypatch.setattr(
        scoring,
        "list_partitioned_symbol_paths",
        lambda daily_dir: [missing_path],
    )

    def fake_process(args):
        processed.append(Path(args[0]).stem)
        return pd.DataFrame(
            {
                "symbol": ["000002.SZ"],
                "date": [target_date],
                "signal": [True],
            }
        ), None, None

    monkeypatch.setattr(scoring, "_process_symbol", fake_process)

    result = scoring.build_latest_dataset(
        tmp_path,
        ["signal"],
        "2020-01-01",
        "2026-08-12",
        workers=1,
        feature_cache_path=cache_path,
    )

    assert result["symbol"].tolist() == ["000001.SZ", "000002.SZ"]
    assert processed == ["000002.SZ"]
    assert result.attrs["feature_cache_symbols"] == 1
    assert result.attrs["calculated_symbols"] == 1


def test_latest_scoring_rejects_current_cache_schema_mismatch(
    monkeypatch,
    tmp_path,
) -> None:
    target_date = pd.Timestamp("2026-08-12")
    signal_rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [target_date],
            "signal": [True],
        }
    )
    row = _active_cache_row("000001.SZ", target_date)
    row["factor_schema_version"] = "stale-schema"
    cache_path = tmp_path / "active.parquet"
    pd.DataFrame([row]).to_parquet(cache_path, index=False)
    monkeypatch.setattr(scoring, "_load_signal_cache", lambda signals, start_date: signal_rows)

    with pytest.raises(RuntimeError, match="schema mismatch"):
        scoring.build_latest_dataset(
            tmp_path,
            ["signal"],
            "2020-01-01",
            "2026-08-12",
            workers=1,
            feature_cache_path=cache_path,
        )


def test_latest_scoring_represents_zero_candidates_with_dated_manifest_data(
    monkeypatch,
    tmp_path,
) -> None:
    prior_date = pd.Timestamp("2026-08-11")
    signal_rows = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [prior_date],
            "signal": [True],
        }
    )
    monkeypatch.setattr(
        scoring,
        "_load_signal_cache",
        lambda signals, start_date: signal_rows,
    )

    result = scoring.build_latest_dataset(
        tmp_path,
        ["signal"],
        "2020-01-01",
        "2026-08-12",
        workers=1,
    )

    assert result.empty
    assert result.attrs["target_date"] == "2026-08-12"
    assert result.attrs["empty_candidate_set"] is True

    class Model:
        feature_names_in_ = ["feature"]

    coverage = scoring._empty_feature_coverage(
        {("signal", "up5"): Model()},
        "2026-08-12",
    )
    assert coverage["status"] == "valid"
    assert coverage["empty_candidate_set"] is True
    assert coverage["required_feature_count"] == 1
