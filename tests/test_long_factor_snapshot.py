from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pytest

from quant.data.atomic_io import atomic_write_parquet
from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.features.factor_registry import LONG_PRODUCTION_FACTOR_COLUMNS
from quant.features.long_factor_snapshot import publish_long_factor_snapshot, read_long_factor_snapshot
from quant.routine import daily_dependency_runtime
from quant.webapp import services


def _frame(day: str) -> pd.DataFrame:
    row = {column: 1.0 for column in LONG_PRODUCTION_FACTOR_COLUMNS}
    row.update(
        ts_code="000001.SZ", date=pd.Timestamp(day),
        factor_schema_version=services.LONG_FACTOR_SNAPSHOT_SCHEMA_VERSION,
    )
    return pd.DataFrame([row])


def test_concurrent_historical_writers_preserve_newest_snapshot(tmp_path):
    days = ["2026-08-26", "2026-07-23", "2026-08-24", "2026-08-25"] * 3
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda day: publish_long_factor_snapshot(_frame(day), tmp_path), days))

    latest, manifest = read_long_factor_snapshot(tmp_path, latest=True)
    assert manifest["signal_date"] == "2026-08-26"
    assert latest["date"].eq(pd.Timestamp("2026-08-26")).all()


@pytest.mark.parametrize("corruption", ["date", "value", "column", "duplicate"])
def test_latest_manifest_cannot_hide_corrupt_parquet(tmp_path, corruption):
    frame = _frame("2026-08-26")
    publish_long_factor_snapshot(frame, tmp_path)
    if corruption == "date":
        frame["date"] = pd.Timestamp("2026-07-23")
    elif corruption == "value":
        frame["roe"] = 99.0
    elif corruption == "column":
        frame = frame.drop(columns="roe")
    else:
        frame = pd.concat([frame, frame], ignore_index=True)
    atomic_write_parquet(frame, tmp_path / "latest.parquet", index=False)

    with pytest.raises(RuntimeError, match="long_snapshot"):
        read_long_factor_snapshot(tmp_path, "2026-08-26", latest=True)
    assert daily_dependency_runtime._evidence_value(
        tmp_path, {}, "long_factor_snapshot", ".", "signal_date",
    ) is None


def test_valid_dated_snapshot_repairs_regressed_latest_without_recalculation(monkeypatch, tmp_path):
    publish_long_factor_snapshot(_frame("2026-07-23"), tmp_path)
    atomic_write_parquet(_frame("2026-08-26"), tmp_path / "20260826.parquet", index=False)
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(services, "_selector_active_model_features", lambda: ("roe",))
    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", None)

    result = services._ensure_selector_long_factor_snapshot("2026-08-26", force_refresh=False)

    assert result["latest_restored_from_dated"] is True
    assert read_long_factor_snapshot(tmp_path, latest=True)[1]["signal_date"] == "2026-08-26"


def test_daily_dependency_cannot_fall_back_to_valid_manifest_when_parquet_is_corrupt(tmp_path):
    directory = tmp_path / "data/features/long"
    frame = _frame("2026-08-26")
    publish_long_factor_snapshot(frame, directory)
    before = daily_dependency_runtime.collect_node_states(DEFAULT_DAILY_DEPENDENCY_REGISTRY, tmp_path)
    assert "feature.long_snapshot" in before
    frame["roe"] = 99.0
    atomic_write_parquet(frame, directory / "latest.parquet", index=False)

    after = daily_dependency_runtime.collect_node_states(DEFAULT_DAILY_DEPENDENCY_REGISTRY, tmp_path)

    assert "feature.long_snapshot" not in after


def test_historical_read_does_not_use_future_latest(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(services, "LONG_FACTOR_SNAPSHOT_DIR", tmp_path / "long")
    monkeypatch.setattr(services, "_selector_active_model_features", lambda: ("roe",))
    publish_long_factor_snapshot(_frame("2026-08-26"), tmp_path / "long")
    old = _frame("2026-08-24")
    old["roe"] = 2.0
    publish_long_factor_snapshot(old, tmp_path / "long")

    rows = services._selector_production_snapshot_rows("2026-08-24")

    assert rows["000001.SZ"]["roe"] == 2.0
    assert rows["000001.SZ"]["_selector_layer_coverage"]["long_snapshot"]["date"] == "2026-08-24"


def test_native_missing_values_cannot_mask_missing_layer(monkeypatch):
    artifact = {
        "features": ["close", "pct_chg", "roe"],
        "preprocessing": "xgboost_native_nan_float32_v1",
        "model": object(),
    }
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {"buy": artifact, "hold": artifact})
    row = {"symbol": "000001.SZ", "date": "2026-08-26", "matched_groups": []}
    values = {
        "date": "2026-08-26", "close": 10.0, "pct_chg": 2.0, "roe": np.nan,
        "_score_feature_source": "live_daily",
        "_selector_layer_coverage": {"project_daily": {"status": "complete", "date": "2026-08-26"}},
    }

    services._apply_historical_score_normalization([row], {"000001.SZ": values})

    assert row["model_score_available"] is False
    assert row["feature_quality"]["missing_layers"] == ["long_snapshot"]


def test_scoring_failure_clears_previously_cached_model_success(monkeypatch):
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {})
    row = {
        "symbol": "000001.SZ", "date": "2026-08-26",
        "buy_score_source": "historical_return_model", "hold_score_source": "historical_return_model",
        "historical_buy_score": 88.0, "historical_hold_score": 90.0,
        "model_score_available": True,
    }

    services._apply_historical_score_normalization([row])

    assert row["model_score_available"] is False
    assert "buy_score_source" not in row
    assert "hold_score_source" not in row
    assert "historical_buy_score" not in row
    assert row["feature_quality"]["status"] == "failed"


@pytest.mark.parametrize("quality", [{"status": "failed"}, {}, None])
def test_incomplete_scoring_never_overwrites_published_page(monkeypatch, tmp_path, quality):
    snapshot_dir = tmp_path / "snapshots"
    monkeypatch.setattr(services, "SELECTOR_SNAPSHOT_DIR", snapshot_dir)
    def unexpected_store(config):
        raise AssertionError("publication reached storage before validating model scores")
    monkeypatch.setattr(services, "MarketDataStore", unexpected_store)
    payload = {"stocks": [{
        "symbol": "000001.SZ", "model_score_available": False,
        "feature_quality": quality,
    }]}

    with pytest.raises(RuntimeError, match="incomplete model scores"):
        services._write_selector_snapshot_batch([(payload, None, True)])

    assert not snapshot_dir.exists()


def test_failed_model_scores_are_not_displayed_as_numeric_model_results():
    row = {
        "opportunity_score": 88.0, "holding_score": 90.0,
        "model_score_available": False, "feature_quality": {"status": "failed"},
    }

    services._apply_selector_score_presentation([row])

    assert row["buy_score_normalized"] is None
    assert row["hold_score_normalized"] is None
    assert row["buy_score_rank"] is None


def test_hold_success_does_not_hide_buy_failure(monkeypatch):
    class Model:
        def __init__(self, fail):
            self.fail = fail

        def predict(self, values):
            if self.fail:
                raise RuntimeError("buy failed")
            return np.array([1.0])

    artifacts = {
        mode: {"features": ["close"], "model": Model(mode == "buy"), "score_reference": np.array([0.0, 1.0])}
        for mode in ("buy", "hold")
    }
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: artifacts)
    row = {"symbol": "000001.SZ", "date": "2026-08-26"}

    services._apply_historical_score_normalization([row], {"000001.SZ": {"date": "2026-08-26", "close": 10.0}})

    assert row["model_score_available"] is False
    assert row["feature_quality"]["status"] == "failed"
    assert "buy failed" in row["feature_quality"]["error"]
