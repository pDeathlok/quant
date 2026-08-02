import json
import queue
import threading
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quant.research import similar_patterns as similar_patterns_module
from quant.research.similar_patterns import SimilarPatternResult, TargetSpec, summarize_forecast
from quant.webapp.app import app
from quant.webapp import services
from quant.webapp import api as webapp_api


client = TestClient(app)


def _stub_successful_global_refresh(
    monkeypatch,
    *,
    build_features=None,
    generate_daily_plan=None,
    generate_dashboard=None,
    score_latest_models=None,
    refresh_chan_model_scores=None,
    convertible_bond_plan=None,
    convertible_bond_allotment=None,
    byd_daily_plan=None,
):
    from quant.routine import pipeline

    status = {
        "status": "idle",
        "run_id": None,
        "started_at": None,
        "finished_at": None,
        "updated_at": None,
        "message": "idle",
        "percent": 0,
        "current_step": None,
        "steps": [],
        "result": None,
        "error": None,
    }
    success = lambda *args, **kwargs: {"status": "success"}
    source_success = lambda *args, **kwargs: {
        "status": "success",
        "expected_trade_date": "20260723",
        "dataset_trade_date": "20260723",
    }
    daily_basic_success = lambda *args, **kwargs: {
        "status": "success",
        "latest_trade_date": "20260723",
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services, "_write_terminal_refresh_manifest_unlocked", lambda: None)
    monkeypatch.setattr(services, "run_cache_cleanup", lambda project_root: {"status": "success"})
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260723")
    monkeypatch.setattr(pipeline, "refresh_data", source_success)
    monkeypatch.setattr(pipeline, "refresh_daily_basic_data", daily_basic_success)
    monkeypatch.setattr(pipeline, "refresh_reference_inputs", success)
    monkeypatch.setattr(pipeline, "build_features", build_features or success)
    monkeypatch.setattr(pipeline, "refresh_strategy_signal_cache", success)
    monkeypatch.setattr(pipeline, "generate_daily_plan", generate_daily_plan or success)
    monkeypatch.setattr(pipeline, "generate_dashboard", generate_dashboard or success)
    monkeypatch.setattr(pipeline, "score_latest_models", score_latest_models or success)
    monkeypatch.setattr(
        pipeline,
        "refresh_chan_model_scores",
        refresh_chan_model_scores or success,
    )
    monkeypatch.setattr(services, "_clear_selector_caches", lambda: None)
    monkeypatch.setattr(
        services,
        "get_stock_selector_payload",
        lambda **kwargs: {"signal_date": "2026-07-23", "stocks": []},
    )
    monkeypatch.setattr(
        services,
        "get_convertible_bond_grid_plan",
        convertible_bond_plan
        or (lambda *args, **kwargs: {"trade_date": "20260723", "candidates": []}),
    )
    monkeypatch.setattr(
        services,
        "get_convertible_bond_allotments",
        convertible_bond_allotment
        or (lambda *args, **kwargs: {"asof": "2026-07-23", "records": []}),
    )
    monkeypatch.setattr(
        services,
        "get_byd_daily_strategy",
        byd_daily_plan
        or (lambda *args, **kwargs: {"planned_t": {"signal_date": "2026-07-23"}, "alerts": []}),
    )
    monkeypatch.setattr(
        services,
        "get_chan_model_strategy_plan",
        lambda *args, **kwargs: {
            "signal_date": "2026-07-23",
            "candidates": [],
            "primary_count": 0,
            "expanded_count": 0,
        },
    )
    monkeypatch.setattr(
        services,
        "_refresh_long_stock_pool_variants",
        lambda variants, signal_date: [
            {"variant": variant, "signal_date": signal_date, "stocks": 0}
            for variant in variants
        ],
    )
    monkeypatch.setattr(
        services,
        "_run_similar_pattern_analysis_isolated",
        lambda: {"generated_at": "2026-07-23T18:00:00", "results": []},
    )
    monkeypatch.setattr(services, "_write_strategy_pool_snapshots", lambda *args, **kwargs: [])
    monkeypatch.setattr(services, "_run_post_snapshot_cache_cleanup", lambda results: None)

    class DummyStoreConfig:
        @classmethod
        def from_env(cls, *args, **kwargs):
            return cls()

    class DummyStore:
        def __init__(self, config):
            self.config = type("Config", (), {"sql_url": None})()

        def latest_dataset_trade_date(self, dataset):
            return pd.Timestamp("2026-07-23")

    monkeypatch.setattr(services, "MarketDataStoreConfig", DummyStoreConfig)
    monkeypatch.setattr(services, "MarketDataStore", DummyStore)
    return status


def test_historical_scores_are_independent_of_current_candidate_pool(monkeypatch) -> None:
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {})
    base_row = {
        "symbol": "000001.SZ",
        "historical_buy_score": 72.4,
        "historical_hold_score": 61.3,
    }
    other_row = {
        "symbol": "000002.SZ",
        "historical_buy_score": 99.0,
        "historical_hold_score": 5.0,
    }

    score_alone = services._apply_historical_score_normalization([dict(base_row)])[0]
    score_with_other = services._apply_historical_score_normalization([dict(base_row), dict(other_row)])[0]

    assert score_alone["opportunity_score"] == score_with_other["opportunity_score"] == 72.4
    assert score_alone["holding_score"] == score_with_other["holding_score"] == 61.3
    assert score_alone["selector_score"] == 72.4
    assert score_alone["score_target"] == "historical_return_model_score"


def test_return_model_scores_use_fixed_historical_reference(monkeypatch) -> None:
    class FakeImputer:
        def transform(self, frame):
            return frame[["selector_return_5d"]].to_numpy()

    class FakeModel:
        def predict(self, values):
            return values[:, 0]

    artifact = {
        "features": ["selector_return_5d"],
        "imputer": FakeImputer(),
        "model": FakeModel(),
        "score_reference": np.array([0.0, 1.0, 2.0, 3.0]),
    }
    monkeypatch.setattr(services, "_selector_buy_hold_models", lambda: {"buy": artifact})
    monkeypatch.setattr(
        services,
        "_selector_model_feature_rows",
        lambda signal_date: {
            "000001.SZ": {"selector_return_5d": 1.0},
            "000002.SZ": {"selector_return_5d": 3.0},
        },
    )
    rows = [
        {"symbol": "000001.SZ", "date": "2026-07-21", "matched_groups": ["B1"]},
        {"symbol": "000002.SZ", "date": "2026-07-21", "matched_groups": ["B2"]},
    ]

    services._apply_return_model_scores(rows)

    assert rows[0]["historical_buy_score"] < rows[1]["historical_buy_score"]
    assert rows[1]["historical_buy_score"] < 100.0
    assert rows[0]["buy_score_source"] == "historical_return_model"


def test_watchlist_buy_hold_scores_accepts_numpy_matched_groups(monkeypatch) -> None:
    observed_groups = []
    monkeypatch.setattr(
        services,
        "_latest_similar_pattern_target_date",
        lambda symbols: "2026-07-21",
    )
    monkeypatch.setattr(
        services,
        "_selector_model_score_date",
        lambda market_date: "2026-07-21",
    )
    monkeypatch.setattr(
        services,
        "_selector_model_feature_rows",
        lambda signal_date: {
            "000001.SZ": {
                "matched_groups": np.array(["B1", "B2"]),
                "best_profit_factor": 1.5,
                "best_avg_return_pct": 2.0,
            },
        },
    )

    def apply_scores(rows, feature_rows):
        observed_groups.extend(row["matched_groups"] for row in rows)
        for row in rows:
            row.update(
                opportunity_score=70.0,
                holding_score=60.0,
                score_target="historical_return_model_score",
                buy_score_source="historical_return_model",
                hold_score_source="historical_return_model",
            )

    monkeypatch.setattr(services, "_apply_historical_score_normalization", apply_scores)

    result = services._watchlist_buy_hold_scores(("000001.SZ",))

    assert observed_groups == [["B1", "B2"]]
    assert result["000001.SZ"]["opportunity_score"] == 70.0


def test_family_signal_cache_does_not_fall_back_to_prior_trade_date(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "family.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "date": [pd.Timestamp("2026-07-23")],
            "signal_b1": [1],
        }
    ).to_parquet(cache, index=False)
    monkeypatch.setattr(services, "FAMILY_SIGNAL_CACHE", cache)
    monkeypatch.setattr(
        services,
        "_model_scored_candidates_for_date",
        lambda signal_date: {},
    )
    services._family_signals_for_date.cache_clear()

    signals = services._family_signals_for_date("2026-07-24")

    assert signals == {}
    services._family_signals_for_date.cache_clear()


@pytest.mark.parametrize("mode", ["buy", "hold"])
def test_historical_buy_and_hold_scores_preserve_return_order(monkeypatch, mode: str) -> None:
    distribution = np.linspace(-20.0, 20.0, 401)
    monkeypatch.setattr(
        services,
        "_selector_score_calibration",
        lambda: {
            "global": {mode: distribution},
            "group": {mode: {"B1": distribution}},
        },
    )
    common = {
        "strategy_group": "B1",
        "metrics": {
            "trades": 240,
            "win_rate": 0.5,
            "max_drawdown_pct": 10.0,
            "profit_factor": 1.5,
        },
    }
    lower_return = {**common, "metrics": {**common["metrics"], "avg_return_pct": 1.0}}
    higher_return = {**common, "metrics": {**common["metrics"], "avg_return_pct": 5.0}}

    assert services._normalized_signal_score(higher_return, mode=mode) > services._normalized_signal_score(
        lower_return,
        mode=mode,
    )


def test_api_refresh_runs_cleanup_before_data_refresh_even_when_refresh_fails(monkeypatch) -> None:
    from quant.routine import pipeline

    call_order: list[str] = []
    cleanup_result = {"status": "success", "reclaimed_bytes": 456, "errors": []}
    status = {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "updated_at": None,
        "message": "idle",
        "percent": 0,
        "current_step": None,
        "steps": [],
        "result": None,
        "error": None,
    }

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services, "_write_terminal_refresh_manifest_unlocked", lambda: None)
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda project_root: call_order.append("cleanup") or cleanup_result,
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_data",
        lambda **kwargs: call_order.append("refresh") or {"status": "failed", "stderr_tail": "boom"},
    )

    services._run_latest_refresh_job("all")

    assert call_order == ["cleanup", "refresh"]
    assert status["status"] == "failed"
    assert status["result"]["cache_cleanup"] == cleanup_result


def test_global_refresh_starts_independent_workspaces_before_feature_build(
    monkeypatch,
) -> None:
    rendezvous = threading.Barrier(4, timeout=3)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def early_step(name: str, payload: dict):
        def run(*args, **kwargs):
            with calls_lock:
                calls.append(name)
            rendezvous.wait()
            return payload

        return run

    def build_features(**kwargs):
        rendezvous.wait()
        return {"status": "success"}

    status = _stub_successful_global_refresh(
        monkeypatch,
        build_features=build_features,
        convertible_bond_plan=early_step(
            "convertible_bond_plan",
            {"trade_date": "20260723", "candidates": []},
        ),
        convertible_bond_allotment=early_step(
            "convertible_bond_allotment",
            {"asof": "2026-07-23", "records": []},
        ),
        byd_daily_plan=early_step(
            "byd_daily_plan",
            {"planned_t": {"signal_date": "2026-07-23"}, "alerts": []},
        ),
    )

    services._run_latest_refresh_job("all", run_id="parallel-early-workspaces")

    assert status["status"] == "success"
    assert sorted(calls) == [
        "byd_daily_plan",
        "convertible_bond_allotment",
        "convertible_bond_plan",
    ]
    assert status["result"]["convertible_bond_plan"]["status"] == "success"
    assert status["result"]["convertible_bond_allotment"]["status"] == "success"
    assert status["result"]["byd_daily_plan"]["status"] == "success"


def test_allotment_quality_accepts_negative_j_values() -> None:
    payload = {
        "records": [
            {
                "stock_price_date": "2026-07-23",
                "kdj_daily_j": -6.58,
                "kdj_weekly_j": -9.08,
                "kdj_monthly_j": -3.21,
            }
        ],
        "data_sources": {
            "stock_daily": {"requested": 1, "matched": 1, "error": None},
            "daily_basic": {"matched": 1, "error": None},
        },
    }

    quality = services._convertible_bond_allotment_quality(
        payload,
        expected_trade_date="20260723",
    )

    assert quality["status"] == "success"
    assert quality["metrics"]["kdj_weekly_j"]["count"] == 1
    assert quality["metrics"]["kdj_monthly_j"]["count"] == 1


def test_allotment_quality_gate_rejects_stale_or_incomplete_payload(
    monkeypatch,
    tmp_path,
) -> None:
    payload = {
        "generated_at": "2026-07-23T18:00:00",
        "asof": "2026-07-23",
        "records": [
            {
                "stock_price_date": "2026-07-22",
                "kdj_daily_j": 12.0,
                "kdj_weekly_j": None,
                "kdj_monthly_j": None,
            }
        ],
        "data_sources": {
            "stock_daily": {"requested": 1, "matched": 1, "error": None},
            "daily_basic": {"matched": 1, "error": None},
        },
    }
    writes = []
    monkeypatch.setattr(
        services,
        "CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH",
        tmp_path / "allotments.json",
    )
    monkeypatch.setattr(
        services,
        "build_convertible_bond_allotment_payload",
        lambda **kwargs: payload,
    )
    monkeypatch.setattr(
        services,
        "_write_workspace_snapshot",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="配债股数据质量门禁失败") as exc_info:
        services.get_convertible_bond_allotments(
            refresh=True,
            expected_trade_date="2026-07-23",
            validate_quality=True,
        )

    assert "行情日期新鲜度不足" in str(exc_info.value)
    assert "周线 J 完整率不足" in str(exc_info.value)
    assert "月线 J 完整率不足" in str(exc_info.value)
    assert payload["quality"]["status"] == "failed"
    assert writes


def test_allotment_scope_refreshes_daily_basic_inputs(monkeypatch) -> None:
    from quant.routine import pipeline

    status = _stub_successful_global_refresh(monkeypatch)
    daily_basic_calls = []
    monkeypatch.setattr(
        pipeline,
        "refresh_daily_basic_data",
        lambda **kwargs: daily_basic_calls.append(kwargs)
        or {"status": "success", "latest_trade_date": "20260723"},
    )

    services._run_latest_refresh_job(
        "cbAllotment",
        run_id="allotment-daily-basic-refresh",
    )

    assert status["status"] == "success"
    assert len(daily_basic_calls) == 1


def test_global_refresh_parallelizes_outputs_and_caps_cpu_workers(monkeypatch) -> None:
    rendezvous = threading.Barrier(4, timeout=3)
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    worker_args: dict[str, int] = {}

    def parallel_output(payload: dict):
        def run(*args, **kwargs):
            nonlocal active, max_active
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            rendezvous.wait()
            with active_lock:
                active -= 1
            return payload

        return run

    plan = parallel_output({"status": "success", "output": "plan.json"})
    dashboard = parallel_output({"status": "success", "output": "dashboard.json"})
    score_body = parallel_output({"status": "success"})
    chan_body = parallel_output({"status": "success"})

    def score(*, workers: int):
        worker_args["model"] = workers
        return score_body()

    def chan(*, progress_callback, workers: int):
        worker_args["chan"] = workers
        return chan_body()

    monkeypatch.setenv("ROUTINE_MODEL_SCORE_WORKERS", "99")
    monkeypatch.setenv("ROUTINE_CHAN_WORKERS", "99")
    status = _stub_successful_global_refresh(
        monkeypatch,
        generate_daily_plan=plan,
        generate_dashboard=dashboard,
        score_latest_models=score,
        refresh_chan_model_scores=chan,
    )

    services._run_latest_refresh_job("all", run_id="parallel-daily-outputs")

    assert status["status"] == "success"
    assert max_active == 4
    assert worker_args == {"model": 4, "chan": 4}
    assert status["result"]["generate_daily_plan"]["status"] == "success"
    assert status["result"]["generate_dashboard"]["status"] == "success"
    assert status["result"]["model_score"]["status"] == "success"
    assert status["result"]["refresh_chan_model_scores"]["status"] == "success"


def test_global_refresh_propagates_early_workspace_failure(monkeypatch) -> None:
    calls = 0

    def fail_convertible_bond(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("bond upstream unavailable")

    status = _stub_successful_global_refresh(
        monkeypatch,
        convertible_bond_plan=fail_convertible_bond,
    )

    services._run_latest_refresh_job("all", run_id="parallel-early-failure")

    assert calls == 1
    assert status["status"] == "failed"
    assert "可转债策略计划刷新失败" in status["error"]
    assert "bond upstream unavailable" in status["error"]


def test_global_refresh_checkpoints_early_results_before_terminal_failure(
    monkeypatch,
) -> None:
    rendezvous = threading.Barrier(4, timeout=3)

    def early_success(payload: dict):
        def run(*args, **kwargs):
            rendezvous.wait()
            return payload

        return run

    def fail_features(**kwargs):
        rendezvous.wait()
        raise RuntimeError("feature build failed")

    status = _stub_successful_global_refresh(
        monkeypatch,
        build_features=fail_features,
        convertible_bond_plan=early_success(
            {"trade_date": "20260723", "candidates": [{"code": "123001"}]},
        ),
        convertible_bond_allotment=early_success(
            {"asof": "2026-07-23", "records": [{"code": "600001"}]},
        ),
        byd_daily_plan=early_success(
            {
                "planned_t": {"signal_date": "2026-07-23"},
                "alerts": [{"level": "info"}],
            },
        ),
    )

    services._run_latest_refresh_job("all", run_id="parallel-early-checkpoint")

    assert status["status"] == "failed"
    assert "feature build failed" in status["error"]
    assert status["result"]["convertible_bond_plan"]["candidates"] == 1
    assert status["result"]["convertible_bond_allotment"]["records"] == 1
    assert status["result"]["byd_daily_plan"]["alerts"] == 1


def test_terminal_refresh_writes_immutable_manifest_with_step_timings(
    monkeypatch,
    tmp_path,
) -> None:
    status = {
        "status": "running",
        "run_id": "all-20260723170000-abcd1234",
        "started_at": "2026-07-23T17:00:00",
        "finished_at": None,
        "updated_at": "2026-07-23T17:00:00",
        "message": "running",
        "percent": 1,
        "current_step": "refresh_data",
        "steps": services._progress_steps("chan"),
        "scope": "chan",
        "scope_label": "缠论策略",
        "result": None,
        "error": None,
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "REFRESH_STATUS_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(services, "REFRESH_MANIFEST_ROOT", tmp_path / "runs")
    services._REFRESH_CONTEXT.run_id = status["run_id"]
    try:
        services._set_refresh_progress(
            step_key="refresh_data",
            message="data done",
            percent=35,
            step_status="success",
            complete_previous=False,
        )
        services._set_refresh_progress(
            status="success",
            step_key="chan_model_strategy",
            message="done",
            percent=100,
            result={"chan_model_strategy": {"status": "success"}},
        )
    finally:
        delattr(services._REFRESH_CONTEXT, "run_id")

    manifest_path = Path(status["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "web_daily_refresh"
    assert payload["status"] == "success"
    assert all(step["started_at"] for step in payload["steps"])
    assert all(step["finished_at"] for step in payload["steps"])
    assert all(step["elapsed_seconds"] is not None for step in payload["steps"])
    assert (tmp_path / "latest.json").exists()

    original = manifest_path.read_text(encoding="utf-8")
    status["message"] = "must not overwrite terminal manifest"
    services._write_terminal_refresh_manifest_unlocked()
    assert manifest_path.read_text(encoding="utf-8") == original


def test_checkpoint_reused_step_has_zero_attempt_elapsed(
    monkeypatch,
) -> None:
    status = {
        "status": "running",
        "run_id": "resume-attempt",
        "attempt": 2,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": services._progress_steps("short"),
        "percent": 35,
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    services._REFRESH_CONTEXT.run_id = status["run_id"]
    try:
        services._set_refresh_progress(
            step_key="refresh_data",
            step_status="success",
            message="checkpoint reused",
            percent=35,
            complete_previous=False,
            checkpoint_reused=True,
        )
    finally:
        delattr(services._REFRESH_CONTEXT, "run_id")

    step = next(
        item
        for item in status["steps"]
        if item["key"] == "refresh_data"
    )
    assert step["status"] == "success"
    assert step["checkpoint_reused"] is True
    assert step["elapsed_seconds"] == 0.0
    assert step["started_at"] == step["finished_at"]


def test_resumed_refresh_starts_independent_attempt_timing(
    monkeypatch,
) -> None:
    old_steps = services._progress_steps("all")
    old_steps[0].update(
        {
            "status": "success",
            "started_at": "2026-07-27T15:00:00",
            "finished_at": "2026-07-27T15:05:00",
            "elapsed_seconds": 300.0,
        }
    )
    status = {
        "status": "failed",
        "run_id": "old-run",
        "attempt": 1,
        "started_at": "2026-07-27T15:00:00",
        "finished_at": "2026-07-27T15:30:00",
        "elapsed_seconds": 1800.0,
        "updated_at": "2026-07-27T15:30:00",
        "message": "failed",
        "percent": 35,
        "current_step": "feature_cache",
        "steps": old_steps,
        "scope": "all",
        "result": {},
        "error": "old failure",
        "manifest_path": "/tmp/old-run/manifest.json",
    }

    class InlineThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(
        services,
        "_refresh_resume_ready",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(services.threading, "Thread", InlineThread)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)

    payload = services.start_latest_refresh("all")

    assert payload["status"] == "queued"
    assert payload["run_id"] != "old-run"
    assert payload["attempt"] == 2
    assert payload["started_at"] != "2026-07-27T15:00:00"
    assert payload["resumed_from"] == {
        "run_id": "old-run",
        "attempt": 1,
        "started_at": "2026-07-27T15:00:00",
        "finished_at": "2026-07-27T15:30:00",
        "elapsed_seconds": 1800.0,
        "manifest_path": "/tmp/old-run/manifest.json",
    }
    assert all(step["started_at"] is None for step in payload["steps"])
    assert all(step["elapsed_seconds"] is None for step in payload["steps"])
    assert all(
        step["checkpoint_reused"] is False
        for step in payload["steps"]
    )


def test_input_resume_marks_only_reused_steps_as_checkpoints(
    monkeypatch,
) -> None:
    status = _stub_successful_global_refresh(monkeypatch)
    source_started_at = pd.Timestamp.now().isoformat()
    resume_status = {
        "status": "failed",
        "run_id": "source-attempt",
        "attempt": 1,
        "started_at": source_started_at,
        "finished_at": source_started_at,
        "elapsed_seconds": 900.0,
        "steps": services._progress_steps("all"),
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260723",
                "dataset_trade_date": "20260723",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260723",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "steps": {
                    "analyst_forecast_snapshot": {"status": "success"}
                },
            },
            "feature_cache": {"status": "success"},
            "model_score": {"status": "success"},
        },
    }
    monkeypatch.setattr(
        services.pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame(
            {"date": [pd.Timestamp("2026-07-23")]}
        ),
    )

    services._run_latest_refresh_job(
        "all",
        resume_status=resume_status,
        run_id="resumed-run",
    )

    assert status["status"] == "success"
    assert status["attempt"] == 2
    assert status["started_at"] != source_started_at
    step_map = {
        step["key"]: step
        for step in status["steps"]
    }
    for key in (
        "refresh_data",
        "feature_cache",
        "signal_cache",
        "model_score",
    ):
        assert step_map[key]["checkpoint_reused"] is True
        assert step_map[key]["elapsed_seconds"] == 0.0
    assert step_map["daily_plan"]["checkpoint_reused"] is False


def test_starting_refresh_clears_stale_manifest_path(monkeypatch, tmp_path) -> None:
    stale_manifest = tmp_path / "old" / "manifest.json"
    status = {
        "status": "failed",
        "run_id": "old-run",
        "started_at": "2026-07-23T17:00:00",
        "finished_at": "2026-07-23T17:30:00",
        "updated_at": "2026-07-23T17:30:00",
        "message": "failed",
        "percent": 35,
        "current_step": "feature_cache",
        "steps": services._progress_steps("all"),
        "scope": "all",
        "scope_label": "全部工作区",
        "result": None,
        "error": "old error",
        "manifest_path": str(stale_manifest),
        "manifest_error": "old manifest error",
    }

    class InlineThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_refresh_resume_ready", lambda *args, **kwargs: False)
    monkeypatch.setattr(services.threading, "Thread", InlineThread)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)

    payload = services.start_latest_refresh("all")

    assert payload["status"] == "queued"
    assert payload["manifest_path"] is None
    assert payload["manifest_error"] is None


def test_health_endpoint_reports_service_status() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "quant-webapp"}


def test_byd_daily_plan_only_forwards_permanent_holding(monkeypatch) -> None:
    captured = {}

    def fake_strategy(**kwargs):
        captured.update(kwargs)
        return {"validation": {"execution_enabled": False}, "alerts": []}

    monkeypatch.setattr(webapp_api, "get_byd_daily_strategy", fake_strategy)

    response = client.get(
        "/api/byd/daily-plan",
        params={
            "shares": 10300,
            "cost": 108.25,
            "refresh": True,
        },
    )

    assert response.status_code == 200
    assert captured["shares"] == 10300
    assert captured["cost"] == 108.25
    assert captured["refresh"] is True
    assert set(captured) == {"shares", "cost", "refresh"}


def test_refresh_endpoint_rejects_unknown_scope() -> None:
    response = client.post("/api/selector/refresh-latest", json={"scope": "unknown"})

    assert response.status_code == 400
    assert "未知刷新范围" in response.json()["detail"]


def test_global_refresh_includes_daily_similar_pattern_step() -> None:
    step_keys = [step["key"] for step in services._progress_steps("all")]

    assert "convertible_bond_allotment" in step_keys
    assert "similar_patterns" in step_keys


def test_similar_refresh_scope_runs_only_the_isolated_analysis(monkeypatch) -> None:
    status = {
        "status": "queued",
        "run_id": "similar-run",
        "attempt": 1,
        "started_at": "2026-07-30T20:00:00",
        "steps": services._progress_steps("similar"),
        "scope": "similar",
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services, "_write_terminal_refresh_manifest_unlocked", lambda: None)
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda *args, **kwargs: pytest.fail("单独刷新自选分析不应清理全站缓存"),
    )
    monkeypatch.setattr(
        services,
        "_run_similar_pattern_analysis_isolated",
        lambda: {
            "generated_at": "2026-07-30T20:01:00",
            "results": [{"target": {"symbol": "002594.SZ"}}],
        },
    )

    services._run_latest_refresh_job("similar", run_id="similar-run")

    assert [step["key"] for step in status["steps"]] == ["similar_patterns"]
    assert status["status"] == "success"
    assert status["percent"] == 100
    assert status["result"]["similar_patterns"]["targets"] == 1


def test_refresh_progress_ignores_stale_run_context(monkeypatch) -> None:
    status = {
        "status": "running",
        "run_id": "current-run",
        "started_at": "2026-07-22T19:00:00",
        "finished_at": None,
        "updated_at": "2026-07-22T19:00:00",
        "message": "current",
        "percent": 10,
        "current_step": "refresh_data",
        "steps": services._progress_steps("all"),
        "result": None,
        "error": None,
    }

    monkeypatch.setattr(services, "_REFRESH_STATUS", status)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    services._REFRESH_CONTEXT.run_id = "old-run"
    try:
        services._set_refresh_progress(
            step_key="feature_cache",
            message="old worker should not win",
            percent=40,
        )
    finally:
        delattr(services._REFRESH_CONTEXT, "run_id")

    assert status["message"] == "current"
    assert status["percent"] == 10
    assert status["current_step"] == "refresh_data"


def test_expiring_stale_refresh_terminates_known_child_processes(monkeypatch) -> None:
    status = {
        "status": "running",
        "run_id": "stale-run",
        "started_at": "2000-01-01T00:00:00",
        "finished_at": None,
        "updated_at": "2000-01-01T00:00:00",
        "message": "stale",
        "percent": 35,
        "current_step": "feature_cache",
        "steps": [{"key": "feature_cache", "status": "running"}],
        "result": None,
        "error": None,
    }
    killed: list[tuple[int, int]] = []

    class Result:
        returncode = 0
        stdout = "\n".join(
            [
                "123 /Users/didi/miniforge3/bin/python scripts/research/refresh_b1_feature_cache.py --workers 8",
                "456 /Users/didi/miniforge3/bin/python scripts/research/rebuild_strategy_signal_cache.py --workers 8",
            ]
        )
        stderr = ""

    monkeypatch.setattr(services, "_REFRESH_STATUS", dict(status))
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services.subprocess, "run", lambda *args, **kwargs: Result())
    monkeypatch.setattr(services.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    expired = services._expire_stale_refresh_status_unlocked(status)

    assert expired["status"] == "failed"
    assert expired["steps"][0]["status"] == "failed"
    assert (123, services.signal.SIGTERM) in killed
    assert (456, services.signal.SIGTERM) in killed
    assert "已终止卡住的后台节点" in expired["error"]


def test_latest_extended_signals_reuses_candidate_parquet(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "z_skill_daily_candidates.parquet"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "date": pd.Timestamp("2026-07-19"),
                "name": "平安银行",
                "close": 10.0,
                "KEY_K": True,
                "NANA": False,
            },
            {
                "symbol": "000002.SZ",
                "date": pd.Timestamp("2026-07-20"),
                "name": "万科A",
                "close": 8.0,
                "KEY_K": False,
                "NANA": True,
            },
        ]
    ).to_parquet(cache_path)

    monkeypatch.setattr(services, "EXTENDED_SIGNAL_CACHE", tmp_path / "missing.json")
    monkeypatch.setattr(services, "EXTENDED_CANDIDATE_CACHE", cache_path)
    monkeypatch.setattr(
        services,
        "_stock_basic_map",
        lambda: {
            "000001.SZ": {"name": "平安银行", "industry": "银行"},
            "000002.SZ": {"name": "万科A", "industry": "地产"},
        },
    )
    monkeypatch.setattr(
        services,
        "build_extended_daily_signals",
        lambda *args, **kwargs: pytest.fail("should reuse candidate parquet"),
    )
    services._latest_extended_signals.cache_clear()

    signals = services._latest_extended_signals("2026-07-20")

    assert set(signals) == {"000002.SZ"}
    assert signals["000002.SZ"]["industry"] == "地产"
    assert signals["000002.SZ"]["signals"][0]["strategy_key"] == "NANA"


def test_long_stock_pool_variants_run_in_parallel_and_keep_order(monkeypatch) -> None:
    observed = []

    class FakeExecutor:
        def __init__(self, max_workers):
            observed.append(("max_workers", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            observed.extend(("variant", item) for item in items)
            return [fn(item) for item in reversed(list(items))]

    monkeypatch.setattr(services, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        services,
        "_refresh_long_stock_pool_variant",
        lambda variant, signal_date: {"variant": variant, "signal_date": signal_date, "stocks": len(variant)},
    )

    result = services._refresh_long_stock_pool_variants(["tea", "tea_safe", "v44"], "2026-07-20")

    assert ("max_workers", 3) in observed
    assert [item["variant"] for item in result] == ["tea", "tea_safe", "v44"]
    assert [item["signal_date"] for item in result] == ["2026-07-20"] * 3


def test_long_research_module_waits_for_concurrent_import(monkeypatch) -> None:
    module_name = "quant_long_dividend_quality_research"
    import_started = threading.Event()
    allow_import_to_finish = threading.Event()
    second_returned = threading.Event()
    import_calls = []

    class FakeLoader:
        def exec_module(self, module) -> None:
            import_calls.append(module)
            import_started.set()
            assert allow_import_to_finish.wait(timeout=2)
            module.load_stock_basic = lambda: pd.DataFrame()

    class FakeSpec:
        loader = FakeLoader()

    class FakeModule:
        pass

    monkeypatch.delitem(services.sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        services.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: FakeSpec(),
    )
    monkeypatch.setattr(
        services.importlib.util,
        "module_from_spec",
        lambda spec: FakeModule(),
    )
    services._long_research_module.cache_clear()

    def load_second():
        module = services._long_research_module()
        second_returned.set()
        return module

    try:
        with services.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(services._long_research_module)
            assert import_started.wait(timeout=2)
            second = executor.submit(load_second)
            returned_before_import_finished = second_returned.wait(timeout=0.1)
            allow_import_to_finish.set()
            first_module = first.result(timeout=2)
            second_module = second.result(timeout=2)
    finally:
        allow_import_to_finish.set()
        services._long_research_module.cache_clear()

    assert not returned_before_import_finished
    assert first_module is second_module
    assert len(import_calls) == 1
    assert first_module._quant_services_import_complete is True


def test_tea_master_research_module_discards_partial_import(monkeypatch) -> None:
    module_name = "quant_tea_master_long_research"
    imported_modules = []

    class FakeLoader:
        def exec_module(self, module) -> None:
            imported_modules.append(module)
            module.CONFIGS = []
            if len(imported_modules) == 1:
                raise RuntimeError("incomplete import")

    class FakeSpec:
        loader = FakeLoader()

    class FakeModule:
        pass

    monkeypatch.delitem(services.sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        services.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: FakeSpec(),
    )
    monkeypatch.setattr(
        services.importlib.util,
        "module_from_spec",
        lambda spec: FakeModule(),
    )
    services._tea_master_research_module.cache_clear()

    try:
        with pytest.raises(RuntimeError, match="incomplete import"):
            services._tea_master_research_module()
        recovered = services._tea_master_research_module()
    finally:
        services._tea_master_research_module.cache_clear()

    assert len(imported_modules) == 2
    assert recovered is imported_modules[1]
    assert recovered._quant_services_import_complete is True


def test_long_stock_pool_worker_does_not_redirect_process_stdout(monkeypatch) -> None:
    original_stdout = services.sys.stdout
    observed_stdout = []
    payload = {"signal_date": "2026-07-30", "stocks": []}

    def build(variant, signal_date):
        observed_stdout.append(services.sys.stdout)
        return payload

    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", build)
    monkeypatch.setattr(services, "_write_long_stock_pool_snapshot", lambda *args, **kwargs: None)

    result = services._refresh_long_stock_pool_variant("tea", "2026-07-30")

    assert observed_stdout == [original_stdout]
    assert result == {
        "variant": "tea",
        "signal_date": "2026-07-30",
        "stocks": 0,
    }


def test_live_long_base_skips_persistent_backtest_cache_and_reuses_memory(monkeypatch) -> None:
    calls = []
    features = pd.DataFrame({"date": pd.to_datetime(["2026-07-20"]), "ts_code": ["000001.SZ"]})
    daily_basic = pd.DataFrame({"date": pd.to_datetime(["2026-07-20"]), "ts_code": ["000001.SZ"]})
    stock_basic = pd.DataFrame({"ts_code": ["000001.SZ"]})

    class FakeModule:
        @staticmethod
        def parse_date(value):
            return pd.to_datetime(value)

        @staticmethod
        def load_stock_basic():
            return stock_basic

        @staticmethod
        def load_daily_basic_monthly(start, end):
            return daily_basic, {"first_trade_date": "20130101"}

        @staticmethod
        def load_daily_monthly_features(start, end, basic, candidate_symbols, **kwargs):
            calls.append(kwargs)
            return features, pd.DataFrame()

    services._load_live_long_base_cached.cache_clear()
    monkeypatch.setattr(services, "_long_research_module", lambda: FakeModule())

    first = services._load_live_long_base("2026-07-20")
    second = services._load_live_long_base("2026-07-20")

    assert len(calls) == 1
    assert calls[0] == {"use_cache": False, "include_daily_returns": False}
    assert first[0] is second[0]
    services._load_live_long_base_cached.cache_clear()


def test_live_long_base_keeps_two_sections_and_bounded_history(monkeypatch) -> None:
    observed = {}
    dates = pd.to_datetime(["2026-05-29", "2026-06-30", "2026-07-21"])
    daily_basic = pd.DataFrame(
        {
            "date": dates,
            "trade_date": dates.strftime("%Y%m%d"),
            "ts_code": ["000001.SZ"] * 3,
        }
    )
    features = pd.DataFrame({"date": dates[-2:], "ts_code": ["000001.SZ"] * 2})
    stock_basic = pd.DataFrame({"ts_code": ["000001.SZ"]})

    class FakeModule:
        @staticmethod
        def load_stock_basic():
            return stock_basic

        @staticmethod
        def load_daily_basic_monthly(start, end):
            observed["basic_start"] = start
            observed["basic_end"] = end
            return daily_basic, {"first_trade_date": "20260529"}

        @staticmethod
        def load_daily_monthly_features(start, end, basic, candidate_symbols, **kwargs):
            observed["price_start"] = start
            return features, pd.DataFrame()

    services._load_live_long_base_cached.cache_clear()
    monkeypatch.setattr(services, "_long_research_module", lambda: FakeModule())

    loaded_features, loaded_basic, _, coverage = services._load_live_long_base("2026-07-21")

    assert observed["basic_start"] == pd.Timestamp("2023-03-21")
    assert observed["price_start"] == pd.Timestamp("2026-06-30")
    assert loaded_features["date"].tolist() == list(dates[-2:])
    assert loaded_basic["date"].tolist() == list(dates[-2:])
    assert coverage["live_rebalance_dates"] == ["2026-06-30", "2026-07-21"]
    services._load_live_long_base_cached.cache_clear()


def test_tea_checkpoint_recovers_previous_month_targets(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_read_long_stock_pool_snapshot",
        lambda *args, **kwargs: {
            "signal_date": "2026-07-20",
            "stocks": [
                {"ts_code": "000001.SZ", "rank": 1, "previous_state": "CORE"},
                {"ts_code": "000002.SZ", "rank": 2, "previous_state": "WATCH"},
                {"ts_code": "000003.SZ", "rank": None, "previous_state": "CORE"},
            ],
        },
    )

    same_month = services._tea_previous_target_checkpoint("tea", pd.Timestamp("2026-07-21"))
    next_month = services._tea_previous_target_checkpoint("tea", pd.Timestamp("2026-08-03"))

    assert same_month == {"000001.SZ", "000003.SZ"}
    assert next_month == {"000001.SZ", "000002.SZ"}


def test_tea_analyst_display_recomputes_growth_score_and_preserves_missing(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-21"] * 3),
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "close": [10.0, 20.0, 30.0],
            "analyst_forward_growth_score": [0.0, 0.0, 0.0],
        }
    )

    class FakeModule:
        @staticmethod
        def load_analyst_forecast_asof(out):
            return out.assign(
                analyst_report_count_180d=[9.0, 9.0, 5.0],
                analyst_org_count_180d=[2.0, 2.0, 1.0],
                analyst_forward_years_180d=[3.0, 3.0, 3.0],
                analyst_forward_eps_growth_180d=[0.20, 0.10, float("nan")],
                analyst_forward_revenue_growth_180d=[0.15, 0.05, float("nan")],
                analyst_forward_net_profit_growth_180d=[0.25, 0.08, float("nan")],
                analyst_target_upside_180d=[0.10, 0.05, float("nan")],
            )

        @staticmethod
        def percentile_score(series, higher_is_better=True):
            rank = pd.to_numeric(series, errors="coerce").rank(pct=True)
            if not higher_is_better:
                rank = 1.0 - rank
            return (rank * 100).fillna(0.0)

    monkeypatch.setattr(services, "_long_research_module", lambda: FakeModule())

    result = services._attach_analyst_forecast_for_display(frame).set_index("ts_code")

    assert result.loc["000001.SZ", "analyst_forward_growth_score"] > result.loc["000002.SZ", "analyst_forward_growth_score"]
    assert pd.isna(result.loc["000003.SZ", "analyst_forward_growth_score"])


def test_continuous_recommendation_days_follow_latest_unbroken_streak() -> None:
    targets = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                [
                    "2026-03-31",
                    "2026-04-30",
                    "2026-05-29",
                    "2026-05-29",
                    "2026-06-30",
                    "2026-06-30",
                    "2026-07-21",
                    "2026-07-21",
                ]
            ),
            "ts_code": [
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000001.SZ",
                "000003.SZ",
                "000001.SZ",
                "000003.SZ",
            ],
        }
    )

    starts = services._continuous_recommendation_starts(targets, pd.Timestamp("2026-07-21"))

    assert starts == {
        "000001.SZ": pd.Timestamp("2026-03-31"),
        "000003.SZ": pd.Timestamp("2026-06-30"),
    }

    recommended = services._decorate_long_recommendation_display(
        {
            "state": "BUILDING",
            "close": 9.8,
            "price_levels": {
                "entry_aggressive_price": 10.0,
                "entry_target_price": 10.2,
                "entry_small_position_price": 11.0,
            },
        },
        selected=True,
        signal_ts=pd.Timestamp("2026-07-21"),
        recommendation_since=starts["000001.SZ"],
    )
    assert recommended["recommendation_level"] == "RECOMMENDED"
    assert recommended["recommendation_days"] == 113
    assert recommended["price_state"] == "AGGRESSIVE"
    assert recommended["display_reason"] == "本期进入推荐池，价格与风险条件支持分批执行"
    assert "持有" not in recommended["display_reason"]

    cautious = services._decorate_long_recommendation_display(
        {
            "state": "REDUCE",
            "reason": "价格跌破 MA60，先降仓控制风险",
            "close": 9.8,
            "price_levels": {"reduce_ma60_price": 10.0},
        },
        selected=False,
        signal_ts=pd.Timestamp("2026-07-21"),
    )
    assert cautious["recommendation_level"] == "CAUTION"
    assert "降仓" not in cautious["display_reason"]


def test_legacy_tea_snapshot_is_upgraded_without_full_pool_rebuild(monkeypatch) -> None:
    payload = {
        "variant": "tea",
        "signal_date": "2026-07-21",
        "stocks": [
            {"ts_code": "000001.SZ", "close": 10.0, "analyst_forward_growth_score": 0.0},
            {"ts_code": "000002.SZ", "close": 20.0, "analyst_forward_growth_score": 0.0},
        ],
    }
    enriched = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "analyst_report_count_180d": [9.0, 5.0],
            "analyst_org_count_180d": [2.0, 1.0],
            "analyst_forward_years_180d": [3.0, 2.0],
            "analyst_forward_growth_score": [72.5, float("nan")],
            "analyst_target_upside_180d": [0.12, float("nan")],
        }
    )
    monkeypatch.setattr(services, "_attach_analyst_forecast_for_display", lambda frame: enriched)

    upgraded, changed = services._upgrade_cached_tea_analyst_display(payload)

    assert changed is True
    assert upgraded["schema_version"] == services.LONG_STOCK_POOL_SCHEMA_VERSION
    assert upgraded["stocks"][0]["analyst_forward_growth_score"] == 72.5
    assert upgraded["stocks"][1]["analyst_forward_growth_score"] is None


def test_tail_resume_runs_pending_steps_via_executor(monkeypatch) -> None:
    submitted = []
    maintenance_order: list[str] = []

    class FakeFuture:
        def __init__(self, fn):
            self.fn = fn

        def result(self):
            return self.fn()

    class FakeExecutor:
        def __init__(self, max_workers):
            submitted.append(("max_workers", max_workers))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn):
            submitted.append(("submit", fn))
            return FakeFuture(fn)

    monkeypatch.setattr(services, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(services, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(services, "_set_refresh_progress", lambda **kwargs: None)
    monkeypatch.setattr("quant.routine.pipeline._incremental_daily_start", lambda: "20260720")
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260720")
    monkeypatch.setattr(
        services,
        "get_stock_selector_payload",
        lambda **kwargs: {"signal_date": "2026-07-20", "stocks": []},
    )
    monkeypatch.setattr(services, "_build_tea_master_stock_pool_cached", type("Cache", (), {"cache_clear": staticmethod(lambda: None)})())
    monkeypatch.setattr(services, "_build_long_stock_pool_cached", type("Cache", (), {"cache_clear": staticmethod(lambda: None)})())
    monkeypatch.setattr(services, "get_chan_model_strategy_plan", lambda *args, **kwargs: {"signal_date": "2026-07-20", "candidates": []})
    monkeypatch.setattr(services, "_refresh_long_stock_pool_variants", lambda *args, **kwargs: [])
    monkeypatch.setattr(services, "get_convertible_bond_grid_plan", lambda *args, **kwargs: {"trade_date": "20260720", "candidates": []})
    monkeypatch.setattr(services, "get_convertible_bond_allotments", lambda *args, **kwargs: {"asof": "2026-07-20", "records": []})
    monkeypatch.setattr(services, "get_byd_daily_strategy", lambda *args, **kwargs: {"planned_t": {"signal_date": "2026-07-20"}, "alerts": []})
    monkeypatch.setattr(services, "_run_similar_pattern_analysis_isolated", lambda: {"generated_at": "2026-07-20T20:00:00", "results": []})
    monkeypatch.setattr(
        services,
        "_write_strategy_pool_snapshots",
        lambda *args, **kwargs: maintenance_order.append("snapshot") or {"ALL": 0},
    )
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda project_root: maintenance_order.append("cleanup")
        or {"status": "success", "sql_snapshots": {"status": "success"}, "errors": []},
    )

    result = services._resume_tail_refresh_from_cached_selector(
        "all",
        {
            "steps": [
                {"key": "selector_extended", "status": "success"},
                {"key": "snapshot", "status": "failed"},
            ],
            "result": {
                "refresh_data": {
                    "status": "success",
                    "expected_trade_date": "20260720",
                    "dataset_trade_date": "20260720",
                }
            },
        },
    )

    assert ("max_workers", 6) in submitted
    assert len([item for item in submitted if item[0] == "submit"]) == 6
    assert result["snapshot"]["strategy_pools"] == {"ALL": 0}
    assert result["similar_patterns"]["status"] == "success"
    assert result["cache_cleanup_after_snapshot"]["status"] == "success"
    assert maintenance_order[-2:] == ["snapshot", "cleanup"]


def test_post_snapshot_cleanup_failure_is_not_silently_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "run_cache_cleanup",
        lambda project_root: {
            "status": "partial",
            "sql_snapshots": {"status": "failed"},
            "errors": ["database unavailable"],
        },
    )
    results: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="database unavailable"):
        services._run_post_snapshot_cache_cleanup(results)

    assert results["cache_cleanup_after_snapshot"]["status"] == "partial"


def test_same_day_failed_run_can_reuse_completed_source_inputs(monkeypatch) -> None:
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260724")
    today = pd.Timestamp.now().isoformat()
    status = {
        "status": "failed",
        "started_at": today,
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260724",
                "dataset_trade_date": "20260724",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260724",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "steps": {"analyst_forecast_snapshot": {"status": "success"}},
            },
            "feature_cache": {"status": "failed"},
        },
    }

    assert services._input_resume_ready(status, "all") is True
    status["result"]["refresh_daily_basic"]["status"] = "failed"
    assert services._input_resume_ready(status, "all") is False


def test_same_day_failed_run_reuses_degraded_analyst_fallback(monkeypatch) -> None:
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260724")
    status = {
        "status": "failed",
        "started_at": pd.Timestamp.now().isoformat(),
        "result": {
            "refresh_data": {
                "status": "success",
                "expected_trade_date": "20260724",
                "dataset_trade_date": "20260724",
            },
            "refresh_daily_basic": {
                "status": "success",
                "latest_trade_date": "20260724",
            },
            "refresh_reference_inputs": {
                "status": "success",
                "critical_errors": [],
                "steps": {"analyst_forecast_snapshot": {"status": "degraded"}},
            },
        },
    }

    assert services._input_resume_ready(status, "all") is True


def test_similar_pattern_refresh_updates_vector_caches(monkeypatch, tmp_path) -> None:
    calls = {"cache": 0}

    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: [])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-20")

    def fake_build(*args, **kwargs):
        calls["cache"] += 1
        return pd.DataFrame({"status": ["built", "cache_hit"]})

    monkeypatch.setattr(services, "build_vector_caches_parallel", fake_build)
    monkeypatch.setattr(services, "analyze_targets_by_threshold", lambda *args, **kwargs: {})

    payload = services.refresh_similar_pattern_analysis(force_vector_cache=True)

    assert calls["cache"] == 1
    assert payload["cache"]["rebuilt"] == 1
    assert payload["cache"]["reused"] == 1
    assert payload["cache"]["reference_library_policy"] == "weekly"
    assert payload["cache"]["reference_library_refreshed"] is True
    assert payload["cache"]["reference_library_reason"] == "forced"
    assert payload["cache"]["reference_library_minimum_refresh_age_days"] == 5
    assert payload["cache"]["target_vectors"] == "live_from_latest_daily_data"


def test_similar_pattern_refresh_reuses_current_weekly_library_but_analyzes_live_targets(
    monkeypatch,
    tmp_path,
) -> None:
    calls = {"cache": 0, "analysis": 0}
    cache_root = tmp_path / "vector_cache"
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", cache_root)
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: ["002594.SZ"])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-21")

    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps(
            {
                "config_key": services.vector_cache_key(services.SIMILAR_PATTERN_CONFIG),
                "refreshed_at": datetime.now().isoformat(timespec="seconds"),
                "source_trade_date": "2026-07-20",
                "cached_files": 1,
            }
        ),
        encoding="utf-8",
    )

    def fake_build(*args, **kwargs):
        calls["cache"] += 1
        return pd.DataFrame()

    def fake_analyze(*args, **kwargs):
        calls["analysis"] += 1
        assert kwargs["target_symbols"] == ["002594.SZ"]
        return {"002594.SZ": object()}

    monkeypatch.setattr(services, "build_vector_caches_parallel", fake_build)
    monkeypatch.setattr(services, "analyze_targets_by_threshold", fake_analyze)
    monkeypatch.setattr(
        services,
        "_similar_pattern_result_payload",
        lambda *args, **kwargs: {"target": {"symbol": "002594.SZ"}},
    )
    monkeypatch.setattr(services, "_attach_watchlist_strategy_hits", lambda payload: payload)

    payload = services.refresh_similar_pattern_analysis()

    assert calls == {"cache": 0, "analysis": 1}
    assert payload["cache"]["reference_library_refreshed"] is False
    assert payload["cache"]["reference_library_reason"] in {
        "waiting_for_friday_close",
        "waiting_for_friday_trade_close",
        "friday_close_window_already_refreshed",
        "minimum_refresh_age_not_reached",
    }
    assert payload["cache"]["reused"] == 1


def test_similar_pattern_refresh_rejects_missing_watchlist_targets(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: ["002594.SZ"])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-22")
    monkeypatch.setattr(
        services,
        "_similar_pattern_vector_cache_refresh_decision",
        lambda **kwargs: {
            "due": False,
            "reason": "weekly_cache_current",
            "metadata": {},
            "inferred_legacy": False,
            "refreshed_at": None,
            "cached_files": 0,
        },
    )
    monkeypatch.setattr(services, "analyze_targets_by_threshold", lambda *args, **kwargs: {})

    with pytest.raises(RuntimeError, match="missing=1/1"):
        services.refresh_similar_pattern_analysis()


def test_similar_pattern_refresh_coalesces_concurrent_requests(monkeypatch) -> None:
    read_count = 0
    compute_count = 0
    read_lock = threading.Lock()
    second_request_started = threading.Event()
    computation_started = threading.Event()
    release_computation = threading.Event()
    results: list[dict] = []
    errors: list[BaseException] = []

    monkeypatch.setattr(services, "_SIMILAR_PATTERN_REFRESH_INFLIGHT", None)

    def fake_symbols() -> list[str]:
        nonlocal read_count
        with read_lock:
            read_count += 1
            if read_count >= 2:
                second_request_started.set()
        return ["002594.SZ"]

    def fake_refresh_once(symbols, progress_callback=None, *, force_vector_cache=False):
        nonlocal compute_count
        compute_count += 1
        computation_started.set()
        assert release_computation.wait(timeout=3)
        return {
            "generated_at": "2026-07-28T13:21:53",
            "results": [{"target": {"symbol": symbols[0]}}],
            "cache": {"backend": "generated"},
        }

    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", fake_symbols)
    monkeypatch.setattr(services, "_refresh_similar_pattern_analysis_once", fake_refresh_once)

    def invoke_refresh() -> None:
        try:
            results.append(services.refresh_similar_pattern_analysis())
        except BaseException as exc:  # pragma: no cover - diagnostic collection
            errors.append(exc)

    first = threading.Thread(target=invoke_refresh)
    second = threading.Thread(target=invoke_refresh)
    first.start()
    assert computation_started.wait(timeout=3)
    second.start()
    assert second_request_started.wait(timeout=3)
    release_computation.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert errors == []
    assert compute_count == 1
    assert len(results) == 2
    assert sum(bool(item["cache"].get("coalesced")) for item in results) == 1


def test_similar_pattern_weekly_library_waits_for_friday_close_after_seven_days(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-14T08:00:00"}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 21, 8, 0, 0)
    )

    assert decision["due"] is False
    assert decision["reason"] == "waiting_for_friday_close"
    assert decision["next_refresh_at"] == "2026-07-24T15:00:00"


def test_similar_pattern_weekly_library_becomes_due_on_friday_after_close(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-19T15:01:00", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is True
    assert decision["reason"] == "friday_close_window"
    assert decision["refresh_age_days"] == 5
    assert decision["minimum_refresh_age_days"] == 5


def test_similar_pattern_weekly_library_waits_until_minimum_age_on_friday(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-20T20:25:57", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is False
    assert decision["reason"] == "minimum_refresh_age_not_reached"
    assert decision["refresh_age_days"] < 5


def test_similar_pattern_weekly_library_waits_when_friday_is_not_trade_date(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-18T15:01:00", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-23",
    )

    assert decision["due"] is False
    assert decision["reason"] == "waiting_for_friday_trade_close"


def test_similar_pattern_weekly_library_does_not_rebuild_twice_in_friday_window(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps({"refreshed_at": "2026-07-24T15:05:00", "cached_files": 1}),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 16, 0, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is False
    assert decision["reason"] == "friday_close_window_already_refreshed"
    assert decision["next_refresh_at"] == "2026-07-31T15:00:00"


def test_similar_pattern_weekly_library_rebuilds_when_cache_count_changes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps(
            {
                "refreshed_at": "2026-07-17T15:01:00",
                "cached_files": 2,
                "errors": 0,
            }
        ),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is True
    assert decision["reason"] == "cache_file_count_changed"


def test_similar_pattern_cache_repair_waits_for_minimum_refresh_age(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    state_dir = services._similar_pattern_vector_cache_state_dir()
    state_dir.mkdir(parents=True)
    (state_dir / "000001_SZ.npz").write_bytes(b"cache")
    (state_dir / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA).write_text(
        json.dumps(
            {
                "refreshed_at": "2026-07-23T15:01:00",
                "cached_files": 2,
                "errors": 0,
            }
        ),
        encoding="utf-8",
    )

    decision = services._similar_pattern_vector_cache_refresh_decision(
        now=datetime(2026, 7, 24, 15, 1, 0),
        source_trade_date="2026-07-24",
    )

    assert decision["due"] is False
    assert decision["reason"] == "minimum_refresh_age_not_reached"


def test_similar_pattern_weekly_library_does_not_advance_watermark_on_build_errors(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_VECTOR_CACHE_DIR", tmp_path / "vector_cache")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: [])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())
    monkeypatch.setattr(services, "_latest_similar_pattern_target_date", lambda symbols: "2026-07-23")
    monkeypatch.setattr(
        services,
        "build_vector_caches_parallel",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"symbol": "000001.SZ", "status": "built"},
                {"symbol": "000002.SZ", "status": "error", "error": "bad source"},
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="errors=1.*000002.SZ: bad source"):
        services.refresh_similar_pattern_analysis(force_vector_cache=True)

    metadata_path = (
        services._similar_pattern_vector_cache_state_dir()
        / services.SIMILAR_PATTERN_VECTOR_CACHE_METADATA
    )
    assert not metadata_path.exists()


def test_similar_pattern_payload_includes_optimized_decision_and_validation() -> None:
    cases = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "name": ["甲", "乙", "丙"],
            "industry": ["汽车整车", "汽车整车", "软件服务"],
            "date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-04"]),
            "similarity": [0.080, 0.075, 0.065],
            "fwd_1d": [0.02, 0.01, -0.01],
            "fwd_20d": [0.05, 0.02, -0.03],
            "fwd_60d": [0.10, 0.04, -0.06],
            "max_drawdown_60d": [-0.03, -0.05, -0.12],
        }
    )
    result = SimilarPatternResult(
        target=TargetSpec("002594.SZ", "比亚迪", pd.Timestamp("2025-01-02")),
        latest_snapshot={"close": 100.0, "dist_ma20": 1.0, "dist_ma60": 2.0, "drawdown_60d": -3.0, "vol_ratio20": 1.0},
        similar_cases=cases,
        forecast=summarize_forecast(cases),
        status_probs={"上升": 50.0, "震荡": 20.0, "下跌": 30.0, "高波动": 0.0},
        scan_summary={"matched_cases": 3},
    )
    market = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-04", "2025-01-02"]), "market_regime": ["risk_on"] * 4})
    industry = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-02-02", "2025-01-02"]), "industry_regime": ["risk_on"] * 3})
    validation = {
        "calibrations": {"next_1d": {"x": [0, 100], "y": [0, 100]}},
        "model_selection": {
            "next_1d": {
                "selected": {
                    "source": "regime_industry",
                    "bearish_max": 49.0,
                    "bullish_min": 51.0,
                    "enable_risk_gate": False,
                }
            }
        },
        "summary": [{"symbol": "002594.SZ", "horizon": "next_1d", "coverage": 40.0}],
    }

    payload = services._similar_pattern_result_payload(
        result,
        profile={"industry": "汽车整车"},
        market_regime=market,
        industry_regime=industry,
        validation=validation,
    )

    assert payload["optimized_forecast"]
    assert payload["decisions"][0]["signal"] in {"bullish", "bearish", "observe"}
    assert payload["optimization_summary"]["effective_sample_size"] > 0
    assert payload["validation_summary"][0]["coverage"] == 40.0
    assert payload["optimized_forecast"][0]["probability_source"] == "regime_industry"
    assert payload["decisions"][0]["bearish_max"] == 49.0
    top_cases = payload["top_cases"]
    ceiling = services._similarity_score_ceiling()
    assert top_cases[0]["similarity_score"] == round(
        np.log1p(services.SIMILARITY_SCORE_CONTRAST * top_cases[0]["forecast_weight"] / ceiling)
        / np.log1p(services.SIMILARITY_SCORE_CONTRAST)
        * 100,
        1,
    )
    assert top_cases[0]["similarity_score"] < 100.0
    assert [row["similarity_score"] for row in top_cases] == sorted(
        [row["similarity_score"] for row in top_cases], reverse=True
    )


def test_watchlist_strategy_hits_merge_only_strategy_workspaces(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"策略命中合并不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(services, "_latest_candidate_signal_date", lambda: "2026-07-15")
    monkeypatch.setattr(
        services,
        "_read_selector_snapshot",
        lambda *args, **kwargs: {
            "signal_date": "2026-07-15",
            "stocks": [{"symbol": "002594.SZ", "matched_families": ["B1", "强K"], "matched_count": 2}],
        },
    )
    monkeypatch.setattr(
        services,
        "get_chan_model_strategy_plan",
        lambda **kwargs: {
            "signal_date": "2026-07-15",
            "candidates": [{"symbol": "002594.SZ", "rule_name": "主策略"}],
        },
    )
    monkeypatch.setattr(
        services,
        "get_long_stock_pool",
        lambda **kwargs: {
            "signal_date": "2026-07-15",
            "stocks": [{"ts_code": "002788.SZ", "state": "WATCH", "action": "观察"}],
        },
    )
    monkeypatch.setattr(
        services,
        "_read_daily_payload_cache",
        lambda path: {
            "asof": "2026-07-15",
            "records": [{"stock_code": "002788", "status": "证监会注册"}],
        },
    )

    hits = services._collect_watchlist_strategy_hits(["002594.SZ", "002788.SZ"])

    assert [item["strategy_key"] for item in hits["002594.SZ"]] == ["short", "chan"]
    assert [item["strategy_key"] for item in hits["002788.SZ"]] == ["long"]
    assert hits["002594.SZ"][0]["detail"] == "B1 / 强K"


def test_attach_watchlist_strategy_hits_adds_badges_to_each_result(monkeypatch) -> None:
    payload = {
        "results": [
            {"target": {"symbol": "002594.SZ"}},
            {"target": {"symbol": "002788.SZ"}},
        ]
    }
    monkeypatch.setattr(
        services,
        "_collect_watchlist_strategy_hits",
        lambda symbols: {"002594.SZ": [{"strategy_key": "short"}], "002788.SZ": []},
    )

    enriched = services._attach_watchlist_strategy_hits(payload)

    assert enriched["results"][0]["strategy_hit_count"] == 1
    assert enriched["results"][1]["strategy_hit_count"] == 0


def test_watchlist_notes_are_saved_without_recalculating_scores(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["002594.SZ"],"notes":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"笔记保存不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail(f"笔记保存不应重新计算评分: {symbols}"),
    )

    payload = services.save_similar_pattern_watch_note(
        "002594.SZ",
        "操作计划：回踩 20 日线后分批关注",
    )

    stock = payload["stocks"][0]
    persisted = watchlist_path.read_text(encoding="utf-8")
    assert stock["note"] == "操作计划：回踩 20 日线后分批关注"
    assert stock["note_updated_at"]
    assert "opportunity_score" not in stock
    assert "holding_score" not in stock
    assert "回踩 20 日线" in persisted


def test_strategy_add_duplicate_preserves_existing_note(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    original = (
        '{"symbols":["002594.SZ"],"notes":{"002594.SZ":'
        '{"content":"手工计划：回踩观察","updated_at":"2026-07-16T20:00:00"}}}'
    )
    watchlist_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )

    payload = services.add_similar_pattern_watch_symbol("002594.SZ", note="7.17 触发 B1 策略")

    note = payload["stocks"][0]["note"]
    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert note == "手工计划：回踩观察"
    assert watchlist_path.read_text(encoding="utf-8") == original
    assert persisted["notes"]["002594.SZ"] == {
        "content": "手工计划：回踩观察",
        "updated_at": "2026-07-16T20:00:00",
    }


def test_watchlist_add_returns_without_calculating_scores(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["002594.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [
                {"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"},
                {"ts_code": "000001.SZ", "name": "平安银行", "industry": "银行"},
            ]
        ),
    )

    def fail_if_scores_run(symbols):
        raise AssertionError(f"add should not calculate scores synchronously: {symbols}")

    monkeypatch.setattr(services, "_watchlist_buy_hold_scores", fail_if_scores_run)

    payload = services.add_similar_pattern_watch_symbol("000001.SZ")

    assert [item["symbol"] for item in payload["stocks"]] == ["002594.SZ", "000001.SZ"]
    assert payload["stocks"][1]["name"] == "平安银行"
    assert "opportunity_score" not in payload["stocks"][1]


def test_watchlist_remove_returns_without_market_or_score_recalculation(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        json.dumps(
            {
                "symbols": ["002594.SZ", "000001.SZ"],
                "pinned": ["002594.SZ"],
                "notes": {"002594.SZ": {"content": "观察", "updated_at": "2026-07-30T10:00:00"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [
                {"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"},
                {"ts_code": "000001.SZ", "name": "平安银行", "industry": "银行"},
            ]
        ),
    )
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail("删除现有自选股不应重新校验行情"),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail("删除响应不应重新计算评分"),
    )

    payload = services.remove_similar_pattern_watch_symbol("002594.SZ")

    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert [item["symbol"] for item in payload["stocks"]] == ["000001.SZ"]
    assert persisted["symbols"] == ["000001.SZ"]
    assert persisted["pinned"] == []
    assert persisted["notes"] == {}


def test_watchlist_api_forwards_default_source_note(monkeypatch) -> None:
    captured = {}

    def fake_add(symbol, note=""):
        captured.update({"symbol": symbol, "note": note})
        return {"stocks": []}

    monkeypatch.setattr(webapp_api, "add_similar_pattern_watch_symbol", fake_add)

    response = client.post(
        "/api/similar-patterns/watchlist",
        json={"symbol": "002594.SZ", "note": "7.15 配债股"},
    )

    assert response.status_code == 200
    assert captured == {"symbol": "002594.SZ", "note": "7.15 配债股"}


def test_watchlist_api_can_return_lightweight_profiles(monkeypatch) -> None:
    captured = {}

    def fake_get(*, include_scores=True):
        captured["include_scores"] = include_scores
        return {"stocks": []}

    monkeypatch.setattr(webapp_api, "get_similar_pattern_watchlist", fake_get)

    response = client.get("/api/similar-patterns/watchlist?include_scores=false")

    assert response.status_code == 200
    assert captured == {"include_scores": False}


def test_watchlist_note_rejects_stock_outside_watchlist(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["002594.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    with pytest.raises(ValueError, match="不在自选池"):
        services.save_similar_pattern_watch_note("000001.SZ", "观察")


def test_watchlist_independent_reminders_with_notes_are_persisted(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["002594.SZ"],"notes":{},"pinned":[]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"提醒保存不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail(f"提醒保存不应重新计算评分: {symbols}"),
    )

    payload = services.save_similar_pattern_watch_alerts(
        "002594.SZ",
        {
            "enabled": True,
            "reminders": [
                {
                    "id": "reminder-1",
                    "note": "突破前高并放量",
                    "conditions": [
                        {
                            "id": "price-high",
                            "conjunction": "and",
                            "kind": "price",
                            "operator": "gt",
                            "value": 118.5,
                        },
                        {
                            "id": "volume",
                            "conjunction": "and",
                            "kind": "indicator",
                            "indicator": "vol_ratio20",
                            "operator": "gt",
                            "value": 1.8,
                        },
                    ],
                },
                {
                    "id": "reminder-2",
                    "note": "回撤或评分进入关注区",
                    "conditions": [
                        {
                            "id": "drawdown",
                            "conjunction": "and",
                            "kind": "indicator",
                            "indicator": "kdj_daily_j",
                            "operator": "lt",
                            "value": -10,
                        },
                        {
                            "id": "buy-score",
                            "conjunction": "or",
                            "kind": "indicator",
                            "indicator": "opportunity_score",
                            "operator": "gt",
                            "value": 75,
                        },
                    ],
                },
            ],
        },
    )

    alerts = payload["stocks"][0]["alerts"]
    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert alerts["enabled"] is True
    assert len(alerts["reminders"]) == 2
    assert alerts["reminders"][0]["note"] == "突破前高并放量"
    assert alerts["reminders"][1]["conditions"][1]["conjunction"] == "or"
    assert alerts["updated_at"]
    assert payload["stocks"][0]["alert_count"] == 2
    assert payload["stocks"][0]["alert_condition_count"] == 4
    assert persisted["alerts"]["002594.SZ"]["reminders"][0]["conditions"][1]["indicator"] == "vol_ratio20"
    assert persisted["alerts"]["002594.SZ"]["reminders"][1]["conditions"][0]["indicator"] == "kdj_daily_j"


def test_watchlist_alert_api_accepts_reminder_notes_and_condition_logic(monkeypatch) -> None:
    captured = {}

    def fake_save(symbol, config):
        captured.update({"symbol": symbol, "config": config})
        return {"stocks": []}

    monkeypatch.setattr(webapp_api, "save_similar_pattern_watch_alerts", fake_save)
    response = client.put(
        "/api/similar-patterns/watchlist/002594.SZ/alerts",
        json={
            "enabled": True,
            "reminders": [
                {
                    "id": "reminder-1",
                    "note": "价格或涨幅满足",
                    "conditions": [
                        {
                            "id": "p1",
                            "conjunction": "and",
                            "kind": "price",
                            "operator": "gt",
                            "value": 120,
                        },
                        {
                            "id": "i1",
                            "conjunction": "or",
                            "kind": "indicator",
                            "indicator": "kdj_daily_j",
                            "operator": "lt",
                            "value": -10,
                        },
                    ],
                }
            ],
        },
    )
    invalid = client.put(
        "/api/similar-patterns/watchlist/002594.SZ/alerts",
        json={
            "reminders": [
                {
                    "id": "bad",
                    "conditions": [
                        {
                            "id": "bad-condition",
                            "kind": "price",
                            "operator": "gte",
                            "value": 100,
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured["symbol"] == "002594.SZ"
    assert captured["config"]["reminders"][0]["note"] == "价格或涨幅满足"
    assert captured["config"]["reminders"][0]["conditions"][1]["conjunction"] == "or"
    assert captured["config"]["reminders"][0]["conditions"][1]["indicator"] == "kdj_daily_j"
    assert invalid.status_code == 422


def test_legacy_watchlist_alert_shape_is_discarded(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        json.dumps(
            {
                "symbols": ["002594.SZ"],
                "alerts": {
                    "002594.SZ": {
                        "enabled": True,
                        "logic": "and",
                        "price_rules": [{"id": "old", "operator": "gte", "value": 100}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    state = services._read_similar_pattern_watchlist_state()

    assert state["alerts"] == {}


def test_watchlist_order_and_pins_are_persisted(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["000001.SZ","000002.SZ","000003.SZ"],"notes":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(
        services,
        "_normalize_watch_symbol",
        lambda symbol: pytest.fail(f"置顶或排序不应读取行情文件: {symbol}"),
    )
    monkeypatch.setattr(
        services,
        "_watchlist_buy_hold_scores",
        lambda symbols: pytest.fail(f"置顶或排序不应重新计算评分: {symbols}"),
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "name": "甲", "industry": "银行"},
                {"ts_code": "000002.SZ", "name": "乙", "industry": "地产"},
                {"ts_code": "000003.SZ", "name": "丙", "industry": "制造"},
            ]
        ),
    )

    pinned_payload = services.set_similar_pattern_watch_pin("000002.SZ", True)
    reordered_payload = services.reorder_similar_pattern_watchlist(
        ["000002.SZ", "000003.SZ", "000001.SZ"]
    )

    assert [item["symbol"] for item in pinned_payload["stocks"]] == [
        "000002.SZ",
        "000001.SZ",
        "000003.SZ",
    ]
    assert pinned_payload["stocks"][0]["pinned"] is True
    assert [item["symbol"] for item in reordered_payload["stocks"]] == [
        "000002.SZ",
        "000003.SZ",
        "000001.SZ",
    ]
    persisted = json.loads(watchlist_path.read_text(encoding="utf-8"))
    assert persisted["symbols"] == ["000002.SZ", "000003.SZ", "000001.SZ"]
    assert persisted["pinned"] == ["000002.SZ"]


def test_watchlist_reorder_rejects_incomplete_symbol_set(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["000001.SZ","000002.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    with pytest.raises(ValueError, match="当前全部股票"):
        services.reorder_similar_pattern_watchlist(["000001.SZ"])


def test_daily_payload_freshness_uses_generated_date() -> None:
    today = date(2026, 7, 13)

    assert services._is_daily_payload_current({"generated_at": "2026-07-13T08:30:00"}, today=today)
    assert not services._is_daily_payload_current({"generated_at": "2026-06-28T22:52:04"}, today=today)
    assert not services._is_daily_payload_current({}, today=today)


def test_workspace_snapshot_uses_filesystem_and_nearest_prior_date(monkeypatch, tmp_path) -> None:
    class NoSqlStore:
        def __init__(self, config):
            self.config = config

    no_sql_config = type("NoSqlConfig", (), {"sql_url": None})()
    monkeypatch.setattr(services, "WEB_WORKSPACE_SNAPSHOT_DIR", tmp_path / "snapshots")
    monkeypatch.setattr(services, "MarketDataStore", NoSqlStore)
    monkeypatch.setattr(services.MarketDataStoreConfig, "from_env", lambda **kwargs: no_sql_config)

    services._write_workspace_snapshot(
        "convertible_bond_grid_plan",
        "20260717",
        {"trade_date": "20260717", "candidates": [{"ts_code": "123001.SZ"}]},
        params={"limit": 18},
    )

    payload = services._read_workspace_snapshot(
        "convertible_bond_grid_plan",
        snapshot_date="2026-07-18",
        params={"limit": 18},
    )

    assert payload is not None
    assert payload["candidates"][0]["ts_code"] == "123001.SZ"
    assert payload["cache"]["backend"] == "filesystem"
    assert payload["cache"]["snapshot_date"] == "2026-07-17"
    assert payload["cache"]["requested_date"] == "2026-07-18"
    assert payload["cache"]["stale"] is True


def test_convertible_bond_plan_bootstraps_from_legacy_snapshot(monkeypatch, tmp_path) -> None:
    legacy_path = tmp_path / "convertible_bond_grid_plan.json"
    legacy_path.write_text(
        '{"trade_date":"20260717","generated_at":"2026-07-17T18:00:00","candidates":[]}',
        encoding="utf-8",
    )
    writes = []
    monkeypatch.setattr(services, "CONVERTIBLE_BOND_GRID_PLAN_PATH", legacy_path)
    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        services,
        "_write_filesystem_workspace_snapshot",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        services,
        "build_convertible_bond_grid_plan",
        lambda **kwargs: pytest.fail("tab read must not synchronously rebuild when a durable snapshot exists"),
    )

    payload = services.get_convertible_bond_grid_plan(trade_date="2026-07-18", limit=18)

    assert payload["cache"]["backend"] == "legacy_filesystem"
    assert payload["cache"]["snapshot_date"] == "2026-07-17"
    assert payload["cache"]["stale"] is True
    assert len(writes) == 1


def test_allotment_tab_serves_stale_daily_cache_without_blocking_refresh(monkeypatch, tmp_path) -> None:
    calls = []
    stale = {
        "generated_at": "2026-06-28T22:52:04",
        "records": [],
        "data_sources": {
            "stock_daily": {"requested": 0, "matched": 0, "error": None},
            "daily_basic": {"matched": 0, "error": None},
        },
    }

    monkeypatch.setattr(services, "CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH", tmp_path / "allotments.json")
    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *args, **kwargs: stale)
    monkeypatch.setattr(services, "_write_workspace_snapshot", lambda *args, **kwargs: None)

    def fake_build(**kwargs):
        calls.append(kwargs["refresh"])
        return {"generated_at": "2026-07-13T08:30:00", "records": [{"stock_code": "000001"}]}

    monkeypatch.setattr(services, "build_convertible_bond_allotment_payload", fake_build)

    payload = services.get_convertible_bond_allotments()

    assert calls == []
    assert payload["records"] == []
    assert payload["cache"]["stale"] is True
    assert payload["quality"]["status"] == "success"


def test_similar_pattern_tab_serves_stale_cache_without_blocking_refresh(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_analysis_cache",
        lambda: {"generated_at": "2026-06-28T22:52:04", "results": []},
    )

    def fake_refresh():
        calls.append(True)
        return {"generated_at": "2026-07-13T08:30:00", "results": []}

    monkeypatch.setattr(services, "refresh_similar_pattern_analysis", fake_refresh)

    payload = services.get_similar_pattern_analysis()

    assert calls == []
    assert payload["generated_at"].startswith("2026-06-28")
    assert payload["cache"]["stale"] is True


def test_similar_pattern_tab_without_cache_returns_immediately(monkeypatch) -> None:
    monkeypatch.setattr(services, "_read_similar_pattern_analysis_cache", lambda: None)
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_watchlist_symbols",
        lambda: ["002594.SZ"],
    )
    monkeypatch.setattr(
        services,
        "get_similar_pattern_watchlist",
        lambda *, include_scores=True: {
            "updated_at": "2026-07-30T20:00:00",
            "stocks": [{"symbol": "002594.SZ", "name": "比亚迪"}],
        },
    )
    monkeypatch.setattr(
        services,
        "refresh_similar_pattern_analysis",
        lambda: pytest.fail("打开自选池不应同步生成完整分析"),
    )

    payload = services.get_similar_pattern_analysis()

    assert payload["results"] == []
    assert payload["watchlist"][0]["symbol"] == "002594.SZ"
    assert payload["cache"]["missing"] is True


def test_similar_pattern_cache_filters_results_removed_from_watchlist(monkeypatch) -> None:
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_analysis_cache",
        lambda: {
            "generated_at": "2026-07-28T13:21:53",
            "watchlist": [
                {
                    "symbol": "002594.SZ",
                    "opportunity_score": 73.2,
                    "holding_score": 61.8,
                    "score_date": "2026-07-28",
                }
            ],
            "results": [
                {"target": {"symbol": "002594.SZ"}},
                {"target": {"symbol": "000792.SZ"}},
            ],
        },
    )
    monkeypatch.setattr(
        services,
        "_read_similar_pattern_watchlist_symbols",
        lambda: ["002594.SZ"],
    )
    monkeypatch.setattr(
        services,
        "_similar_pattern_watchlist_profiles",
        lambda basic=None, include_scores=True: [{"symbol": "002594.SZ"}],
    )
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(),
    )
    monkeypatch.setattr(
        services,
        "_attach_watchlist_strategy_hits",
        lambda payload: payload,
    )

    payload = services.get_similar_pattern_analysis()

    assert [item["target"]["symbol"] for item in payload["results"]] == ["002594.SZ"]
    assert payload["watchlist"][0]["opportunity_score"] == 73.2
    assert payload["watchlist"][0]["holding_score"] == 61.8
    assert payload["cache"]["watchlist_changed"] is True


def test_vector_cache_builder_uses_thread_pool_in_daemon_process(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        similar_patterns_module.mp,
        "current_process",
        lambda: type("Proc", (), {"daemon": True})(),
    )

    daily_dir = tmp_path / "daily"
    cache_dir = tmp_path / "cache"
    daily_dir.mkdir()
    cache_dir.mkdir()
    for symbol in ["000001.SZ", "000002.SZ"]:
        (daily_dir / f"{symbol}.parquet").write_text("", encoding="utf-8")

    worker_calls = []

    def fake_worker(task):
        worker_calls.append(task[0])
        return {"status": "cache_hit", "symbol": task[0], "vectors": 1}

    class InlineFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class InlineThreadPool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task):
            return InlineFuture(fn(task))

    monkeypatch.setattr(similar_patterns_module, "_build_stock_vector_cache_worker", fake_worker)
    monkeypatch.setattr(similar_patterns_module, "ThreadPoolExecutor", InlineThreadPool)
    monkeypatch.setattr(
        similar_patterns_module,
        "ProcessPoolExecutor",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("process pool should not be used")),
    )
    monkeypatch.setattr(similar_patterns_module, "as_completed", lambda futures: futures)

    result = similar_patterns_module.build_vector_caches_parallel(
        daily_dir=daily_dir,
        basic=pd.DataFrame(),
        config=services.SIMILAR_PATTERN_CONFIG,
        cache_dir=cache_dir,
        workers=2,
    )

    assert len(worker_calls) == 2
    assert set(result["status"]) == {"cache_hit"}


def test_isolated_similar_pattern_process_is_not_daemon(monkeypatch) -> None:
    captured = {}

    class FakeQueue:
        def __init__(self):
            self._messages = [
                {"type": "result", "ok": True, "payload": {"generated_at": "2026-07-14T17:20:00", "results": []}}
            ]

        def get_nowait(self):
            if not self._messages:
                raise queue.Empty
            return self._messages.pop(0)

    class FakeProcess:
        pid = 12345

        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon
            self._alive = False

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

        def terminate(self):
            return None

        def kill(self):
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args, daemon):
            return FakeProcess(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(services.mp, "get_context", lambda _: FakeContext())
    monkeypatch.setattr(services, "_register_active_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "_clear_active_worker", lambda *args, **kwargs: None)

    payload = services._run_similar_pattern_analysis_isolated(timeout_seconds=1)

    assert captured["daemon"] is False
    assert payload["generated_at"] == "2026-07-14T17:20:00"


def test_isolated_similar_pattern_process_forwards_progress(monkeypatch) -> None:
    progress_messages = []

    class FakeQueue:
        def __init__(self):
            self._messages = [
                {"type": "progress", "message": "vector cache 200/5538 files usable=184 elapsed=14.4s"},
                {"type": "result", "ok": True, "payload": {"generated_at": "2026-07-16T18:40:00", "results": []}},
            ]

        def get_nowait(self):
            if not self._messages:
                raise queue.Empty
            return self._messages.pop(0)

    class FakeProcess:
        pid = 12345

        def __init__(self, *, target, args, daemon):
            self._alive = False

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

        def terminate(self):
            return None

        def kill(self):
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args, daemon):
            return FakeProcess(target=target, args=args, daemon=daemon)

    def fake_set_refresh_progress(**kwargs):
        progress_messages.append(kwargs["message"])

    monkeypatch.setattr(services.mp, "get_context", lambda _: FakeContext())
    monkeypatch.setattr(services, "_register_active_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "_clear_active_worker", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "_set_refresh_progress", fake_set_refresh_progress)

    payload = services._run_similar_pattern_analysis_isolated(timeout_seconds=1)

    assert payload["generated_at"] == "2026-07-16T18:40:00"
    assert progress_messages == ["相似走势决策台：vector cache 200/5538 files usable=184 elapsed=14.4s"]
