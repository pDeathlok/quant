import io
import threading
import time
import json

import pandas as pd

from quant.routine import pipeline
from quant.webapp import services


def test_refresh_daily_basic_reports_partial_failure(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "_incremental_daily_basic_start", lambda: "20260720")
    monkeypatch.setattr(
        "quant.routine.daily_basic_refresh.refresh_daily_basic",
        lambda **kwargs: {"trade_dates": 2, "success": 1, "failed": 1, "start_date": kwargs["start_date"]},
    )

    result = pipeline.refresh_daily_basic_data(dry_run=False)

    assert result["status"] == "failed"
    assert result["start_date"] == "20260720"


def test_analyst_snapshot_reuses_today_checkpoint(monkeypatch, tmp_path) -> None:
    output = tmp_path / "data/raw/analyst_forecasts.parquet"
    output.parent.mkdir(parents=True)
    today = pd.Timestamp.now().normalize()
    pd.DataFrame(
        {
            "source": ["akshare_em_snapshot"],
            "report_date": [today],
        }
    ).to_parquet(output, index=False)
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess should not run")),
    )

    result = pipeline._refresh_analyst_forecast_snapshot()

    assert result["status"] == "skipped"
    assert result["latest_report_date"] == today.date().isoformat()


def test_chan_refresh_requires_current_completion_manifest(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports/chan_daily/model_filter"
    report_dir.mkdir(parents=True)
    scored_path = report_dir / "chan_model_scored_candidates.parquet"
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20260721")

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def run_without_manifest(*args, **kwargs):
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", run_without_manifest)

    stale = pipeline.refresh_chan_model_scores()
    assert stale["status"] == "failed"

    (report_dir / "live_refresh_manifest.json").write_text(
        json.dumps({"processed_through": "2026-07-20"}),
        encoding="utf-8",
    )

    def run_with_current_manifest(*args, **kwargs):
        (report_dir / "live_refresh_manifest.json").write_text(
            json.dumps({"processed_through": "2026-07-21"}),
            encoding="utf-8",
        )
        return Result()

    monkeypatch.setattr(pipeline.subprocess, "run", run_with_current_manifest)
    current = pipeline.refresh_chan_model_scores()
    assert current["status"] == "success"
    assert current["processed_through"] == "2026-07-21"


def test_build_features_uses_process_executor_by_default(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        stdout = io.StringIO("")

        def __init__(self, command, **kwargs) -> None:
            captured["command"] = command

        def wait(self) -> int:
            return 0

    monkeypatch.delenv("ROUTINE_FEATURE_EXECUTOR", raising=False)
    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(pipeline, "_incremental_feature_start", lambda: "20260721")

    result = pipeline.build_features()

    command = captured["command"]
    assert command[command.index("--executor") + 1] == "processes"
    assert result["status"] == "success"


def test_daily_web_workspaces_refreshes_every_non_short_tab_in_parallel(monkeypatch) -> None:
    calls: list[str] = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(6)

    def record(name: str) -> None:
        with calls_lock:
            calls.append(name)
        barrier.wait(timeout=2)

    monkeypatch.setattr(services, "_latest_candidate_signal_date", lambda: "2026-07-13")
    monkeypatch.setattr(
        services,
        "get_chan_model_strategy_plan",
        lambda **kwargs: record("chan") or {"signal_date": kwargs["signal_date"], "candidates": []},
    )

    def get_long_stock_pool(**kwargs):
        if kwargs["variant"] == "tea":
            record("long")
        return {"signal_date": kwargs["signal_date"], "stocks": []}

    monkeypatch.setattr(services, "get_long_stock_pool", get_long_stock_pool)
    monkeypatch.setattr(
        services,
        "get_convertible_bond_grid_plan",
        lambda **kwargs: record("convertible_bond") or {"trade_date": kwargs["trade_date"], "candidates": []},
    )

    monkeypatch.setattr(
        services,
        "get_convertible_bond_allotments",
        lambda **kwargs: record("allotment")
        or {"generated_at": "2026-07-13T08:30:00", "records": []},
    )
    monkeypatch.setattr(
        services,
        "get_byd_daily_strategy",
        lambda **kwargs: record("byd") or {"planned_t": {"signal_date": "2026-07-13"}, "alerts": []},
    )
    monkeypatch.setattr(
        services,
        "refresh_similar_pattern_analysis",
        lambda: record("similar")
        or {"generated_at": "2026-07-13T08:31:00", "results": []},
    )

    result = pipeline.refresh_daily_web_workspaces()

    assert set(calls) == {"chan", "long", "convertible_bond", "allotment", "byd", "similar"}
    assert set(result) == {
        "chan_model_strategy",
        "long_stock_pool",
        "convertible_bond_plan",
        "convertible_bond_allotments",
        "byd_daily_plan",
        "similar_patterns",
    }
    assert all(item["status"] == "success" for item in result.values())
    assert result["convertible_bond_allotments"]["status"] == "success"
    assert result["similar_patterns"]["status"] == "success"


def test_daily_web_workspaces_isolates_individual_failure(monkeypatch) -> None:
    monkeypatch.setattr(services, "_latest_candidate_signal_date", lambda: "2026-07-13")
    monkeypatch.setattr(
        services,
        "get_chan_model_strategy_plan",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("chan failed")),
    )
    monkeypatch.setattr(services, "get_long_stock_pool", lambda **kwargs: {"stocks": []})
    monkeypatch.setattr(services, "get_convertible_bond_grid_plan", lambda **kwargs: {"candidates": []})
    monkeypatch.setattr(services, "get_convertible_bond_allotments", lambda **kwargs: {"records": []})
    monkeypatch.setattr(services, "get_byd_daily_strategy", lambda **kwargs: {"planned_t": {}, "alerts": []})
    monkeypatch.setattr(services, "refresh_similar_pattern_analysis", lambda: {"results": []})

    result = pipeline.refresh_daily_web_workspaces(max_workers=1)

    assert result["chan_model_strategy"] == {"status": "failed", "error": "chan failed"}
    assert result["long_stock_pool"]["status"] == "success"
    assert result["similar_patterns"]["status"] == "success"


def test_daily_pipeline_parallelizes_independent_stages(monkeypatch, tmp_path) -> None:
    active = 0
    max_active = 0
    call_order: list[str] = []
    lock = threading.Lock()

    def parallel_step() -> dict:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"status": "success"}

    monkeypatch.setattr(pipeline, "load_strategy_configs", lambda path: [])
    monkeypatch.setattr(
        pipeline,
        "run_cache_cleanup",
        lambda project_root: call_order.append("cleanup") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_data",
        lambda dry_run: call_order.append("refresh") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_daily_basic_data",
        lambda dry_run: call_order.append("daily_basic") or {"status": "success"},
    )
    monkeypatch.setattr(
        pipeline,
        "refresh_reference_inputs",
        lambda dry_run, include_financials: call_order.append("reference") or {"status": "success"},
    )
    monkeypatch.setattr(pipeline, "build_features", parallel_step)
    monkeypatch.setattr(pipeline, "refresh_strategy_signal_cache", parallel_step)
    monkeypatch.setattr(pipeline, "score_latest_models", lambda: {"status": "success"})
    monkeypatch.setattr(pipeline, "refresh_chan_model_scores", lambda: {"status": "success"})
    monkeypatch.setattr(pipeline, "generate_daily_plan", parallel_step)
    monkeypatch.setattr(pipeline, "generate_dashboard", parallel_step)
    monkeypatch.setattr(pipeline, "refresh_daily_web_workspaces", lambda: {"status": "success"})
    monkeypatch.setattr(pipeline, "write_run_manifest", lambda results, strategies: tmp_path / "manifest.json")

    result = pipeline.run_daily_pipeline(skip_data=False, skip_backtest=True)

    assert call_order[:4] == ["cleanup", "refresh", "daily_basic", "reference"]
    assert max_active == 2
    assert result["steps"]["cache_cleanup"]["status"] == "success"
    assert result["steps"]["build_features"]["status"] == "success"
    assert result["steps"]["generate_dashboard"]["status"] == "success"


def test_daily_pipeline_stops_before_features_when_source_refresh_is_incomplete(monkeypatch, tmp_path) -> None:
    feature_called = False

    def build_should_not_run():
        nonlocal feature_called
        feature_called = True
        return {"status": "success"}

    monkeypatch.setattr(pipeline, "load_strategy_configs", lambda path: [])
    monkeypatch.setattr(pipeline, "run_cache_cleanup", lambda project_root: {"status": "success"})
    monkeypatch.setattr(pipeline, "refresh_data", lambda dry_run: {"status": "failed", "failed": 1})
    monkeypatch.setattr(pipeline, "build_features", build_should_not_run)
    monkeypatch.setattr(pipeline, "write_run_manifest", lambda results, strategies: tmp_path / "manifest.json")

    result = pipeline.run_daily_pipeline(skip_data=False, skip_backtest=True)

    assert result["status"] == "failed"
    assert result["steps"]["pipeline"]["status"] == "failed"
    assert feature_called is False
