import threading
import time

from quant.routine import pipeline
from quant.webapp import services


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
    monkeypatch.setattr(pipeline, "refresh_data", lambda dry_run: {"status": "success"})
    monkeypatch.setattr(pipeline, "build_features", parallel_step)
    monkeypatch.setattr(pipeline, "refresh_strategy_signal_cache", parallel_step)
    monkeypatch.setattr(pipeline, "score_latest_models", lambda: {"status": "success"})
    monkeypatch.setattr(pipeline, "generate_daily_plan", parallel_step)
    monkeypatch.setattr(pipeline, "generate_dashboard", parallel_step)
    monkeypatch.setattr(pipeline, "refresh_daily_web_workspaces", lambda: {"status": "success"})
    monkeypatch.setattr(pipeline, "write_run_manifest", lambda results, strategies: tmp_path / "manifest.json")

    result = pipeline.run_daily_pipeline(skip_data=False, skip_backtest=True)

    assert max_active == 2
    assert result["steps"]["build_features"]["status"] == "success"
    assert result["steps"]["generate_dashboard"]["status"] == "success"
