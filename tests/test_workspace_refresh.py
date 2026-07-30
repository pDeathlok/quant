from __future__ import annotations

import threading

from quant.application.workspace_refresh import (
    WorkspaceRefreshOperations,
    refresh_daily_workspaces,
)


def test_refresh_daily_workspaces_runs_independent_jobs_in_parallel() -> None:
    calls: list[str] = []
    calls_lock = threading.Lock()
    barrier = threading.Barrier(6)

    def record(name: str) -> None:
        with calls_lock:
            calls.append(name)
        barrier.wait(timeout=2)

    def refresh_long(**kwargs):
        if kwargs["variant"] == "tea":
            record("long")
        return {"signal_date": kwargs["signal_date"], "stocks": []}

    operations = WorkspaceRefreshOperations(
        latest_signal_date=lambda: "2026-07-13",
        refresh_chan=lambda **kwargs: record("chan")
        or {"signal_date": kwargs["signal_date"], "candidates": []},
        refresh_long=refresh_long,
        refresh_convertible_bonds=lambda **kwargs: record("convertible_bond")
        or {"trade_date": kwargs["trade_date"], "candidates": []},
        refresh_allotments=lambda **kwargs: record("allotment")
        or {"generated_at": "2026-07-13T08:30:00", "records": []},
        refresh_byd=lambda **kwargs: record("byd")
        or {"planned_t": {"signal_date": "2026-07-13"}, "alerts": []},
        refresh_similar_patterns=lambda: record("similar")
        or {"generated_at": "2026-07-13T08:31:00", "results": []},
    )

    result = refresh_daily_workspaces(operations)

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
    assert result["convertible_bond_allotments"]["records"] == 0
    assert result["similar_patterns"]["targets"] == 0


def test_refresh_daily_workspaces_isolates_individual_failure() -> None:
    operations = WorkspaceRefreshOperations(
        latest_signal_date=lambda: "2026-07-13",
        refresh_chan=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("chan failed")),
        refresh_long=lambda **kwargs: {"stocks": []},
        refresh_convertible_bonds=lambda **kwargs: {"candidates": []},
        refresh_allotments=lambda **kwargs: {"records": []},
        refresh_byd=lambda **kwargs: {"planned_t": {}, "alerts": []},
        refresh_similar_patterns=lambda: {"results": []},
    )

    result = refresh_daily_workspaces(operations, max_workers=1)

    assert result["chan_model_strategy"] == {"status": "failed", "error": "chan failed"}
    assert result["long_stock_pool"]["status"] == "success"
    assert result["similar_patterns"]["status"] == "success"
