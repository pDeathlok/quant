from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from quant.webapp.app import app
from quant.webapp import services


client = TestClient(app)


def test_health_endpoint_reports_service_status() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "quant-webapp"}


def test_refresh_endpoint_rejects_unknown_scope() -> None:
    response = client.post("/api/selector/refresh-latest", json={"scope": "unknown"})

    assert response.status_code == 400
    assert "未知刷新范围" in response.json()["detail"]


def test_global_refresh_includes_daily_similar_pattern_step() -> None:
    step_keys = [step["key"] for step in services._progress_steps("all")]

    assert "convertible_bond_allotment" in step_keys
    assert "similar_patterns" in step_keys


def test_similar_pattern_refresh_updates_vector_caches(monkeypatch, tmp_path) -> None:
    calls = {"cache": 0}

    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: [])
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: pd.DataFrame())

    def fake_build(*args, **kwargs):
        calls["cache"] += 1
        return pd.DataFrame({"status": ["built", "cache_hit"]})

    monkeypatch.setattr(services, "build_vector_caches_parallel", fake_build)
    monkeypatch.setattr(services, "analyze_targets_by_threshold", lambda *args, **kwargs: {})

    payload = services.refresh_similar_pattern_analysis()

    assert calls["cache"] == 1
    assert payload["cache"]["rebuilt"] == 1
    assert payload["cache"]["reused"] == 1


def test_daily_payload_freshness_uses_generated_date() -> None:
    today = date(2026, 7, 13)

    assert services._is_daily_payload_current({"generated_at": "2026-07-13T08:30:00"}, today=today)
    assert not services._is_daily_payload_current({"generated_at": "2026-06-28T22:52:04"}, today=today)
    assert not services._is_daily_payload_current({}, today=today)


def test_allotment_tab_refreshes_stale_daily_cache(monkeypatch, tmp_path) -> None:
    calls = []
    stale = {"generated_at": "2026-06-28T22:52:04", "records": []}
    refreshed = {"generated_at": "2026-07-13T08:30:00", "records": [{"stock_code": "000001"}]}

    monkeypatch.setattr(services, "CONVERTIBLE_BOND_ALLOTMENT_DAILY_PATH", tmp_path / "allotments.json")
    monkeypatch.setattr(services, "_read_workspace_snapshot", lambda *args, **kwargs: stale)
    monkeypatch.setattr(services, "_write_workspace_snapshot", lambda *args, **kwargs: None)

    def fake_build(**kwargs):
        calls.append(kwargs["refresh"])
        return refreshed

    monkeypatch.setattr(services, "build_convertible_bond_allotment_payload", fake_build)

    payload = services.get_convertible_bond_allotments()

    assert calls == [True]
    assert payload == refreshed


def test_similar_pattern_tab_refreshes_stale_daily_cache(monkeypatch) -> None:
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

    assert calls == [True]
    assert payload["generated_at"].startswith("2026-07-13")
