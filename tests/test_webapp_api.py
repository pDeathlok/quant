from fastapi.testclient import TestClient
import pandas as pd

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
