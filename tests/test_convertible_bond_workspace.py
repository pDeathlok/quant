from __future__ import annotations

import pytest

from quant.routine import convertible_bond_allotment as allotment_module

from quant.application.workspaces.convertible_bonds import (
    ConvertibleBondAllotmentDependencies,
    ConvertibleBondGridDependencies,
    build_convertible_bond_allotment_workspace,
    build_convertible_bond_grid_workspace,
    evaluate_convertible_bond_allotment_quality,
)


def test_grid_workspace_returns_cached_snapshot_without_rebuilding() -> None:
    cached = {"trade_date": "2026-07-18", "candidates": [{"ts_code": "123001.SZ"}]}
    dependencies = ConvertibleBondGridDependencies(
        read_snapshot=lambda *args, **kwargs: cached,
        read_legacy_snapshot=lambda: pytest.fail("legacy cache must not be read"),
        promote_legacy_snapshot=lambda *args, **kwargs: pytest.fail("legacy cache must not be promoted"),
        write_snapshot=lambda *args, **kwargs: pytest.fail("cache hit must not be rewritten"),
        refresh_daily=lambda *args, **kwargs: pytest.fail("cache hit must not refresh data"),
        build_plan=lambda *args, **kwargs: pytest.fail("cache hit must not rebuild"),
    )

    payload = build_convertible_bond_grid_workspace(
        trade_date="2026-07-18",
        limit=18,
        dependencies=dependencies,
    )

    assert payload is cached


def test_allotment_event_poll_advances_after_successful_empty_subsource_polls() -> None:
    result = allotment_module._event_poll_contract(
        refresh=True,
        today_text="20260812",
        sources={
            "pipeline": {"poll_status": "success", "rows": 0},
            "cninfo_issue": {"poll_status": "success", "rows": 0},
            "basic": {"poll_status": "success", "rows": 0},
            "issue": {"poll_status": "success", "rows": 0},
        },
    )

    assert result["event_poll_status"] == "success"
    assert result["event_polled_through"] == "2026-08-12"


def test_allotment_event_fallback_does_not_advance_poll_watermark() -> None:
    result = allotment_module._event_poll_contract(
        refresh=True,
        today_text="20260812",
        sources={
            "pipeline": {"poll_status": "success"},
            "cninfo_issue": {
                "poll_status": "failed",
                "available": True,
                "error": "provider unavailable",
            },
            "basic": {"poll_status": "success"},
            "issue": {"poll_status": "success"},
        },
    )

    assert result["event_poll_status"] == "failed"
    assert result["event_polled_through"] is None


def test_grid_workspace_promotes_eligible_legacy_snapshot() -> None:
    promoted: list[tuple[str, dict]] = []
    legacy = {
        "trade_date": "20260717",
        "generated_at": "2026-07-17T18:00:00",
        "candidates": [],
    }
    dependencies = ConvertibleBondGridDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        read_legacy_snapshot=lambda: legacy,
        promote_legacy_snapshot=lambda snapshot_date, payload: promoted.append(
            (snapshot_date, payload)
        ),
        write_snapshot=lambda *args, **kwargs: pytest.fail("legacy hit must not rebuild"),
        refresh_daily=lambda *args, **kwargs: pytest.fail("legacy hit must not refresh"),
        build_plan=lambda *args, **kwargs: pytest.fail("legacy hit must not rebuild"),
    )

    payload = build_convertible_bond_grid_workspace(
        trade_date="2026-07-18",
        limit=18,
        dependencies=dependencies,
    )

    assert payload["cache"]["backend"] == "legacy_filesystem"
    assert payload["cache"]["snapshot_date"] == "2026-07-17"
    assert payload["cache"]["requested_date"] == "2026-07-18"
    assert payload["cache"]["stale"] is True
    assert promoted == [("2026-07-17", payload)]


def test_grid_workspace_rejects_unparameterized_legacy_for_custom_limit() -> None:
    promoted: list[tuple[str, dict]] = []
    writes: list[tuple[tuple, dict]] = []
    legacy = {
        "trade_date": "20260717",
        "generated_at": "2026-07-17T18:00:00",
        "candidates": [{"ts_code": "123001.SZ"}],
    }
    dependencies = ConvertibleBondGridDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        read_legacy_snapshot=lambda: legacy,
        promote_legacy_snapshot=lambda snapshot_date, payload: promoted.append(
            (snapshot_date, payload)
        ),
        write_snapshot=lambda *args, **kwargs: writes.append((args, kwargs)),
        refresh_daily=lambda *args, **kwargs: pytest.fail("non-refresh build must not refresh data"),
        build_plan=lambda **kwargs: {
            "trade_date": "20260718",
            "candidates": [{"ts_code": f"12300{index}.SZ"} for index in range(kwargs["limit"])],
        },
    )

    payload = build_convertible_bond_grid_workspace(
        trade_date="2026-07-18",
        limit=8,
        dependencies=dependencies,
    )

    assert promoted == []
    assert len(payload["candidates"]) == 8
    assert payload["request_params"] == {"limit": 8}
    assert writes[0][1]["params"] == {"limit": 8}


def test_grid_workspace_refreshes_builds_and_persists() -> None:
    events: list[tuple[str, object]] = []
    dependencies = ConvertibleBondGridDependencies(
        read_snapshot=lambda *args, **kwargs: pytest.fail("refresh bypasses cache"),
        read_legacy_snapshot=lambda: pytest.fail("refresh bypasses legacy cache"),
        promote_legacy_snapshot=lambda *args, **kwargs: pytest.fail("refresh bypasses legacy cache"),
        write_snapshot=lambda *args, **kwargs: events.append(("write", (args, kwargs))),
        refresh_daily=lambda **kwargs: events.append(("refresh", kwargs)) or {"rows": 12},
        build_plan=lambda **kwargs: {
            "trade_date": kwargs["trade_date"],
            "candidates": [{"ts_code": "123001.SZ"}],
        },
    )

    payload = build_convertible_bond_grid_workspace(
        trade_date="20260718",
        limit=8,
        refresh=True,
        dependencies=dependencies,
    )

    assert payload["data_refresh"] == {"rows": 12}
    assert payload["request_params"] == {"limit": 8}
    assert events[0] == ("refresh", {"trade_date": "20260718"})
    assert events[1][0] == "write"


def test_allotment_quality_accepts_negative_finite_j_values() -> None:
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

    quality = evaluate_convertible_bond_allotment_quality(
        payload,
        expected_trade_date="20260723",
        minimum_coverage=0.90,
    )

    assert quality["status"] == "success"
    assert quality["metrics"]["kdj_weekly_j"]["count"] == 1
    assert quality["metrics"]["kdj_monthly_j"]["count"] == 1


def test_allotment_workspace_serves_stale_snapshot_when_daily_cache_is_missing() -> None:
    cached = {
        "generated_at": "2026-06-28T22:52:04",
        "records": [],
        "data_sources": {
            "stock_daily": {"requested": 0, "matched": 0, "error": None},
            "daily_basic": {"matched": 0, "error": None},
        },
    }
    dependencies = ConvertibleBondAllotmentDependencies(
        read_snapshot=lambda *args, **kwargs: cached,
        read_daily_cache=lambda: None,
        write_daily_cache=lambda payload: pytest.fail("cache hit must not write"),
        write_snapshot=lambda *args, **kwargs: pytest.fail("cache hit must not write"),
        build_payload=lambda **kwargs: pytest.fail("cache hit must not rebuild"),
        is_daily_current=lambda payload: False,
    )

    payload = build_convertible_bond_allotment_workspace(
        dependencies=dependencies,
    )

    assert payload["cache"]["hit"] is True
    assert payload["cache"]["stale"] is True
    assert payload["cache"]["source"] == "workspace_snapshot"
    assert payload["quality"]["status"] == "success"


def test_allotment_workspace_prefers_newer_daily_cache_to_stale_snapshot() -> None:
    snapshot = {
        "generated_at": "2026-07-30T21:16:27",
        "records": [{"stock_code": "300453"}],
        "data_sources": {
            "stock_daily": {"requested": 0, "matched": 0, "error": None},
            "daily_basic": {"matched": 0, "error": None},
        },
    }
    daily = {
        "generated_at": "2026-08-07T17:24:53",
        "records": [],
        "data_sources": {
            "stock_daily": {"requested": 0, "matched": 0, "error": None},
            "daily_basic": {"matched": 0, "error": None},
        },
    }
    dependencies = ConvertibleBondAllotmentDependencies(
        read_snapshot=lambda *args, **kwargs: snapshot,
        read_daily_cache=lambda: daily,
        write_daily_cache=lambda payload: pytest.fail("cache hit must not write"),
        write_snapshot=lambda *args, **kwargs: pytest.fail("cache hit must not write"),
        build_payload=lambda **kwargs: pytest.fail("cache hit must not rebuild"),
        is_daily_current=lambda payload: False,
    )

    payload = build_convertible_bond_allotment_workspace(
        dependencies=dependencies,
    )

    assert payload is daily
    assert payload["records"] == []
    assert payload["cache"]["source"] == "daily_cache"
    assert payload["cache"]["stale"] is True


def test_allotment_quality_gate_persists_failed_payload_before_raising() -> None:
    writes: list[str] = []
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
    dependencies = ConvertibleBondAllotmentDependencies(
        read_snapshot=lambda *args, **kwargs: None,
        read_daily_cache=lambda: None,
        write_daily_cache=lambda value: writes.append("daily"),
        write_snapshot=lambda *args, **kwargs: writes.append("snapshot"),
        build_payload=lambda **kwargs: payload,
        is_daily_current=lambda value: False,
    )

    with pytest.raises(RuntimeError, match="配债股数据质量门禁失败"):
        build_convertible_bond_allotment_workspace(
            refresh=True,
            expected_trade_date="2026-07-23",
            validate_quality=True,
            dependencies=dependencies,
        )

    assert payload["quality"]["status"] == "failed"
    assert writes == ["daily", "snapshot"]


@pytest.mark.parametrize("publish_success", [True, False])
def test_standalone_allotment_refresh_uses_committed_page_generation(
    monkeypatch, tmp_path, publish_success,
):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from quant.webapp import services
    from quant.infrastructure.publication import PublicationView, current_publication

    cases = json.loads((Path(__file__).parent / "fixtures/convertible_bond_announcement_cases.json").read_text())
    case = next(row for row in cases["timeline_cases"] if row["id"] == "hangya_20260912_registration_after_published_committee")
    views = [PublicationView(tmp_path, tmp_path / "data/publications", "old", ())]
    monkeypatch.setattr(services, "_publication_store", lambda: SimpleNamespace(view=lambda: views[0]))
    monkeypatch.setattr(services, "_REFRESH_STATUS", {})
    runs = []
    builds = []
    def run(scope, *, run_id):
        runs.append(scope)
        services._REFRESH_STATUS.update(run_id=run_id, status="success" if publish_success else "failed", error=None if publish_success else "quality failure")
        if publish_success:
            views[0] = PublicationView(tmp_path, tmp_path / "data/publications", "new", ())
    monkeypatch.setattr(services, "_run_latest_refresh_job", run)
    def build(**kwargs):
        builds.append(kwargs["refresh"])
        view = current_publication()
        index = 1 if kwargs["refresh"] or (view and view.generation == "new") else 0
        announcement = case["announcements"][index]
        return {"asof": "2026-09-12", "event_poll_status": "success", "event_polled_through": "2026-09-12", "quality": {"status": "success", "expected_trade_date": "2026-09-11"}, "data_sources": {"stock_daily": {"requested": 1, "matched": 1}, "daily_basic": {"matched": 1}}, "records": [{"stock_code": case["stock_code"], "stock_price_date": "2026-09-11", "kdj_daily_j": 10.0, "kdj_weekly_j": 20.0, "kdj_monthly_j": 30.0, "stage": "registered" if index else "exchange_approved", "announcement_title": announcement["title"], "announcement_url": announcement["url"]}]}
    monkeypatch.setattr(services, "build_convertible_bond_allotment_workspace", build)
    if publish_success:
        result = services.get_convertible_bond_allotments(refresh=True, validate_quality=True)
        assert runs == ["cbAllotment"]
        assert builds == [False]
        assert result["records"][0]["stage"] == "registered"
        assert result["records"][0]["announcement_url"] == case["expected"]["url"]
    else:
        with pytest.raises(RuntimeError, match="配债股发布失败"):
            services.get_convertible_bond_allotments(refresh=True, validate_quality=True)
        assert runs == ["cbAllotment"]
        assert builds == []
        assert views[0].generation == "old"


@pytest.mark.parametrize("gate_passes", [True, False])
def test_score_sync_updates_published_cache_atomically(monkeypatch, tmp_path, gate_passes):
    import json
    from quant.webapp import services
    from quant.routine import pipeline
    from quant.infrastructure.publication import PublicationStore, publication_path, publication_context
    from quant.data.atomic_io import atomic_write_json

    path = tmp_path / "scores.json"
    old = {"target_date": "2026-09-11", "watchlist": [{"symbol": "688510.SH", "opportunity_score": 1, "holding_score": 2}], "results": [{"target": {"symbol": "688510.SH"}}]}
    path.write_text(json.dumps(old))
    store = PublicationStore(tmp_path, ("scores.json",))
    audit = {"status": "success", "freshness_audit": {"status": "success", "failures": []}, "refresh_node_ids": []}
    with store.begin("initial") as view:
        store.commit(view, audit)
    before = store.view().generation
    monkeypatch.setattr(services, "_publication_store", lambda: store)
    monkeypatch.setattr(services, "SIMILAR_PATTERN_ANALYSIS_PATH", path)
    monkeypatch.setattr(services, "_stock_basic_for_similar_patterns", lambda: None)
    monkeypatch.setattr(services, "_similar_pattern_watchlist_profiles", lambda *a, **kw: [{"symbol": "688510.SH", "opportunity_score": 70, "holding_score": 80, "score_date": "2026-09-11", "feature_quality": {"status": "complete"}}])
    monkeypatch.setattr(services, "_read_similar_pattern_watchlist_symbols", lambda: ["688510.SH"])
    monkeypatch.setattr(services, "_REFRESH_STATUS", {})
    def run(scope, *, run_id):
        assert scope == "similar"
        try:
            with store.begin("score-refresh") as staged:
                services.refresh_similar_pattern_watchlist_scores()
                store.commit(staged, audit if gate_passes else {"status": "failed"})
        except Exception as exc:
            services._REFRESH_STATUS.update(run_id=run_id, status="failed", error=str(exc))
        else:
            services._REFRESH_STATUS.update(run_id=run_id, status="success")
    monkeypatch.setattr(services, "_run_latest_refresh_job", run)
    if gate_passes:
        result = services.refresh_similar_pattern_watchlist_scores()
        assert result["scored_count"] == 1
        assert store.view().generation != before
        with publication_context(store.view()):
            saved = json.loads(publication_path(path).read_text())
        assert saved["watchlist"][0]["opportunity_score"] == 70
        assert saved["results"] == old["results"]
    else:
        with pytest.raises(Exception, match="freshness audit"):
            services.refresh_similar_pattern_watchlist_scores()
        assert store.view().generation == before
        with publication_context(store.view()):
            assert json.loads(publication_path(path).read_text()) == old
    assert json.loads(path.read_text()) == old
