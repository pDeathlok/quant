import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from quant.routine import left_side_unified_production, pipeline, right_side_unified_production
from quant.webapp import services


SCRIPT = Path(__file__).parents[1] / "scripts/research/postflight_selector_buy_hold_registry_v3.py"


def test_selector_postflight_validates_upstreams_before_freezing_dependency_baseline(monkeypatch):
    namespace = runpy.run_path(str(SCRIPT))
    regenerate = namespace["_regenerate_latest"]
    events = []
    monkeypatch.setitem(regenerate.__globals__, "_load_json", lambda path: {"result": {}})
    monkeypatch.setitem(regenerate.__globals__, "_page_score_audit", lambda *args: {"status": "success"})

    def ranker(side):
        def run(target_date):
            events.append(side)
            return {"status": "success", "checkpoint_reused": True}
        return run

    def dependency(*args, **kwargs):
        events.append(kwargs["phase"])
        return {"status": "success", "baseline_committed": kwargs["phase"] == "postflight"}

    monkeypatch.setattr(right_side_unified_production, "run_right_side_unified_production", ranker("right"))
    monkeypatch.setattr(left_side_unified_production, "run_left_side_production", ranker("left"))
    monkeypatch.setattr(pipeline, "publish_daily_dependency_contract", dependency)
    monkeypatch.setattr(services, "_ensure_selector_long_factor_snapshot", lambda *args, **kwargs: {"status": "success"})
    monkeypatch.setattr(services, "_clear_selector_caches", lambda: None)
    monkeypatch.setattr(services, "get_stock_selector_payload", lambda **kwargs: {"signal_date": kwargs["signal_date"], "stocks": []})
    monkeypatch.setattr(services, "_write_strategy_pool_snapshots", lambda *args, **kwargs: {})
    monkeypatch.setattr(services, "MarketDataStore", lambda config: SimpleNamespace(config=SimpleNamespace(sql_url=None)))

    _, report = regenerate("2026-08-26")

    assert events == ["right", "left", "preflight", "postflight"]
    assert report["ranker_refresh"]["right"]["checkpoint_reused"] is True


def _page():
    return {
        "signal_date": "2026-08-26",
        "model_release_id": services.SELECTOR_BUY_HOLD_RELEASE_ID,
        "factor_validation_schema": "selector-required-layers-v1",
        "stocks": [{
            "symbol": "000001.SZ", "date": "2026-08-26", "score_date": "2026-08-26",
            "model_score_available": True, "feature_quality": {"status": "complete"},
            "buy_score_source": "historical_return_model", "hold_score_source": "historical_return_model",
        }],
    }


def test_postflight_rejects_pre_validation_cached_release():
    audit = runpy.run_path(str(SCRIPT))["_page_score_audit"]
    page = _page()
    page.pop("factor_validation_schema")

    assert audit(page, "2026-08-26")["status"] == "failed"


def test_postflight_rejects_fallback_to_an_older_snapshot_date():
    audit = runpy.run_path(str(SCRIPT))["_page_score_audit"]

    report = audit(_page(), "2026-08-27")

    assert report["status"] == "failed"
    assert report["snapshot_date_matches"] is False


def test_postflight_rejects_unavailable_score_even_with_stale_success_sources():
    audit = runpy.run_path(str(SCRIPT))["_page_score_audit"]
    page = _page()
    page["stocks"][0]["model_score_available"] = False

    report = audit(page, "2026-08-26")

    assert report["status"] == "failed"
    assert report["incomplete_symbols"] == ["000001.SZ"]


@pytest.mark.parametrize("database_matches", [True, False])
def test_mysql_audit_checks_all_and_per_strategy_snapshot_scopes(
    monkeypatch, tmp_path, database_matches,
):
    audit = runpy.run_path(str(SCRIPT))["_mysql_snapshot_audit"]
    monkeypatch.setattr(services, "SELECTOR_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(services, "EXTENDED_STRATEGIES", [{"key": "CHANGAN"}])
    monkeypatch.setattr(services, "STRATEGY_GROUP_MEMBERS", {"B1": {"B1"}, "CHANGAN": {"CHANGAN"}})
    engine = create_engine("sqlite://")
    monkeypatch.setattr(
        services, "MarketDataStore",
        lambda config: SimpleNamespace(config=SimpleNamespace(sql_url="sqlite://"), _engine=lambda: engine),
    )
    with engine.begin() as connection:
        connection.execute(text(f"CREATE TABLE {services.SELECTOR_SNAPSHOT_TABLE} (snapshot_key TEXT, payload_json TEXT)"))
        for scope, extended in ((None, True), (["B1"], False), (["CHANGAN"], True)):
            key = services._selector_snapshot_key("2026-08-26", scope, extended)[0]
            current = {"generated_at": "2026-08-27T16:00:00", "scope": scope}
            services.atomic_write_json(current, services._selector_snapshot_path(key))
            stored = current if database_matches or scope != ["B1"] else {"generated_at": "old"}
            connection.execute(
                text(f"INSERT INTO {services.SELECTOR_SNAPSHOT_TABLE} VALUES (:key, :payload)"),
                {"key": key, "payload": json.dumps(stored)},
            )

    report = audit({"available_strategies": [{"key": "B1"}, {"key": "CHANGAN"}]}, "2026-08-26")

    assert report["status"] == ("success" if database_matches else "failed")
    assert report["checked_snapshot_count"] == 3
    assert len(report["mismatched_snapshot_keys"]) == (0 if database_matches else 1)


def test_mysql_audit_fails_closed_when_snapshot_file_is_missing(monkeypatch, tmp_path):
    audit = runpy.run_path(str(SCRIPT))["_mysql_snapshot_audit"]
    monkeypatch.setattr(services, "SELECTOR_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(
        services, "MarketDataStore",
        lambda config: SimpleNamespace(config=SimpleNamespace(sql_url="sqlite://")),
    )

    report = audit({}, "2026-08-26")

    assert report["status"] == "failed"
    assert report["error_type"] == "FileNotFoundError"


def test_postflight_accepts_today_all_scope_baseline_instead_of_yesterday_short(monkeypatch, tmp_path):
    read = runpy.run_path(str(SCRIPT))["_committed_selector_dependency_snapshot"]
    monkeypatch.setitem(read.__globals__, "PROJECT_ROOT", tmp_path)
    directory = tmp_path / "data/contracts/daily_dependencies"
    for scope, day in (("short", "2026-08-26"), ("all", "2026-08-27")):
        services.atomic_write_json({
            "status": "success", "baseline_committed": True, "target_trade_date": day,
            "scope": scope, "model_contract_hashes": {"score.selector": "validated"},
        }, directory / f"latest-{scope}.json")

    assert read("2026-08-27")["scope"] == "all"


def test_postflight_rejects_a_failed_or_stale_committed_baseline(monkeypatch, tmp_path):
    read = runpy.run_path(str(SCRIPT))["_committed_selector_dependency_snapshot"]
    monkeypatch.setitem(read.__globals__, "PROJECT_ROOT", tmp_path)
    services.atomic_write_json({
        "status": "failed", "baseline_committed": False, "target_trade_date": "2026-08-27",
        "model_contract_hashes": {"score.selector": "not_committed"},
    }, tmp_path / "data/contracts/daily_dependencies/latest-all.json")

    with pytest.raises(RuntimeError, match="no successful committed selector dependency baseline"):
        read("2026-08-27")
