import queue
from datetime import date

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
        "get_stock_selector_payload",
        lambda **kwargs: {
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


def test_watchlist_notes_are_saved_and_returned_with_stock_profiles(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text(
        '{"symbols":["002594.SZ"],"notes":{}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())
    monkeypatch.setattr(
        services,
        "_stock_basic_for_similar_patterns",
        lambda: pd.DataFrame(
            [{"ts_code": "002594.SZ", "name": "比亚迪", "industry": "汽车整车"}]
        ),
    )

    payload = services.save_similar_pattern_watch_note(
        "002594.SZ",
        "操作计划：回踩 20 日线后分批关注",
    )

    stock = payload["stocks"][0]
    persisted = watchlist_path.read_text(encoding="utf-8")
    assert stock["note"] == "操作计划：回踩 20 日线后分批关注"
    assert stock["note_updated_at"]
    assert "回踩 20 日线" in persisted


def test_watchlist_note_rejects_stock_outside_watchlist(monkeypatch, tmp_path) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    watchlist_path.write_text('{"symbols":["002594.SZ"]}', encoding="utf-8")
    monkeypatch.setattr(services, "SIMILAR_PATTERN_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(services, "_normalize_watch_symbol", lambda symbol: str(symbol).upper())

    with pytest.raises(ValueError, match="不在自选池"):
        services.save_similar_pattern_watch_note("000001.SZ", "观察")


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
