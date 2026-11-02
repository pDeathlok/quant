from __future__ import annotations

from quant.routine import pipeline
from quant.routine.web_refresh_runner import RefreshApiClient
from quant.webapp import api, services


def test_refresh_client_forwards_requested_date():
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"status": "queued"}

    class Session:
        def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            return Response()

    client = RefreshApiClient("http://localhost/api", session=Session())
    client.start_refresh("all", "2026-10-28")
    client.start_refresh("all")
    assert calls == [{"scope": "all", "trade_date": "2026-10-28"}, {"scope": "all"}]


def test_api_forwards_requested_date(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "start_latest_refresh", lambda **kwargs: calls.append(kwargs) or {})
    api.selector_refresh_latest(api.RefreshLatestRequest(trade_date="2026-10-28"))
    assert calls == [{"scope": "all", "target_trade_date": "2026-10-28"}]


def test_source_commands_are_bounded_by_requested_date(monkeypatch):
    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20261029")
    result = pipeline.refresh_data(source_end_date="20261028")
    assert "--end 20261028" in result["command"]
    assert "--start 20261028" in result["command"]


def test_requested_date_reaches_source_through_production_composer(monkeypatch, tmp_path):
    from quant.routine.composed_operations import execute_composed_operation

    monkeypatch.setattr(pipeline, "_incremental_daily_start", lambda: "20261018")
    result = execute_composed_operation(
        "refresh_data", pipeline.refresh_data, target_trade_date="2026-10-28",
        scope="all", project_root=tmp_path, dry_run=True, source_end_date="20261028",
    )
    assert "--end 20261028" in result["command"]


def test_daily_basic_receives_requested_date(monkeypatch):
    import quant.routine.daily_basic_refresh as basic

    calls = []
    monkeypatch.setattr(pipeline, "_incremental_daily_basic_start", lambda: "20261029")
    monkeypatch.setattr(basic, "refresh_daily_basic", lambda **kwargs: calls.append(kwargs) or {})
    pipeline.refresh_daily_basic_data(dry_run=False, source_end_date="20261028")
    assert calls[0]["end_date"] == calls[0]["start_date"] == "20261028"


def test_service_passes_date_to_background_job_without_cross_date_resume(monkeypatch):
    calls = []

    class Thread:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(pipeline, "_latest_daily_trade_date", lambda: "20261028")
    monkeypatch.setattr(services, "_REFRESH_STATUS", {"status": "failed"})
    monkeypatch.setattr(services, "_refresh_job_active_unlocked", lambda _: False)
    monkeypatch.setattr(services, "_refresh_resume_ready", lambda *args: True)
    monkeypatch.setattr(services, "_completed_checkpoint_ready", lambda *args: False)
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    monkeypatch.setattr(services.threading, "Thread", Thread)
    result = services.start_latest_refresh(target_trade_date="2026-10-28")
    assert calls[0]["kwargs"] == {"target_trade_date": "20261028"}
    assert calls[0]["args"][1] is None
    assert result["trade_date"] == "20261028"


def test_other_dates_success_does_not_satisfy_requested_date(monkeypatch, tmp_path):
    from quant.routine import web_refresh_runner as runner

    class Client:
        starts = 0

        def get_status(self):
            return {"status": "idle"}

        def start_refresh(self, scope):
            self.starts += 1
            return {"status": "queued"}

    client = Client()
    results = iter([
        {"status": "success", "trade_date": "20261028"},
        {"status": "success", "trade_date": "20261029"},
    ])
    monkeypatch.setattr(runner, "RefreshApiClient", lambda *a, **k: client)
    monkeypatch.setattr(runner, "ensure_local_service", lambda **k: None)
    monkeypatch.setattr(runner, "decide_trade_day", lambda **k:
        runner.TradeDayDecision(True, "open", "20261029"))
    monkeypatch.setattr(runner, "wait_for_terminal_status", lambda **k: next(results))
    result = runner.run_refresh_workflow(
        runner.RefreshRunnerConfig(
            project_root=tmp_path, env_path=tmp_path / "missing.env",
            runner_lock_path=tmp_path / "runner.lock",
        ), print_fn=lambda _: None,
    )
    assert result["status"] == "success"
    assert client.starts == 2
