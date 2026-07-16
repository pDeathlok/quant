from __future__ import annotations

from datetime import date
from pathlib import Path

from quant.routine import web_refresh_runner as runner


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def _pop(self):
        if not self.responses:
            raise AssertionError("no more fake responses")
        return self.responses.pop(0)

    def get(self, url, timeout):
        self.requests.append(("GET", url, timeout, None))
        return self._pop()

    def post(self, url, json, timeout):
        self.requests.append(("POST", url, timeout, json))
        return self._pop()


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.text = payload if isinstance(payload, str) else ""

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeFetcher:
    def __init__(self, rows):
        self.rows = rows

    def get_trade_calendar(self, **kwargs):
        import pandas as pd

        return pd.DataFrame(self.rows)


def test_load_env_file_parses_comments_and_quotes(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "TUSHARE_TOKEN=\"abc\"",
                "EMPTY=",
                "RAW=value",
            ]
        ),
        encoding="utf-8",
    )

    loaded = runner.load_env_file(env_path)

    assert loaded == {"TUSHARE_TOKEN": "abc", "EMPTY": "", "RAW": "value"}


def test_decide_trade_day_skips_closed_day() -> None:
    decision = runner.decide_trade_day(
        target_date=date(2026, 7, 18),
        fetcher_factory=lambda: FakeFetcher([]),
    )

    assert decision.should_run is False
    assert "跳过刷新" in decision.reason


def test_run_refresh_workflow_retries_after_failed_status(tmp_path: Path) -> None:
    responses = [
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse("<html>quant</html>"),
        FakeResponse({"status": "idle"}),
        FakeResponse({"status": "queued", "percent": 0, "message": "queued"}),
        FakeResponse(
            {
                "status": "failed",
                "started_at": "2026-07-15T16:00:00",
                "finished_at": "2026-07-15T16:10:00",
                "updated_at": "2026-07-15T16:10:00",
                "percent": 97,
                "current_step": "similar_patterns",
                "message": "failed",
                "error": "similar_patterns failed",
            }
        ),
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse("<html>quant</html>"),
        FakeResponse({"status": "failed"}),
        FakeResponse({"status": "queued", "percent": 90, "message": "resume queued"}),
        FakeResponse(
            {
                "status": "success",
                "started_at": "2026-07-15T16:00:00",
                "finished_at": "2026-07-15T16:20:00",
                "updated_at": "2026-07-15T16:20:00",
                "percent": 100,
                "current_step": "snapshot",
                "message": "done",
                "result": {"refresh_data": {"failed": 11}},
            }
        ),
    ]
    session = FakeSession(responses)
    logs: list[str] = []
    sleep_calls: list[float] = []
    env_path = tmp_path / ".env"
    env_path.write_text("TUSHARE_TOKEN=test\n", encoding="utf-8")
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        env_path=env_path,
        poll_seconds=0,
        retry_delay_seconds=0,
        max_attempts=2,
        service_log_path=tmp_path / "service.log",
    )

    result = runner.run_refresh_workflow(
        config=config,
        target_date=date(2026, 7, 15),
        session=session,
        fetcher_factory=lambda: FakeFetcher([{"exchange": "SSE", "cal_date": "20260715", "is_open": 1}]),
        sleep_fn=sleep_calls.append,
        monotonic_fn=lambda: 0.0,
        print_fn=logs.append,
    )

    assert result["status"] == "success"
    assert result["attempts"] == 2
    assert result["failed_count"] == 11
    assert result["cache_cleanup"]["status"] == "success"
    assert any("自动重试" in line for line in logs)


def test_ensure_local_service_checks_frontend_and_starts_stack(monkeypatch, tmp_path: Path) -> None:
    responses = [
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse(""),
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse("<html>quant</html>"),
    ]
    session = FakeSession(responses)
    client = runner.RefreshApiClient("http://127.0.0.1:8088/api", session=session)
    logs: list[str] = []
    sleep_calls: list[float] = []
    started = {}

    class FakeProcess:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        started["args"] = args
        started["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        service_log_path=tmp_path / "service.log",
        health_timeout_seconds=5,
    )

    process = runner.ensure_local_service(
        config=config,
        client=client,
        sleep_fn=sleep_calls.append,
        print_fn=logs.append,
    )

    assert process is not None
    assert started["args"][0][-1] == "scripts/run_webapp.py"
    assert ("GET", "http://127.0.0.1:8088/", 10, None) in session.requests
    assert any("前后端未就绪" in line for line in logs)
    assert any("已启动前后端" in line for line in logs)


def test_run_refresh_workflow_skips_when_trade_day_unknown(tmp_path: Path) -> None:
    logs: list[str] = []
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    config = runner.RefreshRunnerConfig(env_path=env_path)

    def failing_fetcher():
        raise RuntimeError("network unavailable")

    result = runner.run_refresh_workflow(
        config=config,
        target_date=date(2026, 7, 15),
        fetcher_factory=failing_fetcher,
        print_fn=logs.append,
    )

    assert result["status"] == "skipped"
    assert "无法可靠确认" in result["reason"]


def test_run_cache_cleanup_reports_error_without_raising(tmp_path: Path) -> None:
    def failing_cleanup(project_root: Path, reference_date: date | None):
        raise OSError("permission denied")

    result = runner.run_cache_cleanup(
        tmp_path,
        date(2026, 7, 15),
        cleanup_fn=failing_cleanup,
    )

    assert result == {
        "status": "failed",
        "reference_date": "2026-07-15",
        "reclaimed_bytes": 0,
        "errors": ["permission denied"],
    }
