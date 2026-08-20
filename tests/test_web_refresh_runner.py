from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

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


def test_decide_trade_day_marks_calendar_failure_as_error() -> None:
    def raise_fetcher():
        raise RuntimeError("calendar unavailable")

    decision = runner.decide_trade_day(
        target_date=date(2026, 7, 15),
        fetcher_factory=raise_fetcher,
    )

    assert decision.should_run is False
    assert decision.error is True
    assert "无法可靠确认" in decision.reason


def test_run_refresh_workflow_rejects_second_process_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "daily-refresh.lock"
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        runner_lock_path=lock_path,
    )

    with runner.acquire_runner_lock(lock_path):
        result = runner.run_refresh_workflow(config=config, print_fn=lambda _: None)

    assert result["status"] == "busy"
    assert str(lock_path) in result["reason"]


def test_runner_reuses_service_by_default_and_supports_explicit_restart() -> None:
    config = runner.RefreshRunnerConfig()
    args = runner.build_parser().parse_args([])

    assert config.restart_service is False
    assert args.restart_service is False
    assert runner.build_parser().parse_args(["--restart-service"]).restart_service is True
    assert runner.build_parser().parse_args(["--no-restart-service"]).restart_service is False


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
                "result": {
                    "refresh_data": {"failed": 11},
                    "cache_cleanup": {
                        "status": "success",
                        "reclaimed_bytes": 123,
                        "errors": [],
                    },
                },
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
        restart_service=False,
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
    assert result["cache_cleanup"]["reclaimed_bytes"] == 123
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

    monkeypatch.setattr(runner, "resolve_service_python", lambda project_root: Path("/test/python"))
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        service_log_path=tmp_path / "service.log",
        service_pid_path=tmp_path / "service.pid",
        health_timeout_seconds=5,
    )

    process = runner.ensure_local_service(
        config=config,
        client=client,
        sleep_fn=sleep_calls.append,
        print_fn=logs.append,
    )

    assert process is not None
    command = started["args"][0]
    assert command[0] == "/test/python"
    assert command[1] == "scripts/run_webapp.py"
    assert command[command.index("--log-file") + 1] == str(tmp_path / "service.log")
    assert started["kwargs"]["stdout"] is runner.subprocess.DEVNULL
    assert started["kwargs"]["stderr"] is runner.subprocess.DEVNULL
    assert started["kwargs"]["start_new_session"] is True
    assert started["kwargs"]["close_fds"] is True
    assert (tmp_path / "service.pid").read_text(encoding="utf-8") == "12345\n"
    assert ("GET", "http://127.0.0.1:8088/", 10, None) in session.requests
    assert any("前后端未就绪" in line for line in logs)
    assert any("已常驻后台启动前后端" in line for line in logs)


def test_ensure_local_service_force_restarts_pid_file_service(monkeypatch, tmp_path: Path) -> None:
    responses = [
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse("<html>quant</html>"),
    ]
    session = FakeSession(responses)
    client = runner.RefreshApiClient("http://127.0.0.1:8088/api", session=session)
    pid_path = tmp_path / "service.pid"
    pid_path.write_text("22257\n", encoding="utf-8")
    logs: list[str] = []
    sleep_calls: list[float] = []
    kill_calls: list[tuple[int, int]] = []
    clock = iter([0.0, 0.1, 0.2, 0.3])
    running_checks = iter([True, False])
    started = {}

    class FakeProcess:
        pid = 33333

        def poll(self):
            return None

    def fake_killpg(pid: int, sig: int) -> None:
        kill_calls.append((pid, sig))

    def fake_process_is_running(pid: int) -> bool:
        return next(running_checks)

    def fake_popen(*args, **kwargs):
        started["args"] = args
        started["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runner, "resolve_service_python", lambda project_root: Path("/test/python"))
    monkeypatch.setattr(runner.os, "killpg", fake_killpg)
    monkeypatch.setattr(runner, "process_is_running", fake_process_is_running)
    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        service_log_path=tmp_path / "service.log",
        service_pid_path=pid_path,
        health_timeout_seconds=5,
    )

    process = runner.ensure_local_service(
        config=config,
        client=client,
        force_restart=True,
        sleep_fn=sleep_calls.append,
        monotonic_fn=lambda: next(clock),
        print_fn=logs.append,
    )

    assert process is not None
    assert kill_calls == [(22257, runner.signal.SIGTERM)]
    assert pid_path.read_text(encoding="utf-8") == "33333\n"
    assert started["kwargs"]["start_new_session"] is True
    assert any("准备重启常驻 web 服务" in line for line in logs)


def test_ensure_local_service_treats_busy_existing_port_as_recoverable(
    monkeypatch, tmp_path: Path
) -> None:
    session = FakeSession([])
    client = runner.RefreshApiClient("http://127.0.0.1:8088/api", session=session)
    logs: list[str] = []
    clock = iter([0.0, 0.1])

    class FakeProcess:
        pid = 44444

        def poll(self):
            return 1

    monkeypatch.setattr(runner, "resolve_service_python", lambda project_root: Path("/test/python"))
    monkeypatch.setattr(runner, "is_service_port_listening", lambda base_url: True)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        service_log_path=tmp_path / "service.log",
        service_pid_path=tmp_path / "service.pid",
        health_timeout_seconds=5,
    )

    process = runner.ensure_local_service(
        config=config,
        client=client,
        sleep_fn=lambda _: None,
        monotonic_fn=lambda: next(clock),
        print_fn=logs.append,
    )

    assert process is None
    assert not (tmp_path / "service.pid").exists()
    assert any("端口仍有服务监听" in line for line in logs)


def test_ensure_local_service_restarts_launchd_managed_service(
    monkeypatch, tmp_path: Path
) -> None:
    responses = [
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse("<html>quant</html>"),
        FakeResponse({"status": "ok", "service": "quant-webapp"}),
        FakeResponse("<html>quant</html>"),
    ]
    session = FakeSession(responses)
    client = runner.RefreshApiClient("http://127.0.0.1:8088/api", session=session)
    logs: list[str] = []

    monkeypatch.setattr(runner, "stop_service_from_pid_file", lambda *args, **kwargs: False)
    monkeypatch.setattr(runner, "restart_launchd_service", lambda: (True, None))

    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        service_log_path=tmp_path / "service.log",
        service_pid_path=tmp_path / "service.pid",
    )
    process = runner.ensure_local_service(
        config=config,
        client=client,
        force_restart=True,
        print_fn=logs.append,
    )

    assert process is None
    assert any("已重启 launchd 托管" in line for line in logs)


def test_ensure_local_service_fails_if_external_service_cannot_restart(
    monkeypatch, tmp_path: Path
) -> None:
    session = FakeSession(
        [
            FakeResponse({"status": "ok", "service": "quant-webapp"}),
            FakeResponse("<html>quant</html>"),
        ]
    )
    client = runner.RefreshApiClient("http://127.0.0.1:8088/api", session=session)

    monkeypatch.setattr(runner, "stop_service_from_pid_file", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        runner,
        "restart_launchd_service",
        lambda: (False, "service not registered"),
    )
    config = runner.RefreshRunnerConfig(
        project_root=tmp_path,
        service_log_path=tmp_path / "service.log",
        service_pid_path=tmp_path / "service.pid",
    )

    with pytest.raises(RuntimeError, match="无法通过 launchd 重启"):
        runner.ensure_local_service(
            config=config,
            client=client,
            force_restart=True,
        )


def test_run_refresh_workflow_fails_when_trade_day_unknown(tmp_path: Path) -> None:
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

    assert result["status"] == "failed"
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
