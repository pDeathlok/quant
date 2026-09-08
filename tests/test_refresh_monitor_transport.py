from datetime import date

import pytest
import requests

from quant.routine import web_refresh_runner as runner


class StatusClient:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.starts = 0

    def get_status(self):
        status = next(self.statuses)
        if isinstance(status, Exception):
            raise status
        return status

    def start_refresh(self, _scope):
        self.starts += 1
        return {"status": "queued"}


def test_transient_poll_failures_do_not_consume_refresh_retry_budget(monkeypatch, tmp_path):
    client = StatusClient([
        {"status": "idle"},
        {"status": "running", "job_active": True, "heartbeat_at": "first"},
        *[requests.ReadTimeout("busy") for _ in range(5)],
        {"status": "success", "job_active": False},
    ])
    monkeypatch.setattr(runner, "RefreshApiClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, "ensure_local_service", lambda **kwargs: None)
    monkeypatch.setattr(
        runner, "decide_trade_day",
        lambda **kwargs: runner.TradeDayDecision(True, "open", "20260908"),
    )

    result = runner.run_refresh_workflow(
        runner.RefreshRunnerConfig(project_root=tmp_path, env_path=tmp_path / "absent.env", max_attempts=1),
        target_date=date(2026, 9, 8), sleep_fn=lambda _: None,
        monotonic_fn=lambda: 0.0, print_fn=lambda _: None,
    )

    assert result["status"] == "success"
    assert result["attempts"] == client.starts == 1


@pytest.mark.parametrize("failure", [requests.ReadTimeout, requests.ConnectionError])
def test_unreachable_monitor_retains_original_deadline(failure):
    clock = iter([0.0, 1.0, 6.0])
    client = StatusClient([failure("offline"), failure("offline")])

    with pytest.raises(TimeoutError, match="无法确认"):
        runner.wait_for_terminal_status(
            client, runner.RefreshRunnerConfig(no_progress_timeout_seconds=5),
            sleep_fn=lambda _: None, monotonic_fn=lambda: next(clock), print_fn=lambda _: None,
        )


def test_programming_errors_are_not_retried_as_transport_failures():
    client = StatusClient([ValueError("invalid state")])
    with pytest.raises(ValueError, match="invalid state"):
        runner.wait_for_terminal_status(client, runner.RefreshRunnerConfig())


@pytest.mark.parametrize("listening", [False, True])
def test_unknown_owner_on_listening_port_prevents_explicit_restart(monkeypatch, tmp_path, listening):
    client = StatusClient([
        requests.ReadTimeout("owner unknown"),
        {"status": "running", "job_active": True},
        {"status": "success", "job_active": False},
    ])
    restarts = []
    monkeypatch.setattr(runner, "RefreshApiClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, "is_service_port_listening", lambda _: listening)
    monkeypatch.setattr(
        runner, "ensure_local_service",
        lambda **kwargs: restarts.append(kwargs["force_restart"]),
    )
    monkeypatch.setattr(
        runner, "decide_trade_day",
        lambda **kwargs: runner.TradeDayDecision(True, "open", "20260908"),
    )

    result = runner.run_refresh_workflow(
        runner.RefreshRunnerConfig(project_root=tmp_path, env_path=tmp_path / "absent.env", restart_service=True),
        sleep_fn=lambda _: None, print_fn=lambda _: None,
    )

    assert result["status"] == "success"
    assert client.starts == 0
    assert restarts == [not listening]
