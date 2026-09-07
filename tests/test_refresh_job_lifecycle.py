from datetime import datetime
import threading

import pytest

from quant.routine import web_refresh_runner as runner
from quant.webapp import services


@pytest.fixture
def job_state(monkeypatch):
    state = {
        "run_id": "owned-run",
        "status": "running",
        "scope": "all",
        "started_at": "2000-01-01T00:00:00",
        "updated_at": "2000-01-01T00:00:00",
        "steps": [],
    }
    monkeypatch.setattr(services, "_REFRESH_STATUS", state)
    monkeypatch.setattr(services, "_REFRESH_JOB_THREADS", {})
    monkeypatch.setattr(services, "_persist_refresh_status_unlocked", lambda: None)
    return state


def test_silent_owned_job_is_not_expired_or_killed(monkeypatch, job_state):
    services._REFRESH_JOB_THREADS["owned-run"] = threading.current_thread()
    monkeypatch.setattr(
        services, "_terminate_active_worker_unlocked",
        lambda *args: pytest.fail("An active owner must not be killed by a status read"),
    )

    status = services.get_latest_refresh_status()

    assert status["status"] == "running"
    assert status["job_active"] is True
    assert "runtime budget" in status["execution_warning"]
    assert services._expire_stale_refresh_status_unlocked(job_state) is job_state


@pytest.mark.parametrize("status", ["running", "queued", "success", "failed"])
def test_retry_waits_for_owner_even_after_terminal_status(monkeypatch, job_state, status):
    job_state["status"] = status
    services._REFRESH_JOB_THREADS["owned-run"] = threading.current_thread()
    monkeypatch.setattr(
        services.threading, "Thread",
        lambda **kwargs: pytest.fail("A new job must not overlap the owner"),
    )

    result = services.start_latest_refresh("all")

    assert result["run_id"] == "owned-run"
    assert result["job_active"] is True


def test_queued_job_is_not_submitted_twice(monkeypatch, job_state):
    job_state.update(status="queued", started_at=datetime.now().isoformat())
    job_state["updated_at"] = job_state["started_at"]
    monkeypatch.setattr(
        services.threading, "Thread",
        lambda **kwargs: pytest.fail("Queued owner has not entered its wrapper yet"),
    )

    assert services.start_latest_refresh("all")["run_id"] == "owned-run"


def test_dead_owner_does_not_suppress_stale_detection(job_state):
    services._REFRESH_JOB_THREADS["owned-run"] = threading.Thread()
    job_state["heartbeat_at"] = datetime.now().isoformat()

    assert services._is_refresh_status_stale(job_state)


def test_heartbeat_does_not_fabricate_step_progress(job_state):
    services._REFRESH_JOB_THREADS["owned-run"] = threading.current_thread()

    class TwoTicks:
        def __init__(self):
            self.ticks = 0

        def wait(self, _seconds):
            self.ticks += 1
            return self.ticks > 1

    services._refresh_job_heartbeat("owned-run", TwoTicks())

    assert job_state["heartbeat_at"]
    assert job_state["updated_at"] == "2000-01-01T00:00:00"
    assert job_state["steps"] == []
    previous = dict(job_state)
    services._refresh_job_heartbeat("old-run", TwoTicks())
    assert job_state == previous


@pytest.mark.parametrize("fail", [False, True])
def test_owner_is_released_only_after_owned_work_returns(monkeypatch, job_state, fail):
    def work(*args):
        assert services._refresh_job_active_unlocked(job_state)
        job_state["status"] = "failed" if fail else "success"
        assert services.get_latest_refresh_status()["job_active"]
        if fail:
            raise RuntimeError("test failure")

    monkeypatch.setattr(services, "_run_latest_refresh_job_owned", work)
    if fail:
        with pytest.raises(RuntimeError, match="test failure"):
            services._run_latest_refresh_job(run_id="owned-run")
    else:
        services._run_latest_refresh_job(run_id="owned-run")

    assert not services._REFRESH_JOB_THREADS
    assert not services.get_latest_refresh_status()["job_active"]


def test_runner_waits_for_terminal_owner_to_release():
    statuses = iter([
        {"status": "failed", "job_active": True, "heartbeat_at": "first"},
        {"status": "failed", "job_active": True, "heartbeat_at": "second"},
        {"status": "failed", "job_active": False},
    ])

    class Client:
        def get_status(self):
            return next(statuses)

    sleeps = []
    terminal = runner.wait_for_terminal_status(
        Client(), runner.RefreshRunnerConfig(), sleep_fn=sleeps.append,
        print_fn=lambda _: None,
    )

    assert not terminal["job_active"]
    assert len(sleeps) == 2


def test_runner_liveness_signature_includes_heartbeat():
    status = {"status": "running", "updated_at": "fixed", "heartbeat_at": "one"}
    assert runner._status_signature(status) != runner._status_signature(
        {**status, "heartbeat_at": "two"}
    )


def test_runner_does_not_restart_or_resubmit_terminal_but_active_job(monkeypatch, tmp_path):
    statuses = iter([
        {"status": "failed", "job_active": True},
        {"status": "failed", "job_active": True},
        {"status": "failed", "job_active": True, "heartbeat_at": "waiting"},
        {"status": "success", "job_active": False},
    ])

    class Client:
        def get_status(self):
            return next(statuses)

        def start_refresh(self, _scope):
            pytest.fail("The original job still owns its publication")

    restarts = []
    monkeypatch.setattr(runner, "RefreshApiClient", lambda *args, **kwargs: Client())
    monkeypatch.setattr(
        runner, "decide_trade_day",
        lambda **kwargs: runner.TradeDayDecision(True, "open", "20260907"),
    )
    monkeypatch.setattr(
        runner, "ensure_local_service",
        lambda **kwargs: restarts.append(kwargs["force_restart"]),
    )

    result = runner.run_refresh_workflow(
        runner.RefreshRunnerConfig(
            project_root=tmp_path, env_path=tmp_path / "absent.env", restart_service=True,
        ),
        sleep_fn=lambda _: None, print_fn=lambda _: None,
    )

    assert result["status"] == "success"
    assert restarts == [False]
