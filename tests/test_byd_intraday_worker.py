from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant.routine import byd_intraday_worker as worker


@pytest.mark.parametrize("mode", ["success", "error", "timeout", "eof"])
def test_worker_cleanup_and_failure_propagation(monkeypatch, mode: str) -> None:
    events = []
    class Pipe:
        def close(self):
            events.append("close")
        def poll(self, seconds):
            assert seconds == 0.1
            return mode != "timeout"
        def recv(self):
            if mode == "eof":
                raise EOFError
            return {"ok": mode == "success", "result": {"status": "success"}, "error": "provider failure"}
    class Process:
        alive = True
        def start(self):
            events.append("start")
        def join(self, timeout):
            events.append("join")
        def is_alive(self):
            return self.alive
        def terminate(self):
            events.append("terminate")
            self.alive = False
        def kill(self):
            events.append("kill")
            self.alive = False
    context = SimpleNamespace(Pipe=lambda **kwargs: (Pipe(), Pipe()), Process=lambda **kwargs: Process())
    monkeypatch.setattr(worker.mp, "get_context", lambda method: context)
    if mode == "success":
        assert worker.refresh_byd_training_isolated(expected_trade_date="2026-09-15", timeout_seconds=0.1) == {"status": "success"}
    else:
        with pytest.raises(TimeoutError if mode == "timeout" else RuntimeError):
            worker.refresh_byd_training_isolated(expected_trade_date="2026-09-15", timeout_seconds=0.1)
    assert events.count("close") == 2
    assert "terminate" in events


def test_byd_workspace_reader_never_starts_network_refresh(monkeypatch) -> None:
    from quant.webapp import services

    monkeypatch.setattr(worker, "refresh_byd_training_isolated", lambda **kwargs: pytest.fail("page read starts refresh"))
    monkeypatch.setattr(services, "build_byd_daily_strategy", lambda **kwargs: {"cached": True})
    assert services.get_byd_daily_strategy(refresh=False) == {"cached": True}


def test_byd_workspace_refresh_repairs_training_before_build(monkeypatch) -> None:
    from quant.webapp import services

    calls = []
    def refresh(**kwargs):
        assert kwargs["expected_trade_date"] == "2026-09-15"
        calls.append("refresh")
        return {"status": "success", "provider": "tushare"}
    def build(**kwargs):
        calls.append("build")
        return {"planned_t": {"signal_date": "2026-09-15"}}
    monkeypatch.setattr(worker, "refresh_byd_training_isolated", refresh)
    monkeypatch.setattr(services, "_local_market_trade_date", lambda: "20260915")
    monkeypatch.setattr(services, "build_byd_daily_strategy", build)
    result = services.get_byd_daily_strategy(refresh=True)
    assert calls == ["refresh", "build"]
    assert result["intraday_training_refresh"]["provider"] == "tushare"
