from __future__ import annotations

import json

from quant.routine import cli


def test_cache_cleanup_command_runs_retention_immediately(monkeypatch, capsys) -> None:
    expected = {"status": "success", "reclaimed_bytes": 789, "errors": []}
    monkeypatch.setattr(cli, "run_cache_cleanup", lambda project_root: expected)
    monkeypatch.setattr("sys.argv", ["quant-routine", "cache-cleanup"])

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected


def test_daily_refresh_data_delegates_to_canonical_web_workflow(monkeypatch, capsys) -> None:
    captured = {}

    def fake_web_refresh(config):
        captured["config"] = config
        return {"status": "success", "attempts": 1}

    monkeypatch.setattr(cli, "run_refresh_workflow", fake_web_refresh)
    monkeypatch.setattr(
        cli,
        "run_daily_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy pipeline called")),
    )

    exit_code = cli.main(["daily", "--refresh-data", "--skip-backtest"])

    assert exit_code == 0
    assert captured["config"].restart_service is False
    assert json.loads(capsys.readouterr().out)["status"] == "success"


def test_direct_pipeline_requires_explicit_flag(monkeypatch, capsys) -> None:
    captured = {}
    monkeypatch.setattr(
        cli,
        "run_daily_pipeline",
        lambda **kwargs: captured.update(kwargs) or {"status": "success"},
    )

    exit_code = cli.main(
        ["daily", "--refresh-data", "--skip-backtest", "--direct-pipeline"]
    )

    assert exit_code == 0
    assert captured == {"skip_data": False, "skip_backtest": True}
    assert json.loads(capsys.readouterr().out)["status"] == "success"
