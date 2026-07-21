from __future__ import annotations

import json

from quant.routine import cli


def test_cache_cleanup_command_runs_retention_immediately(monkeypatch, capsys) -> None:
    expected = {"status": "success", "reclaimed_bytes": 789, "errors": []}
    monkeypatch.setattr(cli, "run_cache_cleanup", lambda project_root: expected)
    monkeypatch.setattr("sys.argv", ["quant-routine", "cache-cleanup"])

    cli.main()

    assert json.loads(capsys.readouterr().out) == expected
