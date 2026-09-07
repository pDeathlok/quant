"""Execute real frontend loaders with controlled, out-of-order Node responses."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_request_races() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend request-race regression tests")
    result = subprocess.run(
        [node, "--test", "tests/frontend/workspace_race.test.js"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
