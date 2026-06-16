"""Start the long-running Datayes analyst consensus refresh."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / ".run"
LOG_PATH = RUN_DIR / "datayes_consensus_refresh.log"
PID_PATH = RUN_DIR / "datayes_consensus_refresh.pid"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/research/refresh_analyst_forecasts.py"),
        "--source",
        "datayes_consensus",
        "--batch-size",
        "60",
        "--concurrency",
        "8",
        "--sleep",
        "0.5",
        "--retries",
        "1",
        "--timeout",
        "180",
    ]
    env = os.environ.copy()
    env.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
    log_file = LOG_PATH.open("ab")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(process.pid)


if __name__ == "__main__":
    main()
