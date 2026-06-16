"""Start the long-running Tushare report_rc refresh as a detached process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / ".run"
LOG_PATH = RUN_DIR / "tushare_report_rc_refresh.log"
PID_PATH = RUN_DIR / "tushare_report_rc_refresh.pid"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts/research/refresh_tushare_financials.py"),
        "--dataset",
        "report_rc",
        "--start",
        "20130101",
        "--sleep",
        "1",
        "--retries",
        "3",
        "--page-size",
        "3000",
    ]
    log_file = LOG_PATH.open("ab")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(process.pid)


if __name__ == "__main__":
    main()
