#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant.routine.web_refresh_runner import main


def daily_refresh_args(argv: list[str]) -> list[str]:
    restart_flags = {"--restart-service", "--no-restart-service"}
    if any(arg in restart_flags for arg in argv):
        return list(argv)
    return ["--restart-service", *argv]


if __name__ == "__main__":
    raise SystemExit(main(daily_refresh_args(sys.argv[1:])))
