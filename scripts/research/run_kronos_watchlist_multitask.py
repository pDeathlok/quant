"""Run isolated watchlist Kronos ablations; no production writes."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant.research.kronos_watchlist_multitask import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kronos_watchlist_multitask.yaml"))
    args = parser.parse_args()
    print(run(args.config), flush=True)


if __name__ == "__main__":
    main()
