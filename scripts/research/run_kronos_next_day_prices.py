"""Run the daily-sampled, OHLC-only next-session Kronos experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant.research.kronos_next_day_prices import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kronos_next_day_prices.yaml"))
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
