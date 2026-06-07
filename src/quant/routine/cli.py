from __future__ import annotations

import argparse
import json

from quant.routine.dashboard import write_dashboard_json
from quant.routine.b1_daily_plan import write_daily_plan
from quant.routine.pipeline import run_daily_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routine B1 strategy operations.")
    parser.add_argument(
        "command",
        choices=["daily", "dashboard", "plan"],
        help="daily runs the routine pipeline; dashboard regenerates dashboard.json; plan generates next-day B1 plan",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Actually run the data refresh script.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip formal combo backtest and only refresh dashboard.")
    args = parser.parse_args()

    if args.command == "dashboard":
        output = write_dashboard_json()
        print(json.dumps({"status": "success", "output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "plan":
        output = write_daily_plan()
        print(json.dumps({"status": "success", "output": str(output)}, ensure_ascii=False, indent=2))
        return

    result = run_daily_pipeline(skip_data=not args.refresh_data, skip_backtest=args.skip_backtest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
