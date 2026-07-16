from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant.routine.dashboard import write_dashboard_json
from quant.routine.b1_daily_plan import write_daily_plan
from quant.routine.convertible_bond_plan import write_convertible_bond_plan
from quant.routine.pipeline import run_daily_pipeline
from quant.routine.web_refresh_runner import (
    RefreshRunnerConfig,
    run_refresh_workflow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routine B1 strategy operations.")
    parser.add_argument(
        "command",
        choices=["daily", "dashboard", "plan", "cb-plan", "web-refresh"],
        help="daily runs the routine pipeline; dashboard regenerates dashboard.json; plan generates next-day B1 plan; cb-plan generates convertible-bond plan; web-refresh orchestrates the page-equivalent latest-data refresh",
    )
    parser.add_argument("--trade-date", help="Trade date in YYYYMMDD format for cb-plan.")
    parser.add_argument("--refresh-data", action="store_true", help="Actually run the data refresh script.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip formal combo backtest and only refresh dashboard.")
    parser.add_argument("--env-file", default=str(PROJECT_ROOT / ".env"), help="Env file for web-refresh.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8088/api", help="Local web API base URL for web-refresh.")
    parser.add_argument("--frontend-url", default=None, help="Local frontend URL for web-refresh health checks.")
    parser.add_argument("--scope", default="all", help="Refresh scope for web-refresh.")
    parser.add_argument("--poll-seconds", type=float, default=20.0, help="Status poll interval for web-refresh.")
    parser.add_argument("--health-timeout", type=float, default=60.0, help="Service startup timeout for web-refresh.")
    parser.add_argument("--no-progress-timeout", type=float, default=35 * 60.0, help="No-progress timeout for web-refresh.")
    parser.add_argument("--retry-delay", type=float, default=5.0, help="Retry delay for web-refresh.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max trigger attempts for web-refresh.")
    parser.add_argument(
        "--log-file",
        default=str(PROJECT_ROOT / ".run" / "daily_web_refresh.log"),
        help="Service log path for web-refresh.",
    )
    args = parser.parse_args()

    if args.command == "dashboard":
        output = write_dashboard_json()
        print(json.dumps({"status": "success", "output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "plan":
        output = write_daily_plan()
        print(json.dumps({"status": "success", "output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "cb-plan":
        if not args.trade_date:
            raise SystemExit("--trade-date is required for cb-plan")
        output = write_convertible_bond_plan(trade_date=args.trade_date)
        print(json.dumps({"status": "success", "output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "web-refresh":
        result = run_refresh_workflow(
            RefreshRunnerConfig(
                env_path=Path(args.env_file).expanduser().resolve(),
                base_url=args.base_url,
                frontend_url=args.frontend_url,
                scope=args.scope,
                poll_seconds=args.poll_seconds,
                health_timeout_seconds=args.health_timeout,
                no_progress_timeout_seconds=args.no_progress_timeout,
                retry_delay_seconds=args.retry_delay,
                max_attempts=args.max_attempts,
                service_log_path=Path(args.log_file).expanduser().resolve(),
            )
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = run_daily_pipeline(skip_data=not args.refresh_data, skip_backtest=args.skip_backtest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
