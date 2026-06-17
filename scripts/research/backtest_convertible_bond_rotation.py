from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.routine.convertible_bond_plan import load_convertible_bond_config
from quant.strategies.convertible_bond.backtest import (
    ConvertibleBondBacktestConfig,
    backtest_convertible_bond_rotation,
    collect_convertible_bond_history,
    load_convertible_bond_history,
    write_backtest_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Tushare convertible-bond history and backtest double-low rotation.")
    parser.add_argument("--start-date", default="20180101", help="Backtest start date, YYYYMMDD.")
    parser.add_argument("--end-date", default="20260616", help="Backtest end date, YYYYMMDD.")
    parser.add_argument("--data-dir", default="data/convertible_bond/tushare", help="Local history cache directory.")
    parser.add_argument("--output-dir", default="reports/convertible_bond/rotation", help="Backtest output directory.")
    parser.add_argument("--config", default="configs/strategies/convertible_bond_rotation.yaml", help="Strategy config YAML.")
    parser.add_argument("--refresh", action="store_true", help="Fetch or rebuild Tushare history before backtesting.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore existing local history files.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between Tushare daily requests.")
    parser.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="daily")
    parser.add_argument("--commission-rate", type=float, default=0.0002)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir / f"{args.start_date}_{args.end_date}_{args.rebalance}"
    selector_config = load_convertible_bond_config(PROJECT_ROOT / args.config)

    if args.refresh:
        collect_convertible_bond_history(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=data_dir,
            sleep_seconds=args.sleep,
            force=args.force_refresh,
        )

    daily, basic, call = load_convertible_bond_history(
        data_dir=data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    config = ConvertibleBondBacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        rebalance=args.rebalance,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        initial_cash=args.initial_cash,
        selector=selector_config,
    )
    result = backtest_convertible_bond_rotation(daily=daily, basic=basic, call=call, config=config)
    paths = write_backtest_outputs(result, output_dir)
    payload = {"summary": result.summary, "paths": {key: str(value) for key, value in paths.items()}}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    yaml_path = output_dir / "run_config.yaml"
    yaml_path.write_text(yaml.safe_dump(payload["summary"]["config"], allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
