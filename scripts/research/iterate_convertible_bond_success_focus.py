from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.strategies.convertible_bond.backtest import (  # noqa: E402
    ConvertibleBondBacktestConfig,
    write_backtest_outputs,
)
from quant.strategies.convertible_bond.grid import (  # noqa: E402
    HoldingGridConfig,
    backtest_holding_grid,
)

DATA_DIR = PROJECT_ROOT / "data/convertible_bond/tushare"
DAILY_PATH = DATA_DIR / "cb_daily_20180101_20260616.parquet"
BASIC_PATH = DATA_DIR / "cb_basic_all.parquet"
CALL_PATH = DATA_DIR / "cb_call_20180101_20260616.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "reports/convertible_bond/success_focus_iteration"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def success_configs() -> list[HoldingGridConfig]:
    return [
        HoldingGridConfig(
            name="success_strict_block",
            top_n=6,
            max_holdings=10,
            max_total_weight=0.75,
            max_position_weight=0.14,
            max_entry_price=112.0,
            min_premium_rate=0.0,
            max_premium_rate=18.0,
            max_double_low=130.0,
            max_price_position_252=0.24,
            min_drawdown_from_252_high=0.10,
            min_amount=5_000.0,
            min_momentum_20d=-0.06,
            market_risk_mode="block_downtrend",
            exit_price=124.0,
            exit_premium_rate=35.0,
            exit_double_low=150.0,
            exit_price_position_252=0.68,
            min_credit_rating="AA-",
        ),
        HoldingGridConfig(
            name="success_balanced_block",
            top_n=8,
            max_holdings=12,
            max_total_weight=0.85,
            max_position_weight=0.15,
            max_entry_price=114.0,
            min_premium_rate=-1.0,
            max_premium_rate=22.0,
            max_double_low=134.0,
            max_price_position_252=0.30,
            min_drawdown_from_252_high=0.08,
            min_amount=4_000.0,
            min_momentum_20d=-0.08,
            market_risk_mode="block_downtrend",
            exit_price=128.0,
            exit_premium_rate=40.0,
            exit_double_low=158.0,
            exit_price_position_252=0.76,
            min_credit_rating="AA-",
        ),
        HoldingGridConfig(
            name="success_balanced_scaled",
            top_n=8,
            max_holdings=12,
            max_total_weight=0.90,
            max_position_weight=0.15,
            max_entry_price=115.0,
            min_premium_rate=-2.0,
            max_premium_rate=23.0,
            max_double_low=136.0,
            max_price_position_252=0.32,
            min_drawdown_from_252_high=0.08,
            min_amount=3_000.0,
            min_momentum_20d=-0.10,
            market_risk_mode="scale_downtrend",
            market_entry_scale_weak=0.25,
            market_entry_scale_strong=1.0,
            exit_price=130.0,
            exit_premium_rate=42.0,
            exit_double_low=162.0,
            exit_price_position_252=0.78,
            min_credit_rating="AA-",
        ),
        HoldingGridConfig(
            name="success_deep_value",
            top_n=5,
            max_holdings=8,
            max_total_weight=0.65,
            max_position_weight=0.13,
            max_entry_price=110.0,
            min_premium_rate=0.0,
            max_premium_rate=16.0,
            max_double_low=126.0,
            max_price_position_252=0.18,
            min_drawdown_from_252_high=0.12,
            min_amount=5_000.0,
            min_momentum_20d=-0.05,
            market_risk_mode="scale_downtrend",
            market_entry_scale_weak=0.20,
            market_entry_scale_strong=1.0,
            exit_price=122.0,
            exit_premium_rate=32.0,
            exit_double_low=146.0,
            exit_price_position_252=0.64,
            min_credit_rating="AA",
        ),
    ]


def run_variant(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame,
    grid_config: HoldingGridConfig,
    start_date: str,
    end_date: str,
    rebalance: str,
) -> dict:
    backtest_config = ConvertibleBondBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance=rebalance,
        commission_rate=0.0002,
        slippage_rate=0.0002,
        initial_cash=1_000_000.0,
    )
    result = backtest_holding_grid(
        daily=daily,
        basic=basic,
        call=call,
        backtest_config=backtest_config,
        grid_config=grid_config,
    )
    out_dir = OUTPUT_ROOT / f"{grid_config.name}_{start_date}_{end_date}_{rebalance}"
    paths = write_backtest_outputs(result, out_dir)
    if result.position_trades is not None and not result.position_trades.empty:
        closed = result.position_trades[result.position_trades["status"] == "closed"].copy()
        cases = pd.concat(
            [
                closed.sort_values("return").head(20).assign(case_type="worst_closed"),
                closed.sort_values("return", ascending=False).head(20).assign(case_type="best_closed"),
            ],
            ignore_index=True,
        )
    else:
        cases = pd.DataFrame()
    case_path = out_dir / "position_cases.csv"
    cases.to_csv(case_path, index=False)
    summary = dict(result.summary)
    summary.update(
        {
            "variant": grid_config.name,
            "rebalance": rebalance,
            "summary_path": str(paths["summary"]),
            "position_case_path": str(case_path),
        }
    )
    return summary


def main() -> None:
    daily = pd.read_parquet(DAILY_PATH)
    basic = pd.read_parquet(BASIC_PATH)
    call = pd.read_parquet(CALL_PATH) if CALL_PATH.exists() else pd.DataFrame()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = [
        (config, rebalance, start_date)
        for config in success_configs()
        for rebalance in ["weekly", "monthly"]
        for start_date in ["20200101", "20240101"]
    ]
    summaries: list[dict] = []
    for index, (config, rebalance, start_date) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {config.name} {rebalance} {start_date}", flush=True)
        summaries.append(
            run_variant(
                daily=daily,
                basic=basic,
                call=call,
                grid_config=config,
                start_date=start_date,
                end_date="20260616",
                rebalance=rebalance,
            )
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = OUTPUT_ROOT / "iteration_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    ranked = summary_frame.sort_values(
        [
            "start_date",
            "position_win_rate",
            "position_profit_factor",
            "average_position_return",
            "max_drawdown",
        ],
        ascending=[True, False, False, False, False],
    )
    columns = [
        "variant",
        "rebalance",
        "start_date",
        "total_return",
        "annual_return",
        "max_drawdown",
        "closed_position_trades",
        "position_win_rate",
        "position_profit_factor",
        "average_position_return",
        "median_position_return",
        "average_holding_days",
        "average_exposure",
        "invested_days",
        "trade_count",
    ]
    print(
        json.dumps(
            json_safe(
                {
                "summary_path": str(summary_path),
                "top": ranked[columns].head(12).to_dict(orient="records"),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
