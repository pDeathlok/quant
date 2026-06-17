from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
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
OUTPUT_ROOT = PROJECT_ROOT / "reports/convertible_bond/grid_trend_overlay"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def base_configs() -> list[HoldingGridConfig]:
    return [
        HoldingGridConfig(
            name="core_market_scaled",
            top_n=8,
            max_holdings=12,
            max_total_weight=0.95,
            max_position_weight=0.18,
            max_entry_price=116.0,
            min_premium_rate=0.0,
            max_premium_rate=24.0,
            max_double_low=138.0,
            max_price_position_252=0.35,
            min_drawdown_from_252_high=0.07,
            min_amount=3_000.0,
            min_momentum_20d=-0.08,
            market_risk_mode="scale_downtrend",
            market_entry_scale_weak=0.35,
            market_entry_scale_strong=1.0,
            exit_price=132.0,
            exit_premium_rate=45.0,
            exit_double_low=165.0,
            exit_price_position_252=0.82,
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
            name="return_core",
            top_n=8,
            max_holdings=14,
            max_total_weight=1.00,
            max_position_weight=0.18,
            max_entry_price=118.0,
            min_premium_rate=-1.0,
            max_premium_rate=26.0,
            max_double_low=142.0,
            max_price_position_252=0.38,
            min_drawdown_from_252_high=0.06,
            min_amount=3_000.0,
            min_momentum_20d=-0.10,
            exit_price=140.0,
            exit_premium_rate=55.0,
            exit_double_low=178.0,
            exit_price_position_252=0.90,
        ),
    ]


def overlay_configs(config: HoldingGridConfig) -> list[HoldingGridConfig]:
    return [
        config,
        replace(
            config,
            name=f"{config.name}_trend_confirm",
            min_entry_trend_strength=75.0,
            min_entry_six_sword=4,
            min_entry_consecutive_six_sword=1,
            min_entry_return_5d=-1.0,
            max_entry_return_5d=8.0,
            min_entry_return_1d=-2.0,
            max_entry_return_1d=4.0,
            max_entry_price_position_60d=0.88,
            max_entry_market_median_double_low=150.0,
            min_entry_market_trend_20d=-0.03,
            min_entry_market_trend_breadth=0.12,
        ),
        replace(
            config,
            name=f"{config.name}_trend_rebound",
            min_entry_trend_strength=50.0,
            min_entry_six_sword=3,
            min_entry_return_5d=-4.0,
            max_entry_return_5d=10.0,
            min_entry_return_1d=-3.0,
            max_entry_return_1d=5.0,
            max_entry_price_position_60d=0.82,
            max_entry_market_median_double_low=145.0,
            min_entry_market_trend_20d=-0.04,
            min_entry_market_trend_breadth=0.10,
        ),
        replace(
            config,
            name=f"{config.name}_market_gate",
            max_entry_market_median_double_low=145.0,
            min_entry_market_trend_20d=-0.02,
            min_entry_market_trend_breadth=0.15,
        ),
    ]


def run_variant(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame,
    grid_config: HoldingGridConfig,
    start_date: str,
    rebalance: str,
) -> dict:
    backtest_config = ConvertibleBondBacktestConfig(
        start_date=start_date,
        end_date="20260616",
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
    out_dir = OUTPUT_ROOT / f"{grid_config.name}_{start_date}_{rebalance}"
    paths = write_backtest_outputs(result, out_dir)
    summary = dict(result.summary)
    summary.update(
        {
            "variant": grid_config.name,
            "base_variant": grid_config.name.split("_trend_")[0].replace("_market_gate", ""),
            "rebalance": rebalance,
            "summary_path": str(paths["summary"]),
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
        for base in base_configs()
        for config in overlay_configs(base)
        for rebalance in ["weekly", "monthly"]
        for start_date in ["20180101", "20200101", "20240101"]
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
                rebalance=rebalance,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_path = OUTPUT_ROOT / "iteration_summary.csv"
    summary.to_csv(summary_path, index=False)

    baseline = summary[~summary["variant"].str.contains("trend_|market_gate", regex=True)].copy()
    baseline = baseline.rename(
        columns={
            "total_return": "baseline_total_return",
            "annual_return": "baseline_annual_return",
            "max_drawdown": "baseline_max_drawdown",
            "sharpe": "baseline_sharpe",
        }
    )
    compare = summary.merge(
        baseline[
            [
                "base_variant",
                "rebalance",
                "start_date",
                "baseline_total_return",
                "baseline_annual_return",
                "baseline_max_drawdown",
                "baseline_sharpe",
            ]
        ],
        on=["base_variant", "rebalance", "start_date"],
        how="left",
    )
    compare["delta_total_return"] = compare["total_return"] - compare["baseline_total_return"]
    compare["delta_annual_return"] = compare["annual_return"] - compare["baseline_annual_return"]
    compare["delta_max_drawdown"] = compare["max_drawdown"] - compare["baseline_max_drawdown"]
    compare["delta_sharpe"] = compare["sharpe"] - compare["baseline_sharpe"]
    compare_path = OUTPUT_ROOT / "paired_comparison.csv"
    compare.to_csv(compare_path, index=False)

    columns = [
        "variant",
        "rebalance",
        "start_date",
        "total_return",
        "annual_return",
        "max_drawdown",
        "sharpe",
        "delta_total_return",
        "delta_max_drawdown",
        "delta_sharpe",
        "average_exposure",
        "invested_days",
        "trade_count",
    ]
    top = compare[compare["variant"].ne(compare["base_variant"])].sort_values(
        ["delta_total_return", "delta_sharpe", "delta_max_drawdown"],
        ascending=[False, False, False],
    )[columns].head(16)
    print(
        json.dumps(
            json_safe(
                {
                    "summary_path": str(summary_path),
                    "comparison_path": str(compare_path),
                    "top_improvements": top.to_dict(orient="records"),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
