from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.strategies.convertible_bond import (  # noqa: E402
    ConvertibleBondFilterConfig,
    ConvertibleBondTrendEnhancedBacktestConfig,
    ConvertibleBondTrendEnhancedConfig,
    add_trend_enhanced_features,
    backtest_convertible_bond_trend_enhanced,
)
from quant.strategies.convertible_bond.backtest import write_backtest_outputs  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data/convertible_bond/tushare"
DAILY_PATH = DATA_DIR / "cb_daily_20180101_20260616.parquet"
BASIC_PATH = DATA_DIR / "cb_basic_all.parquet"
CALL_PATH = DATA_DIR / "cb_call_20180101_20260616.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "reports/convertible_bond/trend_enhanced_iteration"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def strategy_configs() -> list[ConvertibleBondTrendEnhancedConfig]:
    base_filter = ConvertibleBondFilterConfig(
        min_price=100.0,
        max_price=160.0,
        max_premium_rate=45.0,
        min_amount=1_000.0,
        min_remaining_size=1.0,
        min_credit_rating="AA-",
        exclude_call_risk=True,
        exclude_not_convertible=True,
    )
    return [
        ConvertibleBondTrendEnhancedConfig(
            top_n=20,
            max_position_weight=0.05,
            min_trend_strength=75.0,
            min_5d_return=3.0,
            max_5d_return=20.0,
            min_1d_return=0.0,
            max_1d_return=7.0,
            min_six_sword=4,
            min_consecutive_six_sword=1,
            require_band_buy=True,
            max_price_position_60d=0.92,
            max_market_median_double_low=155.0,
            min_market_trend_20d=-0.04,
            min_market_trend_breadth=0.10,
            filter=base_filter,
        ),
        ConvertibleBondTrendEnhancedConfig(
            top_n=15,
            max_position_weight=0.06,
            min_trend_strength=75.0,
            min_5d_return=2.0,
            max_5d_return=18.0,
            min_1d_return=-1.0,
            max_1d_return=6.5,
            min_six_sword=4,
            min_consecutive_six_sword=2,
            require_band_buy=True,
            max_price_position_60d=0.90,
            max_market_median_double_low=150.0,
            min_market_trend_20d=-0.03,
            min_market_trend_breadth=0.12,
            filter=base_filter,
        ),
        ConvertibleBondTrendEnhancedConfig(
            top_n=12,
            max_position_weight=0.07,
            min_trend_strength=100.0,
            min_5d_return=2.0,
            max_5d_return=16.0,
            min_1d_return=-0.5,
            max_1d_return=5.5,
            min_six_sword=5,
            min_consecutive_six_sword=2,
            require_band_buy=True,
            max_price_position_60d=0.88,
            max_market_median_double_low=148.0,
            min_market_trend_20d=-0.025,
            min_market_trend_breadth=0.15,
            filter=replace(base_filter, max_price=150.0, max_premium_rate=38.0, min_amount=3_000.0),
        ),
        ConvertibleBondTrendEnhancedConfig(
            top_n=20,
            max_position_weight=0.05,
            min_trend_strength=75.0,
            min_5d_return=1.0,
            max_5d_return=15.0,
            min_1d_return=-1.5,
            max_1d_return=5.0,
            min_six_sword=4,
            min_consecutive_six_sword=1,
            require_band_buy=False,
            max_price_position_60d=0.86,
            max_market_median_double_low=145.0,
            min_market_trend_20d=-0.02,
            min_market_trend_breadth=0.15,
            filter=replace(base_filter, max_price=145.0, max_premium_rate=35.0, min_amount=3_000.0),
        ),
        ConvertibleBondTrendEnhancedConfig(
            top_n=8,
            max_position_weight=0.10,
            min_trend_strength=100.0,
            min_5d_return=3.0,
            max_5d_return=14.0,
            min_1d_return=-0.5,
            max_1d_return=4.5,
            min_six_sword=5,
            min_consecutive_six_sword=3,
            require_band_buy=True,
            max_price_position_60d=0.84,
            max_market_median_double_low=142.0,
            min_market_trend_20d=-0.015,
            min_market_trend_breadth=0.18,
            filter=replace(base_filter, max_price=140.0, max_premium_rate=30.0, min_amount=5_000.0),
        ),
        ConvertibleBondTrendEnhancedConfig(
            top_n=12,
            max_position_weight=0.06,
            min_trend_strength=75.0,
            min_5d_return=0.5,
            max_5d_return=10.0,
            min_1d_return=-2.0,
            max_1d_return=3.5,
            min_six_sword=4,
            min_consecutive_six_sword=1,
            require_band_buy=False,
            max_price_position_60d=0.78,
            max_market_median_double_low=138.0,
            min_market_trend_20d=-0.01,
            min_market_trend_breadth=0.20,
            filter=replace(base_filter, max_price=135.0, max_premium_rate=28.0, min_amount=5_000.0),
        ),
    ]


def run_variant(
    daily: pd.DataFrame,
    basic: pd.DataFrame,
    call: pd.DataFrame,
    selector_config: ConvertibleBondTrendEnhancedConfig,
    variant: str,
    start_date: str,
    rebalance: str,
) -> dict:
    backtest_config = ConvertibleBondTrendEnhancedBacktestConfig(
        start_date=start_date,
        end_date="20260616",
        rebalance=rebalance,
        commission_rate=0.0002,
        slippage_rate=0.0002,
        initial_cash=1_000_000.0,
        selector=selector_config,
    )
    result = backtest_convertible_bond_trend_enhanced(
        daily=daily,
        basic=basic,
        call=call,
        config=backtest_config,
    )
    out_dir = OUTPUT_ROOT / f"{variant}_{start_date}_{rebalance}"
    paths = write_backtest_outputs(result, out_dir)

    worst_days = result.equity.sort_values("daily_return").head(30).copy()
    worst_day_path = out_dir / "worst_days.csv"
    worst_days.to_csv(worst_day_path, index=False)

    if not result.targets.empty and not worst_days.empty:
        worst_dates = set(worst_days["trade_date"].astype(str))
        target_cases = result.targets[result.targets["trade_date"].astype(str).isin(worst_dates)].copy()
        target_cases = target_cases.sort_values(
            ["trade_date", "score"], ascending=[True, False]
        ).head(300)
    else:
        target_cases = pd.DataFrame()
    target_case_path = out_dir / "worst_day_targets.csv"
    target_cases.to_csv(target_case_path, index=False)

    summary = dict(result.summary)
    summary.update(
        {
            "variant": variant,
            "rebalance": rebalance,
            "summary_path": str(paths["summary"]),
            "worst_day_path": str(worst_day_path),
            "target_case_path": str(target_case_path),
            "selector_config": asdict(selector_config),
        }
    )
    return summary


def main() -> None:
    daily = add_trend_enhanced_features(pd.read_parquet(DAILY_PATH))
    basic = pd.read_parquet(BASIC_PATH)
    call = pd.read_parquet(CALL_PATH) if CALL_PATH.exists() else pd.DataFrame()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    jobs = [
        (f"trend_v{index}", config, rebalance, start_date)
        for index, config in enumerate(strategy_configs(), start=1)
        for rebalance in ["daily", "weekly"]
        for start_date in ["20180101", "20200101", "20240101"]
    ]
    for index, (variant, config, rebalance, start_date) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {variant} {rebalance} {start_date}", flush=True)
        summaries.append(
            run_variant(
                daily=daily,
                basic=basic,
                call=call,
                selector_config=config,
                variant=variant,
                start_date=start_date,
                rebalance=rebalance,
            )
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = OUTPUT_ROOT / "iteration_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    ranked = summary_frame.sort_values(
        ["start_date", "sharpe", "annual_return", "max_drawdown", "average_turnover"],
        ascending=[True, False, False, False, True],
    )
    columns = [
        "variant",
        "rebalance",
        "start_date",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "average_turnover",
        "trade_count",
        "summary_path",
        "worst_day_path",
        "target_case_path",
    ]
    print(
        json.dumps(
            json_safe(
                {
                    "summary_path": str(summary_path),
                    "top": ranked[columns].head(15).to_dict(orient="records"),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
