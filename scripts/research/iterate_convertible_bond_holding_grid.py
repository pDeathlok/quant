from __future__ import annotations

import json
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
OUTPUT_ROOT = PROJECT_ROOT / "reports/convertible_bond/holding_grid_iteration"


def build_case_analysis(equity: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    frame = equity.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    recent = frame[frame["trade_date"] >= "20240101"].copy()
    if recent.empty:
        recent = frame.tail(500).copy()
    worst_days = recent.sort_values("daily_return").head(20)
    rows = []
    for row in worst_days.to_dict(orient="records"):
        trade_date = str(row["trade_date"])
        day_targets = (
            targets[targets["trade_date"].astype(str) == trade_date]
            if not targets.empty
            else pd.DataFrame()
        )
        rows.append(
            {
                "trade_date": trade_date,
                "daily_return": row["daily_return"],
                "equity": row["equity"],
                "exposure": row.get("exposure"),
                "positions": row["positions"],
                "top_codes": ",".join(
                    day_targets.head(5).get("ts_code", pd.Series(dtype=str)).astype(str).tolist()
                ),
                "avg_close": day_targets["close"].mean() if "close" in day_targets else None,
                "avg_premium_rate": (
                    day_targets["premium_rate"].mean() if "premium_rate" in day_targets else None
                ),
                "avg_price_position_252": (
                    day_targets["price_position_252"].mean()
                    if "price_position_252" in day_targets
                    else None
                ),
                "avg_target_weight": (
                    day_targets["target_weight"].mean() if "target_weight" in day_targets else None
                ),
            }
        )
    return pd.DataFrame(rows)


def build_position_case_analysis(position_trades: pd.DataFrame | None) -> pd.DataFrame:
    if position_trades is None or position_trades.empty:
        return pd.DataFrame()
    closed = position_trades[position_trades["status"] == "closed"].copy()
    if closed.empty:
        return position_trades.copy()
    closed["return"] = pd.to_numeric(closed["return"], errors="coerce")
    worst = closed.sort_values("return").head(20).assign(case_type="worst_closed")
    best = closed.sort_values("return", ascending=False).head(20).assign(case_type="best_closed")
    return pd.concat([worst, best], ignore_index=True)


def variant_configs() -> list[HoldingGridConfig]:
    return [
        HoldingGridConfig(
            name="hold_success_strict_block",
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
            name="hold_success_balanced_block",
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
            name="hold_success_balanced_scaled",
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
            name="hold_success_deep_value",
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
        HoldingGridConfig(
            name="hold_balanced_monthly",
            top_n=8,
            max_total_weight=0.85,
            max_position_weight=0.14,
            max_entry_price=116.0,
            min_premium_rate=0.0,
            max_premium_rate=24.0,
            max_double_low=138.0,
            max_price_position_252=0.32,
            min_drawdown_from_252_high=0.08,
            min_amount=3_000.0,
            min_momentum_20d=-0.10,
            exit_price=132.0,
            exit_premium_rate=45.0,
            exit_double_low=168.0,
            exit_price_position_252=0.82,
        ),
        HoldingGridConfig(
            name="hold_return_target",
            top_n=10,
            max_total_weight=1.00,
            max_position_weight=0.16,
            max_entry_price=118.0,
            min_premium_rate=-2.0,
            max_premium_rate=28.0,
            max_double_low=145.0,
            max_price_position_252=0.40,
            min_drawdown_from_252_high=0.06,
            min_amount=2_000.0,
            min_momentum_20d=-0.12,
            exit_price=138.0,
            exit_premium_rate=55.0,
            exit_double_low=178.0,
            exit_price_position_252=0.88,
        ),
        HoldingGridConfig(
            name="hold_return_core",
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
        HoldingGridConfig(
            name="hold_core_fast_exit",
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
            exit_price=132.0,
            exit_premium_rate=45.0,
            exit_double_low=165.0,
            exit_price_position_252=0.82,
        ),
        HoldingGridConfig(
            name="hold_core_market_scaled",
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
            name="hold_core_market_block",
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
            market_risk_mode="block_downtrend",
            exit_price=132.0,
            exit_premium_rate=45.0,
            exit_double_low=165.0,
            exit_price_position_252=0.82,
        ),
        HoldingGridConfig(
            name="hold_aggressive_low",
            top_n=12,
            max_total_weight=1.00,
            max_position_weight=0.18,
            max_entry_price=120.0,
            min_premium_rate=-5.0,
            max_premium_rate=32.0,
            max_double_low=150.0,
            max_price_position_252=0.45,
            min_drawdown_from_252_high=0.05,
            min_amount=1_500.0,
            min_momentum_20d=-0.15,
            exit_price=145.0,
            exit_premium_rate=65.0,
            exit_double_low=195.0,
            exit_price_position_252=0.92,
        ),
        HoldingGridConfig(
            name="hold_strict_risk",
            top_n=6,
            max_total_weight=0.65,
            max_position_weight=0.12,
            max_entry_price=114.0,
            min_premium_rate=0.0,
            max_premium_rate=20.0,
            max_double_low=132.0,
            max_price_position_252=0.25,
            min_drawdown_from_252_high=0.10,
            min_amount=5_000.0,
            min_momentum_20d=-0.08,
            exit_price=128.0,
            exit_premium_rate=40.0,
            exit_double_low=158.0,
            exit_price_position_252=0.75,
            min_credit_rating="AA-",
        ),
        HoldingGridConfig(
            name="hold_monthly_swing",
            top_n=10,
            max_total_weight=0.95,
            max_position_weight=0.15,
            max_entry_price=118.0,
            min_premium_rate=0.0,
            max_premium_rate=26.0,
            max_double_low=142.0,
            max_price_position_252=0.38,
            min_drawdown_from_252_high=0.07,
            min_amount=3_000.0,
            min_momentum_20d=-0.10,
            exit_price=136.0,
            exit_premium_rate=52.0,
            exit_double_low=174.0,
            exit_price_position_252=0.86,
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
    cases = build_case_analysis(result.equity, result.targets)
    case_path = out_dir / "cases.csv"
    cases.to_csv(case_path, index=False)
    position_cases = build_position_case_analysis(result.position_trades)
    position_case_path = out_dir / "position_cases.csv"
    position_cases.to_csv(position_case_path, index=False)
    summary = dict(result.summary)
    summary["variant"] = grid_config.name
    summary["rebalance"] = rebalance
    summary["summary_path"] = str(paths["summary"])
    summary["case_path"] = str(case_path)
    summary["position_case_path"] = str(position_case_path)
    return summary


def main() -> None:
    daily = pd.read_parquet(DAILY_PATH)
    basic = pd.read_parquet(BASIC_PATH)
    call = pd.read_parquet(CALL_PATH) if CALL_PATH.exists() else pd.DataFrame()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    for config in variant_configs():
        for rebalance in ["daily", "weekly", "monthly"]:
            for start_date in ["20150101", "20240101"]:
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
    top = summary_frame.sort_values(
        [
            "start_date",
            "position_win_rate",
            "position_profit_factor",
            "average_position_return",
            "max_drawdown",
        ],
        ascending=[True, False, False, False, False],
    )[
        [
            "variant",
            "rebalance",
            "start_date",
            "end_date",
            "total_return",
            "annual_return",
            "max_drawdown",
            "sharpe",
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
    ].head(16)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "top": top.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
