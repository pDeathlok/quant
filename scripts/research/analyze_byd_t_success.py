#!/usr/bin/env python
"""Select a sideways-regime BYD T strategy with chronological validation.

The selection rule first requires positive net profit and profit factor in both
the training and validation periods, then ranks the Wilson lower bound of the
fee-adjusted cycle win rate. Strong trends are excluded using lagged daily
features only. The held-out period is evaluated only afterward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant.research.byd_t_backtest import (
    BydTBacktestConfig,
    ChinaAStockFees,
    backtest_byd_t,
    prepare_byd_intraday_bars,
    sideways_regime_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data/cache/baostock_002594_5min_20150101_20260717_qfq.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports/byd_002594/t_success_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


REGIME_PROFILES = {
    "strict": {
        "sideways_max_abs_return_60": 0.08,
        "sideways_max_abs_ma20_slope_5": 0.015,
        "sideways_max_abs_ma20_ma60_gap": 0.05,
        "sideways_max_range_width_60": 0.28,
    },
    "balanced": {
        "sideways_max_abs_return_60": 0.12,
        "sideways_max_abs_ma20_slope_5": 0.025,
        "sideways_max_abs_ma20_ma60_gap": 0.08,
        "sideways_max_range_width_60": 0.35,
    },
    "broad": {
        "sideways_max_abs_return_60": 0.16,
        "sideways_max_abs_ma20_slope_5": 0.03,
        "sideways_max_abs_ma20_ma60_gap": 0.10,
        "sideways_max_range_width_60": 0.40,
    },
}


def coarse_configs() -> Iterable[BydTBacktestConfig]:
    for profile_name, return_cap, entry, edge in product(
        ["balanced", "broad"],
        [0.0, 0.04],
        [0.006, 0.007, 0.008, 0.009, 0.010],
        [0.40, 0.50],
    ):
        yield BydTBacktestConfig(
            direction="positive",
            entry_deviation=entry,
            profit_target=0.004,
            stop_loss=0.020,
            max_holding_sessions=3,
            latest_entry_minute=14 * 60 + 50,
            tail_entry_minute=14 * 60 + 35,
            tail_extra_deviation=0.003,
            force_flat_minute=None,
            require_sideways_regime=True,
            positive_range_position_max=edge,
            positive_entry_max_prior_return_60=return_cap,
            **REGIME_PROFILES[profile_name],
        )


def evaluate_periods(
    bars_by_period: dict[str, pd.DataFrame],
    config: BydTBacktestConfig,
    fees: ChinaAStockFees,
) -> dict[str, Any]:
    row: dict[str, Any] = asdict(config)
    for name, bars in bars_by_period.items():
        result = backtest_byd_t(bars, config, fees, track_equity=False)
        for key, value in result.metrics.items():
            if isinstance(value, (int, float, bool)):
                row[f"{name}_{key}"] = value
    return row


def eligible(row: pd.Series) -> bool:
    return bool(
        row["train_cycles"] >= 12
        and row["validation_cycles"] >= 12
        and row["train_win_rate"] >= 0.55
        and row["validation_win_rate"] >= 0.55
        and row["train_net_pnl"] > 0
        and row["validation_net_pnl"] > 0
        and row["train_profit_factor"] >= 1.2
        and row["validation_profit_factor"] >= 1.2
    )


def add_selection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["eligible"] = out.apply(eligible, axis=1)
    out["profitable_both_periods"] = (
        out["train_net_pnl"].gt(0)
        & out["validation_net_pnl"].gt(0)
        & out["train_profit_factor"].gt(1)
        & out["validation_profit_factor"].gt(1)
    )
    out["stable_wilson"] = out[
        ["train_win_rate_wilson_lower", "validation_win_rate_wilson_lower"]
    ].min(axis=1)
    out["stable_profit_factor"] = out[
        ["train_profit_factor", "validation_profit_factor"]
    ].min(axis=1)
    out["combined_net_pnl"] = out["train_net_pnl"] + out["validation_net_pnl"]
    out["worst_net_per_cycle"] = (
        out[["train_net_pnl", "validation_net_pnl"]].min(axis=1)
        / out[["train_cycles", "validation_cycles"]].max(axis=1).clip(lower=1)
    )
    return out.sort_values(
        [
            "eligible",
            "profitable_both_periods",
            "stable_wilson",
            "stable_profit_factor",
            "combined_net_pnl",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def config_from_row(row: pd.Series) -> BydTBacktestConfig:
    fields = BydTBacktestConfig.__dataclass_fields__
    integer_fields = {
        "base_shares",
        "minimum_shares",
        "maximum_shares",
        "lot_shares",
        "max_positive_shares",
        "max_reverse_shares",
        "max_holding_sessions",
        "earliest_entry_minute",
        "latest_entry_minute",
        "tail_entry_minute",
    }
    values: dict[str, Any] = {}
    for name in fields:
        value = row[name]
        if name == "force_flat_minute" and pd.isna(value):
            value = None
        elif name in integer_fields:
            value = int(value)
        elif name == "require_sideways_regime":
            value = bool(value)
        values[name] = value
    return BydTBacktestConfig(**values)


def refinement_configs(coarse: pd.DataFrame) -> list[BydTBacktestConfig]:
    candidates: dict[tuple[Any, ...], BydTBacktestConfig] = {}
    config_fields = tuple(BydTBacktestConfig.__dataclass_fields__)
    for _, row in coarse.head(3).iterrows():
        base = config_from_row(row)
        for target, stop, maximum_holding in product(
            [0.003, 0.004, 0.005],
            [0.012, 0.016, 0.020],
            [2, 3],
        ):
            refined = replace(
                base,
                profit_target=target,
                stop_loss=stop,
                max_holding_sessions=maximum_holding,
            )
            key = tuple(getattr(refined, name) for name in config_fields)
            candidates[key] = refined
    return list(candidates.values())


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serializable(item) for item in value]
    if isinstance(value, float) and (pd.isna(value) or value in {float("inf"), float("-inf")}):
        return None
    if hasattr(value, "item"):
        return serializable(value.item())
    return value


def main() -> None:
    args = parse_args()
    prepared = prepare_byd_intraday_bars(pd.read_parquet(args.input))
    train = prepared[prepared["date"].dt.year.le(2023)].reset_index(drop=True)
    validation = prepared[prepared["date"].dt.year.between(2024, 2025)].reset_index(drop=True)
    test = prepared[prepared["date"].dt.year.ge(2026)].reset_index(drop=True)
    fees = ChinaAStockFees()
    selection_periods = {"train": train, "validation": validation}

    coarse_candidates = list(coarse_configs())
    coarse_rows = []
    for index, config in enumerate(coarse_candidates, start=1):
        coarse_rows.append(evaluate_periods(selection_periods, config, fees))
        if index % 8 == 0 or index == len(coarse_candidates):
            print(f"coarse {index}/{len(coarse_candidates)}", flush=True)
    coarse = add_selection_columns(pd.DataFrame(coarse_rows))
    refined_candidates = refinement_configs(coarse)
    refined_rows = []
    for index, config in enumerate(refined_candidates, start=1):
        refined_rows.append(evaluate_periods(selection_periods, config, fees))
        if index % 8 == 0 or index == len(refined_candidates):
            print(f"refined {index}/{len(refined_candidates)}", flush=True)
    refined = add_selection_columns(pd.DataFrame(refined_rows))
    ranked = add_selection_columns(pd.concat([coarse, refined], ignore_index=True))
    selected_row = ranked.iloc[0]
    selected_config = config_from_row(selected_row)
    validation_passed = bool(selected_row["eligible"])

    train_result = backtest_byd_t(train, selected_config, fees)
    validation_result = backtest_byd_t(validation, selected_config, fees)
    test_result = backtest_byd_t(test, selected_config, fees)
    full_result = backtest_byd_t(prepared, selected_config, fees)
    generic_config = replace(selected_config, require_sideways_regime=False)
    ungated_return_config = replace(
        selected_config, positive_entry_max_prior_return_60=10.0
    )
    generic_train = backtest_byd_t(train, generic_config, fees, track_equity=False)
    generic_validation = backtest_byd_t(
        validation, generic_config, fees, track_equity=False
    )
    generic_test = backtest_byd_t(test, generic_config, fees, track_equity=False)
    ungated_return_train = backtest_byd_t(
        train, ungated_return_config, fees, track_equity=False
    )
    ungated_return_validation = backtest_byd_t(
        validation, ungated_return_config, fees, track_equity=False
    )
    ungated_return_test = backtest_byd_t(
        test, ungated_return_config, fees, track_equity=False
    )

    regime_calendar = prepared.groupby("date", sort=True).first().reset_index()
    regime_calendar["sideways"] = sideways_regime_mask(
        regime_calendar, selected_config
    ).to_numpy()
    regime_calendar["strong_uptrend_excluded"] = (
        regime_calendar["prior_return_60"].gt(
            selected_config.sideways_max_abs_return_60
        )
        | regime_calendar["prior_ma20_slope_5"].gt(
            selected_config.sideways_max_abs_ma20_slope_5
        )
        | regime_calendar["prior_ma20_ma60_gap"].gt(
            selected_config.sideways_max_abs_ma20_ma60_gap
        )
    )
    annual_rows = []
    for year, year_bars in prepared.groupby(prepared["date"].dt.year, sort=True):
        year_result = backtest_byd_t(
            year_bars.reset_index(drop=True), selected_config, fees
        )
        annual_rows.append({"year": int(year), **year_result.metrics})
    annual = pd.DataFrame(annual_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coarse.to_csv(args.output_dir / "coarse_grid.csv", index=False)
    refined.to_csv(args.output_dir / "refined_grid.csv", index=False)
    ranked.to_csv(args.output_dir / "ranked_grid.csv", index=False)
    annual.to_csv(args.output_dir / "selected_annual_metrics.csv", index=False)
    regime_calendar.to_csv(args.output_dir / "selected_regime_calendar.csv", index=False)
    full_result.cycles.to_csv(args.output_dir / "selected_cycles.csv", index=False)
    full_result.orders.to_csv(args.output_dir / "selected_orders.csv", index=False)
    summary = {
        "data": {
            "source": str(args.input),
            "bars": len(prepared),
            "sessions": int(prepared["date"].nunique()),
            "train": "2020-2023 (entries only in lagged-feature sideways regimes)",
            "validation": "2024-2025 (entries only in lagged-feature sideways regimes)",
            "held_out_test": "2026-01-01 through latest available date",
        },
        "fees": asdict(fees),
        "selection_rule": (
            "Require >=12 cycles per selection segment, >=55% fee-adjusted win rate, "
            "positive net PnL and profit factor >=1.2 in both train and validation; "
            "then rank the lower of "
            "their 95% Wilson win-rate bounds. The 2026 held-out period is not used "
            "for configuration selection."
        ),
        "regime_rule": (
            "Entries require prior-day 60-session return, MA20 slope, MA20/MA60 gap, "
            "60-session range width and ATR to pass the selected sideways thresholds. "
            "Current price must also be near the appropriate range edge without "
            "breaking the prior range beyond tolerance."
        ),
        "validation_passed": validation_passed,
        "selected_config": asdict(selected_config),
        "train_metrics": train_result.metrics,
        "validation_metrics": validation_result.metrics,
        "held_out_test_metrics": test_result.metrics,
        "full_sample_metrics": full_result.metrics,
        "annual_metrics": annual.to_dict(orient="records"),
        "generic_same_parameters_without_regime_gate": {
            "train": generic_train.metrics,
            "validation": generic_validation.metrics,
            "held_out_test": generic_test.metrics,
        },
        "same_sideways_strategy_without_directional_return_gate": {
            "train": ungated_return_train.metrics,
            "validation": ungated_return_validation.metrics,
            "held_out_test": ungated_return_test.metrics,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(serializable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(serializable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
