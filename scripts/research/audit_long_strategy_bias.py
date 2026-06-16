"""Audit long-strategy universe and factor bias.

This script focuses on leakage risks that do not show up in the per-date
percentile formulas themselves: full-period universe prefiltering, current-name
ST/delist filtering, and early-year universe distortion.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SCRIPT = PROJECT_ROOT / "scripts/research/backtest_long_dividend_quality.py"
OUTPUT_DIR = PROJECT_ROOT / "reports/long_dividend_quality/audit_bias"


def load_long_module() -> Any:
    module_name = "long_dividend_quality_research_for_audit"
    spec = importlib.util.spec_from_file_location(module_name, RESEARCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RESEARCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def annual_returns(equity: pd.DataFrame) -> pd.DataFrame:
    frame = equity.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        first_equity = float(group["equity"].iloc[0])
        last_equity = float(group["equity"].iloc[-1])
        dd = group["equity"] / group["equity"].cummax() - 1.0
        rows.append(
            {
                "year": int(year),
                "return": last_equity / first_equity - 1.0,
                "max_drawdown": float(dd.min()),
                "avg_weight": float(group["total_weight"].mean()),
                "avg_positions": float(group["positions"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_equity(module: Any, equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    benchmark = module.load_benchmark(equity["date"].min(), equity["date"].max())
    summary = module.summarize(equity, trades, benchmark)
    return {
        "annual_return": float(summary["annual_return"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "sharpe": float(summary["sharpe"]),
        "avg_total_weight": float(summary["avg_total_weight"]),
        "avg_positions": float(summary["avg_positions"]),
    }


def build_pipeline(
    module: Any,
    config: Any,
    *,
    point_in_time_universe: bool,
    include_current_st_filter: bool,
    include_analyst: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    start = module.parse_date(config.start)
    end = module.parse_date(config.end)
    stock_basic = module.load_stock_basic()
    daily_basic, coverage = module.load_daily_basic_monthly(start, end)
    if daily_basic.empty:
        raise RuntimeError("No daily_basic data found.")

    if point_in_time_universe:
        candidate_symbols = None
    else:
        candidate_symbols = module.select_candidate_symbols_from_daily_basic(daily_basic, stock_basic, config)

    daily_features, daily_returns = module.load_daily_monthly_features(
        start,
        end,
        stock_basic,
        candidate_symbols=candidate_symbols,
    )
    executable_start_text = max(config.start, coverage.get("first_trade_date") or config.start)
    executable_start = module.parse_date(executable_start_text)
    daily_features = daily_features[daily_features["date"] >= executable_start].copy()
    daily_returns = daily_returns[daily_returns["date"] >= executable_start].copy()
    daily_basic = daily_basic[daily_basic["date"] >= executable_start].copy()
    daily_features = daily_features.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(["date", "ts_code"], keep="last")
    daily_returns = daily_returns.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(["date", "ts_code"], keep="last")
    daily_basic = daily_basic.sort_values(["date", "ts_code", "trade_date"]).drop_duplicates(["date", "ts_code"], keep="last")

    if point_in_time_universe:
        min_dv_ttm = 0.0 if config.variant in module.GROWTH_VARIANTS else (
            0.5 if config.variant in module.MARKET_REGIME_VARIANTS else config.prefilter_min_dv_ttm
        )
        point_mask = (
            (pd.to_numeric(daily_basic["dv_ttm"], errors="coerce").fillna(0) >= min_dv_ttm)
            & (pd.to_numeric(daily_basic["total_mv"], errors="coerce") >= config.prefilter_min_total_mv)
            & (pd.to_numeric(daily_basic["circ_mv"], errors="coerce") >= config.prefilter_min_circ_mv)
        )
        daily_basic = daily_basic[point_mask].copy()

    merged = daily_features.merge(
        daily_basic.drop(columns=["trade_date"]),
        on=["date", "ts_code"],
        how="inner",
    )
    if include_current_st_filter and not stock_basic.empty:
        names = stock_basic.set_index("ts_code")["name"].astype(str)
        st_symbols = set(names[names.str.contains("ST|退", regex=True, na=False)].index)
        merged = merged[~merged["ts_code"].astype(str).isin(st_symbols)].copy()

    merged = module.load_financial_asof(merged)
    if include_analyst and config.variant in module.GROWTH_VARIANTS:
        merged = module.load_analyst_forecast_asof(merged)
    else:
        merged = module.add_empty_analyst_forecast_columns(merged)

    if config.variant in module.MARKET_REGIME_VARIANTS:
        market_regime = module.load_market_regime(merged["date"].min(), merged["date"].max())
        if market_regime.empty:
            merged["market_regime"] = "neutral"
            merged["index_ma_120_slope_20d"] = np.nan
            merged["index_return_20d"] = np.nan
            merged["index_return_60d"] = np.nan
            merged["index_return_120d"] = np.nan
            merged["index_drawdown_60d"] = np.nan
            merged["index_overheat"] = False
        else:
            merged = merged.merge(market_regime, on="date", how="left")
            merged["market_regime"] = merged["market_regime"].fillna("neutral")
    merged["cashflow_quality"] = np.nan

    scored = module.build_scores(merged, config)
    targets = module.make_monthly_targets(scored, config)
    targets = targets.sort_values(["rebalance_date", "ts_code"]).drop_duplicates(["rebalance_date", "ts_code"], keep="last")
    equity, trades = module.run_portfolio_backtest(daily_returns, targets, scored, config)
    audit = {
        "rows": int(len(merged)),
        "symbols": int(merged["ts_code"].nunique()),
        "months": int(merged["date"].nunique()),
        "candidate_symbols": None if candidate_symbols is None else int(len(candidate_symbols)),
        "point_in_time_universe": point_in_time_universe,
        "include_current_st_filter": include_current_st_filter,
    }
    return scored, targets, equity, trades, audit


def universe_diagnostics(module: Any, config: Any) -> pd.DataFrame:
    start = module.parse_date(config.start)
    end = module.parse_date(config.end)
    stock_basic = module.load_stock_basic()
    daily_basic, _ = module.load_daily_basic_monthly(start, end)
    full_symbols = module.select_candidate_symbols_from_daily_basic(daily_basic, stock_basic, config)
    min_dv_ttm = 0.0 if config.variant in module.GROWTH_VARIANTS else (
        0.5 if config.variant in module.MARKET_REGIME_VARIANTS else config.prefilter_min_dv_ttm
    )
    rows: list[dict[str, Any]] = []
    for signal_date, group in daily_basic.groupby("date", sort=True):
        point_symbols = set(
            group[
                (pd.to_numeric(group["dv_ttm"], errors="coerce").fillna(0) >= min_dv_ttm)
                & (pd.to_numeric(group["total_mv"], errors="coerce") >= config.prefilter_min_total_mv)
                & (pd.to_numeric(group["circ_mv"], errors="coerce") >= config.prefilter_min_circ_mv)
            ]["ts_code"].astype(str)
        )
        if not point_symbols:
            continue
        rows.append(
            {
                "date": signal_date,
                "year": pd.Timestamp(signal_date).year,
                "point_symbols": len(point_symbols),
                "full_prefilter_symbols_seen_that_month": len(point_symbols & full_symbols),
                "symbols_added_only_by_future_prefilter": len(full_symbols - point_symbols),
                "point_symbols_missing_from_full_prefilter": len(point_symbols - full_symbols),
                "full_coverage_ratio": len(point_symbols & full_symbols) / len(point_symbols),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    module = load_long_module()
    config = module.BacktestConfig(variant="v33_bull_boost_defensive_bear_sleeve", start="20130101")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    diagnostics = universe_diagnostics(module, config)
    diagnostics.to_csv(OUTPUT_DIR / "universe_diagnostics.csv", index=False)
    yearly_diag = diagnostics.groupby("year", as_index=False).agg(
        months=("date", "count"),
        avg_point_symbols=("point_symbols", "mean"),
        avg_full_coverage_ratio=("full_coverage_ratio", "mean"),
        avg_missing_from_full_prefilter=("point_symbols_missing_from_full_prefilter", "mean"),
    )
    yearly_diag.to_csv(OUTPUT_DIR / "universe_diagnostics_by_year.csv", index=False)

    baseline_summary_path = PROJECT_ROOT / "reports/long_dividend_quality/v33_bull_boost_defensive_bear_sleeve/l1_dividend_quality_summary.json"
    baseline_summary = {}
    if baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8")).get("summary", {})

    runs: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    for label, point_in_time, st_filter in [
        ("fast_current_full_period_prefilter_no_analyst", False, False),
        ("fast_pit_universe_no_current_st_filter_no_analyst", True, False),
        ("fast_pit_universe_with_current_st_filter_no_analyst", True, True),
    ]:
        print(f"running audit variant: {label}", flush=True)
        _, targets, equity, trades, audit = build_pipeline(
            module,
            config,
            point_in_time_universe=point_in_time,
            include_current_st_filter=st_filter,
            include_analyst=False,
        )
        summary = summarize_equity(module, equity, trades)
        runs.append({"label": label, **audit, **summary, "target_rows": int(len(targets))})
        annual = annual_returns(equity)
        annual.insert(0, "label", label)
        annual_frames.append(annual)
        targets.to_csv(OUTPUT_DIR / f"{label}_targets.csv", index=False)
        equity.to_csv(OUTPUT_DIR / f"{label}_equity.csv", index=False)

    summary_df = pd.DataFrame(runs)
    annual_df = pd.concat(annual_frames, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "audit_summary.csv", index=False)
    annual_df.to_csv(OUTPUT_DIR / "audit_annual_returns.csv", index=False)

    focus_years = annual_df[annual_df["year"].isin([2014, 2017])]
    report = {
        "baseline_existing_v33_with_analyst": baseline_summary,
        "note": "Fast audit disables analyst forecasts to isolate universe/survivorship/value-ranking bias. Existing v33 summary is included for reference.",
        "summary": summary_df.to_dict(orient="records"),
        "focus_years": focus_years.to_dict(orient="records"),
        "yearly_universe": yearly_diag[yearly_diag["year"].isin([2014, 2017])].to_dict(orient="records"),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
