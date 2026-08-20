#!/usr/bin/env python3
"""Build reproducible local market, financial, forecast and governance workpapers for one A-share.

The script performs mechanical extraction only.  It does not score the company or
choose valuation assumptions; those decisions remain part of the individual Deep
review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data/raw"
GOV_ROOT = PROJECT_ROOT / "reports/good_company_deep_20260809/sources/tushare_governance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cutoff", default="20260807", help="Latest YYYYMMDD market date")
    return parser.parse_args()


def _read_filtered(
    path: Path, ticker: str, *, columns: list[str] | None = None
) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=columns,
        filters=[("ts_code", "==", ticker)],
    )


def _latest_statement(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    data = frame.copy()
    data["ann_date"] = data["ann_date"].astype(str)
    data["end_date"] = data["end_date"].astype(str)
    data = data[data["ann_date"].le(cutoff)]
    if "report_type" in data:
        data = data[data["report_type"].astype(str).eq("1")]
    data = data.sort_values(["end_date", "ann_date"])
    return data.drop_duplicates("end_date", keep="last")


def build_financials(ticker: str, cutoff: str) -> pd.DataFrame:
    income = _latest_statement(_read_filtered(RAW_ROOT / "income.parquet", ticker), cutoff)
    cash = _latest_statement(_read_filtered(RAW_ROOT / "cashflow.parquet", ticker), cutoff)
    balance = _latest_statement(_read_filtered(RAW_ROOT / "balancesheet.parquet", ticker), cutoff)
    indicator = _latest_statement(_read_filtered(RAW_ROOT / "fina_indicator.parquet", ticker), cutoff)

    income = income[income["end_date"].str.endswith("1231")]
    cash = cash[cash["end_date"].str.endswith("1231")]
    balance = balance[balance["end_date"].str.endswith("1231")]
    indicator = indicator[indicator["end_date"].str.endswith("1231")]

    base = income.copy()
    for extra in (cash, balance, indicator):
        duplicate_columns = [
            column for column in extra.columns if column in base.columns and column not in {"end_date"}
        ]
        extra = extra.drop(columns=duplicate_columns)
        base = base.merge(extra, on="end_date", how="left")

    base["simple_fcf"] = base["n_cashflow_act"] - base["c_pay_acq_const_fiolta"]
    base["ocf_to_consolidated_np"] = base["n_cashflow_act"] / base["n_income"].replace(0, np.nan)
    base["simple_fcf_to_consolidated_np"] = base["simple_fcf"] / base["n_income"].replace(0, np.nan)
    preferred = [
        "ts_code", "ann_date", "end_date", "report_type", "revenue", "operate_profit",
        "total_profit", "n_income", "n_income_attr_p", "total_revenue", "income_tax",
        "minority_gain", "n_cashflow_act", "c_pay_acq_const_fiolta", "total_assets",
        "total_liab", "total_hldr_eqy_exc_min_int", "money_cap", "inventories",
        "fix_assets", "goodwill", "roe_waa", "grossprofit_margin", "netprofit_margin",
        "debt_to_assets", "ar_turn", "inv_turn", "simple_fcf",
        "ocf_to_consolidated_np", "simple_fcf_to_consolidated_np",
    ]
    return base[[column for column in preferred if column in base.columns]].sort_values("end_date")


def build_market(ticker: str, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    prices = _read_filtered(
        RAW_ROOT / "daily_partitioned",
        ticker,
        columns=[
            "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
            "change", "pct_chg", "vol", "amount",
        ],
    )
    prices["trade_date"] = prices["trade_date"].astype(str)
    prices = prices[prices["trade_date"].le(cutoff)].sort_values("trade_date")

    basics = _read_filtered(
        RAW_ROOT / "daily_basic",
        ticker,
        columns=[
            "ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "volume_ratio",
            "pe", "pe_ttm", "pb", "ps", "ps_ttm", "dv_ratio", "dv_ttm", "total_share",
            "float_share", "free_share", "total_mv", "circ_mv",
        ],
    )
    basics["trade_date"] = basics["trade_date"].astype(str)
    basics = basics[basics["trade_date"].le(cutoff)].sort_values("trade_date")

    close = pd.to_numeric(prices["close"], errors="coerce")
    high = pd.to_numeric(prices["high"], errors="coerce")
    low = pd.to_numeric(prices["low"], errors="coerce")
    volume = pd.to_numeric(prices["vol"], errors="coerce")
    latest_close = float(close.iloc[-1])

    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    previous = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    atr_pct = true_range.rolling(14).mean() / close

    index = pd.read_parquet(RAW_ROOT / "index_000300.SH.parquet")
    index["trade_date"] = index["trade_date"].astype(str)
    index = index[index["trade_date"].le(cutoff)].sort_values("trade_date")
    index_close = pd.to_numeric(index["close"], errors="coerce")

    summary: dict[str, object] = {
        "as_of": prices.iloc[-1]["trade_date"],
        "close": latest_close,
        "observations": len(prices),
    }
    for window in (5, 10, 20, 50, 120, 200, 250):
        summary[f"ma{window}"] = float(close.tail(window).mean()) if len(close) >= window else None
    for window in (20, 50, 120, 250):
        summary[f"return_{window}d"] = float(latest_close / close.iloc[-window - 1] - 1) if len(close) > window else None
        summary[f"csi300_return_{window}d"] = (
            float(index_close.iloc[-1] / index_close.iloc[-window - 1] - 1)
            if len(index_close) > window else None
        )
    summary.update(
        {
            "rsi14": float(rsi.iloc[-1]),
            "atr14_pct": float(atr_pct.iloc[-1]),
            "volume_5d_over_20d": float(volume.tail(5).mean() / volume.tail(20).mean()),
            "high_20d": float(high.tail(20).max()),
            "low_20d": float(low.tail(20).min()),
            "high_250d": float(high.tail(250).max()),
            "low_250d": float(low.tail(250).min()),
        }
    )

    current = basics.iloc[-1]
    valuation_summary: dict[str, object] = {
        "as_of": current["trade_date"],
        "observations": len(basics),
        "current": {},
        "percentiles": {},
    }
    for metric in ("pe_ttm", "pb", "ps_ttm", "dv_ttm"):
        series = pd.to_numeric(basics[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if metric == "pe_ttm":
            series = series[series.gt(0)]
        value = float(current[metric]) if pd.notna(current[metric]) else None
        valuation_summary["current"][metric] = value
        valuation_summary["percentiles"][metric] = {
            "current_percentile": float((series.le(value)).mean()) if value is not None and len(series) else None,
            "p10": float(series.quantile(0.10)) if len(series) else None,
            "p25": float(series.quantile(0.25)) if len(series) else None,
            "p50": float(series.quantile(0.50)) if len(series) else None,
            "p75": float(series.quantile(0.75)) if len(series) else None,
            "p90": float(series.quantile(0.90)) if len(series) else None,
            "valid_observations": int(len(series)),
        }
    valuation_summary["current"].update(
        {
            "total_share_10k": float(current["total_share"]),
            "total_mv_10k_cny": float(current["total_mv"]),
        }
    )
    return prices, basics, summary, valuation_summary


def build_forecasts(ticker: str, cutoff: str, shares: float) -> pd.DataFrame:
    forecasts = _read_filtered(RAW_ROOT / "analyst_forecasts.parquet", ticker)
    forecasts["report_date"] = forecasts["report_date"].astype(str).str.replace("-", "", regex=False)
    forecasts = forecasts[forecasts["report_date"].le(cutoff)].copy()
    forecasts["implied_parent_np"] = pd.to_numeric(forecasts["eps"], errors="coerce") * shares
    return forecasts.sort_values(["report_date", "forecast_year"], na_position="last")


def write_governance(ticker: str, output_dir: Path) -> None:
    for name in ("pledge_stat", "dividend", "repurchase", "fina_audit"):
        source = GOV_ROOT / f"{name}.parquet"
        if not source.exists():
            continue
        frame = _read_filtered(source, ticker)
        if "end_date" in frame:
            frame = frame.sort_values("end_date", ascending=False)
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    financials = build_financials(args.ticker, args.cutoff)
    prices, basics, technical, valuation = build_market(args.ticker, args.cutoff)
    shares = float(basics.iloc[-1]["total_share"]) * 10_000
    forecasts = build_forecasts(args.ticker, args.cutoff, shares)

    financials.to_csv(output_dir / "historical_financials.csv", index=False)
    prices.to_csv(output_dir / "price_history.csv", index=False)
    basics.to_csv(output_dir / "historical_valuation_daily.csv", index=False)
    forecasts.to_csv(output_dir / "forecast_detail.csv", index=False)
    (output_dir / "technical_summary.json").write_text(
        json.dumps(technical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "valuation_summary.json").write_text(
        json.dumps(valuation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_governance(args.ticker, output_dir)
    print(
        json.dumps(
            {
                "ticker": args.ticker,
                "output_dir": str(output_dir),
                "financial_years": len(financials),
                "price_observations": len(prices),
                "valuation_observations": len(basics),
                "forecast_rows": len(forecasts),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
