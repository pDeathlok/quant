"""Exploratory Tushare sector ETF turnover-surge backtest, frozen simple rules."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/public_strategy_validation_20260929/sector_turnover_surge"
SECTORS = (
    "512010.SH", "512200.SH", "512400.SH", "512660.SH", "512800.SH", "512880.SH",
    "512480.SH", "512690.SH", "515030.SH", "515220.SH", "515790.SH", "159825.SZ",
)
START, END = "20200101", "20260924"
FEE_ONE_WAY = 0.001

spec = importlib.util.spec_from_file_location("public_validation", ROOT / "scripts/research/validate_public_strategies.py")
validation = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validation)


def fetch_sector_etfs(token: str) -> pd.DataFrame:
    cache = OUT / "source"
    cache.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(token)
    frames = []
    for code in SECTORS:
        path = cache / f"{code.replace('.', '_')}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            dailies, factors = [], []
            for lo, hi in (("20200101", "20221231"), ("20230101", END)):
                dailies.append(pro.fund_daily(ts_code=code, start_date=lo, end_date=hi,
                                              fields="ts_code,trade_date,close,amount"))
                factors.append(pro.fund_adj(ts_code=code, start_date=lo, end_date=hi))
            frame = pd.concat(dailies, ignore_index=True).merge(
                pd.concat(factors, ignore_index=True), on=["ts_code", "trade_date"], how="left"
            )
            frame = frame.drop_duplicates(["ts_code", "trade_date"]).sort_values("trade_date")
            if frame.empty or frame.adj_factor.isna().any():
                raise RuntimeError(f"Tushare ETF price/factor incomplete for {code}")
            frame.to_parquet(path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_targets(prices: pd.DataFrame, amount: pd.DataFrame, benchmark: pd.Series,
                  mode: str) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    """Weekly last close signal, top 2; source data available only through signal close."""
    ret20 = prices.pct_change(20, fill_method=None)
    ma60 = prices.rolling(60, min_periods=60).mean()
    recent_amount = amount.rolling(5, min_periods=5).mean()
    prior_amount = amount.shift(5).rolling(60, min_periods=60).mean()
    surge = recent_amount / prior_amount
    market_ok = benchmark > benchmark.rolling(200, min_periods=200).mean()
    weekly = prices.groupby(prices.index.to_period("W-FRI")).tail(1).index
    targets, audit = {}, []
    for day in weekly:
        eligible = (ret20.loc[day] > 0) & (prices.loc[day] > ma60.loc[day])
        eligible &= (recent_amount.loc[day] > 10000)  # Tushare amount in 1,000 yuan
        if mode.startswith("surge"):
            eligible &= surge.loc[day] >= 1.5
            score = surge.loc[day] * ret20.loc[day]
        else:
            score = ret20.loc[day]
        selected = score[eligible & score.notna()].sort_values(ascending=False).head(2).index.tolist()
        exposure = 0.6 if "60pct" in mode else 1.0
        if mode == "surge_60pct_market_gate" and not bool(market_ok.loc[day]):
            exposure = 0.3
        target = {code: exposure / 2 for code in selected}  # cash for missing second slot
        targets[day] = target
        audit.append({"signal_date": day, "mode": mode, "selected": ",".join(selected),
                      "selected_count": len(selected), "target_exposure": sum(target.values()),
                      "market_ok": bool(market_ok.loc[day])})
    return targets, pd.DataFrame(audit)


def run(token: str) -> dict:
    frame = fetch_sector_etfs(token)
    frame["date"] = pd.to_datetime(frame.trade_date)
    frame["adjusted_close"] = frame.close * frame.adj_factor
    prices = frame.pivot(index="date", columns="ts_code", values="adjusted_close").sort_index()
    amount = frame.pivot(index="date", columns="ts_code", values="amount").reindex(prices.index)
    benchmark_frame = pd.read_parquet(
        ROOT / "reports/public_strategy_validation_20260929/source/510300_SH_2018_2026.parquet"
    )
    benchmark_frame["date"] = pd.to_datetime(benchmark_frame.trade_date)
    benchmark = (benchmark_frame.set_index("date").sort_index().close
                 * benchmark_frame.set_index("date").sort_index().adj_factor)
    dates = prices.index.intersection(benchmark.index)
    prices = prices.reindex(dates)
    amount = amount.reindex(dates)
    benchmark = benchmark.reindex(dates)
    modes = ("momentum_full", "surge_full", "surge_60pct", "surge_60pct_market_gate")
    results, daily, audits = [], {}, []
    start = pd.Timestamp("2021-04-01")  # 60-day amount history for all late-2020 funds
    for mode in modes:
        targets, audit = build_targets(prices, amount, benchmark, mode)
        audit = audit[audit.signal_date >= start]
        audits.append(audit)
        series, turnover, count = validation.simulate(prices, targets)
        previous_fee = validation.FEE_ONE_WAY
        try:
            validation.FEE_ONE_WAY = 0.0
            gross_series, _, _ = validation.simulate(prices, targets)
        finally:
            validation.FEE_ONE_WAY = previous_fee
        series = series[series.index >= start]
        gross_series = gross_series[gross_series.index >= start]
        daily[mode] = series
        for period, sample in (("full", series), ("2023_onward", series[series.index >= "2023-01-01"])):
            gross_sample = gross_series.loc[sample.index]
            results.append({"strategy": mode, "period": period, **validation.metrics(sample),
                            "gross_annual_return_zero_trading_cost": validation.metrics(gross_sample)["annual_return"],
                            "turnover_notional_full": turnover, "rebalance_count_full": count,
                            "average_selected_count": float(audit.selected_count.mean()),
                            "average_target_exposure": float(audit.target_exposure.mean())})
    baseline = benchmark.pct_change(fill_method=None).fillna(0).loc[start:]
    baseline.iloc[0] -= FEE_ONE_WAY
    daily["buy_hold_510300"] = baseline
    for period, sample in (("full", baseline), ("2023_onward", baseline[baseline.index >= "2023-01-01"])):
        results.append({"strategy": "buy_hold_510300", "period": period, **validation.metrics(sample)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daily).to_csv(OUT / "daily_returns.csv", index_label="date")
    pd.concat(audits, ignore_index=True).to_csv(OUT / "weekly_signals.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)
    result = {"sample_start": str(start.date()), "sample_end": str(dates.max().date()),
              "sector_etfs": list(SECTORS), "results": results,
              "cautions": [
                  "Turnover surge is ETF trading activity, not verified net capital inflow.",
                  "The fixed ETF pool uses funds surviving to 2026 and may have selection bias.",
                  "Signals use a weekly close and trade at next close; no intraday response is tested.",
                  "Historical bid-ask spread and market impact beyond 0.10% one-way cost are not modeled.",
                  "2023 onward is a chronological diagnostic, not an untouched blind holdout.",
              ]}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    token = os.getenv("TUSHARE_TOKEN") or dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN missing")
    result = run(token)
    print(pd.DataFrame(result["results"])[["strategy", "period", "annual_return", "max_drawdown"]].to_string(index=False))
