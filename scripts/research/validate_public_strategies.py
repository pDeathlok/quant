"""One-pass, predeclared comparison of public dividend and ETF momentum ideas.

Uses local Tushare snapshots for stocks and Tushare Pro fund_daily/fund_adj
for ETFs. The stock pct_chg field uses ex-rights-adjusted pre_close, so it is
an approximate dividend-adjusted daily return, not a raw close/close return.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
ETFS = ("510300.SH", "510500.SH", "159915.SZ")
START, END = "20190101", "20260924"
FEE_ONE_WAY = 0.001  # commission plus slippage; applied to traded notional


def fetch_etfs(cache: Path, token: str) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(token)
    frames = []
    for code in ETFS:
        path = cache / f"{code.replace('.', '_')}_2018_2026.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            daily, factors = [], []
            for year in range(2018, 2027):
                lo, hi = f"{year}0101", min(f"{year}1231", END)
                if lo > hi:
                    continue
                a = pro.fund_daily(ts_code=code, start_date=lo, end_date=hi,
                                   fields="ts_code,trade_date,close,vol,amount")
                b = pro.fund_adj(ts_code=code, start_date=lo, end_date=hi)
                daily.append(a)
                factors.append(b)
            frame = pd.concat(daily, ignore_index=True).merge(
                pd.concat(factors, ignore_index=True), on=["ts_code", "trade_date"], how="left"
            )
            frame = frame.drop_duplicates(["ts_code", "trade_date"]).sort_values("trade_date")
            if frame.empty or frame.adj_factor.isna().any():
                raise RuntimeError(f"Tushare ETF data incomplete: {code}")
            frame.to_parquet(path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metrics(returns: pd.Series) -> dict:
    returns = returns.fillna(0.0).astype(float)
    wealth = (1 + returns).cumprod()
    years = len(returns) / 252
    return {
        "annual_return": float(wealth.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "sharpe_zero_rf": float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() else None,
        "total_return": float(wealth.iloc[-1] - 1),
    }


def stock_signals() -> pd.DataFrame:
    basic = pd.read_parquet(
        ROOT / "data/research/long_dividend_quality/daily_basic_monthly_period_ends_v1.parquet",
        columns=["ts_code", "trade_date", "dv_ttm", "total_mv", "pe_ttm"],
    )
    basic = basic[(basic.trade_date >= "20181201") & (basic.trade_date <= END)].copy()
    basic["date"] = pd.to_datetime(basic.trade_date)
    basic["dv_ttm"] = pd.to_numeric(basic.dv_ttm, errors="coerce")
    fin = pd.read_parquet(ROOT / "data/raw/fina_indicator.parquet",
                          columns=["ts_code", "end_date", "ann_date", "roe", "basic_eps_yoy"])
    fin = fin[fin.ann_date.notna()].copy()
    fin["ann"] = pd.to_datetime(fin.ann_date)
    fin = fin.sort_values(["ts_code", "ann", "end_date"]).drop_duplicates(
        ["ts_code", "ann"], keep="last"
    )
    basic = pd.merge_asof(
        basic.sort_values(["date", "ts_code"]), fin.sort_values(["ann", "ts_code"]),
        left_on="date", right_on="ann", by="ts_code", direction="backward",
    )
    meta = pd.read_parquet(ROOT / "data/raw/stock_basic_history.parquet",
                           columns=["ts_code", "list_date"])
    meta["list_date"] = pd.to_datetime(meta.list_date, errors="coerce")
    basic = basic.merge(meta, on="ts_code", how="left")
    basic = basic[
        (basic.dv_ttm > 0) & (basic.dv_ttm < 15) & (basic.roe > 6)
        & (basic.basic_eps_yoy > 1) & (basic.pe_ttm > 0)
        & (basic.date - basic.list_date >= pd.Timedelta(days=365))
    ]
    selected = basic.sort_values(["date", "dv_ttm", "ts_code"], ascending=[True, False, True])
    return selected.groupby("date", sort=True).head(50)[["date", "ts_code", "dv_ttm", "roe", "basic_eps_yoy"]]


def stock_returns(symbols: set[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    base = ROOT / "data/raw/daily_partitioned"
    for path in sorted(base.glob("year_month=*/data.parquet")):
        ym = path.parent.name.split("=")[-1]
        if ym < "201901" or ym > "202609":
            continue
        part = pd.read_parquet(path, columns=["ts_code", "trade_date", "pct_chg"])
        part = part[part.ts_code.isin(symbols)]
        frames.append(part)
    if not frames:
        raise RuntimeError("No Tushare stock daily partitions")
    daily = pd.concat(frames, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    daily["date"] = pd.to_datetime(daily.trade_date)
    result = daily.pivot(index="date", columns="ts_code", values="pct_chg") / 100
    return result.reindex(dates)


def simulate(prices: pd.DataFrame, targets: dict[pd.Timestamp, dict[str, float]],
             *, tradable: pd.DataFrame | None = None) -> tuple[pd.Series, float, int]:
    """Signal at close t, rebalance at next close, earn new weights from t+2."""
    days = prices.index
    rets = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    weights: dict[str, float] = {}
    output = []
    turnover = 0.0
    count = 0
    pending: dict[str, float] | None = None
    for day in days:
        row = rets.loc[day]
        available = {s: w for s, w in weights.items() if s in row.index and pd.notna(row[s])}
        daily_ret = sum(w * float(row[s]) for s, w in available.items())
        if pending is not None:
            desired = {s: w for s, w in pending.items() if s in prices.columns
                       and (tradable is None or bool(tradable.at[day, s]))}
            trade = sum(abs(desired.get(s, 0) - weights.get(s, 0)) for s in set(desired) | set(weights))
            daily_ret -= FEE_ONE_WAY * trade
            turnover += trade
            count += 1
            weights = desired
            pending = None
        if day in targets:
            pending = targets[day]
        output.append(daily_ret)
    return pd.Series(output, index=days), turnover, count


def run(output: Path, token: str) -> dict:
    funds = fetch_etfs(output / "source", token)
    funds["date"] = pd.to_datetime(funds.trade_date)
    funds["adjusted_close"] = funds.close * funds.adj_factor
    etf = funds.pivot(index="date", columns="ts_code", values="adjusted_close").sort_index()
    etf = etf[(etf.index >= pd.Timestamp(START)) & (etf.index <= pd.Timestamp(END))].dropna()
    if len(etf) < 1000:
        raise RuntimeError("ETF common trading history too short")
    dates = etf.index
    close_to_close = etf.pct_change(fill_method=None).fillna(0)
    # Frozen rule: last trading day of each month, 20 trading-day winner; cash if all negative.
    monthly = etf.groupby(etf.index.to_period("M")).tail(1)
    mom = etf.pct_change(20).reindex(monthly.index)
    etf_targets = {}
    for date, row in mom.iterrows():
        if date >= dates[-1] or row.isna().any():
            continue
        winner = row.idxmax()
        etf_targets[date] = {winner: 1.0} if row.max() > 0 else {}
    etf_ret, etf_turnover, etf_rebalances = simulate(etf, etf_targets)
    benchmark_ret = close_to_close["510300.SH"].copy()

    signals = stock_signals()
    stock_targets = {}
    for day, group in signals.groupby("date"):
        if day in dates and len(group) >= 20:
            stock_targets[day] = {s: 1 / len(group) for s in group.ts_code}
    symbols = set(signals.ts_code)
    stock_daily = stock_returns(symbols, dates)
    # Convert ex-rights-adjusted daily returns into an index for the common simulator.
    stock_prices = (1 + stock_daily.fillna(0)).cumprod()
    stock_ret, stock_turnover, stock_rebalances = simulate(
        stock_prices, stock_targets, tradable=stock_daily.notna()
    )

    frame = pd.DataFrame({"dividend_quality_price_only": stock_ret,
                          "etf_momentum_adjusted": etf_ret,
                          "buy_hold_510300_adjusted": benchmark_ret})
    frame = frame.loc[frame.index >= pd.Timestamp("2019-02-01")]
    frame.iloc[0, frame.columns.get_loc("buy_hold_510300_adjusted")] -= FEE_ONE_WAY
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "daily_returns.csv", index_label="date")
    annual = (1 + frame).groupby(frame.index.year).prod() - 1
    annual.to_csv(output / "annual_returns.csv", index_label="year")
    summary = {
        "start": str(frame.index.min().date()), "end": str(frame.index.max().date()),
        "trading_days": len(frame), "one_way_cost": FEE_ONE_WAY,
        "metrics": {name: metrics(frame[name]) for name in frame},
        "turnover_notional": {"dividend_quality": stock_turnover, "etf_momentum": etf_turnover},
        "rebalances": {"dividend_quality": stock_rebalances, "etf_momentum": etf_rebalances},
        "signal_months": {"dividend_quality": len(stock_targets), "etf_momentum": len(etf_targets)},
        "source": "Tushare local daily_basic/fina_indicator/daily_partitioned; Tushare Pro fund_daily/fund_adj",
        "limitations": [
            "The column name dividend_quality_price_only is legacy: Tushare pct_chg uses ex-rights-adjusted pre_close and approximates reinvested distributions.",
            "Exact cash dividend payment timing, taxes, and explicit reinvestment are not modeled.",
            "Current stock_basic history lacks historical ST status; no point-in-time ST exclusion is applied.",
            "Stock missing-price days are marked flat; unavailable buys move to cash, but sells are assumed executable.",
            "Monthly signal uses financial data only after ann_date; ROE>6 and EPS YoY>1 proxy the public post's quality rules; PEG omitted.",
            "ETF fund_adj adjusts for distributions; same next-close lag and 0.10% per traded notional used for strategies.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/public_strategy_validation_20260929")
    args = parser.parse_args()
    token = os.getenv("TUSHARE_TOKEN") or dotenv_values(ROOT / ".env").get("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN missing")
    print(json.dumps(run(args.output, token), ensure_ascii=False, indent=2))
