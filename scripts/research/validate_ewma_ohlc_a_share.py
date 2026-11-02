"""Reproduce the commodity correlation and EWMA/OHLC exit study.

Run from the repository root: python3 scripts/research/validate_ewma_ohlc_a_share.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/ewma_ohlc_a_share_20260923"
START = "2024-01-01"
END = "2026-09-23"
FEE_PER_SIDE = 0.0015
ALPHA = 2 / 21

# Business-exposure candidates fixed before computing correlations.
CANDIDATES = {
    "600547.SH": ("山东黄金", "黄金", "AU0_SHF"),
    "600489.SH": ("中金黄金", "黄金", "AU0_SHF"),
    "000975.SZ": ("山金国际", "黄金", "AU0_SHF"),
    "601899.SH": ("紫金矿业", "金铜混合", "AU0_SHF"),
    "600362.SH": ("江西铜业", "铜", "CU0_SHF"),
    "000630.SZ": ("铜陵有色", "铜", "CU0_SHF"),
    "601168.SH": ("西部矿业", "铜等有色", "CU0_SHF"),
    "603993.SH": ("洛阳钼业", "铜钴混合", "CU0_SHF"),
    "601600.SH": ("中国铝业", "铝", "AL0_SHF"),
    "000933.SZ": ("神火股份", "铝煤混合", "AL0_SHF"),
    "600938.SH": ("中国海油", "油气开采", "SCM_INE"),
    "601857.SH": ("中国石油", "油气综合", "SCM_INE"),
    "600028.SH": ("中国石化", "炼化综合", "SCM_INE"),
    "601808.SH": ("中海油服", "油服", "SCM_INE"),
    "002353.SZ": ("杰瑞股份", "油服设备", "SCM_INE"),
}


def load_stocks() -> pd.DataFrame:
    paths = sorted((ROOT / "data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
    pieces = []
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "name"]
    for path in paths:
        month = path.parent.name.split("=")[-1]
        if month < "202310" or month > "202609":
            continue
        frame = pd.read_parquet(path, columns=cols)
        pieces.append(frame[frame.ts_code.isin(CANDIDATES)])
    data = pd.concat(pieces, ignore_index=True)
    data["date"] = pd.to_datetime(data.trade_date, format="%Y%m%d")
    data = data.sort_values(["ts_code", "date"]).drop_duplicates(["ts_code", "date"], keep="last")
    return data


def adjust_and_signal(group: pd.DataFrame) -> pd.DataFrame:
    x = group.copy().sort_values("date").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "pre_close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    valid = (x[["open", "high", "low", "close", "pre_close"]] > 0).all(axis=1)
    valid &= (x.high >= x[["open", "close", "low"]].max(axis=1))
    valid &= (x.low <= x[["open", "close"]].min(axis=1))
    x = x[valid].copy().reset_index(drop=True)

    # Stitch ex-dividend/split pre-close into a synthetic adjusted OHLC series.
    # This approximates total return; it is not an audited cash-dividend series.
    factor = np.empty(len(x))
    factor[0] = 1.0
    raw_close = x.close.to_numpy()
    raw_pre = x.pre_close.to_numpy()
    for i in range(1, len(x)):
        factor[i] = factor[i - 1] * raw_close[i - 1] / raw_pre[i]
    for col in ("open", "high", "low", "close"):
        x["adj_" + col] = x[col].to_numpy() * factor

    returns = np.log(x.adj_close / x.adj_close.shift(1))
    x["sigma_prev"] = np.sqrt(returns.pow(2).ewm(alpha=ALPHA, adjust=False, min_periods=20).mean().shift(1))
    x["drawdown_z"] = np.log(x.high / x.close) / x.sigma_prev
    gk_var = 0.5 * np.log(x.high / x.low).pow(2) - (2 * np.log(2) - 1) * np.log(x.close / x.open).pow(2)
    x["gk_z"] = np.sqrt(gk_var.clip(lower=0)) / x.sigma_prev
    x["close_location"] = (x.close - x.low) / (x.high - x.low).replace(0, np.nan)
    x["A"] = x.drawdown_z > 1.5
    x["B"] = x.gk_z > 1.5
    x["C"] = x.close_location < 1 / 3
    x["trigger"] = x[["A", "B", "C"]].sum(axis=1) >= 2
    x["ret"] = returns
    return x[x.date.between(START, END)].copy().reset_index(drop=True)


def load_commodities() -> pd.DataFrame:
    a = pd.read_csv(OUT / "commodity_main_continuous.csv", parse_dates=["date"])
    b = pd.read_csv(OUT / "aluminum_main_continuous.csv", parse_dates=["date"])
    c = a.merge(b, on="date", how="outer", validate="one_to_one").sort_values("date").set_index("date")
    return np.log(c / c.shift(1)).rename(columns=lambda n: n + "_ret")


def correlation_row(x: pd.DataFrame, commodity: pd.DataFrame, benchmark: str) -> dict:
    joined = x.set_index("date")[["ret"]].join(commodity[[benchmark + "_ret"]], how="inner").dropna()
    joined.columns = ["stock", "commodity"]
    last120 = joined.tail(120)
    return {
        "aligned_days": len(joined),
        "corr_full": joined.stock.corr(joined.commodity),
        "corr_last120": last120.stock.corr(last120.commodity),
        "corr_2024_2025": joined.loc[:"2025-12-31"].stock.corr(joined.loc[:"2025-12-31"].commodity),
        "corr_2026": joined.loc["2026-01-01":].stock.corr(joined.loc["2026-01-01":].commodity),
    }


def backtest_monthly(x: pd.DataFrame, use_signal: bool, threshold: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = 1.0
    curves, trades = [], []
    data = x.copy()
    if threshold != 1.5:
        data["trigger"] = ((data.drawdown_z > threshold).astype(int) + (data.gk_z > threshold).astype(int) + data.C.astype(int)) >= 2
    for month, block in data.groupby(data.date.dt.to_period("M"), sort=True):
        block = block.reset_index(drop=True)
        if len(block) < 5:
            continue
        shares = equity * (1 - FEE_PER_SIDE) / block.at[0, "adj_open"]
        entry = block.at[0, "adj_open"]
        exit_at = None
        signal_at = None
        for i, row in block.iterrows():
            if exit_at == i:
                equity = shares * row.adj_open * (1 - FEE_PER_SIDE)
                shares = 0.0
                exit_at = None
            invested_today = int(shares > 0)
            if shares:
                mark = shares * row.adj_close
                if use_signal and i < len(block) - 1 and row.trigger and row.adj_close > entry and exit_at is None:
                    exit_at = i + 1
                    signal_at = row.date
                if i == len(block) - 1:
                    equity = mark * (1 - FEE_PER_SIDE)
                    shares = 0.0
                    mark = equity
            else:
                mark = equity
            curves.append({"date": row.date, "equity": mark, "invested": invested_today})
        exit_i = next((i for i, row in block.iterrows() if signal_at is not None and row.date > signal_at), None)
        exit_price = block.at[exit_i, "adj_open"] if exit_i is not None else block.iloc[-1].adj_close
        trades.append({
            "month": str(month), "entry_date": str(block.iloc[0].date.date()), "entry_price": entry,
            "exit_date": str(block.at[exit_i, "date"].date()) if exit_i is not None else str(block.iloc[-1].date.date()),
            "exit_price": exit_price, "signal_date": str(signal_at.date()) if signal_at is not None else "",
            "trade_return_net": (exit_price / entry) * (1 - FEE_PER_SIDE) ** 2 - 1,
        })
    return pd.DataFrame(curves).drop_duplicates("date", keep="last"), pd.DataFrame(trades)


def stats(curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
    if curve.empty:
        return {}
    daily = curve.equity.pct_change().fillna(curve.equity.iloc[0] - 1)
    years = (curve.date.iloc[-1] - curve.date.iloc[0]).days / 365.25
    return {
        "total_return": curve.equity.iloc[-1] - 1,
        "cagr": curve.equity.iloc[-1] ** (1 / years) - 1,
        "max_drawdown": (curve.equity / curve.equity.cummax() - 1).min(),
        "sharpe": np.sqrt(252) * daily.mean() / daily.std() if daily.std() > 0 else np.nan,
        "exposure": curve.invested.mean(),
        "months": len(trades),
        "signal_exits": trades.signal_date.ne("").sum(),
    }


def main() -> None:
    raw = load_stocks()
    commodities = load_commodities()
    results, all_trades, all_signals, sensitivities = [], [], [], []
    for code, (name, group, benchmark) in CANDIDATES.items():
        stock = adjust_and_signal(raw[raw.ts_code.eq(code)])
        corr = correlation_row(stock, commodities, benchmark)
        base_curve, base_trades = backtest_monthly(stock, False)
        rule_curve, rule_trades = backtest_monthly(stock, True)
        b, r = stats(base_curve, base_trades), stats(rule_curve, rule_trades)
        fixed75_curve = base_curve.copy()
        baseline_daily = base_curve.equity.pct_change().fillna(base_curve.equity.iloc[0] - 1)
        fixed75_curve["equity"] = (1 + 0.75 * baseline_daily).cumprod()
        fixed75_curve["invested"] = 0.75
        f = stats(fixed75_curve, base_trades)
        for label, trades in (("base", base_trades), ("rule", rule_trades)):
            for period, mask in (("2024_2025", trades.month.str[:4].isin(["2024", "2025"])),
                                 ("2026_ytd", trades.month.str[:4].eq("2026"))):
                period_return = (1 + trades.loc[mask, "trade_return_net"]).prod() - 1
                if label == "base":
                    b["return_" + period] = period_return
                else:
                    r["return_" + period] = period_return
        for threshold in (1.25, 1.5, 1.75):
            candidate_curve, candidate_trades = backtest_monthly(stock, True, threshold)
            candidate_stats = stats(candidate_curve, candidate_trades)
            sensitivities.append({"code": code, "name": name, "threshold": threshold,
                                  "total_return": candidate_stats["total_return"],
                                  "max_drawdown": candidate_stats["max_drawdown"],
                                  "signal_exits": candidate_stats["signal_exits"]})
        result = {"code": code, "name": name, "group": group, "benchmark": benchmark,
                  "first_date": str(stock.date.min().date()), "last_date": str(stock.date.max().date()),
                  "stock_days": len(stock), "trigger_days": int(stock.trigger.sum()), **corr,
                  **{"base_" + k: v for k, v in b.items()}, **{"rule_" + k: v for k, v in r.items()},
                  **{"fixed75_" + k: v for k, v in f.items()}}
        results.append(result)
        rule_trades.insert(0, "code", code)
        rule_trades.insert(1, "name", name)
        all_trades.append(rule_trades)
        triggered = stock[stock.trigger].copy()
        triggered.insert(0, "code", code)
        all_signals.append(triggered[["code", "date", "drawdown_z", "gk_z", "close_location", "A", "B", "C"]])
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "results.csv", index=False)
    pd.concat(all_trades).to_csv(OUT / "trades.csv", index=False)
    pd.concat(all_signals).to_csv(OUT / "signals.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(OUT / "sensitivity.csv", index=False)
    summary = {
        "stocks": len(result_df),
        "period": [START, END],
        "commodity_rows": len(commodities),
        "median_corr_full": result_df.corr_full.median(),
        "median_corr_last120": result_df.corr_last120.median(),
        "median_return_delta": (result_df.rule_total_return - result_df.base_total_return).median(),
        "median_drawdown_delta": (result_df.rule_max_drawdown - result_df.base_max_drawdown).median(),
        "improved_return_count": int((result_df.rule_total_return > result_df.base_total_return).sum()),
        "improved_drawdown_count": int((result_df.rule_max_drawdown > result_df.base_max_drawdown).sum()),
        "rule_vs_fixed75_return_count": int((result_df.rule_total_return > result_df.fixed75_total_return).sum()),
        "rule_vs_fixed75_drawdown_count": int((result_df.rule_max_drawdown > result_df.fixed75_max_drawdown).sum()),
        "signal_exits": int(result_df.rule_signal_exits.sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n")
    print(result_df[["name", "benchmark", "corr_full", "corr_last120", "base_total_return", "rule_total_return", "base_max_drawdown", "rule_max_drawdown", "rule_signal_exits"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
