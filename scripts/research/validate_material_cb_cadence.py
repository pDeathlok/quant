"""Tushare convertible-bond double-low cadence study with one-bar signal lag."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/public_strategy_validation_20260929/material_backtests"
SOURCE = ROOT / "data/convertible_bond/tushare/cb_daily_20180101_20260616.parquet"


def metrics(returns: pd.Series) -> dict[str, float]:
    wealth = (1 + returns).cumprod()
    years = len(returns) / 252
    return {
        "annual_return": float(wealth.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "total_return": float(wealth.iloc[-1] - 1),
        "sharpe_zero_rf": float(returns.mean() / returns.std() * np.sqrt(252)),
    }


def prepare() -> tuple[list[str], dict[str, dict[str, float]], dict[str, list[str]], dict[str, str]]:
    frame = pd.read_parquet(SOURCE, columns=["ts_code", "trade_date", "close", "amount"])
    frame = frame[(frame.trade_date >= "20180101") & (frame.trade_date <= "20260616")].copy()
    frame = frame.drop_duplicates(["trade_date", "ts_code"])
    premium = pd.concat([
        pd.read_parquet(OUT / f"source/cb_over_rate_{year}.parquet")
        for year in range(2018, 2027)
    ], ignore_index=True).drop_duplicates(["trade_date", "ts_code"])
    frame = frame.merge(premium, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    frame["close"] = pd.to_numeric(frame.close, errors="coerce")
    frame["cb_over_rate"] = pd.to_numeric(frame.cb_over_rate, errors="coerce")
    frame["amount"] = pd.to_numeric(frame.amount, errors="coerce")
    # Tushare includes zero-close terminal rows for some retired bonds; zero is not a fillable market quote.
    frame = frame.dropna(subset=["close"])
    frame = frame[frame.close > 0]
    all_dates = sorted(frame.trade_date.unique())
    prices = {day: dict(zip(group.ts_code, group.close)) for day, group in frame.groupby("trade_date")}
    eligible = frame[
        frame.close.between(80, 140) & frame.cb_over_rate.notna()
        & (frame.cb_over_rate >= -100) & (frame.amount >= 1000)
    ].copy()
    eligible["double_low"] = eligible.close + eligible.cb_over_rate
    ranked = {day: group.sort_values(["double_low", "ts_code"]).ts_code.tolist()
              for day, group in eligible.groupby("trade_date")}
    basic = pd.read_parquet(ROOT / "data/convertible_bond/tushare/cb_basic_all.parquet",
                            columns=["ts_code", "delist_date"])
    delist = dict(zip(basic.ts_code, basic.delist_date.fillna("")))
    return all_dates, prices, ranked, delist


def simulate(
    dates: list[str], prices: dict[str, dict[str, float]], ranked: dict[str, list[str]],
    delist: dict[str, str],
    top_n: int, cadence: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    weights: dict[str, float] = {}
    last_price: dict[str, float] = {}
    wealth = 1.0
    rows = []
    traded_notional, trades, missing_held_days, assumed_delist_exits = 0.0, 0, 0, 0
    for i, day in enumerate(dates):
        close = prices[day]
        before = wealth
        asset_returns = {code: (close[code] / last_price[code] - 1)
                         if code in close and code in last_price and last_price[code] > 0 else 0.0
                         for code in weights}
        gross = sum(weight * asset_returns[code] for code, weight in weights.items())
        wealth *= 1 + gross
        weights = {code: weight * (1 + asset_returns[code]) / (1 + gross)
                   for code, weight in weights.items()}
        # Approximate proceeds at the last quote when a bond leaves the exchange.
        for code in list(weights):
            if code not in close and delist.get(code, "") and day > delist[code]:
                traded_notional += weights[code]
                wealth *= 1 - 0.0004 * weights[code]
                weights.pop(code)
                assumed_delist_exits += 1
        suspended = {code: weight for code, weight in weights.items() if code not in close}
        missing_held_days += len(suspended)
        # Date i uses the ranking available only after the previous close.
        if i >= 20 and (i - 20) % cadence == 0:
            signal_day = dates[i - 1]
            chosen = ranked.get(signal_day, [])[:top_n]
            target = {code: 1 / top_n for code in chosen if code in close}
            room = max(0.0, 1.0 - sum(suspended.values()))
            if sum(target.values()) > room:
                target = {code: weight * room / sum(target.values()) for code, weight in target.items()}
            target.update(suspended)
            turnover = sum(abs(target.get(code, 0.0) - weights.get(code, 0.0))
                           for code in set(target) | set(weights))
            wealth *= 1 - 0.0004 * turnover
            traded_notional += turnover
            trades += sum(abs(target.get(code, 0.0) - weights.get(code, 0.0)) > 1e-10
                          for code in set(target) | set(weights))
            weights = target
        rows.append({"trade_date": day, "daily_return": wealth / before - 1, "equity": wealth,
                     "positions": len(weights)})
        last_price.update(close)
    stats = {"turnover_notional": traded_notional, "trade_count": trades,
             "missing_held_symbol_days": missing_held_days,
             "assumed_delist_exits": assumed_delist_exits}
    return pd.DataFrame(rows).iloc[20:].reset_index(drop=True), stats


def run() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    dates, prices, ranked, delist = prepare()
    rows = []
    recent_rows = []
    for top_n in (10, 15):
        for name, cadence in (("daily", 1), ("weekly", 5), ("biweekly", 10), ("monthly", 21)):
            equity, stats = simulate(dates, prices, ranked, delist, top_n, cadence)
            equity.to_csv(OUT / f"cb_pure_top{top_n}_{name}_equity.csv", index=False)
            rows.append({"top_n": top_n, "cadence": name, "start_date": equity.trade_date.iloc[0],
                         "end_date": equity.trade_date.iloc[-1],
                         **metrics(equity.daily_return), **stats})
            recent = equity[equity.trade_date >= "20230101"]
            recent_rows.append({"top_n": top_n, "cadence": name,
                                "start_date": recent.trade_date.iloc[0],
                                "end_date": recent.trade_date.iloc[-1],
                                **metrics(recent.daily_return)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "cb_cadence_comparison.csv", index=False)
    pd.DataFrame(recent_rows).to_csv(OUT / "cb_cadence_recent_comparison.csv", index=False)
    (OUT / "cb_cadence_notes.json").write_text(json.dumps({
        "source": str(SOURCE),
        "selection": "Rank previous-close bonds by close + cb_over_rate (conversion premium percent); top 10 or 15; price 80-140, amount >= 1000 (Tushare amount units), valid premium",
        "execution": "Rebalance every 1/5/10/21 bond sessions at current close using previous-close ranking, 4bp per traded notional, vacant slots cash",
        "return_basis": "Raw bond close-to-close; coupon/redemption payments are not added",
        "limitations": ["Suspended bonds mark at last close until quotation resumes", "Delisted bonds are assumed to exit at their last quote on the first subsequent session", "No limit-order execution or market impact", "Coupon, call and redemption cashflows absent", "No independent untouched holdout"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(run().to_string(index=False))
