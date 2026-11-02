"""Chronological, costed ETF strategy lab for return, hit rate and stability."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/strategy_lab_20260930"
END = "2026-09-24"
FEE = 0.001
CORE_CODES = ("510300.SH", "510500.SH", "159915.SZ", "159920.SZ",
              "513100.SH", "513500.SH", "518880.SH", "511010.SH")
RISKY_CODES = CORE_CODES[:7]
DOMESTIC_CODES = CORE_CODES[:3]


def load_prices(paths: list[Path]) -> pd.DataFrame:
    columns = []
    for path in paths:
        frame = pd.read_parquet(path)
        series = (frame.close * frame.adj_factor).astype(float)
        series.index = pd.to_datetime(frame.trade_date)
        series.name = str(frame.ts_code.iloc[0])
        columns.append(series)
    return pd.concat(columns, axis=1, sort=True).sort_index().dropna()


def core_prices() -> pd.DataFrame:
    paths = []
    for code in CORE_CODES:
        if code in DOMESTIC_CODES:
            paths.append(ROOT / "reports/public_strategy_validation_20260929/source"
                         / f"{code.replace('.', '_')}_2018_2026.parquet")
        else:
            paths.append(OUT / "source" / f"{code.replace('.', '_')}.parquet")
    return load_prices(paths)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.to_series().groupby(index.to_period("M")).last())


def capped_inverse_vol(vol: pd.Series, cap: float = 0.5) -> dict[str, float]:
    inverse = 1 / vol.clip(lower=0.005)
    weights = inverse / inverse.sum()
    fixed: dict[str, float] = {}
    free = list(weights.index)
    while free:
        remaining = 1 - sum(fixed.values())
        pool = inverse.loc[free]
        proposal = pool / pool.sum() * remaining
        excess = proposal[proposal > cap]
        if excess.empty:
            fixed.update(proposal.to_dict())
            break
        for code in excess.index:
            fixed[code] = cap
            free.remove(code)
    return fixed


def build_core_signals(prices: pd.DataFrame) -> dict[str, dict[pd.Timestamp, dict[str, float]]]:
    returns = prices.pct_change(fill_method=None)
    ma120 = prices.rolling(120, min_periods=120).mean()
    ma200 = prices.rolling(200, min_periods=200).mean()
    vol60 = returns.rolling(60, min_periods=60).std() * np.sqrt(252)
    signals: dict[str, dict[pd.Timestamp, dict[str, float]]] = {
        name: {} for name in (
            "buy_300", "buy_nasdaq", "buy_gold", "buy_bond",
            "mom120_top1", "mom120_top2", "mom240_top1", "risk_adj120_top1",
            "mom120_domestic_gate", "nasdaq_ma200_bond",
            "fixed_40_30_30", "invvol60", "invvol60_market_gate",
            "fixed_market_gate", "bond_gold_50_50", "fixed_25_25_50",
            "fixed_20_30_50", "fixed_nasdaq_25_gold25_bond50",
            "invvol60_nasdaq", "invvol60_nasdaq_trend",
            "nasdaq_gold_bond_30_30_40", "nasdaq_gold_switch",
        )
    }
    for day in month_ends(prices.index):
        if day < pd.Timestamp("2019-01-01"):
            continue
        for name, code in (("buy_300", "510300.SH"), ("buy_nasdaq", "513100.SH"),
                           ("buy_gold", "518880.SH"), ("buy_bond", "511010.SH")):
            signals[name][day] = {code: 1.0}
        market_on = bool(prices.at[day, "510300.SH"] > ma200.at[day, "510300.SH"])
        for lookback, name, top_n in ((120, "mom120_top1", 1), (120, "mom120_top2", 2),
                                      (240, "mom240_top1", 1), (120, "risk_adj120_top1", 1),
                                      (120, "mom120_domestic_gate", 1)):
            momentum = prices.loc[day, RISKY_CODES] / prices.shift(lookback).loc[day, RISKY_CODES] - 1
            eligible = (momentum > 0) & (prices.loc[day, RISKY_CODES] > ma120.loc[day, RISKY_CODES])
            if name == "mom120_domestic_gate" and not market_on:
                eligible.loc[list(DOMESTIC_CODES)] = False
            score = momentum / vol60.loc[day, RISKY_CODES] if name == "risk_adj120_top1" else momentum
            chosen = score[eligible & score.notna()].nlargest(top_n).index.tolist()
            signals[name][day] = {code: 1 / top_n for code in chosen}
        signals["nasdaq_ma200_bond"][day] = (
            {"513100.SH": 1.0}
            if prices.at[day, "513100.SH"] > ma200.at[day, "513100.SH"]
            else {"511010.SH": 1.0}
        )
        fixed = {"510300.SH": 0.4, "518880.SH": 0.3, "511010.SH": 0.3}
        signals["bond_gold_50_50"][day] = {"518880.SH": 0.5, "511010.SH": 0.5}
        signals["fixed_25_25_50"][day] = {
            "510300.SH": 0.25, "518880.SH": 0.25, "511010.SH": 0.5}
        signals["fixed_20_30_50"][day] = {
            "510300.SH": 0.2, "518880.SH": 0.3, "511010.SH": 0.5}
        signals["fixed_nasdaq_25_gold25_bond50"][day] = {
            "513100.SH": 0.25, "518880.SH": 0.25, "511010.SH": 0.5}
        signals["nasdaq_gold_bond_30_30_40"][day] = {
            "513100.SH": 0.3, "518880.SH": 0.3, "511010.SH": 0.4}
        nasdaq_ok = (prices.at[day, "513100.SH"] > ma200.at[day, "513100.SH"]
                     and prices.at[day, "513100.SH"] > prices.shift(120).at[day, "513100.SH"])
        gold_ok = prices.at[day, "518880.SH"] > ma120.at[day, "518880.SH"]
        signals["nasdaq_gold_switch"][day] = (
            {"513100.SH": 1.0} if nasdaq_ok else
            {"518880.SH": 1.0} if gold_ok else {"511010.SH": 1.0}
        )
        signals["fixed_40_30_30"][day] = fixed
        defensive = ["510300.SH", "518880.SH", "511010.SH"]
        scaled = capped_inverse_vol(vol60.loc[day, defensive])
        signals["invvol60"][day] = scaled
        scaled_nasdaq = capped_inverse_vol(vol60.loc[day, ["513100.SH", "518880.SH", "511010.SH"]])
        signals["invvol60_nasdaq"][day] = scaled_nasdaq
        signals["invvol60_nasdaq_trend"][day] = (
            scaled_nasdaq if prices.at[day, "513100.SH"] > ma200.at[day, "513100.SH"]
            else {"518880.SH": 0.5, "511010.SH": 0.5}
        )
        signals["invvol60_market_gate"][day] = scaled if market_on else {
            "518880.SH": 0.5, "511010.SH": 0.5}
        signals["fixed_market_gate"][day] = fixed if market_on else {
            "518880.SH": 0.5, "511010.SH": 0.5}
    return signals


@dataclass
class Run:
    daily: pd.DataFrame
    trades: pd.DataFrame
    turnover: float


def simulate_targets(prices: pd.DataFrame, targets: dict[pd.Timestamp, dict[str, float]],
                     *, fee: float = FEE, lag: int = 1) -> Run:
    dates = prices.index
    rets = prices.pct_change(fill_method=None).fillna(0)
    executions = {dates[dates.get_loc(signal) + lag]: target
                  for signal, target in targets.items()
                  if signal in dates and dates.get_loc(signal) + lag < len(dates)}
    weights: dict[str, float] = {}
    entries: dict[str, tuple[pd.Timestamp, float]] = {}
    wealth, turnover_sum = 1.0, 0.0
    rows, trades = [], []
    for day in dates:
        before = wealth
        gross = sum(weight * rets.at[day, code] for code, weight in weights.items())
        wealth *= 1 + gross
        weights = {code: weight * (1 + rets.at[day, code]) / (1 + gross)
                   for code, weight in weights.items()}
        if day in executions:
            desired = executions[day]
            for code, (entry_day, entry_price) in list(entries.items()):
                if code not in desired:
                    exit_price = prices.at[day, code]
                    trades.append({"code": code, "entry_date": entry_day, "exit_date": day,
                                   "net_return": exit_price / entry_price * (1 - fee) ** 2 - 1,
                                   "holding_days": int(dates.get_loc(day) - dates.get_loc(entry_day))})
                    entries.pop(code)
            for code in desired:
                if code not in entries:
                    entries[code] = (day, prices.at[day, code])
            turnover = sum(abs(weights.get(code, 0.0) - desired.get(code, 0.0))
                           for code in set(weights) | set(desired))
            turnover_sum += turnover
            wealth *= 1 - fee * turnover
            weights = desired.copy()
        rows.append({"date": day, "daily_return": wealth / before - 1,
                     "equity": wealth, "exposure": sum(weights.values()),
                     "holdings": ",".join(sorted(weights))})
    return Run(pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades), turnover_sum)


def pullback(prices: pd.DataFrame, *, dip: float, take_profit: float, max_hold: int,
             fee: float = FEE, lag: int = 1, rising_ma: bool = False,
             max_vol: float | None = None, stop_loss: float | None = None,
             candidates: tuple[str, ...] = ("510300.SH", "513100.SH", "518880.SH")) -> Run:
    dates = prices.index
    ma200 = prices.rolling(200, min_periods=200).mean()
    ret5 = prices.pct_change(5, fill_method=None)
    ret120 = prices.pct_change(120, fill_method=None)
    day_returns = prices.pct_change(fill_method=None).fillna(0)
    vol60 = day_returns.rolling(60, min_periods=60).std() * np.sqrt(252)
    position: str | None = None
    entry_date: pd.Timestamp | None = None
    entry_price = 0.0
    entry_index = -1
    execution: dict[int, tuple[str, str, str]] = {}
    wealth, turnover_sum = 1.0, 0.0
    rows, trades = [], []
    for i, day in enumerate(dates):
        before = wealth
        if position:
            wealth *= 1 + day_returns.at[day, position]
        just_exited = False
        if i in execution:
            action, code, reason = execution[i]
            if action == "exit" and position:
                net = prices.at[day, position] / entry_price * (1 - fee) ** 2 - 1
                trades.append({"code": position, "entry_date": entry_date, "exit_date": day,
                               "net_return": net, "holding_days": i - entry_index,
                               "exit_reason": reason})
                wealth *= 1 - fee
                turnover_sum += 1
                position, entry_date, entry_price, entry_index = None, None, 0.0, -1
                just_exited = True
            elif action == "enter" and not position:
                position, entry_date, entry_price, entry_index = code, day, prices.at[day, code], i
                wealth *= 1 - fee
                turnover_sum += 1
        pending = any(execution_day > i for execution_day in execution)
        if (i >= 200 and day >= pd.Timestamp("2019-01-01")
                and i + lag < len(dates) and not just_exited and not pending):
            if position:
                profit = prices.at[day, position] / entry_price - 1
                broken = prices.at[day, position] < ma200.at[day, position]
                stopped = stop_loss is not None and profit <= -stop_loss
                if profit >= take_profit or i - entry_index >= max_hold or broken or stopped:
                    reason = ("take_profit" if profit >= take_profit else
                              "trend_break" if broken else
                              "stop_loss" if stopped else "max_hold")
                    execution[i + lag] = ("exit", position, reason)
            else:
                eligible = [code for code in candidates
                            if prices.at[day, code] > ma200.at[day, code]
                            and ret120.at[day, code] > 0
                            and (not rising_ma or ma200.at[day, code] > ma200.iloc[i - 20][code])
                            and (max_vol is None or vol60.at[day, code] <= max_vol)
                            and -0.10 <= ret5.at[day, code] <= -dip]
                if eligible:
                    code = max(eligible, key=lambda item: ret120.at[day, item])
                    execution[i + lag] = ("enter", code, "pullback")
        rows.append({"date": day, "daily_return": wealth / before - 1,
                     "equity": wealth, "exposure": float(position is not None),
                     "holdings": position or ""})
    result = Run(pd.DataFrame(rows).set_index("date"), pd.DataFrame(trades), turnover_sum)
    return result


def metrics(series: pd.Series, exposure: pd.Series) -> dict[str, float | int]:
    series = series.fillna(0.0)
    wealth = (1 + series).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    monthly = (1 + series).groupby(series.index.to_period("M")).prod() - 1
    annual = (1 + series).groupby(series.index.year).prod() - 1
    annual = annual[series.groupby(series.index.year).size() >= 230]
    return {
        "annual_return": float(wealth.iloc[-1] ** (252 / len(series)) - 1),
        "max_drawdown": float((wealth / peak - 1).min()),
        "total_return": float(wealth.iloc[-1] - 1),
        "sharpe_zero_rf": float(series.mean() / series.std() * np.sqrt(252)) if series.std() else 0.0,
        "positive_month_rate": float((monthly > 0).mean()),
        "positive_months": int((monthly > 0).sum()),
        "months": int(len(monthly)),
        "worst_year": float(annual.min()),
        "annual_return_std": float(annual.std()) if len(annual) > 1 else 0.0,
        "positive_years": int((annual > 0).sum()),
        "years_count": int(len(annual)),
        "average_exposure": float(exposure.mean()),
    }


def summarize(name: str, run: Run, start: str, end: str) -> list[dict]:
    output = []
    for label, lo, hi in (("full", start, end), ("selection_2019_2022", "2019-02-01", "2022-12-31"),
                          ("review_2023_onward", "2023-01-01", end)):
        frame = run.daily.loc[lo:hi]
        if frame.empty:
            continue
        trade = run.trades
        if not trade.empty:
            trade = trade[(trade.entry_date >= frame.index.min()) &
                          (trade.exit_date <= frame.index.max())]
        output.append({"strategy": name, "period": label,
                       **metrics(frame.daily_return, frame.exposure),
                       "closed_trades": int(len(trade)),
                       "trade_win_rate": float((trade.net_return > 0).mean()) if len(trade) else None,
                       "mean_trade_return": float(trade.net_return.mean()) if len(trade) else None,
                       "mean_win": float(trade.loc[trade.net_return > 0, "net_return"].mean()) if len(trade) else None,
                       "mean_loss": float(trade.loc[trade.net_return <= 0, "net_return"].mean()) if len(trade) else None,
                       "turnover_full": run.turnover})
    return output


def sector_results(core: pd.DataFrame) -> tuple[list[dict], dict[str, Run]]:
    paths = sorted((ROOT / "reports/public_strategy_validation_20260929/sector_turnover_surge/source").glob("*.parquet"))
    paths = [path for path in paths if not path.stem.startswith(("159825", "515790"))]
    sector = load_prices(paths)
    common = sector.index.intersection(core.index)
    sector = sector.reindex(common)
    benchmark = core.reindex(common)["510300.SH"]
    ma120 = sector.rolling(120, min_periods=120).mean()
    ma200 = core["510300.SH"].rolling(200, min_periods=200).mean().reindex(common)
    ret120 = sector.pct_change(120, fill_method=None)
    signals = {"sector_mom120_top1": {}, "sector_mom120_top2": {},
               "sector_mom120_market_gate": {}}
    for day in month_ends(common):
        if day < pd.Timestamp("2021-03-01"):
            continue
        qualified = (ret120.loc[day] > 0) & (sector.loc[day] > ma120.loc[day])
        scores = ret120.loc[day][qualified].dropna()
        for name, n in (("sector_mom120_top1", 1), ("sector_mom120_top2", 2),
                        ("sector_mom120_market_gate", 1)):
            chosen = scores.nlargest(n).index.tolist()
            if name == "sector_mom120_market_gate" and benchmark.at[day] < ma200.at[day]:
                chosen = []
            signals[name][day] = {code: 1 / n for code in chosen}
    runs = {name: simulate_targets(sector, target) for name, target in signals.items()}
    comparison = simulate_targets(core.reindex(common), {
        day: {"510300.SH": 1.0} for day in month_ends(common)
        if day >= pd.Timestamp("2021-03-01")
    })
    runs["sector_window_buy_300"] = comparison
    rows = []
    for name, run in runs.items():
        for label, lo, hi in (("full", "2021-04-01", END),
                              ("review_2023_onward", "2023-01-01", END)):
            frame = run.daily.loc[lo:hi]
            rows.append({"strategy": name, "period": label,
                         **metrics(frame.daily_return, frame.exposure)})
    return rows, runs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = core_prices()
    signals = build_core_signals(prices)
    runs = {name: simulate_targets(prices, target) for name, target in signals.items()}
    runs["pullback_2pct_tp3_20d"] = pullback(prices, dip=0.02, take_profit=0.03, max_hold=20)
    runs["pullback_4pct_tp5_30d"] = pullback(prices, dip=0.04, take_profit=0.05, max_hold=30)
    runs["pullback_2pct_tp2_20d"] = pullback(prices, dip=0.02, take_profit=0.02, max_hold=20)
    runs["pullback_2pct_tp3_rising_ma"] = pullback(
        prices, dip=0.02, take_profit=0.03, max_hold=20, rising_ma=True)
    runs["pullback_2pct_tp3_vol35"] = pullback(
        prices, dip=0.02, take_profit=0.03, max_hold=20, max_vol=0.35)
    runs["pullback_2pct_tp3_stop7"] = pullback(
        prices, dip=0.02, take_profit=0.03, max_hold=20, stop_loss=0.07)
    runs["pullback_2pct_tp2_gold_300"] = pullback(
        prices, dip=0.02, take_profit=0.02, max_hold=20,
        candidates=("510300.SH", "518880.SH"))
    runs["pullback_1pct_tp2_gold_300"] = pullback(
        prices, dip=0.01, take_profit=0.02, max_hold=20,
        candidates=("510300.SH", "518880.SH"))
    runs["pullback_3pct_tp2_gold_300"] = pullback(
        prices, dip=0.03, take_profit=0.02, max_hold=20,
        candidates=("510300.SH", "518880.SH"))
    runs["pullback_4pct_tp2_gold_300"] = pullback(
        prices, dip=0.04, take_profit=0.02, max_hold=20,
        candidates=("510300.SH", "518880.SH"))
    runs["pullback_2pct_tp2_gold_only"] = pullback(
        prices, dip=0.02, take_profit=0.02, max_hold=20,
        candidates=("518880.SH",))
    rows = []
    for name, run in runs.items():
        rows.extend(summarize(name, run, "2019-02-01", END))
        run.daily.loc["2019-02-01":].to_csv(OUT / f"{name}_daily.csv", index_label="date")
        if not run.trades.empty:
            run.trades.to_csv(OUT / f"{name}_trades.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "core_comparison.csv", index=False)
    sector_rows, sector_runs = sector_results(prices)
    pd.DataFrame(sector_rows).to_csv(OUT / "sector_comparison.csv", index=False)
    for name, run in sector_runs.items():
        run.daily.loc["2021-04-01":].to_csv(OUT / f"{name}_daily.csv", index_label="date")
        if not run.trades.empty:
            run.trades.to_csv(OUT / f"{name}_trades.csv", index=False)
    metadata = {"source": "Tushare fund_daily and fund_adj, local caches",
                "core_symbols": list(CORE_CODES), "core_start": "2019-02-01",
                "core_end": END, "cost_per_traded_notional": FEE,
                "execution": "signal at close, execute at next close, earn new position from subsequent close",
                "selection_period": "2019-2022", "retrospective_review_period": "2023-2026-09-24",
                "core_trading_days": int(len(prices.loc["2019-02-01":END])),
                "missing_from_300_calendar": ["2021-02-08", "2022-01-13", "2022-03-29"],
                "caveat": "Review period was indirectly viewed in prior research and reused for adaptive tuning; not an untouched holdout"}
    (OUT / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    full = pd.DataFrame(rows)
    print(full[full.period == "full"][["strategy", "annual_return", "max_drawdown",
                                       "trade_win_rate", "closed_trades"]].to_string(index=False))
    print(pd.DataFrame(sector_rows).query("period=='full'")[
        ["strategy", "annual_return", "max_drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
