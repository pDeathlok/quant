#!/usr/bin/env python
"""Full monthly-ladder backtest for the selected V2 long-entry model.

The one-year sleeve recycle is a standardized measurement horizon, not a sell
recommendation.  Existing holdings remain a separate self-pool workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quant.research.long_entry_v2 import (
    equity_metrics,
    month_end_week_mask,
    select_industry_capped,
)


REPORT_DIR = PROJECT_ROOT / "reports/long_entry_model_v2"
INDEX_PATH = PROJECT_ROOT / "data/raw/index_000300.SH.parquet"
STOCK_BASIC_PATH = PROJECT_ROOT / "data/raw/stock_basic_history.parquet"
DAILY_CACHE_DIR = PROJECT_ROOT / "data/research/long_dividend_quality"


@dataclass
class Position:
    ts_code: str
    shares: float
    anchor_price: float
    base_budget: float
    entry_calendar_index: int
    filled_5: bool = False
    filled_10: bool = False
    delist_counted: bool = False


@dataclass
class Sleeve:
    cash: float
    positions: list[Position] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buy-cost", type=float, default=0.0010)
    parser.add_argument("--sell-cost", type=float, default=0.0020)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--start", default="2020-01-01")
    return parser.parse_args()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def find_full_daily_cache() -> Path:
    candidates: list[tuple[int, pd.Timestamp, Path]] = []
    for path in DAILY_CACHE_DIR.glob("daily_returns_*.parquet"):
        dates = pd.read_parquet(path, columns=["date"])["date"]
        if dates.empty or pd.Timestamp(dates.min()).year > 2020:
            continue
        candidates.append((len(dates), pd.Timestamp(dates.max()), path))
    if not candidates:
        raise FileNotFoundError("no full daily return cache found")
    return max(candidates, key=lambda item: (item[1], item[0]))[2]


def load_prices(path: Path, symbols: set[str]) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(str(path), format="parquet")
        frame = dataset.to_table(
            columns=["date", "ts_code", "close"],
            filter=ds.field("ts_code").isin(sorted(symbols)),
        ).to_pandas()
    except Exception:
        frame = pd.read_parquet(path, columns=["date", "ts_code", "close"])
        frame = frame[frame["ts_code"].astype(str).isin(symbols)]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna().drop_duplicates(["date", "ts_code"], keep="last").sort_values(["ts_code", "date"])


def build_price_lookup(prices: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, dict[np.datetime64, float]]]:
    lookup = {}
    for code, group in prices.groupby("ts_code", sort=False):
        dates = group["date"].to_numpy(dtype="datetime64[ns]")
        closes = group["close"].to_numpy(dtype=float)
        exact = {date: float(close) for date, close in zip(dates, closes, strict=True)}
        lookup[str(code)] = (dates, closes, exact)
    return lookup


def exact_price(lookup: dict, code: str, date: pd.Timestamp) -> float:
    series = lookup.get(str(code))
    if series is None:
        return np.nan
    return float(series[2].get(np.datetime64(date, "ns"), np.nan))


def last_price(lookup: dict, code: str, date: pd.Timestamp) -> float:
    series = lookup.get(str(code))
    if series is None:
        return np.nan
    dates, closes, _ = series
    index = int(np.searchsorted(dates, np.datetime64(date, "ns"), side="right")) - 1
    return float(closes[index]) if index >= 0 else np.nan


def load_delist_dates() -> dict[str, pd.Timestamp]:
    if not STOCK_BASIC_PATH.exists():
        return {}
    frame = pd.read_parquet(STOCK_BASIC_PATH, columns=["ts_code", "delist_date"])
    frame["delist_date"] = pd.to_datetime(frame["delist_date"].astype(str), format="%Y%m%d", errors="coerce")
    return dict(frame.dropna(subset=["delist_date"])[["ts_code", "delist_date"]].itertuples(index=False, name=None))


def apply_price_guardrail(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "none":
        return frame
    historical_value = pd.to_numeric(frame["historical_value_score_5y"], errors="coerce")
    pr = pd.to_numeric(frame["pr"], errors="coerce")
    trend = pd.to_numeric(frame["close_to_ma120"], errors="coerce")
    momentum = pd.to_numeric(frame["return_120d_cross_section_pct"], errors="coerce")
    if name == "historical_value_50":
        mask = historical_value.ge(50)
    elif name == "value_or_absolute_trend":
        mask = (historical_value.ge(55) | pr.between(0, 2, inclusive="neither")) & trend.ge(-0.10)
    elif name == "momentum_and_value":
        mask = momentum.ge(0.40) & (historical_value.ge(50) | pr.between(0, 2, inclusive="neither"))
    else:
        raise ValueError(name)
    return frame[mask].copy()


def build_strategies(start: pd.Timestamp) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads((REPORT_DIR / "experiment_manifest.json").read_text(encoding="utf-8"))
    chosen = manifest["chosen"]
    predictions = pd.read_parquet(REPORT_DIR / "walk_forward_predictions.parquet")
    all_good_stocks = predictions[predictions["candidate"].eq(chosen["candidate"])].copy()
    all_good_stocks = all_good_stocks[
        month_end_week_mask(all_good_stocks) & all_good_stocks["date"].ge(start)
    ]
    chosen_frame = apply_price_guardrail(
        all_good_stocks, str(chosen.get("guardrail", "none"))
    )
    model = select_industry_capped(
        chosen_frame,
        score_column="entry_score",
        top_n=20,
        max_per_industry=int(chosen["industry_cap"]),
    )
    quality = select_industry_capped(
        chosen_frame,
        score_column="good_stock_score",
        top_n=20,
        max_per_industry=int(chosen["industry_cap"]),
    )

    v1_path = PROJECT_ROOT / "reports/long_entry_model_v1/walk_forward_predictions.parquet"
    v1 = pd.read_parquet(v1_path)
    v1 = v1[month_end_week_mask(v1) & v1["date"].ge(start)]
    v1_score = "calibrated_score" if "calibrated_score" in v1.columns else "entry_score"
    v1_selected = select_industry_capped(
        v1,
        score_column=v1_score,
        top_n=20,
        max_per_industry=int(chosen["industry_cap"]),
    )
    common_dates = set(v1_selected["date"])
    return {
        "v2_model": model,
        "v2_model_common_v1_dates": model[model["date"].isin(common_dates)].copy(),
        "all_good_stocks": all_good_stocks,
        "guardrail_good_stocks": chosen_frame,
        "good_stock_score": quality,
        "v1_model_common_dates": v1_selected,
    }, manifest


def run_ladder(
    picks: pd.DataFrame,
    *,
    lookup: dict,
    calendar: pd.DatetimeIndex,
    delist_dates: dict[str, pd.Timestamp],
    slots: int,
    buy_cost: float,
    sell_cost: float,
    staged: bool,
    conservative_delist: bool,
) -> tuple[pd.DataFrame, dict]:
    schedule: dict[pd.Timestamp, pd.DataFrame] = {}
    market = calendar.to_numpy(dtype="datetime64[ns]")
    for signal_date, group in picks.groupby("date", sort=True):
        index = int(np.searchsorted(market, np.datetime64(signal_date), side="right"))
        if index < len(market):
            schedule[pd.Timestamp(market[index])] = group
    if not schedule:
        return pd.DataFrame(), {}
    sleeves = [Sleeve(cash=1.0 / slots) for _ in range(slots)]
    first_date = min(schedule)
    last_date = calendar.max()
    active_calendar = calendar[(calendar >= first_date) & (calendar <= last_date)]
    schedule_counter = 0
    total_cost = 0.0
    buy_turnover = 0.0
    sell_turnover = 0.0
    skipped_entries = 0
    delisted_positions = 0
    rows: list[dict] = []

    def marked_value(position: Position, date: pd.Timestamp) -> float:
        delist = delist_dates.get(position.ts_code)
        if conservative_delist and delist is not None and date >= delist:
            return 0.0
        price = last_price(lookup, position.ts_code, date)
        return position.shares * price if np.isfinite(price) else 0.0

    for calendar_index, date in enumerate(active_calendar):
        date = pd.Timestamp(date)
        if date in schedule:
            sleeve = sleeves[schedule_counter % slots]
            schedule_counter += 1
            liquidation = sleeve.cash
            for position in sleeve.positions:
                value = marked_value(position, date)
                fee = value * sell_cost
                total_cost += fee
                sell_turnover += value
                liquidation += value - fee
            sleeve.cash = liquidation
            sleeve.positions = []
            selected = schedule[date]
            allocation = liquidation / len(selected) if len(selected) else 0.0
            for row in selected.itertuples(index=False):
                code = str(row.ts_code)
                price = exact_price(lookup, code, date)
                if not np.isfinite(price) or price <= 0:
                    skipped_entries += 1
                    continue
                initial_fraction = 0.50 if staged else 1.0
                budget = allocation * initial_fraction
                shares = budget / (price * (1.0 + buy_cost))
                fee = budget - shares * price
                total_cost += fee
                buy_turnover += budget
                sleeve.cash -= budget
                sleeve.positions.append(
                    Position(
                        ts_code=code,
                        shares=shares,
                        anchor_price=price,
                        base_budget=allocation,
                        entry_calendar_index=calendar_index,
                    )
                )

        if staged:
            for sleeve in sleeves:
                for position in sleeve.positions:
                    if calendar_index - position.entry_calendar_index > 63:
                        continue
                    price = exact_price(lookup, position.ts_code, date)
                    if not np.isfinite(price):
                        continue
                    for threshold, attribute in ((0.95, "filled_5"), (0.90, "filled_10")):
                        if getattr(position, attribute) or price > position.anchor_price * threshold:
                            continue
                        budget = min(position.base_budget * 0.25, sleeve.cash)
                        if budget <= 0:
                            continue
                        extra_shares = budget / (price * (1.0 + buy_cost))
                        fee = budget - extra_shares * price
                        total_cost += fee
                        buy_turnover += budget
                        sleeve.cash -= budget
                        position.shares += extra_shares
                        setattr(position, attribute, True)

        invested = 0.0
        active_positions = 0
        for sleeve in sleeves:
            for position in sleeve.positions:
                delist = delist_dates.get(position.ts_code)
                if conservative_delist and delist is not None and date >= delist and not position.delist_counted:
                    delisted_positions += 1
                    position.delist_counted = True
                invested += marked_value(position, date)
                active_positions += 1
        cash = sum(sleeve.cash for sleeve in sleeves)
        rows.append(
            {
                "date": date,
                "equity": cash + invested,
                "cash": cash,
                "invested": invested,
                "invested_ratio": invested / (cash + invested) if cash + invested > 0 else 0.0,
                "positions": active_positions,
            }
        )
    diagnostics = {
        "signals": int(picks["date"].nunique()),
        "selected_rows": int(len(picks)),
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "total_cost": total_cost,
        "skipped_entries": skipped_entries,
        "delisted_positions": delisted_positions,
        "average_invested_ratio": float(pd.DataFrame(rows)["invested_ratio"].mean()),
        "staged": staged,
        "conservative_delist": conservative_delist,
    }
    return pd.DataFrame(rows), diagnostics


def add_market_regime(equity: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    market = index.copy().sort_values("date")
    close = market["close"]
    market["ma120"] = close.rolling(120).mean()
    market["ma120_slope20"] = market["ma120"] / market["ma120"].shift(20) - 1
    market["market_regime"] = "neutral"
    market.loc[(close > market["ma120"]) & (market["ma120_slope20"] > 0), "market_regime"] = "risk_on"
    market.loc[(close < market["ma120"]) & (market["ma120_slope20"] < 0), "market_regime"] = "risk_off"
    return equity.merge(market[["date", "market_regime"]], on="date", how="left")


def regime_metrics(equity: pd.DataFrame) -> pd.DataFrame:
    frame = equity.copy()
    frame["daily_return"] = frame["equity"].pct_change()
    rows = []
    for regime, group in frame.groupby("market_regime", dropna=False):
        ret = group["daily_return"].dropna()
        rows.append(
            {
                "market_regime": regime,
                "days": len(ret),
                "annualized_return_approx": float(ret.mean() * 252),
                "annualized_volatility": float(ret.std() * np.sqrt(252)),
                "positive_day_rate": float(ret.gt(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def yearly_returns(equity: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = equity.copy().sort_values("date")
    frame["year"] = frame["date"].dt.year
    rows = []
    for year, group in frame.groupby("year"):
        rows.append(
            {
                "strategy": name,
                "year": int(year),
                "return": float(group["equity"].iloc[-1] / group["equity"].iloc[0] - 1),
                "max_drawdown": float((group["equity"] / group["equity"].cummax() - 1).min()),
            }
        )
    return pd.DataFrame(rows)


def build_report(summary: pd.DataFrame, yearly: pd.DataFrame, regimes: pd.DataFrame, daily_path: Path) -> str:
    lines = [
        "# 长线好价格 V2：完整资金回测",
        "",
        "## 口径",
        "",
        "- 每月末生成一次推荐，下一交易日进入一个独立资金袖套；12个袖套轮换，约一年后复用该袖套。这个一年周期只是统一评价窗口，不是页面卖出建议。",
        "- 即时方案一次建满；分批方案为50%首笔、未来63个市场交易日收盘较首笔低5%/10%时各加25%，未触发资金保持现金。",
        "- 成本假设：买入10bp、卖出20bp（研究敏感性参数，不声称复刻每个历史时点的真实费率）。前复权收盘价估值。",
        "- 主结果对信号后退市且无法继续交易的持仓按价值归零；同时给出持有最后收盘价的宽松敏感性。历史ST/涨跌停可交易性数据不完整，尚未模拟。",
        "",
        "## 总体结果",
        "",
        "| 策略 | 总收益 | 年化 | 年化波动 | Sharpe(0利率) | 最大回撤 | 平均投入 | 成本 | 跳过成交 | 退市归零持仓 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {row.total_return:.2%} | {row.annual_return:.2%} | {row.annual_volatility:.2%} | {row.sharpe_zero_rf:.2f} | {row.max_drawdown:.2%} | {row.average_invested_ratio:.2%} | {row.total_cost:.4f} | {int(row.skipped_entries)} | {int(row.delisted_positions)} |"
        )
    pivot = yearly.pivot(index="year", columns="strategy", values="return").sort_index()
    lines.extend(["", "## 分年收益", "", pivot.to_markdown(floatfmt=".2%"), "", "## V2即时方案的市场状态", "", regimes.to_markdown(index=False, floatfmt=".4f"), "", "## 解释", ""])
    lines.extend(
        [
            "- `all_good_stocks_equal_weight` 是每月全部好股票等权基线；`guardrail_good_stocks_equal_weight` 是通过历史价值分护栏后的等权基线；`good_stock_score` 是护栏内只按质量分前20。",
            "- V1 只对有完整未来标签的日期生成过分数，因此仅比较 `v2_model_common_v1_dates` 与 `v1_model_common_dates`；二者使用完全相同的信号月份。",
            "- `csi300` 为同期指数价格基准；组合使用前复权价，未单独重建现金分红流。",
            "- 分批方案的目标是降低初始时点风险，不保证提高收益；应同时看最大回撤、现金占用和收益，而不能只看最终收益。",
            "- 2020–2023 是模型选择期，2024+ 是已复用诊断期；整段资金曲线不是一次全新独立检验。",
            f"- 使用日线缓存：`{daily_path}`。这是研究回测，不构成投资建议。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    strategies, manifest = build_strategies(pd.Timestamp(args.start))
    symbols = set().union(*(set(frame["ts_code"].astype(str)) for frame in strategies.values()))
    daily_path = find_full_daily_cache()
    prices = load_prices(daily_path, symbols)
    lookup = build_price_lookup(prices)
    index = pd.read_parquet(INDEX_PATH)
    index["date"] = pd.to_datetime(index["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    index["close"] = pd.to_numeric(index["close"], errors="coerce")
    index = index.dropna(subset=["date", "close"]).sort_values("date")
    calendar = pd.DatetimeIndex(index["date"].unique()).sort_values()
    calendar = calendar[calendar <= prices["date"].max()]
    delist_dates = load_delist_dates()

    variants = {
        "v2_immediate_conservative": (strategies["v2_model"], False, True),
        "v2_staged_50_25_25_conservative": (strategies["v2_model"], True, True),
        "v2_immediate_last_close_sensitivity": (strategies["v2_model"], False, False),
        "all_good_stocks_equal_weight": (strategies["all_good_stocks"], False, True),
        "guardrail_good_stocks_equal_weight": (strategies["guardrail_good_stocks"], False, True),
        "good_stock_score_immediate": (strategies["good_stock_score"], False, True),
        "v2_model_common_v1_dates": (strategies["v2_model_common_v1_dates"], False, True),
        "v1_model_common_dates": (strategies["v1_model_common_dates"], False, True),
    }
    summary_rows: list[dict] = []
    yearly_frames: list[pd.DataFrame] = []
    regime = pd.DataFrame()
    diagnostics: dict[str, dict] = {}
    for name, (picks, staged, conservative) in variants.items():
        print(f"backtesting {name}", flush=True)
        equity, detail = run_ladder(
            picks,
            lookup=lookup,
            calendar=calendar,
            delist_dates=delist_dates,
            slots=args.slots,
            buy_cost=args.buy_cost,
            sell_cost=args.sell_cost,
            staged=staged,
            conservative_delist=conservative,
        )
        equity = add_market_regime(equity, index)
        _atomic_csv(equity, REPORT_DIR / f"equity_{name}.csv")
        metrics = equity_metrics(equity)
        summary_rows.append({"strategy": name, **metrics, **detail})
        yearly_frames.append(yearly_returns(equity, name))
        diagnostics[name] = detail
        if name == "v2_immediate_conservative":
            regime = regime_metrics(equity)
    first_backtest_date = pd.Timestamp(summary_rows[0]["start"])
    benchmark = index[index["date"].between(first_backtest_date, calendar.max())][["date", "close"]].copy()
    benchmark["equity"] = benchmark["close"] / benchmark["close"].iloc[0]
    benchmark["cash"] = 0.0
    benchmark["invested"] = benchmark["equity"]
    benchmark["invested_ratio"] = 1.0
    benchmark["positions"] = 1
    benchmark = add_market_regime(benchmark.drop(columns=["close"]), index)
    _atomic_csv(benchmark, REPORT_DIR / "equity_csi300.csv")
    benchmark_metrics = equity_metrics(benchmark)
    benchmark_detail = {
        "signals": 0,
        "selected_rows": 0,
        "buy_turnover": 0.0,
        "sell_turnover": 0.0,
        "total_cost": 0.0,
        "skipped_entries": 0,
        "delisted_positions": 0,
        "average_invested_ratio": 1.0,
        "staged": False,
        "conservative_delist": False,
    }
    summary_rows.append({"strategy": "csi300", **benchmark_metrics, **benchmark_detail})
    yearly_frames.append(yearly_returns(benchmark, "csi300"))
    diagnostics["csi300"] = {"benchmark": True}
    summary = pd.DataFrame(summary_rows)
    yearly = pd.concat(yearly_frames, ignore_index=True)
    _atomic_csv(summary, REPORT_DIR / "full_backtest_summary.csv")
    _atomic_csv(yearly, REPORT_DIR / "full_backtest_yearly.csv")
    _atomic_csv(regime, REPORT_DIR / "full_backtest_regime.csv")
    result = {
        "status": "success",
        "model": manifest["chosen"],
        "daily_cache": str(daily_path),
        "costs": {"buy": args.buy_cost, "sell": args.sell_cost},
        "slots": args.slots,
        "diagnostics": diagnostics,
    }
    _atomic_json(result, REPORT_DIR / "full_backtest_manifest.json")
    report = build_report(summary, yearly, regime, daily_path)
    report_path = REPORT_DIR / "full_backtest.md"
    temporary = report_path.with_suffix(f".{os.getpid()}.tmp.md")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    print(json.dumps({**result, "report": str(report_path)}, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run(parse_args())
