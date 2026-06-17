#!/usr/bin/env python
"""BYD 002594.SZ buy/sell scoring strategy on qfq daily data.

The script is intentionally single-stock and rule-based. It produces a daily
score table, a latest action report, and a simple next-day-position backtest.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant.features.variable_library import calc_bbi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/byd_002594"
SYMBOL = "002594.SZ"
DISPLAY_NAME = "比亚迪"


@dataclass(frozen=True)
class StrategyConfig:
    start_date: str = "20150101"
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.0002


def bounded_score(value: pd.Series, low: float, high: float, reverse: bool = False) -> pd.Series:
    """Map a numeric series into 0..1 with clipping."""
    score = (value - low) / (high - low)
    score = score.clip(0, 1)
    return 1 - score if reverse else score


def latest_qfq_cache(cache_dir: Path) -> Path:
    files = sorted(cache_dir.glob("sz002594_*_qfq.parquet"))
    files += sorted(cache_dir.glob("tushare_002594.SZ_*_qfq.parquet"))
    if not files:
        raise FileNotFoundError(f"No BYD qfq cache found under {cache_dir}")
    return max(files, key=lambda path: path.stat().st_mtime)


def refresh_tushare_qfq(start_date: str, end_date: str, cache_dir: Path) -> Path:
    from quant.data.tushare_fetcher import TushareDataFetcher

    fetcher = TushareDataFetcher(token=os.environ.get("TUSHARE_TOKEN"), cache_dir=cache_dir)
    fetcher.get_stock_daily(SYMBOL, start_date, end_date, adjust="qfq")
    return cache_dir / f"tushare_{SYMBOL}_{start_date}_{end_date}_qfq.parquet"


def load_qfq_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "date" not in df.columns and "trade_date" in df.columns:
        df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        date_text = df["date"].astype(str).str.replace("-", "", regex=False)
        if date_text.str.fullmatch(r"\d{8}").all():
            df["date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "pct_chg" in df.columns and "pct_change" not in df.columns:
        df["pct_change"] = df["pct_chg"]
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100
    if "symbol" not in df.columns:
        df["symbol"] = SYMBOL
    keep = ["date", "symbol", "open", "high", "low", "close", "volume", "pct_change"]
    optional = [col for col in ["pre_close", "change", "pct_chg", "turnover", "amount"] if col in df.columns]
    out = df[keep + optional].dropna(subset=["date", "open", "high", "low", "close"]).copy()
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Input is already qfq. Do not re-adjust with pre_close because some qfq
    # caches keep pre_close in raw-price scale while OHLC is adjusted.
    price_df = out
    close = price_df["close"]
    high = price_df["high"]
    low = price_df["low"]
    volume = out["volume"].replace(0, np.nan)

    for window in [5, 10, 20, 60, 120, 250]:
        out[f"ma{window}"] = close.rolling(window).mean()
        out[f"ma{window}_slope20"] = out[f"ma{window}"].pct_change(20)

    out["bbi"] = calc_bbi(close)
    out["ema12"] = close.ewm(span=12, adjust=False).mean()
    out["ema26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_dif"] = out["ema12"] - out["ema26"]
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd_dif"] - out["macd_dea"]
    out["macd_hist_delta"] = out["macd_hist"].diff(3)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = tr.rolling(14).mean() / close.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    lowest_low = low.rolling(9).min()
    highest_high = high.rolling(9).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    out["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(com=2, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

    out["vol_ma20"] = volume.rolling(20).mean()
    out["vol_ratio20"] = volume / out["vol_ma20"].replace(0, np.nan)
    out["vol_z20"] = (volume - out["vol_ma20"]) / volume.rolling(20).std().replace(0, np.nan)

    out["high_60"] = close.rolling(60).max()
    out["high_120"] = close.rolling(120).max()
    out["low_120"] = close.rolling(120).min()
    out["drawdown_60"] = close / out["high_60"].replace(0, np.nan) - 1
    out["drawdown_120"] = close / out["high_120"].replace(0, np.nan) - 1
    out["dist_ma20"] = close / out["ma20"].replace(0, np.nan) - 1
    out["dist_ma60"] = close / out["ma60"].replace(0, np.nan) - 1
    out["dist_ma120"] = close / out["ma120"].replace(0, np.nan) - 1
    out["dist_ma250"] = close / out["ma250"].replace(0, np.nan) - 1
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)

    body = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    close_pos = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["distribution_day"] = ((body < -0.015) & (close_pos < 0.35) & (out["vol_ratio20"] > 1.4)).astype(float)
    out["distribution_10d"] = out["distribution_day"].rolling(10, min_periods=1).sum()
    out["trend_regime"] = np.select(
        [
            (close > out["ma250"]) & (out["ma250_slope20"] > 0),
            (close > out["ma120"]) & (out["ma120_slope20"] > 0),
            (close < out["ma250"]) & (out["ma250_slope20"] < 0),
        ],
        ["bull", "repair", "bear"],
        default="neutral",
    )
    return out


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    trend_score = (
        8 * (close > out["ma250"]).astype(float)
        + 5 * (out["ma250_slope20"] > 0).astype(float)
        + 5 * (close > out["ma120"]).astype(float)
        + 4 * (out["ma60_slope20"] > 0).astype(float)
        + 3 * (out["bbi"] > out["ma60"]).astype(float)
    )

    pullback_depth = (-out["drawdown_60"]).clip(lower=0)
    pullback_score = (
        10 * bounded_score(pullback_depth, 0.04, 0.18)
        + 8 * bounded_score(out["dist_ma60"].abs(), 0.12, 0.02, reverse=True)
        + 5 * bounded_score(out["dist_ma120"].abs(), 0.18, 0.03, reverse=True)
        + 2 * (out["ret_20"] > -0.20).astype(float)
    )

    momentum_score = (
        5 * (close > out["ma20"]).astype(float)
        + 5 * (out["macd_hist"] > out["macd_hist"].shift(1)).astype(float)
        + 4 * (out["macd_hist_delta"] > 0).astype(float)
        + 4 * ((out["kdj_j"] > out["kdj_d"]) & (out["kdj_j"] < 85)).astype(float)
        + 2 * (out["ret_5"] > 0).astype(float)
    )

    volume_score = (
        6 * bounded_score(out["vol_ratio20"], 0.75, 1.8)
        + 4 * ((out["vol_ratio20"] > 1.0) & (out["pct_change"] > 0)).astype(float)
        + 3 * (out["distribution_10d"] <= 1).astype(float)
        + 2 * bounded_score(out["vol_z20"].abs(), 2.5, 0.2, reverse=True)
    )

    risk_score = (
        6 * (out["dist_ma250"] > -0.08).astype(float)
        + 4 * (out["drawdown_120"] > -0.30).astype(float)
        + 3 * bounded_score(out["atr14_pct"], 0.065, 0.025, reverse=True)
        + 2 * (out["distribution_10d"] <= 2).astype(float)
    )

    out["buy_score"] = (trend_score + pullback_score + momentum_score + volume_score + risk_score).clip(0, 100)

    trend_break = (
        10 * (close < out["ma60"]).astype(float)
        + 8 * (close < out["ma120"]).astype(float)
        + 7 * ((close < out["ma250"]) & (out["ma250_slope20"] < 0)).astype(float)
        + 5 * (out["bbi"] < out["ma60"]).astype(float)
    )
    overheat = (
        8 * bounded_score(out["dist_ma20"], 0.06, 0.18)
        + 6 * bounded_score(out["dist_ma60"], 0.12, 0.28)
        + 4 * (out["rsi14"] > 76).astype(float)
        + 2 * (out["kdj_j"] > 105).astype(float)
    )
    distribution = (
        8 * out["distribution_day"]
        + 8 * bounded_score(out["distribution_10d"], 1, 4)
        + 4 * ((out["vol_ratio20"] > 1.8) & (out["pct_change"] < 0)).astype(float)
    )
    momentum_bad = (
        7 * (out["macd_hist"] < out["macd_hist"].shift(1)).astype(float)
        + 5 * (out["macd_hist_delta"] < 0).astype(float)
        + 4 * ((out["kdj_j"] < out["kdj_d"]) & (out["kdj_j"] > 50)).astype(float)
        + 4 * (out["ma20_slope20"] < 0).astype(float)
    )
    stop_risk = (
        6 * (out["drawdown_120"] < -0.28).astype(float)
        + 4 * (out["atr14_pct"] > 0.065).astype(float)
    )

    out["sell_score"] = (trend_break + overheat + distribution + momentum_bad + stop_risk).clip(0, 100)
    raw_target = 0.12 + out["buy_score"] * 0.0085 - out["sell_score"] * 0.0075
    out["target_position_pct"] = (raw_target.clip(0, 0.92) * 100).round(1)
    out.loc[out["trend_regime"].eq("bear"), "target_position_pct"] = out.loc[
        out["trend_regime"].eq("bear"), "target_position_pct"
    ].clip(0, 25)
    out.loc[out["sell_score"] >= 70, "target_position_pct"] = out.loc[
        out["sell_score"] >= 70, "target_position_pct"
    ].clip(0, 30)
    out.loc[out["sell_score"] >= 85, "target_position_pct"] = 0

    conditions = [
        out["sell_score"] >= 85,
        out["sell_score"] >= 70,
        (out["sell_score"] >= 55) & (out["buy_score"] < 65),
        (out["buy_score"] >= 75) & (out["sell_score"] < 45),
        (out["buy_score"] >= 62) & (out["sell_score"] < 55),
        (out["buy_score"] >= 50) & (out["sell_score"] < 60),
    ]
    choices = ["清仓/退出", "大幅降仓", "减机动仓", "买入/回补", "小幅回补", "持有观察"]
    out["action"] = np.select(conditions, choices, default="防守等待")
    return out


def backtest_scores(df: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    bt = df[df["date"] >= pd.to_datetime(config.start_date)].dropna(subset=["target_position_pct"]).copy()
    bt["daily_return"] = bt["close"].pct_change().fillna(0)
    bt["position"] = (bt["target_position_pct"].shift(1).fillna(0) / 100).clip(0, 1)
    bt["turnover"] = bt["position"].diff().abs().fillna(bt["position"].abs())
    sell_turnover = (-bt["position"].diff()).clip(lower=0).fillna(0)
    cost = bt["turnover"] * (config.commission_rate + config.slippage_rate) + sell_turnover * config.stamp_tax_rate
    bt["strategy_return"] = bt["position"] * bt["daily_return"] - cost
    bt["strategy_equity"] = (1 + bt["strategy_return"]).cumprod()
    bt["buy_hold_equity"] = (1 + bt["daily_return"]).cumprod()

    metrics = {
        "start": bt["date"].min().strftime("%Y-%m-%d"),
        "end": bt["date"].max().strftime("%Y-%m-%d"),
        "days": float(len(bt)),
        "strategy_total_return": float(bt["strategy_equity"].iloc[-1] - 1),
        "buy_hold_total_return": float(bt["buy_hold_equity"].iloc[-1] - 1),
        "strategy_max_drawdown": max_drawdown(bt["strategy_equity"]),
        "buy_hold_max_drawdown": max_drawdown(bt["buy_hold_equity"]),
        "strategy_annual_return": annual_return(bt["strategy_equity"], len(bt)),
        "buy_hold_annual_return": annual_return(bt["buy_hold_equity"], len(bt)),
        "strategy_sharpe": sharpe(bt["strategy_return"]),
        "avg_position": float(bt["position"].mean()),
        "turnover": float(bt["turnover"].sum()),
    }
    return bt, metrics


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def annual_return(equity: pd.Series, days: int) -> float:
    if days <= 0 or equity.empty:
        return math.nan
    return float(equity.iloc[-1] ** (252 / days) - 1)


def sharpe(returns: pd.Series) -> float:
    std = returns.std()
    if not std or pd.isna(std):
        return math.nan
    return float(returns.mean() / std * np.sqrt(252))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(scored: pd.DataFrame, bt: pd.DataFrame, metrics: dict[str, float], output_dir: Path, source: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    try:
        source_display = source.relative_to(PROJECT_ROOT)
    except ValueError:
        source_display = source
    latest = scored.dropna(subset=["buy_score", "sell_score"]).iloc[-1]
    recent = scored.dropna(subset=["buy_score", "sell_score"]).tail(20).copy()
    score_cols = [
        "date",
        "close",
        "buy_score",
        "sell_score",
        "target_position_pct",
        "action",
        "trend_regime",
        "dist_ma60",
        "dist_ma250",
        "drawdown_120",
        "vol_ratio20",
        "rsi14",
        "kdj_j",
    ]
    latest_scores_path = output_dir / "byd_002594_scores_latest.csv"
    scored[score_cols].to_csv(latest_scores_path, index=False)
    bt.to_csv(output_dir / "byd_002594_backtest_daily.csv", index=False)

    recent_display = recent[score_cols].copy()
    recent_display["date"] = recent_display["date"].dt.strftime("%Y-%m-%d")
    for col in ["dist_ma60", "dist_ma250", "drawdown_120"]:
        recent_display[col] = (recent_display[col] * 100).round(2)
    for col in ["buy_score", "sell_score", "target_position_pct", "vol_ratio20", "rsi14", "kdj_j"]:
        recent_display[col] = recent_display[col].round(2)

    lines = [
        "# 比亚迪 002594.SZ 买卖点评分策略",
        "",
        f"- 数据源：`{source_display}`",
        "- 价格口径：前复权 qfq；所有均线、BBI、MACD、KDJ、ATR 与回测收益均基于该口径。",
        f"- 最新信号日：{latest['date'].strftime('%Y-%m-%d')}",
        "- 边界：这是规则化交易研究输出，不构成投资建议；真实交易需结合账户风险承受能力、流动性和基本面事件。",
        "",
        "## 最新结论",
        "",
        f"- 收盘价：{latest['close']:.2f}",
        f"- 买入分：{latest['buy_score']:.1f} / 100",
        f"- 卖出分：{latest['sell_score']:.1f} / 100",
        f"- 建议目标仓位：{latest['target_position_pct']:.1f}%",
        f"- 动作：{latest['action']}",
        f"- 趋势状态：{latest['trend_regime']}",
        "",
        "## 评分解释",
        "",
        "- `buy_score >= 75 且 sell_score < 45`：趋势未坏、回调质量和修复动能共振，适合回补或加仓。",
        "- `buy_score 62-75 且 sell_score < 55`：小幅回补区，重仓账户更适合等确认而不是猛加。",
        "- `sell_score 55-70`：减机动仓，优先把仓位降回策略目标。",
        "- `sell_score 70-85`：大幅降仓，趋势破位、放量派发或动能恶化已成主线。",
        "- `sell_score >= 85`：清仓/退出信号，等待重新站回长期趋势再评估。",
        "",
        "## 简化回测",
        "",
        f"- 区间：{metrics['start']} 至 {metrics['end']}",
        f"- 策略累计收益：{pct(metrics['strategy_total_return'])}",
        f"- 买入持有累计收益：{pct(metrics['buy_hold_total_return'])}",
        f"- 策略年化收益：{pct(metrics['strategy_annual_return'])}",
        f"- 买入持有年化收益：{pct(metrics['buy_hold_annual_return'])}",
        f"- 策略最大回撤：{pct(metrics['strategy_max_drawdown'])}",
        f"- 买入持有最大回撤：{pct(metrics['buy_hold_max_drawdown'])}",
        f"- 策略 Sharpe：{metrics['strategy_sharpe']:.2f}",
        f"- 平均仓位：{pct(metrics['avg_position'])}",
        f"- 累计换手：{metrics['turnover']:.2f} 倍",
        "",
        "## 最近 20 个交易日",
        "",
        recent_display.to_markdown(index=False),
        "",
        "## 执行纪律",
        "",
        "1. 重仓账户先看 `target_position_pct`，不要只看买卖动作文本。",
        "2. 目标仓位低于当前仓位 20 个百分点以上时，优先减机动仓或高成本仓。",
        "3. 只有当买入分重新超过 75 且卖出分低于 45，才允许把仓位加回重仓区。",
        "4. 跌破年线且年线斜率向下时，目标仓位自动封顶 25%，避免用补仓对抗熊段。",
        "5. 单票仓位建议设置账户级硬上限；比亚迪这类高波动成长股不适合无限摊平。",
        "",
    ]
    report_path = output_dir / "byd_002594_trading_strategy_latest.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BYD 002594.SZ qfq buy/sell scoring strategy")
    parser.add_argument("--data", type=Path, default=None, help="Optional qfq parquet path")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/cache")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--refresh-tushare", action="store_true", help="Refresh qfq data through Tushare first")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.data
    if args.refresh_tushare:
        source = refresh_tushare_qfq("20110101", args.end_date, args.cache_dir)
    if source is None:
        source = latest_qfq_cache(args.cache_dir)

    config = StrategyConfig(start_date=args.start_date)
    daily = load_qfq_data(source)
    scored = add_scores(add_indicators(daily))
    bt, metrics = backtest_scores(scored, config)
    report_path = write_report(scored, bt, metrics, args.output_dir, source)
    latest = scored.dropna(subset=["buy_score", "sell_score"]).iloc[-1]
    print(f"Report written: {report_path}")
    print(
        f"{DISPLAY_NAME} {latest['date'].strftime('%Y-%m-%d')} "
        f"buy={latest['buy_score']:.1f} sell={latest['sell_score']:.1f} "
        f"target={latest['target_position_pct']:.1f}% action={latest['action']}"
    )


if __name__ == "__main__":
    main()
