#!/usr/bin/env python
"""Range trading / T+0-style inventory strategy for BYD 002594.SZ.

The strategy is designed for an investor who already holds a large BYD
inventory. It sells strength first, buys back only near range support, and keeps
the total position capped so "doing T" does not turn into uncontrolled averaging
down.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from byd_002594_trading_strategy import (
    PROJECT_ROOT,
    SYMBOL,
    annual_return,
    latest_qfq_cache,
    load_qfq_data,
    max_drawdown,
    pct,
    sharpe,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/byd_002594"


@dataclass(frozen=True)
class RangeTConfig:
    start_date: str = "20240101"
    initial_position: float = 0.85
    current_position: float = 0.85
    core_position: float = 0.25
    floor_position: float = 0.15
    range_cap_position: float = 0.55
    buyback_cap_position: float = 0.45
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    slippage_rate: float = 0.0002


def add_range_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"].replace(0, np.nan)

    for window in [5, 10, 20, 60, 120, 250]:
        out[f"ma{window}"] = close.rolling(window).mean()
    out["ma20_slope5"] = out["ma20"].pct_change(5)
    out["ma60_slope20"] = out["ma60"].pct_change(20)
    out["ma250_slope20"] = out["ma250"].pct_change(20)

    out["range_high_60"] = high.shift(1).rolling(60, min_periods=40).max()
    out["range_low_60"] = low.shift(1).rolling(60, min_periods=40).min()
    out["range_mid_60"] = (out["range_high_60"] + out["range_low_60"]) / 2
    out["range_width_60"] = (out["range_high_60"] - out["range_low_60"]) / out["range_mid_60"].replace(0, np.nan)
    out["range_pos_60"] = (close - out["range_low_60"]) / (
        out["range_high_60"] - out["range_low_60"]
    ).replace(0, np.nan)

    out["std20"] = close.rolling(20).std()
    out["boll_upper"] = out["ma20"] + 2 * out["std20"]
    out["boll_lower"] = out["ma20"] - 2 * out["std20"]
    out["dist_ma20"] = close / out["ma20"].replace(0, np.nan) - 1
    out["dist_ma60"] = close / out["ma60"].replace(0, np.nan) - 1
    out["dist_ma250"] = close / out["ma250"].replace(0, np.nan) - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    lowest_low = low.rolling(9).min()
    highest_high = high.rolling(9).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    out["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(com=2, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]

    out["ema12"] = close.ewm(span=12, adjust=False).mean()
    out["ema26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_dif"] = out["ema12"] - out["ema26"]
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd_dif"] - out["macd_dea"]
    out["macd_hist_delta"] = out["macd_hist"].diff(3)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = tr.rolling(14).mean() / close.replace(0, np.nan)
    out["vol_ma20"] = volume.rolling(20).mean()
    out["vol_ratio20"] = volume / out["vol_ma20"].replace(0, np.nan)

    body = (out["close"] - out["open"]) / out["open"].replace(0, np.nan)
    close_pos = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    out["distribution_day"] = ((body < -0.012) & (close_pos < 0.35) & (out["vol_ratio20"] > 1.25)).astype(float)
    out["distribution_5d"] = out["distribution_day"].rolling(5, min_periods=1).sum()

    out["range_regime"] = (
        out["range_width_60"].between(0.10, 0.42)
        & (out["ma60_slope20"].abs() <= 0.055)
        & (out["atr14_pct"] <= 0.055)
    )
    out["breakdown"] = (
        (close < out["range_low_60"] * 0.985)
        & (close < out["ma20"])
        & (out["ma20_slope5"] < 0)
    )
    out["breakout_up"] = (
        (close > out["range_high_60"] * 1.015)
        & (out["vol_ratio20"] > 1.25)
        & (out["macd_hist_delta"] > 0)
    )
    return out


def classify_signal(row: pd.Series) -> tuple[str, float, str]:
    """Return action, position delta, short reason."""
    if pd.isna(row.get("range_pos_60")):
        return "WAIT", 0.0, "数据预热不足"

    upper = bool(
        row["range_pos_60"] >= 0.78
        or row["close"] >= row["boll_upper"]
        or (row["dist_ma20"] >= 0.035 and row["rsi14"] >= 58)
    )
    mid_upper_weak = bool(
        row["range_pos_60"] >= 0.62
        and (row["macd_hist_delta"] < 0 or row["kdj_j"] < row["kdj_d"])
    )
    lower = bool(
        row["range_pos_60"] <= 0.24
        or row["close"] <= row["boll_lower"]
        or (row["dist_ma20"] <= -0.035 and row["rsi14"] <= 42)
    )
    lower_confirmed = bool(
        lower
        and not row["breakdown"]
        and (
            row["kdj_j"] > row["kdj_j_prev"]
            or row["macd_hist_delta"] > 0
            or row["close"] > row["low"] * 1.015
        )
    )

    if bool(row["breakdown"]):
        return "RISK_SELL", -0.15, "跌破 60 日箱体下沿，先降风险"
    if bool(row["breakout_up"]):
        return "HOLD_BREAKOUT", 0.0, "放量上破箱体，暂停反 T，避免卖飞突破"
    if upper and row["distribution_5d"] >= 1:
        return "SELL_T_STRONG", -0.15, "上沿区叠加派发，卖出 15% 做 T"
    if upper:
        return "SELL_T", -0.10, "接近箱体上沿或布林上轨，卖出 10% 做 T"
    if mid_upper_weak:
        return "SELL_T_LIGHT", -0.05, "箱体中上部动能走弱，卖出 5%"
    if lower_confirmed:
        return "BUY_BACK", 0.08, "下沿区出现企稳，买回 8%"
    if lower:
        return "WATCH_SUPPORT", 0.0, "靠近下沿但确认不足，等企稳再买"
    return "HOLD_RANGE", 0.0, "箱体中部，不主动交易"


def add_t_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["kdj_j_prev"] = out["kdj_j"].shift(1)
    signals = out.apply(classify_signal, axis=1, result_type="expand")
    out["t_action"] = signals[0]
    out["raw_delta"] = signals[1].astype(float)
    out["reason"] = signals[2]
    return out.drop(columns=["kdj_j_prev"])


def apply_position_delta(position: float, delta: float, row: pd.Series, config: RangeTConfig) -> float:
    if delta < 0:
        floor = config.floor_position if row["t_action"] == "RISK_SELL" else config.core_position
        return max(position + delta, floor)
    if delta > 0:
        if bool(row["breakdown"]):
            return position
        if position >= config.buyback_cap_position:
            return position
        return min(position + delta, config.buyback_cap_position)
    if position > config.range_cap_position and row["t_action"] in {"HOLD_RANGE", "WATCH_SUPPORT"}:
        return max(position - 0.03, config.range_cap_position)
    return position


def backtest_range_t(scored: pd.DataFrame, config: RangeTConfig) -> tuple[pd.DataFrame, dict[str, float]]:
    bt = scored[scored["date"] >= pd.to_datetime(config.start_date)].dropna(subset=["range_pos_60"]).copy()
    bt = bt.reset_index(drop=True)
    if len(bt) < 2:
        raise RuntimeError("Not enough rows to backtest range T strategy")

    position = float(config.initial_position)
    equity = 1.0
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    sell_turnover = 0.0
    buy_turnover = 0.0

    prev_close = float(bt.loc[0, "close"])
    prev_signal = bt.loc[0]
    for idx in range(1, len(bt)):
        row = bt.loc[idx]
        open_price = float(row["open"])
        close_price = float(row["close"])
        old_position = position

        overnight_return = old_position * (open_price / prev_close - 1)
        target_position = apply_position_delta(old_position, float(prev_signal["raw_delta"]), prev_signal, config)
        turnover = abs(target_position - old_position)
        sell_part = max(old_position - target_position, 0)
        buy_part = max(target_position - old_position, 0)
        trade_cost = turnover * (config.commission_rate + config.slippage_rate) + sell_part * config.stamp_tax_rate
        intraday_return = target_position * (close_price / open_price - 1)
        daily_return = overnight_return + intraday_return - trade_cost

        equity *= 1 + daily_return
        position = target_position
        sell_turnover += sell_part
        buy_turnover += buy_part

        rows.append({
            "date": row["date"],
            "open": open_price,
            "close": close_price,
            "signal_date": prev_signal["date"],
            "executed_action": prev_signal["t_action"],
            "reason": prev_signal["reason"],
            "old_position": old_position,
            "new_position": target_position,
            "turnover": turnover,
            "sell_turnover": sell_part,
            "buy_turnover": buy_part,
            "strategy_return": daily_return,
            "strategy_equity": equity,
            "range_pos_60": row["range_pos_60"],
            "range_low_60": row["range_low_60"],
            "range_high_60": row["range_high_60"],
        })
        prev_close = close_price
        prev_signal = row

    result = pd.DataFrame(rows)
    constant_return = bt["close"].pct_change().fillna(0).iloc[1:] * config.initial_position
    result["constant_high_position_equity"] = (1 + constant_return.reset_index(drop=True)).cumprod()
    result["cash_released_pct"] = (config.initial_position - result["new_position"]).clip(lower=0)

    metrics = {
        "start": result["date"].min().strftime("%Y-%m-%d"),
        "end": result["date"].max().strftime("%Y-%m-%d"),
        "strategy_total_return": float(result["strategy_equity"].iloc[-1] - 1),
        "constant_total_return": float(result["constant_high_position_equity"].iloc[-1] - 1),
        "strategy_annual_return": annual_return(result["strategy_equity"], len(result)),
        "constant_annual_return": annual_return(result["constant_high_position_equity"], len(result)),
        "strategy_max_drawdown": max_drawdown(result["strategy_equity"]),
        "constant_max_drawdown": max_drawdown(result["constant_high_position_equity"]),
        "strategy_sharpe": sharpe(result["strategy_return"]),
        "avg_position": float(result["new_position"].mean()),
        "final_position": float(result["new_position"].iloc[-1]),
        "min_position": float(result["new_position"].min()),
        "max_position": float(result["new_position"].max()),
        "sell_turnover": float(sell_turnover),
        "buy_turnover": float(buy_turnover),
        "total_turnover": float(result["turnover"].sum()),
        "cash_released_pct": float(result["cash_released_pct"].iloc[-1]),
    }
    return result, metrics


def next_trade_plan(latest: pd.Series, config: RangeTConfig) -> dict[str, float | str]:
    current_position = config.current_position
    target_position = apply_position_delta(current_position, float(latest["raw_delta"]), latest, config)
    delta = target_position - current_position
    if latest["raw_delta"] > 0 and current_position >= config.buyback_cap_position:
        action = "NO_BUY_HIGH_POSITION"
        reason = "已有仓位高于买回上限，下沿信号只用于停止卖出，不再加仓"
    elif current_position > config.range_cap_position and latest["t_action"] in {"HOLD_RANGE", "WATCH_SUPPORT"}:
        action = "TRIM_TO_RANGE_CAP"
        reason = "当前仓位高于震荡策略上限，反弹不足也先小步降仓"
    else:
        action = str(latest["t_action"])
        reason = str(latest["reason"])
    return {
        "action": action,
        "current_position_pct": current_position * 100,
        "target_position_pct": target_position * 100,
        "trade_delta_pct": delta * 100,
        "reason": reason,
    }


def write_report(
    scored: pd.DataFrame,
    bt: pd.DataFrame,
    metrics: dict[str, float],
    output_dir: Path,
    source: Path,
    config: RangeTConfig,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    try:
        source_display = source.relative_to(PROJECT_ROOT)
    except ValueError:
        source_display = source

    scored.to_csv(output_dir / "byd_002594_range_t_signals.csv", index=False)
    bt.to_csv(output_dir / "byd_002594_range_t_backtest_daily.csv", index=False)

    latest = scored.dropna(subset=["range_pos_60"]).iloc[-1]
    plan = next_trade_plan(latest, config)
    range_low = float(latest["range_low_60"])
    range_high = float(latest["range_high_60"])
    range_width = range_high - range_low
    support_break = range_low * 0.985
    mid_trim = range_low + 0.50 * range_width
    weak_trim = range_low + 0.62 * range_width
    strong_trim = range_low + 0.78 * range_width
    recent = scored.dropna(subset=["range_pos_60"]).tail(15).copy()
    recent_cols = [
        "date",
        "close",
        "range_low_60",
        "range_high_60",
        "range_pos_60",
        "rsi14",
        "kdj_j",
        "vol_ratio20",
        "t_action",
        "raw_delta",
        "reason",
    ]
    recent_display = recent[recent_cols].copy()
    recent_display["date"] = recent_display["date"].dt.strftime("%Y-%m-%d")
    for col in ["close", "range_low_60", "range_high_60", "rsi14", "kdj_j", "vol_ratio20"]:
        recent_display[col] = recent_display[col].round(2)
    recent_display["range_pos_60"] = (recent_display["range_pos_60"] * 100).round(1)
    recent_display["raw_delta"] = (recent_display["raw_delta"] * 100).round(1)

    lines = [
        "# 比亚迪 002594.SZ 区间做 T 降仓策略",
        "",
        f"- 数据源：`{source_display}`",
        "- 价格口径：前复权 qfq。",
        f"- 最新信号日：{latest['date'].strftime('%Y-%m-%d')}",
        f"- 默认当前仓位：{plan['current_position_pct']:.1f}%",
        "- 设计目标：高仓位持有者在震荡箱体内反弹减仓、下沿确认再买回，逐步把仓位降到可控区。",
        "",
        "## 下一交易日操作单",
        "",
        f"- 最新收盘：{latest['close']:.2f}",
        f"- 60 日箱体下沿：{latest['range_low_60']:.2f}",
        f"- 60 日箱体上沿：{latest['range_high_60']:.2f}",
        f"- 当前箱体位置：{latest['range_pos_60'] * 100:.1f}%",
        f"- 动作：{plan['action']}",
        f"- 仓位变化：{plan['trade_delta_pct']:+.1f} 个百分点",
        f"- 目标仓位：{plan['target_position_pct']:.1f}%",
        f"- 理由：{plan['reason']}",
        "",
        "## 价格阶梯",
        "",
        f"- 跌破 `{support_break:.2f}`：视为箱体下破，卖出 15 个百分点，保留防守底仓。",
        f"- 反弹到 `{mid_trim:.2f}` 附近：若仓位仍高于 55%，先卖 3-5 个百分点，把仓位往 55% 以下压。",
        f"- 反弹到 `{weak_trim:.2f}` 附近且动能转弱：卖出 5 个百分点。",
        f"- 反弹到 `{strong_trim:.2f}` 以上或触及布林上轨：卖出 10 个百分点；若叠加放量阴线/派发，卖出 15 个百分点。",
        f"- 回落到 `{range_low:.2f}` 附近：只有仓位低于 {config.buyback_cap_position * 100:.0f}% 且出现企稳，才买回 8 个百分点。",
        "",
        "## 操作规则",
        "",
        "- 上沿区：箱体位置 >= 78%、触及布林上轨、或明显高于 20 日线时，卖出 10%；若叠加派发，卖出 15%。",
        "- 中上区转弱：箱体位置 >= 62% 且 MACD/KDJ 走弱，卖出 5%。",
        "- 下沿区：箱体位置 <= 24%、触及布林下轨、或明显低于 20 日线时，只观察；必须有 KDJ/MACD/下影企稳才买回 8%。",
        "- 跌破箱体：有效跌破 60 日箱体下沿且 20 日线向下，卖出 15%，仓位最低降到防守底仓。",
        "- 放量上破：放量突破箱体上沿时暂停反 T，不机械卖飞突破。",
        "- 高仓位额外纪律：如果仓位高于 55%，即使在箱体中部，也每天小步降 3%，直到回到 55% 以下。",
        "",
        "## 回测结果",
        "",
        f"- 区间：{metrics['start']} 至 {metrics['end']}",
        f"- 初始仓位：{config.initial_position * 100:.1f}%",
        f"- 策略累计收益：{pct(metrics['strategy_total_return'])}",
        f"- 同仓位不做 T 收益：{pct(metrics['constant_total_return'])}",
        f"- 策略年化收益：{pct(metrics['strategy_annual_return'])}",
        f"- 同仓位不做 T 年化：{pct(metrics['constant_annual_return'])}",
        f"- 策略最大回撤：{pct(metrics['strategy_max_drawdown'])}",
        f"- 同仓位不做 T 最大回撤：{pct(metrics['constant_max_drawdown'])}",
        f"- Sharpe：{metrics['strategy_sharpe']:.2f}",
        f"- 平均仓位：{metrics['avg_position'] * 100:.1f}%",
        f"- 期末仓位：{metrics['final_position'] * 100:.1f}%",
        f"- 释放现金：{metrics['cash_released_pct'] * 100:.1f} 个百分点",
        f"- 卖出换手：{metrics['sell_turnover']:.2f} 倍",
        f"- 买回换手：{metrics['buy_turnover']:.2f} 倍",
        "",
        "## 最近信号",
        "",
        recent_display.to_markdown(index=False),
        "",
        "## 边界",
        "",
        "本策略只使用日线，无法知道真实盘中先上后下还是先下后上；回测按收盘产生信号、下一交易日开盘执行。它适合制定纪律，不适合替代盘中盘口判断。",
        "",
    ]
    report_path = output_dir / "byd_002594_range_t_strategy_latest.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BYD range T strategy")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data/cache")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--initial-position", type=float, default=0.85)
    parser.add_argument("--current-position", type=float, default=0.85)
    parser.add_argument("--core-position", type=float, default=0.25)
    parser.add_argument("--floor-position", type=float, default=0.15)
    parser.add_argument("--range-cap-position", type=float, default=0.55)
    parser.add_argument("--buyback-cap-position", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.data or latest_qfq_cache(args.cache_dir)
    config = RangeTConfig(
        start_date=args.start_date,
        initial_position=args.initial_position,
        current_position=args.current_position,
        core_position=args.core_position,
        floor_position=args.floor_position,
        range_cap_position=args.range_cap_position,
        buyback_cap_position=args.buyback_cap_position,
    )
    daily = load_qfq_data(source)
    scored = add_t_signals(add_range_indicators(daily))
    bt, metrics = backtest_range_t(scored, config)
    report_path = write_report(scored, bt, metrics, args.output_dir, source, config)
    latest = scored.dropna(subset=["range_pos_60"]).iloc[-1]
    plan = next_trade_plan(latest, config)
    print(f"Report written: {report_path}")
    print(
        f"{SYMBOL} {latest['date'].strftime('%Y-%m-%d')} "
        f"action={plan['action']} delta={plan['trade_delta_pct']:+.1f}pp "
        f"target={plan['target_position_pct']:.1f}% range_pos={latest['range_pos_60'] * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
