#!/usr/bin/env python3
"""诊断 B1 策略每笔交易的卖出原因"""

import sys
sys.path.insert(0, "src")

from datetime import date as date_type
import pandas as pd
from quant.data.fetcher import DataFetcher
from quant.data.factors import KDJ

SYMBOL = "sz300750"
START_DATE = "20200101"
END_DATE = "20260521"

fetcher = DataFetcher()
df = fetcher.get_stock_daily(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

# 预计算指标
df["pct_change"] = df["close"].pct_change() * 100
ma3 = df["close"].rolling(window=3).mean()
ma6 = df["close"].rolling(window=6).mean()
ma12 = df["close"].rolling(window=12).mean()
ma24 = df["close"].rolling(window=24).mean()
df["bbi"] = (ma3 + ma6 + ma12 + ma24) / 4

kdj = KDJ()
kdj_df = kdj.compute(df)
df["K"] = kdj_df["K"]
df["D"] = kdj_df["D"]
df["J"] = kdj_df["J"]

# 交易列表
trades = [
    (1, date_type(2023,2,9), date_type(2023,2,13), 230.61, 232.56),
    (2, date_type(2023,5,29), date_type(2023,5,30), 207.60, 202.74),
    (3, date_type(2024,6,3), date_type(2024,6,11), 189.30, 180.91),
    (4, date_type(2024,6,14), date_type(2024,6,17), 178.14, 174.74),
    (5, date_type(2025,5,30), date_type(2025,6,3), 246.14, 247.22),
    (6, date_type(2025,8,8), date_type(2025,8,20), 258.99, 273.22),
    (7, date_type(2025,8,11), date_type(2025,8,20), 258.48, 273.22),
    (8, date_type(2025,8,12), date_type(2025,8,20), 258.50, 273.22),
    (9, date_type(2026,5,15), date_type(2026,5,19), 428.08, 415.86),
]

print(f"=== {SYMBOL} 卖出原因诊断 ===\n")

for tid, entry_dt, exit_dt, entry_price, exit_price in trades:
    entry_idx = df.index[df["date"] == entry_dt].tolist()
    entry_i = entry_idx[0] if entry_idx else None

    exit_idx = df.index[df["date"] == exit_dt].tolist()
    exit_i = exit_idx[0] if exit_idx else None
    if exit_i is None:
        print(f"#{tid}: {entry_dt} -> {exit_dt} (无法找到 exit 日期)")
        continue

    row = df.iloc[exit_i]
    prev_low = df.iloc[entry_i - 1]["low"] if entry_i and entry_i > 0 else None
    stop_loss_price = prev_low * 0.98 if prev_low else None

    bars_held = len(df[(df["date"] > entry_dt) & (df["date"] <= exit_dt)])
    mask = (df["date"] >= entry_dt) & (df["date"] <= exit_dt)
    peak = df[mask]["high"].max()
    drawdown = (peak - row["close"]) / peak if peak > 0 else 0.0

    reasons = []

    # 1. 长上影线
    upper_shadow = row["high"] - row["close"]
    lower_part = row["close"] - row["low"]
    if upper_shadow > lower_part and exit_i > 0:
        prev_close = df.iloc[exit_i - 1]["close"]
        if (row["close"] - prev_close) / prev_close < -0.01:
            reasons.append(f"长上影线(上影{upper_shadow/row['close']*100:.2f}%, 收盘跌{(prev_close-row['close'])/prev_close*100:.2f}%)")

    # 2. 止损
    if stop_loss_price and row["low"] <= stop_loss_price:
        reasons.append(f"止损(low {row['low']:.2f} <= 止损价 {stop_loss_price:.2f})")

    # 3. 时间止损
    if bars_held >= 5 and drawdown >= 0.015:
        reasons.append(f"时间止损(持有{bars_held}天, 回撤{drawdown*100:.2f}%)")

    print(f"#{tid}: {entry_dt} -> {exit_dt}")
    print(f"  买入={entry_price:.2f}, 卖出={exit_price:.2f}, 盈亏={(exit_price-entry_price)*100:.0f}元")
    print(f"  持有 {bars_held} 天, 峰值 {peak:.2f}, 回撤 {drawdown*100:.2f}%")
    print(f"  J={row['J']:.1f}, 上影={upper_shadow:.2f}, 实体={lower_part:.2f}")
    if stop_loss_price:
        print(f"  止损价={stop_loss_price:.2f}, exit当日low={row['low']:.2f}")
    if reasons:
        print(f"  >>> 卖出原因: {', '.join(reasons)}")
    else:
        print(f"  >>> 卖出原因: 未匹配任何条件")
    print()
