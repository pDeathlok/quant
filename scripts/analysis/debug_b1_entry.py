#!/usr/bin/env python3
"""诊断 B1 策略在特定时间段的入场条件"""

import sys
sys.path.insert(0, "src")

import pandas as pd
from quant.data.fetcher import DataFetcher
from quant.data.factors import KDJ

SYMBOL = "sh600000"
START_DATE = "20240101"
END_DATE = "20241231"

# 获取数据
fetcher = DataFetcher()
df = fetcher.get_stock_daily(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

# 计算各项指标
df["pct_change"] = df["close"].pct_change() * 100
df["amplitude"] = (df["high"] - df["low"]) / df["low"] * 100

# BBI
ma3 = df["close"].rolling(window=3).mean()
ma6 = df["close"].rolling(window=6).mean()
ma12 = df["close"].rolling(window=12).mean()
ma24 = df["close"].rolling(window=24).mean()
df["bbi"] = (ma3 + ma6 + ma12 + ma24) / 4

# MA60
df["ma60"] = df["close"].rolling(window=60).mean()

# KDJ
kdj = KDJ()
kdj_df = kdj.compute(df)
df["K"] = kdj_df["K"]
df["D"] = kdj_df["D"]
df["J"] = kdj_df["J"]

# 成交量对比
df["vol_prev"] = df["volume"].shift(1)
df["vol_up"] = df["volume"] > df["vol_prev"]

# 检查每个条件
df["c1_non_st"] = True
df["c2_pct"] = (-2 <= df["pct_change"]) & (df["pct_change"] <= 2)
df["c3_amp"] = df["amplitude"] < 7
df["c4_bbi"] = df["bbi"] > df["ma60"]
df["c5_j"] = df["J"] < -5
df["c6_vol"] = df["vol_up"]

df["all_pass"] = df["c1_non_st"] & df["c2_pct"] & df["c3_amp"] & df["c4_bbi"] & df["c5_j"] & df["c6_vol"]

# 打印结果
print(f"=== {SYMBOL} 入场条件诊断 ({START_DATE} ~ {END_DATE}) ===\n")
print(f"J < -5 的天数: {df['c5_j'].sum()}")
print(f"所有条件通过的天数: {df['all_pass'].sum()}\n")

# 打印 J < -5 的日子，看其他条件
candidates = df[df["c5_j"]]
for idx, row in candidates.iterrows():
    print(f"日期: {row['date']}, 收盘: {row['close']:.2f}, J: {row['J']:.1f}")
    print(f"  涨跌幅: {row['pct_change']:.2f}%  {'[通过]' if row['c2_pct'] else '[不通过]'}")
    print(f"  振幅: {row['amplitude']:.2f}%  {'[通过]' if row['c3_amp'] else '[不通过]'}")
    print(f"  BBI: {row['bbi']:.2f}, MA60: {row['ma60']:.2f}  {'[通过]' if row['c4_bbi'] else '[不通过]'}")
    print(f"  成交量: {row['volume']:.0f} > 前日 {row['vol_prev']:.0f}?  {'[通过]' if row['c6_vol'] else '[不通过]'}")
    print()

# 也打印所有通过的日子
passed = df[df["all_pass"]]
if not passed.empty:
    print(f"\n=== 所有条件通过的日期 ({len(passed)} 天) ===")
    for idx, row in passed.iterrows():
        print(f"  {row['date']}, 收盘: {row['close']:.2f}, J: {row['J']:.1f}")
else:
    print("\n=== 没有一天所有条件全部通过 ===")
