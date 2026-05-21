"""
B1 策略测试脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from quant.strategies.custom.b1 import B1Strategy
from quant.data.factors import KDJ

# 加载测试数据
df = pd.read_parquet("./data/stocks/sh600000.parquet")
print(f"数据形状: {df.shape}")
print(f"数据时间范围: {df['date'].min()} ~ {df['date'].max()}")

# 计算策略所需指标
# 1. 计算振幅
df["amplitude"] = (df["high"] - df["low"]) / df["low"] * 100

# 2. 计算 BBI
close = df["close"]
ma3 = close.rolling(window=3).mean()
ma6 = close.rolling(window=6).mean()
ma12 = close.rolling(window=12).mean()
ma24 = close.rolling(window=24).mean()
df["bbi"] = (ma3 + ma6 + ma12 + ma24) / 4

# 3. 计算 MA60
df["ma60"] = close.rolling(window=60).mean()

# 4. 计算 KDJ
kdj = KDJ()
kdj_result = kdj.compute(df)
df = pd.concat([df, kdj_result], axis=1)

# 5. 上一日成交量
df["prev_volume"] = df["volume"].shift(1)

# 筛选满足条件的日期
conditions = [
    (df["pct_change"] >= -2) & (df["pct_change"] <= 2),  # 涨跌幅 -2% ~ +2%
    (df["amplitude"] < 7),  # 振幅 < 7%
    (df["bbi"] > df["ma60"]),  # BBI > MA60
    (df["J"] < 10),  # KDJ J值 < 10
    (df["volume"] > df["prev_volume"])  # 成交量高于上一日
]

# 合并所有条件
df["signal"] = True
for cond in conditions:
    df["signal"] = df["signal"] & cond

# 统计信号数量
signal_dates = df[df["signal"]].copy()
print(f"\n满足所有条件的信号数量: {len(signal_dates)}")

if not signal_dates.empty:
    print("\n信号日期详情:")
    print(signal_dates[["date", "close", "pct_change", "amplitude", "bbi", "ma60", "J", "volume", "prev_volume"]].head(10))

# 查看最近的信号
print("\n最近的信号:")
recent_signals = signal_dates.sort_values("date", ascending=False).head(5)
print(recent_signals[["date", "close", "pct_change", "amplitude", "bbi", "ma60", "J"]])

# 保存信号数据
signal_dates.to_parquet("./data/results/b1_strategy_signals.parquet")
print(f"\n信号数据已保存到: ./data/results/b1_strategy_signals.parquet")