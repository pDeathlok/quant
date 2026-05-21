"""
因子库测试脚本
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from quant.data.factors import MA, EMA, MACD, RSI, BollingerBands, ATR


def test_factors():
    """测试各类因子计算"""
    print("加载测试数据...")
    df = pd.read_parquet("./data/stocks/sh600000.parquet")
    from datetime import date
    df = df[(df["date"] >= date(2023, 1, 1)) & (df["date"] <= date(2023, 6, 30))].copy()
    
    print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"数据条数: {len(df)}")
    
    # 测试 MA 因子
    print("\n--- 测试 MA 因子 ---")
    ma5 = MA(window=5)
    ma20 = MA(window=20)
    
    df["MA5"] = ma5.compute(df)
    df["MA20"] = ma20.compute(df)
    
    print("MA5 前5个值:")
    print(df[["date", "close", "MA5"]].dropna().head())
    
    # 测试 EMA 因子
    print("\n--- 测试 EMA 因子 ---")
    ema12 = EMA(window=12)
    ema26 = EMA(window=26)
    
    df["EMA12"] = ema12.compute(df)
    df["EMA26"] = ema26.compute(df)
    
    print("EMA12 和 EMA26 前5个值:")
    print(df[["date", "close", "EMA12", "EMA26"]].dropna().head())
    
    # 测试 MACD 因子
    print("\n--- 测试 MACD 因子 ---")
    macd = MACD()
    
    df["MACD"] = macd.compute(df)
    df["Signal"] = macd.compute_signal(df)
    df["Histogram"] = macd.compute_histogram(df)
    
    print("MACD 指标前5个值:")
    print(df[["date", "MACD", "Signal", "Histogram"]].dropna().head())
    
    # 测试 RSI 因子
    print("\n--- 测试 RSI 因子 ---")
    rsi = RSI(window=14)
    
    df["RSI"] = rsi.compute(df)
    
    print("RSI 前5个值:")
    print(df[["date", "close", "RSI"]].dropna().head())
    
    # 测试布林带因子
    print("\n--- 测试布林带因子 ---")
    bb = BollingerBands(window=20, num_std=2.0)
    
    bb_result = bb.compute(df)
    df = pd.concat([df, bb_result], axis=1)
    
    print("布林带前5个值:")
    print(df[["date", "close", "upper", "middle", "lower"]].dropna().head())
    
    # 测试 ATR 因子
    print("\n--- 测试 ATR 因子 ---")
    atr = ATR(window=14)
    
    df["ATR"] = atr.compute(df)
    
    print("ATR 前5个值:")
    print(df[["date", "high", "low", "close", "ATR"]].dropna().head())
    
    # 保存测试结果
    output_file = "./data/factors_test_result.parquet"
    df.to_parquet(output_file)
    print(f"\n测试结果已保存到: {output_file}")
    
    return df


if __name__ == "__main__":
    test_factors()
