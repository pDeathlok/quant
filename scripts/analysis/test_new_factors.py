"""
测试新增因子
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pandas as pd
from datetime import date
from quant.data.factors import KDJ, WilliamsR, BIAS, Momentum, PSY, VR, OBV, CCI, DMI


def test_new_factors():
    """测试新增的因子"""
    print("加载测试数据...")
    df = pd.read_parquet("./data/stocks/sh600000.parquet")
    df = df[(df["date"] >= date(2023, 1, 1)) & (df["date"] <= date(2023, 6, 30))].copy()
    
    print(f"数据范围: {df['date'].min()} ~ {df['date'].max()}")
    print(f"数据条数: {len(df)}")
    
    # 测试 KDJ 因子
    print("\n--- 测试 KDJ 因子 ---")
    kdj = KDJ()
    kdj_result = kdj.compute(df)
    df = pd.concat([df, kdj_result], axis=1)
    print("KDJ 前5个值:")
    print(df[["date", "K", "D", "J"]].dropna().head())
    
    # 测试威廉指标
    print("\n--- 测试 WilliamsR 因子 ---")
    wr = WilliamsR(window=14)
    df["WilliamsR"] = wr.compute(df)
    print("WilliamsR 前5个值:")
    print(df[["date", "close", "WilliamsR"]].dropna().head())
    
    # 测试乖离率
    print("\n--- 测试 BIAS 因子 ---")
    bias = BIAS(window=6)
    df["BIAS6"] = bias.compute(df)
    print("BIAS6 前5个值:")
    print(df[["date", "close", "BIAS6"]].dropna().head())
    
    # 测试动量指标
    print("\n--- 测试 Momentum 因子 ---")
    momentum = Momentum(window=12)
    df["Momentum"] = momentum.compute(df)
    print("Momentum 前5个值:")
    print(df[["date", "close", "Momentum"]].dropna().head())
    
    # 测试心理线
    print("\n--- 测试 PSY 因子 ---")
    psy = PSY(window=12)
    df["PSY"] = psy.compute(df)
    print("PSY 前5个值:")
    print(df[["date", "close", "PSY"]].dropna().head())
    
    # 测试成交量变异率
    print("\n--- 测试 VR 因子 ---")
    vr = VR(window=24)
    df["VR"] = vr.compute(df)
    print("VR 前5个值:")
    print(df[["date", "close", "VR"]].dropna().head())
    
    # 测试 OBV
    print("\n--- 测试 OBV 因子 ---")
    obv = OBV()
    df["OBV"] = obv.compute(df)
    print("OBV 前5个值:")
    print(df[["date", "close", "OBV"]].dropna().head())
    
    # 测试 CCI
    print("\n--- 测试 CCI 因子 ---")
    cci = CCI(window=20)
    df["CCI"] = cci.compute(df)
    print("CCI 前5个值:")
    print(df[["date", "close", "CCI"]].dropna().head())
    
    # 测试 DMI
    print("\n--- 测试 DMI 因子 ---")
    dmi = DMI()
    dmi_result = dmi.compute(df)
    df = pd.concat([df, dmi_result], axis=1)
    print("DMI 前5个值:")
    print(df[["date", "+DI", "-DI", "ADX"]].dropna().head())
    
    # 保存测试结果
    output_file = "./data/factors_test_result_all.parquet"
    df.to_parquet(output_file)
    print(f"\n测试结果已保存到: {output_file}")
    
    return df


if __name__ == "__main__":
    test_new_factors()
