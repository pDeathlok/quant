#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
财务数据下载脚本 - 考虑积分消耗的优化方案

用户当前积分: 5000分
财务接口消耗: 50积分/次

全市场股票: ~5500只
如果下载全部4个财务接口: 5500 × 4 × 50 = 1,100,000积分 (超出预算)

优化方案: 优先下载fina_indicator(包含大部分因子), 按需要选择性下载其他数据
"""

import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 创建目录
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tushare Token（从环境变量读取）
TOKEN = os.environ.get("TUSHARE_TOKEN", "")

# 积分预算
BUDGET_POINTS = 5000
COST_PER_REQUEST = 50  # 财务接口每次调用消耗50积分


def estimate_points_needed(num_stocks, include_income=True, include_balance=True, include_cashflow=True):
    """估算需要的积分"""
    interfaces = 1  # fina_indicator 是必须的
    if include_income:
        interfaces += 1
    if include_balance:
        interfaces += 1
    if include_cashflow:
        interfaces += 1
    return num_stocks * interfaces * COST_PER_REQUEST


def download_financial_data(num_stocks=500):
    """下载财务数据"""
    import tushare as ts
    
    # 设置token
    os.environ['TUSHARE_TOKEN'] = TOKEN
    pro = ts.pro_api(TOKEN)
    
    print("=" * 60)
    print("财务数据下载方案")
    print("=" * 60)
    print(f"当前积分: {BUDGET_POINTS}")
    print(f"计划下载股票数: {num_stocks}")
    print(f"预计消耗积分: {num_stocks * COST_PER_REQUEST}")
    print("=" * 60)
    
    # 获取股票列表
    print("\n获取股票列表...")
    stocks = pro.stock_basic(
        list_status='L',
        fields='ts_code,symbol,name,industry,list_date,market'
    )
    stocks = stocks.head(num_stocks)  # 只取前num_stocks只股票
    print(f"获取 {len(stocks)} 只股票")
    
    # 下载股票基本信息（0积分）
    print("\n下载股票基本信息...")
    stocks.to_parquet(RAW_DIR / "stock_basic.parquet")
    print(f"股票基本信息下载完成: {len(stocks)} 条记录")
    
    # 下载财务指标（50积分/次）
    print("\n下载财务指标（包含ROE/ROA/PE/PB等核心因子）...")
    all_fina = []
    success_count = 0
    fail_count = 0
    
    for i, row in stocks.iterrows():
        ts_code = row['ts_code']
        
        if i % 50 == 0:
            print(f"  进度: {i}/{len(stocks)} ({success_count}成功, {fail_count}失败)")
        
        try:
            df = pro.fina_indicator(
                ts_code=ts_code,
                start_date="20100101",
                end_date="20241231",
                fields='ts_code,end_date,eps,roe,roe_waa,roe_dt,roa,roa_dt,'
                       'netprofit_margin,profit_margin,grossprofit_margin,'
                       'pe,pe_ttm,pb,ps,ps_ttm,'
                       'debt_to_assets,debt_to_equity,current_ratio,quick_ratio,'
                       'ar_turn,inv_turn,assets_turn,'
                       'revenue,operate_profit,total_profit,n_income,'
                       'total_assets,total_liab,total_hldr_eqy_exc_min_int,'
                       'op_cash_flow_ps,free_cash_flow_ps,n_cashflow_act,'
                       'shares_b,shares_a,shares_h,total_share'
            )
            if df is not None and len(df) > 0:
                all_fina.append(df)
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
    
    if all_fina:
        fina_df = pd.concat(all_fina, ignore_index=True)
        fina_df.to_parquet(RAW_DIR / "fina_indicator.parquet")
        print(f"\n财务指标下载完成: {len(fina_df)} 条记录, 涉及 {fina_df['ts_code'].nunique()} 只股票")
    
    # 估算剩余积分
    points_used = success_count * COST_PER_REQUEST
    points_left = BUDGET_POINTS - points_used
    print(f"\n已消耗积分: {points_used}")
    print(f"剩余积分: {points_left}")
    
    return stocks


def verify_download():
    """验证下载结果"""
    print("\n" + "=" * 60)
    print("验证下载结果")
    print("=" * 60)
    
    files = list(RAW_DIR.glob("*.parquet"))
    
    if not files:
        print("未找到数据文件")
        return
    
    for f in files:
        df = pd.read_parquet(f)
        unique_stocks = df['ts_code'].nunique() if 'ts_code' in df.columns else 'N/A'
        print(f"  {f.name}:")
        print(f"    记录数: {len(df)}")
        print(f"    涉及股票数: {unique_stocks}")
        print(f"    字段数: {len(df.columns)}")
        if len(df.columns) < 20:
            print(f"    字段: {df.columns.tolist()}")
        print()


if __name__ == "__main__":
    # 根据积分预算计算可下载的股票数量
    max_stocks = BUDGET_POINTS // COST_PER_REQUEST
    print(f"根据积分预算({BUDGET_POINTS}分)，最多可下载 {max_stocks} 只股票的财务数据")
    
    # 下载500只股票的财务数据（消耗约25000积分，如果积分不够则按预算下载）
    num_stocks = min(500, max_stocks)
    print(f"本次下载 {num_stocks} 只股票")
    
    download_financial_data(num_stocks)
    verify_download()