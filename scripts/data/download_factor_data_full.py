#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子数据下载脚本 - 基于因子设计文档

根据因子设计文档下载所有相关数据，用于构建因子库。

数据来源：Tushare Pro
权限说明：积分仅作为权限约束，不会被消耗
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 创建目录
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tushare Token
TOKEN = "wwqxe0122b7c9829941beb898d20d5c19db0eb0c62ea8fee51c100qq"


class FactorDataDownloader:
    """因子数据下载器"""
    
    def __init__(self, token: str = TOKEN):
        import tushare as ts
        
        self.token = token
        self.pro = ts.pro_api(token)
        self.stocks = None
        
    def get_stock_list(self, list_status: str = 'L') -> pd.DataFrame:
        """获取股票列表"""
        print("获取股票列表...")
        
        stocks = self.pro.stock_basic(
            exchange='',
            list_status=list_status,
            fields='ts_code,symbol,name,area,industry,list_date,market'
        )
        
        self.stocks = stocks
        print(f"获取 {len(stocks)} 只股票")
        
        return stocks
    
    def download_stock_basic(self) -> pd.DataFrame:
        """下载股票基本信息"""
        print("\n下载股票基本信息...")
        
        if self.stocks is None:
            self.get_stock_list()
        
        # 保存股票基本信息
        self.stocks.to_parquet(RAW_DIR / "stock_basic.parquet")
        print(f"股票基本信息下载完成: {len(self.stocks)} 条记录")
        
        return self.stocks
    
    def download_daily_data(self, start_date: str = "20100101", end_date: str = None) -> None:
        """
        下载日线行情数据
        
        数据用途：
        - 量价因子（价格、成交量）
        - 趋势因子（MACD、ADX等）
        - 波动率因子
        - 动量/反转因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载日线行情数据 ({start_date} - {end_date})...")
        
        if self.stocks is None:
            self.get_stock_list()
        
        # 创建日线数据目录
        daily_dir = RAW_DIR / "daily"
        daily_dir.mkdir(exist_ok=True)
        
        success_count = 0
        fail_count = 0
        
        for i, row in self.stocks.iterrows():
            ts_code = row['ts_code']
            
            if i % 100 == 0:
                print(f"  进度: {i}/{len(self.stocks)} ({success_count}成功, {fail_count}失败)")
            
            try:
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and len(df) > 0:
                    df.to_parquet(daily_dir / f"{ts_code}.parquet")
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                fail_count += 1
        
        print(f"日线数据下载完成: {success_count}成功, {fail_count}失败")
    
    def download_financial_indicator(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载财务指标数据
        
        数据用途：
        - 估值因子（PE、PB、PS等）
        - 盈利能力因子（ROE、ROA、EPS等）
        - 成长因子（营收/利润同比环比）
        - 质量因子（盈利质量等）
        - 杠杆因子（资产负债率等）
        - 流动性因子（流动比率等）
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载财务指标数据 ({start_date} - {end_date})...")
        
        if self.stocks is None:
            self.get_stock_list()
        
        all_fina = []
        success_count = 0
        fail_count = 0
        
        for i, row in self.stocks.iterrows():
            ts_code = row['ts_code']
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(self.stocks)} ({success_count}成功, {fail_count}失败)")
            
            try:
                df = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,end_date,ann_date,eps,roe,roe_waa,roe_dt,roa,roa_dt,'
                           'netprofit_margin,profit_margin,grossprofit_margin,'
                           'pe,pe_ttm,pb,ps,ps_ttm,'
                           'debt_to_assets,debt_to_equity,current_ratio,quick_ratio,'
                           'ar_turn,inv_turn,assets_turn,'
                           'revenue,operate_profit,total_profit,n_income,'
                           'total_assets,total_liab,total_hldr_eqy_exc_min_int,'
                           'op_cash_flow_ps,free_cash_flow_ps,n_cashflow_act,'
                           'shares_b,shares_a,total_share,'
                           'profit_to_gr,o_profit_to_gr,'
                           'basic_eps_yoy,diluted_eps_yoy,'
                           'or_yoy,revenue_yoy'
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
            return fina_df
        
        return pd.DataFrame()
    
    def download_income_statement(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载利润表数据
        
        数据用途：
        - 盈利能力因子
        - 成长因子
        - 质量因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载利润表数据 ({start_date} - {end_date})...")
        
        if self.stocks is None:
            self.get_stock_list()
        
        all_income = []
        success_count = 0
        fail_count = 0
        
        for i, row in self.stocks.iterrows():
            ts_code = row['ts_code']
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(self.stocks)} ({success_count}成功, {fail_count}失败)")
            
            try:
                df = self.pro.income(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,report_type,'
                           'revenue,operate_profit,total_profit,n_income,n_income_attr_p,'
                           'total_revenue,income_tax,minority_gain'
                )
                if df is not None and len(df) > 0:
                    all_income.append(df)
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                fail_count += 1
        
        if all_income:
            income_df = pd.concat(all_income, ignore_index=True)
            income_df.to_parquet(RAW_DIR / "income.parquet")
            print(f"利润表下载完成: {len(income_df)} 条记录, 涉及 {income_df['ts_code'].nunique()} 只股票")
            return income_df
        
        return pd.DataFrame()
    
    def download_balance_sheet(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载资产负债表数据
        
        数据用途：
        - 杠杆因子
        - 流动性因子
        - 市值因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载资产负债表数据 ({start_date} - {end_date})...")
        
        if self.stocks is None:
            self.get_stock_list()
        
        all_balance = []
        success_count = 0
        fail_count = 0
        
        for i, row in self.stocks.iterrows():
            ts_code = row['ts_code']
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(self.stocks)} ({success_count}成功, {fail_count}失败)")
            
            try:
                df = self.pro.balancesheet(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,report_type,'
                           'total_assets,total_liab,total_hldr_eqy_exc_min_int,total_hldr_eqy_inc_min_int,'
                           'money_cap,inventories,amor_exp,lt_eqt_invest,fix_assets,'
                           'intan_assets,goodwill,notes_payable,taxes_payable,surplus_rese,minority_int'
                )
                if df is not None and len(df) > 0:
                    all_balance.append(df)
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                fail_count += 1
        
        if all_balance:
            balance_df = pd.concat(all_balance, ignore_index=True)
            balance_df.to_parquet(RAW_DIR / "balancesheet.parquet")
            print(f"资产负债表下载完成: {len(balance_df)} 条记录, 涉及 {balance_df['ts_code'].nunique()} 只股票")
            return balance_df
        
        return pd.DataFrame()
    
    def download_cash_flow(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载现金流量表数据
        
        数据用途：
        - 质量因子（盈利质量）
        - 现金流因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载现金流量表数据 ({start_date} - {end_date})...")
        
        if self.stocks is None:
            self.get_stock_list()
        
        all_cashflow = []
        success_count = 0
        fail_count = 0
        
        for i, row in self.stocks.iterrows():
            ts_code = row['ts_code']
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(self.stocks)} ({success_count}成功, {fail_count}失败)")
            
            try:
                df = self.pro.cashflow(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,report_type,'
                           'n_cashflow_act,c_pay_acq_const_fiolta,n_cashflow_inv,'
                           'n_cash_flows_fro_act_sub_tota,net_profit'
                )
                if df is not None and len(df) > 0:
                    all_cashflow.append(df)
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception as e:
                fail_count += 1
        
        if all_cashflow:
            cashflow_df = pd.concat(all_cashflow, ignore_index=True)
            cashflow_df.to_parquet(RAW_DIR / "cashflow.parquet")
            print(f"现金流量表下载完成: {len(cashflow_df)} 条记录, 涉及 {cashflow_df['ts_code'].nunique()} 只股票")
            return cashflow_df
        
        return pd.DataFrame()
    
    def download_margin(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载融资融券数据
        
        数据用途：
        - 资金流因子（融资融券）
        - 另类数据因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载融资融券数据 ({start_date} - {end_date})...")
        
        try:
            df = self.pro.margin(
                start_date=start_date,
                end_date=end_date,
                fields='trade_date,exchange_id,ts_code,fin_buy_amount,fin_refund_amount,'
                       'fin_balance,fin_buy_vol,sec_sell_amount,sec_refund_amount,'
                       'sec_balance,sec_sell_vol'
            )
            
            if df is not None and len(df) > 0:
                df.to_parquet(RAW_DIR / "margin.parquet")
                print(f"融资融券数据下载完成: {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"融资融券数据下载失败: {e}")
        
        return pd.DataFrame()
    
    def download_top_list(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载龙虎榜数据
        
        数据用途：
        - 另类数据因子（龙虎榜）
        - 市场情绪因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载龙虎榜数据 ({start_date} - {end_date})...")
        
        try:
            df = self.pro.top_list(
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,exalter,reason,'
                       'buy_amount,sell_amount,net_amount,'
                       'buy_elg_vol,sell_elg_vol,net_elg_vol'
            )
            
            if df is not None and len(df) > 0:
                df.to_parquet(RAW_DIR / "top_list.parquet")
                print(f"龙虎榜数据下载完成: {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"龙虎榜数据下载失败: {e}")
        
        return pd.DataFrame()
    
    def download_dividend(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载分红送股数据
        
        数据用途：
        - 分红因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载分红送股数据 ({start_date} - {end_date})...")
        
        try:
            df = self.pro.dividend(
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,div_proc,stk_div,stk_bo_rate,stk_co_rate,'
                       'cash_div,cash_div_tax,record_date,ex_date,pay_date'
            )
            
            if df is not None and len(df) > 0:
                df.to_parquet(RAW_DIR / "dividend.parquet")
                print(f"分红送股数据下载完成: {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"分红送股数据下载失败: {e}")
        
        return pd.DataFrame()
    
    def download_holder_trade(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载增减持数据
        
        数据用途：
        - 股东因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载增减持数据 ({start_date} - {end_date})...")
        
        try:
            df = self.pro.stk_holdertrade(
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,ann_date,holder_name,holder_type,'
                       'change_vol,change_ratio,after_ratio,avg_price'
            )
            
            if df is not None and len(df) > 0:
                df.to_parquet(RAW_DIR / "holder_trade.parquet")
                print(f"增减持数据下载完成: {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"增减持数据下载失败: {e}")
        
        return pd.DataFrame()
    
    def download_share_float(self, start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载限售解禁数据
        
        数据用途：
        - 风险因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载限售解禁数据 ({start_date} - {end_date})...")
        
        try:
            df = self.pro.share_float(
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,float_date,float_share,float_ratio,holder_name'
            )
            
            if df is not None and len(df) > 0:
                df.to_parquet(RAW_DIR / "share_float.parquet")
                print(f"限售解禁数据下载完成: {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"限售解禁数据下载失败: {e}")
        
        return pd.DataFrame()
    
    def download_index_daily(self, index_code: str = "000300.SH", 
                           start_date: str = "20100101", end_date: str = None) -> pd.DataFrame:
        """
        下载指数日线数据
        
        数据用途：
        - 市场基准（计算相对强弱、特质波动率等）
        - 行业因子
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print(f"\n下载指数日线数据 ({index_code}) ({start_date} - {end_date})...")
        
        try:
            df = self.pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and len(df) > 0:
                df.to_parquet(RAW_DIR / f"index_{index_code}.parquet")
                print(f"指数日线数据下载完成: {len(df)} 条记录")
                return df
        except Exception as e:
            print(f"指数日线数据下载失败: {e}")
        
        return pd.DataFrame()
    
    def download_all(self, start_date: str = "20100101", end_date: str = None):
        """
        下载所有因子数据
        
        数据用途：
        - 量价因子：日线行情
        - 基本面因子：财务指标、三大报表
        - 资金流因子：融资融券、龙虎榜
        - 股东因子：增减持
        - 风险因子：限售解禁
        - 分红因子：分红送股
        - 市场基准：指数日线
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        print("=" * 60)
        print("因子数据下载")
        print("=" * 60)
        print(f"时间范围: {start_date} - {end_date}")
        print(f"数据目录: {RAW_DIR}")
        print("=" * 60)
        
        # 获取股票列表
        self.get_stock_list()
        
        # 下载股票基本信息
        self.download_stock_basic()
        
        # 下载日线行情（量价因子）
        self.download_daily_data(start_date, end_date)
        
        # 下载财务数据（基本面因子）
        self.download_financial_indicator(start_date, end_date)
        self.download_income_statement(start_date, end_date)
        self.download_balance_sheet(start_date, end_date)
        self.download_cash_flow(start_date, end_date)
        
        # 下载资金流数据
        self.download_margin(start_date, end_date)
        self.download_top_list(start_date, end_date)
        
        # 下载股东数据
        self.download_holder_trade(start_date, end_date)
        
        # 下载风险数据
        self.download_share_float(start_date, end_date)
        
        # 下载分红数据
        self.download_dividend(start_date, end_date)
        
        # 下载市场基准数据
        self.download_index_daily("000300.SH", start_date, end_date)  # 沪深300
        self.download_index_daily("000001.SH", start_date, end_date)  # 上证指数
        
        print("\n" + "=" * 60)
        print("所有数据下载完成！")
        print("=" * 60)
    
    def verify_download(self):
        """验证下载结果"""
        print("\n" + "=" * 60)
        print("验证下载结果")
        print("=" * 60)
        
        files = list(RAW_DIR.glob("*.parquet"))
        
        if not files:
            print("未找到数据文件")
            return
        
        for f in files:
            try:
                df = pd.read_parquet(f)
                unique_stocks = df['ts_code'].nunique() if 'ts_code' in df.columns else 'N/A'
                date_range = f"{df['trade_date'].min()} - {df['trade_date'].max()}" if 'trade_date' in df.columns else "N/A"
                
                print(f"\n{f.name}:")
                print(f"  记录数: {len(df)}")
                print(f"  涉及股票数: {unique_stocks}")
                print(f"  日期范围: {date_range}")
                print(f"  字段数: {len(df.columns)}")
            except Exception as e:
                print(f"\n{f.name}: 读取失败 - {e}")
        
        # 检查日线数据目录
        daily_dir = RAW_DIR / "daily"
        if daily_dir.exists():
            daily_files = list(daily_dir.glob("*.parquet"))
            print(f"\ndaily/ 目录:")
            print(f"  文件数: {len(daily_files)}")


if __name__ == "__main__":
    # 创建下载器
    downloader = FactorDataDownloader()
    
    # 下载所有数据
    downloader.download_all(start_date="20100101")
    
    # 验证下载结果
    downloader.verify_download()