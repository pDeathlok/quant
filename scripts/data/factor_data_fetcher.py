"""
AKShare 数据获取工具

提供获取因子计算所需的各类数据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import akshare as ak
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Optional, List


class FinancialDataFetcher:
    """财务数据获取器"""
    
    @staticmethod
    def get_stock_profit_report(symbol: str, year: int, quarter: int) -> pd.DataFrame:
        """
        获取单只股票的利润表
        
        参数:
            symbol: 股票代码（如 'SH600000'）
            year: 年份
            quarter: 季度（1-4）
        
        返回:
            利润表数据
        """
        try:
            # 使用东方财富接口
            df = ak.stock_profit_sheet_by_yearly_em(symbol=symbol)
            return df
        except Exception as e:
            print(f"获取利润表失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_stock_balance_report(symbol: str, year: int, quarter: int) -> pd.DataFrame:
        """
        获取单只股票的资产负债表
        
        参数:
            symbol: 股票代码（如 'SH600000'）
            year: 年份
            quarter: 季度（1-4）
        
        返回:
            资产负债表数据
        """
        try:
            df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
            return df
        except Exception as e:
            print(f"获取资产负债表失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_stock_cash_flow_report(symbol: str, year: int, quarter: int) -> pd.DataFrame:
        """
        获取单只股票的现金流量表
        
        参数:
            symbol: 股票代码（如 'SH600000'）
            year: 年份
            quarter: 季度（1-4）
        
        返回:
            现金流量表数据
        """
        try:
            df = ak.stock_cash_flow_sheet_by_quarterly_em(symbol=symbol)
            return df
        except Exception as e:
            print(f"获取现金流量表失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_stock_financial_summary(symbol: str) -> pd.DataFrame:
        """
        获取股票财务摘要（包含多个指标）
        
        参数:
            symbol: 股票代码（如 'SH600000'）
        
        返回:
            财务摘要数据
        """
        try:
            # 使用业绩报表接口
            df = ak.stock_yjbb_em(date="20241231")
            # 筛选指定股票
            df = df[df["股票代码"] == symbol.replace("SH", "").replace("SZ", "")]
            return df
        except Exception as e:
            print(f"获取财务摘要失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_all_financial_data(date: str = "20241231") -> pd.DataFrame:
        """
        获取所有股票的财务数据（业绩报表）
        
        参数:
            date: 报告期（格式 'YYYYMMDD'，如 '20241231'）
        
        返回:
            所有股票的财务数据
        """
        try:
            df = ak.stock_yjbb_em(date=date)
            return df
        except Exception as e:
            print(f"获取财务数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_balance_sheet_all(date: str = "20241231") -> pd.DataFrame:
        """
        获取所有股票的资产负债表数据
        
        参数:
            date: 报告期（格式 'YYYYMMDD'，如 '20241231'）
        
        返回:
            所有股票的资产负债表数据
        """
        try:
            df = ak.stock_zcfz_em(date=date)
            return df
        except Exception as e:
            print(f"获取资产负债表数据失败: {e}")
            return pd.DataFrame()


class MarketDataFetcher:
    """市场数据获取器"""
    
    @staticmethod
    def get_index_daily(symbol: str = "sh000300", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取指数日线数据
        
        参数:
            symbol: 指数代码（默认沪深300 sh000300）
            start_date: 开始日期（格式 'YYYYMMDD'）
            end_date: 结束日期（格式 'YYYYMMDD'）
        
        返回:
            指数日线数据
        """
        try:
            df = ak.index_zh_a_hist(
                symbol=symbol,
                start_date=start_date or "20100101",
                end_date=end_date or date.today().strftime("%Y%m%d")
            )
            
            # 计算收益率
            df["return"] = df["收盘"].pct_change()
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount"
            })
            
            df["date"] = pd.to_datetime(df["date"]).dt.date
            
            return df
        except Exception as e:
            print(f"获取指数数据失败: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_market_return(start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取市场基准收益率（沪深300）
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            市场收益率数据
        """
        df = MarketDataFetcher.get_index_daily("sh000300", start_date, end_date)
        df = df[["date", "return"]].rename(columns={"return": "market_return"})
        return df


class IndustryDataFetcher:
    """行业数据获取器"""
    
    @staticmethod
    def get_stock_industry(symbol: str) -> str:
        """
        获取单只股票的行业分类
        
        参数:
            symbol: 股票代码（如 '600000'）
        
        返回:
            行业名称
        """
        try:
            # 使用业绩报表接口获取行业信息
            df = ak.stock_yjbb_em(date="20241231")
            stock_code = symbol.replace("SH", "").replace("SZ", "")
            stock_data = df[df["股票代码"] == stock_code]
            if not stock_data.empty:
                return stock_data.iloc[0].get("所处行业", "")
            return ""
        except Exception as e:
            print(f"获取行业分类失败: {e}")
            return ""
    
    @staticmethod
    def get_all_stock_industries() -> pd.DataFrame:
        """
        获取所有股票的行业分类
        
        返回:
            包含股票代码和行业的 DataFrame
        """
        try:
            # 从业绩报表中获取行业信息
            df = ak.stock_yjbb_em(date="20241231")
            result = df[["股票代码", "股票简称", "所处行业"]].rename(columns={
                "股票代码": "symbol",
                "股票简称": "name",
                "所处行业": "industry"
            })
            return result
        except Exception as e:
            print(f"获取行业分类失败: {e}")
            return pd.DataFrame()


class FactorDataFetcher:
    """因子数据获取器（整合各类数据）"""
    
    def __init__(self):
        self.financial_fetcher = FinancialDataFetcher()
        self.market_fetcher = MarketDataFetcher()
        self.industry_fetcher = IndustryDataFetcher()
    
    def get_factor_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取完整的因子数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            包含所有因子数据的 DataFrame
        """
        # 获取行情数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover"
        })
        
        df["date"] = pd.to_datetime(df["date"]).dt.date
        
        # 获取市场基准收益
        market_df = self.market_fetcher.get_market_return(start_date, end_date)
        df = df.merge(market_df, on="date", how="left")
        
        # 获取行业分类
        industry = self.industry_fetcher.get_stock_industry(symbol)
        df["industry"] = industry
        
        # 获取财务数据（最近季度）
        today = datetime.now()
        current_year = today.year
        current_quarter = (today.month - 1) // 3 + 1
        
        financial_df = self.financial_fetcher.get_stock_financial_summary(symbol)
        if not financial_df.empty:
            # 添加财务指标
            for col in ["净利润", "营业收入", "每股收益", "净资产收益率"]:
                if col in financial_df.columns:
                    df[col.lower()] = financial_df[col].iloc[0]
        
        # 计算市值
        if "流通股本" in df.columns:
            df["market_cap"] = df["close"] * df["流通股本"] * 10000
        
        return df
    
    def batch_get_factor_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量获取多只股票的因子数据
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        返回:
            合并后的因子数据
        """
        all_data = []
        
        for symbol in symbols:
            try:
                df = self.get_factor_data(symbol, start_date, end_date)
                df["symbol"] = symbol
                all_data.append(df)
            except Exception as e:
                print(f"获取 {symbol} 因子数据失败: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


# 示例用法
if __name__ == "__main__":
    print("=== AKShare 数据获取测试 ===\n")
    
    # 1. 获取所有股票的财务数据（业绩报表）
    print("=== 获取所有股票财务数据 ===")
    financial_df = FinancialDataFetcher.get_all_financial_data("20241231")
    print(f"数据形状: {financial_df.shape}")
    print(f"字段列表: {financial_df.columns.tolist()}")
    print(financial_df.head())
    
    # 2. 获取资产负债表数据
    print("\n=== 获取资产负债表数据 ===")
    balance_df = FinancialDataFetcher.get_balance_sheet_all("20241231")
    print(f"数据形状: {balance_df.shape}")
    print(f"字段列表: {balance_df.columns.tolist()}")
    print(balance_df.head())
    
    # 3. 获取市场基准收益率
    print("\n=== 获取市场基准收益率 ===")
    market_df = MarketDataFetcher.get_market_return("20230101", "20231231")
    print(market_df.head())
    
    # 4. 获取行业分类
    print("\n=== 获取行业分类 ===")
    industry_df = IndustryDataFetcher.get_all_stock_industries()
    print(f"数据形状: {industry_df.shape}")
    print(industry_df.head(10))
    
    # 5. 获取单只股票行业
    print("\n=== 获取单只股票行业 ===")
    industry = IndustryDataFetcher.get_stock_industry("600000")
    print(f"浦发银行行业: {industry}")